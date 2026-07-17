"""ICT 再設計 — フェーズ0: 執行と時計の土台（信号を1本も作る前に凍結する）。

背骨（`docs/findings/s01_entries.md` の ICT v3 項）: 各層は正しい帰無に勝ってから次を積む。
執行モデル（ASK基準の指値約定・realistic コスト・同足タイブレーク）と時計（tz変換）は
"外枠" として最初に固定する。ここが甘いと病巣「薄い時間帯の偽約定」「スプレッドで崩壊」が
全層をすり抜ける（v1→v2 で執行を後付けして全部再走した反省）。

このモジュールが持つのは:
  - 銘柄マップ / コストモデル（spread, commission）/ PIP
  - load_ny  : Vantage CSV(ブローカー時刻 EET/EEST) → NY 壁時計(naive) へ一度だけ変換 + ATR14
  - prep     : ny_wall を昇順配列にし、日ごとの窓を timestamp で切れるようにする
  - window_pos / clock_check
  - walk     : ASK基準の指値約定 + 前進走査（約定足も損切り判定に含める・同足は損切り優先）
  - stats / sc : 成績レポータ

walk は setups（L/H/atr/kz を持つ dict のリスト）を受け取るだけで、
"どう作られたか"（狩り+MSS 等）は知らない ＝ フェーズ1 との綺麗な継ぎ目。

自己検査: .venv/bin/python scratchpad/ict_exec.py
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv

# --- 窓の定義（全て NY 壁時計。ロング=安値を狩って上へ MSS / ショート=鏡像） ---
ASIA_HOURS = (19, 2)       # 前日 19:00 〜 当日 02:00
LONDON_HOURS = (2, 7)      # 当日 02:00 〜 07:00
KZ_HOURS = (7, 10)         # 当日 07:00 〜 10:00  （≒ JST 20:00-22:00）
ATR_LEN = 14
FWD_CAP = 500
BUF = 0.1                  # 損切りバッファ（× ATR）
F_CANON, RR_CANON = 0.25, 4.0   # 浅い押し目・遠い固定目標（本リポジトリ正典。ICT の深OTE/RR2 ではない）

SYMS = {
    "gold":   "data/vantage_xauusd_m15.csv",
    "eurusd": "data/vantage_eurusd_m15.csv",
    "gbpusd": "data/vantage_gbpusd_m15.csv",
    "usdjpy": "data/vantage_usdjpy_m15.csv",
    "audusd": "data/vantage_audusd_m15.csv",
    "nzdusd": "data/vantage_nzdusd_m15.csv",
    "usdcad": "data/vantage_usdcad_m15.csv",
    "btcusd": "data/vantage_btcusd_m15.csv",
}

# (fill spread, commission-only cost) — CLAUDE.md の実測値を物理分解。
#   ASK で買い BID で決済する時点でスプレッドは損切り幅に埋まっているので二重計上しない。
MODEL = {
    "gold":   (0.15, 0.06),
    "eurusd": (0.3e-4, 0.6e-4), "gbpusd": (0.3e-4, 0.6e-4), "audusd": (0.3e-4, 0.6e-4),
    "nzdusd": (0.3e-4, 0.6e-4), "usdcad": (0.3e-4, 0.6e-4),
    "usdjpy": (0.3e-2, 0.6e-2),
    "btcusd": (15.0, 0.0),
}
PIP = {"eurusd": 1e-4, "gbpusd": 1e-4, "audusd": 1e-4, "nzdusd": 1e-4, "usdcad": 1e-4,
       "usdjpy": 1e-2, "gold": 0.1, "btcusd": 1.0}
CUT2000 = {"usdjpy"}       # USDJPY m5/m15 は 1999 以前が日足ラベルの罠（CLAUDE.md）


# ---------------------------------------------------------------------------
def load_ny(path, cut2000=False):
    """ブローカー時刻 (Europe/Riga) → NY 壁時計(naive) へ一度だけ変換。以降は naive 比較のみ。"""
    df = load_mt5_csv(path)
    if cut2000:
        df = df.loc["2000-01-01":]
    naive_idx = df.index.tz_localize(None)
    riga_idx = naive_idx.tz_localize("Europe/Riga", ambiguous="NaT", nonexistent="shift_forward")
    nat_mask = riga_idx.isna()
    n_nat = int(nat_mask.sum())
    if n_nat:
        df = df.loc[~nat_mask].copy()
        naive_idx = naive_idx[~nat_mask]
        riga_idx = riga_idx[~nat_mask]
    else:
        df = df.copy()
    ny_wall = riga_idx.tz_convert("America/New_York").tz_localize(None)
    df["broker_dt"] = naive_idx
    df["ny_wall"] = ny_wall
    df["ny_hour"] = ny_wall.hour
    df = df.reset_index(drop=True)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN).values
    return df, n_nat


def prep(df):
    """ny_wall を datetime64 昇順配列にし、NY暦の日付リストを返す。"""
    ny = pd.to_datetime(df["ny_wall"].values)
    df = df.copy()
    df["_t"] = ny
    dates = np.array(sorted(set(pd.DatetimeIndex(ny).normalize())))
    return df, ny.values.astype("datetime64[ns]"), dates


def span_years(df):
    y = pd.to_datetime(df["broker_dt"]).dt.year.value_counts()
    return int((y > 5000).sum())     # 実質データのある年数（疎データ年を除く）


def window_pos(tarr, t0, t1):
    """[t0, t1) に入るバーの位置（tarr は昇順）。"""
    a = np.searchsorted(tarr, np.datetime64(t0), "left")
    b = np.searchsorted(tarr, np.datetime64(t1), "left")
    return a, b


def clock_check(df, name):
    """時計の自己検査: NY 時刻別の平均値幅が実セッション・プロファイルを再現するか。"""
    d = df.dropna(subset=["atr14"])
    d = d[d["atr14"] > 0]
    g = ((d["high"] - d["low"]) / d["atr14"]).groupby(d["ny_hour"]).mean().reindex(range(24))
    top5 = list(g.sort_values(ascending=False).index[:5])
    print(f"  [{name}] NY時刻別 平均(high-low)/ATR14 の TOP5 = {top5}")
    return g


# ---------------------------------------------------------------------------
def walk(df, setups, f, rr, buf, spread, cost, side, lim_fn=None,
         tgt_fn=None, tgt_fn_full=None, skip_log=None, rr_log=None, trade_log=None,
         partial_r=None, partial_frac=0.5):
    """ASK基準の指値約定 + 前進走査。約定足も損切り判定に含める。同足タイブレーク＝損切り優先。
    setups[*][side] = dict(L, H, atr, kz=(k0,k1)[, fvg_lo, fvg_hi, pdh/pdl, asiaH/asiaL, swingH/L(N)])。
    返り値 = (date, net_R, gross_R, risk) のリスト（後方互換の4-tuple、tgt_fn有無に関わらず不変）。
    lim_fn: 省略時は従来通り lim=H-f*(H-L)（ロング）/L+f*(H-L)（ショート）の固定リトレース。
    与えると lim_fn(s)->price で入口アンカーを上書きする（例: FVG近位端タップ）。f はその場合無視される。
    tgt_fn: 省略時は従来通り tgt=entry+rr*risk（ロング）/entry-rr*risk（ショート）の固定RR。
    与えると tgt_fn(s, entry, risk)->(price_or_None, reason) で出口を上書きする（外部流動性ターゲット,
    優先3）。None が返るとそのトレードは見送り（skip_log があれば (date, reason) を記録）。
    rr が固定入力でなく出力になる（実現RR = |tgt-entry|/risk）ため、rr_log があれば (date, realized_rr)
    を記録する。rr 引数はその場合 walk() 内では不使用（tgt_fn 側のロジックのみが出口を決める）。
    trade_log: 省略時は挙動不変（既存呼び出し元は無改変）。与えると約定した各トレードについて
    dict(date, side, fill_dt(broker_dt), entry, stop, tgt, r_rr, R, net, reason∈{TP,SL,timeout}) を
    追記する（レポート用の観測のみ、判定ロジックには一切影響しない）。
    partial_r: 省略時(None)は挙動完全不変（このブロックに一切入らない＝既存呼び出しビット一致）。
    与えると ICT ep41 の部分利確: +partial_r*risk 到達で partial_frac(既定50%)を利確し、残りの
    ストップを建値(entry)へ移動、残りは従来通り tgt(固定RRまたはtgt_fn)を狙う二段階出口になる。
    同足タイブレークは全箇所で不利側優先（損切り/建値ヒットが同足の利確より先に処理される）。
    最終目標価格が partial_r 到達価格より近い（rr_final<=partial_r）場合は、部分利確が発生する前に
    100%が最終目標へ到達しうる＝その時は reason="TP" で通常の単段トレードとして閉じる（部分利確は
    「価格レベル」として評価するため、これを取り違えると「近い目標のときだけ部分利確が起きない」を
    見落とすバグになる）。reason ∈ {TP, SL, timeout, partial_be, partial_tp, partial_timeout}。
    R/net の会計: partial_credit=partial_frac*partial_r（部分利確ぶんの寄与）+ (1-partial_frac)*残り
    （建値=0 / 最終RR / タイムアウト時のクローズ価格ベースR）。cost は既存同様トレード1件につき
    1回のみ（net = R - cost/risk）＝部分約定による手数料の重複計上はモデル化しない（明記のみ）。
    tgt_fn_full: 省略時(None)は挙動完全不変（既存呼び出し元は無改変・tgt_fn と併用不可、tgt_fn が
    優先される）。tgt_fn では約定時刻(fill_dt)が渡らず時間依存の外部データ（例: 約定時点で確定済み・
    未タップの上位足流動性プール）を参照できないための拡張。与えると
    tgt_fn_full(s, entry, risk, fill_dt)->(price_or_None, reason) を呼ぶ（fill_dt=約定足の broker_dt、
    pd.Timestamp）。戻り値の意味・skip_log/rr_log の扱いは tgt_fn と同一（Draw on Liquidity 検証用）。"""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    fill_dt = df["broker_dt"].values if (trade_log is not None or tgt_fn_full is not None) else None
    trades = []
    for rec in setups:
        s = rec[side]
        if s is None:
            continue
        L, H, A = s["L"], s["H"], s["atr"]
        k0, k1 = s["kz"]
        if side == "long":
            lim = lim_fn(s) if lim_fn is not None else H - f * (H - L)
            stop = L - buf * A
            if lim <= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if l[p] <= lim - spread:      # BID データで ASK 指値が約定する条件
                    fp = p; break
            if fp is None:
                continue
            entry = min(lim, o[fp] + spread)
            risk = entry - stop
            if risk <= 0:
                continue
            if tgt_fn is not None:
                tgt, reason = tgt_fn(s, entry, risk)
                if tgt is None:
                    if skip_log is not None:
                        skip_log.append((rec["date"], reason))
                    continue
                r_rr = (tgt - entry) / risk
            elif tgt_fn_full is not None:
                tgt, reason = tgt_fn_full(s, entry, risk, pd.Timestamp(fill_dt[fp]))
                if tgt is None:
                    if skip_log is not None:
                        skip_log.append((rec["date"], reason))
                    continue
                r_rr = (tgt - entry) / risk
            else:
                tgt = entry + rr * risk
                r_rr = rr
            if rr_log is not None:
                rr_log.append((rec["date"], r_rr))
            end_idx = min(fp + FWD_CAP, n)
            if partial_r is None:
                R = None; reason = "timeout"
                for p in range(fp, end_idx):   # 約定足 fp から走査（タダ乗り防止）
                    if l[p] <= stop:
                        R = -1.0; reason = "SL"; break       # 同足は損切り優先
                    if h[p] >= tgt:
                        R = r_rr; reason = "TP"; break
                if R is None:
                    R = (c[end_idx - 1] - entry) / risk
            else:
                partial_level = entry + partial_r * risk
                partial_before_tgt = partial_level < tgt
                partial_credit = 0.0
                partial_done = False
                R = None; reason = "timeout"
                for p in range(fp, end_idx):
                    if not partial_done:
                        if l[p] <= stop:
                            R = -1.0; reason = "SL"; break         # 即SL（部分利確なし）
                        if partial_before_tgt:
                            if h[p] >= partial_level:
                                partial_done = True
                                partial_credit = partial_frac * partial_r
                                if l[p] <= entry:                   # 同足で建値も割る＝不利側優先
                                    R = partial_credit; reason = "partial_be"; break
                                if h[p] >= tgt:
                                    R = partial_credit + (1 - partial_frac) * r_rr
                                    reason = "partial_tp"; break
                                # 部分利確のみ約定、残玉は建値ストップで継続
                        else:
                            if h[p] >= tgt:                          # 最終目標が部分水準より近い
                                R = r_rr; reason = "TP"; break
                    else:
                        if l[p] <= entry:
                            R = partial_credit; reason = "partial_be"; break
                        if h[p] >= tgt:
                            R = partial_credit + (1 - partial_frac) * r_rr
                            reason = "partial_tp"; break
                if R is None:
                    last_close = c[end_idx - 1]
                    if partial_done:
                        R = partial_credit + (1 - partial_frac) * ((last_close - entry) / risk)
                        reason = "partial_timeout"
                    else:
                        R = (last_close - entry) / risk
                        reason = "timeout"
        else:
            lim = lim_fn(s) if lim_fn is not None else L + f * (H - L)
            stop = H + buf * A
            if lim >= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if h[p] >= lim + spread:
                    fp = p; break
            if fp is None:
                continue
            entry = max(lim, o[fp] - spread)
            risk = stop - entry
            if risk <= 0:
                continue
            if tgt_fn is not None:
                tgt, reason = tgt_fn(s, entry, risk)
                if tgt is None:
                    if skip_log is not None:
                        skip_log.append((rec["date"], reason))
                    continue
                r_rr = (entry - tgt) / risk
            elif tgt_fn_full is not None:
                tgt, reason = tgt_fn_full(s, entry, risk, pd.Timestamp(fill_dt[fp]))
                if tgt is None:
                    if skip_log is not None:
                        skip_log.append((rec["date"], reason))
                    continue
                r_rr = (entry - tgt) / risk
            else:
                tgt = entry - rr * risk
                r_rr = rr
            if rr_log is not None:
                rr_log.append((rec["date"], r_rr))
            end_idx = min(fp + FWD_CAP, n)
            if partial_r is None:
                R = None; reason = "timeout"
                for p in range(fp, end_idx):
                    if h[p] >= stop:
                        R = -1.0; reason = "SL"; break
                    if l[p] <= tgt:
                        R = r_rr; reason = "TP"; break
                if R is None:
                    R = (entry - c[end_idx - 1]) / risk
            else:
                partial_level = entry - partial_r * risk
                partial_before_tgt = partial_level > tgt
                partial_credit = 0.0
                partial_done = False
                R = None; reason = "timeout"
                for p in range(fp, end_idx):
                    if not partial_done:
                        if h[p] >= stop:
                            R = -1.0; reason = "SL"; break
                        if partial_before_tgt:
                            if l[p] <= partial_level:
                                partial_done = True
                                partial_credit = partial_frac * partial_r
                                if h[p] >= entry:
                                    R = partial_credit; reason = "partial_be"; break
                                if l[p] <= tgt:
                                    R = partial_credit + (1 - partial_frac) * r_rr
                                    reason = "partial_tp"; break
                        else:
                            if l[p] <= tgt:
                                R = r_rr; reason = "TP"; break
                    else:
                        if h[p] >= entry:
                            R = partial_credit; reason = "partial_be"; break
                        if l[p] <= tgt:
                            R = partial_credit + (1 - partial_frac) * r_rr
                            reason = "partial_tp"; break
                if R is None:
                    last_close = c[end_idx - 1]
                    if partial_done:
                        R = partial_credit + (1 - partial_frac) * ((entry - last_close) / risk)
                        reason = "partial_timeout"
                    else:
                        R = (entry - last_close) / risk
                        reason = "timeout"
        net = R - cost / risk
        if trade_log is not None:
            trade_log.append(dict(date=rec["date"], side=side, fill_dt=pd.Timestamp(fill_dt[fp]),
                                   entry=entry, stop=stop, tgt=tgt, r_rr=r_rr, R=R, net=net, reason=reason))
        trades.append((rec["date"], net, R, risk))
    return trades


def mfe_scan(df, setups, f, buf, spread, side):
    """目標を被せずに約定後の巡行幅を測る（反発率→巡行幅を RR より先に見るため）。
    各約定について MFE（favorable な最大 R）を、損切りに当たるまでの範囲で記録する。
    同足タイブレーク＝損切り優先（損切り足の favorable は数えない＝到達可能な RR を過大評価しない）。
    返り値 = (date, mfe_R, stopped, final_R) のリスト。mfe_R>=k の割合が「RR=k での勝率」に一致する。"""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    out = []
    for rec in setups:
        s = rec[side]
        if s is None:
            continue
        L, H, A = s["L"], s["H"], s["atr"]
        k0, k1 = s["kz"]
        if side == "long":
            lim = H - f * (H - L); stop = L - buf * A
            if lim <= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if l[p] <= lim - spread:
                    fp = p; break
            if fp is None:
                continue
            entry = min(lim, o[fp] + spread); risk = entry - stop
            if risk <= 0:
                continue
            mfe = 0.0; stopped = False; endp = fp
            for p in range(fp, min(fp + FWD_CAP, n)):
                endp = p
                if l[p] <= stop:
                    stopped = True; break
                fav = (h[p] - entry) / risk
                if fav > mfe:
                    mfe = fav
            final = (c[endp] - entry) / risk
        else:
            lim = L + f * (H - L); stop = H + buf * A
            if lim >= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if h[p] >= lim + spread:
                    fp = p; break
            if fp is None:
                continue
            entry = max(lim, o[fp] - spread); risk = stop - entry
            if risk <= 0:
                continue
            mfe = 0.0; stopped = False; endp = fp
            for p in range(fp, min(fp + FWD_CAP, n)):
                endp = p
                if h[p] >= stop:
                    stopped = True; break
                fav = (entry - l[p]) / risk
                if fav > mfe:
                    mfe = fav
            final = (entry - c[endp]) / risk
        out.append((rec["date"], mfe, stopped, final))
    return out


def stats(tr, span):
    if len(tr) < 10:
        return None
    net = np.array([t[1] for t in tr]); g = np.array([t[2] for t in tr])
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    cum = np.cumsum(net); dd = float((np.maximum.accumulate(cum) - cum).max())
    yrs = np.array([pd.Timestamp(t[0]).year for t in tr])
    by = pd.Series(net).groupby(yrs).sum()
    half = len(net) // 2
    return dict(n=len(net), npy=len(net) / span, win=100 * (g > 0).mean(),
                gross=g.mean(), net=net.mean(), pf=pos / neg if neg > 0 else np.inf,
                tot=net.sum(), dd=dd, rdd=net.sum() / dd if dd > 0 else np.inf,
                IS=net[:half].sum(), OOS=net[half:].sum(),
                gy=100 * (by > 0).mean(), ny=len(by))


def sc(tr, minn=20):
    """軽量サマリ（totR/DD 中心。ゲート/棄権の比較用）。"""
    if len(tr) < minn:
        return None
    net = np.array([t[1] for t in tr])
    cum = np.cumsum(net); dd = float((np.maximum.accumulate(cum) - cum).max())
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    return dict(n=len(net), net=net.mean(), pf=pos / neg if neg > 0 else np.inf,
                tot=net.sum(), dd=dd, rdd=net.sum() / dd if dd > 0 else np.inf)


def spread_cost(name):
    sp, cost = MODEL[name]
    return sp, cost


if __name__ == "__main__":
    import io, contextlib
    print("フェーズ0 自己検査: 時計プロファイル（gold は NY 08-09時、EURUSD は 02-03時ロンドンが最大なら正常）")
    for name in ("gold", "eurusd", "usdjpy"):
        with contextlib.redirect_stderr(io.StringIO()):
            df, n_nat = load_ny(SYMS[name], cut2000=(name in CUT2000))
        print(f"{name:7s} bars={len(df):7d} span={span_years(df)}年 NaT落ち={n_nat}", end="  ")
        clock_check(df, name)
