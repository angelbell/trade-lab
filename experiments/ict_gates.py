"""ICT 再設計 — フェーズ2-3: 方向ゲートと棄権（1つずつ、base との差でしか測らない）。

背骨: 方向は片側ずつ測る（両サイド平均は今回2回私を騙した ―― EURUSD 陰性対照の誤報、
BTC 棄権バグ）。1ゲート足すごとに原因を分離。棄権は最も効果が薄いので最後・サイド固定。

フェーズ2（方向ゲート）:
  gate_discount_long : ICT 日足バイアス＝premium/discount（平均回帰極性）。日足の直近L日レンジ内で
                       終値位置 pos<0.5-band（discount）の時だけロングを通す。EUR/GBP/JPY メジャー3種で実在。
  smt_short_gate     : 死んでいる short@premium 側を DXY＋クロスペア SMT で蘇生する候補（未実装 scaffold）。
  gate_pd            : discount→L / premium→S の両サイド版（＝符号確認の対照。invert で逆極性）。

フェーズ3（棄権）:
  gate_muddy         : サイド固定で muddy（重なり合う＝displacement 無し）な日を棄権。

pd_frame は全て前日までに確定（+1日 tz-shift で先読み排除）。join_days は各トレード日を
ロンドン窓の開始(+2h)で直近の確定日足に merge_asof で結合する。

自己検査（discount ロングの台帳アンカーを再現）:
    .venv/bin/python experiments/ict_gates.py
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import (ASIA_HOURS, LONDON_HOURS, MODEL, walk, window_pos, BUF,
                      F_CANON, RR_CANON, sc)
from ict_population import canonical_setups, trade_pool, side_list, load_prepped, prev_day_extremes
from breakout_wave import resample

LOOKBACKS = [10, 20, 40]
BANDS = [0.0, 0.10, 0.20, 0.30]


# ---------------------------------------------------------------------------
def pd_frame(df):
    """日足の premium/discount 位置(posL)と displacement(march_k)を、前日までに確定した形で返す。"""
    b = df.set_index("broker_dt")[["open", "high", "low", "close"]]
    d = resample(b, "1D")
    h, l, c = d["high"], d["low"], d["close"]
    D = pd.DataFrame(index=d.index)
    for L in LOOKBACKS:
        hi = h.rolling(L).max(); lo = l.rolling(L).min()
        D[f"pos{L}"] = ((c - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    # 行進度 = (union range) / (Σ individual range)。1に近い=一方向に変位(clean) / 小=重なりチョップ(muddy)
    for k in (3, 5):
        rng = (h - l)
        union = h.rolling(k).max() - l.rolling(k).min()
        D[f"march{k}"] = (union / rng.rolling(k).sum()).replace([np.inf, -np.inf], np.nan)
    conf = (D.index + pd.Timedelta(days=1)).tz_localize(
        "Europe/Riga", ambiguous="NaT", nonexistent="shift_forward"
    ).tz_convert("America/New_York").tz_localize(None)
    D["conf"] = conf
    return D[~D["conf"].isna()].sort_values("conf").reset_index(drop=True)


def join_days(dates, D):
    """各トレード日を、ロンドン窓の開始(+2h)時点で直近に確定した日足行へ結合（先読み無し）。"""
    q = pd.DataFrame({"date": list(dates)})
    q["t"] = pd.to_datetime(q["date"]) + pd.Timedelta(hours=LONDON_HOURS[0])
    m = pd.merge_asof(q.sort_values("t"), D, left_on="t", right_on="conf", direction="backward")
    return m.set_index("date")


# ---------------- フェーズ2: 方向ゲート ----------------
def gate_discount_long(long_pool, J, poscol="pos10", band=0.20):
    """ICT 本命: 日足が discount（pos < 0.5-band）の日だけロングを通す。"""
    thr = 0.5 - band
    return [(d, long_pool[d]) for d, r in J.iterrows()
            if d in long_pool and not pd.isna(r[poscol]) and r[poscol] < thr]


def gate_pd(pool, J, poscol="pos10", band=0.20, invert=False):
    """両サイド版（符号確認の対照）: discount→ロング / premium→ショート。invert=逆極性(トレンド追随)。"""
    lo_thr, hi_thr = 0.5 - band, 0.5 + band
    out = []
    for d, row in J.iterrows():
        pos = row[poscol]
        if pd.isna(pos):
            continue
        disc, prem = pos < lo_thr, pos > hi_thr
        long_ok = prem if invert else disc
        short_ok = disc if invert else prem
        if long_ok and d in pool["long"]:
            out.append((d, pool["long"][d]))
        if short_ok and d in pool["short"]:
            out.append((d, pool["short"][d]))
    return out


def sweep_frame(df, tarr, dates, shift=0):
    """各 NY日について「ロンドン窓が buyside/sellside 流動性を掃除したか」を返す（相方の SMT 判定用）。
    buyside_swept = ロンドン窓高値 > (アジア高値 or 前日高値) / sellside 同様。ロンドン窓終了時に確定。
    返り値 = {date: (buyside_swept, sellside_swept)}。窓が薄い日は None（保守側でゲートを通さない）。"""
    hi, lo = df["high"].values, df["low"].values
    pdh, pdl = prev_day_extremes(df, dates)
    A0H, A1H = ASIA_HOURS[0] + shift, ASIA_HOURS[1] + shift
    L0H, L1H = LONDON_HOURS[0] + shift, LONDON_HOURS[1] + shift
    out = {}
    for d in dates:
        day = pd.Timestamp(d)
        a0, a1 = window_pos(tarr, day - pd.Timedelta(days=1) + pd.Timedelta(hours=A0H),
                            day + pd.Timedelta(hours=A1H))
        l0, l1 = window_pos(tarr, day + pd.Timedelta(hours=L0H), day + pd.Timedelta(hours=L1H))
        if (a1 - a0) < 4 or (l1 - l0) < 6:
            out[d] = None; continue
        asia_lo, asia_hi = lo[a0:a1].min(), hi[a0:a1].max()
        p_lo, p_hi = pdl.get(d, np.nan), pdh.get(d, np.nan)
        lon_hi, lon_lo = hi[l0:l1].max(), lo[l0:l1].min()
        buy = bool((np.isfinite(asia_hi) and lon_hi > asia_hi) or (np.isfinite(p_hi) and lon_hi > p_hi))
        sell = bool((np.isfinite(asia_lo) and lon_lo < asia_lo) or (np.isfinite(p_lo) and lon_lo < p_lo))
        out[d] = (buy, sell)
    return out


def smt_short_gate(short_pool, partner_sweep):
    """クロスペア SMT（第一版・DXY不要）: ショート設定日のうち、相方ペアが同じロンドン窓で buyside を
    掃除しなかった（higher high を作らなかった）日だけ通す＝弱気ダイバージェンス。

    このペアのショート母集団は「自分は buyside を掃除して反転（MSS下）」＝既に自分の掃除を含む。
    そこに相方の非掃除を重ねると「片方だけが流動性を取りに行って失敗」＝ICT の SMT。
    short_pool = {date: net}（ict_population.trade_pool の "short"）。
    partner_sweep = sweep_frame(相方)。相方データ欠損日(None)は通さない（保守側）。"""
    out = []
    for d, net in short_pool.items():
        sw = partner_sweep.get(d)
        if sw is None:
            continue
        buy_swept, _ = sw
        if not buy_swept:            # 相方は高値を取りに行かなかった＝ダイバージェンス
            out.append((d, net))
    return out


# ---------------- フェーズ3: 棄権 ----------------
def gate_muddy(side_pool, J, marchcol="march5", q=0.35):
    """サイド固定で muddy（下位 q の march＝重なり合う日）を棄権。side_pool = {date: net}。"""
    thr = J[marchcol].quantile(q) if q > 0 else -np.inf
    out = []
    for d, row in J.iterrows():
        if d not in side_pool:
            continue
        m = row[marchcol]
        if q > 0 and (pd.isna(m) or m < thr):
            continue
        out.append((d, side_pool[d]))
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("フェーズ2-3 自己検査: discount ロングの台帳アンカー（long-only base → discount band別 totR/DD）")
    print("  台帳: usdjpy base 1.26 → band0.20 2.18 / band0.00 3.47 ; eurusd 2.56→3.80 ; gbpusd 0.25→1.90")
    print(f"  {'銘柄':8s} {'base(素L)':>10} {'band0.00':>10} {'band0.10':>10} {'band0.20':>10} {'band0.30':>10}")
    for name in ("eurusd", "gbpusd", "usdjpy"):
        df, tarr, dates, span = load_prepped(name)
        S = canonical_setups(df, tarr, dates, 0)
        pool = trade_pool(df, S, name)
        P = pd_frame(df)
        J = join_days(sorted(pool["long"]), P)
        base = [(d, pool["long"][d]) for d in J.index if d in pool["long"]]
        b = sc(base)
        cells = []
        for band in BANDS:
            g = gate_discount_long(pool["long"], J, "pos10", band)
            s = sc(g)
            cells.append(f"{s['rdd']:5.2f}n{s['n']:<4d}" if s else "   n/a   ")
        print(f"  {name:8s} {b['rdd']:6.2f}n{b['n']:<4d} " + " ".join(f"{c:>10}" for c in cells))
