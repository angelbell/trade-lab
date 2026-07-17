"""ICT 忠実版 EURUSD 15m ロング — 週足レジーム・ゲート C1/C2（凍結仕様カード, 2026-07-16）。

目的: 連敗が「欠けていた WHEN ゲート」で説明・改善できるか（=WHENレバー候補）、それとも
「一時代ドルベータで救えない」か（=忠実化しても救えない）を白黒つける。

ベース（変更禁止・ict_extliq_target.py の生存セル = ict_fvg_dsr_audit の凍結仕様と同一）:
  EURUSD 15m long-only, 狩り+MSS+FVG(min_atr=0.15)母集団, 入口=FVG-CE(mid)指値・KZ内約定,
  stop=L-0.1*ATR14, target=PDH-5pip（外部流動性）, cost=realistic tier（spread0.3+comm0.6=0.9pip RT）。
  自己検査アンカー: ext_PDH_fluff5 (realistic) = n=313, win%=34.5, PF=1.41, meanR=+0.281, totR/DD=4.05,
  maxDD=21.7 (scratchpad/out_ict_extliq_target.txt に既存)。

足すゲート（ロング許可条件のみ、出口・入口は不変）:
  週足データ = data/vantage_eurusd_w1.csv。各トレードの「実際の約定足の broker 時刻」(trade_log の
  fill_dt。walk() が ASK指値約定した bar の broker_dt = ict_exec.load_ny の naive 時刻。週足 CSV も
  同じ load_mt5_csv→tz_localize(None) 経路の naive broker 時刻なので直接比較できる) を基準に、
  「その時刻より前に完全に閉じた最後の週足バー」= [w-1] を特定する（searchsorted で「開始<=t の
  最後のバー」= 進行中の当該週バー p、[w-1] = p-1。p 自身は閉じていないので絶対に使わない）。
  C1(win): close[w-1] > SMA(win)[w-1]（win∈{20,30}、共に[w-1]で計算=先読み無し）。
  C2     : high[w-1] > high[w-2]（前の確定週足高値を更新）。

再利用（車輪の再発明禁止）: ict_exec.walk/stats/sc, ict_population.canonical_setups/load_prepped,
ict_fvg_anchor.fvg_anchor_fn, ict_extliq_target.make_ext_tgt_fn/cost_tiers,
ict_audit.block_boot/random_drop_null, research.overfit_audit.psr/sr0。
新規で書くのは「週足ゲート判定」（searchsorted+rolling、既存関数の組合せに無い部分）と、
「ゲート版 vs 無ゲートを同じ再標本パスで比べる paired block bootstrap」
（block_boot は単一系列の P(totR>0) しか返さないため、ここだけ新規実装。同じ月グループ化手法を流用）。

自己検査: 本スクリプト実行時に無ゲートが ext_PDH_fluff5 アンカーを再現するか先頭で確認する。

Run: .venv/bin/python scratchpad/ict_weekly_gate_c1c2.py [--smoke] 2>&1 | tee scratchpad/out_ict_weekly_gate_c1c2.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv
from ict_exec import BUF, F_CANON, RR_CANON, walk, stats, sc
from ict_population import canonical_setups, load_prepped
from ict_audit import block_boot, random_drop_null
from ict_fvg_anchor import fvg_anchor_fn
from ict_extliq_target import make_ext_tgt_fn, cost_tiers
from research.overfit_audit import psr, sr0

RNG = np.random.default_rng(20260716)

EURUSD_MA = 0.15
LIM_FN = fvg_anchor_fn("mid", "long")
TGT_FN = make_ext_tgt_fn("pdh", 5, "eurusd", "long")   # objkey = rec["long"]["pdh"]（小文字キー、ict_population.build() 準拠）
YEARS = list(range(2018, 2027))
WSMA = [30, 20]
NREP_BLOCK = 3000


# ---------------------------------------------------------------------------
def build_base_trades(smoke=False):
    """凍結ベース(ext_PDH_fluff5)のトレードを trade_log 付きで再構成する。"""
    df, tarr, dates, span = load_prepped("eurusd")
    if smoke:
        dates = dates[-int(len(dates) * 0.25):]
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA,
                         use_liq=True, liq_ns=(20, 40))
    sp, cost = cost_tiers("eurusd")["realistic"]
    trade_log = []
    tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, "long",
              lim_fn=LIM_FN, tgt_fn=TGT_FN, trade_log=trade_log)
    return df, span, tr, trade_log


def load_weekly():
    dfw = load_mt5_csv("data/vantage_eurusd_w1.csv")
    week_start = dfw.index.tz_localize(None).values.astype("datetime64[ns]")
    close_w = dfw["close"].values.astype(float)
    high_w = dfw["high"].values.astype(float)
    sma = {w: pd.Series(close_w).rolling(w).mean().values for w in WSMA}
    return week_start, close_w, high_w, sma


def confirmed_week_idx(week_start, ts):
    """[w-1] の位置。ts より前に完全に閉じた最後の週足バー（進行中の当該週バー p は絶対に使わない）。"""
    p = np.searchsorted(week_start, np.datetime64(ts), side="right") - 1
    return p - 1


def gate_arrays(trade_log, week_start, close_w, high_w, sma):
    """trade_log の各トレードについて C1(30)/C1(20)/C2 の bool を返す（先読み無し、[w-1]のみ参照）。"""
    n = len(trade_log)
    c1 = {w: np.zeros(n, dtype=bool) for w in WSMA}
    c2 = np.zeros(n, dtype=bool)
    for i, t in enumerate(trade_log):
        wm1 = confirmed_week_idx(week_start, t["fill_dt"])
        if wm1 < 0:
            continue
        for w in WSMA:
            if wm1 >= w - 1 and np.isfinite(sma[w][wm1]):
                c1[w][i] = close_w[wm1] > sma[w][wm1]
        if wm1 >= 1:
            c2[i] = high_w[wm1] > high_w[wm1 - 1]
    return c1, c2


def to_tr(trade_log, mask=None):
    """trade_log(dict list) -> stats()/sc() が食える (date, net, R, risk) タプルへ。"""
    idxs = range(len(trade_log)) if mask is None else np.where(mask)[0]
    return [(trade_log[i]["date"], trade_log[i]["net"], trade_log[i]["R"], 0.0) for i in idxs]


def fmt_stats(label, st):
    if st is None:
        return f"  {label:14s} n<10 skip"
    return (f"  {label:14s} n={st['n']:5d} n/yr={st['npy']:5.1f} win%={st['win']:5.1f} "
            f"PF={st['pf']:5.2f} meanR={st['net']:+.3f} totR={st['tot']:+7.1f} "
            f"maxDD={st['dd']:6.1f} totR/DD={st['rdd']:6.2f} IS={st['IS']:+7.1f} OOS={st['OOS']:+7.1f}")


def year_table(trade_log, mask, years):
    """年別 n/win%/totR（gross R>0 を勝ちと数える既存流儀に合わせる）。"""
    idxs = range(len(trade_log)) if mask is None else np.where(mask)[0]
    yrs = np.array([pd.Timestamp(trade_log[i]["date"]).year for i in idxs])
    net = np.array([trade_log[i]["net"] for i in idxs])
    R = np.array([trade_log[i]["R"] for i in idxs])
    out = {}
    for y in years:
        sel = yrs == y
        n = int(sel.sum())
        if n == 0:
            out[y] = dict(n=0, win=float("nan"), tot=0.0)
        else:
            out[y] = dict(n=n, win=100.0 * (R[sel] > 0).mean(), tot=float(net[sel].sum()))
    return out


def paired_block_boot(trade_log, mask, months, nrep=NREP_BLOCK):
    """block_boot と同じ「月をブロック単位で再標本」を、無ゲート全トレードに対して行い、
    同じ再標本パスの上で ゲート版 vs 無ゲート版 の totR/DD を比べる（レバレッジではなく
    同じ経路での勝ち負けを問う。CLAUDE.md チェックリスト7の「同じ価格経路」思想をペアで適用）。
    戻り値: (P(gate_rdd > nogate_rdd) [%], 有効ブロック数)。ブロック<4は block_boot 同様 NaN。"""
    dates = pd.to_datetime([t["date"] for t in trade_log])
    net = np.array([t["net"] for t in trade_log])
    gate = np.asarray(mask, dtype=bool)
    s = pd.DataFrame({"net": net, "gate": gate}, index=dates).sort_index()
    periods = s.index.to_period("M")
    groups = [g for _, g in s.groupby(periods)]
    nb = max(1, len(groups) // months)
    blocks = [pd.concat(groups[i * months:(i + 1) * months]) for i in range(nb)
              if len(groups[i * months:(i + 1) * months])]
    blocks = [b for b in blocks if len(b)]
    if len(blocks) < 4:
        return float("nan"), len(blocks)
    wins = 0
    valid = 0
    for _ in range(nrep):
        idx = RNG.integers(0, len(blocks), len(blocks))
        path = pd.concat([blocks[i] for i in idx])
        pn = path["net"].values
        cum = np.cumsum(pn); dd_ng = float((np.maximum.accumulate(cum) - cum).max())
        rdd_ng = pn.sum() / dd_ng if dd_ng > 0 else np.inf
        gn = path.loc[path["gate"].values, "net"].values
        if len(gn) < 2:
            continue
        cumg = np.cumsum(gn); dd_g = float((np.maximum.accumulate(cumg) - cumg).max())
        rdd_g = gn.sum() / dd_g if dd_g > 0 else np.inf
        valid += 1
        if rdd_g > rdd_ng:
            wins += 1
    if valid == 0:
        return float("nan"), len(blocks)
    return 100.0 * wins / valid, len(blocks)


def eurusd_annual_pct():
    dfw = load_mt5_csv("data/vantage_eurusd_w1.csv")
    w = dfw.copy()
    w.index = w.index.tz_localize(None)
    w["year"] = w.index.year
    ann = w.groupby("year").agg(o=("open", "first"), c=("close", "last"))
    return (ann["c"] / ann["o"] - 1.0) * 100.0


def annual_totR(trade_log, mask=None):
    idxs = range(len(trade_log)) if mask is None else np.where(mask)[0]
    yrs = pd.Series([trade_log[i]["net"] for i in idxs],
                    index=pd.to_datetime([trade_log[i]["date"] for i in idxs]).year)
    return yrs.groupby(level=0).sum()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("0. 自己検査: 無ゲートが ext_PDH_fluff5 アンカーを再現するか")
    print("   台帳: n=313, win%=34.5, PF=1.41, meanR=+0.281, totR/DD=4.05, maxDD=21.7 (realistic)")
    print("#" * 110)
    df, span, tr0, trade_log = build_base_trades(smoke=args.smoke)
    st0 = stats(tr0, span)
    print(fmt_stats("無ゲート(自己検査)", st0))
    if not args.smoke:
        assert st0["n"] == 313, f"自己検査 n 不一致: {st0['n']} != 313"

    week_start, close_w, high_w, sma = load_weekly()
    c1, c2 = gate_arrays(trade_log, week_start, close_w, high_w, sma)

    versions = {
        "無ゲート": None,
        "C1(30)": c1[30],
        "C1(20)": c1[20],
        "C2(HH)": c2,
    }

    print("\n" + "#" * 110)
    print("1. 無ゲート vs C1(30) vs C1(20) vs C2 — 全史(2000-2026)")
    print("#" * 110)
    trs = {}
    for label, mask in versions.items():
        tr = to_tr(trade_log, mask)
        trs[label] = tr
        st = stats(tr, span)
        print(fmt_stats(label, st))

    print("\n" + "#" * 110)
    print("2. 年別（2018-2026）: n / 許可日ロング win% / totR / ON%（=gate採用trade数 / 無ゲートtrade数）")
    print("#" * 110)
    yt = {label: year_table(trade_log, mask, YEARS) for label, mask in versions.items()}
    for y in YEARS:
        ng = yt["無ゲート"][y]
        line = f"  {y}: 無ゲート n={ng['n']:3d} win%={ng['win']:5.1f} totR={ng['tot']:+7.1f}"
        for label in ("C1(30)", "C1(20)", "C2(HH)"):
            g = yt[label][y]
            onpct = 100.0 * g["n"] / ng["n"] if ng["n"] > 0 else float("nan")
            line += f"  |  {label} n={g['n']:3d}(ON={onpct:5.1f}%) win%={g['win']:5.1f} totR={g['tot']:+6.1f}"
        print(line)

    print("\n" + "#" * 110)
    print("3. 掟の検定")
    print("#" * 110)
    print("\n[3a] random-drop null（無ゲート母集団から同数間引き, n=2000, 必要条件どまり）")
    base_pairs = [(t[0], t[1]) for t in tr0]
    for label in ("C1(30)", "C1(20)", "C2(HH)"):
        tr = trs[label]
        st = stats(tr, span)
        if st is None:
            print(f"  {label:10s} n<10 skip"); continue
        null = random_drop_null(base_pairs, st["n"])
        pct = 100.0 * (null < st["rdd"]).mean()
        print(f"  {label:10s} 実測totR/DD={st['rdd']:6.2f}  間引きnull分布での%ile={pct:5.1f}%  "
              f"(null median={np.median(null):.2f})")

    print("\n[3b] 巡回ブロック・ブートストラップ（1/3/6/12ヶ月, 各%d回, P=ゲート版のtotR/DDが無ゲートを上回る確率）"
          % NREP_BLOCK)
    for label in ("C1(30)", "C1(20)", "C2(HH)"):
        mask = versions[label]
        row = f"  {label:10s} "
        for m in (1, 3, 6, 12):
            p, nb = paired_block_boot(trade_log, mask, m)
            row += f"{m:2d}mo: P={p:5.1f}%(nb={nb:3d})  "
        print(row)

    print("\n[3c] Deflated Sharpe（試行数=3: C1(30)/C1(20)/C2、参考値=サイズ入力・棄却の門ではない）")
    srs = []
    per_variant = {}
    for label in ("C1(30)", "C1(20)", "C2(HH)"):
        tr = trs[label]
        r = np.array([t[1] for t in tr])
        if len(r) < 5:
            continue
        sr_v = r.mean() / r.std(ddof=1)
        srs.append(sr_v)
        per_variant[label] = (r, sr_v)
    if len(srs) >= 2:
        Vsr = float(np.var(srs))
        best_label = max(per_variant, key=lambda k: per_variant[k][1])
        r_best, sr_best = per_variant[best_label]
        p0, sr, g1, g4 = psr(r_best, 0.0)
        Ns = [1, 3, 10, 25]
        dsrs = [psr(r_best, sr0(N, Vsr))[0] for N in Ns]
        print(f"  最良変種={best_label} SR/tr={sr:.3f} skew={g1:.2f} kurt={g4:.1f} V_SR={Vsr:.5f}")
        print("  " + "  ".join(f"DSR@{N}={d:.3f}" for N, d in zip(Ns, dsrs)))
    else:
        print("  変種が不足（n<5）のため DSR 計算をスキップ")

    print("\n" + "#" * 110)
    print("4. 独立性の割引: 年別totR と EURUSD年間騰落 の相関、2024-26単一ブロックの見立て")
    print("#" * 110)
    ann_ret = eurusd_annual_pct()
    for label in ("無ゲート", "C1(30)", "C1(20)", "C2(HH)"):
        at = annual_totR(trade_log, versions[label])
        common = sorted(set(at.index) & set(ann_ret.index))
        common = [y for y in common if 2000 <= y <= 2026]
        if len(common) < 5:
            print(f"  {label:10s} 年数不足で相関スキップ"); continue
        x = at.reindex(common).values
        yv = ann_ret.reindex(common).values
        corr = np.corrcoef(x, yv)[0, 1]
        print(f"  {label:10s} 年別totR vs EURUSD年間騰落%  相関(n={len(common)}年, {common[0]}-{common[-1]}) = {corr:+.3f}")
    print("\n  参考: EURUSD 年間騰落%(2018-2026) = " +
          "  ".join(f"{y}:{ann_ret.get(y, float('nan')):+5.1f}%" for y in YEARS))


if __name__ == "__main__":
    main()
