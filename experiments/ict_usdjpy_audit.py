"""USDJPY discount-long の詰め: 過学習監査(PBO/CSCV・Deflated Sharpe) + ブック独立性 + band 台地。

横展開で USDJPY が最頑健と出た（base 1.26 → discount band0.00 で totR/DD 3.47・%ile97・全時代+・
ブロック99）。採用前の標準ゲートを通す:
  1. PBO via CSCV（IS 最良設定が OOS で沈むか）: overfit_audit.cscv を USDJPY discount-long の
     108設定グリッド(band×L×RR×押し目)に当てる。PBO<0.2 = 頑健。
  2. Deflated Sharpe（N試行の運の最大値を引いた後も SR が生きるか）: overfit_audit.psr/sr0。
  3. band/L 台地の再確認（realistic コスト）。
  4. ブック6脚との年別R相関（法則6: 独立なら価値／冗長なら焼き直し）。

母集団: 狩り+MSS + NYキルゾーン + ロング + ASK指値約定(0.9pip RT)。
Run: .venv/bin/python experiments/ict_usdjpy_audit.py
"""
import sys, io, contextlib, itertools
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import ict_ablation as A
import ict_v2_mss as V
from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk
from ict_ablation import build, BUF
from ict_abstain import join_days, sc
from ict_pd_bias import pd_frame
from research.overfit_audit import psr, sr0, cscv
from book_deployed_spec import build as book_build, SIX

RNG = np.random.default_rng(20260715)
NAME = "usdjpy"
SPREAD = 0.3   # realistic: 0.3pip spread + 0.6 commission = 0.9pip RT
BANDS = [0.0, 0.10, 0.20, 0.30]
POSCOLS = ["pos10", "pos20", "pos40"]
RRS = [3.0, 4.0, 5.0]
FRACS = [0.20, 0.25, 0.30]


def long_trades(df, S, F, rr, sp, cost):
    return {d: net for (d, net, g, risk) in walk(df, S, F, rr, BUF, sp, cost, "long")}


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[NAME], cut2000=True)
    df, tarr, dates = prep(df)
    span = int((pd.to_datetime(df["broker_dt"]).dt.year.value_counts() > 5000).sum())
    P = pd_frame(df)
    S = build(df, tarr, dates, True, True, "mss", 0)
    sp = SPREAD * A.PIP[NAME]; _, cost = V.MODEL[NAME]

    # ---- 設定グリッド: (F,rr) ごとに walk、(band,poscol) で discount 濾過 ----
    print(f"USDJPY discount-long 監査（{span}年）。グリッド = "
          f"{len(BANDS)}band × {len(POSCOLS)}L × {len(RRS)}RR × {len(FRACS)}押し目 = "
          f"{len(BANDS)*len(POSCOLS)*len(RRS)*len(FRACS)}設定")
    cols, srs, flagship = {}, [], None
    cid = 0
    for F, rr in itertools.product(FRACS, RRS):
        L = long_trades(df, S, F, rr, sp, cost)
        J = join_days(sorted(L), P)
        posmap = {d: J.loc[d] for d in J.index}
        for band, poscol in itertools.product(BANDS, POSCOLS):
            disc = [(d, L[d]) for d in J.index
                    if d in L and not pd.isna(posmap[d][poscol]) and posmap[d][poscol] < 0.5 - band]
            if len(disc) < 20:
                continue
            r = np.array([x[1] for x in disc])
            s = pd.Series(r, index=pd.to_datetime([x[0] for x in disc]))
            cols[f"c{cid}"] = s.groupby(s.index.to_period("M")).sum()
            srs.append(r.mean() / r.std(ddof=1))
            if abs(F - 0.25) < 1e-9 and abs(rr - 4.0) < 1e-9 and band == 0.0 and poscol == "pos10":
                flagship = disc
            cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0).values
    Vsr = float(np.var(srs))

    # ---- 1. PBO via CSCV ----
    print("\n" + "=" * 78)
    print("1. PBO via CSCV（IS 最良設定が OOS で中央値以下に沈む確率）。<0.2=頑健、~0.5=過学習")
    pbo, oos_med, ploss = cscv(M)
    pbo_n, _, _ = cscv(RNG.standard_normal((M.shape[0], M.shape[1])))
    print(f"  USDJPY disc-long grid={M.shape[1]}cfg × {M.shape[0]}mo  "
          f"PBO={pbo:.2f}  IS-best 平均OOS-Sharpe={oos_med:+.2f}  P(OOS損)={ploss:.2f}")
    print(f"  [ノイズ健全性] 同形状の乱数行列 PBO={pbo_n:.2f}（~0.50 なら CSCV 実装は正常）")

    # ---- 2. Deflated Sharpe on flagship ----
    print("\n" + "=" * 78)
    print("2. Deflated Sharpe（flagship = band0.0/pos10/RR4/押し目0.25）。DSR>0.95 = 試行数の運を引いても生存")
    r = np.array([x[1] for x in flagship])
    p0, sr, g1, g4 = psr(r, 0.0)
    tstat = sr * np.sqrt(len(r))
    Ns = [1, 25, 50, 100, 108, 200]
    dsrs = [psr(r, sr0(N, Vsr))[0] for N in Ns]
    print(f"  n={len(r)} SR/tr={sr:.3f} t={tstat:.2f} skew={g1:.2f} kurt={g4:.1f}  V_SR={Vsr:.4f}")
    print("  " + "  ".join(f"DSR@{N}={d:.2f}" for N, d in zip(Ns, dsrs)))

    # ---- 3. band/L 台地 ----
    print("\n" + "=" * 78)
    print("3. band × L 台地（realistic コスト、totR/DD（間引き%ile は割愛、上の横展開で確認済み））")
    Lbase = long_trades(df, S, 0.25, 4.0, sp, cost)
    J = join_days(sorted(Lbase), P)
    base = sc([(d, Lbase[d]) for d in J.index if d in Lbase])
    print(f"  base(素ロング) totR/DD={base['rdd']:.2f}  n={base['n']}")
    print(f"  {'':8s} " + " ".join(f"{b:>14}" for b in ("band0.00", "band0.10", "band0.20", "band0.30")))
    for poscol in POSCOLS:
        row = []
        for band in BANDS:
            disc = [(d, Lbase[d]) for d in J.index
                    if d in Lbase and not pd.isna(J.loc[d][poscol]) and J.loc[d][poscol] < 0.5 - band]
            s = sc(disc)
            row.append(f"{s['rdd']:5.2f} n{s['n']:<4d}" if s else "   n/a   ")
        print(f"  {poscol:8s} " + " ".join(f"{c:>14}" for c in row))

    # ---- 4. ブック6脚との年別R相関 ----
    print("\n" + "=" * 78)
    print("4. ブック6脚との年別R相関（法則6: 独立なら価値／冗長なら焼き直し）")
    disc = [(d, Lbase[d]) for d in J.index
            if d in Lbase and not pd.isna(J.loc[d]["pos10"]) and J.loc[d]["pos10"] < 0.5]
    jpy_ann = pd.Series([x[1] for x in disc],
                        index=pd.to_datetime([x[0] for x in disc])).groupby(
                        lambda t: t.year).sum()
    legs = book_build(200, 4.5)
    print(f"  {'脚':14s} {'年別R相関':>9}  (共通年数)")
    for k in SIX:
        la = legs[k].groupby(lambda t: t.year).sum()
        j, l = jpy_ann.align(la, join="inner")
        if len(j) >= 4:
            print(f"  {k:14s} {j.corr(l):+9.2f}  ({len(j)})")


if __name__ == "__main__":
    main()
