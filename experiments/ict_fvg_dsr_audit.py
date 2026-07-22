"""ICT 再設計 項目4 続き — FVG-MSS 濃縮候補（usdjpy long / audusd long）の採用前ゲート。

前段（ict_fvg_mss_audit.py, out_fvg_ablation_full.txt）で FVG 閾値を上げるほど totR/DD が
単調に上がる候補として usdjpy long・audusd long が浮上した（eurusd long は山型で c で天井→d で失速、
btcusd long も単調だが窓プレミアムが薄く24h市場の疑いあり）。これを USDJPY discount-long と
同じ標準ゲート（DSR@N・PBO via CSCV・ブック相関）に通し、「試行数の運」か「本物」かを判定する。

A. overfit_grid: 1銘柄あたり FVG段(exist/0.15/0.25) × RR(2/3/4) × frac(0.20/0.25/0.30) = 27設定。
   月次リターン行列 → CSCV で PBO、flagship(fvg0.25/RR4/frac0.25 = 前段の "d_FVG0.25" と同一設定)で
   Deflated Sharpe を N=27(この銘柄のみ)・N=108(discount-long anchor と同じ試行数で直接比較)ほかで算出。
B. 時代分解（4era）+ USDX 年間騰落との符号関係（記述統計のみ）。
C. 候補4銘柄の年別R相互相関 + 現役ブック6脚との相関 + 同日重複率。

Run:
  .venv/bin/python experiments/ict_fvg_dsr_audit.py --smoke   # 直近25%のみ・動作確認
  .venv/bin/python experiments/ict_fvg_dsr_audit.py           # 本番（全期間）
"""
import sys, io, itertools, contextlib, argparse
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, walk
from ict_population import canonical_setups, load_prepped
from research.overfit_audit import psr, sr0, cscv

RNG = np.random.default_rng(20260715)
FVG_STAGES = [("exist", True, 0.00), ("fvg0.15", True, 0.15), ("fvg0.25", True, 0.25)]
RRS = [2.0, 3.0, 4.0]
FRACS = [0.20, 0.25, 0.30]
CANDIDATES = ["usdjpy", "audusd", "eurusd", "btcusd"]   # usdjpy/audusd=候補、eurusd/btcusd=対照
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]
FLAG_LAB = "fvg0.25/RR4.0/frac0.25"   # = 前段 out_fvg_ablation_full.txt の "d_FVG0.25" と同一設定


def net_map(df, S, name, f, rr):
    sp, cost = MODEL[name]
    return {d: net for (d, net, g, risk) in walk(df, S, f, rr, BUF, sp, cost, "long")}


def build_grid(df, tarr, dates, name):
    """27設定(FVG3×RR3×frac3) -> 月次リターン行列 M・per-cfg SR分散・flagship trades(list of (date,net))。"""
    cols, srs, flagship = {}, [], None
    cid = 0
    for lab, uf, fm in FVG_STAGES:
        S = canonical_setups(df, tarr, dates, 0, use_fvg=uf, fvg_min_atr=fm)
        for rr, f in itertools.product(RRS, FRACS):
            L = net_map(df, S, name, f, rr)
            if len(L) < 20:
                cid += 1
                continue
            r = np.array(list(L.values()))
            s = pd.Series(r, index=pd.to_datetime(list(L.keys())))
            cols[f"c{cid}"] = s.groupby(s.index.to_period("M")).sum()
            srs.append(r.mean() / r.std(ddof=1))
            if lab == "fvg0.25" and abs(rr - 4.0) < 1e-9 and abs(f - 0.25) < 1e-9:
                flagship = sorted(L.items())
            cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0).values
    return M, float(np.var(srs)), flagship


def eras_of(tr):
    return " ".join(
        f"{sum(x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b):+7.1f}"
        if any(a <= pd.Timestamp(x[0]).year <= b for x in tr) else "    n/a"
        for a, b in ERAS)


def annual(tr):
    return pd.Series([x[1] for x in tr],
                     index=pd.to_datetime([x[0] for x in tr])).groupby(lambda t: t.year).sum()


# ============================== A. DSR / PBO ==============================
def part_a(data):
    print("\n" + "=" * 100)
    print("A. Deflated Sharpe / PBO via CSCV — グリッド = FVG段(3)×RR(3)×frac(3) = 27設定/銘柄")
    print("   flagship = " + FLAG_LAB + "（前段 d_FVG0.25 と同一設定）")
    print("   アンカー: USDJPY discount-long  PBO=0.40(ノイズ0.34)  SR/tr=0.099 t=2.12 skew+1.16  "
          "DSR@108=0.56（合格線>0.95）")
    Ns = [1, 25, 27, 50, 100, 108, 200]
    flagships = {}
    for name in CANDIDATES:
        df, tarr, dates, span, M, Vsr, flagship = data[name]
        if flagship is None or len(flagship) < 20:
            print(f"\n  [{name}] flagship n<20, skip"); continue
        flagships[name] = flagship
        pbo, oos_med, ploss = cscv(M)
        pbo_n, _, _ = cscv(RNG.standard_normal(M.shape))
        r = np.array([x[1] for x in flagship])
        p0, sr, g1, g4 = psr(r, 0.0)
        tstat = sr * np.sqrt(len(r))
        dsrs = [psr(r, sr0(N, Vsr))[0] for N in Ns]
        print(f"\n  [{name}] grid={M.shape[1]}cfg x {M.shape[0]}mo  PBO={pbo:.2f}  "
              f"IS-best平均OOS-Sharpe={oos_med:+.2f}  P(OOS損)={ploss:.2f}  [ノイズ健全性PBO={pbo_n:.2f}]")
        print(f"       flagship n={len(r)} SR/tr={sr:.3f} t={tstat:.2f} skew={g1:.2f} kurt={g4:.1f} V_SR={Vsr:.4f}")
        print("       " + "  ".join(f"DSR@{N}={d:.2f}" for N, d in zip(Ns, dsrs)))
    return flagships


# ============================== B. 時代分解 + USDX ==============================
def part_b(data, flagships):
    print("\n" + "=" * 100)
    print("B. 時代分解（4era）: " + "/".join(f"{a}-{b}" for a, b in ERAS))
    for name in CANDIDATES:
        if name not in flagships:
            continue
        print(f"  {name:8s} totR/era: {eras_of(flagships[name])}")

    print("\n  USDX 年間騰落 vs 候補の年別totRの符号（記述統計のみ・USDXは2020年以降のみデータ有）:")
    import os, contextlib as cl
    try:
        from src.data_loader import load_mt5_csv
        usdx = load_mt5_csv("data/vantage_usdx.r_h1.csv")
        usdx_d = usdx["close"].resample("1D").last().dropna()
        usdx_ann = usdx_d.groupby(usdx_d.index.year).apply(lambda s: (s.iloc[-1] / s.iloc[0] - 1) * 100)
        print(f"  USDX annual %chg: " + "  ".join(f"{y}={v:+.2f}%" for y, v in usdx_ann.items()))
        for name in CANDIDATES:
            if name not in flagships:
                continue
            ann = annual(flagships[name])
            common = sorted(set(ann.index) & set(usdx_ann.index))
            if not common:
                continue
            agree = sum(1 for y in common if np.sign(ann[y]) == np.sign(usdx_ann[y]))
            print(f"  {name:8s} 候補totR: " + "  ".join(f"{y}={ann[y]:+.1f}" for y in common)
                  + f"   | USDXと同符号 {agree}/{len(common)}年")
    except Exception as e:
        print(f"  [USDX読み込み失敗: {e}]")


# ============================== C. 独立性 ==============================
def part_c(flagships):
    print("\n" + "=" * 100)
    print("C. 独立性: 候補間の年別R相関 + 現役ブック6脚との相関 + 同日重複率")
    anns = {name: annual(tr) for name, tr in flagships.items()}
    names = list(anns.keys())
    print("\n  候補間 年別R相関:")
    print("  " + " " * 10 + " ".join(f"{n:>10}" for n in names))
    for a in names:
        row = []
        for b in names:
            j, l = anns[a].align(anns[b], join="inner")
            row.append(f"{j.corr(l):+.2f}(n{len(j)})" if len(j) >= 3 else "n/a")
        print(f"  {a:10s} " + " ".join(f"{c:>10}" for c in row))

    print("\n  現役ブック6脚との年別R相関:")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from book_deployed_spec import build as book_build, SIX
        legs = book_build(200, 4.5)
    for name, tr in flagships.items():
        ann = anns[name]
        row = []
        for k in SIX:
            la = legs[k].groupby(lambda t: t.year).sum()
            j, l = ann.align(la, join="inner")
            row.append(f"{k}={j.corr(l):+.2f}" if len(j) >= 4 else f"{k}=n/a")
        print(f"  {name:8s} " + "  ".join(row))

    print("\n  同日重複率（トレード日の交差 / 短い方の日数）:")
    dsets = {n: set(d for d, _ in tr) for n, tr in flagships.items()}
    for a, b in itertools.combinations(names, 2):
        inter = len(dsets[a] & dsets[b])
        denom = min(len(dsets[a]), len(dsets[b]))
        print(f"  {a:8s} x {b:8s}: 交差{inter}日 / min({len(dsets[a])},{len(dsets[b])}) = "
              f"{100*inter/denom:.1f}%")


def main(smoke=False):
    data = {}
    for name in CANDIDATES:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.25):]
        M, Vsr, flagship = build_grid(df, tarr, dates, name)
        data[name] = (df, tarr, dates, span, M, Vsr, flagship)
        print(f"[{name}] span={span}年 grid={M.shape} flagship_n={len(flagship) if flagship else 0}")

    flagships = part_a(data)
    part_b(data, flagships)
    part_c(flagships)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
