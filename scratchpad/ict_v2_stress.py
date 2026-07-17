"""ICT v2 の生存セルに決定的なストレスをかける（2026-07-14）。

v2 で残った2種類:
  (A) EURUSD ロング = 浅い戻り(0.25-0.30) + 遠いRR(3-4)  ＝トレンド継続の形（法則9と一致）
  (B) GBPUSD/AUDUSD ショート = 深い戻り(0.705-0.79) + 近いRR(1.5) ＝フェードの形
(B) は v1 でプラセボ窓が偽エッジを出したときと同じ形（薄い損切り＋ヒゲ約定）。ここを潰す。

S1. 約定スプレッドのストレス: 0.3 / 0.6 / 1.0 / 1.5 / 2.0 pip
    → 薄いヒゲ約定に依存しているなら、スプレッドを厳しくすると崩壊する。
       (A) は損切りが厚いので影響が小さいはず（＝この2つを分離できる）。
S2. 年別 net totR（時代に偏っていないか）
S3. 巡回ブロック・ブートストラップ（1/3/6/12か月ブロック）＝「別の月の並びでも成り立つか」
S4. ドル単一因子の検査: GBPUSDショート と AUDUSDショート は両方「ドル買い」。年別Rの相関を見る。
    さらに EURUSDロング（ドル売り）とも比較＝符号が割れるなら単一ドル因子ではない。
S5. プラセボ窓（ロンドン窓・アジア窓・キルゾーンをセットで +8h ずらす）＝v2の機構でも窓は飾りか。

Run: .venv/bin/python scratchpad/ict_v2_stress.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import ict_v2_mss as V2
from ict_v2_mss import (prep, build_setups, walk, stats, MODEL, F_LIST, RR_LIST)
from ict_killzone import load_ny, SYMS

RNG = np.random.default_rng(20260714)
PIP = {"eurusd": 1e-4, "gbpusd": 1e-4, "audusd": 1e-4, "usdjpy": 1e-2, "gold": 0.1, "btcusd": 1.0}

# 生存セル: (銘柄, サイド, sweep, mss, f, rr, ラベル)
CELLS = [
    ("eurusd", "long",  "either", "wick", 0.25, 4.0, "A: EURUSD L 浅0.25/RR4 (トレンド形)"),
    ("eurusd", "long",  "either", "wick", 0.30, 4.0, "A: EURUSD L 浅0.30/RR4 (トレンド形)"),
    ("eurusd", "long",  "pdl",    "close", 0.705, 2.0, "  EURUSD L 正典(PDL/close)"),
    ("gbpusd", "short", "either", "wick", 0.79, 1.5, "B: GBPUSD S 深0.79/RR1.5 (フェード形)"),
    ("gbpusd", "short", "either", "wick", 0.705, 2.0, "B: GBPUSD S 正典 0.705/RR2"),
    ("gbpusd", "short", "pdl",    "close", 0.705, 2.0, "B: GBPUSD S PDL/close 0.705/RR2"),
    ("audusd", "short", "either", "wick", 0.705, 2.0, "B: AUDUSD S 正典 0.705/RR2"),
    ("audusd", "short", "pdl",    "close", 0.705, 2.0, "B: AUDUSD S PDL/close 0.705/RR2"),
    ("gbpusd", "long",  "either", "wick", 0.25, 4.0, "  GBPUSD L 浅0.25/RR4 (対照)"),
]
SPREAD_PIPS = [0.3, 0.6, 1.0, 1.5, 2.0]


def load_all():
    D, S, SP = {}, {}, {}
    for name in ("eurusd", "gbpusd", "audusd"):
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(SYMS[name])
        df, tarr, dates = prep(df)
        D[name] = (df, tarr, dates)
        y = pd.to_datetime(df["broker_dt"]).dt.year.value_counts()
        SP[name] = int((y > 5000).sum())
    return D, SP


def block_boot(net, dates, months, nrep=2000):
    """巡回ブロック・ブートストラップ: 月ブロックを巡回的に並べ替え、totR>0 の割合を返す。"""
    s = pd.Series(net, index=pd.to_datetime(dates)).sort_index()
    key = s.index.to_period("M")
    groups = [g.values for _, g in s.groupby(key)]
    nb = max(1, len(groups) // months)
    blocks = [np.concatenate(groups[i * months:(i + 1) * months])
              for i in range(nb) if len(groups[i * months:(i + 1) * months])]
    blocks = [b for b in blocks if len(b)]
    if len(blocks) < 4:
        return np.nan
    wins = 0
    for _ in range(nrep):
        pick = [blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]
        if np.concatenate(pick).sum() > 0:
            wins += 1
    return 100.0 * wins / nrep


def main():
    D, SPAN = load_all()
    setup_cache = {}

    def setups_of(name, sw, ms, shift=0):
        key = (name, sw, ms, shift)
        if key not in setup_cache:
            df, tarr, dates = D[name]
            if shift:
                V2.ASIA = (-1, 19 + shift, 0, 2 + shift)
                V2.LONDON = (2 + shift, 7 + shift)
                V2.KZ = (7 + shift, 10 + shift)
            else:
                V2.ASIA, V2.LONDON, V2.KZ = (-1, 19, 0, 2), (2, 7), (7, 10)
            setup_cache[key] = build_setups(df, tarr, dates, sw, ms)[0]
            V2.ASIA, V2.LONDON, V2.KZ = (-1, 19, 0, 2), (2, 7), (7, 10)
        return setup_cache[key]

    print("=" * 122)
    print("S1. 約定スプレッドのストレス（買い指値は BID <= lim - spread でしか約定しない）")
    print("    net meanR / PF / n。生スプレッド 0.3pip が現行。1.0-1.5pip でも生きるか？")
    print("=" * 122)
    print(f"  {'cell':38s} " + "".join(f"{f'{p}pip':>17}" for p in SPREAD_PIPS))
    surv = {}
    for (name, side, sw, ms, f, rr, lab) in CELLS:
        df, _, _ = D[name]
        _, cost = MODEL[name]
        st = setups_of(name, sw, ms)
        cells = []
        for p in SPREAD_PIPS:
            tr = walk(df, st, f, rr, 0.1, p * PIP[name], cost, side)
            s = stats(tr, SPAN[name])
            cells.append(f"{s['net']:+.3f}/{s['pf']:.2f}/{s['n']:4d}".rjust(17) if s else "            n<10")
            if p == 0.3:
                surv[lab] = (tr, s)
        print(f"  {lab:38s} " + "".join(cells))

    print("\n" + "=" * 122)
    print("S2/S3. 年別 net totR と 巡回ブロック・ブートストラップ（totR>0 の割合、%）")
    print("       真の改善はブロックを長くするほど割合が上がる。経路当てはめは50%へ縮む。")
    print("=" * 122)
    for lab, (tr, s) in surv.items():
        if s is None:
            continue
        net = np.array([t[1] for t in tr]); dts = [t[0] for t in tr]
        yr = pd.Series(net).groupby(pd.to_datetime(dts).year).sum()
        bs = {m: block_boot(net, dts, m) for m in (1, 3, 6, 12)}
        print(f"\n  {lab}")
        print(f"    n={s['n']} 年{s['npy']:.0f}本 PF={s['pf']:.2f} net={s['net']:+.3f} "
              f"totR={s['tot']:+.1f} maxDD={s['dd']:.1f}R 黒字年={s['gy']:.0f}%")
        print(f"    ブロック・ブートストラップ  1か月 {bs[1]:.0f}% / 3か月 {bs[3]:.0f}% / "
              f"6か月 {bs[6]:.0f}% / 12か月 {bs[12]:.0f}%")
        print("    年別: " + " ".join(f"{int(y)}:{v:+.0f}" for y, v in yr.items()))

    print("\n" + "=" * 122)
    print("S4. ドル単一因子の検査（年別 net R の相関）")
    print("=" * 122)
    keys = [k for k in surv if surv[k][1] is not None]
    ser = {}
    for k in keys:
        tr = surv[k][0]
        ser[k] = pd.Series([t[1] for t in tr]).groupby(pd.to_datetime([t[0] for t in tr]).year).sum()
    M = pd.DataFrame(ser).fillna(0.0)
    if len(M.columns) > 1:
        C = M.corr()
        print("  " + "  ".join(f"{i:>6d}" for i in range(len(C))))
        for i, k in enumerate(C.columns):
            print(f"  {i}: " + "  ".join(f"{C.iloc[i, j]:+6.2f}" for j in range(len(C))) + f"   <- {k}")

    print("\n" + "=" * 122)
    print("S5. プラセボ窓（アジア/ロンドン/キルゾーンをセットで +8h ずらす）— v2 の機構でも窓は飾りか")
    print("=" * 122)
    for (name, side, sw, ms, f, rr, lab) in CELLS:
        if not lab.startswith(("A:", "B:")):
            continue
        df, _, _ = D[name]
        _, cost = MODEL[name]
        a = stats(walk(df, setups_of(name, sw, ms, 0), f, rr, 0.1, 0.3 * PIP[name], cost, side), SPAN[name])
        b = stats(walk(df, setups_of(name, sw, ms, 8), f, rr, 0.1, 0.3 * PIP[name], cost, side), SPAN[name])
        fa = f"n={a['n']:4d} net={a['net']:+.3f} PF={a['pf']:.2f}" if a else "n<10"
        fb = f"n={b['n']:4d} net={b['net']:+.3f} PF={b['pf']:.2f}" if b else "n<10"
        print(f"  {lab:38s} 本物: {fa:34s} +8h: {fb}")


if __name__ == "__main__":
    main()
