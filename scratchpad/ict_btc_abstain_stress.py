"""BTC の「迷う日（前日が十字線）は入らない」に決定的なストレスをかける（2026-07-15）。

ict_abstain.py で引っかかったセル:
  母集団 = 狩り(sweep) + MSS + 浅い押し目 f=0.25 + RR4 + NYキルゾーン（ASK約定0.3pip・手数料0）
  方向   = 前日の日足の陰陽（動画のルール）
  棄権   = 前日の実体比 |C-O|/(H-L) が下位 q% の日は入らない（＝十字線＝迷いの日）
  → q=35%: net +0.399 / PF 1.56 / totR/DD 3.69（ランダム間引き帰無の 100%ile）

疑うべき点を全部潰す:
  1. n/年 の再計算（前回 26.5年で割っていた。BTC は 2018-2026 の 9年）
  2. 年別 net totR（一部の年に集中していないか）
  3. 巡回ブロック・ブートストラップ（1/3/6/12か月）
  4. プラセボ窓（+8h）— この効果はキルゾーンに依存するのか、それとも時間帯と無関係か
  5. ★ベータ対照 — BTC は上昇ドリフトがある。「同じ日に、同じ損切り幅で、ただ寄りで買う」だけの
     ナイーブ版と比べる（＝入口の構造が効いているのか、ただ BTC が上がっただけか）
  6. 多重比較の自白: 4銘柄×3方向×5明確さ×4閾値 = 240セル引いた中の1つである
  7. 閾値の台地（20/35/50/65%）と、方向定義をまたぐ一貫性
  8. 約定スプレッドのストレス（$15 / $25）

Run: .venv/bin/python scratchpad/ict_btc_abstain_stress.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, BUF
from ict_abstain import daily_frame, join_days, sc, random_drop_null

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
NAME = "btcusd"


def pool_of(df, setups, spread, cost):
    p = {"long": {}, "short": {}}
    for side in ("long", "short"):
        for (d, net, g, risk) in walk(df, setups, F, RR, BUF, spread, cost, side):
            p[side][d] = net
    return p


def build_trades(pool, J, dircol, clarcol, q):
    thr = J[clarcol].quantile(q) if q > 0 else -np.inf
    out = []
    for d, row in J.iterrows():
        v = row[dircol]
        if pd.isna(v):
            continue
        if q > 0 and (pd.isna(row[clarcol]) or row[clarcol] < thr):
            continue
        side = "long" if bool(v) else "short"
        if d in pool[side]:
            out.append((d, pool[side][d]))
    return out


def block_boot(tr, months, nrep=3000):
    s = pd.Series([t[1] for t in tr], index=pd.to_datetime([t[0] for t in tr])).sort_index()
    groups = [g.values for _, g in s.groupby(s.index.to_period("M"))]
    nb = max(1, len(groups) // months)
    blocks = [np.concatenate(groups[i*months:(i+1)*months]) for i in range(nb)
              if len(groups[i*months:(i+1)*months])]
    blocks = [b for b in blocks if len(b)]
    if len(blocks) < 4:
        return np.nan
    w = sum(1 for _ in range(nrep)
            if np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]).sum() > 0)
    return 100.0 * w / nrep


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[NAME])
    df, tarr, dates = prep(df)
    yrs = pd.to_datetime(df["broker_dt"]).dt.year.value_counts()
    span = int((yrs > 5000).sum())
    _, cost = MODEL[NAME]
    S0 = build(df, tarr, dates, True, True, "mss", 0)
    S8 = build(df, tarr, dates, True, True, "mss", 8)
    D = daily_frame(df)

    print(f"BTC: 実質データのある年数 = {span}年（前回は全銘柄を26.5年で割っていた＝n/年が過小報告）")
    print("=" * 118)
    print("1/7/8. 閾値の台地 × 方向定義 × 約定スプレッド（net / PF / totR/DD / ランダム間引き帰無の%ile）")
    print("=" * 118)
    for spread in (15.0, 25.0):
        p0 = pool_of(df, S0, spread, cost)
        J = join_days(sorted(set(list(p0["long"]) + list(p0["short"]))), D)
        print(f"\n--- 約定スプレッド ${spread:.0f}（BTC は手数料0・スプレッドのみ） ---")
        print(f"  {'方向':12s} {'棄権':>5} {'n':>5} {'年':>5} {'net':>7} {'PF':>5} {'totR':>7} "
              f"{'DD':>6} {'totR/DD':>8} {'null中央':>8} {'%ile':>5}")
        for dircol, dlab in (("dir_body", "前日陽線(動画)"), ("dir_kama", "日足KAMA↑"), ("dir_sma", "日足SMA150↑")):
            base = build_trades(p0, J, dircol, "E4_body", 0.0)
            b = sc(base)
            if b:
                print(f"  {dlab:12s} {'0%':>5} {b['n']:5d} {b['n']/span:5.1f} {b['net']:+7.3f} "
                      f"{b['pf']:5.2f} {b['tot']:+7.1f} {b['dd']:6.1f} {b['rdd']:8.2f}")
            for q in (0.20, 0.35, 0.50, 0.65):
                tr = build_trades(p0, J, dircol, "E4_body", q)
                s = sc(tr)
                if not s:
                    continue
                nul = random_drop_null(base, s["n"])
                pc = 100 * (s["rdd"] > nul).mean()
                star = " *" if pc >= 90 else ""
                print(f"  {dlab:12s} {int(q*100):4d}% {s['n']:5d} {s['n']/span:5.1f} {s['net']:+7.3f} "
                      f"{s['pf']:5.2f} {s['tot']:+7.1f} {s['dd']:6.1f} {s['rdd']:8.2f} "
                      f"{np.median(nul):8.2f} {pc:4.0f}%{star}")

    # ---- 主役セル: 前日陽線 × E4 35% ----
    p0 = pool_of(df, S0, 15.0, cost)
    J = join_days(sorted(set(list(p0["long"]) + list(p0["short"]))), D)
    tr = build_trades(p0, J, "dir_body", "E4_body", 0.35)
    s = sc(tr)
    print("\n" + "=" * 118)
    print(f"2/3. 主役セル（前日陽線 × 十字線35%を棄権）: n={s['n']} 年{s['n']/span:.1f}本 "
          f"PF={s['pf']:.2f} net={s['net']:+.3f} totR={s['tot']:+.1f} maxDD={s['dd']:.1f}R")
    print("=" * 118)
    yr = pd.Series([t[1] for t in tr]).groupby(pd.to_datetime([t[0] for t in tr]).year).agg(["sum", "size"])
    print("  年別 totR(本数): " + "  ".join(f"{int(y)}:{r['sum']:+.0f}({int(r['size'])})" for y, r in yr.iterrows()))
    print("  巡回ブロック・ブートストラップ(totR>0の割合): " +
          " / ".join(f"{m}か月 {block_boot(tr, m):.0f}%" for m in (1, 3, 6, 12)))

    # ---- 4. プラセボ窓 ----
    print("\n" + "=" * 118)
    print("4. プラセボ窓（アジア/ロンドン/キルゾーンをセットで +8h ずらす）")
    print("=" * 118)
    p8 = pool_of(df, S8, 15.0, cost)
    J8 = join_days(sorted(set(list(p8["long"]) + list(p8["short"]))), D)
    for q in (0.0, 0.35, 0.65):
        a = sc(build_trades(p0, J, "dir_body", "E4_body", q))
        b = sc(build_trades(p8, J8, "dir_body", "E4_body", q))
        fa = f"n={a['n']:4d} net={a['net']:+.3f} PF={a['pf']:.2f} totR/DD={a['rdd']:5.2f}" if a else "n<20"
        fb = f"n={b['n']:4d} net={b['net']:+.3f} PF={b['pf']:.2f} totR/DD={b['rdd']:5.2f}" if b else "n<20"
        pm = f"{a['net']-b['net']:+.3f}" if (a and b) else "n/a"
        print(f"  棄権{int(q*100):3d}%  本物: {fa}   +8h: {fb}   premium={pm}")

    # ---- 5. ベータ対照 ----
    print("\n" + "=" * 118)
    print("5. ベータ対照 — 「同じ日・同じ損切り幅で、キルゾーンの寄りに成行で建てる」だけの版")
    print("   構造（狩り+MSS+押し目指値）が効いているのか、ただ BTC が上がっただけかを分ける")
    print("=" * 118)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    thr = J["E4_body"].quantile(0.35)
    naive = []
    for rec in S0:
        d = rec["date"]
        if d not in J.index:
            continue
        row = J.loc[d]
        if pd.isna(row["dir_body"]) or pd.isna(row["E4_body"]) or row["E4_body"] < thr:
            continue
        side = "long" if bool(row["dir_body"]) else "short"
        st = rec[side]
        if st is None:
            continue
        k0, k1 = st["kz"]
        if k0 >= k1:
            continue
        entry = o[k0]
        risk = abs(entry - (st["L"] - BUF * st["atr"])) if side == "long" \
            else abs((st["H"] + BUF * st["atr"]) - entry)
        if risk <= 0:
            continue
        stop = entry - risk if side == "long" else entry + risk
        tgt = entry + RR * risk if side == "long" else entry - RR * risk
        R = None
        for p in range(k0, min(k0 + 500, len(c))):
            if side == "long":
                if l[p] <= stop: R = -1.0; break
                if h[p] >= tgt: R = RR; break
            else:
                if h[p] >= stop: R = -1.0; break
                if l[p] <= tgt: R = RR; break
        if R is None:
            R = ((c[min(k0+500, len(c))-1] - entry) if side == "long"
                 else (entry - c[min(k0+500, len(c))-1])) / risk
        naive.append((d, R - 15.0 / risk))
    nb = sc(naive)
    print(f"  押し目指値（本物）: n={s['n']:4d} net={s['net']:+.3f} PF={s['pf']:.2f} totR/DD={s['rdd']:5.2f}")
    if nb:
        print(f"  成行・寄り（対照）: n={nb['n']:4d} net={nb['net']:+.3f} PF={nb['pf']:.2f} "
              f"totR/DD={nb['rdd']:5.2f}")

    print("\n" + "=" * 118)
    print("6. 多重比較の自白: このセルは 4銘柄 × 3方向 × 5明確さ × 4閾値 = 240 セルの中から出てきた。")
    print("   240回引けば 95%ile は期待値12セル出る。判定は「台地か」「方向定義をまたぐか」「別の窓でも出るか」で行う。")
    print("=" * 118)


if __name__ == "__main__":
    main()
