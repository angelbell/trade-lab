"""gold ロングの「muddy(重なり合う)日は入るな」に決定的ストレスをかける（2026-07-15）。

ict_pd_bias.py の結果:
  - ゲートA（ICT 本命 = premium/discount 極性: discount→ロング）は全滅。台地なし、
    ランダム間引き %ile が 90 未満、しかも gold/btc では *逆極性(トレンド追随)* の方が良い
    ＝ ICT の平均回帰バイアスは狩り+MSS の上に価値を足さない（sweep+MSS が既に明確さ検出器）。
  - ゲートB（muddy 棄権・サイド固定 = ユーザーの「不明な日は入るな」）は gold ロングだけ点灯:
    base totR/DD -0.23 → march5 棄権35% で +1.38（%ile 99）/ 棄権50% +1.15（%ile 95）。台地。

だが: 4銘柄×2サイド×2定義×3閾値=48セル中の1ブロック。random-drop は必要条件どまり。
決定的に潰す:
  1. 年別 totR（9年しかない。1年に集中していないか）
  2. 巡回ブロック・ブートストラップ 1/3/6/12か月（別の月の並びでも totR>0 か）
  3. プラセボ窓：同じ muddy 棄権を、キルゾーンを +4/+8/+12h ずらした偽窓にも適用。
     偽窓でも同じだけ持ち上がるなら「muddy な gold 日が悪いだけ」でキルゾーン非依存。
  4. 執行コスト・スプレッドのストレス（$1.5 / $3.0 / $6.0）
  5. march(行進度)の連続性：閾値を絶対値でも振る（分位でなく）＝定義の頑健性

Run: .venv/bin/python experiments/ict_muddy_gold.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, BUF
from ict_abstain import join_days, sc, random_drop_null
from ict_pd_bias import pd_frame
from breakout_wave import resample

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
NAME = "gold"


def long_pool(df, setups, spread_pips):
    sp = spread_pips * PIP[NAME]; _, cost = MODEL[NAME]
    return {d: net for (d, net, g, risk) in walk(df, setups, F, RR, BUF, sp, cost, "long")}


def muddy_skip(pool, J, marchcol, q):
    thr = J[marchcol].quantile(q) if q > 0 else -np.inf
    out = []
    for d, row in J.iterrows():
        if d not in pool:
            continue
        m = row[marchcol]
        if q > 0 and (pd.isna(m) or m < thr):
            continue
        out.append((d, pool[d]))
    return out


def block_boot(tr, months, nrep=3000):
    s = pd.Series([t[1] for t in tr], index=pd.to_datetime([t[0] for t in tr])).sort_index()
    groups = [g.values for _, g in s.groupby(s.index.to_period("M"))]
    nb = max(1, len(groups) // months)
    blocks = [np.concatenate(groups[i * months:(i + 1) * months]) for i in range(nb)
              if len(groups[i * months:(i + 1) * months])]
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
    span = int((pd.to_datetime(df["broker_dt"]).dt.year.value_counts() > 5000).sum())
    P = pd_frame(df)

    # 本窓 + プラセボ窓
    pools = {}
    for sh in (0, 4, 8, 12):
        S = build(df, tarr, dates, True, True, "mss", sh)
        pools[sh] = long_pool(df, S, 1.5)
    J = join_days(sorted(pools[0].keys()), P)

    base = [(d, pools[0][d]) for d in J.index if d in pools[0]]
    b = sc(base)
    print(f"=== gold ロング muddy 棄権ストレス ===  ({span}年)  base: n={b['n']} 年{b['n']/span:.0f}本 "
          f"PF={b['pf']:.2f} net={b['net']:+.3f} totR={b['tot']:+.1f} totR/DD={b['rdd']:.2f}")

    # ---- 1. 主役セル: march5 棄権35% / 50% ----
    print("\n" + "=" * 100)
    print("1/2/4. 主役セル × 年別 × ブロック・ブートストラップ × コスト")
    print("=" * 100)
    for mk, q in (("march5", 0.35), ("march5", 0.50), ("march3", 0.35)):
        for spread in (1.5, 3.0, 6.0):
            pool = long_pool(df, build(df, tarr, dates, True, True, "mss", 0), spread)
            tr = muddy_skip(pool, J, mk, q)
            s = sc(tr)
            nul = random_drop_null(base, s["n"])
            pc = 100 * (s["rdd"] > nul).mean()
            print(f"  {mk} 棄権{int(q*100)}% spread${spread:<4} n={s['n']:4d} 年{s['n']/span:4.1f} "
                  f"PF={s['pf']:.2f} net={s['net']:+.3f} totR={s['tot']:+6.1f} totR/DD={s['rdd']:5.2f} "
                  f"間引き%ile={pc:3.0f}%")
        # 年別・ブロック（$1.5 で）
        pool = long_pool(df, build(df, tarr, dates, True, True, "mss", 0), 1.5)
        tr = muddy_skip(pool, J, mk, q)
        yr = pd.Series([t[1] for t in tr]).groupby(pd.to_datetime([t[0] for t in tr]).year).sum()
        print(f"    └ 年別 totR: " + "  ".join(f"{int(y)}:{v:+.0f}" for y, v in yr.items()))
        print(f"    └ ブロック・ブートストラップ(totR>0%): " +
              " / ".join(f"{m}か月 {block_boot(tr, m):.0f}%" for m in (1, 3, 6, 12)))
        print()

    # ---- 3. プラセボ窓 ----
    print("=" * 100)
    print("3. プラセボ窓（キルゾーンを +4/+8/+12h ずらして同じ muddy 棄権を適用）")
    print("   偽窓でも同じだけ持ち上がるなら、キルゾーン非依存＝「muddy な gold 日が悪いだけ」")
    print("=" * 100)
    for mk, q in (("march5", 0.35), ("march5", 0.50)):
        print(f"  [{mk} 棄権{int(q*100)}%]")
        for sh in (0, 4, 8, 12):
            Js = join_days(sorted(pools[sh].keys()), P)
            base_s = sc([(d, pools[sh][d]) for d in Js.index if d in pools[sh]])
            tr = muddy_skip(pools[sh], Js, mk, q)
            s = sc(tr)
            tag = "本窓" if sh == 0 else f"+{sh}h"
            lift = s["rdd"] - base_s["rdd"] if (s and base_s) else np.nan
            print(f"    {tag:5s} base totR/DD={base_s['rdd']:6.2f} → 棄権後={s['rdd']:6.2f} "
                  f"(net {base_s['net']:+.3f}→{s['net']:+.3f})  持ち上げ={lift:+.2f}")


if __name__ == "__main__":
    main()
