"""ICT 忠実化・優先1: 入口アンカーを固定リトレース f=0.25 から FVG 近位端タップへ張り替える。

背景（忠実性監査 A-1）: 現行は MSS 認定に FVG の"存在"（displacement条件）だけを使い、
約定価格は別座標の lim = H - f*(H-L)（f=0.25固定リトレース）のまま ＝ 二重定義。
ICT 正典の entry は「MSS を通した FVG 帯そのもの」への指値（quant-audit.md 項目4）。

変更点（因果分離: 動かすのは入口アンカーだけ）:
  - ict_population.py: bullish_fvg_size/bearish_fvg_size が (size, (lo_edge,hi_edge)) を返すよう改修、
    build() が MSSを通した最大サイズFVGの帯を rec[side]["fvg_lo"/"fvg_hi"] に保持（use_fvg=True 時のみ）。
  - ict_exec.py walk(): 引数 lim_fn を追加（省略時は従来の f 固定リトレースとビット一致・後方互換）。
    lim_fn(s) が与えられると、その戻り値を指値価格として使う（f は無視）。
  - ict_audit.py placebo_premium(): 同じ lim_fn をパススルー。
  stop=L-0.1ATR・RR=4固定・持ち切り・KZ窓・ASK基準指値・同足損切り優先・狩り+MSS+FVG母集団は全て不変。

アンカー定義（ロング、bullish FVG帯=[c1.high, c3.low]。c3.low>c1.high）:
  top    = c3.low （浅い＝戻り小、hi_edge）
  mid    = (c1.high+c3.low)/2 （CE 50%）
  bottom = c1.high （深い＝全ギャップ埋め、lo_edge）
ショートは鏡像（bearish FVG帯=[c3.high, c1.low]。c1.low>c3.high）。「浅い」は常に MSS 後の直近極値に
近い側なので、ロング top=hi_edge に対しショート top=lo_edge=c3.high（帯の価格としては下端）になる:
  top(浅)    = c3.high （lo_edge）
  mid        = (c3.high+c1.low)/2
  bottom(深) = c1.low （hi_edge）
※ 名前(top/bottom)は「浅い/深い」という意味論を鏡映しており、価格の上下ではない。

base = 現行 f=0.25 固定リトレース（= v4 の d_FVG0.25 と同一設定、fvg_min_atr=0.25 の時に台帳と一致）。
fvg_min_atr は 0.15 を主として全アンカー(base込み)を掃引、0/0.25 は従（頑健性チェック）として同じ
表を併走する。曖昧さのフラグ: 仕様は「fvg_min_atr は0.15を主に」の適用範囲（baseを含むか、FVGアンカー
3種だけか）を明記していなかったため、両読みを吸収できるよう base 含む全4アンカーを3水準とも計算する
（計算コストは許容範囲、情報はどちらの読みに対しても上位互換）。

Run: .venv/bin/python experiments/ict_fvg_anchor.py [--smoke]
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, RR_CANON, walk, stats, sc
from ict_population import canonical_setups, load_prepped
from ict_audit import placebo_premium, block_boot

RNG = np.random.default_rng(20260715)
PRIMARY = ["eurusd", "gbpusd", "usdjpy"]
SECONDARY = ["audusd", "nzdusd", "usdcad", "gold", "btcusd"]
ALL_SYMS = PRIMARY + SECONDARY
MIN_ATRS = [0.15, 0.00, 0.25]      # 0.15 主 / 0,0.25 従
ANCHORS = ["base", "top", "mid", "bottom"]
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]


def fvg_anchor_fn(anchor, side):
    def fn(s):
        lo_e, hi_e = s["fvg_lo"], s["fvg_hi"]
        if side == "long":
            if anchor == "top":
                return hi_e      # 浅い = c3.low
            if anchor == "bottom":
                return lo_e      # 深い = c1.high
            return 0.5 * (lo_e + hi_e)
        else:
            if anchor == "top":
                return lo_e      # 浅い(鏡像) = c3.high
            if anchor == "bottom":
                return hi_e      # 深い(鏡像) = c1.low
            return 0.5 * (lo_e + hi_e)
    return fn


def frac_lim(s, side, f=F_CANON):
    L, H = s["L"], s["H"]
    return (H - f * (H - L)) if side == "long" else (L + f * (H - L))


def era_split(tr):
    out = []
    for a, b in ERAS:
        v = sum(t[1] for t in tr if a <= pd.Timestamp(t[0]).year <= b)
        n = sum(1 for t in tr if a <= pd.Timestamp(t[0]).year <= b)
        out.append((a, b, v, n))
    return out


def fmt_era(tr):
    return "  ".join(f"{a}-{b}:{v:+6.1f}(n={n})" for a, b, v, n in era_split(tr))


def run_ablation(symbols, smoke=False):
    """全銘柄×サイド×min_atr×アンカー の表。戻り値 = dict[(sym,side,min_atr,anchor)] = (n_pop, tr, st)。"""
    results = {}
    for name in symbols:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.25):]
        sp, cost = MODEL[name]
        print(f"\n=== {name} (span={span}年, dates={len(dates)}) ===")
        for ma in MIN_ATRS:
            S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=ma)
            for side in ("long", "short"):
                n_pop = sum(1 for rec in S if rec[side] is not None)
                for anchor in ANCHORS:
                    lim_fn = None if anchor == "base" else fvg_anchor_fn(anchor, side)
                    tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, side, lim_fn=lim_fn)
                    st = stats(tr, span)
                    fr = 100.0 * len(tr) / n_pop if n_pop else float("nan")
                    if st is None:
                        print(f"  min_atr={ma:.2f} {side:5s} {anchor:6s} n_pop={n_pop:5d} "
                              f"n_fill={len(tr):5d}({fr:4.1f}%)  n<10 skip")
                    else:
                        print(f"  min_atr={ma:.2f} {side:5s} {anchor:6s} n_pop={n_pop:5d} "
                              f"n_fill={st['n']:5d}({fr:4.1f}%) n/yr={st['npy']:5.1f} win%={st['win']:5.1f} "
                              f"meanR={st['net']:+.3f} PF={st['pf']:5.2f} totR/DD={st['rdd']:6.2f} "
                              f"IS={st['IS']:+7.0f} OOS={st['OOS']:+7.0f}")
                    results[(name, side, ma, anchor)] = (n_pop, tr, st)
    return results


def deviation_stats(symbols, smoke=False, ma=0.15):
    """FVGアンカー価格が0.25固定リトレースからどれだけ乖離するか（ATR単位、母集団=min_atr=ma）。"""
    print(f"\n{'='*100}\nFVGアンカー vs f=0.25 固定リトレースの乖離（ATR単位、min_atr={ma}）\n{'='*100}")
    for name in symbols:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.25):]
        S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=ma)
        for side in ("long", "short"):
            devs = {a: [] for a in ("top", "mid", "bottom")}
            for rec in S:
                s = rec[side]
                if s is None:
                    continue
                fl = frac_lim(s, side)
                A = s["atr"]
                for a in ("top", "mid", "bottom"):
                    dev = (fvg_anchor_fn(a, side)(s) - fl) / A
                    devs[a].append(dev)
            if not devs["mid"]:
                print(f"  {name:8s} {side:5s} n=0 skip")
                continue
            parts = []
            for a in ("top", "mid", "bottom"):
                x = np.array(devs[a])
                parts.append(f"{a}:med={np.median(x):+.3f} q25={np.percentile(x,25):+.3f} "
                             f"q75={np.percentile(x,75):+.3f}")
            print(f"  {name:8s} {side:5s} n={len(devs['mid']):5d}  " + "  ".join(parts))


def judge_cell(df, tarr, dates, name, side, span, ma, anchor):
    """生存候補セルの審判: ブロック1/3/6/12・プラセボ窓+4/8/12h・era split。"""
    sp, cost = MODEL[name]
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=ma)
    lim_fn = None if anchor == "base" else fvg_anchor_fn(anchor, side)
    tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, side, lim_fn=lim_fn)
    st = stats(tr, span)
    if st is None:
        print(f"  [{name} {side} ma={ma} {anchor}] n<10, skip")
        return
    print(f"\n  --- {name} {side} min_atr={ma} anchor={anchor} ---")
    print(f"  素の数値: n={st['n']} n/yr={st['npy']:.1f} win%={st['win']:.1f} meanR={st['net']:+.3f} "
          f"PF={st['pf']:.2f} totR/DD={st['rdd']:.2f} IS={st['IS']:+.0f} OOS={st['OOS']:+.0f}")
    pp = placebo_premium(df, tarr, dates, name, side, span, use_fvg=True, fvg_min_atr=ma, lim_fn=lim_fn)
    print("  プラセボ窓: " + "  ".join(
        f"+{sh}h(n={pp[sh]['n'] if pp[sh] else 0},net={pp[sh]['net'] if pp[sh] else float('nan'):+.3f},"
        f"PF={pp[sh]['pf'] if pp[sh] else float('nan'):.2f})" for sh in (0, 4, 8, 12)))
    prem = {sh: (st['net'] - pp[sh]['net']) if pp[sh] else float("nan") for sh in (4, 8, 12)}
    print("  窓プレミアム(0h-Xh): " + "  ".join(f"+{sh}h={prem[sh]:+.3f}" for sh in (4, 8, 12)))
    bb = {m: block_boot(tr, m) for m in (1, 3, 6, 12)}
    print("  ブロックブートストラップ P(totR>0): " + "  ".join(f"{m}mo={bb[m]:.0f}%" for m in (1, 3, 6, 12)))
    print("  時代別: " + fmt_era(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 100)
    print("0. 検算アンカー: use_fvg=True, fvg_min_atr=0.25, f=0.25, lim_fn=None (=base) が v4 の d_FVG0.25 を再現するか")
    print("   台帳: usdjpy long n=311/PF1.29, eurusd long n=336, audusd long n=291")
    print("#" * 100)
    for name in ("usdjpy", "eurusd", "audusd"):
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=0.25)
        sp, cost = MODEL[name]
        tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, "long")
        st = stats(tr, span)
        print(f"  {name:8s} long n={st['n']} PF={st['pf']:.3f} meanR={st['net']:+.3f} totR/DD={st['rdd']:.2f}")

    print("\n" + "#" * 100)
    print("1. Ablation: base(f=0.25固定) vs FVG_top/mid/bottom(FVG帯アンカー) × min_atr(0.15主/0,0.25従)")
    print("#" * 100)
    print("\n########## 主軸 (eurusd/gbpusd/usdjpy) ##########")
    r1 = run_ablation(PRIMARY, smoke=args.smoke)
    print("\n########## 対照 (audusd/nzdusd/usdcad/gold/btcusd) ##########")
    r2 = run_ablation(SECONDARY, smoke=args.smoke)
    results = {**r1, **r2}

    deviation_stats(ALL_SYMS, smoke=args.smoke, ma=0.15)

    # 生存候補: min_atr=0.15 の各アンカー(base込み)で meanR>0 かつ PF>1 かつ n_fill>=30(smoke時は10)
    minn = 10 if args.smoke else 30
    print(f"\n{'='*100}\n2. 生存候補セルの審判（min_atr=0.15, meanR>0 かつ PF>1 かつ n>={minn}）\n{'='*100}")
    cand = []
    for (name, side, ma, anchor), (n_pop, tr, st) in results.items():
        if ma != 0.15 or st is None:
            continue
        if st['net'] > 0 and st['pf'] > 1.0 and st['n'] >= minn:
            cand.append((name, side, anchor))
    if not cand:
        print("  生存候補なし（min_atr=0.15 で meanR>0 かつ PF>1 のセルが無い）")
    else:
        cache = {}
        for name, side, anchor in sorted(cand):
            if name not in cache:
                with contextlib.redirect_stderr(io.StringIO()):
                    cache[name] = load_prepped(name)
            df, tarr, dates, span = cache[name]
            judge_cell(df, tarr, dates, name, side, span, 0.15, anchor)


if __name__ == "__main__":
    main()
