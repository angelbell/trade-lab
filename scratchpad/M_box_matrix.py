"""M_squeeze_matrix.py の続き: 投稿の「箱」(レジスタンス直下に貼り付いた狭い横ばい)を、粗い
TR比版(comp)とは別に忠実実装し、同じ銘柄xTFマトリクスで並走させる。TR比版は「どこかで静かに
なった」だけで、投稿の「箱」形状(狭い・かつ水準の直下に張り付いている)を捉えていない可能性が
あるため。

流用(車輪の再発明禁止): M_squeeze_matrix.py の load_cell/INSTR_TFS/run_bare_breakout、
M_squeeze_screen.py の mfe_stop_only/layer_stats/random_subset_null/block_boot_diff を import。
新規実装は「箱」判定(box_flag)のみ。

箱の定義(先読み厳禁、ブレイク確定足 b より前の確定足のみ; b自体・以降は不使用):
  box_high = max(high[b-K..b-1]), box_low = min(low[b-K..b-1])
  ATR = 確定ATR14 at b-1
  条件1(狭い箱)     : (box_high - box_low) < c * ATR            (c = 1.5 固定)
  条件2(レベル貼り付き): dist = (e_px - box_high) / ATR が [0, d] の範囲
  箱あり = 条件1 かつ 条件2

K は TF依存(中心値 + 小スイープ):
  5min:[10,14,20]center14 / 15min:[8,10,14]center10 / 1h:[4,6,8]center6 /
  4h:[3,4,6]center4 / 1d:[3,4,5]center4
primary = 中心K, c=1.5, d=0.5。d スイープ = [0.25,0.5,1.0,1.5,2.0]。

重い演算(全d x 全K x 33セル)は避け、以下の2段構成にする(仕様カードの許容範囲内):
  - primary(中心K, d=0.5) を全33セルで実行 -> メインのマトリクス表
  - d スイープ・K スイープは代表セルのみ(BTC全TF・gold全TF・FX代表=eurusd全TF)

Run:
  .venv/bin/python scratchpad/M_box_matrix.py --smoke
  .venv/bin/python scratchpad/M_box_matrix.py
"""
import sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from M_squeeze_screen import mfe_stop_only, layer_stats, random_subset_null, block_boot_diff
from M_squeeze_matrix import load_cell, INSTR_TFS, run_bare_breakout

K_CENTER = {"5min": 14, "15min": 10, "1h": 6, "4h": 4, "1d": 4}
K_SWEEP = {"5min": [10, 14, 20], "15min": [8, 10, 14], "1h": [4, 6, 8],
           "4h": [3, 4, 6], "1d": [3, 4, 5]}
D_SWEEP = [0.25, 0.5, 1.0, 1.5, 2.0]
D_PRIMARY = 0.5
C_FIXED = 1.5
REP_CELLS = ([("gold", tf) for tf in INSTR_TFS["gold"]]
             + [("btcusd", tf) for tf in INSTR_TFS["btcusd"]]
             + [("eurusd", tf) for tf in INSTR_TFS["eurusd"]])


# ---------------------------------------------------------------- cached per-cell base data
def build_cell_base(sym, tf, smoke=False):
    d = load_cell(sym, tf, smoke=smoke)
    if d is None or len(d) < 500:
        return None
    t = run_bare_breakout(d)
    if t is None or len(t) < 20:
        return None
    MFE, idx = mfe_stop_only(d, t)
    atrv = ta.atr(d["high"], d["low"], d["close"], 14).values
    h, l = d["high"].values, d["low"].values
    e_px = t["e_px"].values
    times = t["time"].values
    return dict(d=d, MFE=MFE, idx=idx, atrv=atrv, h=h, l=l, e_px=e_px, times=times)


# ---------------------------------------------------------------- box flag (先読み無し: b以前の確定足のみ)
def box_flag(base, K, c=C_FIXED, d=D_PRIMARY):
    idx, atrv, h, l, e_px = base["idx"], base["atrv"], base["h"], base["l"], base["e_px"]
    n = len(idx)
    valid = np.zeros(n, dtype=bool)
    box = np.zeros(n, dtype=bool)
    for i in range(n):
        b = idx[i]
        if b - K < 0 or b - 1 < 0 or not (atrv[b - 1] > 0):
            continue
        bh = h[b - K:b].max()
        bl = l[b - K:b].min()
        atr = atrv[b - 1]
        narrow = (bh - bl) < c * atr
        dist = (e_px[i] - bh) / atr
        stick = (dist >= 0.0) and (dist <= d)
        valid[i] = True
        box[i] = narrow and stick
    return valid, box


def sign_of(delta, pct_med):
    if delta > 0 and pct_med >= 60:
        return "順"
    if delta < 0 and pct_med <= 40:
        return "逆"
    return "無"


# ---------------------------------------------------------------- main matrix (primary K,d=0.5, all cells)
def analyze_cell_box(sym, tf, base):
    K = K_CENTER[tf]
    valid, box = box_flag(base, K, c=C_FIXED, d=D_PRIMARY)
    MFE = base["MFE"]
    n = int(valid.sum())
    n_box = int((valid & box).sum())
    if n < 20 or n_box < 10 or (n - n_box) < 10:
        return dict(sym=sym, tf=tf, skip=True, n=n, n_box=n_box, K=K)
    m_all = valid
    m_box = valid & box
    s_all = layer_stats(MFE[m_all])
    s_box = layer_stats(MFE[m_box])
    meds, _ = random_subset_null(MFE[m_all], n_box, reps=1000)
    pct_med = 100 * np.mean(meds < s_box["median"])
    delta = s_box["median"] - s_all["median"]
    outside = (pct_med >= 97.5) or (pct_med <= 2.5)
    return dict(sym=sym, tf=tf, skip=False, K=K, n=n, n_box=n_box,
                med_all=s_all["median"], med_box=s_box["median"], delta=delta,
                p_lt1_box=s_box["p_lt1"], p_lt1_all=s_all["p_lt1"],
                pct_med=pct_med, outside=outside, sign=sign_of(delta, pct_med))


def fmt_row(r):
    if r.get("skip", True):
        return (f"  {r['sym']:<8}{r['tf']:<7}{'K='+str(r.get('K','-')):<6}{'-':>6}{'-':>6}{'-':>9}{'-':>9}"
                f"{'-':>8}{'-':>10}{'-':>10}{'-':>9}{'-':>5}   n={r.get('n',0)} n箱={r.get('n_box',0)} 不足でskip")
    return (f"  {r['sym']:<8}{r['tf']:<7}K={r['K']:<4}{r['n']:>6}{r['n_box']:>6}{r['med_all']:>9.2f}"
            f"{r['med_box']:>9.2f}{r['delta']:>+8.2f}{r['p_lt1_box']:>9.1f}%{r['p_lt1_all']:>8.1f}%"
            f"{r['pct_med']:>8.0f}%ile{r['sign']:>5}")


# ---------------------------------------------------------------- d sweep (representative cells)
def d_sweep_table(bases):
    print(f"\n{'='*100}\nd スイープ（代表セル: gold/btc/eurusd 全TF、primary K・c=1.5固定、d のみ振る）")
    print(f"Δ中央値(箱あり中央値 - 全体中央値) が d とともにどう動くか。貼り付きが効くなら d 小で効果大→大で薄まるはず。")
    print(f"{'='*100}")
    header = f"  {'銘柄':<8}{'TF':<7}{'K':<5}{'n':>6}" + "".join(f"{'d='+str(dd):>10}" for dd in D_SWEEP)
    print(header)
    for sym, tf in REP_CELLS:
        base = bases.get((sym, tf))
        if base is None:
            print(f"  {sym:<8}{tf:<7}{'-':<5}{'-':>6}  (データ不足)")
            continue
        K = K_CENTER[tf]
        row = f"  {sym:<8}{tf:<7}{K:<5}"
        n_ref = None
        cells = []
        for dd in D_SWEEP:
            valid, box = box_flag(base, K, c=C_FIXED, d=dd)
            n = int(valid.sum()); n_box = int((valid & box).sum())
            if n_ref is None:
                n_ref = n
            if n_box < 10 or (n - n_box) < 10:
                cells.append("  n不足")
                continue
            med_all = np.median(base["MFE"][valid])
            med_box = np.median(base["MFE"][valid & box])
            cells.append(f"{med_box - med_all:>+9.2f}")
        print(f"  {sym:<8}{tf:<7}{K:<5}{n_ref if n_ref else '-':>6}" + "".join(f"{c:>10}" for c in cells))


# ---------------------------------------------------------------- K sweep (representative cells)
def k_sweep_table(bases):
    print(f"\n{'='*100}\nK スイープ（代表セル、d=0.5・c=1.5固定、K のみ振る）-- primary近傍で符号が安定するか(プラトー)")
    print(f"{'='*100}")
    for sym, tf in REP_CELLS:
        base = bases.get((sym, tf))
        if base is None:
            continue
        print(f"  {sym} {tf}:")
        for K in K_SWEEP[tf]:
            valid, box = box_flag(base, K, c=C_FIXED, d=D_PRIMARY)
            n = int(valid.sum()); n_box = int((valid & box).sum())
            if n_box < 10 or (n - n_box) < 10:
                print(f"    K={K:<3}  n不足でskip")
                continue
            med_all = np.median(base["MFE"][valid])
            med_box = np.median(base["MFE"][valid & box])
            delta = med_box - med_all
            meds, _ = random_subset_null(base["MFE"][valid], n_box, reps=1000)
            pct = 100 * np.mean(meds < med_box)
            mark = "  <- primary" if K == K_CENTER[tf] else ""
            print(f"    K={K:<3}  n={n:>5}  n箱={n_box:>5}  Δ中央値={delta:>+.2f}  null%ile={pct:>3.0f}%"
                  f"  符号={sign_of(delta, pct)}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("箱検知(忠実版): box_height < 1.5*ATR かつ 0<=(e_px-box_high)/ATR<=0.5(primary) のトレードのみ「溜め(箱)あり」")
    print(f"{'銘柄':<8}{'TF':<7}{'K':<6}{'n':>6}{'n箱':>6}{'MFE中央値base':>9}{'MFE中央値箱':>9}"
          f"{'Δ中央値':>8}{'P(<1R)箱':>9}{'P(<1R)base':>8}{'null%ile':>8}{'符号':>5}")
    print("-" * 112)

    rows = []
    bases = {}
    for sym, tfs in INSTR_TFS.items():
        for tf in tfs:
            base = build_cell_base(sym, tf, smoke=args.smoke)
            if (sym, tf) in REP_CELLS:
                bases[(sym, tf)] = base
            if base is None:
                r = dict(sym=sym, tf=tf, skip=True, n=0, n_box=0, K=K_CENTER[tf])
            else:
                r = analyze_cell_box(sym, tf, base)
            rows.append(r)
            print(fmt_row(r))

    valid_rows = [r for r in rows if not r.get("skip", True)]
    print(f"\n有効セル数: {len(valid_rows)} / {len(rows)}")

    print("\n符号の内訳(箱・忠実版):")
    from collections import Counter
    cnt = Counter(r["sign"] for r in valid_rows)
    for k, v in cnt.items():
        print(f"  {k}: {v}")

    print("\n銘柄別の符号:")
    by_sym = {}
    for r in valid_rows:
        by_sym.setdefault(r["sym"], []).append(f"{r['tf']}:{r['sign']}")
    for sym, signs in by_sym.items():
        print(f"  {sym:<8}: {' '.join(signs)}")

    print("\nTF別の符号:")
    by_tf = {}
    for r in valid_rows:
        by_tf.setdefault(r["tf"], []).append(f"{r['sym']}:{r['sign']}")
    for tf in ["5min", "15min", "1h", "4h", "1d"]:
        if tf in by_tf:
            print(f"  {tf:<7}: {' '.join(by_tf[tf])}")

    out_rows = [r for r in valid_rows if r["outside"]]
    print("\nnull帯の外(97.5%ile超 or 2.5%ile未満)に出たセル(箱・忠実版):")
    if out_rows:
        for r in out_rows:
            print(f"  {r['sym']:<8}{r['tf']:<7}null%ile={r['pct_med']:.0f}  Δ中央値={r['delta']:+.2f}")
            # robustness cross-check via block bootstrap for any candidate hit
            base = bases.get((r["sym"], r["tf"]))
            if base is None:
                base = build_cell_base(r["sym"], r["tf"], smoke=args.smoke)
            if base is not None:
                valid, box = box_flag(base, r["K"], c=C_FIXED, d=D_PRIMARY)
                diffs1 = block_boot_diff(base["times"][valid], base["MFE"][valid], box[valid], k=1, nb=1000)
                diffs12 = block_boot_diff(base["times"][valid], base["MFE"][valid], box[valid], k=12, nb=1000)
                if diffs1 is not None and diffs12 is not None:
                    p1 = 100 * np.mean(diffs1[~np.isnan(diffs1)] > 0)
                    p12 = 100 * np.mean(diffs12[~np.isnan(diffs12)] > 0)
                    print(f"      (裏取り) 巡回ブロック・ブートストラップ P(箱あり中央値>全体中央値): "
                          f"k=1ヶ月={p1:.0f}%  k=12ヶ月={p12:.0f}%")
    else:
        print("  無し(全セル帯の内)")

    d_sweep_table(bases)
    k_sweep_table(bases)


if __name__ == "__main__":
    main()
