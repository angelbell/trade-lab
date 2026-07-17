"""M_box_matrix.py の「狭さ」条件を自己正規化に差し替え。

前段(M_box_matrix.py)で c=1.5 の絶対ATR倍率しきいを使ったところ、5min/15min(K=10,14)は
ほぼ全セルで箱あり本数が0〜数本しか出ず(n=2000超のセルでも n箱<10)、4h/1d(K=3,4)ですら
半分近くのFXセルで不足だった。実測: K本レンジ/ATR14 の中央値は K=6で約2.5、K=10で約3.0
と、Kが増えるほど分布そのものが右に伸びる(レンジはKの単調増加関数なのでATRの一定倍数
という絶対しきいはKに対してスケールしない)。これは検知の欠陥であり、対処療法(c を K ごとに
変える)ではなく、そもそも絶対しきいをやめて **自己正規化(その銘柄・TFの過去分布の中での
順位)** に差し替える。

新しい「狭い箱」条件(K非依存・自己正規化・先読み無し):
  box_range[j] = max(high[j-K+1..j]) - min(low[j-K+1..j])   (j = b-1, ブレイク確定足の直前確定足)
  trailing 250本ぶんの box_range 分布( j 以前のみ、window=[j-249..j] )の中での box_range[j] の
  パーセンタイル順位 <= 1/3(primary) なら「狭い」。
「箱あり」 = 狭い(自己正規化) かつ 貼り付き(dist=(e_px-box_high)/ATR <= d)。

流用: M_box_matrix.build_cell_base / K_CENTER / K_SWEEP / D_SWEEP / D_PRIMARY / REP_CELLS /
      sign_of をそのまま import。M_squeeze_matrix.INSTR_TFS。M_squeeze_screen の
      layer_stats / random_subset_null / block_boot_diff。新規実装は narrow 判定のみ。

Run:
  .venv/bin/python scratchpad/M_box_matrix_v2.py --smoke
  .venv/bin/python scratchpad/M_box_matrix_v2.py
"""
import sys, re, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from M_squeeze_screen import layer_stats, random_subset_null, block_boot_diff
from M_squeeze_matrix import INSTR_TFS
from M_box_matrix import build_cell_base, K_CENTER, K_SWEEP, D_SWEEP, D_PRIMARY, REP_CELLS, sign_of

Q_PRIMARY = 1.0 / 3.0
Q_SWEEP = [1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0]
TRAIL_WIN = 250
MIN_VALID = 50


# ---------------------------------------------------------------- 自己正規化「狭さ」
def rolling_box_range(d_df, K):
    """box_range[j] = max(high[j-K+1..j]) - min(low[j-K+1..j])。pandas rolling は末尾バー基準
    (先読み無し=jまでの確定足のみ使用)。"""
    rh = d_df["high"].rolling(K).max().values
    rl = d_df["low"].rolling(K).min().values
    return rh - rl


def narrow_pctrank_at(box_range_all, j, win=TRAIL_WIN, min_valid=MIN_VALID):
    """j 時点の box_range が、[j-win+1 .. j] の trailing 分布(j以前のみ=先読み無し)の中で
    どのパーセンタイルに位置するか。値そのものを含めて順位を取る(自己参照だが未来は見ない)。"""
    if j < 0 or np.isnan(box_range_all[j]):
        return np.nan
    lo = max(0, j - win + 1)
    seg = box_range_all[lo:j + 1]
    seg = seg[~np.isnan(seg)]
    if len(seg) < min_valid:
        return np.nan
    return float(np.mean(seg <= box_range_all[j]))


def box_flag_v2(base, K, q=Q_PRIMARY, d=D_PRIMARY, box_range_all=None):
    idx, atrv, h, e_px = base["idx"], base["atrv"], base["h"], base["e_px"]
    if box_range_all is None:
        box_range_all = rolling_box_range(base["d"], K)
    n = len(idx)
    valid = np.zeros(n, dtype=bool)
    box = np.zeros(n, dtype=bool)
    pcts = np.full(n, np.nan)
    for i in range(n):
        b = idx[i]
        j = b - 1
        if j < 0 or b - K < 0 or not (atrv[j] > 0):
            continue
        pct = narrow_pctrank_at(box_range_all, j)
        if np.isnan(pct):
            continue
        bh = h[b - K:b].max()
        dist = (e_px[i] - bh) / atrv[j]
        stick = (dist >= 0.0) and (dist <= d)
        narrow = pct <= q
        valid[i] = True
        pcts[i] = pct
        box[i] = narrow and stick
    return valid, box, pcts


# ---------------------------------------------------------------- coarse TR-ratio matrix (for cross-reference)
def load_tr_signs(path="/home/angelbell/dev/auto-trade/scratchpad/out_squeeze_matrix_full.txt"):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"\s*(\w+)\s+(\d+min|\d+h|1d)\s+(\d+)\s+(\d+)", line)
                if m:
                    sym, tf = m.group(1), m.group(2)
                    sign_m = re.search(r"(順|逆|無)\s*$", line)
                    if sign_m:
                        out[(sym, tf)] = sign_m.group(1)
    except FileNotFoundError:
        pass
    return out


# ---------------------------------------------------------------- main matrix (primary K, q=1/3, d=0.5)
def analyze_cell(sym, tf, base):
    K = K_CENTER[tf]
    valid, box, pcts = box_flag_v2(base, K, q=Q_PRIMARY, d=D_PRIMARY)
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
    frac = 100 * n_box / n
    return dict(sym=sym, tf=tf, skip=False, K=K, n=n, n_box=n_box, frac=frac,
                med_all=s_all["median"], med_box=s_box["median"], delta=delta,
                p_lt1_box=s_box["p_lt1"], p_lt1_all=s_all["p_lt1"],
                pct_med=pct_med, outside=outside, sign=sign_of(delta, pct_med))


def fmt_row(r, tr_signs):
    tr = tr_signs.get((r["sym"], r["tf"]), "?")
    if r.get("skip", True):
        return (f"  {r['sym']:<8}{r['tf']:<7}{'K='+str(r.get('K','-')):<6}{'-':>6}{'-':>6}{'-':>6}{'-':>9}{'-':>9}"
                f"{'-':>8}{'-':>10}{'-':>10}{'-':>9}{'-':>5}{tr:>5}   n={r.get('n',0)} n箱={r.get('n_box',0)} 不足でskip")
    return (f"  {r['sym']:<8}{r['tf']:<7}K={r['K']:<4}{r['n']:>6}{r['n_box']:>6}{r['frac']:>5.0f}%"
            f"{r['med_all']:>9.2f}{r['med_box']:>9.2f}{r['delta']:>+8.2f}{r['p_lt1_box']:>9.1f}%"
            f"{r['p_lt1_all']:>8.1f}%{r['pct_med']:>8.0f}%ile{r['sign']:>5}{tr:>5}")


# ---------------------------------------------------------------- q sweep (representative cells)
def q_sweep_table(bases):
    print(f"\n{'='*100}\n狭さしきい q スイープ（代表セル、primary K・d=0.5固定、q(下位何割)のみ振る）")
    print(f"{'='*100}")
    for sym, tf in REP_CELLS:
        base = bases.get((sym, tf))
        if base is None:
            continue
        K = K_CENTER[tf]
        box_range_all = rolling_box_range(base["d"], K)
        print(f"  {sym} {tf} (K={K}):")
        for q in Q_SWEEP:
            valid, box, _ = box_flag_v2(base, K, q=q, d=D_PRIMARY, box_range_all=box_range_all)
            n = int(valid.sum()); n_box = int((valid & box).sum())
            if n_box < 10 or (n - n_box) < 10:
                print(f"    q<={q:.2f}  n不足でskip")
                continue
            med_all = np.median(base["MFE"][valid])
            med_box = np.median(base["MFE"][valid & box])
            delta = med_box - med_all
            meds, _ = random_subset_null(base["MFE"][valid], n_box, reps=1000)
            pct = 100 * np.mean(meds < med_box)
            mark = "  <- primary" if abs(q - Q_PRIMARY) < 1e-9 else ""
            print(f"    q<={q:.2f}  n={n:>5}  n箱={n_box:>5}({100*n_box/n:.0f}%)  Δ中央値={delta:>+.2f}"
                  f"  null%ile={pct:>3.0f}%  符号={sign_of(delta, pct)}{mark}")


# ---------------------------------------------------------------- d sweep (representative cells)
def d_sweep_table(bases):
    print(f"\n{'='*100}\nd スイープ（代表セル、primary K・q=1/3固定、d のみ振る）")
    print(f"{'='*100}")
    header = f"  {'銘柄':<8}{'TF':<7}{'K':<5}{'n':>6}" + "".join(f"{'d='+str(dd):>10}" for dd in D_SWEEP)
    print(header)
    for sym, tf in REP_CELLS:
        base = bases.get((sym, tf))
        if base is None:
            print(f"  {sym:<8}{tf:<7}{'-':<5}{'-':>6}  (データ不足)")
            continue
        K = K_CENTER[tf]
        box_range_all = rolling_box_range(base["d"], K)
        n_ref = None
        cells = []
        for dd in D_SWEEP:
            valid, box, _ = box_flag_v2(base, K, q=Q_PRIMARY, d=dd, box_range_all=box_range_all)
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
    print(f"\n{'='*100}\nK スイープ（代表セル、q=1/3・d=0.5固定、K のみ振る）-- primary近傍で符号が安定するか")
    print(f"{'='*100}")
    for sym, tf in REP_CELLS:
        base = bases.get((sym, tf))
        if base is None:
            continue
        print(f"  {sym} {tf}:")
        for K in K_SWEEP[tf]:
            valid, box, _ = box_flag_v2(base, K, q=Q_PRIMARY, d=D_PRIMARY)
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
            print(f"    K={K:<3}  n={n:>5}  n箱={n_box:>5}({100*n_box/n:.0f}%)  Δ中央値={delta:>+.2f}"
                  f"  null%ile={pct:>3.0f}%  符号={sign_of(delta, pct)}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tr_signs = load_tr_signs()

    print("箱検知v2(自己正規化): 狭さ=trailing250本中パーセンタイル<=1/3(primary) かつ 貼り付き dist<=0.5(primary)")
    print(f"{'銘柄':<8}{'TF':<7}{'K':<6}{'n':>6}{'n箱':>6}{'箱%':>5}{'MFE中央値base':>9}{'MFE中央値箱':>9}"
          f"{'Δ中央値':>8}{'P(<1R)箱':>9}{'P(<1R)base':>8}{'null%ile':>8}{'符号':>5}{'TR比版':>5}")
    print("-" * 128)

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
                r = analyze_cell(sym, tf, base)
            rows.append(r)
            print(fmt_row(r, tr_signs))

    valid_rows = [r for r in rows if not r.get("skip", True)]
    print(f"\n有効セル数: {len(valid_rows)} / {len(rows)}")

    print("\nsanity check: 箱あり比率(n箱/n)の分布（期待値: 狭さ1/3 x 貼り付き~30-40% ≈ 全体の1〜2割前後）")
    if valid_rows:
        fracs = [r["frac"] for r in valid_rows]
        print(f"  中央値={np.median(fracs):.0f}%  平均={np.mean(fracs):.0f}%  範囲=[{min(fracs):.0f}%, {max(fracs):.0f}%]")
        odd = [r for r in valid_rows if r["frac"] < 5 or r["frac"] > 30]
        if odd:
            print("  期待レンジ(5〜30%)から外れたセル:")
            for r in odd:
                print(f"    {r['sym']:<8}{r['tf']:<7}箱%={r['frac']:.0f}%")
        else:
            print("  全セルおおむね期待レンジ内")

    print("\n符号の内訳(箱v2・自己正規化):")
    from collections import Counter
    cnt = Counter(r["sign"] for r in valid_rows)
    for k, v in cnt.items():
        print(f"  {k}: {v}")

    print("\n符号が TR比版(粗い版) と一致したか:")
    same = diff_ = noref = 0
    for r in valid_rows:
        tr = tr_signs.get((r["sym"], r["tf"]))
        if tr is None:
            noref += 1
        elif tr == r["sign"]:
            same += 1
        else:
            diff_ += 1
    print(f"  一致={same}  不一致={diff_}  TR比版に対応セル無し={noref}")
    print("  詳細(箱v2符号 vs TR比版符号):")
    for r in valid_rows:
        tr = tr_signs.get((r["sym"], r["tf"]), "?")
        mark = "==" if tr == r["sign"] else ("!=" if tr != "?" else "? ")
        print(f"    {r['sym']:<8}{r['tf']:<7}箱v2={r['sign']}  {mark}  TR比={tr}")

    print("\n銘柄別の符号(箱v2):")
    by_sym = {}
    for r in valid_rows:
        by_sym.setdefault(r["sym"], []).append(f"{r['tf']}:{r['sign']}")
    for sym, signs in by_sym.items():
        print(f"  {sym:<8}: {' '.join(signs)}")

    print("\nTF別の符号(箱v2):")
    by_tf = {}
    for r in valid_rows:
        by_tf.setdefault(r["tf"], []).append(f"{r['sym']}:{r['sign']}")
    for tf in ["5min", "15min", "1h", "4h", "1d"]:
        if tf in by_tf:
            print(f"  {tf:<7}: {' '.join(by_tf[tf])}")

    out_rows = [r for r in valid_rows if r["outside"]]
    print("\nnull帯の外(97.5%ile超 or 2.5%ile未満)に出たセル(箱v2):")
    if out_rows:
        for r in out_rows:
            print(f"  {r['sym']:<8}{r['tf']:<7}null%ile={r['pct_med']:.0f}  Δ中央値={r['delta']:+.2f}")
            base = bases.get((r["sym"], r["tf"]))
            if base is None:
                base = build_cell_base(r["sym"], r["tf"], smoke=args.smoke)
            if base is not None:
                valid, box, _ = box_flag_v2(base, r["K"], q=Q_PRIMARY, d=D_PRIMARY)
                diffs1 = block_boot_diff(base["times"][valid], base["MFE"][valid], box[valid], k=1, nb=1000)
                diffs12 = block_boot_diff(base["times"][valid], base["MFE"][valid], box[valid], k=12, nb=1000)
                if diffs1 is not None and diffs12 is not None:
                    p1 = 100 * np.mean(diffs1[~np.isnan(diffs1)] > 0)
                    p12 = 100 * np.mean(diffs12[~np.isnan(diffs12)] > 0)
                    print(f"      (裏取り)巡回ブロック・ブートストラップ P(箱あり中央値>全体中央値): "
                          f"k=1ヶ月={p1:.0f}%  k=12ヶ月={p12:.0f}%")
    else:
        print("  無し(全セル帯の内)")

    q_sweep_table(bases)
    d_sweep_table(bases)
    k_sweep_table(bases)


if __name__ == "__main__":
    main()
