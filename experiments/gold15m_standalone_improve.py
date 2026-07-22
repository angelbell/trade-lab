"""仕様カード experiments/spec_gold15m_standalone_improve.md の実装。

gold15m 単体改善 A(入口)/B(選別)/C(RR) を1ジョブで測る。判定は必ずブックCAGR/DDではなく単体軸
(勝率・PF・N/年・meanR・IS/OOS・@1%固定リスクのCAGR・巡回ブロックブートストラップ中央値DD・
CAGR/中央値DD・資金倍率分布=1年窓の口座倍率 中央値/p25/p75)。固定0.01ロットの前提でサイズ写像は
しない(1%はあくまで比較のスケール)。

土台(照合ゲート): experiments/strength_gateslope_generalize.py の build_gold15m / gate1_check /
gate2_check と experiments/strength_btc15mL.py の rebuild_entries / match_entries_to_trades /
random_drop_null をそのまま import して使う(車輪の再発明禁止)。R/netRの定義は research/book.py
get_book_legs()['gold15m'] と一字一句一致(0.3/risk)。

A.入口の3構成のうち frac0.25(現行) と market(pf=0) は entries(plan()の出力)が共通で、
src.engine.walk.walk() をそのまま pullback_frac だけ変えて呼ぶ(reuse そのまま、追加コード無し)。
構造アンカー(H1レベル戻り指値)だけは walk() が「1トレードにつき1つの固定pf」しか受け付けない
ため、pH1(setups由来、param無し)を per-trade の指値にする必要がある。これは walk() に無い機能
なので、walk()の PULLBACK-LIMIT分岐(fill_win内の指値待ち→同一足ストップ優先→fwd本の順張り
決済)を「指値の決め方だけ」差し替えた小関数 walk_struct_lim() をこのファイル内に書いた
(walk()の68-134行目の忠実な鏡像、split-exec/tp1/against分岐は gold15m が使わないため省略)。
検出(make_swings/pattern_b)・ゲート(gate_sma/gate_kama)・コスト式・fill_win・同一足ストップ優先
はすべて engine の既存関数をそのまま呼んでいる。pH1 自体は detect.pattern_b の出力(setup直後、
plan()がstop/tgt計算のため捨てる前)から (e_i, i_origin)キーで引く。

Run:
  .venv/bin/python experiments/gold15m_standalone_improve.py --smoke 2>&1 | tee experiments/out_gold15m_standalone_improve_smoke.txt
  .venv/bin/python experiments/gold15m_standalone_improve.py 2>&1 | tee experiments/out_gold15m_standalone_improve.txt
"""
import argparse
import contextlib
import io
import os
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_gateslope_generalize as sgg   # build_gold15m / gate1_check / gate2_check
import strength_btc15mL as base               # rebuild_entries / match_entries_to_trades / random_drop_null

from breakout_wave import run
from src.engine.presets import BASE
from src.engine.detect import make_swings, pattern_b
from src.engine.gates import gate_sma, gate_kama, exit_flip
from src.engine.plan import plan
from src.engine.walk import walk
from src.engine.arbiter import cd, Boot, months_union

RNG_SEED = 20260719


# ============================================================== 共通の単体指標

def pf_of(R):
    R = np.asarray(R, dtype=float)
    neg = abs(R[R <= 0].sum())
    return R[R > 0].sum() / neg if neg > 0 else np.inf


def basic_stats(R, times):
    R = np.asarray(R, dtype=float)
    times = pd.DatetimeIndex(times)
    n = len(R)
    days = max((times[-1] - times[0]).days, 1)
    yrs = days / 365.25
    mid = times[0] + (times[-1] - times[0]) / 2
    isr = R[times <= mid]; oos = R[times > mid]
    return dict(n=n, npy=n / yrs, win=100.0 * (R > 0).mean(), pf=pf_of(R), meanR=R.mean(),
                totR=R.sum(), IS=isr.mean() if len(isr) else np.nan,
                OOS=oos.mean() if len(oos) else np.nan, days=days, yrs=yrs)


def fmt_row(name, s):
    pf_s = f"{s['pf']:.2f}" if np.isfinite(s['pf']) else "inf"
    return (f"  {name:<16}{s['n']:>6}{s['npy']:>7.1f}{s['win']:>7.1f}%{pf_s:>7}"
            f"{s['meanR']:>+8.3f}{s['totR']:>+9.1f}{s['IS']:>+7.2f}/{s['OOS']:<+6.2f}")


def block_dd_table(months, R, times, f=0.01, nb=3000, seed=RNG_SEED):
    """1/3/6/12mo 巡回ブロックbootstrap(nb回)の中央値DD、その CAGR/中央値DD。CAGRは実測1経路
    (@risk f固定)。maxDDだけをブートストラップで置き換える(単一経路禁止、CLAUDE.md#8)。"""
    s = pd.Series(np.asarray(R, dtype=float) * f, index=pd.DatetimeIndex(times))
    days = max((s.index[-1] - s.index[0]).days, 1)
    cagr, dd_single = cd(s.values, days)
    rows = []
    for k in (1, 3, 6, 12):
        bt = Boot(months, nb=nb, k=k, seed=seed)
        ddm = bt.dd_median(s)
        rows.append(dict(k=k, dd_median=ddm, cagr=cagr, ratio=cagr / max(ddm, 1e-9)))
    return cagr, dd_single, rows


def window_multiplier(months, R, times, f=0.01, k=3, nb=3000, seed=RNG_SEED):
    """1年窓の口座倍率分布: 巡回ブロック(k=3mo)で全期間を並べ替え、n/年(実測の本数密度)本を
    "1年分"として複利させた倍率。中央値/p25/p75 を返す(標準偏差/歪みも併記できるよう生配列も返す)。"""
    s = pd.Series(np.asarray(R, dtype=float) * f, index=pd.DatetimeIndex(times))
    days = max((s.index[-1] - s.index[0]).days, 1)
    yrs = days / 365.25
    n1 = max(1, int(round(len(s) / yrs)))
    bt = Boot(months, nb=nb, k=k, seed=seed)
    mk = s.index.to_period("M")
    by = {m: s.values[mk == m] for m in bt.months}
    mult = np.empty(len(bt.layout))
    for i, seq in enumerate(bt.layout):
        v = np.concatenate([by[bt.months[j]] for j in seq])[:len(s)]
        mult[i] = np.prod(1.0 + v[:n1])
    return dict(n1=n1, median=np.median(mult), p25=np.percentile(mult, 25),
                p75=np.percentile(mult, 75), std=np.std(mult), mult=mult)


def print_config_report(name, R, times, months, f=0.01):
    s = basic_stats(R, times)
    cagr, dd_single, ddrows = block_dd_table(months, R, times, f=f)
    wm = window_multiplier(months, R, times, f=f)
    pf_s = f"{s['pf']:.2f}" if np.isfinite(s['pf']) else "inf"
    print(f"  {name}")
    print(f"    n={s['n']}  n/年={s['npy']:.1f}  勝率={s['win']:.1f}%  PF={pf_s}  meanR={s['meanR']:+.3f}"
          f"  totR={s['totR']:+.1f}  IS/OOS={s['IS']:+.2f}/{s['OOS']:+.2f}")
    print(f"    @1%固定リスク: CAGR(実測1経路)={cagr:+.1f}%  maxDD(単一経路)={dd_single:.1f}%")
    print(f"    巡回ブロックbootstrap 中央値DD (nb=3000): "
          + "  ".join(f"{r['k']}mo={r['dd_median']:.1f}%(CAGR/DD={r['ratio']:.2f})" for r in ddrows))
    print(f"    資金倍率分布(1年窓={wm['n1']}本, 固定{f*100:.0f}%, k=3moブロックnb=3000): "
          f"中央値={wm['median']:.2f}倍  p25={wm['p25']:.2f}倍  p75={wm['p75']:.2f}倍  std={wm['std']:.2f}")
    return dict(stats=s, cagr=cagr, dd_single=dd_single, ddrows=ddrows, wm=wm)


# ============================================================== 構造アンカー用: setups(pH1)復元

def rebuild_entries_with_setups(d15, args):
    """base.rebuild_entries と同じ手順だが、pattern_b の setups(pH1 を含む)も返す。
    engine の既存関数(make_swings/pattern_b/plan/gate_sma/gate_kama/exit_flip/walk)を
    そのまま呼んでいるだけで、独自の検出/ゲート/執行ロジックは足していない。"""
    h, l, c = d15["high"].values, d15["low"].values, d15["close"].values
    a = ta.atr(d15["high"], d15["low"], d15["close"], length=args.atr).values
    es = (d15["close"].ewm(span=args.trend_ema, adjust=False).mean().values
          if args.trend_ema > 0 else None)
    reg, ext_arr = gate_sma(d15, args)
    kreg = gate_kama(d15, args)
    against = exit_flip(d15, args)
    sw = make_swings(h, l, c, a, args)
    setups = pattern_b(c, l, a, es, sw, args)
    entries = plan(c, l, a, sw, setups, reg, ext_arr, kreg, args)
    t2, _ = walk(d15, entries, against, args)
    return setups, entries, t2, a, against


def ph1_lookup(setups):
    d = {}
    for (e_i, pH1, pL0, pL2, iL0) in setups:
        key = (e_i, iL0)
        if key in d and d[key] != pH1:
            raise RuntimeError(f"pH1 lookup collision at key={key}: {d[key]} vs {pH1}")
        d[key] = pH1
    return d


def walk_struct_lim(d15, entries_ext, args):
    """walk()の PULLBACK-LIMIT分岐(src/engine/walk.py 68-134行)の忠実な鏡像。差分は1点だけ:
    lim を e-pf*(e-stop) ではなく、per-trade の構造アンカー pH1 にする。gold15mは
    exec_split/tp1_frac/against(exit_flip)を使わないため、それらの分岐は省略している
    (このレッグの args では全部 off/None で発火しない — build_gold15m の args 参照)。
    同一足ストップ優先・fill_win内の指値待ち・fwd本の順張り決済・de-dup済みentries前提の
    busy_until/max_pos ループは walk() と一字一句同じ。"""
    h, l, c = d15["high"].values, d15["low"].values, d15["close"].values
    trades = []
    open_x = []
    maxpos = max(1, int(getattr(args, "max_pos", 1)))
    fw = getattr(args, "fill_win", 0) or args.fwd
    n_invalid_anchor = 0
    n_miss_nofill = 0
    depths = []
    for (i, e, stop, tgt, i_origin, ph1) in entries_ext:
        open_x = [x for x in open_x if x >= i]
        if len(open_x) >= maxpos:
            continue
        lim = ph1
        if not (stop < lim < e):
            n_invalid_anchor += 1
            continue
        depths.append((e - lim) / (e - stop))
        fj = None
        for j in range(i + 1, min(i + 1 + fw, len(c))):
            if h[j] >= tgt:
                break                      # ran to target first = limit missed
            if l[j] <= lim:
                fj = j; break
        if fj is None:
            n_miss_nofill += 1
            continue
        fill_bar_stopped = l[fj] <= stop
        e_px, e_bar = lim, fj
        risk = e_px - stop
        reward = tgt - e_px
        if risk <= 0:
            continue
        exit_j = min(e_bar + args.fwd, len(c) - 1)
        if fill_bar_stopped:
            R, exit_j = -1.0, e_bar
        else:
            R = None
            for j in range(e_bar + 1, min(e_bar + 1 + args.fwd, len(c))):
                if l[j] <= stop:
                    R = -1.0; exit_j = j; break
                if h[j] >= tgt:
                    R = reward / risk; exit_j = j; break
            if R is None:
                R = (c[exit_j] - e_px) / risk
        hold = (d15.index[exit_j] - d15.index[e_bar]).total_seconds() / 86400.0
        trades.append((d15.index[e_bar], R, hold, risk, e_px, R, 1.0, i - i_origin))
        open_x.append(exit_j)
    if not trades:
        return None, n_invalid_anchor, n_miss_nofill, depths
    t = pd.DataFrame(trades, columns=["time", "R", "hold", "risk", "e_px", "r_mkt", "filled", "base_bars"])
    t["y"] = t["time"].dt.year
    return t, n_invalid_anchor, n_miss_nofill, depths


# ============================================================== PART B: 選別ヘルパ

def group_gap_bootstrap(times, is_top, R, k_months, n_boot=3000, seed=RNG_SEED):
    """top群 vs rest群 の meanR ギャップの巡回ブロックbootstrap(strength_gold15m.py の
    top/rest bootstrap と同じ手法: 月ブロックを並べ替え、各トレードの元のtop/rest所属は
    固定したまま、再連結したサンプル内で群平均の差を測る)。"""
    s = pd.DataFrame({"top": is_top, "R": np.asarray(R, dtype=float)},
                      index=pd.DatetimeIndex(times))
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(nm / k_months))
    gaps = []
    for _ in range(n_boot):
        starts = rng.integers(0, nm, size=nblk)
        seq = np.concatenate([[(st + j) % nm for j in range(k_months)] for st in starts])
        samp = pd.concat([by_month[months[j]] for j in seq])
        tg = samp.loc[samp["top"], "R"]; rg = samp.loc[~samp["top"], "R"]
        if len(tg) < 5 or len(rg) < 5:
            continue
        gaps.append(tg.mean() - rg.mean())
    gaps = np.array(gaps)
    return float(np.median(gaps)), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5)), len(gaps)


def topX_mask(x, Xpct):
    """rank降順(値が大きい方が上位)で上位 Xpct% を True にする。同値はrank method='first'で
    タイブレークし、n*Xpct% を四捨五入した本数を厳密に取る。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_keep = int(round(n * Xpct / 100.0))
    order = np.argsort(-x, kind="stable")   # 降順
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_keep]] = True
    return mask


# ============================================================== main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    print(f"{'='*90}\ngold15m 単体改善 A/B/C 同時測定 (smoke={cli.smoke})\n{'='*90}")

    # ---------------------------------------------------------------- 土台 + 照合ゲート
    g15, args, t, netR = sgg.build_gold15m(cli.smoke)
    print(f"\ngold15m 再構築: n={len(t)}  span={t['time'].iloc[0]} -> {t['time'].iloc[-1]}  (smoke={cli.smoke})")

    mine = pd.Series(netR, index=pd.DatetimeIndex(t["time"]))
    gate1 = sgg.gate1_check("gold15m", mine, cli.smoke)
    if gate1 is False:
        print("!!! 照合ゲート1 FAIL -- 停止する。"); return

    entries, t2 = base.rebuild_entries(g15, args)
    gate2 = sgg.gate2_check("gold15m", t2, t)
    if not gate2:
        print("!!! 照合ゲート2 FAIL -- 停止する。"); return

    i_arr = base.match_entries_to_trades(entries, t, args.pullback_frac)
    print(f"[照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(t)} 本すべて一意対応 => PASS")

    R0 = t["R"].values
    times0 = pd.DatetimeIndex(t["time"])
    netR0 = R0 - 0.3 / t["risk"].values
    s0 = basic_stats(netR0, times0)
    pf0_s = f"{s0['pf']:.2f}" if np.isfinite(s0['pf']) else "inf"
    print(f"\n[baseline(現行) 自己点検] n={s0['n']} ({s0['npy']:.1f}本/年)  勝率={s0['win']:.1f}%  "
          f"PF={pf0_s}  meanR={s0['meanR']:+.4f}  totR={s0['totR']:+.1f}"
          f"   (spec記載の目安: n≈325/46本年/win24.3%/PF1.64/meanR+0.517)")

    MONTHS0 = months_union(pd.Series(netR0, index=times0))

    # ================================================================ A. 入口: frac0.25 / market / 構造アンカー
    print(f"\n\n{'#'*90}\n# A. 入口: frac0.25(現行) vs market(frac0) vs 構造アンカー(H1レベル戻り指値・param無)\n{'#'*90}")

    against = exit_flip(g15, args)
    if against is not None:
        print("  !! 注意: exit_flip が非Noneを返した(gold15mの想定=None)。仕様と食い違うため要確認。")

    args_market = SimpleNamespace(**{**vars(args), "pullback_frac": 0.0})
    t_market, _ = walk(g15, entries, against, args_market)
    R_market = t_market["R"].values
    times_market = pd.DatetimeIndex(t_market["time"])
    netR_market = R_market - 0.3 / t_market["risk"].values

    setups, entries_chk, t2_chk, a_arr, against_chk = rebuild_entries_with_setups(g15, args)
    same_entries = (len(entries_chk) == len(entries)
                    and all(abs(e1[1] - e2[1]) < 1e-9 and abs(e1[2] - e2[2]) < 1e-9
                            and abs(e1[3] - e2[3]) < 1e-9 and e1[0] == e2[0] and e1[4] == e2[4]
                            for e1, e2 in zip(entries_chk, entries)))
    print(f"  [構造アンカー用 自己整合ゲート] rebuild_entries_with_setups の entries が "
          f"base.rebuild_entries の entries と一致 => {'PASS' if same_entries else 'FAIL'}")
    if not same_entries:
        print("  !!! 一致しない -- 構造アンカーのpH1対応付けは信用できない。A構成の構造アンカー行はスキップする。")
        struct_ok = False
    else:
        struct_ok = True

    if struct_ok:
        ph1_map = ph1_lookup(setups)
        entries_ext = []
        n_missing_key = 0
        for (e_i, e, stop, tgt, i_origin) in entries:
            key = (e_i, i_origin)
            if key not in ph1_map:
                n_missing_key += 1
                continue
            entries_ext.append((e_i, e, stop, tgt, i_origin, ph1_map[key]))
        print(f"  [pH1対応付け] entries {len(entries)}本中 {len(entries_ext)}本に pH1 が付いた "
              f"(欠落={n_missing_key}) => {'PASS' if n_missing_key == 0 else 'WARN'}")

        t_struct, n_inv, n_missfill, depths = walk_struct_lim(g15, entries_ext, args)
        if t_struct is None:
            print("  構造アンカー構成: トレードが1本も生成されなかった。")
            struct_ok = False
        else:
            R_struct = t_struct["R"].values
            times_struct = pd.DatetimeIndex(t_struct["time"])
            netR_struct = R_struct - 0.3 / t_struct["risk"].values
            depths = np.array(depths)
            print(f"\n  構造アンカー(H1戻り指値)の押し目深さ(frac of risk換算, (e-H1)/(e-stop)):"
                  f" n_valid={len(depths)}  median={np.median(depths):.2f}"
                  f"  [25/75={np.percentile(depths,25):.2f}/{np.percentile(depths,75):.2f}]"
                  f"   (参考: frac0.25 は定数0.25)")
            print(f"  無効アンカー(pH1がstopとeの間に無い)={n_inv}本  fill_win内に約定しなかった={n_missfill}本"
                  f"  (母数={len(entries_ext)})")

    MONTHS_A = months_union(pd.Series(netR0, index=times0), pd.Series(netR_market, index=times_market),
                             *([pd.Series(netR_struct, index=times_struct)] if struct_ok else []))

    print(f"\n  {'構成':<16}{'n':>6}{'本/年':>7}{'勝率':>8}{'PF':>7}{'meanR':>8}{'totR':>9}{'IS/OOS':>14}")
    print(fmt_row("frac0.25(現行)", s0))
    s_mkt = basic_stats(netR_market, times_market)
    print(fmt_row("market(frac0)", s_mkt))
    if struct_ok:
        s_struct = basic_stats(netR_struct, times_struct)
        print(fmt_row("構造アンカー", s_struct))

    print("\n  -- 各構成の詳細(@1%固定リスクのCAGR・巡回ブロックbootstrap中央値DD・資金倍率分布) --")
    resA = {}
    resA["frac0.25(現行)"] = print_config_report("frac0.25(現行)", netR0, times0, MONTHS_A)
    resA["market(frac0)"] = print_config_report("market(frac0)", netR_market, times_market, MONTHS_A)
    if struct_ok:
        resA["構造アンカー"] = print_config_report("構造アンカー(H1戻り指値)", netR_struct, times_struct, MONTHS_A)

    print("\n  判定A: 構造アンカーが frac0.25 を単体軸(特に資金倍率分布・IS/OOS均衡)で上回るか")
    if struct_ok:
        b, m = resA["frac0.25(現行)"], resA["構造アンカー"]
        print(f"    frac0.25: meanR={b['stats']['meanR']:+.3f} 倍率中央値={b['wm']['median']:.2f}x"
              f" IS/OOS={b['stats']['IS']:+.2f}/{b['stats']['OOS']:+.2f}")
        print(f"    構造:     meanR={m['stats']['meanR']:+.3f} 倍率中央値={m['wm']['median']:.2f}x"
              f" IS/OOS={m['stats']['IS']:+.2f}/{m['stats']['OOS']:+.2f}")
    else:
        print("    構造アンカー構成は自己整合ゲート不通過のため比較不能。")

    # ================================================================ B. 選別: stop_atr 上位X%
    print(f"\n\n{'#'*90}\n# B. 選別: stop_atr(=risk/ATR14[確定足i]) 上位X%のみ採用 (X=100/80/60/40)\n{'#'*90}")

    atr_g15 = ta.atr(g15["high"], g15["low"], g15["close"], length=args.atr).values
    stop_atr = t["risk"].values / atr_g15[i_arr]
    n_nan = int(np.isnan(stop_atr).sum())
    print(f"  stop_atr 有効n={len(stop_atr) - n_nan}/{len(stop_atr)} (NaN={n_nan}, ATR14ウォームアップ等)")
    valid = ~np.isnan(stop_atr)
    sa = stop_atr[valid]; R0v = netR0[valid]; tv = times0[valid]

    print(f"\n  {'X%':>5}{'n':>6}{'本/年':>7}{'勝率':>8}{'PF':>7}{'meanR':>8}{'totR':>9}{'IS/OOS':>14}")
    resB = {}
    for X in (100, 80, 60, 40):
        mask = topX_mask(sa, X) if X < 100 else np.ones(len(sa), dtype=bool)
        Rx, tx = R0v[mask], tv[mask]
        sx = basic_stats(Rx, tx)
        print(fmt_row(f"X={X}%", sx) + ("  ← baseline(全採用)" if X == 100 else ""))
        resB[X] = dict(mask=mask, R=Rx, times=tx, stats=sx)

    print("\n  -- 各Xの詳細(@1%固定リスクのCAGR・巡回ブロックbootstrap中央値DD・資金倍率分布) --")
    for X in (100, 80, 60, 40):
        resB[X]["report"] = print_config_report(f"X={X}%", resB[X]["R"], resB[X]["times"], MONTHS0)

    print("\n  -- ランダム除去null(法則7): 上位X%と同数をランダム抽出したmeanR分布に対する実測パーセンタイル --")
    for X in (80, 60, 40):
        n_top = resB[X]["stats"]["n"]
        pct, null_mean, null_std = base.random_drop_null(R0v, resB[X]["stats"]["meanR"], n_top, n_reps=5000)
        print(f"    X={X}%: n={n_top}  実測meanR={resB[X]['stats']['meanR']:+.3f}  "
              f"null(平均{null_mean:+.3f}±{null_std:.3f}) に対し {pct:.1f}パーセンタイル")

    print("\n  -- 巡回ブロックbootstrap: 採用群 vs 除外群 の meanRギャップ (法則7: 別の月の並びでも成り立つか) --")
    for X in (80, 60, 40):
        mask = resB[X]["mask"]
        gap0 = R0v[mask].mean() - R0v[~mask].mean()
        print(f"    X={X}%: 実測ギャップ(採用-除外)meanR = {gap0:+.3f}")
        for k in (1, 3, 6, 12):
            med, lo, hi, nvalid = group_gap_bootstrap(tv, mask, R0v, k, n_boot=3000)
            straddles = "0またぎ" if lo < 0 < hi else ("常に正" if lo > 0 else "常に負")
            print(f"      {k:>2}mo: 中央値={med:+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  ({straddles})"
                  f"  有効draw={nvalid}/3000")

    print("\n  -- 資金倍率レンズでの裁定(N減 vs PF増): X=100基準比 --")
    b100 = resB[100]["report"]
    for X in (80, 60, 40):
        r = resB[X]["report"]
        pf_x = resB[X]["stats"]["pf"]; pf_100 = resB[100]["stats"]["pf"]
        pf_diff_s = "inf" if not np.isfinite(pf_x) or not np.isfinite(pf_100) else f"{pf_x-pf_100:+.2f}"
        print(f"    X={X}%: 倍率中央値 {r['wm']['median']:.2f}x (基準比 {r['wm']['median']-b100['wm']['median']:+.2f})"
              f"  n/年 {resB[X]['stats']['npy']:.1f} (基準比 {resB[X]['stats']['npy']-resB[100]['stats']['npy']:+.1f})"
              f"  PF {pf_x:.2f} (基準比 {pf_diff_s})")

    # ================================================================ C. RR: {3,4,5,6}
    print(f"\n\n{'#'*90}\n# C. RR掃引 (入口frac0.25固定): RR in {{3,4,5,6}}\n{'#'*90}")
    print(f"\n  {'RR':>5}{'n':>6}{'本/年':>7}{'勝率':>8}{'PF':>7}{'meanR':>8}{'totR':>9}{'IS/OOS':>14}")
    resC = {}
    for RR in (3, 4, 5, 6):
        args_rr = SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                                      "pullback_frac": 0.25, "fill_win": 200, "rr": float(RR)})
        t_rr = run(g15, args_rr)
        R_rr = t_rr["R"].values
        times_rr = pd.DatetimeIndex(t_rr["time"])
        netR_rr = R_rr - 0.3 / t_rr["risk"].values
        s_rr = basic_stats(netR_rr, times_rr)
        tag = "  ← 現行(baselineと一致するはず)" if RR == 4 else ""
        print(fmt_row(f"RR={RR}", s_rr) + tag)
        resC[RR] = dict(R=netR_rr, times=times_rr, stats=s_rr)
        if RR == 4:
            same_n = len(R_rr) == len(R0)
            same_val = same_n and np.allclose(netR_rr, netR0, atol=1e-9)
            print(f"    [RR=4 自己点検] baseline(t)と n一致={same_n}  値一致={same_val}"
                  f"  => {'PASS' if same_n and same_val else 'FAIL(要確認)'}")

    print("\n  -- 各RRの詳細(@1%固定リスクのCAGR・巡回ブロックbootstrap中央値DD・資金倍率分布) --")
    for RR in (3, 4, 5, 6):
        resC[RR]["report"] = print_config_report(f"RR={RR}", resC[RR]["R"], resC[RR]["times"], MONTHS0)

    means = [resC[RR]["stats"]["meanR"] for RR in (3, 4, 5, 6)]
    mults = [resC[RR]["report"]["wm"]["median"] for RR in (3, 4, 5, 6)]
    print(f"\n  meanR系列(RR3->6): {[round(x,3) for x in means]}")
    print(f"  資金倍率中央値系列(RR3->6): {[round(x,2) for x in mults]}")
    peak_idx = int(np.argmax(mults))
    print(f"  資金倍率のピークは RR={[3,4,5,6][peak_idx]} (丘型なら中間、壁/棘なら端に単発ピーク)")

    # ================================================================ 総括
    print(f"\n\n{'#'*90}\n# 総括: 単体で現行(frac0.25/RR4/全採用)を上回る構成の有無\n{'#'*90}")
    base_wm = resA["frac0.25(現行)"]["wm"]["median"]
    base_ratio3mo = [r for r in resA["frac0.25(現行)"]["ddrows"] if r["k"] == 3][0]["ratio"]
    print(f"  baseline: 倍率中央値={base_wm:.2f}x  CAGR/中央値DD(3mo)={base_ratio3mo:.2f}")

    beats = []
    if struct_ok:
        m_wm = resA["構造アンカー"]["wm"]["median"]
        if m_wm > base_wm:
            beats.append(f"A.構造アンカー(倍率中央値 {m_wm:.2f}x > baseline {base_wm:.2f}x)")
    for X in (80, 60, 40):
        wm = resB[X]["report"]["wm"]["median"]
        if wm > base_wm:
            beats.append(f"B.X={X}%(倍率中央値 {wm:.2f}x > baseline {base_wm:.2f}x)")
    for RR in (3, 5, 6):
        wm = resC[RR]["report"]["wm"]["median"]
        if wm > base_wm:
            beats.append(f"C.RR={RR}(倍率中央値 {wm:.2f}x > baseline {base_wm:.2f}x)")

    if beats:
        print("  現行を資金倍率中央値で上回った構成:")
        for b in beats:
            print(f"    - {b}")
    else:
        print("  現行(frac0.25/RR4/全採用)を資金倍率中央値で上回る構成は見つからなかった"
              " => 現行が単体でも最良、改善余地は小さい。")

    print(f"\n実行コマンド: .venv/bin/python experiments/gold15m_standalone_improve.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
