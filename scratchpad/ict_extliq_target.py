"""ICT 忠実化・優先3: 出口を固定RR4 から外部流動性ターゲット（反対側の流動性プール-fluff）へ張り替える。

忠実性監査 A-2（`quant-audit.md` 項目3）: ICT正典の target は固定RRではなく
"draw on liquidity"＝反対側の流動性レベル（PDH/直近スイング高値/アジア高値など）。
現行の生存セル（EURUSD-long-FVG-CE(mid), s01 優先1）は出口だけ固定RR4のまま＝残る忠実性ギャップ。

入口・stop は固定（因果分離、優先1の張替と対称）。動かすのは出口(target)だけ:
  entry: EURUSD ロング = FVG-CE(mid, fvg_min_atr=0.15)（優先1の生存セル、lim_fn=fvg_anchor_fn("mid","long")）
         対照(usdjpy/audusd/gold/btcusd)ロング = base f=0.25固定リトレース, fvg_min_atr=0.25
         （= v4 の "0.25 入口" と同一設定。ict_fvg_anchor.py の「0. 検算アンカー」がこの表記の原典 —
         「対照は各々v4の0.25入口＝そのセルの最良」を、既存コードが確立した唯一の具体的な"0.25入口"の
         定義として採用。仕様の他の読み方は無いと判断したが、解釈である旨をここに明記する。）
  stop : L - 0.1ATR（据え置き）

出口の変種（stop据え置きなので risk=entry-stop は不変、targetだけ替える）:
  固定RR: RR2 / RR3 / RR4（現行ベースライン）
  外部流動性: objective ∈ {PDH（前日高値）, swingH20/swingH40（直近20/40本の3本フラクタル・スイング
    高値の最大値、"直近N本のスイング高値"の字義通りの実装 — N=20〜40の両端を掃引）, asiaH（アジア窓高値）}
    、target = objective - fluff, fluff ∈ {0, 3pip, 5pip}。
  見送り規則: objective が entry 以下 → skip("objective_at_or_below_entry")。
             (target-entry)/risk < 0.5 → skip("too_close")。objective 欠損 → skip("no_objective")。
             クランプはしない（ICTは届かない目標を取らない、の字義通り）。skip率を分母=試行済み
             （lim約定した=fp発見済みトレード）に対する割合として報告する。

先読み排除: PDH/asiaHはbuild()内でMSS確定足(jm)より前の窓（前日終値/アジア窓終了時）で確定済みの
既存ローカル変数を流用。swingH(N)は新規ヘルパ recent_swing_high(hi,end,N) を追加し、
end=jm（MSS確定足）までの範囲でのみ3本フラクタルを探索（k+1<end=確定済み）。

再利用（車輪の再発明禁止）:
  - ict_population.py: build()/canonical_setups() に use_liq/liq_ns を追加（pdh/asiaH/swingH20/40 を
    rec[side] に付与するだけ、既存の sweep/MSS/FVG 判定ロジックは一切変更なし。デフォルト False で
    既存呼び出しはビット一致）。新規ヘルパ recent_swing_high/recent_swing_low は last_fractal_high と
    同じフラクタル条件を流用（"最後の1つ"ではなく"窓内の最大"を取るだけの違い）。
  - ict_exec.py: walk() に tgt_fn/skip_log/rr_log を追加（tgt_fn=None時は既存のrr固定ロジックと
    ビット一致、returnの4-tupleも不変=既存呼び出し元は無改変で動く）。同足損切り優先はそのまま適用。
  - ict_audit.py: placebo_premium() に tgt_fn/use_liq/liq_ns のパススルーを追加。
  - ict_fvg_anchor.py: fvg_anchor_fn（EURUSD mid入口）, era_split/fmt_era をそのままimport。
  - ict_dxy_smt.py: cost_tiers（realistic/conservative の2コスト段、spread+2pipのconservative定義）
    をそのままimport（車輪の再発明禁止）。

判定は勝率重視（ユーザー明示）。DSRゲートは課さない。ロングのみ（ショートは優先2で死、確定済み）。

検算アンカー: EURUSD-long-FVG-CE(mid) の RR4 が優先1の値 n=328・PF1.39・net+0.296・totR/DD 3.17前後
を再現するか（ma=0.15, lim_fn=mid, rr=4.0, cost=realistic）。

Run: .venv/bin/python scratchpad/ict_extliq_target.py [--smoke] 2>&1 | tee scratchpad/out_ict_extliq_target.txt
"""
import sys, io, argparse, contextlib
from collections import Counter
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, PIP, BUF, F_CANON, RR_CANON, walk, stats, sc
from ict_population import canonical_setups, load_prepped
from ict_audit import placebo_premium, block_boot
from ict_fvg_anchor import fvg_anchor_fn, era_split, fmt_era
from ict_dxy_smt import cost_tiers

RNG = np.random.default_rng(20260716)

EURUSD_LIM_FN = fvg_anchor_fn("mid", "long")
EURUSD_MA = 0.15
CONTROL_SYMS = ["usdjpy", "audusd", "gold", "btcusd"]
CONTROL_MA = 0.25   # = v4 の "0.25 入口"（fvg_min_atr=0.25, base f=0.25固定リトレース, lim_fn=None）

OBJ_LONG = {"PDH": "pdh", "asiaH": "asiaH", "swingH20": "swingH20", "swingH40": "swingH40"}
FLUFF_PIPS = [0, 3, 5]
FIXED_RRS = [2.0, 3.0, 4.0]


def make_ext_tgt_fn(objkey, fluff_pips, name, side="long"):
    fluff = fluff_pips * PIP[name]
    def fn(s, entry, risk):
        obj = s.get(objkey)
        if obj is None or not np.isfinite(obj):
            return None, "no_objective"
        if side == "long":
            tgt = obj - fluff
            if tgt <= entry:
                return None, "objective_at_or_below_entry"
            if (tgt - entry) / risk < 0.5:
                return None, "too_close"
        else:
            tgt = obj + fluff
            if tgt >= entry:
                return None, "objective_at_or_below_entry"
            if (entry - tgt) / risk < 0.5:
                return None, "too_close"
        return tgt, None
    return fn


def run_cell(df, tarr, dates, name, side, span, ma, lim_fn, cost_tier,
             rr=None, tgt_fn=None):
    use_liq = tgt_fn is not None
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=ma,
                         use_liq=use_liq, liq_ns=(20, 40))
    sp, cost = cost_tiers(name)[cost_tier]
    skip_log, rr_log = [], []
    tr = walk(df, S, F_CANON, rr if rr is not None else RR_CANON, BUF, sp, cost, side,
              lim_fn=lim_fn, tgt_fn=tgt_fn, skip_log=skip_log, rr_log=rr_log)
    st = stats(tr, span)
    n_pop = sum(1 for rec in S if rec[side] is not None)
    n_attempt = len(tr) + len(skip_log)
    skip_pct = 100.0 * len(skip_log) / n_attempt if n_attempt else float("nan")
    reasons = Counter(r for _, r in skip_log)
    rr_arr = np.array([x[1] for x in rr_log]) if rr_log else np.array([])
    rr_dist = None
    if len(rr_arr):
        rr_dist = dict(med=float(np.median(rr_arr)), q25=float(np.percentile(rr_arr, 25)),
                       q75=float(np.percentile(rr_arr, 75)), mean=float(np.mean(rr_arr)),
                       sd=float(np.std(rr_arr, ddof=1)) if len(rr_arr) > 1 else 0.0)
    return dict(tr=tr, st=st, n_pop=n_pop, n_attempt=n_attempt, skip_pct=skip_pct,
                reasons=reasons, rr_dist=rr_dist, S=S)


def fmt_row(label, res):
    st = res["st"]
    if st is None:
        return f"  {label:22s} n<10 skip (n_fill={len(res['tr'])}, n_attempt={res['n_attempt']}, skip%={res['skip_pct']:.1f})"
    rr = res["rr_dist"]
    rr_s = f"RRmed={rr['med']:.2f}(q25={rr['q25']:.2f},q75={rr['q75']:.2f})" if rr else "RR=fixed"
    return (f"  {label:22s} win%={st['win']:5.1f} PF={st['pf']:5.2f} meanR={st['net']:+.3f} "
            f"n={st['n']:5d} n/yr={st['npy']:5.1f} totR/DD={st['rdd']:6.2f} maxDD={st['dd']:6.1f} "
            f"IS={st['IS']:+7.0f} OOS={st['OOS']:+7.0f} skip%={res['skip_pct']:5.1f} " + rr_s)


def ablation_eurusd(df, tarr, dates, span, smoke=False):
    print("\n" + "#" * 110)
    print("1. EURUSD ロング Ablation: RR2/RR3/RR4(固定) vs 外部流動性(PDH/swingH20/swingH40/asiaH x fluff0/3/5)")
    print("   入口=FVG-CE(mid, fvg_min_atr=0.15) stop=L-0.1ATR 固定。win%を先頭列。")
    print("#" * 110)
    rows = {}
    for tier in ("realistic", "conservative"):
        print(f"\n--- cost={tier} ---")
        for rr in FIXED_RRS:
            res = run_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, tier, rr=rr)
            print(fmt_row(f"RR{rr:.1f}", res))
            rows[("eurusd", tier, f"RR{rr:.1f}")] = res
        for objname, objkey in OBJ_LONG.items():
            for fl in FLUFF_PIPS:
                tgt_fn = make_ext_tgt_fn(objkey, fl, "eurusd", "long")
                res = run_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, tier, tgt_fn=tgt_fn)
                label = f"ext_{objname}_fluff{fl}"
                print(fmt_row(label, res))
                rows[("eurusd", tier, label)] = res
    return rows


def pick_best_extliq(rows, tier="realistic", minn=30):
    """勝率重視で最良の extLiq セルを選ぶ（同点はPF、次にmeanRでtie-break）。"""
    best = None
    for (name, t, label), res in rows.items():
        if name != "eurusd" or t != tier or not label.startswith("ext_"):
            continue
        st = res["st"]
        if st is None or st["n"] < minn:
            continue
        key = (st["win"], st["pf"], st["net"])
        if best is None or key > best[0]:
            best = (key, label, res)
    return best


def judge_cell(df, tarr, dates, name, side, span, ma, lim_fn, tier, label, rr=None, tgt_fn=None):
    print(f"\n  --- 審判: {name} {side} {label} (cost={tier}) ---")
    res = run_cell(df, tarr, dates, name, side, span, ma, lim_fn, tier, rr=rr, tgt_fn=tgt_fn)
    print("  " + fmt_row(label, res))
    tr = res["tr"]
    if res["st"] is None:
        print("  n<10、審判スキップ")
        return res
    sp0, cost0 = cost_tiers(name)[tier]
    pp = placebo_premium(df, tarr, dates, name, side, span, f=F_CANON, rr=(rr if rr is not None else RR_CANON),
                         use_fvg=True, fvg_min_atr=ma, lim_fn=lim_fn,
                         use_liq=(tgt_fn is not None), liq_ns=(20, 40), tgt_fn=tgt_fn)
    print("  プラセボ窓: " + "  ".join(
        f"+{sh}h(n={pp[sh]['n'] if pp[sh] else 0},net={pp[sh]['net'] if pp[sh] else float('nan'):+.3f},"
        f"PF={pp[sh]['pf'] if pp[sh] else float('nan'):.2f})" for sh in (0, 4, 8, 12)))
    prem = {sh: (res["st"]["net"] - pp[sh]["net"]) if pp[sh] else float("nan") for sh in (4, 8, 12)}
    print("  窓プレミアム(0h-Xh): " + "  ".join(f"+{sh}h={prem[sh]:+.3f}" for sh in (4, 8, 12)))
    bb = {m: block_boot(tr, m) for m in (1, 3, 6, 12)}
    print("  ブロックブートストラップ P(totR>0): " + "  ".join(f"{m}mo={bb[m]:.0f}%" for m in (1, 3, 6, 12)))
    print("  時代別: " + fmt_era(tr))
    if res["reasons"]:
        print("  skip内訳: " + ", ".join(f"{k}={v}" for k, v in res["reasons"].most_common()))
    return res


def control_symbols(best_label, best_objkey, best_fl, smoke=False):
    print("\n" + "#" * 110)
    print(f"2. 対照銘柄 (usdjpy/audusd/gold/btcusd) ロング: RR4(固定) vs 最良extLiq({best_label})")
    print("   入口=base f=0.25固定リトレース, fvg_min_atr=0.25（=v4の0.25入口）固定。")
    print("#" * 110)
    rows = {}
    for name in CONTROL_SYMS:
        with contextlib.redirect_stderr(io.StringIO()):
            df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.25):]
        print(f"\n--- {name} (span={span}年) ---")
        for tier in ("realistic", "conservative"):
            print(f" cost={tier}")
            res_rr4 = run_cell(df, tarr, dates, name, "long", span, CONTROL_MA, None, tier, rr=4.0)
            print(" " + fmt_row("RR4", res_rr4))
            tgt_fn = make_ext_tgt_fn(best_objkey, best_fl, name, "long")
            res_ext = run_cell(df, tarr, dates, name, "long", span, CONTROL_MA, None, tier, tgt_fn=tgt_fn)
            print(" " + fmt_row(best_label, res_ext))
            rows[(name, tier, "RR4")] = res_rr4
            rows[(name, tier, best_label)] = res_ext
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("0. 検算アンカー: EURUSD-long-FVG-CE(mid, ma=0.15) RR4 が優先1の値を再現するか")
    print("   台帳(優先1): n=328・PF1.39・net+0.296・totR/DD 3.17前後")
    print("#" * 110)
    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    if args.smoke:
        dates = dates[-int(len(dates) * 0.25):]
    res0 = run_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, "realistic", rr=4.0)
    print(fmt_row("RR4(anchor)", res0))

    rows = ablation_eurusd(df, tarr, dates, span, smoke=args.smoke)

    best = pick_best_extliq(rows, tier="realistic", minn=(10 if args.smoke else 30))
    print("\n" + "=" * 110)
    if best is None:
        print("生存候補 extLiq セルなし（n>=閾値 の中で meanR/PF 基準を満たすものが無い）")
        best_label, best_objkey, best_fl = None, None, None
    else:
        key, best_label, best_res = best
        objname = best_label.replace("ext_", "").rsplit("_fluff", 1)[0]
        best_fl = int(best_label.rsplit("fluff", 1)[1])
        best_objkey = OBJ_LONG[objname]
        print(f"勝率重視の最良extLiq(realistic, n>=閾値): {best_label}  win%={key[0]:.1f} PF={key[1]:.2f} meanR={key[2]:+.3f}")

    print("\n" + "=" * 110)
    print("3. 生存候補セルの審判（EURUSD: RR4 vs 最良extLiq、realistic + conservative）")
    print("=" * 110)
    for tier in ("realistic", "conservative"):
        judge_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, tier, "RR4", rr=4.0)
    if best_label is not None:
        tgt_fn = make_ext_tgt_fn(best_objkey, best_fl, "eurusd", "long")
        for tier in ("realistic", "conservative"):
            judge_cell(df, tarr, dates, "eurusd", "long", span, EURUSD_MA, EURUSD_LIM_FN, tier, best_label, tgt_fn=tgt_fn)

    if best_label is not None:
        control_symbols(best_label, best_objkey, best_fl, smoke=args.smoke)
    else:
        print("\n2. 対照銘柄: 最良extLiqが無いためスキップ")


if __name__ == "__main__":
    main()
