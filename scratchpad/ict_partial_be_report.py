"""ICT 忠実版（EURUSD 15m long, 凍結済み）に ep41「部分利確＋建値ストップ」を足した二段階出口の測定。

凍結仕様（変更しない）:
  母集団   = 狩り(London安値<Asia安値 or PDL) + MSS(直前フラクタル高値上抜け) + ブレイク脚内
             bullish FVG (帯幅/ATR>=0.15, AND)。canonical_setups(use_fvg=True, fvg_min_atr=0.15)。
  入口     = FVG-CE(50%, mid) 買い指値・KZ内約定・来る前に走り抜けたら見送り。
             lim_fn = ict_fvg_anchor.fvg_anchor_fn("mid","long")。
  損切     = 狩られた安値 - 0.10*ATR14。
  利確     = PDH - 5pip（ict_extliq_target.make_ext_tgt_fn("pdh", 5, "eurusd", "long")）。
  最小レンジ H-L >= 0.25*ATR（build() 内で既定・不変）。
  コスト   = EURUSD RT 0.9pip = spread0.3pip + commission0.6pip（MODEL["eurusd"]="realistic"tier）。
  先読み禁止 = walk() の指値約定+前進走査+同足SL優先を流用（ict_exec.walk）。

このスクリプトが追加するのは出口の二段階化のみ（ict_exec.walk の partial_r/partial_frac、
2026-07-16 追加、partial_r=None 時は既存呼び出しとビット一致 — 自己検査で確認済み）:
  V0 = 現行           : 100%玉、PDH-5pip 単一利確（対照、partial_r=None）
  V1 = ep41正典 +1R   : +1R到達で50%利確+残り建値、残り50%はPDH-5pip目標（partial_r=1.0）
  V2 = +2R版          : +2R到達で50%利確+残り建値、残り50%はPDH-5pip目標（partial_r=2.0）

会計（ict_exec.walk のdocstringに明記済み、ここでも要約）: 部分利確後に残玉が建値で切れたら
その場のR = partial_frac*partial_r（部分利確ぶんのみ確定、残玉は0R=チャラ）。同足で建値と利確が
両方触れる場合は不利側（建値）優先。最終目標が partial_r 到達価格より近い（rr_final<=partial_r）
トレードは、部分利確が発生する前に100%が目標に達し得るため reason="TP"（通常の単段勝ち）になる
点に注意（部分利確は「価格レベル」で判定するため、近い目標では必ずしも発動しない）。
cost は他バリアントと同じく1トレード1回のみ課す（部分約定による手数料の重複計上はモデル化しない
＝仮定として明記。V0/V1/V2の比較はこの前提の下で行う）。

判定は勝率軸で見る（ユーザー指示）。フル21年のmeanRを「正しい物差し」として持ち込まない。
年別・直近窓(2025-01-01〜2026-07-10)を必ず併記し、フル集計は参考として最後に置く。

自己検査（ビット一致・実行前に必ず通す）:
  1. ict_population.py 単体実行 → n=1148/PF1.17 の台帳アンカーを再現するか
  2. 本スクリプト内 --check で、partial_r=None のV0が ict_extliq_target.py の
     ext_PDH_fluff5 ledger 行 (n=313, win%=34.5, PF=1.41, meanR=+0.281, totR/DD=4.05, maxDD=21.7,
     IS=+60, OOS=+28; ただしその win% は stats()のgross基準。本スクリプトの主指標=net基準win%は別途report) を
     再現するか

Run: .venv/bin/python scratchpad/ict_partial_be_report.py [--smoke] [--check] 2>&1 | \
     tee scratchpad/out_ict_partial_be_report.txt
"""
import sys, io, argparse, contextlib
from collections import Counter
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, PIP, BUF, F_CANON, RR_CANON, walk, stats
from ict_population import canonical_setups, load_prepped
from ict_fvg_anchor import fvg_anchor_fn
from ict_extliq_target import make_ext_tgt_fn, cost_tiers

EURUSD_LIM_FN = fvg_anchor_fn("mid", "long")
EURUSD_MA = 0.15
TIER = "realistic"
RECENT_START = pd.Timestamp("2025-01-01")
RECENT_END = pd.Timestamp("2026-07-10")
YEARS = list(range(2018, 2027))

VARIANTS = [
    ("V0_現行(単一TP)", None, None),
    ("V1_ep41(+1R 50%+BE)", 1.0, 0.5),
    ("V2_(+2R 50%+BE)", 2.0, 0.5),
]


def run_variant(df, tarr, dates, span, partial_r, partial_frac):
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA,
                         use_liq=True, liq_ns=(20, 40))
    sp, cost = cost_tiers("eurusd")[TIER]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")
    skip_log, trade_log = [], []
    tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, "long", lim_fn=EURUSD_LIM_FN,
              tgt_fn=tgt_fn, skip_log=skip_log, trade_log=trade_log,
              partial_r=partial_r, partial_frac=partial_frac)
    n_pop = sum(1 for rec in S if rec["long"] is not None)
    return dict(tr=tr, trade_log=trade_log, skip_log=skip_log, n_pop=n_pop)


def trade_resolution_maxdd(net_arr):
    if len(net_arr) == 0:
        return float("nan")
    cum = np.cumsum(net_arr)
    return float((np.maximum.accumulate(cum) - cum).max())


def summarize(trade_log, label):
    """全体集計: 勝率(net>0)前面, 建値チャラ率, PF/meanR/totR/maxDD(R)/n, 決済理由内訳。"""
    if not trade_log:
        print(f"  [{label}] n=0")
        return None
    net = np.array([t["net"] for t in trade_log])
    n = len(net)
    win = 100.0 * (net > 0).mean()
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    pf = pos / neg if neg > 0 else float("inf")
    meanR = net.mean()
    totR = net.sum()
    dd = trade_resolution_maxdd(net)
    reasons = Counter(t["reason"] for t in trade_log)
    be_n = reasons.get("partial_be", 0)
    be_rate = 100.0 * be_n / n
    half = n // 2
    IS, OOS = net[:half].sum(), net[half:].sum()
    print(f"  [{label}] n={n} win%(net>0)={win:5.1f} 建値チャラ率={be_rate:5.1f}% "
          f"PF={pf:5.2f} meanR={meanR:+.3f} totR={totR:+7.1f} maxDD(R)={dd:6.2f} "
          f"IS={IS:+7.1f} OOS={OOS:+7.1f}")
    tot_n = n
    parts = []
    for key, jp in [("TP", "フルTP"), ("partial_be", "部分TP+建値"),
                    ("partial_tp", "部分TP+残TP"), ("SL", "即SL"),
                    ("partial_timeout", "部分TP+タイムアウト"), ("timeout", "無部分タイムアウト")]:
        c = reasons.get(key, 0)
        if c:
            parts.append(f"{jp}={c}({100.0*c/tot_n:.1f}%)")
    print(f"    決済理由内訳: " + ", ".join(parts))
    return dict(n=n, win=win, be_rate=be_rate, pf=pf, meanR=meanR, totR=totR, dd=dd,
                IS=IS, OOS=OOS, reasons=reasons)


def yearly(trade_log, label):
    print(f"  [{label}] 年別:")
    rows = []
    for t in trade_log:
        y = pd.Timestamp(t["date"]).year
        rows.append((y, t["net"]))
    if not rows:
        print("    (no trades)")
        return
    dfy = pd.DataFrame(rows, columns=["year", "net"])
    for y in YEARS:
        sub = dfy[dfy["year"] == y]
        if len(sub) == 0:
            print(f"    {y}: n=0")
            continue
        win = 100.0 * (sub["net"] > 0).mean()
        print(f"    {y}: n={len(sub):3d} win%={win:5.1f} totR={sub['net'].sum():+7.2f}")


def recent_window(trade_log, label):
    rows = [t for t in trade_log if RECENT_START <= pd.Timestamp(t["date"]) <= RECENT_END]
    if not rows:
        print(f"  [{label}] 直近窓(2025-01-01〜2026-07-10): n=0")
        return None
    net = np.array([t["net"] for t in rows])
    win = 100.0 * (net > 0).mean()
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    pf = pos / neg if neg > 0 else float("inf")
    dd = trade_resolution_maxdd(net)
    print(f"  [{label}] 直近窓(2025-01-01〜2026-07-10): n={len(net)} win%={win:5.1f} "
          f"PF={pf:5.2f} totR={net.sum():+7.2f} maxDD(R)={dd:5.2f}")
    return dict(n=len(net), win=win, pf=pf, totR=net.sum(), dd=dd)


def self_check(df, tarr, dates, span):
    """partial_r=None が ict_extliq_target.py の ext_PDH_fluff5 ledger 行を再現するか。"""
    print("#" * 100)
    print("自己検査: partial_r=None (V0) が ext_PDH_fluff5 の台帳値を再現するか")
    print("  台帳(ict_extliq_target.py, cost=realistic): n=313 win%=34.5(gross基準) PF=1.41 "
          "meanR=+0.281 totR/DD=4.05 maxDD=21.7 IS=+60 OOS=+28")
    print("#" * 100)
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA,
                         use_liq=True, liq_ns=(20, 40))
    sp, cost = cost_tiers("eurusd")[TIER]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")
    tr = walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, "long", lim_fn=EURUSD_LIM_FN,
              tgt_fn=tgt_fn, partial_r=None)
    st = stats(tr, span)
    print(f"  再現値: n={st['n']} win%(gross基準,stats())={st['win']:.1f} PF={st['pf']:.2f} "
          f"meanR={st['net']:+.3f} totR/DD={st['rdd']:.2f} maxDD={st['dd']:.1f} "
          f"IS={st['IS']:+.0f} OOS={st['OOS']:+.0f}")
    ok = (st['n'] == 313)
    print(f"  n一致: {ok}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")
    if args.smoke:
        dates = dates[-int(len(dates) * 0.25):]

    if args.check:
        self_check(df, tarr, dates, span)

    print("=" * 100)
    print("EURUSD 15m long: V0(現行PDH単一TP) vs V1(ep41 +1R 50%利確+建値) vs V2(+2R版)")
    print("=" * 100)

    results = {}
    for label, pr, pf_frac in VARIANTS:
        res = run_variant(df, tarr, dates, span, pr, pf_frac)
        results[label] = res
        print(f"\n--- {label} --- n_pop={res['n_pop']} n_skip={len(res['skip_log'])}")

    print("\n" + "#" * 100)
    print("1. 勝率前面の全体集計（正味R>0基準）+ 建値チャラ率 + 決済理由内訳")
    print("#" * 100)
    summaries = {}
    for label, _, _ in VARIANTS:
        summaries[label] = summarize(results[label]["trade_log"], label)

    print("\n" + "#" * 100)
    print("2. 年別 (2018-2026)")
    print("#" * 100)
    for label, _, _ in VARIANTS:
        yearly(results[label]["trade_log"], label)

    print("\n" + "#" * 100)
    print("3. 直近窓 2025-01-01 〜 2026-07-10（連敗局面）")
    print("#" * 100)
    recents = {}
    for label, _, _ in VARIANTS:
        recents[label] = recent_window(results[label]["trade_log"], label)

    print("\n" + "#" * 100)
    print("4. 参考: フル期間集計（正しい物差しとして持ち込まない。上の1と同じ数値の再掲）")
    print("#" * 100)
    for label, _, _ in VARIANTS:
        s = summaries[label]
        if s is None:
            continue
        print(f"  [{label}] n={s['n']} win%={s['win']:.1f} PF={s['pf']:.2f} meanR={s['meanR']:+.3f} "
              f"totR={s['totR']:+.1f} maxDD(R)={s['dd']:.2f}")

    print("\n" + "#" * 100)
    print("5. 直近窓で V0 の出血を V1/V2 がどれだけ止めたか（一言。samplesが薄い点は必ず併記）")
    print("#" * 100)
    r0, r1, r2 = recents.get(VARIANTS[0][0]), recents.get(VARIANTS[1][0]), recents.get(VARIANTS[2][0])
    if r0:
        print(f"  V0: n={r0['n']} totR={r0['totR']:+.2f} maxDD(R)={r0['dd']:.2f} win%={r0['win']:.1f}")
        if r1:
            print(f"  V1: n={r1['n']} totR={r1['totR']:+.2f} maxDD(R)={r1['dd']:.2f} win%={r1['win']:.1f} "
                  f"(totR差 V1-V0={r1['totR']-r0['totR']:+.2f}, DD差={r1['dd']-r0['dd']:+.2f})")
        if r2:
            print(f"  V2: n={r2['n']} totR={r2['totR']:+.2f} maxDD(R)={r2['dd']:.2f} win%={r2['win']:.1f} "
                  f"(totR差 V2-V0={r2['totR']-r0['totR']:+.2f}, DD差={r2['dd']-r0['dd']:+.2f})")
    print(f"  [注] n が一桁〜十数件の窓なので、この差は参考程度（サンプルが薄い）。")


if __name__ == "__main__":
    main()
