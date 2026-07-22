"""ICT 忠実版 — Draw on Liquidity を「後付けゲート」でなく正典どおりに機械化して測る。

背骨: 前回の週足MAゲート(縮退版)は12ヶ月ブロックでコイン投げ〜悪化で否定済み（docs/verified_findings.md）。
今回は正典そのもの（プール検出＋動的目標＋ディスカウント）を測る。EURUSD 15m long-only。
入口=FVG-CE(mid, fvg_min_atr=0.15)・stop=L-0.1ATR・コスト=realistic(spread0.3pip+comm0.6pip)は
全バリアント共通・不変（因果分離。優先1/優先3 と同じ作法）。

バリアント（全て同じ setups 母集団・同じ entry から派生。差は tgt_fn/tgt_fn_full だけ）:
  G0        : 現行（PDH-5pip固定、ゲート無し） = ict_extliq_target.make_ext_tgt_fn("pdh",5,...) そのまま再利用
  T_dyn     : 利確 = 入口の上で最も近い未タップ buy-side プール（日足優先・無ければ週足、距離<=6*ATR14）-5pip
              射程内のプールが無ければそのトレードを建てない。RRは出力。
  G_dir     : 目標PDHのまま。直近の未タップ週足流動性(buy/sell両側で最近接)が buy-side の時だけロング許可。
  G_disc    : 目標PDHのまま。入口CE(=entry)が直近の確定日足スイング高安の中点50%より下の時だけ許可。
  T_dyn+G_disc : ディスカウント入口ゲート × 動的ドロー目標（正典フュージョン）。

再利用（車輪の再発明禁止）:
  - ict_population.canonical_setups/load_prepped, ict_exec.walk/stats/sc/MODEL/PIP/BUF,
    ict_fvg_anchor.fvg_anchor_fn（entry=mid）, ict_extliq_target.make_ext_tgt_fn（G0のPDH-fluffそのもの）,
    ict_audit.random_drop_null/block_boot, research.overfit_audit.psr/sr0。
  - walk() の tgt_fn_full 拡張（本タスクで ict_exec.py に追加。tgt_fn_full=None 時は挙動不変・
    既存呼び出し元は無改変。約定時刻 fill_dt を tgt_fn に渡す必要があるための最小拡張）。
  - プール検出は ict_draw_on_liq_pools.py（新規・先読み厳禁で自己検査済み）。

先読み厳禁の要石: tgt_fn_full は walk() から「約定足の broker_dt」を渡されるだけで、確定済み・未タップの
判定は untapped_mask(confirm_ts, tap_ts, fill_dt) が「fill_dt までの事実」だけで行う
（ict_draw_on_liq_pools.py の自己検査を参照）。

掟の検定（本丸）:
  - random-drop null: G0母集団から同数間引きした totR/DD 分布に対する各バリアントの %ile（必要条件どまり）
  - 巡回ブロック・ブートストラップ 1/3/6/12ヶ月（各3000回）: 「同じ月ブロック抽選をG0とバリアント双方に
    適用し、バリアントの totR が G0 の totR を上回る確率 P」（ペア比較・全期間の月グリッドで揃える）
  - Deflated Sharpe（変種数=4: T_dyn/G_dir/G_disc/T_dyn+G_disc の相互SR分散で補正、参考値）

Run: .venv/bin/python experiments/ict_draw_on_liq.py [--smoke] 2>&1 | tee experiments/out_ict_draw_on_liq.txt
"""
import sys, io, argparse, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, PIP, BUF, RR_CANON, walk, stats, sc
from ict_population import canonical_setups, load_prepped
from ict_audit import random_drop_null, block_boot
from ict_fvg_anchor import fvg_anchor_fn
from ict_extliq_target import make_ext_tgt_fn
from ict_draw_on_liq_pools import build_pools, untapped_mask, load_htf_naive, DAILY_PATH, WEEKLY_PATH, WEEKLY_CUTOFF
from research.overfit_audit import psr, sr0

RNG_BOOT = np.random.default_rng(20260716)
NAME = "eurusd"
SIDE = "long"
MA = 0.15                      # fvg_min_atr（優先1/優先3の生存セルと同一）
ENTRY_LIM_FN = fvg_anchor_fn("mid", "long")
ATR_MULT = 6.0
FLUFF_PIPS = 5
YEARS = list(range(2018, 2027))
BLOCK_MONTHS = [1, 3, 6, 12]
NREP = 3000


# ---------------------------------------------------------------------------
# バリアント別 tgt_fn / tgt_fn_full の構築
# ---------------------------------------------------------------------------
def discount_ok(pools, entry, fd):
    """G_disc の中核: 直近の確定日足スイング高値/安値（タップ有無は不問）の中点より entry が下か。"""
    dh, dl = pools["daily_buy"], pools["daily_sell"]
    mh, ml = dh["confirm"] <= fd, dl["confirm"] <= fd
    if not mh.any() or not ml.any():
        return None, "no_range"
    idx_h = np.where(mh)[0][np.argmax(dh["confirm"][mh])]
    idx_l = np.where(ml)[0][np.argmax(dl["confirm"][ml])]
    hi, lo = dh["level"][idx_h], dl["level"][idx_l]
    mid = 0.5 * (hi + lo)
    if entry >= mid:
        return None, "premium_blocked"
    return mid, None


def tdyn_target(pools, entry, atr, fd, pip, fluff_pips, source_log=None):
    """T_dyn の中核: 日足優先・無ければ週足、距離<=ATR_MULT*atr の最近接 untapped buy-side プール - fluff。"""
    limit_price = entry + ATR_MULT * atr
    db = pools["daily_buy"]
    md = untapped_mask(db["confirm"], db["tap"], fd)
    lv_d = db["level"][md]
    cand_d = lv_d[(lv_d > entry) & (lv_d <= limit_price)]
    if len(cand_d):
        obj, src = float(cand_d.min()), "daily"
    else:
        wb = pools["weekly_buy"]
        mw = untapped_mask(wb["confirm"], wb["tap"], fd)
        lv_w = wb["level"][mw]
        cand_w = lv_w[(lv_w > entry) & (lv_w <= limit_price)]
        if len(cand_w):
            obj, src = float(cand_w.min()), "weekly"
        else:
            return None, "no_pool_in_range"
    tgt = obj - fluff_pips * pip
    if tgt <= entry:
        return None, "objective_at_or_below_entry"
    if source_log is not None:
        source_log.append((fd, src))
    return tgt, None


def make_g0_fn():
    return make_ext_tgt_fn("pdh", FLUFF_PIPS, NAME, "long")


def make_tdyn_fn(pools, pip, source_log=None):
    def fn(s, entry, risk, fill_dt):
        fd = np.datetime64(fill_dt)
        return tdyn_target(pools, entry, s["atr"], fd, pip, FLUFF_PIPS, source_log=source_log)
    return fn


def make_gdir_fn(pools, base_fn):
    def fn(s, entry, risk, fill_dt):
        fd = np.datetime64(fill_dt)
        wb, ws = pools["weekly_buy"], pools["weekly_sell"]
        mb = untapped_mask(wb["confirm"], wb["tap"], fd)
        ms = untapped_mask(ws["confirm"], ws["tap"], fd)
        lv_b, lv_s = wb["level"][mb], ws["level"][ms]
        cands = []
        if len(lv_b):
            d = np.abs(lv_b - entry); cands.append((float(d.min()), "buy"))
        if len(lv_s):
            d = np.abs(lv_s - entry); cands.append((float(d.min()), "sell"))
        if not cands:
            return None, "no_weekly_pool"
        cands.sort(key=lambda x: x[0])
        if cands[0][1] != "buy":
            return None, "draw_down_blocked"
        return base_fn(s, entry, risk)
    return fn


def make_gdisc_fn(pools, base_fn):
    def fn(s, entry, risk, fill_dt):
        fd = np.datetime64(fill_dt)
        _, reason = discount_ok(pools, entry, fd)
        if reason is not None:
            return None, reason
        return base_fn(s, entry, risk)
    return fn


def make_tdyn_gdisc_fn(pools, pip, source_log=None):
    def fn(s, entry, risk, fill_dt):
        fd = np.datetime64(fill_dt)
        _, reason = discount_ok(pools, entry, fd)
        if reason is not None:
            return None, reason
        return tdyn_target(pools, entry, s["atr"], fd, pip, FLUFF_PIPS, source_log=source_log)
    return fn


# ---------------------------------------------------------------------------
def run_variant(df, S, name, tgt_fn=None, tgt_fn_full=None, skip_log=None, rr_log=None):
    sp, cost = MODEL[name]
    tr = walk(df, S, 0.25, RR_CANON, BUF, sp, cost, "long", lim_fn=ENTRY_LIM_FN,
              tgt_fn=tgt_fn, tgt_fn_full=tgt_fn_full, skip_log=skip_log, rr_log=rr_log)
    return tr


def fmt_row(label, tr, span):
    st = stats(tr, span)
    if st is None:
        return f"  {label:16s} n={len(tr):5d} (<10, skip)"
    return (f"  {label:16s} n={st['n']:5d} n/yr={st['npy']:5.1f} win%={st['win']:5.1f} PF={st['pf']:5.2f} "
            f"meanR={st['net']:+.3f} totR={st['tot']:+7.1f} maxDD={st['dd']:6.1f} totR/DD={st['rdd']:6.2f} "
            f"IS={st['IS']:+7.0f} OOS={st['OOS']:+7.0f}")


def annual_row(tr, g0_tr, years):
    ydf = pd.DataFrame([(pd.Timestamp(d).year, net, g) for d, net, g, risk in tr],
                       columns=["year", "net", "g"])
    g0y = pd.DataFrame([(pd.Timestamp(d).year, ) for d, net, g, risk in g0_tr], columns=["year"])
    g0_counts = g0y["year"].value_counts()
    out = []
    for y in years:
        sub = ydf[ydf["year"] == y]
        n = len(sub)
        win = 100 * (sub["g"] > 0).mean() if n else float("nan")
        tot = sub["net"].sum() if n else 0.0
        g0n = int(g0_counts.get(y, 0))
        on_pct = 100.0 * n / g0n if g0n else float("nan")
        out.append((y, n, win, tot, on_pct))
    return out


def month_universe(dates):
    return pd.period_range(start=pd.Timestamp(dates.min()), end=pd.Timestamp(dates.max()), freq="M")


def month_sums(tr, universe):
    if not tr:
        return pd.Series(0.0, index=universe)
    s = pd.Series([t[1] for t in tr], index=pd.to_datetime([t[0] for t in tr])).sort_index()
    m = s.groupby(s.index.to_period("M")).sum()
    return m.reindex(universe, fill_value=0.0)


def paired_block_boot(var_m, g0_m, months, nrep=NREP, rng=None):
    if rng is None:
        rng = RNG_BOOT
    periods = var_m.index
    nb = len(periods) // months
    if nb < 4:
        return np.nan
    blocks = [periods[i * months:(i + 1) * months] for i in range(nb)]
    var_arr = var_m.values
    g0_arr = g0_m.values
    block_var_sums = np.array([var_arr[i * months:(i + 1) * months].sum() for i in range(nb)])
    block_g0_sums = np.array([g0_arr[i * months:(i + 1) * months].sum() for i in range(nb)])
    wins = 0
    for _ in range(nrep):
        sel = rng.integers(0, nb, nb)
        v_tot = block_var_sums[sel].sum()
        g_tot = block_g0_sums[sel].sum()
        if v_tot > g_tot:
            wins += 1
    return 100.0 * wins / nrep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print("#" * 110)
    print("0. 検算アンカー: EURUSD-long-FVG-CE(mid,ma=0.15)/PDH-5pip固定(G0) が ict_extliq_target の")
    print("   ext_PDH_fluff5(realistic) の値 n=313/win34.5/PF1.41/totR-DD4.05/maxDD21.7 を再現するか")
    print("#" * 110)

    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped(NAME)
    if args.smoke:
        dates = dates[-int(len(dates) * 0.25):]

    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=MA, use_liq=True, liq_ns=(20, 40))

    g0_fn = make_g0_fn()
    tr_g0 = run_variant(df, S, NAME, tgt_fn=g0_fn)
    st0 = stats(tr_g0, span)
    print(f"  G0(検算) n={st0['n']} win%={st0['win']:.1f} PF={st0['pf']:.2f} totR/DD={st0['rdd']:.2f} "
          f"maxDD={st0['dd']:.1f}")

    print("\nプール構築 (daily/weekly, EURUSD)...")
    m15_ts = df["broker_dt"].values.astype("datetime64[ns]")
    m15_high, m15_low = df["high"].values, df["low"].values
    daily_df = load_htf_naive(DAILY_PATH)
    weekly_df = load_htf_naive(WEEKLY_PATH, cutoff=WEEKLY_CUTOFF)
    pools = build_pools(m15_ts, m15_high, m15_low, daily_df, weekly_df)
    for k, v in pools.items():
        print(f"  {k:12s} n_pool={len(v['level']):5d}")

    pip = PIP[NAME]
    src_tdyn, src_fusion = [], []
    skip_g0, skip_tdyn, skip_gdir, skip_gdisc, skip_fusion = [], [], [], [], []
    rr_g0, rr_tdyn, rr_gdir, rr_gdisc, rr_fusion = [], [], [], [], []

    tr_g0 = run_variant(df, S, NAME, tgt_fn=g0_fn, skip_log=skip_g0, rr_log=rr_g0)
    tr_tdyn = run_variant(df, S, NAME, tgt_fn_full=make_tdyn_fn(pools, pip, src_tdyn),
                          skip_log=skip_tdyn, rr_log=rr_tdyn)
    tr_gdir = run_variant(df, S, NAME, tgt_fn_full=make_gdir_fn(pools, g0_fn),
                          skip_log=skip_gdir, rr_log=rr_gdir)
    tr_gdisc = run_variant(df, S, NAME, tgt_fn_full=make_gdisc_fn(pools, g0_fn),
                           skip_log=skip_gdisc, rr_log=rr_gdisc)
    tr_fusion = run_variant(df, S, NAME, tgt_fn_full=make_tdyn_gdisc_fn(pools, pip, src_fusion),
                            skip_log=skip_fusion, rr_log=rr_fusion)

    variants = [("G0", tr_g0), ("T_dyn", tr_tdyn), ("G_dir", tr_gdir),
               ("G_disc", tr_gdisc), ("T_dyn+G_disc", tr_fusion)]

    print("\n" + "#" * 110)
    print("1. 全バリアント横並び (n / n/yr / win% / PF / meanR / totR / maxDD / totR-DD / IS-OOS)")
    print("#" * 110)
    for label, tr in variants:
        print(fmt_row(label, tr, span))
    print(f"\n  skip件数: G0={len(skip_g0)} T_dyn={len(skip_tdyn)} G_dir={len(skip_gdir)} "
          f"G_disc={len(skip_gdisc)} T_dyn+G_disc={len(skip_fusion)}")
    for label, sl in [("G0", skip_g0), ("T_dyn", skip_tdyn), ("G_dir", skip_gdir),
                      ("G_disc", skip_gdisc), ("T_dyn+G_disc", skip_fusion)]:
        if sl:
            c = pd.Series([r for _, r in sl]).value_counts()
            print(f"    {label:14s}: " + ", ".join(f"{k}={v}" for k, v in c.items()))

    print("\n" + "#" * 110)
    print("2. 年別 (2018-2026): n / win% / totR / ON%(=採用/G0同年n)")
    print("#" * 110)
    for label, tr in variants:
        rows = annual_row(tr, tr_g0, YEARS)
        line = "  ".join(f"{y}:n={n:3d},win={w:5.1f},tot={t:+6.1f},ON={o:5.1f}%"
                         if n else f"{y}:n=0" for y, n, w, t, o in rows)
        print(f"  {label:14s} " + line)

    print("\n" + "#" * 110)
    print("3. 掟の検定")
    print("#" * 110)
    print("\n3a. random-drop null（G0母集団から同数間引き、totR/DDの%ile、必要条件どまり）")
    for label, tr in variants[1:]:
        st = stats(tr, span)
        if st is None:
            print(f"  {label:14s} n<10 skip")
            continue
        null = random_drop_null(tr_g0, st["n"], nrep=NREP)
        pct = 100.0 * (null < st["rdd"]).mean()
        print(f"  {label:14s} 実測totR/DD={st['rdd']:+.2f}  null分布(n={st['n']}抽出,{NREP}回) "
              f"median={np.median(null):+.2f}  %ile={pct:.1f}")

    print("\n3b. 巡回ブロック・ブートストラップ（対G0ペア比較, 各3000回, P(バリアントtotR > G0 totR)）")
    universe = month_universe(dates)
    g0_month = month_sums(tr_g0, universe)
    for label, tr in variants[1:]:
        var_month = month_sums(tr, universe)
        row = []
        for m in BLOCK_MONTHS:
            p = paired_block_boot(var_month, g0_month, m)
            row.append(f"{m}mo={p:.1f}%")
        print(f"  {label:14s} " + "  ".join(row))

    print("\n3c. Deflated Sharpe（変種数=4: T_dyn/G_dir/G_disc/T_dyn+G_disc の相互SR分散で補正、参考値）")
    srs = {}
    for label, tr in variants[1:]:
        net = np.array([t[1] for t in tr])
        if len(net) >= 10:
            srs[label] = net.mean() / net.std(ddof=1)
    if len(srs) >= 2:
        Vsr = float(np.var(list(srs.values())))
        thr = sr0(4, Vsr)
        for label, tr in variants[1:]:
            net = np.array([t[1] for t in tr])
            if len(net) < 10:
                print(f"  {label:14s} n<10 skip")
                continue
            dsr, sr, g1, g4 = psr(net, thr)
            print(f"  {label:14s} n={len(net):5d} SR/tr={sr:+.3f} V_SR(4変種)={Vsr:.5f} "
                  f"sr0@N=4={thr:+.3f} DSR={dsr:.2f}")
    else:
        print("  変種のSR算出に十分なnが無い（skip）")

    print("\n" + "#" * 110)
    print("4. T_dyn の実効RR分布 と 目標プールの日足/週足内訳")
    print("#" * 110)
    if rr_tdyn:
        rr_arr = np.array([x[1] for x in rr_tdyn])
        print(f"  T_dyn RR: n={len(rr_arr)} median={np.median(rr_arr):.2f} mean={rr_arr.mean():.2f} "
              f"sd={rr_arr.std(ddof=1):.2f} q25={np.percentile(rr_arr,25):.2f} q75={np.percentile(rr_arr,75):.2f}")
    if rr_g0:
        rr_arr0 = np.array([x[1] for x in rr_g0])
        print(f"  G0(PDH固定) RR:  n={len(rr_arr0)} median={np.median(rr_arr0):.2f} mean={rr_arr0.mean():.2f} "
              f"sd={rr_arr0.std(ddof=1):.2f}")
    if src_tdyn:
        c = pd.Series([s for _, s in src_tdyn]).value_counts()
        tot = len(src_tdyn)
        print(f"  T_dyn 目標の由来: " + ", ".join(f"{k}={v}({100*v/tot:.1f}%)" for k, v in c.items()))
    if src_fusion:
        c = pd.Series([s for _, s in src_fusion]).value_counts()
        tot = len(src_fusion)
        print(f"  T_dyn+G_disc 目標の由来: " + ", ".join(f"{k}={v}({100*v/tot:.1f}%)" for k, v in c.items()))

    print("\n" + "#" * 110)
    print("5. 2025年ピンポイント（週足↑だが G0 は不発の年）")
    print("#" * 110)
    for label, tr in variants:
        sub = [(d, net, g) for d, net, g, risk in tr if pd.Timestamp(d).year == 2025]
        if not sub:
            print(f"  {label:14s} n=0")
            continue
        net = np.array([x[1] for x in sub]); g = np.array([x[2] for x in sub])
        print(f"  {label:14s} n={len(sub):3d} win%={100*(g>0).mean():5.1f} totR={net.sum():+.2f}")


if __name__ == "__main__":
    main()
