"""ICT: 忠実 EURUSD-long システムの最後の未測定の芯 --- 死んだ(アルファ・ディケイ)のか
生きてるが獲れてない(出口の問題)のかを、年代別の素の MFE/MAE で決着する。

背景（docs/findings/s01_entries.md #66）: 出口/週足ゲート/Draw-on-Liquidity の忠実化3方向は
全滅（12ヶ月ブロックでコイン投げ）。凍結アンカー = EURUSD ロング / 狩り+MSS+FVG(displacement,
fvg_min_atr=0.15) / 入口=FVG-CE(mid) / stop=L-0.1ATR / 目標=PDH-5pip / NYキルゾーン
（n=313・PF1.41・totR/DD4.05・maxDD21.7、realistic cost）。T_dyn(n=18)の「上に磁石が無い」は
死の兆候だが未測定 --- これを直接測る。

流用（車輪の再発明禁止）: ict_exec.{SYMS,MODEL,PIP,CUT2000,BUF,F_CANON,load_ny,prep,span_years,
window_pos,walk,stats}／ict_population.{canonical_setups,load_prepped,prev_day_extremes,
last_fractal_high,last_fractal_low}／ict_fvg_anchor.fvg_anchor_fn／ict_extliq_target.{EURUSD_LIM_FN,
EURUSD_MA,make_ext_tgt_fn}／ict_dxy_smt.cost_tiers。新規実装は
(1) MFE+MAE 二重スキャン（既存 mfe_scan は MFE のみ・lim_fn 非対応なので拡張が必要 --- 既存関数を
    壊さず"新規追加"、walk()/tgt_fn 追加と同じ加法パターン）、
(2) 日足/週足フラクタル磁石チェック（3本フラクタルの検出自体は last_fractal_high と同じ条件を再利用）、
(3) 5分足執行エンジン（15分で検出したセットアップを5分足のバー配列で約定判定するだけ --- walk() は
    df の中身を知らないので、5分足の NY壁時計配列に対して window_pos で kz を引き直せば walk() を
    そのまま再利用できる。新規ロジックは無い）。

先読み: population(15分)は全て確定足＋shift(1)（既存コードのまま不変）。日足/週足フラクタルは
「確定日/週」のみ・「その後 未タップ」を fill 時点までの完了済み日/週だけで判定。5分実行も
約定足から前進走査（タダ乗り防止、同足は損切り優先）。

Run:
  .venv/bin/python experiments/ict_alpha_decay.py --smoke
  .venv/bin/python experiments/ict_alpha_decay.py
"""
import sys, io, argparse, contextlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv
from ict_exec import (SYMS, MODEL, PIP, CUT2000, BUF, F_CANON, load_ny, prep, span_years,
                      window_pos, walk, stats, ASIA_HOURS, LONDON_HOURS, KZ_HOURS)
from ict_population import canonical_setups, load_prepped, prev_day_extremes
from ict_fvg_anchor import fvg_anchor_fn
from ict_extliq_target import EURUSD_LIM_FN, EURUSD_MA, make_ext_tgt_fn
from ict_dxy_smt import cost_tiers

FX6 = ["eurusd", "gbpusd", "usdjpy", "audusd", "nzdusd", "usdcad"]
ERAS = [(2018, 2020), (2021, 2023), (2024, 2026)]
RNG = np.random.default_rng(20260716)
NREP = 500


# ============================================================== (1) MFE+MAE scan (新規、lim_fn対応)
def scan_mfe_mae(df, setups, side, lim_fn, spread, cost_tier_sp, fwd_cap=500):
    """狩り+MSS+FVG母集団の各約定について、目標無し(stop-onlyの素の巡行幅)で
    MFE(R, 好都合な最大)・MAE(R, 不都合な最大, stop到達で1.0に打ち切り)・stopped・
    bars_to_fill(kz内)・bars_to_outcome(約定〜損切り/打ち切り)を返す。
    lim_fn=None なら f=0.25固定リトレース、与えれば FVG-CE 等のアンカー(walk()と同じ規約)。
    同足タイブレーク=損切り優先（既存の walk()/mfe_scan と同一規律）。"""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    out = []
    for rec in setups:
        s = rec[side]
        if s is None:
            continue
        L, H, A = s["L"], s["H"], s["atr"]
        k0, k1 = s["kz"]
        if side == "long":
            lim = lim_fn(s) if lim_fn is not None else H - F_CANON * (H - L)
            stop = L - BUF * A
            if lim <= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if l[p] <= lim - spread:
                    fp = p; break
            if fp is None:
                continue
            entry = min(lim, o[fp] + spread)
            risk = entry - stop
            if risk <= 0:
                continue
            mfe = 0.0; mae = 0.0; stopped = False
            bar_stop = None
            for p in range(fp, min(fp + fwd_cap, n)):
                if l[p] <= stop:
                    stopped = True; mae = 1.0; bar_stop = p; break
                fav = (h[p] - entry) / risk
                adv = (entry - l[p]) / risk
                if fav > mfe:
                    mfe = fav
                if adv > mae:
                    mae = adv
            endp = bar_stop if bar_stop is not None else min(fp + fwd_cap, n) - 1
            final = (c[endp] - entry) / risk
        else:
            lim = lim_fn(s) if lim_fn is not None else L + F_CANON * (H - L)
            stop = H + BUF * A
            if lim >= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if h[p] >= lim + spread:
                    fp = p; break
            if fp is None:
                continue
            entry = max(lim, o[fp] - spread)
            risk = stop - entry
            if risk <= 0:
                continue
            mfe = 0.0; mae = 0.0; stopped = False
            bar_stop = None
            for p in range(fp, min(fp + fwd_cap, n)):
                if h[p] >= stop:
                    stopped = True; mae = 1.0; bar_stop = p; break
                fav = (entry - l[p]) / risk
                adv = (h[p] - entry) / risk
                if fav > mfe:
                    mfe = fav
                if adv > mae:
                    mae = adv
            endp = bar_stop if bar_stop is not None else min(fp + fwd_cap, n) - 1
            final = (entry - c[endp]) / risk
        bars_to_outcome = endp - fp
        out.append(dict(date=rec["date"], fp=fp, entry=entry, stop=stop, risk=risk, atr=A,
                        mfe=mfe, mae=mae, stopped=stopped, final=final,
                        bars_to_outcome=bars_to_outcome))
    return out


def era_of(y):
    for a, b in ERAS:
        if a <= y <= b:
            return f"{a}-{b}"
    return None


def summarize(scans, label=""):
    if not scans:
        return None
    mfe = np.array([x["mfe"] for x in scans])
    mae = np.array([x["mae"] for x in scans])
    bars = np.array([x["bars_to_outcome"] for x in scans])
    n = len(mfe)
    return dict(n=n, mfe_med=np.median(mfe), mfe_mean=mfe.mean(),
                mfe_sd=mfe.std(ddof=1) if n > 1 else 0.0,
                p1=100 * np.mean(mfe >= 1), p2=100 * np.mean(mfe >= 2),
                p3=100 * np.mean(mfe >= 3), p4=100 * np.mean(mfe >= 4),
                mae_med=np.median(mae), mae_mean=mae.mean(),
                mae_sd=mae.std(ddof=1) if n > 1 else 0.0,
                stop_rate=100 * np.mean([x["stopped"] for x in scans]),
                bars_med=np.median(bars), bars_mean=bars.mean())


def print_summary_row(label, s):
    if s is None:
        print(f"    {label:<14} n<5 skip")
        return
    print(f"    {label:<14} n={s['n']:>5}  MFE中央値={s['mfe_med']:>6.2f} 平均={s['mfe_mean']:>6.2f} "
          f"sd={s['mfe_sd']:>5.2f}  P(>=1R)={s['p1']:>5.1f}% P(>=2R)={s['p2']:>5.1f}% "
          f"P(>=3R)={s['p3']:>5.1f}% P(>=4R)={s['p4']:>5.1f}%  "
          f"MAE中央値={s['mae_med']:.2f} 平均={s['mae_mean']:.2f} sd={s['mae_sd']:.2f} "
          f"損切り率={s['stop_rate']:>5.1f}%  保有(バー)中央値={s['bars_med']:.0f}")


# ============================================================== (2) magnet check（日足/週足フラクタル）
def build_daily_fractals(df):
    """NY暦の日足高値配列 + 3本フラクタル高値の (confirm_day_idx, day_idx, level) リスト。"""
    g = df.groupby(df["_t"].dt.normalize()).agg(hi=("high", "max"))
    days = g.index.values
    hi = g["hi"].values
    fr = []
    for k in range(1, len(hi) - 1):
        if hi[k] >= hi[k - 1] and hi[k] >= hi[k + 1]:
            fr.append((k + 1, k, hi[k]))     # confirm at k+1 (次日終値で確定)
    return days, hi, fr


def build_weekly_fractals(name):
    with contextlib.redirect_stderr(io.StringIO()):
        w = load_mt5_csv(f"/home/angelbell/dev/auto-trade/data/vantage_{name}_w1.csv")
    hi = w["high"].values
    idx = w.index
    fr = []
    for k in range(1, len(hi) - 1):
        if hi[k] >= hi[k - 1] and hi[k] >= hi[k + 1]:
            fr.append((k + 1, k, hi[k]))
    return idx, hi, fr


def magnet_at_fill(fill_ts, entry, atr, day_index, daily_hi, daily_fr, week_index, weekly_hi, weekly_fr,
                    n_atr=6.0):
    """fill時点(day単位)で「確定済み・未タップの buy-side 流動性プール」が entry の上・射程n_atr*ATR以内に
    あるか。日足/週足それぞれについて判定し、(daily_hit, weekly_hit) を返す。"""
    fday = pd.Timestamp(fill_ts).normalize()
    # -- daily --
    d_hit = False
    di = np.searchsorted(day_index, np.datetime64(fday))       # 最初に fday 以上になる位置
    for confirm_k, pivot_k, level in daily_fr:
        if confirm_k >= di:            # 確定が fill 日以降 -> まだ確定していない
            continue
        if not (entry < level <= entry + n_atr * atr):
            continue
        # untapped: confirm_k(確定日) 〜 di-1(fill前日まで) の日足高値が level 以上に達していないか
        if di > confirm_k and (daily_hi[confirm_k:di] >= level).any():
            continue
        d_hit = True
        break
    # -- weekly --
    w_hit = False
    wi = np.searchsorted(week_index.values, np.datetime64(fday))
    for confirm_k, pivot_k, level in weekly_fr:
        if confirm_k >= wi:
            continue
        if not (entry < level <= entry + n_atr * atr):
            continue
        if wi > confirm_k and (weekly_hi[confirm_k:wi] >= level).any():
            continue
        w_hit = True
        break
    return d_hit, w_hit


def magnet_report(name, df, scans):
    days, daily_hi, daily_fr = build_daily_fractals(df)
    week_index, weekly_hi, weekly_fr = build_weekly_fractals(name)
    rows = []
    for x in scans:
        d_hit, w_hit = magnet_at_fill(x["date"], x["entry"], x["atr"], days, daily_hi, daily_fr,
                                      week_index, weekly_hi, weekly_fr)
        rows.append(dict(date=x["date"], daily=d_hit, weekly=w_hit, either=d_hit or w_hit, mfe=x["mfe"]))
    return rows


# ============================================================== (3) 同数ランダム入場 null（KZ内、同risk）
def random_kz_null(df, setups, side, lim_fn, spread, nrep=NREP):
    """既存 ict_killzone.null2_random_entry と同じ方式: 実際に約定した日について、その日の
    KZ窓の全バーを「同じ risk・open約定」で評価し、1本/日を Monte Carlo 再抽選して null帯を作る。
    stop-onlyのMFEで比較する（今回はRRを課さないため）。"""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    per_day_mfe = []
    real_mfe = []
    for rec in setups:
        s = rec[side]
        if s is None:
            continue
        L, H, A = s["L"], s["H"], s["atr"]
        k0, k1 = s["kz"]
        if side != "long":
            continue
        lim = lim_fn(s) if lim_fn is not None else H - F_CANON * (H - L)
        stop = L - BUF * A
        if lim <= stop:
            continue
        fp = None
        for p in range(k0, k1):
            if l[p] <= lim - spread:
                fp = p; break
        if fp is None:
            continue
        entry_real = min(lim, o[fp] + spread)
        risk = entry_real - stop
        if risk <= 0:
            continue
        real_mfe.append(mfe_for_entry(o, h, l, fp, entry_real, stop, risk))
        # 同じ risk を使い、KZ 窓の全バー(open約定)を仮想エントリーとして評価
        outcomes = np.empty(k1 - k0)
        for j, p in enumerate(range(k0, k1)):
            e2 = o[p]
            s2 = e2 - risk
            outcomes[j] = mfe_for_entry(o, h, l, p, e2, s2, risk)
        per_day_mfe.append(outcomes)
    if not per_day_mfe:
        return None
    real_mfe = np.array(real_mfe)
    rep_meds = np.empty(nrep)
    rep_p3 = np.empty(nrep)
    for rep in range(nrep):
        picks = np.array([arr[RNG.integers(0, len(arr))] for arr in per_day_mfe])
        rep_meds[rep] = np.median(picks)
        rep_p3[rep] = 100 * np.mean(picks >= 3)
    real_med = np.median(real_mfe)
    real_p3 = 100 * np.mean(real_mfe >= 3)
    pct_med = 100 * np.mean(rep_meds < real_med)
    pct_p3 = 100 * np.mean(rep_p3 < real_p3)
    return dict(n=len(real_mfe), real_med=real_med, real_p3=real_p3,
                null_med_lo=np.percentile(rep_meds, 2.5), null_med_hi=np.percentile(rep_meds, 97.5),
                null_p3_lo=np.percentile(rep_p3, 2.5), null_p3_hi=np.percentile(rep_p3, 97.5),
                pct_med=pct_med, pct_p3=pct_p3)


def mfe_for_entry(o, h, l, p0, entry, stop, risk, fwd_cap=500):
    n = len(o)
    mfe = 0.0
    for p in range(p0, min(p0 + fwd_cap, n)):
        if l[p] <= stop:
            break
        fav = (h[p] - entry) / risk
        if fav > mfe:
            mfe = fav
    return mfe


# ============================================================== stage A: tie-back
def stage_tieback():
    print("=" * 110)
    print("STAGE A: tie-back --- 凍結アンカー EURUSD-long-FVG-CE(mid) / stop L-0.1ATR / tgt PDH-5pip の再現")
    print("台帳: n=313 / win%34.5 / PF1.41 / totR/DD4.05 / maxDD21.7 (realistic cost)")
    print("=" * 110)
    df, tarr, dates, span = load_prepped("eurusd")
    S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA, use_liq=True, liq_ns=(20, 40))
    sp, cost = cost_tiers("eurusd")["realistic"]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")
    tr = walk(df, S, F_CANON, 4.0, BUF, sp, cost, "long", lim_fn=EURUSD_LIM_FN, tgt_fn=tgt_fn)
    st = stats(tr, span)
    print(f"  再現値: n={st['n']}  win%={st['win']:.1f}  PF={st['pf']:.2f}  totR/DD={st['rdd']:.2f}  "
          f"maxDD={st['dd']:.1f}  meanR={st['net']:+.3f}  IS={st['IS']:+.0f} OOS={st['OOS']:+.0f}")
    ok = abs(st['n'] - 313) <= 3 and abs(st['pf'] - 1.41) < 0.05 and abs(st['rdd'] - 4.05) < 0.3
    print(f"  tie-back {'OK（許容誤差内で一致）' if ok else '不一致 --- 要確認'}")
    return df, tarr, dates, span, S


# ============================================================== stage B: 15m MFE/MAE by era, FX6
def stage_b_15m_mfe(smoke=False):
    print("\n" + "=" * 110)
    print("STAGE B: 15分執行 --- 狩り+MSS+FVG(displacement,ma=0.15) 母集団, 入口=FVG-CE(mid) 全銘柄共通,")
    print("  stop=L-0.1ATR, 目標なし(素のMFE/MAE)。FX6 x 年代(2018-20/2021-23/2024-26) + 年別")
    print("  ⚠️ 宣言: 5銘柄(gbp/jpy/aud/nzd/cad)への lim_fn=FVG-CE(mid) 適用は本タスク専用の統一比較")
    print("     （個々の銘柄で最良と確認された設定ではない。目的=同一検出器での年代別MFE比較）")
    print("=" * 110)
    all_scans = {}
    all_dfs = {}
    for name in FX6:
        df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.2):]
        S = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=0.15, use_liq=True, liq_ns=(20, 40))
        sp, cost = cost_tiers(name)["realistic"]
        lim_fn = fvg_anchor_fn("mid", "long")
        scans = scan_mfe_mae(df, S, "long", lim_fn, sp, cost)
        all_scans[name] = scans
        all_dfs[name] = df
        n_pop = sum(1 for rec in S if rec["long"] is not None)
        print(f"\n  --- {name} (母集団n={n_pop}, 約定n={len(scans)}, 約定率={100*len(scans)/max(n_pop,1):.0f}%) ---")
        print_summary_row("全期間", summarize(scans))
        by_era = {}
        for x in scans:
            e = era_of(pd.Timestamp(x["date"]).year)
            if e:
                by_era.setdefault(e, []).append(x)
        for a, b in ERAS:
            e = f"{a}-{b}"
            print_summary_row(e, summarize(by_era.get(e, [])))
        print("    年別:")
        by_year = {}
        for x in scans:
            y = pd.Timestamp(x["date"]).year
            by_year.setdefault(y, []).append(x)
        for y in sorted(by_year):
            print_summary_row(str(y), summarize(by_year[y]))
    return all_scans, all_dfs


# ============================================================== stage C: random-KZ null by era
def stage_c_null(smoke=False):
    print("\n" + "=" * 110)
    print("STAGE C: 同数ランダムKZ内入場 null（同risk・500回再抽選）--- 年代別に real vs null帯")
    print("=" * 110)
    for name in FX6:
        df, tarr, dates, span = load_prepped(name)
        if smoke:
            dates = dates[-int(len(dates) * 0.2):]
        sp, cost = cost_tiers(name)["realistic"]
        lim_fn = fvg_anchor_fn("mid", "long")
        print(f"\n  --- {name} ---")
        for a, b in ERAS:
            d_era = np.array([d for d in dates if a <= pd.Timestamp(d).year <= b])
            if len(d_era) < 30:
                print(f"    {a}-{b}: 日数不足でskip"); continue
            S = canonical_setups(df, tarr, d_era, 0, use_fvg=True, fvg_min_atr=0.15, use_liq=True, liq_ns=(20, 40))
            res = random_kz_null(df, S, "long", lim_fn, sp)
            if res is None:
                print(f"    {a}-{b}: 約定なしでskip"); continue
            print(f"    {a}-{b}: n={res['n']}  実測MFE中央値={res['real_med']:.2f} "
                  f"(null帯[{res['null_med_lo']:.2f},{res['null_med_hi']:.2f}] -> {res['pct_med']:.0f}%ile)  "
                  f"実測P(>=3R)={res['real_p3']:.1f}% (null帯[{res['null_p3_lo']:.1f}%,{res['null_p3_hi']:.1f}%] "
                  f"-> {res['pct_p3']:.0f}%ile)")


# ============================================================== stage D: magnet check by era
def stage_d_magnet(all_scans, all_dfs):
    print("\n" + "=" * 110)
    print("STAGE D: 磁石チェック（fill時点で entry の上・射程6ATR以内に、確定済み・未タップの")
    print("  日足/週足3本フラクタル高値があるか）--- 年代別「磁石あり率」")
    print("=" * 110)
    for name in FX6:
        scans = all_scans[name]
        df = all_dfs[name]
        if not scans:
            print(f"\n  --- {name}: scan無し ---"); continue
        rows = magnet_report(name, df, scans)
        print(f"\n  --- {name} (n={len(rows)}) ---")
        by_era = {}
        for r in rows:
            e = era_of(pd.Timestamp(r["date"]).year)
            if e:
                by_era.setdefault(e, []).append(r)
        for a, b in ERAS:
            e = f"{a}-{b}"
            rs = by_era.get(e, [])
            if len(rs) < 5:
                print(f"    {e}: n={len(rs)} 不足でskip"); continue
            d_rate = 100 * np.mean([r["daily"] for r in rs])
            w_rate = 100 * np.mean([r["weekly"] for r in rs])
            either = 100 * np.mean([r["either"] for r in rs])
            mfe_with = [r["mfe"] for r in rs if r["either"]]
            mfe_without = [r["mfe"] for r in rs if not r["either"]]
            mw = f"{np.median(mfe_with):.2f}" if mfe_with else "n/a"
            mwo = f"{np.median(mfe_without):.2f}" if mfe_without else "n/a"
            print(f"    {e}: n={len(rs):>4}  日足磁石あり率={d_rate:>5.1f}%  週足磁石あり率={w_rate:>5.1f}%  "
                  f"いずれか={either:>5.1f}%   MFE中央値(磁石あり)={mw}  MFE中央値(磁石なし)={mwo}")


# ============================================================== stage E: 5分執行版（EURUSD先行）
def load_5m_ny(name):
    with contextlib.redirect_stderr(io.StringIO()):
        df, n_nat = load_ny(f"/home/angelbell/dev/auto-trade/data/vantage_{name}_m5.csv",
                            cut2000=(name in CUT2000))
    return df


def setups_with_5m_kz(setups15, dates, tarr5):
    """15分で検出したセットアップの kz窓だけを、5分足のNY壁時計配列上の位置に引き直す。
    L/H/atr/fvg_lo/fvg_hi/pdh 等の価格レベルはTF非依存でそのまま流用（新規ロジック無し、
    walk() が「どのTFのdfか」を知らない性質をそのまま使う）。"""
    K0H, K1H = KZ_HOURS
    out = []
    for rec in setups15:
        s = rec.get("long")
        newrec = {"date": rec["date"], "long": None, "short": None}
        if s is not None:
            day = pd.Timestamp(rec["date"])
            k0, k1 = window_pos(tarr5, day + pd.Timedelta(hours=K0H), day + pd.Timedelta(hours=K1H))
            if k1 > k0:
                s2 = dict(s); s2["kz"] = (k0, k1)
                newrec["long"] = s2
        out.append(newrec)
    return out


def stage_e_5m_eurusd(smoke=False):
    print("\n" + "=" * 110)
    print("STAGE E: 5分足執行版（EURUSD先行 --- gbpusd/usdjpyもm5データありだが今回はEURUSDのみ、")
    print("  audusd/nzdusd/usdcad は m5 CSV が存在しないため5分執行は実施不可 --- データ制約）")
    print("  検出=15分（狩り+MSS+FVG, ma=0.15）のまま／執行=5分（FVG-CE mid のタップ・stop判定を5分足で）")
    print("=" * 110)
    df15, tarr15, dates, span = load_prepped("eurusd")
    if smoke:
        dates = dates[-int(len(dates) * 0.2):]
    S15 = canonical_setups(df15, tarr15, dates, 0, use_fvg=True, fvg_min_atr=0.15, use_liq=True, liq_ns=(20, 40))

    df5 = load_5m_ny("eurusd")
    df5, tarr5, dates5 = prep(df5)
    S5 = setups_with_5m_kz(S15, dates, tarr5)

    sp, cost = cost_tiers("eurusd")["realistic"]
    lim_fn = fvg_anchor_fn("mid", "long")

    scans15 = scan_mfe_mae(df15, S15, "long", lim_fn, sp, cost)
    scans5 = scan_mfe_mae(df5, S5, "long", lim_fn, sp, cost)
    print(f"\n  15分執行: n={len(scans15)}")
    print_summary_row("全期間(15m)", summarize(scans15))
    print(f"\n  5分執行 : n={len(scans5)}")
    print_summary_row("全期間(5m)", summarize(scans5))
    print("\n  年代別比較（15分 vs 5分執行）:")
    for a, b in ERAS:
        e15 = [x for x in scans15 if a <= pd.Timestamp(x["date"]).year <= b]
        e5 = [x for x in scans5 if a <= pd.Timestamp(x["date"]).year <= b]
        print(f"    {a}-{b}:")
        print_summary_row("  15m", summarize(e15))
        print_summary_row("  5m", summarize(e5))

    # tie-back to RR4/PDH-5pip目標付きの正式トレードで、5分執行が生成する n が妥当か(桁の一致)確認
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")
    tr15 = walk(df15, S15, F_CANON, 4.0, BUF, sp, cost, "long", lim_fn=lim_fn, tgt_fn=tgt_fn)
    tr5 = walk(df5, S5, F_CANON, 4.0, BUF, sp, cost, "long", lim_fn=lim_fn, tgt_fn=tgt_fn)
    st15 = stats(tr15, span); st5 = stats(tr5, span)
    print(f"\n  参考(PDH-5pip目標込み・フルシステム): 15分執行 n={st15['n'] if st15 else 0} "
          f"PF={st15['pf']:.2f} totR/DD={st15['rdd']:.2f}" if st15 else "  15分執行: n<10")
    if st5:
        print(f"                                      5分執行  n={st5['n']} PF={st5['pf']:.2f} "
              f"totR/DD={st5['rdd']:.2f}")
    else:
        print("                                      5分執行: n<10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stages", default="ABCDE", help="実行するステージ文字（例 AB）")
    args = ap.parse_args()

    if "A" in args.stages:
        stage_tieback()
    all_scans, all_dfs = None, None
    if "B" in args.stages:
        all_scans, all_dfs = stage_b_15m_mfe(smoke=args.smoke)
    if "C" in args.stages:
        stage_c_null(smoke=args.smoke)
    if "D" in args.stages:
        if all_scans is None:
            all_scans, all_dfs = stage_b_15m_mfe(smoke=args.smoke)
        stage_d_magnet(all_scans, all_dfs)
    if "E" in args.stages:
        stage_e_5m_eurusd(smoke=args.smoke)


if __name__ == "__main__":
    main()
