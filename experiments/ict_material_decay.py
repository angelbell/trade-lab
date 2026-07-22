"""ICT 素材の経年診断 — 狩り+MSS(+FVGなし) という「原材料」が年代とともに腐ったか。

出口・約定の最適化は一切しない診断スクリプト。母集団は ict_population.canonical_setups を
そのまま使う（ロング側のみ、shift=0）。基準点・リスク単位は仕様カード通り:
  - 基準点 = KZ 開始バーの始値 o[k0]（セットアップ確定後、最初に建てられる価格。指値約定の
    選別＝どのバーで指値が刺さったかは一切混ぜない）
  - リスク単位 = 基準点 − (狩られた安値 L − 0.1×ATR14)   … BUF=0.1 は ict_exec.BUF と同一
  - 前進窓 = 500本（ict_exec.FWD_CAP と同一の地平）

「狩り→MSS の所要バー数」と「狩りの深さ」は canonical_setups() の返り値（L, H, atr, kz のみ）
には入っていないため、build() のロング側ロジックを診断用に複製した long_diag() で別途抽出する。
複製が正しいことは自己検査で保証する: long_diag() の (date, L, H) が canonical_setups() の
ロング側と完全一致すること（tie-back）。母集団の生成そのもの（採否判定）は複製せず
canonical_setups() の出力をそのまま使うので、母集団の数字は二重実装になっていない。

ランダム対照: 同じ年代のランダムな KZ 開始日（狩り/MSS条件は無視、窓の有効性のみ要求）を
300本（不足時は全数）サンプルし、同じ基準点(KZ開始バー始値)・同じ前進窓(500本)で MFE(R) を
測る。リスク単位は「その日の KZ 開始バーの ATR14（係数1.0の近似）」を使う（セットアップが
無い日にはストップ幅の定義が無いための近似 — 仕様カードの許可通り、方法をここに明記する）。

自己検査: .venv/bin/python experiments/ict_material_decay.py --selfcheck
本測定:   .venv/bin/python experiments/ict_material_decay.py [--smoke] [--pairs eurusd,gbpusd,...]
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import argparse
import io
import contextlib
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import (SYMS, BUF, F_CANON, RR_CANON, ASIA_HOURS, LONDON_HOURS, KZ_HOURS,
                       FWD_CAP, load_ny, prep, span_years, window_pos, walk, stats, CUT2000)
from ict_population import (canonical_setups, prev_day_extremes, last_fractal_high,
                             load_prepped)

PAIRS = ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad"]
ERAS = [(2000, 2005), (2006, 2011), (2012, 2017), (2018, 2023), (2024, 2024),
        (2025, 2025), (2026, 2026)]
N_RAND = 300
SEED_BASE = 20260716  # 今日の日付を種に固定（再現性用）


def era_label(lo, hi):
    return f"{lo}" if lo == hi else f"{lo}-{str(hi)[2:]}"


def era_of_year(y):
    for lo, hi in ERAS:
        if lo <= y <= hi:
            return era_label(lo, hi)
    return None


# ---------------------------------------------------------------------------
def long_diag(df, tarr, dates):
    """build() のロング側ロジックを診断用に複製（狩り/MSS の位置を抽出するため）。
    canonical_setups() と同じ採否条件（狩り+MSS+H-L>=0.25ATR+無効化なし）。
    返り値: date -> dict(L,H,atr,iL,jm,k0,k1,bars_hunt_to_mss,depth_atr)"""
    hi, lo = df["high"].values, df["low"].values
    atr = df["atr14"].values
    pdh, pdl = prev_day_extremes(df, dates)
    A0H, A1H = ASIA_HOURS
    L0H, L1H = LONDON_HOURS
    K0H, K1H = KZ_HOURS
    out = {}
    for d in dates:
        day = pd.Timestamp(d)
        a0, a1 = window_pos(tarr, day - pd.Timedelta(days=1) + pd.Timedelta(hours=A0H),
                             day + pd.Timedelta(hours=A1H))
        l0, l1 = window_pos(tarr, day + pd.Timedelta(hours=L0H), day + pd.Timedelta(hours=L1H))
        k0, k1 = window_pos(tarr, day + pd.Timedelta(hours=K0H), day + pd.Timedelta(hours=K1H))
        if (a1 - a0) < 4 or (l1 - l0) < 6 or (k1 - k0) < 2 or not np.isfinite(atr[l1 - 1]):
            continue
        A = atr[l1 - 1]
        asia_lo = lo[a0:a1].min()
        p_lo = pdl.get(d, np.nan)
        iL = l0 + int(np.argmin(lo[l0:l1])); L = lo[iL]
        breached = ((np.isfinite(asia_lo) and L < asia_lo) or (np.isfinite(p_lo) and L < p_lo))
        if not breached:
            continue
        sh = last_fractal_high(hi, a0, iL)
        if sh is None:
            continue
        lvl = hi[sh]
        jm = None
        for j in range(iL + 1, l1):
            if hi[j] > lvl:
                jm = j; break
        if jm is None:
            continue
        end = jm + 1
        if end <= iL + 1:
            continue
        H = hi[iL:end].max()
        if H - L < 0.25 * A:
            continue
        if (lo[end:l1] <= L).any():
            continue
        cands = [v - L for v in (asia_lo, p_lo) if np.isfinite(v) and v > L]
        depth_atr = (min(cands) / A) if cands else np.nan
        out[d] = dict(L=L, H=H, atr=A, iL=iL, jm=jm, k0=k0, k1=k1,
                       bars_hunt_to_mss=jm - iL, depth_atr=depth_atr)
    return out


def selfcheck_tieback(df, tarr, dates, name):
    """long_diag() が canonical_setups() のロング側と (L,H) 完全一致することを検査する。"""
    S0 = canonical_setups(df, tarr, dates, shift=0)
    diag = long_diag(df, tarr, dates)
    setup_dates = {rec["date"] for rec in S0 if rec["long"] is not None}
    diag_dates = set(diag.keys())
    missing_in_diag = setup_dates - diag_dates
    extra_in_diag = diag_dates - setup_dates
    n_common = len(setup_dates & diag_dates)
    max_dL = max_dH = 0.0
    for rec in S0:
        if rec["long"] is None:
            continue
        d = rec["date"]
        if d in diag:
            max_dL = max(max_dL, abs(rec["long"]["L"] - diag[d]["L"]))
            max_dH = max(max_dH, abs(rec["long"]["H"] - diag[d]["H"]))
    print(f"  [{name}] tie-back: setups={len(setup_dates)} diag={len(diag_dates)} "
          f"common={n_common} missing_in_diag={len(missing_in_diag)} extra_in_diag={len(extra_in_diag)} "
          f"maxΔL={max_dL:.8f} maxΔH={max_dH:.8f}")
    ok = (len(missing_in_diag) == 0 and len(extra_in_diag) == 0 and max_dL < 1e-9 and max_dH < 1e-9)
    print(f"  [{name}] tie-back {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
def mfe_mae_scan(df, setups, diag):
    """基準点=KZ開始バー始値、リスク単位=基準点-(L-BUF*atr)、前進窓500本でMFE/MAEを測る。
    setups=canonical_setups()のロング側レコード（採否はここで一切いじらない）、
    diag=long_diag()の出力（bars_hunt_to_mss・depth_atrを引くためだけに使う）。"""
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    n = len(o)
    out = []
    for rec in setups:
        s = rec["long"]
        if s is None:
            continue
        d = rec["date"]
        k0, k1 = s["kz"]
        if k0 >= n:
            continue
        L, atr = s["L"], s["atr"]
        basis = o[k0]
        stop = L - BUF * atr
        risk = basis - stop
        if risk <= 0:
            continue
        end = min(k0 + FWD_CAP, n)
        mfe = (h[k0:end].max() - basis) / risk
        mae = (l[k0:end].min() - basis) / risk
        dg = diag.get(d)
        out.append(dict(date=d, year=pd.Timestamp(d).year, mfe=mfe, mae=mae, risk=risk,
                         bars_hunt_to_mss=(dg["bars_hunt_to_mss"] if dg else np.nan),
                         depth_atr=(dg["depth_atr"] if dg else np.nan)))
    return out


def valid_kz_days(df, tarr, dates):
    """狩り/MSS条件を無視し、窓の有効性だけを要求した「任意の日」のリスト（ランダム対照用）。"""
    atr = df["atr14"].values
    A0H, A1H = ASIA_HOURS
    L0H, L1H = LONDON_HOURS
    K0H, K1H = KZ_HOURS
    n = len(df)
    out = []
    for d in dates:
        day = pd.Timestamp(d)
        a0, a1 = window_pos(tarr, day - pd.Timedelta(days=1) + pd.Timedelta(hours=A0H),
                             day + pd.Timedelta(hours=A1H))
        l0, l1 = window_pos(tarr, day + pd.Timedelta(hours=L0H), day + pd.Timedelta(hours=L1H))
        k0, k1 = window_pos(tarr, day + pd.Timedelta(hours=K0H), day + pd.Timedelta(hours=K1H))
        if (a1 - a0) < 4 or (l1 - l0) < 6 or (k1 - k0) < 2 or not np.isfinite(atr[l1 - 1]):
            continue
        if k0 >= n or not np.isfinite(atr[k0]) or atr[k0] <= 0:
            continue
        out.append(dict(date=d, year=day.year, k0=k0, atr_k0=atr[k0]))
    return out


def random_mfe_by_era(df, tarr, dates, seed):
    """era毎にランダム300日(不足時は全数)のKZ開始バー始値基準MFE(R)を測る（診断: 検出器バイアス対照）。
    リスク単位はその日のATR14(係数1.0)で近似 — セットアップが無い日にはストップ幅の定義が無いため。"""
    vdays = valid_kz_days(df, tarr, dates)
    o, h = df["open"].values, df["high"].values
    n = len(o)
    by_era = {}
    for lo, hi in ERAS:
        lab = era_label(lo, hi)
        pool = [v for v in vdays if lo <= v["year"] <= hi]
        rng = np.random.default_rng(seed + hash((lo, hi)) % 100000)
        if len(pool) == 0:
            by_era[lab] = dict(n=0, mfe_med=np.nan)
            continue
        take = min(N_RAND, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        mfes = []
        for i in idx:
            v = pool[i]
            k0 = v["k0"]
            end = min(k0 + FWD_CAP, n)
            basis = o[k0]
            mfe = (h[k0:end].max() - basis) / v["atr_k0"]
            mfes.append(mfe)
        mfes = np.array(mfes)
        by_era[lab] = dict(n=take, mfe_med=float(np.median(mfes)))
    return by_era


# ---------------------------------------------------------------------------
def summarize(rows_by_era):
    out = {}
    for lab, rows in rows_by_era.items():
        if not rows:
            out[lab] = None
            continue
        mfe = np.array([r["mfe"] for r in rows])
        mae = np.array([r["mae"] for r in rows])
        bars = np.array([r["bars_hunt_to_mss"] for r in rows if np.isfinite(r["bars_hunt_to_mss"])])
        depth = np.array([r["depth_atr"] for r in rows if np.isfinite(r["depth_atr"])])
        out[lab] = dict(
            n=len(rows),
            mfe_med=float(np.median(mfe)), mfe_mean=float(mfe.mean()), mfe_std=float(mfe.std(ddof=1)) if len(mfe) > 1 else 0.0,
            mfe_q25=float(np.percentile(mfe, 25)), mfe_q75=float(np.percentile(mfe, 75)),
            p1=100 * float((mfe >= 1.0).mean()), p2=100 * float((mfe >= 2.0).mean()), p275=100 * float((mfe >= 2.75).mean()),
            mae_med=float(np.median(mae)),
            bars_med=float(np.median(bars)) if len(bars) else np.nan,
            depth_med=float(np.median(depth)) if len(depth) else np.nan,
        )
    return out


def run_pair(name, smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        df, n_nat = load_ny(SYMS[name], cut2000=(name in CUT2000))
    df, tarr, dates = prep(df)
    if smoke:
        dates = np.array([d for d in dates if pd.Timestamp(d) >= pd.Timestamp("2023-01-01")])
    ok = selfcheck_tieback(df, tarr, dates, name)
    S0 = canonical_setups(df, tarr, dates, shift=0)
    diag = long_diag(df, tarr, dates)
    scan = mfe_mae_scan(df, S0, diag)
    rows_by_era = {era_label(lo, hi): [] for lo, hi in ERAS}
    for r in scan:
        lab = era_of_year(r["year"])
        if lab is not None:
            rows_by_era[lab].append(r)
    summ = summarize(rows_by_era)
    rand = random_mfe_by_era(df, tarr, dates, seed=SEED_BASE + hash(name) % 100000)
    return ok, summ, rand


def print_table(name, summ, rand):
    print(f"\n=== {name.upper()} ===")
    hdr = (f"{'era':8s} {'n':>5s} {'MFEmed':>7s} {'MFEmean':>8s} {'MFEsd':>7s} "
           f"{'q25':>6s} {'q75':>6s} {'P>=1R':>6s} {'P>=2R':>6s} {'P>=2.75R':>8s} "
           f"{'MAEmed':>7s} {'bars_med':>8s} {'depth_med':>9s} {'rand_MFEmed':>11s} {'rand_n':>6s}")
    print(hdr)
    for lo, hi in ERAS:
        lab = era_label(lo, hi)
        s = summ.get(lab)
        r = rand.get(lab, {})
        if s is None:
            print(f"{lab:8s} {'n/a':>5s}")
            continue
        print(f"{lab:8s} {s['n']:5d} {s['mfe_med']:7.3f} {s['mfe_mean']:8.3f} {s['mfe_std']:7.3f} "
              f"{s['mfe_q25']:6.3f} {s['mfe_q75']:6.3f} {s['p1']:6.1f} {s['p2']:6.1f} {s['p275']:8.1f} "
              f"{s['mae_med']:7.3f} {s['bars_med']:8.1f} {s['depth_med']:9.3f} "
              f"{r.get('mfe_med', float('nan')):11.3f} {r.get('n', 0):6d}")


def flagship_selfcheck():
    """フェーズ1 自己検査アンカーの再現（既存ライブラリの整合を確認）。"""
    print("フェーズ1 自己検査アンカー再現: EURUSD ロング旗艦 n=1148・PF1.17・totR/DD2.56 を確認")
    df, tarr, dates, span = load_prepped("eurusd")
    S0 = canonical_setups(df, tarr, dates, 0)
    from ict_exec import MODEL
    sp, cost = MODEL["eurusd"]
    r = stats(walk(df, S0, F_CANON, RR_CANON, BUF, sp, cost, "long"), span)
    print(f"  n={r['n']} win%={r['win']:.1f} net={r['net']:+.3f} PF={r['pf']:.2f} totR/DD={r['rdd']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2023年以降のみで全ペアを軽く流す")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--selfcheck", action="store_true", help="旗艦アンカー再現とtie-backのみ実行")
    args = ap.parse_args()
    pairs = args.pairs.split(",")

    flagship_selfcheck()

    if args.selfcheck:
        for name in pairs:
            with contextlib.redirect_stderr(io.StringIO()):
                df, n_nat = load_ny(SYMS[name], cut2000=(name in CUT2000))
            df, tarr, dates = prep(df)
            selfcheck_tieback(df, tarr, dates, name)
        sys.exit(0)

    all_ok = True
    for name in pairs:
        ok, summ, rand = run_pair(name, smoke=args.smoke)
        all_ok = all_ok and ok
        print_table(name, summ, rand)
    print(f"\ntie-back全体: {'PASS' if all_ok else 'FAIL（要確認）'}")
