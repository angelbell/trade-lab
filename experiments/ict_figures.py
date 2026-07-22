"""follow-up 3: 「チャートの雰囲気」の可視化（裁量トレーダー向け・数字と目視をつなぐ）。

matplotlib は利用可能（mplfinance は無し -> ローソク足は矩形/線分で手描き、車輪の再発明は
最小限に留める:実体=Rectangle・ヒゲ=Line2D のみ）。

A. セットアップ・スナップショット: EURUSD 15分、C3(旗艦,狩り+MSS+FVG/入口FVG-CE/損切りL-0.1ATR/
   目標PDH-5pip)の実トレードを 2018-20 era・2024-26 era から各3件（シード固定・勝敗混在・
   チェリーピック禁止=era内から等確率ランダム抽出）。セットアップ前30本〜約定後60本のローソク足に、
   狩られた安値(L)・MSSで抜いたフラクタル高値・FVG帯・CE指値(entry)・損切り(stop)・PDH目標(tgt)を
   重ねる。ファイル: experiments/fig_ict_setups_<era>_<n>.png

B. 正規化した経路の中央値: 同じ C3 母集団の全トレードについて、約定を基準(x=0)に0-100本の
   "生の価格路"（ポジションの結果でなく、その後の値動きそのもの）を (price-entry)/risk で
   正規化し、年代別(2018-20/2021-23/2024-26)に中央値の折れ線+25/75分位帯を重ねる。
   0R・-1R(損切り)・目標R(中央値)の水平線つき。ファイル: experiments/fig_ict_path_by_era.png

C. MFE/MAE散布図: ict_alpha_decay.scan_mfe_mae(C3,E1)の結果を年代別に色分けして散布。
   ファイル: experiments/fig_ict_mfe_mae.png

流用（車輪の再発明禁止）: ict_population.{canonical_setups, load_prepped, last_fractal_high}、
ict_exec.{walk, ASIA_HOURS, KZ_HOURS, window_pos, F_CANON, RR_CANON, BUF}、
ict_extliq_target.{make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA}、ict_dxy_smt.cost_tiers、
ict_fvg_anchor.fvg_anchor_fn、ict_alpha_decay.scan_mfe_mae。

日本語ラベルはフォント無しで文字化けする環境が多いため、図中ラベルは英語で統一する。

Run: .venv/bin/python experiments/ict_figures.py [--smoke]
"""
import sys, io, contextlib, argparse
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from ict_exec import (ASIA_HOURS, KZ_HOURS, window_pos, walk, F_CANON, RR_CANON, BUF)
from ict_population import canonical_setups, load_prepped, last_fractal_high
from ict_extliq_target import make_ext_tgt_fn, EURUSD_LIM_FN, EURUSD_MA
from ict_dxy_smt import cost_tiers
from ict_fvg_anchor import fvg_anchor_fn
from ict_alpha_decay import scan_mfe_mae

OUT = "/home/angelbell/dev/auto-trade/experiments"
SEED = 20260716


def build_flagship(df, tarr, dates):
    S3 = canonical_setups(df, tarr, dates, 0, use_fvg=True, fvg_min_atr=EURUSD_MA, use_liq=True, liq_ns=(20, 40))
    sp, cost = cost_tiers("eurusd")["realistic"]
    tgt_fn = make_ext_tgt_fn("pdh", 5, "eurusd", "long")
    tlog = []
    walk(df, S3, F_CANON, RR_CANON, BUF, sp, cost, "long", lim_fn=EURUSD_LIM_FN, tgt_fn=tgt_fn, trade_log=tlog)
    S3_by_date = {rec["date"]: rec["long"] for rec in S3 if rec["long"] is not None}
    return S3, S3_by_date, tlog


def find_fp(df, fill_dt):
    bd = df["broker_dt"].values
    idx = np.searchsorted(bd, np.datetime64(fill_dt))
    if idx < len(bd) and bd[idx] == np.datetime64(fill_dt):
        return idx
    return None


# ============================================================== A. setup snapshots
def plot_snapshot(df, tarr, s, tr, fp, era_label, k):
    h, l, o, c = df["high"].values, df["low"].values, df["open"].values, df["close"].values
    lo_i, hi_i = max(0, fp - 30), min(len(df), fp + 61)
    xs = np.arange(lo_i, hi_i)

    # recompute the broken fractal high level (last_fractal_high over the Asia window), matching
    # build()'s own logic exactly (re-derived only for the handful of plotted trades).
    day = pd.Timestamp(tr["date"])
    a0, a1 = window_pos(tarr, day - pd.Timedelta(days=1) + pd.Timedelta(hours=ASIA_HOURS[0]),
                        day + pd.Timedelta(hours=ASIA_HOURS[1]))
    iL_guess = np.argmin(l[a1:fp + 1]) + a1 if fp >= a1 else None
    sh = last_fractal_high(h, a0, iL_guess) if iL_guess is not None else None
    mss_level = h[sh] if sh is not None else s["H"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for x in xs:
        color = "#2a9d5c" if c[x] >= o[x] else "#d1495b"
        ax.plot([x, x], [l[x], h[x]], color=color, lw=0.8, zorder=2)
        ybot, ytop = min(o[x], c[x]), max(o[x], c[x])
        ax.add_patch(Rectangle((x - 0.3, ybot), 0.6, max(ytop - ybot, 1e-6), color=color, zorder=3))

    ax.axhline(s["L"], color="orange", ls="--", lw=1.2, label=f"swept low L={s['L']:.5f}")
    ax.axhline(mss_level, color="purple", ls="--", lw=1.2, label=f"MSS fractal high={mss_level:.5f}")
    if "fvg_lo" in s and "fvg_hi" in s:
        ax.axhspan(s["fvg_lo"], s["fvg_hi"], color="steelblue", alpha=0.15, label="FVG band")
    ax.axhline(tr["entry"], color="black", lw=1.4, label=f"entry(CE)={tr['entry']:.5f}")
    ax.axhline(tr["stop"], color="red", lw=1.2, ls=":", label=f"stop={tr['stop']:.5f}")
    ax.axhline(tr["tgt"], color="green", lw=1.2, ls=":", label=f"target(PDH-5pip)={tr['tgt']:.5f}")
    ax.axvline(fp, color="gray", lw=0.8, alpha=0.6)

    outcome = "WIN" if tr["R"] > 0 else ("LOSS" if tr["R"] < 0 else "FLAT")
    ax.set_title(f"EURUSD 15m  {day.date()}  {outcome} (R={tr['R']:+.2f}, reason={tr['reason']})  [{era_label}]")
    ax.set_xlabel("bar index"); ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = f"{OUT}/fig_ict_setups_{era_label}_{k}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def stage_a(df, tarr, dates, S3_by_date, tlog, smoke=False):
    rng = np.random.default_rng(SEED)
    eras = [(2018, 2020, "2018-20"), (2024, 2026, "2024-26")]
    paths = []
    for a, b, elabel in eras:
        pool = [t for t in tlog if a <= pd.Timestamp(t["date"]).year <= b]
        if len(pool) == 0:
            print(f"  [A] {elabel}: 該当トレード無し、skip"); continue
        take = min(3, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        for k, i in enumerate(idx, start=1):
            tr = pool[i]
            fp = find_fp(df, tr["fill_dt"])
            if fp is None:
                print(f"  [A] {elabel} #{k}: fill_dt {tr['fill_dt']} 見つからず skip"); continue
            s = S3_by_date[tr["date"]]
            p = plot_snapshot(df, tarr, s, tr, fp, elabel, k)
            paths.append(p)
            print(f"  [A] saved {p}")
    return paths


# ============================================================== B. normalized path by era
def stage_b(df, tlog):
    c = df["close"].values
    n = len(c)
    eras = [(2018, 2020, "2018-20"), (2021, 2023, "2021-23"), (2024, 2026, "2024-26")]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"2018-20": "#1b7837", "2021-23": "#2166ac", "2024-26": "#b2182b"}
    tgt_r_all = []
    for a, b, elabel in eras:
        paths = []
        for tr in tlog:
            if not (a <= pd.Timestamp(tr["date"]).year <= b):
                continue
            fp = find_fp(df, tr["fill_dt"])
            if fp is None:
                continue
            risk = tr["entry"] - tr["stop"]
            if risk <= 0:
                continue
            end = min(fp + 101, n)
            seg = (c[fp:end] - tr["entry"]) / risk
            if len(seg) < 101:
                seg = np.concatenate([seg, np.full(101 - len(seg), np.nan)])
            paths.append(seg)
            tgt_r_all.append(tr["r_rr"])
        if not paths:
            continue
        arr = np.vstack(paths)
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        x = np.arange(101)
        ax.plot(x, med, color=colors[elabel], lw=2, label=f"{elabel} (n={len(paths)})")
        ax.fill_between(x, q25, q75, color=colors[elabel], alpha=0.15)

    ax.axhline(0, color="black", lw=1)
    ax.axhline(-1, color="red", ls="--", lw=1, label="stop (-1R)")
    if tgt_r_all:
        ax.axhline(np.median(tgt_r_all), color="green", ls="--", lw=1,
                  label=f"target (median R={np.median(tgt_r_all):.2f})")
    ax.set_xlabel("bars since fill (15m bars)")
    ax.set_ylabel("price path, R units ((price-entry)/risk)")
    ax.set_title("EURUSD 15m ICT flagship: normalized price path after fill, by era\n"
                 "(raw price path, not position P&L -- ignores own stop/target)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = f"{OUT}/fig_ict_path_by_era.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [B] saved {path}")
    return path


# ============================================================== C. MFE/MAE scatter
def stage_c(df, S3):
    lim_fn = fvg_anchor_fn("mid", "long")
    sp, cost = cost_tiers("eurusd")["realistic"]
    scans = scan_mfe_mae(df, S3, "long", lim_fn, sp, cost)
    eras = [(2018, 2020, "2018-20"), (2021, 2023, "2021-23"), (2024, 2026, "2024-26")]
    colors = {"2018-20": "#1b7837", "2021-23": "#2166ac", "2024-26": "#b2182b"}
    fig, ax = plt.subplots(figsize=(8, 7))
    for a, b, elabel in eras:
        sub = [x for x in scans if a <= pd.Timestamp(x["date"]).year <= b]
        if not sub:
            continue
        mae = np.array([x["mae"] for x in sub])
        mfe = np.array([x["mfe"] for x in sub])
        ax.scatter(mae, mfe, s=18, alpha=0.6, color=colors[elabel], label=f"{elabel} (n={len(sub)})")
    ax.set_xlabel("MAE (R, capped at 1.0 when stopped)")
    ax.set_ylabel("MFE (R, stop-only excursion)")
    ax.set_title("EURUSD 15m ICT flagship (C3xE1): MFE vs MAE by era")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    path = f"{OUT}/fig_ict_mfe_mae.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [C] saved {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        df, tarr, dates, span = load_prepped("eurusd")

    S3, S3_by_date, tlog = build_flagship(df, tarr, dates)
    print(f"flagship population n_setups={len(S3_by_date)}  n_filled={len(tlog)}")

    print("\nA. setup snapshots")
    paths_a = stage_a(df, tarr, dates, S3_by_date, tlog)

    print("\nB. normalized path by era")
    path_b = stage_b(df, tlog)

    print("\nC. MFE/MAE scatter")
    path_c = stage_c(df, S3)

    print("\n生成PNG一覧:")
    for p in paths_a + [path_b, path_c]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
