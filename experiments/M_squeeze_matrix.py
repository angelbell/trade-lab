"""続き: 「溜め(ブレイク前圧縮)は伸びしろを予言するか」を銘柄x TF のマトリクスに拡張。

前回(experiments/M_squeeze_screen.py)は btc15m_L / gold15m の2レッグのみで、押し目指値
(pullback_frac 0.3/0.25)が入っていたためアンカー b が「約定(フィル)足」であり、真の
ブレイク確定足からずれていた。今回はそれを排除し、**pullback_frac=0 の素の Pattern-B
ブレイク**（ZigZag 2xATR・確定足ブレイク・成行・trend_ema=80 の構造フィルタのみ、regime
ゲート無し）に統一する。BASE (radar_gate_race.BASE) はもともと pullback_frac キーを
持たない(=getattrのデフォルト0.0)ので、CFG = BASE のまま rr=100/fwd=500/cost=0 を上書き
するだけで「素のブレイク」になる。この場合 e_bar == e_i（ブレイク確定足そのもの）なので、
アンカー b の先読みは無い。

流用(車輪の再発明禁止): breakout_wave.run/resample, radar_gate_race.BASE, arb_common.Boot,
M_squeeze_screen.py の comp_scores/mfe_stop_only/layer_stats/random_subset_null/block_boot_diff
をそのまま import して使う(コピペで書き直さない)。

データの罠(CLAUDE.md 準拠):
  - gold: 15m/5mは m5.csv を .loc["2018-09-14":] (m15直読みは疎データ罠、正典 book_deployed_spec.py
    と同じ)。h1以上は h1.csv を .loc["2018-01-01":] (GOLD_H1_START、疎データ罠)。
  - btcusd: h4/d1 の専用CSVが無いので h1 から resample。**h1/m15 とも 2018-03 以前は
    22本/月=日足がラベル違いで紛れ込む罠**(今回 h1 の月次本数を実測して確認)。安全側で
    全TF共通 start=2018-10-01(既存 btc15m_L 系スクリプトの慣例と同じ閾値)。
  - FX majors(eurusd/gbpusd/audusd/nzdusd/usdcad) と usdjpy: m15/h1/h4/d1 とも専用CSVが
    2000年から存在するので resample せず直読み(m5は今回スコープ外=usdjpy m5の日足紛れ込み
    罠を踏まない)。
  - ファイルの存在が真実。無い組み合わせは実行前に os.path.exists でスキップし、表に「-」。

溜めスコア・MFE・null帯・ブロックブートストラップの定義は前回のまま。primary (N,M)=(8,16)。
ブロックブートストラップは今回 k=1 と k=12 のみ(仕様カード指定)。

Run:
  .venv/bin/python experiments/M_squeeze_matrix.py --smoke   # 直近データで先に通す
  .venv/bin/python experiments/M_squeeze_matrix.py           # フル(全銘柄xTF)
"""
import sys, io, os, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv, GOLD_H1_START
from breakout_wave import run, resample
from radar_gate_race import BASE
from M_squeeze_screen import (mfe_stop_only, comp_scores, layer_stats,
                               random_subset_null, block_boot_diff, PRIMARY)

ROOT = "/home/angelbell/dev/auto-trade"
BTC_START = "2018-10-01"


# ---------------------------------------------------------------- loaders (data-trap aware)
def load_gold(tf, smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        if tf in ("5min", "15min"):
            m5 = load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":]
            d = m5 if tf == "5min" else resample(m5, "15min")
        else:
            h1 = load_mt5_csv(f"{ROOT}/data/vantage_xauusd_h1.csv").loc[GOLD_H1_START:]
            d = h1 if tf == "1h" else resample(h1, tf)
    return d.loc["2025-01-01":] if smoke else d


def load_btc(tf, smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        if tf == "5min":
            d = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m5.csv").loc[BTC_START:]
        elif tf == "15min":
            d = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc[BTC_START:]
        else:
            h1 = load_mt5_csv(f"{ROOT}/data/vantage_btcusd_h1.csv").loc[BTC_START:]
            d = h1 if tf == "1h" else resample(h1, tf)
    return d.loc["2025-01-01":] if smoke else d


def load_fx(sym, tf, smoke=False):
    suf = {"15min": "m15", "1h": "h1", "4h": "h4", "1d": "d1"}[tf]
    path = f"{ROOT}/data/vantage_{sym}_{suf}.csv"
    if not os.path.exists(path):
        return None
    with contextlib.redirect_stderr(io.StringIO()):
        d = load_mt5_csv(path)
    return d.loc["2025-01-01":] if smoke else d


INSTR_TFS = {
    "gold":   ["5min", "15min", "1h", "4h", "1d"],
    "btcusd": ["5min", "15min", "1h", "4h", "1d"],
    "eurusd": ["15min", "1h", "4h", "1d"],
    "gbpusd": ["15min", "1h", "4h", "1d"],
    "audusd": ["15min", "1h", "4h", "1d"],
    "nzdusd": ["15min", "1h", "4h", "1d"],
    "usdcad": ["15min", "1h", "4h", "1d"],
    "usdjpy": ["15min", "1h", "4h", "1d"],
}


def load_cell(sym, tf, smoke=False):
    if sym == "gold":
        return load_gold(tf, smoke)
    if sym == "btcusd":
        return load_btc(tf, smoke)
    return load_fx(sym, tf, smoke)


# ---------------------------------------------------------------- run the bare Pattern-B breakout
def run_bare_breakout(d, rr=100.0, fwd=500, cost=0.0):
    """CFG = BASE のまま(pullback_frac無し=0, regime ゲート無し, trend_ema=80 の構造フィルタのみ)。
    rr/fwd/cost だけ上書き = 素の巡行幅測定用。pullback_frac=0 なので e_bar==e_i
    (ブレイク確定足そのもの) -- アンカーの先読みなし。"""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        t = run(d, SimpleNamespace(**{**BASE, "rr": rr, "fwd": fwd, "cost": cost}))
    if t is None or len(t) == 0:
        return None
    return t.reset_index(drop=True)


# ---------------------------------------------------------------- per-cell analysis
def analyze_cell(sym, tf, d):
    t = run_bare_breakout(d)
    if t is None or len(t) < 20:
        return dict(sym=sym, tf=tf, n=0 if t is None else len(t), skip=True)
    MFE, idx = mfe_stop_only(d, t)
    times = t["time"].values
    comps = comp_scores(d, idx, [PRIMARY])
    primary = comps[PRIMARY]
    valid = ~np.isnan(primary)
    if valid.sum() < 20:
        return dict(sym=sym, tf=tf, n=int(valid.sum()), skip=True)
    comp_flag = valid & (primary < 1.0)
    n = int(valid.sum())
    n_comp = int(comp_flag.sum())
    if n_comp < 10 or (n - n_comp) < 10:
        return dict(sym=sym, tf=tf, n=n, n_comp=n_comp, skip=True)

    m_all = valid
    s_all = layer_stats(MFE[m_all])
    s_comp = layer_stats(MFE[comp_flag])
    s_not = layer_stats(MFE[valid & ~comp_flag])

    meds, _ = random_subset_null(MFE[m_all], n_comp, reps=1000)
    pct_med = 100 * np.mean(meds < s_comp["median"])

    boot_p = {}
    for k in (1, 12):
        diffs = block_boot_diff(times[m_all], MFE[m_all], comp_flag[m_all], k, nb=1000)
        if diffs is None:
            boot_p[k] = np.nan
        else:
            ok = ~np.isnan(diffs)
            boot_p[k] = 100 * np.mean(diffs[ok] > 0) if ok.sum() > 0 else np.nan

    delta = s_comp["median"] - s_all["median"]
    b1, b12 = boot_p[1], boot_p[12]
    if delta > 0 and not np.isnan(b1) and not np.isnan(b12) and b1 >= 55 and b12 >= 55:
        sign = "順"
    elif delta < 0 and not np.isnan(b1) and not np.isnan(b12) and b1 <= 45 and b12 <= 45:
        sign = "逆"
    else:
        sign = "無"
    outside = (pct_med >= 97.5) or (pct_med <= 2.5)

    return dict(sym=sym, tf=tf, skip=False, n=n, n_comp=n_comp,
                med_all=s_all["median"], med_comp=s_comp["median"], delta=delta,
                p_lt1_all=s_all["p_lt1"], p_lt1_comp=s_comp["p_lt1"],
                pct_med=pct_med, outside=outside, b1=b1, b12=b12, sign=sign)


def fmt_row(r):
    if r.get("skip", True):
        return (f"  {r['sym']:<8}{r['tf']:<7}{'-':>6}{'-':>6}{'-':>9}{'-':>9}{'-':>8}"
                f"{'-':>10}{'-':>10}{'-':>9}{'-':>7}{'-':>7}{'-':>5}   n={r.get('n',0)} 不足でskip")
    return (f"  {r['sym']:<8}{r['tf']:<7}{r['n']:>6}{r['n_comp']:>6}{r['med_all']:>9.2f}"
            f"{r['med_comp']:>9.2f}{r['delta']:>+8.2f}{r['p_lt1_comp']:>9.1f}%{r['p_lt1_all']:>8.1f}%"
            f"{r['pct_med']:>8.0f}%ile{r['b1']:>6.0f}%{r['b12']:>6.0f}%{r['sign']:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    print(f"{'銘柄':<8}{'TF':<7}{'n':>6}{'n圧縮':>6}{'MFE中央値base':>9}{'MFE中央値圧縮':>9}"
          f"{'Δ中央値':>8}{'P(<1R)圧縮':>9}{'P(<1R)base':>8}{'null%ile':>8}{'boot k1':>6}{'boot k12':>6}{'符号':>5}")
    print("-" * 118)

    rows = []
    for sym, tfs in INSTR_TFS.items():
        for tf in tfs:
            d = load_cell(sym, tf, smoke=args.smoke)
            if d is None or len(d) < 500:
                r = dict(sym=sym, tf=tf, skip=True, n=0 if d is None else len(d))
            else:
                r = analyze_cell(sym, tf, d)
            rows.append(r)
            print(fmt_row(r))

    valid_rows = [r for r in rows if not r.get("skip", True)]
    print(f"\n有効セル数: {len(valid_rows)} / {len(rows)}")

    print("\n符号の内訳:")
    from collections import Counter
    cnt = Counter(r["sign"] for r in valid_rows)
    for k, v in cnt.items():
        print(f"  {k}: {v}")

    print("\n銘柄別の符号（見えるパターンを確認）:")
    by_sym = {}
    for r in valid_rows:
        by_sym.setdefault(r["sym"], []).append(r["sign"])
    for sym, signs in by_sym.items():
        print(f"  {sym:<8}: {' '.join(signs)}")

    print("\nnull帯の外(97.5%ile超 or 2.5%ile未満)に出たセル:")
    out_rows = [r for r in valid_rows if r["outside"]]
    if out_rows:
        for r in out_rows:
            print(f"  {r['sym']:<8}{r['tf']:<7}null%ile={r['pct_med']:.0f}  Δ中央値={r['delta']:+.2f}"
                  f"  boot(k1/k12)={r['b1']:.0f}%/{r['b12']:.0f}%")
    else:
        print("  無し(全セル帯の内)")


if __name__ == "__main__":
    main()
