"""仕様カード: btc15m_L と gold15m で「ブレイク前の溜め(ボラ圧縮)」が「ブレイク後の伸びしろ」を
予言するかを、素の巡行幅(目標なし・損切りのみ)だけで測る。RR/目標のスイープはしない。

流用（車輪の再発明禁止）:
  - breakout_wave.run / resample                  … エントリー生成・OHLC整形
  - radar_gate_race.BASE                          … 両レッグの共通 config 土台
  - experiments/A_daily_regime.py の MFE ループ定義 … 素の巡行幅（stop-only, fwd=500, rr=100）
  - experiments/arb_common.Boot                    … 巡回ブロック・ブートストラップの層(月ブロック抽選)

【config の出所・重要な乖離】
  仕様カードは gold15m のデータを data/vantage_xauusd_m15.csv と指定していたが、実際に
  book_deployed_spec.py はじめ現行スクリプト30本以上が例外なく
      data/vantage_xauusd_m5.csv を .loc["2018-09-14":] してから 15min にresample
  している（m15 CSV を直接使っている既存スクリプトは見つからなかった）。CLAUDE.md の
  「gold h1/m15 は2017以前疎データ」の注記とも整合的（m5→resampleがその回避策）。
  ここでは正典（book_deployed_spec.py）に合わせて m5-resample 版を使う。m15 CSV 直読みとの
  差は tie-back セクションで n を報告して透明化する。

  btc15m_L は README の「単体で再現」コマンド（data/vantage_btcusd_m15.csv 直読み, start 2018-10-01,
  gate_kama=14/240min, pullback_frac=0.30, fill_win=200）と仕様カードが一致。

先読み: 溜めスコア comp は「約定(フィル)足 b」より前の確定TRのみで計算（b, b以降は使わない）。
  b は run() が返す "time"（フィル足）であり、pullback_frac>0 のため、レベルを実際に確定終値で
  抜けた「ブレイク足」そのものより後ろにずれる（仕様カードで事前承認済みの簡易化）。つまり
  comp はブレイク直前の溜めというより「フィルまでの直近安静度」を測っている点に注意
  （全trade型に一様に効くのでレイヤー間比較のバイアスにはならないが、意味の解釈はそれを踏まえること）。

Run:
  .venv/bin/python experiments/M_squeeze_screen.py --smoke   # 直近データで先に通す
  .venv/bin/python experiments/M_squeeze_screen.py           # フル
"""
import sys, io, contextlib, warnings, argparse
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from arb_common import Boot

ROOT = "/home/angelbell/dev/auto-trade"
NM_GRID = [(4, 8), (6, 12), (8, 16), (10, 20)]
PRIMARY = (8, 16)


# ---------------------------------------------------------------- data / entries
def load_btc15m_L(smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        d = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    if smoke:
        d = d.loc["2024-01-01":]
    cfg = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": 0.3, "fill_win": 200}
    return d, cfg


def load_gold15m(smoke=False):
    with contextlib.redirect_stderr(io.StringIO()):
        d = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
    if smoke:
        d = d.loc["2024-01-01":]
    cfg = {**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0, "pullback_frac": 0.25,
           "fill_win": 200}
    return d, cfg


def run_trades(d, cfg, rr, fwd, cost=0.0):
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d, SimpleNamespace(**{**cfg, "rr": rr, "fwd": fwd, "cost": cost}))
    if t is None or len(t) == 0:
        return None
    return t.reset_index(drop=True)


# ---------------------------------------------------------------- MFE (stop-only), same def as A_daily_regime.py
def mfe_stop_only(d, t):
    idx = d.index.get_indexer(t["time"])
    e = t["e_px"].values
    risk = t["risk"].values
    stop = e - risk
    h, l = d["high"].values, d["low"].values
    MFE = np.full(len(e), np.nan)
    for i in range(len(e)):
        j0 = idx[i]
        mx = -np.inf
        for j in range(j0 + 1, min(j0 + 501, len(h))):
            mx = max(mx, h[j])
            if l[j] <= stop[i]:
                break
        MFE[i] = (mx - e[i]) / risk[i]
    return MFE, idx


# ---------------------------------------------------------------- 溜めスコア(先読み無し, b以前の確定足のみ)
def comp_scores(d, idx, grid):
    tr = ta.true_range(d["high"], d["low"], d["close"]).values
    out = {}
    for (N, M) in grid:
        s = np.full(len(idx), np.nan)
        for i, b in enumerate(idx):
            lo2 = b - N - M
            if lo2 < 0:
                continue
            w1 = tr[b - N:b]
            w2 = tr[lo2:b - N]
            m2 = np.nanmean(w2)
            if len(w1) < N or len(w2) < M or not (m2 > 0):
                continue
            s[i] = np.nanmean(w1) / m2
        out[(N, M)] = s
    return out


def range_tightness(d, idx, N=14):
    """交差確認: 直近N本の値幅(max high - min low)/確定ATR14。"""
    atrv = ta.atr(d["high"], d["low"], d["close"], 14).values
    h, l = d["high"].values, d["low"].values
    out = np.full(len(idx), np.nan)
    for i, b in enumerate(idx):
        if b - N < 0 or b - 1 < 0 or not (atrv[b - 1] > 0):
            continue
        out[i] = (h[b - N:b].max() - l[b - N:b].min()) / atrv[b - 1]
    return out


# ---------------------------------------------------------------- reporting helpers
def layer_stats(mfe):
    x = mfe[~np.isnan(mfe)]
    n = len(x)
    if n == 0:
        return dict(n=0)
    return dict(n=n, median=np.median(x), mean=x.mean(), std=x.std(ddof=1) if n > 1 else 0.0,
                p1=np.mean(x >= 1) * 100, p2=np.mean(x >= 2) * 100, p3=np.mean(x >= 3) * 100,
                p45=np.mean(x >= 4.5) * 100, p_lt1=np.mean(x < 1) * 100)


def print_layer_row(label, s):
    if s["n"] == 0:
        print(f"    {label:<12}{'n=0':>6}")
        return
    print(f"    {label:<12}{s['n']:>5}{s['median']:>10.2f}{s['mean']:>9.2f}{s['std']:>9.2f}"
          f"{s['p_lt1']:>10.1f}%{s['p1']:>8.1f}%{s['p2']:>8.1f}%{s['p3']:>8.1f}%{s['p45']:>8.1f}%")


def year_spread(times):
    yrs = pd.Series(pd.DatetimeIndex(times)).dt.year.value_counts().sort_index()
    return "  ".join(f"{y}:{v}" for y, v in yrs.items())


def random_subset_null(mfe_all, n_comp, reps=1000, seed=20260716):
    rng = np.random.default_rng(seed)
    N = len(mfe_all)
    meds = np.empty(reps)
    p3s = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(N, size=n_comp, replace=False)
        x = mfe_all[pick]
        meds[r] = np.median(x)
        p3s[r] = np.mean(x >= 3) * 100
    return meds, p3s


def block_boot_diff(times, mfe, comp_flag, k, nb=1000, seed=20260716):
    """圧縮あり(comp_flag=True) の MFE中央値 - 全体 の MFE中央値 を、月の巡回ブロック・ブートストラップで
    (arb_common.Boot のレイアウト機構を流用)。1回の抽選で「圧縮あり」「全体」を同じ抽出パスから作るので対比較。"""
    df = pd.DataFrame({"mfe": mfe, "comp": comp_flag}, index=pd.DatetimeIndex(times))
    months = sorted(df.index.to_period("M").unique())
    if len(months) < k + 1:
        return None
    boot = Boot(months, nb=nb, k=k, seed=seed)
    mk = df.index.to_period("M")
    by = {m: df[mk == m] for m in months}
    n = len(df)
    out = np.full(len(boot.layout), np.nan)
    for i, seq in enumerate(boot.layout):
        v = pd.concat([by[months[j]] for j in seq]).iloc[:n]
        vc = v.loc[v["comp"] == True, "mfe"]
        if len(vc) < 5:
            continue
        out[i] = vc.median() - v["mfe"].median()
    return out


# ---------------------------------------------------------------- per-leg pipeline
def analyze_leg(name, d, cfg, tie_rr, tie_expect_n=None, tie_expect_meanR=None):
    print(f"\n{'='*100}\n{name}\n{'='*100}")

    # --- tie-back: canonical RR config, confirm n / meanR against known figures
    t_tie = run_trades(d, cfg, rr=tie_rr, fwd=500, cost=0.0)
    n_tie = 0 if t_tie is None else len(t_tie)
    meanR_tie = np.nan if t_tie is None else t_tie["R"].mean()
    print(f"\n[tie-back] rr={tie_rr}, fwd=500, cost=0  ->  n={n_tie}  meanR={meanR_tie:+.3f}"
          + (f"   (参考: 既知 n={tie_expect_n}"
             + (f", meanR={tie_expect_meanR:+.2f}" if tie_expect_meanR is not None else "") + ")"
             if tie_expect_n is not None else ""))

    # --- main: rr=100 (~目標なし), fwd=500, cost=0 -- 素の巡行幅測定用のエントリー集合
    t = run_trades(d, cfg, rr=100.0, fwd=500, cost=0.0)
    if t is None:
        print("  no entries -- abort"); return None
    MFE, idx = mfe_stop_only(d, t)
    times = t["time"].values
    comps = comp_scores(d, idx, NM_GRID)
    rtight = range_tightness(d, idx, N=14)

    valid = ~np.isnan(comps[PRIMARY])
    print(f"\n1. n 総数={len(MFE)}  (うち comp 計算可={valid.sum()}, 先頭付近でウィンドウ不足のため除外={(~valid).sum()})")

    primary = comps[PRIMARY]
    comp_flag = np.where(valid, primary < 1.0, False)
    n_comp = int((comp_flag & valid).sum())
    n_notcomp = int(valid.sum() - n_comp)
    print(f"   圧縮あり(comp<1, N,M={PRIMARY}) = {n_comp} / {valid.sum()}  "
          f"({100*n_comp/max(valid.sum(),1):.1f}%)")
    print(f"   圧縮あり subset 年別: {year_spread(pd.DatetimeIndex(times)[comp_flag & valid])}")
    print(f"   圧縮なし subset 年別: {year_spread(pd.DatetimeIndex(times)[valid & ~comp_flag])}")

    print(f"\n2. 層別 MFE 分布（comp<1 / comp>=1 / 全体）  primary N,M={PRIMARY}")
    print(f"    {'':<12}{'n':>5}{'中央値':>10}{'平均':>9}{'標準偏差':>9}"
          f"{'P(<1R)':>11}{'P(>=1R)':>9}{'P(>=2R)':>9}{'P(>=3R)':>9}{'P(>=4.5R)':>9}")
    m_all = valid
    m_comp = valid & comp_flag
    m_notcomp = valid & ~comp_flag
    s_all = layer_stats(MFE[m_all]); print_layer_row("全体base", s_all)
    s_comp = layer_stats(MFE[m_comp]); print_layer_row("圧縮あり", s_comp)
    s_not = layer_stats(MFE[m_notcomp]); print_layer_row("圧縮なし", s_not)

    print(f"\n   グリッド頑健性（各 N,M で comp<1 分割を再計算。単一しきい値の一点勝負にしない）")
    print(f"    {'N,M':<10}{'n圧縮':>7}{'MFE中央値(圧縮)':>16}{'MFE中央値(全体)':>16}"
          f"{'P3R(圧縮)':>11}{'P3R(全体)':>11}")
    for nm in NM_GRID:
        c = comps[nm]
        v2 = ~np.isnan(c)
        cf = np.where(v2, c < 1.0, False)
        mc = v2 & cf
        xa = MFE[v2]; xc = MFE[mc]
        if mc.sum() < 5:
            print(f"    {str(nm):<10}{'(n<5)':>7}")
            continue
        print(f"    {str(nm):<10}{int(mc.sum()):>7}{np.median(xc):>16.2f}{np.median(xa):>16.2f}"
              f"{100*np.mean(xc>=3):>10.1f}%{100*np.mean(xa>=3):>10.1f}%")

    print(f"\n3. comp 四分位バケット（primary N,M={PRIMARY}）-- tighter(Q1)ほど伸びるか？プラトーかスパイクか")
    xv = primary[m_all]; mfev = MFE[m_all]
    try:
        q = pd.qcut(xv, 4, labels=["Q1(tightest)", "Q2", "Q3", "Q4(loosest)"], duplicates="drop")
    except ValueError:
        q = None
    print(f"    {'quartile':<16}{'n':>5}{'comp範囲':>18}{'MFE中央値':>12}{'P(>=3R)':>10}")
    if q is not None:
        for lab in q.categories:
            mm = np.asarray(q == lab)
            if mm.sum() == 0:
                continue
            xx = mfev[mm]
            rng_ = f"[{xv[mm].min():.2f},{xv[mm].max():.2f}]"
            print(f"    {str(lab):<16}{mm.sum():>5}{rng_:>18}{np.median(xx):>12.2f}"
                  f"{100*np.mean(xx>=3):>9.1f}%")

    print(f"\n   交差確認: 直近14本レンジ幅/ATR14 と primary comp の順位相関(Spearman)")
    rt = rtight[m_all]
    ok = ~np.isnan(rt)
    if ok.sum() > 10:
        sp = pd.Series(xv[ok]).corr(pd.Series(rt[ok]), method="spearman")
        print(f"    n={ok.sum()}  Spearman(comp, range/ATR)={sp:+.3f}  (正の相関なら両指標は同じ「溜め」を捉えている)")

    print(f"\n4. 同数ランダム部分集合 null（1000回、n={n_comp}）  primary N,M={PRIMARY}")
    if n_comp >= 5:
        meds, p3s = random_subset_null(MFE[m_all], n_comp)
        real_med = s_comp["median"]; real_p3 = s_comp["p3"]
        pct_med = 100 * np.mean(meds < real_med)
        pct_p3 = 100 * np.mean(p3s < real_p3)
        print(f"    null帯 MFE中央値: [{np.percentile(meds,2.5):.2f}, {np.percentile(meds,97.5):.2f}]"
              f"  実測(圧縮あり)={real_med:.2f}  -> {pct_med:.0f}パーセンタイル")
        print(f"    null帯 P(>=3R):  [{np.percentile(p3s,2.5):.1f}%, {np.percentile(p3s,97.5):.1f}%]"
              f"  実測(圧縮あり)={real_p3:.1f}%  -> {pct_p3:.0f}パーセンタイル")
        in_band = (2.5 <= pct_med <= 97.5) and (2.5 <= pct_p3 <= 97.5)
        print(f"    圧縮あり subset は null 帯の{'内' if in_band else '外'}"
              f"（内=予言力なし、外=予言力の候補）")
    else:
        print("    圧縮あり n<5 -- skip")

    print(f"\n5. 巡回ブロック・ブートストラップ P(圧縮あり中央値 > 全体中央値)  primary N,M={PRIMARY}")
    if n_comp >= 5:
        for k in (1, 3, 6, 12):
            diffs = block_boot_diff(times[m_all], MFE[m_all], comp_flag[m_all], k)
            if diffs is None:
                print(f"    k={k}ヶ月: 月数不足でskip"); continue
            ok = ~np.isnan(diffs)
            p_pos = 100 * np.mean(diffs[ok] > 0)
            print(f"    k={k:>2}ヶ月ブロック: P(圧縮あり中央値>全体中央値) = {p_pos:.0f}%"
                  f"  (中央値の差の中央値={np.nanmedian(diffs):+.2f}R, 有効経路={ok.sum()}/{len(diffs)})")
    else:
        print("    圧縮あり n<5 -- skip")

    return dict(name=name, MFE=MFE, comp=primary, valid=valid, comp_flag=comp_flag, times=times)


def pooled_analysis(results):
    print(f"\n{'='*100}\nプール（btc15m_L + gold15m 合算）\n{'='*100}")
    MFE = np.concatenate([r["MFE"][r["valid"]] for r in results])
    comp_flag = np.concatenate([r["comp_flag"][r["valid"]] for r in results])
    times = np.concatenate([r["times"][r["valid"]] for r in results])
    n_comp = int(comp_flag.sum())
    print(f"\n1. n 総数(プール, comp計算可)={len(MFE)}  圧縮あり={n_comp} ({100*n_comp/len(MFE):.1f}%)")
    print(f"   圧縮あり subset 年別: {year_spread(pd.DatetimeIndex(times)[comp_flag])}")

    print(f"\n2. 層別 MFE 分布（プール, comp<1 primary N,M={PRIMARY}）")
    print(f"    {'':<12}{'n':>5}{'中央値':>10}{'平均':>9}{'標準偏差':>9}"
          f"{'P(<1R)':>11}{'P(>=1R)':>9}{'P(>=2R)':>9}{'P(>=3R)':>9}{'P(>=4.5R)':>9}")
    s_all = layer_stats(MFE); print_layer_row("全体base", s_all)
    s_comp = layer_stats(MFE[comp_flag]); print_layer_row("圧縮あり", s_comp)
    s_not = layer_stats(MFE[~comp_flag]); print_layer_row("圧縮なし", s_not)

    if n_comp >= 5:
        print(f"\n4. 同数ランダム部分集合 null（プール、1000回、n={n_comp}）")
        meds, p3s = random_subset_null(MFE, n_comp)
        real_med = s_comp["median"]; real_p3 = s_comp["p3"]
        pct_med = 100 * np.mean(meds < real_med)
        pct_p3 = 100 * np.mean(p3s < real_p3)
        print(f"    null帯 MFE中央値: [{np.percentile(meds,2.5):.2f}, {np.percentile(meds,97.5):.2f}]"
              f"  実測(圧縮あり)={real_med:.2f}  -> {pct_med:.0f}パーセンタイル")
        print(f"    null帯 P(>=3R):  [{np.percentile(p3s,2.5):.1f}%, {np.percentile(p3s,97.5):.1f}%]"
              f"  実測(圧縮あり)={real_p3:.1f}%  -> {pct_p3:.0f}パーセンタイル")

        print(f"\n5. 巡回ブロック・ブートストラップ（プール）P(圧縮あり中央値 > 全体中央値)")
        for k in (1, 3, 6, 12):
            diffs = block_boot_diff(times, MFE, comp_flag, k)
            if diffs is None:
                print(f"    k={k}ヶ月: 月数不足でskip"); continue
            ok = ~np.isnan(diffs)
            p_pos = 100 * np.mean(diffs[ok] > 0)
            print(f"    k={k:>2}ヶ月ブロック: P(圧縮あり中央値>全体中央値) = {p_pos:.0f}%"
                  f"  (中央値の差の中央値={np.nanmedian(diffs):+.2f}R, 有効経路={ok.sum()}/{len(diffs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    d_L, cfg_L = load_btc15m_L(smoke=args.smoke)
    d_G, cfg_G = load_gold15m(smoke=args.smoke)

    r_L = analyze_leg("btc15m_L  (data/vantage_btcusd_m15.csv, start 2018-10-01)", d_L, cfg_L,
                       tie_rr=4.5, tie_expect_n=759 if not args.smoke else None,
                       tie_expect_meanR=0.59 if not args.smoke else None)
    r_G = analyze_leg("gold15m  (data/vantage_xauusd_m5.csv->15min, start 2018-09-14 -- 正典に合わせ m5 使用)",
                       d_G, cfg_G, tie_rr=4.0)

    results = [r for r in (r_L, r_G) if r is not None]
    if len(results) == 2:
        pooled_analysis(results)

    print(f"\n{'='*100}\nfalsify 判定\n{'='*100}")
    print("上記の null帯(項目4)・ブロック・ブートストラップ(項目5)を見て判定すること:")
    print("  - 圧縮あり subset が null帯の【内】-> 溜めは伸びしろを予言しない(構造法則4: フィルタ~0 lift)")
    print("  - 圧縮あり subset が null帯の【外】(有利側)かつブロックを伸ばしても P が崩れない")
    print("    -> 予言力の候補。ただしその場合、構造法則9 により、これは目標変数(利確を近くに)ではなく")
    print("       サイズ変数として使うべき(次実験: 玉を厚くする方向で検証)。")


if __name__ == "__main__":
    main()
