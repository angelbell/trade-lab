"""仕様カード experiments/spec_gateslope_generalize.md の実装。

問い: btc15m_L で「ゲート指標(4H KAMA)の傾きの急さ」が本物の強度勾配だった(カード2)。
これが他レッグでも成り立つ横断プリミティブかを、gold15m と btc15m_S(ショート鏡像)で検定する。

土台は experiments/strength_btc15mL.py (base) と strength_regime_btc15mL.py (reg) をそのまま
import して使う。rebuild_entries/match_entries_to_trades/quintile_table/monotone_flag/
block_bootstrap_spearman/random_drop_null/report_candidate は完全に汎用(引数に d, args を
渡すだけ)なので、レッグ構築の d/args/コスト/マスクだけを research/book.py の
get_book_legs() L94-98 (gold15m) / L107-112 (btc15m_S) と一字一句一致させて差し替える。

各レッグの強度候補:
  gold15m : 日足SMA150の傾きの急さ = (SMA150_d[i]-SMA150_d[i-1])/SMA150_d[i-1]、
            前日までの確定日足 (gate_sma と同じ shift(1)+ffill 規約、ただし候補式はk=1の
            1本差分 -- gate自体の daily_slope_k=10 とは無関係、spec記載の式をそのまま実装)。
  btc15m_S: 日足KAMA(14)の傾きの急さ。gate は run(inv, ...) の gate_kama が「反転価格が上向き」
            =「実価格が下向き」で発火する。強度候補は実価格(非反転 d15)の日足KAMAの傾きを
            reg.compute_kama_slope(d15_real, tf="1D") で計算し、期待符号は負(下降)なので
            strength = -slope (下向きに急なほど正で大きい) として報告する。

照合ゲート(各レッグ):
  ゲート1: 自作netR (book.py と同じコスト/マスク適用) vs research.book.get_book_legs()[leg] の
           時刻・値一致。
  ゲート2: entries直呼び再構築(base.rebuild_entries)の t2 が run()の生トレード表(gold15mはt、
           btc15m_Sはマスク前のts)と bit一致。
  ゲート3: base.match_entries_to_trades で entries<->trades が全数一意対応。

Run:
  .venv/bin/python experiments/strength_gateslope_generalize.py --smoke 2>&1 | tee experiments/out_strength_gateslope_generalize_smoke.txt
  .venv/bin/python experiments/strength_gateslope_generalize.py 2>&1 | tee experiments/out_strength_gateslope_generalize.txt
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_btc15mL as base          # build/rebuild_entries/match_entries_to_trades +
                                          # quintile_table/monotone_flag/block_bootstrap_spearman/
                                          # random_drop_null/report_candidate
import strength_regime_btc15mL as reg    # compute_kama_slope

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from src.engine.presets import BASE
from src.engine.mirror import invert
from src.engine.size import pdl_break_mask

GOLD_M5 = f"{ROOT}/data/vantage_xauusd_m5.csv"
BTC_M15 = f"{ROOT}/data/vantage_btcusd_m15.csv"


# ---------------------------------------------------------------- gold15m 構築 (book.py L94-98)

def build_gold15m(smoke):
    with contextlib.redirect_stderr(io.StringIO()):
        raw = load_mt5_csv(GOLD_M5).loc["2018-09-14":]
        if smoke:
            raw = raw.loc[:"2019-12-31"]
        g15 = resample(raw, "15min")
    args = SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10, "ext_cap": 8.0,
                               "pullback_frac": 0.25, "fill_win": 200})
    t = run(g15, args)
    if t is None:
        raise SystemExit("no entries (data too short for --smoke?)")
    netR = t["R"].values - 0.3 / t["risk"].values
    return g15, args, t, netR


# ---------------------------------------------------------------- btc15m_S 構築 (book.py L107-112)

def build_btc15mS(smoke):
    with contextlib.redirect_stderr(io.StringIO()):
        raw = load_mt5_csv(BTC_M15).loc["2018-10-01":]
        if smoke:
            raw = raw.loc[:"2019-12-31"]
        d15 = resample(raw, "15min")
    inv = invert(d15)
    C = 2 * d15["high"].max()
    args = SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3, "rr": 4.5,
                               "fill_win": 200})
    ts = run(inv, args)
    if ts is None:
        raise SystemExit("no entries (data too short for --smoke?)")
    mS = pdl_break_mask(d15, ts, C)
    netR = (ts["R"].values - 15.0 / ts["risk"].values)[mS]
    return d15, inv, C, args, ts, mS, netR


# ---------------------------------------------------------------- 強度候補

def compute_sma_slope(d, sma_n=150, tf="1D"):
    """gold15m 候補: 日足SMA150の1本あたり変化率。gate_sma と同じ足付け(前日確定+ffill)、
    式はspec記載どおり k=1 の1本差分(gate自体のdaily_slope_k=10とは独立)。"""
    dc = d["close"].resample(tf).last().dropna()
    sma = dc.rolling(sma_n).mean()
    slope_pct = (sma - sma.shift(1)) / sma.shift(1)
    slope_shifted = slope_pct.shift(1)
    arr = slope_shifted.reindex(d.index, method="ffill").values
    return arr


# ---------------------------------------------------------------- 照合ゲート共通ヘルパ

def gate1_check(leg_name, mine_series, cli_smoke):
    if cli_smoke:
        print(f"\n[{leg_name} 照合ゲート1] --smoke のためスキップ (get_book_legs()はフルデータ前提)")
        return None
    import research.book as book_mod
    with contextlib.redirect_stderr(io.StringIO()):
        legs = book_mod.get_book_legs()
    ref = legs[leg_name]
    same_len = len(ref) == len(mine_series)
    same_idx = same_len and ref.index.equals(mine_series.index)
    same_val = same_idx and np.allclose(ref.values, mine_series.values, rtol=0, atol=1e-12)
    gate1 = same_len and same_idx and same_val
    print(f"\n[{leg_name} 照合ゲート1] netR vs book.get_book_legs()['{leg_name}']: "
          f"len {len(ref)}=={len(mine_series)} -> {same_len} | idx一致 -> {same_idx} | "
          f"値一致(atol=1e-12) -> {same_val}  => {'PASS' if gate1 else 'FAIL'}")
    return gate1


def gate2_check(leg_name, t2, t_ref):
    same_n = len(t2) == len(t_ref)
    cols = ["time", "R", "hold", "risk", "e_px", "r_mkt", "filled", "base_bars"]
    same_vals = same_n and all(
        (np.allclose(t2[c].values.astype(float), t_ref[c].values.astype(float),
                      rtol=0, atol=1e-9) if c != "time" else
         (t2[c].values == t_ref[c].values).all())
        for c in cols
    )
    gate2 = same_n and same_vals
    print(f"[{leg_name} 照合ゲート2] entries直呼び再構築 t2 vs run()の生トレード表: "
          f"n {len(t2)}=={len(t_ref)} -> {same_n} | 列一致({cols}) -> {same_vals}  "
          f"=> {'PASS' if gate2 else 'FAIL'}")
    return gate2


# ---------------------------------------------------------------- 年別 Q5-Q1 (era beta isolation)

def era_report(x, R, times, label_hi="Q5(強)", label_lo="Q1(弱)"):
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "R": np.asarray(R, dtype=float)},
                       index=pd.DatetimeIndex(times)).dropna()
    if len(df) < 10 or df["x"].nunique() < 5:
        print(f"\n  [時代ベータ隔離] n={len(df)} で分位を切るには不足 -- 測定不能")
        return
    ranks = df["x"].rank(method="first")
    df["Q"] = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["y"] = df.index.year
    print(f"\n  [時代ベータ隔離] 年別 {label_hi} vs {label_lo} meanR/n:")
    print(f"  {'年':>6} | {'Q5 n':>5} {'Q5 meanR':>9} | {'Q1 n':>5} {'Q1 meanR':>9} | {'Q5-Q1':>7} | {'全体n':>5}")
    per_year = []
    for y in sorted(df["y"].unique()):
        s = df[df["y"] == y]
        q5, q1 = s[s["Q"] == 5], s[s["Q"] == 1]
        m5 = q5["R"].mean() if len(q5) else float("nan")
        m1 = q1["R"].mean() if len(q1) else float("nan")
        diff = m5 - m1 if (len(q5) and len(q1)) else float("nan")
        print(f"  {y:>6} | {len(q5):>5} {m5:>+9.3f} | {len(q1):>5} {m1:>+9.3f} | {diff:>+7.3f} | {len(s):>5}")
        if len(q5) and len(q1):
            per_year.append(diff)
    per_year = np.array(per_year)
    if len(per_year):
        print(f"  Q5-Q1 が正の年: {(per_year>0).sum()}/{len(per_year)}  中央値={np.median(per_year):+.3f}")
    else:
        print("  Q5-Q1 が正の年: 両分位が同時に存在する年が無い -- 測定不能")


# ---------------------------------------------------------------- レッグ1: gold15m

def run_gold15m(cli_smoke):
    print(f"\n{'#'*78}\n# gold15m -- 強度候補: 日足SMA150の傾きの急さ\n{'#'*78}")
    g15, args, t, netR = build_gold15m(cli_smoke)
    print(f"gold15m 再構築: n={len(t)}  span={t['time'].iloc[0]} -> {t['time'].iloc[-1]}  (smoke={cli_smoke})")

    mine = pd.Series(netR, index=pd.DatetimeIndex(t["time"]))
    gate1 = gate1_check("gold15m", mine, cli_smoke)
    if gate1 is False:
        print("!!! gold15m 照合ゲート1 FAIL -- 以降の数字は信用しないこと。gold15mをスキップする。")
        return

    entries, t2 = base.rebuild_entries(g15, args)
    gate2 = gate2_check("gold15m", t2, t)
    if not gate2:
        print("!!! gold15m 照合ゲート2 FAIL -- entries復元を信用できない。gold15mをスキップする。")
        return

    i_arr = base.match_entries_to_trades(entries, t, args.pullback_frac)
    print(f"[gold15m 照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(t)} 本すべて一意対応 => PASS")

    R = t["R"].values
    times = t["time"].values

    slope_arr = compute_sma_slope(g15, sma_n=args.daily_sma, tf="1D")
    slope = slope_arr[i_arr]
    n_nan = int(np.isnan(slope).sum())
    n_neg = int((slope[~np.isnan(slope)] < 0).sum())
    print(f"\n[sma_slope 自己点検] トレード{len(slope)}本中、SMA150ウォームアップ等でNaN={n_nan}本、"
          f"負(下降)の本数={n_neg}/{len(slope)-n_nan}  "
          f"(gateはSMA150上向き+10本前より高い、を要求するがslope自体は1本差分でk=10とは独立なので"
          f"負の値が出ても矛盾ではない)")
    mask = ~np.isnan(slope)
    rows, rho = base.report_candidate(
        "sma_slope (= (SMA150_d[i]-SMA150_d[i-1])/SMA150_d[i-1], 前日確定, gold15m)",
        slope[mask], R[mask], times[mask], tag=f"[有効n={mask.sum()}, NaN除外={n_nan}]")
    era_report(slope[mask], R[mask], times[mask])
    return rows, rho


# ---------------------------------------------------------------- レッグ2: btc15m_S

def run_btc15mS(cli_smoke):
    print(f"\n{'#'*78}\n# btc15m_S -- 強度候補: 日足KAMA(14)の傾きの急さ(下向き=正で表現)\n{'#'*78}")
    d15, inv, C, args, ts, mS, netR = build_btc15mS(cli_smoke)
    print(f"btc15m_S 再構築(マスク後): n={mS.sum()}/{len(ts)}  (smoke={cli_smoke})")

    mine = pd.Series(netR, index=pd.DatetimeIndex(ts["time"])[mS])
    gate1 = gate1_check("btc15m_S", mine, cli_smoke)
    if gate1 is False:
        print("!!! btc15m_S 照合ゲート1 FAIL -- 以降の数字は信用しないこと。btc15m_Sをスキップする。")
        return

    entries, t2 = base.rebuild_entries(inv, args)
    gate2 = gate2_check("btc15m_S", t2, ts)
    if not gate2:
        print("!!! btc15m_S 照合ゲート2 FAIL -- entries復元を信用できない。btc15m_Sをスキップする。")
        return

    i_arr_full = base.match_entries_to_trades(entries, ts, args.pullback_frac)
    print(f"[btc15m_S 照合ゲート3] entries<->trades 対応付け(マスク前): "
          f"{len(i_arr_full)}/{len(ts)} 本すべて一意対応 => PASS")

    # PDLハードmaskを entries対応後の i_arr / R / times にも同じ順序で適用
    i_arr = i_arr_full[mS]
    R_full = ts["R"].values
    R = R_full[mS]
    times = ts["time"].values[mS]
    print(f"[btc15m_S] PDLハードmask適用後: n={mS.sum()}/{len(ts)}  "
          f"(R一致確認: netR復元前のR平均={R.mean():+.4f})")

    # 強度候補: 実価格(非反転)d15の日足KAMA(14)の傾き。gateはinv上のgate_kamaで発火するが、
    # 強度は実世界の下降の急さで測る -- reg.compute_kama_slope を real d15, tf="1D" で呼ぶ。
    ks_real_arr, ks_median = reg.compute_kama_slope(d15, n=args.gate_kama, tf="1D")
    ks_real = ks_real_arr[i_arr]
    n_nonneg = int((ks_real >= 0).sum())
    print(f"\n[kama_slope(実価格) 自己点検] トレード{len(ks_real)}本中、非負(=gate_kamaの下向き判定と"
          f"矛盾しうる境界)の本数: {n_nonneg}  最大値={np.nanmax(ks_real):+.6f}  最小値={np.nanmin(ks_real):+.6f}  "
          f"(240min粒度ではなく1D粒度の全履歴中央値={ks_median:+.6f})")
    strength = -ks_real  # 下向きに急なほど正で大きい
    rows, rho = base.report_candidate(
        "kama_slope_down (= -1 * 日足KAMA(14)の1本あたり変化率, btc15m_S, 下向きに急なほど正で大)",
        strength, R, times)
    era_report(strength, R, times)
    return rows, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    run_gold15m(cli.smoke)
    run_btc15mS(cli.smoke)

    print(f"\n実行コマンド: .venv/bin/python experiments/strength_gateslope_generalize.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
