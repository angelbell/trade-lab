"""仕様カード scratchpad/spec_strength_btc15mL.md の実装。

btc15m_L 1本を research/book.py の get_book_legs() L100-105 と厳密一致する仕様で再構築し
(照合ゲート)、3つの強度候補 (base_bars / risk_frac / vol_z) それぞれについて5分位EV表・
Spearman(ブロックbootstrap CI)・ランダム除去nullを出す。

候補の中身:
  1. base_bars: tL の既存列（run()/walk() が返す i - i_origin。i=ブレイク確定足、
     i_origin=wave起点(L0)の足 -- 押し目指値の待ち本数ではなく「waveの長さ(本数)」。
     spec の説明文はやや不正確なので下の報告で明記する。列自体はそのまま使う）。
  2. risk_frac: tL.risk / tL.e_px。
  3. vol_z: ブレイク確定足 i (fill足ではない) の 15m tick_volume の96本ローリングz。
     no-lookahead: i を復元するのに src/engine の plan()/detect() を直呼びして
     entries=(e_i,e,stop,tgt,i_origin) を取り、walk() の実トレード列(t2, tL と bit一致のはず)
     に fill価格式 lim=e-pf*(e-stop) と base_bars=e_i-i_origin で厳密対応付けする
     (近似"time - base_bars本"は base_bars の定義上 i_origin 側にズレるため不採用、
     下の報告で理由を明記)。

Run:
  .venv/bin/python scratchpad/strength_btc15mL.py --smoke 2>&1 | tee scratchpad/out_strength_btc15mL_smoke.txt
  .venv/bin/python scratchpad/strength_btc15mL.py 2>&1 | tee scratchpad/out_strength_btc15mL.txt
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from src.engine.presets import BASE
from src.engine.size import pdh_soft
from src.engine.gates import gate_sma, gate_kama, exit_flip
from src.engine.detect import make_swings, pattern_b
from src.engine.plan import plan
from src.engine.walk import walk

DATA = f"{ROOT}/data/vantage_btcusd_m15.csv"


def build(smoke):
    """btc15m_L を book.get_book_legs() と厳密一致する仕様で構築。volume付きの
    raw も返す(resample()がvolume列を落とすため、位置揃えで別持ちする)。"""
    with contextlib.redirect_stderr(io.StringIO()):
        raw = load_mt5_csv(DATA).loc["2018-10-01":]
        if smoke:
            raw = raw.loc[:"2019-12-31"]
        d15 = resample(raw, "15min")
    assert len(d15) == len(raw) and (d15.index == raw.index).all(), \
        "resample('15min') が bar 位置をズラした -- volume の位置揃えが無効"

    args = SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                               "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200})
    tL = run(d15, args)
    if tL is None:
        raise SystemExit("no entries (data too short for --smoke?)")
    WL, _ = pdh_soft(d15, tL)
    netR = (tL["R"].values - 15.0 / tL["risk"].values) * WL
    return d15, raw, args, tL, netR


def rebuild_entries(d15, args):
    """run()内部と同じ手順を直呼びし、entries=(e_i,e,stop,tgt,i_origin) と
    そこから作った t2(walk実行結果) を返す。t2 は tL と bit一致するはず(照合対象)。"""
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
    t2, rr_real2 = walk(d15, entries, against, args)
    return entries, t2


def match_entries_to_trades(entries, t, pf):
    """t の各行(walk()の出力順=entries順を保存)に、対応する entries の e_i を割り当てる。
    entries は e_i 昇順(plan()がsortする)。t も e_bar(fill)昇順で、walk()はentriesを
    順番に舐めて条件を満たすものだけ trades に足すので、ポインタを前進させるだけで
    1:1対応が復元できる(base_bars と fill価格 lim=e-pf*(e-stop) の完全一致で検算)。"""
    idx_i = np.full(len(t), -1, dtype=np.int64)
    k = 0
    bbs = t["base_bars"].values
    epxs = t["e_px"].values
    for j in range(len(t)):
        matched = False
        while k < len(entries):
            e_i, e, stop, tgt, i_origin = entries[k]
            k += 1
            bb = e_i - i_origin
            lim = e - pf * (e - stop)
            if bb == bbs[j] and abs(lim - epxs[j]) < 1e-9:
                idx_i[j] = e_i
                matched = True
                break
        if not matched:
            raise RuntimeError(f"trade row {j} (time={t['time'].iloc[j]}) に対応する entry が"
                                " 見つからない -- 対応付けロジックを見直すこと")
    return idx_i


def vol_z_at(raw_vol, i_arr, window=96):
    vol_s = pd.Series(raw_vol)
    roll_mean = vol_s.rolling(window, min_periods=window).mean()
    roll_std = vol_s.rolling(window, min_periods=window).std()
    z = ((vol_s - roll_mean) / roll_std).values
    return z[i_arr]


# ---------------------------------------------------------------- quintile analysis

def quintile_table(x, R):
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "R": np.asarray(R, dtype=float)})
    df = df.dropna()
    ranks = df["x"].rank(method="first")
    q = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for i in [1, 2, 3, 4, 5]:
        sub = df.loc[q == i, "R"]
        n = len(sub)
        win = 100.0 * (sub > 0).mean()
        pos = sub[sub > 0].sum()
        neg = abs(sub[sub <= 0].sum())
        pf = pos / neg if neg > 0 else np.inf
        rows.append(dict(q=i, n=n, win=win, pf=pf, meanR=sub.mean(), totR=sub.sum()))
    return rows, df


def monotone_flag(rows):
    means = [r["meanR"] for r in rows]
    nondecr = all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1))
    strictly_up_ends = means[-1] > means[0]
    return nondecr, strictly_up_ends, means


def block_bootstrap_spearman(times, x, R, k_months, n_boot=1000, seed=20260718):
    s = pd.DataFrame({"x": x, "R": R}, index=pd.DatetimeIndex(times)).dropna()
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(nm / k_months))
    rhos = []
    for _ in range(n_boot):
        starts = rng.integers(0, nm, size=nblk)
        seq = np.concatenate([[(st + j) % nm for j in range(k_months)] for st in starts])
        samp = pd.concat([by_month[months[j]] for j in seq])
        if len(samp) < 10 or samp["x"].nunique() < 2:
            continue
        rho, _ = spearmanr(samp["x"], samp["R"])
        if not np.isnan(rho):
            rhos.append(rho)
    rhos = np.array(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(np.median(rhos)), float(lo), float(hi), len(rhos)


def random_drop_null(all_R, actual_meanR, n_top, n_reps=5000, seed=1):
    rng = np.random.default_rng(seed)
    all_R = np.asarray(all_R, dtype=float)
    means = np.empty(n_reps)
    for r in range(n_reps):
        idx = rng.choice(len(all_R), size=n_top, replace=False)
        means[r] = all_R[idx].mean()
    pct = 100.0 * (means < actual_meanR).mean()
    return pct, means.mean(), means.std()


def report_candidate(name, x, R, times, tag=""):
    print(f"\n{'='*78}\n候補: {name} {tag}\n{'='*78}")
    rows, df = quintile_table(x, R)
    print(f"  {'Q':>2}{'n':>6}{'win%':>8}{'PF':>8}{'meanR':>9}{'totR':>9}")
    for r in rows:
        pf_s = f"{r['pf']:.2f}" if np.isfinite(r["pf"]) else "inf"
        print(f"  {r['q']:>2}{r['n']:>6}{r['win']:>7.1f}%{pf_s:>8}{r['meanR']:>+9.3f}{r['totR']:>+9.1f}")
    nondecr, up, means = monotone_flag(rows)
    print(f"  meanR系列(Q1->Q5): {[round(m,3) for m in means]}")
    print(f"  単調非減少(Q1<=Q2<=...<=Q5): {'YES' if nondecr else 'NO'}   Q5>Q1: {'YES' if up else 'NO'}")

    rho, p = spearmanr(df["x"], df["R"])
    print(f"  Spearman(x,R) = {rho:+.4f}  (p={p:.4g}, n={len(df)})")
    for k in (1, 3, 6, 12):
        med, lo, hi, nvalid = block_bootstrap_spearman(times, x, R, k, n_boot=1000)
        print(f"    循環ブロック({k}mo) bootstrap median rho={med:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]"
              f"  (有効draw={nvalid}/1000)")

    q5 = rows[4]
    pct, null_mean, null_std = random_drop_null(df["R"].values, q5["meanR"], q5["n"])
    print(f"  ランダム除去null: Q5と同数(n={q5['n']})をランダム抽出した meanR の分布 "
          f"(平均{null_mean:+.3f}±{null_std:.3f}) に対し 実測Q5 meanR={q5['meanR']:+.3f} は "
          f"{pct:.1f}パーセンタイル")
    return rows, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    d15, raw, args, tL, netR = build(cli.smoke)
    print(f"btc15m_L 再構築: n={len(tL)}  span={tL['time'].iloc[0]} -> {tL['time'].iloc[-1]}"
          f"  (smoke={cli.smoke})")

    # ---- 照合ゲート1: 自作 netR vs research.book.get_book_legs()["btc15m_L"] ----
    if cli.smoke:
        print("\n[照合ゲート1] --smoke のためスキップ (get_book_legs()はフルデータ前提)")
        gate1 = None
    else:
        import research.book as book_mod
        with contextlib.redirect_stderr(io.StringIO()):
            legs = book_mod.get_book_legs()
        ref = legs["btc15m_L"]
        mine = pd.Series(netR, index=pd.DatetimeIndex(tL["time"]))
        same_len = len(ref) == len(mine)
        same_idx = same_len and ref.index.equals(mine.index)
        same_val = same_idx and np.allclose(ref.values, mine.values, rtol=0, atol=1e-12)
        gate1 = same_len and same_idx and same_val
        print(f"\n[照合ゲート1] netR vs book.get_book_legs()['btc15m_L']: "
              f"len {len(ref)}=={len(mine)} -> {same_len} | idx一致 -> {same_idx} | "
              f"値一致(atol=1e-12) -> {same_val}  => {'PASS' if gate1 else 'FAIL'}")
        if not gate1:
            print("!!! 照合ゲート1 FAIL -- 以降の数字は信用しないこと。ここで停止する。")
            return

    # ---- entries 復元 + 照合ゲート2 (t2 が tL と bit一致するか) ----
    entries, t2 = rebuild_entries(d15, args)
    same_n = len(t2) == len(tL)
    cols = ["time", "R", "hold", "risk", "e_px", "r_mkt", "filled", "base_bars"]
    same_vals = same_n and all(
        (np.allclose(t2[c].values.astype(float), tL[c].values.astype(float),
                      rtol=0, atol=1e-9) if c != "time" else
         (t2[c].values == tL[c].values).all())
        for c in cols
    )
    gate2 = same_n and same_vals
    print(f"\n[照合ゲート2] entries直呼び再構築 t2 vs run()の tL: n {len(t2)}=={len(tL)} -> {same_n} | "
          f"列一致({cols}) -> {same_vals}  => {'PASS' if gate2 else 'FAIL'}")
    if not gate2:
        print("!!! 照合ゲート2 FAIL -- i の復元(entries対応付け)を信用できない。ここで停止する。")
        return

    i_arr = match_entries_to_trades(entries, tL, args.pullback_frac)
    print(f"[照合ゲート3] entries<->trades 対応付け: {len(i_arr)}/{len(tL)} 本すべて一意対応 => PASS")

    # 参考: spec記載の近似 (tL.time から base_bars 本遡る) は i_origin 側に寄る -- 数本サンプルで確認
    base_bars = tL["base_bars"].values
    approx_i = d15.index.get_indexer(pd.DatetimeIndex(tL["time"])) - base_bars
    n_sample = min(5, len(i_arr))
    print(f"\n[参考] spec記載の近似(fill_bar_pos - base_bars) vs entries経由の真のi (先頭{n_sample}件):")
    fill_pos = d15.index.get_indexer(pd.DatetimeIndex(tL["time"]))
    for j in range(n_sample):
        print(f"    trade{j}: fill_pos={fill_pos[j]} base_bars={base_bars[j]} "
              f"近似i={approx_i[j]}  真のi(entries経由)={i_arr[j]}  差={approx_i[j]-i_arr[j]}")
    print("    (base_bars = e_i - i_origin であり fill待ち本数ではないため、上の近似は"
          "一般に真のiと一致しない -- 本測定は entries 直呼び経由の真のiを使う)")

    R = tL["R"].values  # 強度は"実現R"との単調性で見る(仕様通り。netR/コスト後ではなくR)
    times = tL["time"].values

    # ---- 候補1: base_bars ----
    report_candidate("base_bars (既存列 = e_i - i_origin, wave長の代理)", base_bars, R, times)

    # ---- 候補2: risk_frac ----
    risk_frac = tL["risk"].values / tL["e_px"].values
    report_candidate("risk_frac (= risk / e_px)", risk_frac, R, times)

    # ---- 候補3: vol_z ----
    vol_z = vol_z_at(raw["volume"].values, i_arr, window=96)
    n_nan = np.isnan(vol_z).sum()
    print(f"\n[vol_z] 96本ローリングz、window不足でNaNの本数: {n_nan}/{len(vol_z)} (先頭付近のみのはず)")
    mask = ~np.isnan(vol_z)
    report_candidate("vol_z (breakout確定足iの15m出来高, 96本z, no-lookahead)",
                      vol_z[mask], R[mask], times[mask],
                      tag=f"[有効n={mask.sum()}, NaN除外={n_nan}]")

    print(f"\n実行コマンド: .venv/bin/python scratchpad/strength_btc15mL.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
