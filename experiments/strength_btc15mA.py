"""仕様カード experiments/spec_strength_btc15mA.md の実装。

btc15m_A（= btc15m_L から「エントリー確定足の終値 e_px > 前日高値PDH」のハード選別だけを
残したもの。PDHソフト重みは使わない。README「単独運用: btc15m_A」の定義そのもの）を再構築し
（照合ゲート必須）、以下を測る:

  パートA: btc15m_L で確立した合成強度旗（kama_slope + stop_atr + atr_pctile の等重み
           ランク平均・トップ20%）を A の母集団内でランクを取り直して移植できるか。
  パートB: btc15m_A 単独仕様にある「日足終値 < 日足SMA150 のときサイズ×0.75」（台帳
           s02_exits.md §2.2, src/engine/size.daily_regime_mult）を、倍率でなく
           「日足↑/日足↓」の強度の段として出し直す（倍率そのものは出力しない）。
  パートC: 合成旗(トップ20%/残り) × 日足レジーム(↑/↓) の 2x2 裁量表。

車輪の再発明禁止 — 全て既存コードを import して使う:
  experiments/strength_btc15mL.py:
      build()                    -- btc15m_L を research/book.py 準拠の仕様で再構築
      rebuild_entries()          -- entries直呼び再構築(確定足iの復元, 照合ゲート2)
      match_entries_to_trades()  -- entries<->trades 対応付け(照合ゲート3)
      quintile_table/monotone_flag/block_bootstrap_spearman/random_drop_null/report_candidate
  experiments/strength_regime_btc15mL.py:
      compute_kama_slope() -- 4H(240min) KAMA(14) の1本あたり変化率、gate_kamaと同じ
                              shift(1)+ffill規約（候補1: kama_slope）
  src/engine/size.py:
      pdh_series()        -- btc15m_A のハード選別(e_px>PDH)に使う前日高値系列
      pdh_soft()           -- 照合ゲート1（L側の book.get_book_legs() 突き合わせ）用
      daily_regime_mult()  -- パートBの「日足終値<SMA150」フラグ(down_at)をそのまま流用
                              (倍率multは無視し、down_atのブール値だけ使う＝倍率は出力しない)
  probe_combo_validate.py が固めた合成スコアの定義（rank_pct平均・qcut5でトップ20%=Q5）を
  そのまま再現する（atr_pctile の rolling(500).apply 定義もこのスクリプトに合わせた）。

台帳の再構築照合ゲート(ゲート4, 2026-07-19 メイン指示で差し替え):
仕様カード記載の「s13_pullback_limit.md kama4h(C1攻め)版 n238」は 2026-07-03・RR4.0 時代の
旧測定で指定ミスと判明（メイン確認済み）。正しい照合基準は **現行README の btc15m_A
（RR4.5・fill_win200 = 今回の土台がすでに正確に再現しているもの）**:
  素netR(重みなし)      n=229 / win=34.1% / meanR=+1.170 / PF=2.61 / IS=+1.15 / OOS=+1.19
  参考(日足×0.75込み)   meanR=+1.104 / PF=2.68 / IS=+1.05 / OOS=+1.16
  (README掲載値         meanR=+1.09 / PF=2.70 / IS=+1.012 / OOS=+1.156 -- ほぼ一致、
   残差はREADMEのIS/OOS分割方法(トレード本数半分でなく年で半分等)の違いと見られる)
一致しなければ以降の数字は出さず、この場で報告して停止する。
パートA/B/C は全て**素netR(×0.75の重みを掛けない)**で計算する — ×0.75 はパートBで
「強度の段」に変換する対象そのものなので、Rに混ぜると二重計上になる。

Run:
  .venv/bin/python experiments/strength_btc15mA.py --smoke 2>&1 | tee experiments/out_strength_btc15mA_smoke.txt
  .venv/bin/python experiments/strength_btc15mA.py 2>&1 | tee experiments/out_strength_btc15mA.txt
"""
import argparse
import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import strength_btc15mL as base
import strength_regime_btc15mL as reg
from src.engine.size import pdh_series, pdh_soft, daily_regime_mult, bar_idx

# 2026-07-19 メイン指示で差し替え: 現行README btc15m_A(RR4.5・fill_win200)の素netR(重みなし)。
# 仕様カードが元々指した s13_pullback_limit.md「kama4h(C1攻め)版 n238」は2026-07-03・RR4.0の
# 旧測定で指定ミスと判明(メイン確認済み)。
LEDGER = dict(n=229, win=34.1, meanR=1.170, pf=2.61, isR=1.15, oosR=1.19)


# ---------------------------------------------------------------- 汎用統計ヘルパ

def stats_of(R):
    R = np.asarray(R, dtype=float)
    n = len(R)
    if n == 0:
        return dict(n=0, win=np.nan, pf=np.nan, meanR=np.nan, isR=np.nan, oosR=np.nan, totR=0.0)
    win = 100.0 * (R > 0).mean()
    pos = R[R > 0].sum()
    neg = abs(R[R <= 0].sum())
    pf = pos / neg if neg > 0 else (np.inf if pos > 0 else np.nan)
    h = n // 2
    isR = R[:h].mean() if h > 0 else np.nan
    oosR = R[h:].mean() if (n - h) > 0 else np.nan
    return dict(n=n, win=win, pf=pf, meanR=R.mean(), isR=isR, oosR=oosR, totR=R.sum())


def pf_str(st):
    return f"{st['pf']:.2f}" if np.isfinite(st['pf']) else "inf"


def print_two_row(label_a, st_a, label_b, st_b, span_years, note=""):
    print(f"  {'':<14}{'n':>6}{'本/年':>7}{'win%':>8}{'PF':>8}{'meanR':>9}")
    for lab, st in ((label_a, st_a), (label_b, st_b)):
        print(f"  {lab:<14}{st['n']:>6}{st['n']/span_years:>7.1f}{st['win']:>7.1f}%"
              f"{pf_str(st):>8}{st['meanR']:>+9.3f}")
    if note:
        print(f"  {note}")


# ---------------------------------------------------------------- A の再構築

def build_A(smoke):
    """btc15m_L を strength_btc15mL.build() でそのまま再構築し、e_px>PDH のハード選別
    だけを適用して btc15m_A を作る(PDHソフト重みは使わない -- README/台帳の定義通り)。"""
    d15, raw, args, tL, netR_L = base.build(smoke)
    ei = bar_idx(d15, tL)
    pdh = pdh_series(d15)
    ab = tL["e_px"].values > pdh[ei]
    return d15, raw, args, tL, ab


def block_bootstrap_gap(times, top_mask, R, k_months, n_boot=3000, seed=20260719):
    """トップ20%(固定メンバーシップ) vs 残りの meanR ギャップを、月ブロックの巡回
    ブートストラップで再標本化する。strength_btc15mL.block_bootstrap_spearman と同じ
    「月を単位にブロック化して巡回抽出」の型を、Spearmanでなくグループ間ギャップに適用した
    もの(既存関数はSpearman専用で使い回せないため、この統計量だけ新規に用意する)。"""
    s = pd.DataFrame({"top": top_mask, "R": R}, index=pd.DatetimeIndex(times))
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(nm / k_months))
    gaps = []
    for _ in range(n_boot):
        starts = rng.integers(0, nm, size=nblk)
        seq = np.concatenate([[(st + j) % nm for j in range(k_months)] for st in starts])
        samp = pd.concat([by_month[months[j]] for j in seq])
        top_r = samp.loc[samp["top"], "R"]
        rest_r = samp.loc[~samp["top"], "R"]
        if len(top_r) < 5 or len(rest_r) < 5:
            continue
        gaps.append(top_r.mean() - rest_r.mean())
    gaps = np.array(gaps)
    if len(gaps) == 0:
        return np.nan, np.nan, np.nan, 0
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(np.median(gaps)), float(lo), float(hi), len(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    d15, raw, args, tL, ab = build_A(cli.smoke)
    print(f"btc15m_L 再構築: n={len(tL)}  span={tL['time'].iloc[0]} -> {tL['time'].iloc[-1]}"
          f"  (smoke={cli.smoke})")
    print(f"btc15m_A ハード選別(e_px>PDH, PDHソフト無し): {int(ab.sum())}/{len(tL)} 本を採用")

    # ---- 照合ゲート1: L側 netR(PDH込み) vs book.get_book_legs()['btc15m_L'] ----
    if cli.smoke:
        print("\n[照合ゲート1] --smoke のためスキップ (get_book_legs()はフルデータ前提)")
    else:
        import research.book as book_mod
        with contextlib.redirect_stderr(io.StringIO()):
            legs = book_mod.get_book_legs()
        ref = legs["btc15m_L"]
        WL, _ = pdh_soft(d15, tL)
        netR_pdh = (tL["R"].values - 15.0 / tL["risk"].values) * WL
        mine = pd.Series(netR_pdh, index=pd.DatetimeIndex(tL["time"]))
        same_len = len(ref) == len(mine)
        same_idx = same_len and ref.index.equals(mine.index)
        same_val = same_idx and np.allclose(ref.values, mine.values, rtol=0, atol=1e-12)
        gate1 = same_len and same_idx and same_val
        print(f"\n[照合ゲート1] L側 netR(PDH込) vs book.get_book_legs()['btc15m_L']: "
              f"len {len(ref)}=={len(mine)} -> {same_len} | idx一致 -> {same_idx} | "
              f"値一致(atol=1e-12) -> {same_val}  => {'PASS' if gate1 else 'FAIL'}")
        if not gate1:
            print("!!! 照合ゲート1 FAIL -- 以降の数字は信用しないこと。ここで停止する。")
            return

    # ---- entries復元 + 照合ゲート2 (t2 が tL と bit一致するか) ----
    entries, t2 = base.rebuild_entries(d15, args)
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

    i_arr_L = base.match_entries_to_trades(entries, tL, args.pullback_frac)
    print(f"[照合ゲート3] entries<->trades 対応付け: {len(i_arr_L)}/{len(tL)} 本すべて一意対応 => PASS")

    # ---- A の切り出し ----
    R_A = tL["R"].values[ab]
    risk_A = tL["risk"].values[ab]
    times_A = tL["time"].values[ab]
    i_arr_A = i_arr_L[ab]
    netR_A = R_A - 15.0 / risk_A
    stA = stats_of(netR_A)

    # ---- 照合ゲート4: 現行README btc15m_A(RR4.5・fill_win200・素netR)と一致するか ----
    # (2026-07-19 メイン指示: 仕様カード記載の s13_pullback_limit.md「kama4h(C1攻め)版 n238」は
    #  2026-07-03・RR4.0時代の旧測定で指定ミスと判明。正しい基準は現行README btc15m_A)
    print(f"\n[照合ゲート4] btc15m_A 再構築 vs 現行README基準(RR4.5・fill_win200・素netR):")
    print(f"    基準:   n={LEDGER['n']}  win={LEDGER['win']:.1f}%  meanR={LEDGER['meanR']:+.3f}  "
          f"PF={LEDGER['pf']:.2f}  IS={LEDGER['isR']:+.2f}  OOS={LEDGER['oosR']:+.2f}")
    print(f"    再構築: n={stA['n']}  win={stA['win']:.1f}%  meanR={stA['meanR']:+.3f}  "
          f"PF={pf_str(stA)}  IS={stA['isR']:+.2f}  OOS={stA['oosR']:+.2f}")
    if cli.smoke:
        print("    (--smoke のため台帳照合はスキップ。以降 --smoke時はここで打ち切り)")
        return
    gate4 = (stA['n'] == LEDGER['n']
             and abs(stA['win'] - LEDGER['win']) < 0.5
             and abs(stA['meanR'] - LEDGER['meanR']) < 0.005
             and np.isfinite(stA['pf']) and abs(stA['pf'] - LEDGER['pf']) < 0.01
             and abs(stA['isR'] - LEDGER['isR']) < 0.01
             and abs(stA['oosR'] - LEDGER['oosR']) < 0.01)
    print(f"    => {'PASS' if gate4 else 'FAIL'}  "
          f"(許容差: n完全一致, win%±0.5pt, meanR/IS/OOS±0.01, PF±0.01)")
    if not gate4:
        print("!!! 照合ゲート4 FAIL -- 台帳基準と一致しない。以降の数字は出さず、"
              "ここで報告のため停止する。")
        return

    # ---- 参考クロスチェック: 日足×0.75を掛けた版がREADME掲載値(meanR+1.09/PF2.70/IS+1.012/OOS+1.156)
    #      に近いことを確認する(倍率はここでは確認だけ、パートA/B/Cの計算には一切使わない) ----
    _, down_at_A_ref = daily_regime_mult(d15, tL)
    down_at_A_ref = down_at_A_ref[ab]
    w_ref = np.where(down_at_A_ref, 0.75, 1.0)
    st_w = stats_of(netR_A * w_ref)
    print(f"\n    [参考・日足×0.75込みクロスチェック] meanR={st_w['meanR']:+.3f}  PF={pf_str(st_w)}  "
          f"IS={st_w['isR']:+.3f}  OOS={st_w['oosR']:+.3f}  "
          f"(README掲載 meanR+1.09/PF2.70/IS+1.012/OOS+1.156 と近似 -- 残差はREADMEのIS/OOS"
          f"分割方法の違いと見られる。パートA/B/Cではこの重み付けは使わない＝素netRのみ使用)")

    span_years = (pd.DatetimeIndex(times_A).max() - pd.DatetimeIndex(times_A).min()).days / 365.25
    print(f"\nspan_A = {span_years:.2f}yr  (本/年 = {stA['n']/span_years:.1f})  "
          f"R定義: netR = tL.R - 15/tL.risk (PDHソフト無し)")

    # ================================================================ 3軸 + 日足レジームの計算
    atr14 = ta.atr(d15["high"], d15["low"], d15["close"], length=args.atr).values
    kama_slope_arr, kama_median = reg.compute_kama_slope(d15, n=args.gate_kama, tf=args.gate_kama_tf)

    ks = kama_slope_arr[i_arr_A]
    stop_atr = risk_A / atr14[i_arr_A]
    atr_s = pd.Series(atr14)
    atr_pctile_full = atr_s.rolling(500).apply(lambda w: (w[-1] > w[:-1]).mean(), raw=True).values
    atr_pctile = atr_pctile_full[i_arr_A]

    _, down_at_L = daily_regime_mult(d15, tL)   # 全Lに対して計算 -> Aの位置でマスクするだけ
    down_at_A = down_at_L[ab]

    df = pd.DataFrame({"R": netR_A, "t": pd.DatetimeIndex(times_A), "down": down_at_A,
                        "ks": ks, "sa": stop_atr, "ap": atr_pctile})
    n_all = len(df)
    df = df.dropna(subset=["ks", "sa", "ap"]).reset_index(drop=True)
    n_drop = n_all - len(df)

    # ================================================================ パートA
    print(f"\n{'#'*78}\nパートA: 合成強度旗(kama_slope+stop_atr+atr_pctile)をA母集団内で移植\n{'#'*78}")
    print(f"3軸すべて有効な本数: {len(df)}/{n_all} (NaN除外={n_drop}, ATR/atr_pctile"
          f"のウォームアップ落ち)")

    print("\n--- 冗長性チェック: A母集団内での各軸 単体Spearman(x,R) ---")
    print("  (参考: btc15m_L 全母集団での単体Spearman -- kama_slope +0.089(p=.014,n=763,"
          "`out_strength_regime_btc15mL.txt`) / stop_atr +0.172(p<.001,n=763) / "
          "atr_pctile +0.216(p<.001,n=762) [`out_entryquality_btc15mL.txt`])")
    axis_rho = {}
    for name, col in (("kama_slope", "ks"), ("stop_atr", "sa"), ("atr_pctile", "ap")):
        rho, p = spearmanr(df[col], df["R"])
        axis_rho[name] = rho
        print(f"    {name:<12} A母集団内 Spearman = {rho:+.4f}  (p={p:.4g}, n={len(df)})")

    print("\n--- 各軸: A母集団内での5分位表(strength_btc15mL.report_candidate を流用) ---")
    for name, col in (("kama_slope", "ks"), ("stop_atr", "sa"), ("atr_pctile", "ap")):
        base.report_candidate(f"{name} (A母集団内ランク)", df[col].values, df["R"].values,
                               df["t"].values)

    # ---- 合成スコア: probe_combo_validate.py と同一定義(等重みランク平均)、A母集団内で再ランク ----
    df["combo"] = (df["ks"].rank(pct=True) + df["sa"].rank(pct=True) + df["ap"].rank(pct=True)) / 3
    print(f"\n--- 合成スコア(kama_slope+stop_atr+atr_pctile 等重みランク平均、A母集団内で再ランク) ---")
    rows, _ = base.quintile_table(df["combo"].values, df["R"].values)
    print(f"  {'Q':>2}{'n':>6}{'win%':>8}{'PF':>8}{'meanR':>9}{'totR':>9}")
    for r in rows:
        pf_s = f"{r['pf']:.2f}" if np.isfinite(r["pf"]) else "inf"
        print(f"  {r['q']:>2}{r['n']:>6}{r['win']:>7.1f}%{pf_s:>8}{r['meanR']:>+9.3f}{r['totR']:>+9.1f}")
    nondecr, up, means = base.monotone_flag(rows)
    print(f"  meanR系列(Q1->Q5): {[round(m, 3) for m in means]}")
    print(f"  単調非減少(Q1<=...<=Q5): {'YES' if nondecr else 'NO'}   Q5>Q1: {'YES' if up else 'NO'}")

    rho_combo, p_combo = spearmanr(df["combo"], df["R"])
    print(f"  Spearman(combo,R) = {rho_combo:+.4f}  (p={p_combo:.4g}, n={len(df)})")

    df["Q"] = pd.qcut(df["combo"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    top_mask = (df["Q"] == 5).values
    top_st = stats_of(df.loc[top_mask, "R"].values)
    rest_st = stats_of(df.loc[~top_mask, "R"].values)
    gap = top_st["meanR"] - rest_st["meanR"]
    print(f"\n  トップ20%(旗ON) vs 残り80%（このA母集団, n={len(df)}）:")
    print_two_row("トップ20%", top_st, "残り80%", rest_st, span_years,
                  note=f"ギャップ(トップ-残り) meanR = {gap:+.3f}")

    print("\n  循環ブロック・ブートストラップ(1/3/6/12mo, 3000回): Spearman(combo,R) CI")
    for k in (1, 3, 6, 12):
        med, lo, hi, nvalid = base.block_bootstrap_spearman(df["t"].values, df["combo"].values,
                                                              df["R"].values, k, n_boot=3000)
        print(f"    {k:>2}mo: median rho={med:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"(有効draw={nvalid}/3000)")

    print("\n  循環ブロック・ブートストラップ(1/3/6/12mo, 3000回): "
          "トップ20%-残り meanRギャップ CI(トップの帰属は原本全体で固定)")
    for k in (1, 3, 6, 12):
        med, lo, hi, nvalid = block_bootstrap_gap(df["t"].values, top_mask, df["R"].values,
                                                    k, n_boot=3000)
        tag = "0超" if (np.isfinite(lo) and lo > 0) else "0またぎ"
        print(f"    {k:>2}mo: median gap={med:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
              f"(有効draw={nvalid}/3000)  {tag}")

    print("\n  年別 トップ-残り (時代隔離):")
    yrs = df["t"].dt.year
    for y in sorted(yrs.unique()):
        m = (yrs == y).values
        top_y = df.loc[m & top_mask, "R"]
        rest_y = df.loc[m & ~top_mask, "R"]
        if len(top_y) < 3 or len(rest_y) < 3:
            print(f"    {y}: トップn={len(top_y)} 残りn={len(rest_y)} (薄いのでスキップ)")
            continue
        print(f"    {y}: トップ(n={len(top_y)}) meanR={top_y.mean():+.3f} | "
              f"残り(n={len(rest_y)}) meanR={rest_y.mean():+.3f} | "
              f"差={top_y.mean() - rest_y.mean():+.3f}")

    # ================================================================ パートB
    print(f"\n{'#'*78}\nパートB: 日足レジーム(既存 daily_regime_mult 流用、倍率でなく強度の段として)\n{'#'*78}")
    print("  定義(README/台帳s02_exits.md §2.2 と同一): src/engine/size.daily_regime_mult"
          "(sma_n=150) — 前日までの確定日足終値 < 日足SMA150 なら「日足↓」"
          "(shift(1)+ffill、倍率mult自体はここでは使わずdown_atのブールのみ流用)")
    up_st_full = stats_of(netR_A[~down_at_A])
    dn_st_full = stats_of(netR_A[down_at_A])
    print(f"\n  A の全トレード(n={len(netR_A)})を日足レジームで層別（強度の段として表現。"
          f"倍率×0.75は出力しない）:")
    print_two_row("日足↑", up_st_full, "日足↓(弱段)", dn_st_full, span_years,
                  note=f"差(日足↑−日足↓) meanR = {up_st_full['meanR']-dn_st_full['meanR']:+.3f}")

    # ================================================================ パートC
    print(f"\n{'#'*78}\nパートC: 統合強度表(裁量用の一枚) — 合成旗(トップ/残り) x 日足レジーム(↑/↓)\n{'#'*78}")
    print(f"  (この2x2は3軸すべて有効な部分集合 n={len(df)} 上で計算。パートBの日足単独表(n={len(netR_A)})"
          f"とは母集団が {n_drop} 本だけ異なる)")
    print(f"  {'':<22}{'n':>6}{'本/年':>7}{'win%':>8}{'PF':>8}{'meanR':>9}")
    for tlabel, tmask in (("トップ20%", top_mask), ("残り80%", ~top_mask)):
        for dlabel, dflag in (("日足↑", False), ("日足↓", True)):
            m = tmask & (df["down"].values == dflag)
            st = stats_of(df.loc[m, "R"].values)
            thin = "  ※n薄い、参考値" if st["n"] < 15 else ""
            win_s = f"{st['win']:.1f}%" if st["n"] else "  ·  "
            meanR_s = f"{st['meanR']:+.3f}" if st["n"] else "   ·   "
            label = f"{tlabel}×{dlabel}"
            print(f"  {label:<22}{st['n']:>6}{st['n']/span_years:>7.1f}"
                  f"{win_s:>8}{pf_str(st):>8}{meanR_s:>9}{thin}")

    print(f"\n実行コマンド: .venv/bin/python experiments/strength_btc15mA.py"
          f"{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
