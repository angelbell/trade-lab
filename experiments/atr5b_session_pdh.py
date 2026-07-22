"""(b) アジア時間 / 前日高値フィルタの検定（生き残った2軸だけを検定する・STEP1は済み扱い）。

対象4方向設定（凍結アンカー: fill_win=200, fwd=20, A系, cost=0.0005）:
  long k2.0 RR3 / long k1.5 RR4.5 / short k2.0 RR3 pf0.5 / short k1.5 RR4.5 pf0.382

軸1 セッション: ブローカー時刻 0-7時（引き金足s自身の時刻）。
軸2 pdh_dist = (close[s]-前日高値)/ATR[s-1]（ショートも同じ式を反転フレームへ適用＝mirror規約）。
   閾値は事前に決めた丸い絶対値 {>-1.0, >-0.5, >0} のみ使う（パーセンタイルで決めない）。
   加えて「前半で凍結して後半に当てる」版（前半の下位20%点を閾値として固定→後半に適用）も出す。

検定: ランダム間引き帰無400回（PF・平均%）／年別PF・N（セッション軸は年別安定性を合否の必須条件と
する）／巡回ブロック・ブートストラップ(1,3,6,12か月)／2軸の冗長性(重なり率・順位相関)。

SCREEN = "atr_spike_btc_h1"
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as spstats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atr5_common import (load_frames, atr_prev_of, raw_triggers, build_entries, run_cell,
                          stats, span_years, fmt_row, drop_null, block_bootstrap,
                          session_hour, build_pdh_dist_series, check_no_lookahead_pdh,
                          DIRECTIONS)  # noqa: E402

THRESH = [-1.0, -0.5, 0.0]


def entries_for_mask(d, atr_prev, s_idx, mask, system, rr):
    return build_entries(d, atr_prev, s_idx[mask], system, rr)


def report_cell(label, d, entries, pf, fwd, C, years, pool_pct=None, do_null=True,
                 do_peryear=True, do_bb=True):
    t = run_cell(d, entries, pf=pf, fill_win=200, fwd=fwd, C=C)
    if t is None or len(t) == 0:
        print(f"  {label:<44} (約定0件)")
        return None
    s = stats(t, years)
    null_pf = null_mean = None
    if do_null and pool_pct is not None:
        nl = drop_null(pool_pct, s["N"], t["pnl_pct"].mean(), s["PF"])
        null_pf, null_mean = nl["pf_pct"], nl["mean_pct"]
    print("  " + fmt_row(label, s, null_pf, null_mean))
    if do_peryear:
        yr = t.assign(y=t["time"].dt.year).groupby("y")["pnl_pct"].agg(
            N="size", PF=lambda x: (x[x > 0].sum() / -x[x <= 0].sum()) if (x <= 0).sum() and -x[x <= 0].sum() > 0 else np.nan,
            mean=lambda x: x.mean() * 100)
        pos_years = int((yr["mean"] > 0).sum())
        print(f"      年別: " + " ".join(f"{int(y)}:N{int(r.N)}/PF{r.PF:.2f}/avg{r['mean']:+.2f}%"
                                          for y, r in yr.iterrows()) +
              f"   [プラス年 {pos_years}/{len(yr)}]")
    if do_bb and s["N"] >= 20:
        bbm = block_bootstrap(t, [1, 3, 6, 12], metric="mean")
        print("      ブロックbootstrap(平均%): " +
              " / ".join(f"{k}mo中央値{v[0]:+.3f}%[{v[1]:+.3f},{v[2]:+.3f}]" for k, v in bbm.items()))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()

    df, inv, C = load_frames()
    if cli.smoke:
        df = df.loc[:"2021-12-31"]
        inv = inv.loc[:"2021-12-31"]
    years = span_years(df)

    # ---------------- 先読み検査 ----------------
    print("=" * 100)
    print("[検算] pdh_dist の先読み検査: 末尾を切り落としても過去の値が変わらないこと")
    print("=" * 100)
    atr_prev0 = atr_prev_of(df)
    s_idx0 = raw_triggers(df, atr_prev0, 2.0)
    ok, n_checked = check_no_lookahead_pdh(df, atr_prev0, s_idx0)
    print(f"  一致={ok}  検査本数={n_checked}")
    assert ok, "pdh_dist が先読みしている"
    assert n_checked > 1000, n_checked
    print("  OK: 先読み無し")

    for dcfg in DIRECTIONS:
        k, side, rr, pf = dcfg["k"], dcfg["side"], dcfg["rr"], dcfg["pf"]
        d = df if side == "long" else inv
        Cx = None if side == "long" else C
        atr_prev = atr_prev_of(d)
        s_idx = raw_triggers(d, atr_prev, k)
        hours = session_hour(d, s_idx)
        pdh_full = build_pdh_dist_series(d, atr_prev)
        pdh_s = pdh_full[s_idx]

        print("\n" + "=" * 112)
        print(f"##### {dcfg['name']}  (side={side} k={k} RR={rr} pf={pf}, A系/fwd20/fill_win200/cost0.0005) #####")
        print("=" * 112)

        base_ent = entries_for_mask(d, atr_prev, s_idx, np.ones(len(s_idx), bool), "A", rr)
        t_base = run_cell(d, base_ent, pf=pf, fill_win=200, fwd=20, C=Cx)
        pool_pct = t_base["pnl_pct"].to_numpy()
        report_cell("基準(フィルタ無し)", d, base_ent, pf, 20, Cx, years, pool_pct=None,
                     do_null=False, do_bb=False)

        # ---------------- 軸1: セッション ----------------
        print("\n  -- 軸1: セッション（ブローカー時刻 0-7時=アジア） --")
        m_asia = (hours >= 0) & (hours < 8)
        ent_on = entries_for_mask(d, atr_prev, s_idx, m_asia, "A", rr)
        ent_off = entries_for_mask(d, atr_prev, s_idx, ~m_asia, "A", rr)
        report_cell(f"セッションON  (アジア0-7時, {m_asia.mean()*100:.1f}%が該当)", d, ent_on, pf,
                    20, Cx, years, pool_pct=pool_pct)
        report_cell("セッションOFF (それ以外)", d, ent_off, pf, 20, Cx, years, pool_pct=pool_pct,
                    do_bb=False)

        # ---------------- 軸2: pdh_dist（丸い絶対閾値のみ・事前登録） ----------------
        print("\n  -- 軸2: pdh_dist（前日高値からの距離/ATR、丸い絶対閾値・事前登録） --")
        for th in THRESH:
            m_pdh = pdh_s > th
            ent_pdh = entries_for_mask(d, atr_prev, s_idx, m_pdh, "A", rr)
            report_cell(f"pdh_dist > {th:+.1f} ({m_pdh.mean()*100:.1f}%が該当)", d, ent_pdh, pf,
                        20, Cx, years, pool_pct=pool_pct)

        # 前半で凍結して後半に当てる版（下位20%点=第1findingの「最下位5分位」相当を前半だけで決める）
        half = len(s_idx) // 2
        cut_val = float(np.nanpercentile(pdh_s[:half], 20))
        m_half_full = pdh_s > cut_val
        ent_half_2nd = entries_for_mask(d, atr_prev, s_idx[half:], m_half_full[half:], "A", rr)
        print(f"\n  -- 前半凍結版: 前半(n={half})の下位20%点={cut_val:+.3f} を後半(n={len(s_idx)-half})に適用 --")
        report_cell(f"後半のみ pdh_dist > {cut_val:+.3f} (前半で凍結)", d,
                    ent_half_2nd, pf, 20, Cx, years, pool_pct=None, do_null=False, do_bb=False)

        # ---------------- 組み合わせ ----------------
        print("\n  -- 2軸の組み合わせ（セッションON かつ pdh_dist>0） --")
        m_comb = m_asia & (pdh_s > 0.0)
        ent_comb = entries_for_mask(d, atr_prev, s_idx, m_comb, "A", rr)
        report_cell(f"セッションON ∩ pdh_dist>0 ({m_comb.mean()*100:.1f}%)", d, ent_comb, pf,
                    20, Cx, years, pool_pct=pool_pct)

        # ---------------- 冗長性 ----------------
        m_pdh0 = pdh_s > 0.0
        n_both = int((m_asia & m_pdh0).sum())
        n_asia = int(m_asia.sum()); n_pdh0 = int(m_pdh0.sum())
        jacc = n_both / max(1, len(set(np.flatnonzero(m_asia)) | set(np.flatnonzero(m_pdh0))))
        rho, pval = spstats.spearmanr(hours.astype(float), pdh_s)
        print(f"\n  -- 冗長性: セッションON n={n_asia}, pdh_dist>0 n={n_pdh0}, "
              f"両方 n={n_both} (Jaccard={jacc:.3f})")
        print(f"     時刻 vs pdh_dist の順位相関 Spearman={rho:+.3f} (p={pval:.3f})")

    print(f"\n実行コマンド: .venv/bin/python experiments/atr5b_session_pdh.py{' --smoke' if cli.smoke else ''}")


if __name__ == "__main__":
    main()
