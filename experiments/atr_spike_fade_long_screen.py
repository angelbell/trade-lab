"""ATR拡大足の「陰線フェードロング」— 入口スクリーン段階（凍結仕様、まだRR/出口は決めない）。

構造法則11（BTCには上昇ドリフトがあり、下向きATR拡大足の「継続ショート」は全滅済み＝
上昇相場の押し目を売っている＝反発を叩いている）を逆手に取る。同じ引き金（下向きATR拡大足）
でロング（フェード）に入れば入口エッジがあるか、という仮説の**巡行幅スクリーンだけ**を測る。
tgt/stop/RRの最適化は一切しない（CLAUDE.md バウンス検証の順序＝反発率→選別可否→巡行幅→
やっとRR、を厳守）。

引き金（先読み禁止）: body=|close-open|、body > ATR14[s-1]×k、close<open（陰線）。
  ATRは確定した1本前[s-1]で評価。k を 1.0/1.5/2.0 で掃引。方向=ロング。
  エントリーは引き金足の**次足始値**（確定足）。

「meanR」の定義（このスクリプト固有の選択、要確認）:
  まだ損切り/目標を決めない段階なので、ストップ/トレール等の出口ルールを一切使わず、
  「fwd本後の終値时点の含み損益」を引き金足のATR[s-1]単位で正規化した**終端の水平線リターン**
  だけを「meanR」として扱う（walk()のトレール執行は使わない＝出口ルールの先取りになるため）。
  MFE/MAEの「比」を判定の主に置き、「meanR」はER層別の分離を見るための補助指標。

コストはまだ混ぜない（仕様どおり素の率×幅を先に見る＝判定順の規律）。

流用（車輪の再発明はしない）:
  wilder_atr           <- experiments/atr_spike_barspread.py
  prep (Vantage 2022-)  <- experiments/atr_spike_2025.py
  load (Binance 2018-)  <- experiments/atr_spike_er_binance.py
  er_series(erLen=120)  <- experiments/atr_spike_er_gate.py
  dropnull（間引き帰無） <- experiments/atr_spike_er_short.py
  build（Spike Rider母集団、被り検算用）<- experiments/atr_spike_er_binance.py
  run_screen（MFE/MAE比の正規ツール・フック要件）<- research/screen.py
"""
SCREEN = "atr_spike_fade_long_binance_k1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_barspread import wilder_atr           # noqa: E402
from experiments.atr_spike_2025 import prep as prep_vantage      # noqa: E402
from experiments.atr_spike_er_binance import load as load_binance, build as build_spike_rider, COST  # noqa: E402
from experiments.atr_spike_er_gate import er_series               # noqa: E402
from experiments.atr_spike_er_short import dropnull                # noqa: E402
from research.screen import run_screen                             # noqa: E402

K_LIST = (1.0, 1.5, 2.0)
FWD_LIST = (20, 80)
ERWIN = 120
NNULL = 2000
RNG = np.random.default_rng(9101)


# ---------------------------------------------------------------- 引き金
def bearish_pool(d):
    """k=0 母集団: 陰線かつATR確定＝候補プール。k掃引はこの部分集合。"""
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    ap = wilder_atr(d).shift(1).to_numpy()
    m = (c < o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(m)
    s = s[s + 1 < len(d)]
    return s, ap


def k_mask(d, s, ap, k):
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    body = o[s] - c[s]
    return body > ap[s] * k


# ---------------------------------------------------------- 巡行幅（ATR単位・水平線終端リターン）
def excursion(d, s, ap, fwd):
    """s の各引き金足について、次足始値ロングのMFE/MAE（ATR[s-1]単位）と、
    fwd本後の終値時点の含み損益（＝出口ルール無しの水平線終端リターン、同じATR単位）。"""
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    mfe, mae, term = [], [], []
    for i in s:
        a = ap[i]
        e = o[i + 1]
        fh, fl = h[i + 1:i + 1 + fwd], l[i + 1:i + 1 + fwd]
        if len(fh) == 0:
            continue
        mfe.append((fh.max() - e) / a)
        mae.append((e - fl.min()) / a)
        j = min(i + fwd, len(d) - 1)
        term.append((c[j] - e) / a)
    return np.array(mfe), np.array(mae), np.array(term)


def qtab(x):
    return dict(n=len(x), mean=x.mean(), median=np.median(x), std=x.std(ddof=1),
                q25=np.quantile(x, 0.25), q75=np.quantile(x, 0.75))


def pf_of(term):
    w, ls = term[term > 0].sum(), -term[term < 0].sum()
    return w / ls if ls > 0 else np.nan


def annualize(n, d):
    span = (d.index[-1] - d.index[0]).days / 365.25
    return n / span


if __name__ == "__main__":
    SMOKE = "--smoke" in sys.argv
    FEEDS = {"Binance": load_binance("btcusdt"),
             "Vantage": prep_vantage("btcusd")}
    if SMOKE:
        FEEDS = {k: v.loc[v.index[-6000]:] for k, v in FEEDS.items()}

    # ============================================================ 1. 巡行幅スクリーン
    print("=" * 100)
    print("1. 巡行幅スクリーン（陰線フェードロング・ATR単位・出口ルール無し）")
    print("=" * 100)

    tie_back = {}   # (feed, k, fwd) -> (n, ratio_mean, pf)  検算用に主要セルを保存
    pools = {}   # feed -> (s0, ap)
    for feed, d in FEEDS.items():
        s0, ap = bearish_pool(d)
        pools[feed] = (s0, ap)
        span_n = annualize(len(s0), d)
        print(f"\n--- {feed}  期間 {d.index[0].date()}〜{d.index[-1].date()} "
              f"陰線プール(k=0) N={len(s0)} (年{span_n:.0f}本)")
        print(f"  {'k':>4} {'fwd':>4} {'N':>6} {'年N':>6} | "
              f"{'MFE 平均/中央/σ':>22} | {'|MAE| 平均/中央/σ':>22} | "
              f"{'比(平均)':>8} {'比(中央値)':>10} | {'meanR 平均/σ':>16} {'PF':>6} {'勝率':>6}")
        for k in K_LIST:
            m = k_mask(d, s0, ap, k)
            sk = s0[m]
            for fwd in FWD_LIST:
                mfe, mae, term = excursion(d, sk, ap, fwd)
                if len(mfe) < 10:
                    print(f"  {k:>4.1f} {fwd:>4} 本数不足(N={len(mfe)})")
                    continue
                qm, qa = qtab(mfe), qtab(np.abs(mae))
                ratio_mean = qm["mean"] / qa["mean"] if qa["mean"] > 0 else np.nan
                ratio_med = qm["median"] / qa["median"] if qa["median"] > 0 else np.nan
                pf = pf_of(term)
                win = (term > 0).mean() * 100
                tie_back[(feed, k, fwd)] = (len(mfe), ratio_mean, pf)
                print(f"  {k:>4.1f} {fwd:>4} {len(mfe):>6} {annualize(len(mfe), d):>6.0f} | "
                      f"{qm['mean']:>7.2f}/{qm['median']:>6.2f}/{qm['std']:>6.2f} | "
                      f"{qa['mean']:>7.2f}/{qa['median']:>6.2f}/{qa['std']:>6.2f} | "
                      f"{ratio_mean:>8.2f} {ratio_med:>10.2f} | "
                      f"{term.mean():>+7.2f}/{term.std(ddof=1):>6.2f} {pf:>6.2f} {win:>5.1f}%")

    # research/screen.py の正規ツールでも1本立てて回す（フック要件・分位25/75も見る）
    dB = FEEDS["Binance"]
    s0B, apB = pools["Binance"]
    mB = k_mask(dB, s0B, apB, 1.0)
    skB = s0B[mB]
    oB, lB = dB["open"].to_numpy(), dB["low"].to_numpy()
    entries_B = [(dB.index[i + 1], +1, oB[i + 1], oB[i + 1] - apB[i]) for i in skB]
    run_screen("atr_spike_fade_long_binance_k1", dB, entries_B, windows=[1200, 4800])

    dV = FEEDS["Vantage"]
    s0V, apV = pools["Vantage"]
    mV = k_mask(dV, s0V, apV, 1.0)
    skV = s0V[mV]
    oV, lV = dV["open"].to_numpy(), dV["low"].to_numpy()
    entries_V = [(dV.index[i + 1], +1, oV[i + 1], oV[i + 1] - apV[i]) for i in skV]
    run_screen("atr_spike_fade_long_vantage2022_k1", dV, entries_V, windows=[1200, 4800])

    # ============================================================ 2. ER層別
    print("\n" + "=" * 100)
    print(f"2. ER層別（erLen={ERWIN}、フルサンプル中央値で高/低・fwd=20固定）")
    print("=" * 100)
    for feed, d in FEEDS.items():
        s0, ap = pools[feed]
        er = er_series(d["close"], ERWIN)
        er_at = er.to_numpy()
        print(f"\n--- {feed}")
        for k in K_LIST:
            m = k_mask(d, s0, ap, k)
            sk = s0[m]
            e_sk = er_at[sk]
            valid = np.isfinite(e_sk)
            sk, e_sk = sk[valid], e_sk[valid]
            if len(sk) < 20:
                print(f"  k={k}: 本数不足")
                continue
            med = np.median(e_sk)
            for lab, sel in (("ER低(<中央値)", e_sk < med), ("ER高(>=中央値)", e_sk >= med)):
                ss = sk[sel]
                if len(ss) < 10:
                    print(f"  k={k:.1f} {lab:<14} 本数不足(N={len(ss)})")
                    continue
                mfe, mae, term = excursion(d, ss, ap, 20)
                qm, qa = qtab(mfe), qtab(np.abs(mae))
                ratio = qm["mean"] / qa["mean"] if qa["mean"] > 0 else np.nan
                print(f"  k={k:.1f} {lab:<14} N={len(ss):>5} 比(平均)={ratio:>6.2f} "
                      f"meanR={term.mean():>+.3f}(中央{np.median(term):>+.3f} σ{term.std(ddof=1):.3f}) "
                      f"PF={pf_of(term):>5.2f} 勝率={(term>0).mean()*100:>5.1f}%")

    # ============================================================ 3. 被り検算（Binance限定・Spike Riderと比較）
    print("\n" + "=" * 100)
    print("3. 被り検算 — Spike Rider（陽線継続ロング, Binance BTCUSDT）との別物度")
    print("=" * 100)
    B = FEEDS["Binance"]
    sr = build_spike_rider(B, "long", COST["btcusdt"])   # 採用済み仕様(PDH+weekday filter入り)
    s0B, apB = pools["Binance"]
    mB15 = k_mask(B, s0B, apB, 1.5)
    fade_idx = s0B[mB15]
    fade_entry_t = B.index[fade_idx + 1]
    fade_exit_t = fade_entry_t + pd.to_timedelta(20, unit="h")  # fwd=20固定窓の擬似保有
    sr_entry_t = sr["time"]
    sr_exit_t = sr["time"] + pd.to_timedelta(sr["hold"], unit="D")

    def occupied_frac(entry_a, exit_a, entry_b, exit_b):
        """entry_a の各トレードについて、entry_b/exit_b の【いずれか】と保有窓が重なる割合。
        区間交差 [ea,xa) ∩ [eb,xb) ≠ ∅  <=>  ea<xb かつ xa>eb（総当たり・n×mは高々百万規模で軽い）。"""
        ea = entry_a.to_numpy()[:, None]
        xa = exit_a.to_numpy()[:, None]
        eb = entry_b.to_numpy()[None, :]
        xb = exit_b.to_numpy()[None, :]
        overlap = (ea < xb) & (xa > eb)
        return overlap.any(axis=1).mean() if overlap.size else np.nan

    frac_fade_in_sr = occupied_frac(fade_entry_t, fade_exit_t, sr_entry_t, sr_exit_t)
    frac_sr_in_fade = occupied_frac(sr_entry_t, sr_exit_t, fade_entry_t, fade_exit_t)
    print(f"  フェードロング(k=1.5) N={len(fade_idx)}本 / Spike Rider N={len(sr)}本")
    print(f"  建玉時刻の重なり率: フェード側の {frac_fade_in_sr*100:.1f}% がSpike Riderの保有中に建つ")
    print(f"                    Spike Rider側の {frac_sr_in_fade*100:.1f}% がフェードの保有中に建つ")

    # 同時期リターンの相関（週次合算R）
    _, _, term_fade = excursion(B, fade_idx, apB, 20)
    fade_week = fade_entry_t.isocalendar().week.to_numpy() + fade_entry_t.year.to_numpy() * 100
    fade_wk = pd.Series(term_fade, index=fade_week).groupby(level=0).sum()
    sr_week = (sr["time"].dt.isocalendar().week + sr["time"].dt.year * 100)
    sr_wk = pd.Series(sr["R"].to_numpy(), index=sr_week).groupby(level=0).sum()
    joined = pd.concat([fade_wk.rename("fade"), sr_wk.rename("sr")], axis=1).dropna()
    corr = joined["fade"].corr(joined["sr"]) if len(joined) > 5 else np.nan
    print(f"  週次リターン相関（共通{len(joined)}週）: r={corr:+.3f}")

    # ============================================================ 4. 帰無（間引き）
    print("\n" + "=" * 100)
    print("4. 帰無検定（k=0陰線プールから同本数を無作為抽出×2000、%ile）")
    print("=" * 100)
    for feed, d in FEEDS.items():
        s0, ap = pools[feed]
        print(f"\n--- {feed}")
        for fwd in (20, 80):
            _, _, term0 = excursion(d, s0, ap, fwd)
            for k in K_LIST:
                m = k_mask(d, s0, ap, k)
                # term0 は s0 と同じ順序で構築されているので m がそのままマスクとして使える
                if m.sum() < 12 or len(term0) != len(s0):
                    continue
                dn = dropnull(term0, term0, m)
                if dn is None:
                    continue
                pct_mean, pct_pf = dn
                print(f"  fwd={fwd:>3} k={k:.1f}  N={int(m.sum()):>5}  "
                      f"帰無%ile 平均R={pct_mean:>5.1f}  PF={pct_pf:>5.1f}")

    # ============================================================ 検算（既知値への数値assert）
    n_check = len(pools["Binance"][0][k_mask(FEEDS["Binance"], *pools["Binance"], 1.5)])
    assert n_check > 100, n_check
    if not SMOKE:
        n15, ratio15, pf15 = tie_back[("Binance", 1.5, 20)]
        assert n15 == 1585, n15
        assert 0.75 < ratio15 < 0.81, ratio15
        assert 0.80 < pf15 < 0.88, pf15
        print(f"\nOK(検算): Binance k=1.5 fwd=20  N={n15}  比(平均)={ratio15:.2f}  PF={pf15:.2f}"
              "  ← フルデータ再実行で既知値と一致")
    print(f"\nOK: Binance k=1.5 陰線フェード母集団 N={n_check}")
