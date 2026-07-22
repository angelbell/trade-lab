"""ICT の「流動性の狩り」を、既存の検証済みレッグの前置条件として足す（2026-07-15）。

ICT 調査から生き残った唯一の本物 = 「流動性の狩り + MSS」。
既存レッグ（gold15m / btc15m_L / btc15m_S）は **MSS を既に持っている**
（Pattern-B の「スイング高値 H1 を終値でブレイク」＝市場構造の転換そのもの）。
欠けているのは **狩り** の方: Pattern-B は `pL2 > pL0`（高値切り上げの押し目）を要求するので、
「押し目が前の安値を割る」ことは構造的に排他。
→ 嵌める余地は「**押し目の安値 pL2 が、別の流動性プール（前日安値/アジア安値/直近3日安値）を
   一度割ってから、H1 をブレイクして戻したか**」という前置条件。

pL2 は run() の出力から復元できる（sl_mode=swinglow なので stop = pL2、risk = e_px - stop）。
∴ breakout_wave 本体は触らない。

審判（CLAUDE.md）:
  - レッグ単体（n, PF, meanR, CAGR/DD）だけでなく **ブックの CAGR/DD（トレード解像度）** で裁定
  - 🚨 inv-vol 重みなので、ばらつきが下がると自動でレバレッジを買う（過去3件撤回）
    → **重みを現行に固定した版**を必ず並べる。固定して差が消えたら、それはレバレッジ・ダイヤル。
  - フィルタは N を削るので **ランダム間引き帰無**（同じ本数をランダムに残す）と比較

Run: .venv/bin/python experiments/ict_sweep_on_book.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from research.regime_gate_lab import CFG
from radar_gate_race import BASE
from short_mirror_15m import invert
from book_spec_fix import book, w_trade, cdd
from book_deployed_spec import build, SIX

ROOT = "/home/angelbell/dev/auto-trade"
RNG = np.random.default_rng(20260715)


def pools(df):
    """エントリー時点で既知の流動性プール（前日/前3日/前週の安値・高値、アジア窓の安値・高値）。
    すべて『前日までに確定』したものだけを使う（先読み厳禁）。"""
    d = pd.DataFrame(index=df.index)
    lo1 = df["low"].resample("1D").min().dropna()
    hi1 = df["high"].resample("1D").max().dropna()
    for k, w in (("d1", 1), ("d3", 3), ("d5", 5)):
        d[f"pl_{k}"] = lo1.rolling(w).min().shift(1).reindex(df.index, method="ffill")
        d[f"ph_{k}"] = hi1.rolling(w).max().shift(1).reindex(df.index, method="ffill")
    # アジア窓 = ブローカー時刻 02:00-10:00（東京の日中。ブローカー=EET なので JST-7/-6h）
    h = df.index.hour
    asia = df[(h >= 2) & (h < 10)]
    al = asia["low"].resample("1D").min()
    ah = asia["high"].resample("1D").max()
    d["pl_asia"] = al.reindex(df.index, method="ffill")     # 当日のアジア窓（当日中は確定済み）
    d["ph_asia"] = ah.reindex(df.index, method="ffill")
    return d


def swept_mask(t, df, P, key, side):
    """押し目の安値 pL2 = e_px - risk が、プール `key` を割っていたか（ロング）。
    ショートは鏡像（pH2 = e_px + risk がプールを上抜けていたか）。"""
    idx = df.index.get_indexer(pd.DatetimeIndex(t["time"]))
    if side == "long":
        l2 = t["e_px"].values - t["risk"].values
        pool = P[f"pl_{key}"].values[idx]
        return np.isfinite(pool) & (l2 < pool)
    else:
        h2 = t["e_px"].values + t["risk"].values
        pool = P[f"ph_{key}"].values[idx]
        return np.isfinite(pool) & (h2 > pool)


def leg_stats(s, risk=0.01):
    """レッグ単体の CAGR/DD。cdd() は cumprod(1+vals) なので、賭け率を掛けてから渡す。"""
    if len(s) < 5:
        return None
    y = (s.index[-1] - s.index[0]).days / 365.25
    pf = s[s > 0].sum() / abs(s[s <= 0].sum()) if (s <= 0).any() else np.inf
    c, dd, r = cdd(s.values * risk, (s.index[-1] - s.index[0]).days)
    return dict(n=len(s), npy=len(s) / y, pf=pf, mean=s.mean(), cagr=c, dd=dd, cdd=r)


def book_fixed_w(legs, basket, w):
    """重みを外から与えて（＝現行に固定して）ブックを組む。レバレッジの混入を排除する。"""
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = []
    for k in basket:
        s = legs[k][(legs[k].index >= st) & (legs[k].index <= en)]
        parts.append(pd.Series(s.values * w[k], index=s.index))
    s = pd.concat(parts).sort_index()
    return cdd(s.values, (s.index[-1] - s.index[0]).days) + (len(s),)


def main():
    print("実運用仕様の6レッグを組み立て中（fill_win=200 / btc15m_S の RR=4.5）...")
    legs0 = build(200, 4.5)
    c0, d0, r0, n0 = book(legs0, SIX)
    w0 = w_trade(legs0, SIX)
    print(f"ベース・ブック: n={n0}  CAGR {c0:+.1f}%  maxDD {d0:.2f}%  **CAGR/DD {r0:.2f}**\n")

    # レッグを作り直せるように、各15分レッグの生トレード表とデータを再取得
    with contextlib.redirect_stderr(io.StringIO()):
        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        tg = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                         "ext_cap": 8.0, "pullback_frac": 0.25, "fill_win": 200}))
        Rg = pd.Series(tg["R"].values - 0.3 / tg["risk"].values, index=pd.DatetimeIndex(tg["time"]))
        Pg = pools(g15)

        b15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        tL = run(b15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                         "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
        pdh = b15["high"].resample("1D").max().dropna().shift(1).reindex(b15.index, method="ffill").values
        wgt = np.where(tL["e_px"].values > pdh[b15.index.get_indexer(tL["time"])], 1.0, 0.5)
        RL = pd.Series((tL["R"].values - 15.0 / tL["risk"].values) * wgt,
                       index=pd.DatetimeIndex(tL["time"]))
        Pb = pools(b15)

        inv = invert(b15); C = 2 * b15["high"].max()
        tS = run(inv, SimpleNamespace(**{**BASE, "gate_kama": 14, "pullback_frac": 0.3,
                                         "rr": 4.5, "fill_win": 200}))
        pdl = b15["low"].resample("1D").min().dropna().shift(1).reindex(b15.index, method="ffill").values
        mS = (C - tS["e_px"].values) < pdl[b15.index.get_indexer(tS["time"])]
        RS = pd.Series((tS["R"].values - 15.0 / tS["risk"].values)[mS],
                       index=pd.DatetimeIndex(tS["time"])[mS])
        # ショートは反転系列上で走っているので、狩り判定も反転系列側のプールで行う
        Pinv = pools(inv)

    CELLS = [
        ("gold15m", tg, g15, Pg, "long", Rg),
        ("btc15m_L", tL, b15, Pb, "long", RL),
        ("btc15m_S", tS, inv, Pinv, "long", RS),   # 反転系列上ではショートもロングとして走る
    ]
    KEYS = [("d1", "前日安値"), ("d3", "直近3日安値"), ("d5", "直近5日安値"), ("asia", "アジア窓安値")]

    print("=" * 126)
    print("狩りの前置条件（押し目安値がプールを割ってから H1 ブレイクで戻した）を足す")
    print("  🚨 inv-vol 重みは σ が下がると自動でレバレッジを買う → 「重み固定」列を必ず見ること")
    print("=" * 126)
    for name, t, df, P, side, R0 in CELLS:
        s0 = leg_stats(R0)
        print(f"\n--- {name} ---  ベース: n={s0['n']} 年{s0['npy']:.0f}本 PF={s0['pf']:.2f} "
              f"meanR={s0['mean']:+.3f} レッグCAGR/DD={s0['cdd']:.2f}")
        print(f"  {'条件':22s} {'残':>5} {'割合':>5} {'PF':>6} {'meanR':>8} {'脚C/DD':>7} | "
              f"{'ブックCAGR/DD':>13} {'差':>7} | {'重み固定':>9} {'差':>7} | {'帰無%ile':>8}")
        for key, klab in KEYS:
            m = swept_mask(t, df, P, key, side)
            if name == "btc15m_S":
                m = m[mS]                          # ショートは PDL フィルタ後の部分集合
            for tag, mask in (("狩り あり", m), ("狩り なし(対照)", ~m)):
                Rf = R0[mask]
                if len(Rf) < 20:
                    continue
                sf = leg_stats(Rf)
                legs = dict(legs0); legs[name] = Rf
                c, d, r, _ = book(legs, SIX)                      # 重み自由（＝レバレッジ混入）
                cf, df_, rf, _ = book_fixed_w(legs, SIX, w0)      # 重み固定
                # ランダム間引き帰無（同じ本数をランダムに残す）— 重み固定で比較
                draws = []
                for _ in range(300):
                    ii = np.sort(RNG.choice(len(R0), size=len(Rf), replace=False))
                    lg = dict(legs0); lg[name] = R0.iloc[ii]
                    draws.append(book_fixed_w(lg, SIX, w0)[2])
                pc = 100 * (rf > np.array(draws)).mean()
                star = " *" if pc >= 90 else ""
                print(f"  {klab+' '+tag:22s} {sf['n']:5d} {100*len(Rf)/len(R0):4.0f}% "
                      f"{sf['pf']:6.2f} {sf['mean']:+8.3f} {sf['cdd']:7.2f} | "
                      f"{r:13.2f} {r-r0:+7.2f} | {rf:9.2f} {rf-r0:+7.2f} | {pc:7.0f}%{star}")


if __name__ == "__main__":
    main()
