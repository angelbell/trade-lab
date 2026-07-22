"""銘柄ホールドアウト: ERゲートの符号反転が ETH でも再現するか。

BTC で見つけたこと（窓120・拡張窓の分位・先読みなし・実スプレッド）:
  ロング  ER【高】で効く（%ile 97.2、方向つきに割ると ER高＋上昇が %ile99.6）
  ショート ER【低】で効く（%ile 96.2、ロングのゲートを当てると %ile 7.2）
機構が説明できていないので、銘柄ホールドアウトが最も安くて効く検定。

ETH の条件:
  ロング  = BTC同時確認つき（既に確立した仕様）
  ショート = 素（BTC同時確認はショートに効かないと確認済み）
ゲート変数は2通り試す: ETH自身のER と BTCのER（BTCが先導する市場なので）。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from scratchpad.atr_spike_short_barspread import run as run_short   # noqa: E402
from scratchpad.atr_spike_barspread import spread_series, leg, spikes  # noqa: E402
from scratchpad.atr_spike_er_gate import er_series                  # noqa: E402
from scratchpad.atr_spike_er_short import dropnull                  # noqa: E402

USDJPY, LOT = 150.0, 0.01
WARM = pd.Timedelta(days=365)
WIN = 120


def load(sym):
    d = load_mt5_csv(f"data/vantage_{sym}_h1.csv").loc["2022-01-01":]
    return d[~d.index.duplicated(keep="first")].sort_index()


def attach(t, e_own, e_btc, e_px_real):
    t = t.copy()
    t["e_px_real"] = e_px_real
    t["er_own"] = e_own.reindex(t["time"]).to_numpy()
    t["er_btc"] = e_btc.reindex(t["time"]).to_numpy()
    t = t.dropna(subset=["er_own", "er_btc"]).sort_values("time").reset_index(drop=True)
    for col in ("er_own", "er_btc"):
        for q in (0.33, 0.50, 0.67):
            t[f"{col}_x{int(q*100)}"] = (t[col].expanding(min_periods=20)
                                         .quantile(q).shift(1))
    t = t[t["time"] >= t["time"].iloc[0] + WARM].copy()
    t["yen"] = t["pct"].to_numpy() * t["e_px_real"].to_numpy() * LOT * USDJPY
    return t


def rep(lab, g, span, allp=None, allr=None, mask=None):
    p, r, y = g["pct"].to_numpy(), g["R_net"].to_numpy(), g["yen"].to_numpy()
    eq = np.cumsum(y)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    tag = ""
    if mask is not None:
        dn = dropnull(allp, allr, mask)
        if dn:
            tag = f"  帰無%ile 平均={dn[0]:5.1f} PF={dn[1]:5.1f}"
    print(f"  {lab:<30} N={len(p):4d} 年{len(p)/span:3.0f}本 勝率={(p>0).mean()*100:5.1f}% "
          f"PF={w/ls if ls>0 else np.nan:5.2f} 1本R={r.mean():+.3f} "
          f"年間={y.sum()/span:+8,.0f}円 DD={dd:7,.0f}円{tag}")


if __name__ == "__main__":
    btc, eth = load("btcusd"), load("ethusd")
    spB, spE = spread_series("BTCUSD|h1"), spread_series("ETHUSD|h1")
    e_btc = er_series(btc["close"], WIN)
    e_eth = er_series(eth["close"], WIN)
    sB = spikes(btc)
    lead = (sB | sB.shift(1)).fillna(False)
    C_eth = 2 * eth["high"].max()

    tl = leg(eth, 20, cost_series=spE, lead=lead)
    tl["R_net"] = tl["pct"] / tl["rf"]
    L = attach(tl, e_eth, e_btc, tl["e_px"].to_numpy())
    ts = run_short(eth, 0.0, cost_series=spE)
    S = attach(ts, e_eth, e_btc, (C_eth - ts["e_px"]).to_numpy())

    for nm, T, want_high in (("ETH ロング（BTC同時確認つき）", L, True),
                             ("ETH ショート（素）", S, False)):
        span = (T["time"].max() - T["time"].min()).days / 365.25
        allp, allr = T["pct"].to_numpy(), T["R_net"].to_numpy()
        print(f"\n===== {nm}  助走後 {span:.1f}年")
        rep("全部取る", T, span)
        for col, cl in (("er_own", "ETH自身のER"), ("er_btc", "BTCのER")):
            q = T[col].quantile([1/3, 2/3]).to_numpy()
            print(f"  -- {cl} の3分位")
            for lo, hi, lb in ((-np.inf, q[0], "低"), (q[0], q[1], "中"), (q[1], np.inf, "高")):
                g = T[(T[col] >= lo) & (T[col] < hi)]
                if len(g) >= 12:
                    rep(f"     {lb}", g, span)
            # ゲート（先読みなし）
            for qq in (33, 50, 67):
                thr = T[f"{col}_x{qq}"].to_numpy()
                m = (T[col].to_numpy() >= thr) if want_high else (T[col].to_numpy() < thr)
                if m.sum() < 15:
                    continue
                d = "上回る" if want_high else "下回る"
                rep(f"     ゲート {cl} {qq}%点を{d}", T[m], span, allp, allr, m)

    print("\n（判定: BTC と同じ符号——ロングは高ER・ショートは低ER——が ETH でも "
          "\n  帰無%ile 95 を超えて再現すれば、機構が説明できなくても信頼度は上がる）")
    assert len(L) > 50 and len(S) > 50, (len(L), len(S))
    print(f"\nOK: ETH ロング N={len(L)} / ショート N={len(S)}")
