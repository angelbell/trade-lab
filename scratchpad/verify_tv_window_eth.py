"""TradingView(ETHUSD Vantage 1H, 2025-01-01〜2026-07-22, 0.01枚)との突き合わせ。

TV 実測: 総損益 +4.80 USD / 最大DD 5.31 USD / 勝率 50.00% (30/60) / PF 1.308
"""
SCREEN = "atr_spike_btc_h1"

import numpy as np
import pandas as pd

d = pd.read_csv("scratchpad/crypto_1h_atr_spike_trades.csv", encoding="utf-8-sig")
d["入口時刻"] = pd.to_datetime(d["入口時刻"])
QTY = 0.01

for sym, tv in (("ETHUSD", "TV: N=60 勝率50.0% PF1.308 総+4.80USD"),
                ("BTCUSD", "TV: PF0.894 総-46.44USD")):
    v = d[(d["フィード"] == "Vantage") & (d["銘柄"] == sym)]
    x = v[v["入口時刻"] >= "2025-01-01"]
    p = x["損益(価格)"].to_numpy()
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p * QTY)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    print(f"{sym}  {tv}")
    print(f"        こちら: N={len(x)} 勝率={(p>0).mean()*100:5.1f}% PF={w/l:6.3f} "
          f"総={p.sum()*QTY:+7.2f}USD 最大DD={dd:5.2f}USD")
    yr = x.groupby(x["入口時刻"].dt.year)["損益(価格)"].agg(["size", "sum"])
    print("        年別: " + "  ".join(
        f"{y}:n{int(r['size'])}/{r['sum']*QTY:+.2f}USD" for y, r in yr.iterrows()))

# 検算: ETH の同窓が TV と同じ向き（PF>1）で、件数が近いこと
e = d[(d["フィード"] == "Vantage") & (d["銘柄"] == "ETHUSD")]
e = e[e["入口時刻"] >= "2025-01-01"]
pe = e["損益(価格)"].to_numpy()
pf_e = pe[pe > 0].sum() / -pe[pe < 0].sum()
assert pf_e > 1.0, pf_e
assert 50 <= len(e) <= 75, len(e)
print(f"\nOK: ETH 同窓 N={len(e)} PF={pf_e:.3f}（TV: N=60 PF1.308）")
