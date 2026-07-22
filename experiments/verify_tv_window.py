"""TradingView の Strategy Tester と同じ窓で、こちらのトレード一覧を切って突き合わせる。

TV 側: BTCUSD(Vantage) 1時間・2025-01-01〜2026-07-22・0.01枚・PF 0.894・総損益 -46.44 USD。
同じ窓でこちらの CSV を集計し、実装差なのか単に「その期間が弱いだけ」なのかを分ける。
"""
SCREEN = "atr_spike_btc_h1"

import pandas as pd

d = pd.read_csv("experiments/crypto_1h_atr_spike_trades.csv", encoding="utf-8-sig")
d["入口時刻"] = pd.to_datetime(d["入口時刻"])
v = d[(d["フィード"] == "Vantage") & (d["銘柄"] == "BTCUSD")].copy()

QTY = 0.01          # TV と同じ枚数


def block(lo, hi, lab):
    x = v[(v["入口時刻"] >= lo) & (v["入口時刻"] <= hi)]
    p = x["損益(価格)"].to_numpy()
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    pf = w / l if l > 0 else float("nan")
    print(f"{lab:30s} N={len(x):4d} 勝率={(p>0).mean()*100:5.1f}% PF={pf:6.3f} "
          f"0.01枚の総損益=${p.sum()*QTY:+8.2f}")
    yr = x.groupby(x["入口時刻"].dt.year)["損益(価格)"].agg(["size", "sum"])
    print("      年別: " + "  ".join(
        f"{y}:n{int(r['size'])}/${r['sum']*QTY:+.1f}" for y, r in yr.iterrows()))
    return pf, len(x), p.sum() * QTY


pf_all, n_all, _ = block("2022-01-01", "2026-12-31", "2022-01-01以降（全体）")
pf_tv, n_tv, usd_tv = block("2025-01-01", "2026-12-31", "2025-01-01以降（TVと同じ窓）")
pf_old, _, _ = block("2022-01-01", "2024-12-31", "2022-2024")

# 検算: 全体は既知の PF1.69 / N=207
assert 200 <= n_all <= 215, n_all
assert 1.65 < pf_all < 1.75, pf_all
# TV と同じ窓では PF が 1 を割っていること（＝TVの0.894と同じ向き）
assert pf_tv < 1.0, pf_tv
print(f"\nOK: 全体 N={n_all} PF={pf_all:.2f} を再現。TVと同じ窓は PF={pf_tv:.3f}（TV実測 0.894）")
