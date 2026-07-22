"""crypto_1h_atr_spike.pine と同一仕様のトレード一覧を書き出す。

出口時刻・出口価格は walk() の返り値から復元する（自前ウォーカーは書かない）:
  walk は cost=0 で回すので R = (出口価格 - 入口価格) / risk  → 出口価格 = e_px + R*risk
  hold は日数 → 出口時刻 = 入口時刻 + hold、保有本数 = hold*24（1時間足）
出口理由は保有本数が時間切れ上限に達したかどうかで分ける。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402

K, TRAIL, FWD = 2.0, 3.0, 20


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def trades(path, start, cost_abs, riga):
    d = load_mt5_csv(path)
    if riga:
        idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
        d = d.set_index(idx)
    d = d.loc[start:]
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()

    hit = (c - o > ap * K) & (c > o) & np.isfinite(ap)
    s = np.flatnonzero(hit)
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]                       # 前日高値フィルタ
    s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]             # 入る足が土日なら見送り

    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    sig_time = {d.index[i]: d.index[i - 1] for i in [e[0] + 1 for e in ent]}
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0,
                        trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)

    e_px, risk, R = t["e_px"].to_numpy(), t["risk"].to_numpy(), t["R"].to_numpy()
    bars = np.round(t["hold"].to_numpy() * 24).astype(int)
    out = pd.DataFrame({
        "引き金時刻": [sig_time.get(x, pd.NaT) for x in t["time"]],
        "入口時刻": t["time"].values,
        "入口価格": e_px.round(4),
        "初期損切り": (e_px - risk).round(4),
        "損切り幅": risk.round(4),
        "保有本数": bars,
        "出口時刻": (t["time"] + pd.to_timedelta(t["hold"], unit="D")).values,
        "出口価格": (e_px + R * risk).round(4),
        "出口理由": np.where(bars >= FWD, "時間切れ", "トレール"),
        "損益(価格)": (R * risk - cost_abs).round(4),
        "損益%": ((R * risk - cost_abs) / e_px * 100).round(4),
    })
    out["累積%"] = out["損益%"].cumsum().round(3)
    return out


CASES = [
    ("Vantage", "BTCUSD", "data/vantage_btcusd_h1.csv", "2022-01-01", 15.0, False),
    ("Vantage", "ETHUSD", "data/vantage_ethusd_h1.csv", "2022-01-01", 2.0, False),
    ("Binance", "BTCUSDT", "data/binance_btcusdt_h1.csv", "2018-01-01", 15.0, True),
    ("Binance", "ETHUSDT", "data/binance_ethusdt_h1.csv", "2018-01-01", 2.0, True),
]

frames = []
for feed, sym, path, start, cost, riga in CASES:
    df = trades(path, start, cost, riga)
    df.insert(0, "銘柄", sym)
    df.insert(0, "フィード", feed)
    frames.append(df)
    w = df["損益%"][df["損益%"] > 0].sum()
    ls = -df["損益%"][df["損益%"] < 0].sum()
    span = (df["入口時刻"].iloc[-1] - df["入口時刻"].iloc[0]).days / 365.25
    print(f"{feed:8s} {sym:8s} N={len(df):4d} 年{len(df)/span:5.1f}本 "
          f"勝率{(df['損益%']>0).mean()*100:5.1f}% PF={w/ls:5.2f} "
          f"平均{df['損益%'].mean():+.3f}% 総{df['損益%'].sum():+.1f}% "
          f"時間切れ{(df['出口理由']=='時間切れ').mean()*100:4.1f}%")

allt = pd.concat(frames, ignore_index=True)
path = "experiments/crypto_1h_atr_spike_trades.csv"
allt.to_csv(path, index=False, encoding="utf-8-sig")
print(f"\n書き出し: {path}  合計 {len(allt)} トレード")

# 検算: Vantage BTC は既知の年46.4本、Binance BTC は既知の PF1.80(週末フィルタ有)
v = frames[0]
assert 200 <= len(v) <= 230, len(v)
b = frames[2]
wb = b["損益%"][b["損益%"] > 0].sum(); lb = -b["損益%"][b["損益%"] < 0].sum()
assert 1.70 < wb / lb < 1.90, wb / lb
print("OK: Vantage BTC の本数と Binance BTC の PF が既知値と整合")
