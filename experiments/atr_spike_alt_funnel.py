"""Pine の診断パネルと突き合わせるための絞り込みの段を Python 側から出す。

pine/altcoin_1h_atr_spike.pine の診断表は
  総バー数 → 拡大足 → ＋先導銘柄も同時 → ＋前日高値 → ＋土日除外 → 建てた回数
の順で数える。同じ順序・同じ定義で Python 側の数を出し、TradingView に貼った時に
実装のズレ（先読み・時刻・条件の順番）を数で検出できるようにする。

⚠️ TradingView のフィードは Vantage とも Binance とも違うので、数字は一致しない。
   見るのは【段ごとの通過率】と【桁】。拡大足の本数が半分・倍なら引き金の実装がずれている。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_btc_leader import load as vload, spikes as vspikes, ALTS  # noqa: E402
from experiments.atr_spike_leader_binance import load as bload, spikes as bspikes    # noqa: E402
from experiments.atr_spike_leader_general import wilder_atr, K                       # noqa: E402


def funnel(d, lead_any):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    spike = (c - o > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(spike)
    s = s[s + 1 < len(d)]
    n_spike = len(s)
    lead = lead_any.reindex(d.index).fillna(False).to_numpy()
    s1 = s[lead[s]]
    s2 = s1[(c[s1] - pdh[s1]) / ap[s1] > 0.0]
    s3 = s2[d.index.dayofweek.to_numpy()[s2 + 1] < 5]
    return len(d), n_spike, len(s1), len(s2), len(s3)


def run(tag, loader, spiker, syms, btc, start_lbl):
    lead = spiker(loader(btc))
    lead = (lead | lead.shift(1)).fillna(False)
    print(f"\n===== {tag}（{start_lbl}）  先導＝{btc}（同じ足 or 1本前）")
    print(f"  {'銘柄':<10} {'総バー':>8} {'拡大足':>7} {'＋先導同時':>10} "
          f"{'＋前日高値':>10} {'＋土日除外':>10} {'通過率':>8}")
    tot = np.zeros(5, dtype=int)
    for sym in syms:
        d = loader(sym)
        f = funnel(d, lead)
        tot += np.array(f)
        print(f"  {sym:<10} {f[0]:>8} {f[1]:>7} {f[2]:>10} {f[3]:>10} {f[4]:>10} "
              f"{f[4]/max(f[1],1)*100:>7.0f}%")
    print(f"  {'合計':<10} {tot[0]:>8} {tot[1]:>7} {tot[2]:>10} {tot[3]:>10} {tot[4]:>10} "
          f"{tot[4]/max(tot[1],1)*100:>7.0f}%")
    print(f"  段ごとの通過率: 拡大足→先導同時 {tot[2]/max(tot[1],1)*100:.0f}%"
          f" → 前日高値 {tot[3]/max(tot[2],1)*100:.0f}%"
          f" → 土日除外 {tot[4]/max(tot[3],1)*100:.0f}%")
    return tot


V_USE = ["ethusd", "solusd", "adausd", "dotusd", "xrpusd", "ltcusd"]
B_USE = ["ethusdt", "solusdt", "adausdt", "dotusdt", "xrpusdt", "ltcusdt"]

if __name__ == "__main__":
    tv = run("Vantage", vload, vspikes, V_USE, "btcusd", "2022-01-01 以降")
    tb = run("Binance", bload, bspikes, B_USE, "btcusdt", "2018-01-01 以降")
    print("\n  TradingView に貼ったときの見方: フィードが違うので本数は一致しない。"
          "\n  一致すべきは【段ごとの通過率】。特に「拡大足→先導同時」が 3〜5割から大きく外れたら、"
          "\n  先導銘柄の指定ミス（取引所違い＝足の切れ目のズレ）を疑う。")
    assert 0.25 < tv[2] / tv[1] < 0.65, tv[2] / tv[1]
    assert 0.25 < tb[2] / tb[1] < 0.65, tb[2] / tb[1]
    print(f"\nOK: 先導同時の通過率 Vantage {tv[2]/tv[1]*100:.0f}% / Binance {tb[2]/tb[1]*100:.0f}%")
