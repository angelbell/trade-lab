"""Pine の診断パネルと突き合わせるための数字を出す（btc_1h_atr_spike_er.pine）。

パネルは 総バー数 → 拡大足 → ＋前日高値 → ＋土日除外 → 建てた回数 → うち厚い玉 の順に数える。
同じ順序・同じ定義で Python 側の数を出し、実装のズレを数で検出できるようにする。
⚠️ TradingView のフィードは Vantage とも違うので本数は一致しない。見るのは【段ごとの通過率】と【厚い玉の割合】。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scratchpad.atr_spike_frequency import load                       # noqa: E402
from scratchpad.atr_spike_barspread import wilder_atr                 # noqa: E402

K, KTHICK, ERLEN, MEDLEN = 1.5, 2.0, 120, 500


def funnel(d, lab):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    body = c - o
    spike = (body > ap * K) & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(spike)
    s = s[s + 1 < len(d)]
    n_spike = len(s)
    s1 = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    s2 = s1[d.index.dayofweek.to_numpy()[s1 + 1] < 5]
    # 効率比（当バーを含めず前バーまでで確定）
    cs = pd.Series(c, index=d.index)
    path = cs.diff().abs().rolling(ERLEN).sum()
    erraw = (cs - cs.shift(ERLEN)).abs() / path
    er = erraw.shift(1).to_numpy()
    med = erraw.rolling(MEDLEN, min_periods=MEDLEN // 2).median().shift(1).to_numpy()
    thick = (body >= ap * KTHICK) & np.isfinite(er) & np.isfinite(med) & (er >= med)
    n_thick = int(thick[s2].sum())
    span = (d.index[-1] - d.index[0]).days / 365.25
    print(f"\n  {lab}（{d.index[0]:%Y-%m}〜{d.index[-1]:%Y-%m}・{span:.1f}年）")
    print(f"    総バー数      {len(d):>8,}")
    print(f"    拡大足        {n_spike:>8,}   （総バーの {n_spike/len(d)*100:.2f}%）")
    print(f"    ＋前日高値    {len(s1):>8,}   （拡大足の {len(s1)/max(n_spike,1)*100:.0f}%）")
    print(f"    ＋土日除外    {len(s2):>8,}   （前段の {len(s2)/max(len(s1),1)*100:.0f}%）")
    print(f"    　→ 年 {len(s2)/span:.0f} 本")
    print(f"    うち厚い玉    {n_thick:>8,}   （{n_thick/max(len(s2),1)*100:.0f}%・年 {n_thick/span:.0f} 本）")
    print(f"    効率比の中央値 {np.nanmedian(erraw.to_numpy()):.4f}"
          f"（範囲 {np.nanmin(erraw.to_numpy()):.4f}〜{np.nanmax(erraw.to_numpy()):.4f}）")
    return len(s2), n_thick


if __name__ == "__main__":
    print("=== Pine 診断パネルとの照合用（btc_1h_atr_spike_er.pine）")
    print(f"    設定: k={K} / 厚くする実体={KTHICK}ATR / 効率比の期間={ERLEN}本 / "
          f"中央値の期間={MEDLEN}本")
    n1, t1 = funnel(load("btcusd", "h1"), "Vantage BTCUSD 1時間")
    print("\n  ※ TradingView に貼ったときの見方: フィードが違うので本数は一致しない。")
    print("     一致すべきは【拡大足の出現率 ≈ 総バーの1.9%】【前日高値の通過率 ≈ 6割】")
    print("     【厚い玉の割合 ≈ 3割】。ここが大きくずれたら実装のズレを疑う。")
    assert n1 > 150 and 0.15 < t1 / n1 < 0.5, (n1, t1)
    print(f"\nOK: 建てた回数 {n1} / 厚い玉 {t1}（{t1/n1*100:.0f}%）")
