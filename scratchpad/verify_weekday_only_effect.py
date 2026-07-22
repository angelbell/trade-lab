"""なぜ「平日限定」だと成績が上がるのか。同一データから土日だけ削る対照実験。

Binance（真の24時間）から土日の足を落とすと、銘柄・期間・フィードは同一のまま
「週末の有無」だけが違う2つの系列になる。Vantage 2018-2021 の水増しが再現するかを見る。
再現したら、次に病巣を3つの候補に分解する:
  1. 保有20本が週末をまたぐと実時間が伸びる（勝ちが育つ時間のただ乗り）
  2. 週末の逆行が見えない（損切りに触る動きが月曜の窓になり、しかも損切り値ちょうどで約定扱い）
  3. 引き金の母集団が変わる（月曜の窓足が拡大足として入る）
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

COST = 0.0005


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def load_binance(path):
    d = load_mt5_csv(path)
    idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
    return d.set_index(idx)


def build(d, k=2.0, use_pdh=True):
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    hit = (c - o > ap * k) & (c > o) & np.isfinite(ap)
    s_all = np.flatnonzero(hit)
    s_all = s_all[s_all + 1 < len(d)]
    if use_pdh:
        s_all = s_all[(c[s_all] - pdh[s_all]) / ap[s_all] > 0.0]
    return [(s, o[s + 1], l[s], o[s + 1] + 1000.0 * (o[s + 1] - l[s]), s)
            for s in s_all if o[s + 1] - l[s] > 0]


def go(d, ent, fwd=20, trail=3.0):
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=fwd, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=trail, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 10:
        return None, None
    p = ((t["R"] * t["risk"] - COST * t["e_px"]) / t["e_px"]).to_numpy()
    return t, p


def stats(p):
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max()) * 100
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    return (f"N{len(p):4d} 勝率{np.mean(p>0)*100:5.1f}% PF{w/ls if ls>0 else float('nan'):6.2f} "
            f"平均{p.mean()*100:+7.3f}% 総{p.sum()*100:+8.1f}% maxDD{dd:6.1f}%")


full = load_binance("data/binance_btcusdt_h1.csv")
wd = full[full.index.dayofweek < 5]          # 土日を削る＝Vantage 2018-2021 の構造を再現

print("========== 同一データ・同一期間で「土日の有無」だけを変える（BTCUSDT）")
for start, end, lab in (("2018-01-01", "2021-12-31", "2018-2021"),
                        ("2022-01-01", None, "2022-2026"),
                        ("2018-01-01", None, "2018-2026")):
    for tag, d in (("24時間", full), ("平日のみ", wd)):
        x = d.loc[start:end] if end else d.loc[start:]
        t, p = go(x, build(x))
        print(f"  {lab:<12} {tag:<8} {stats(p)}")
    print()

# ---- 病巣の分解: 週末をまたいだトレードとまたがないトレード（24時間データ側で）
x = full.loc["2018-01-01":]
t, p = go(x, build(x))
pos = {ts: i for i, ts in enumerate(x.index)}
ent_i = np.array([pos[ts] for ts in t["time"]])
# 保有20本ぶんの実時間に土日が含まれるか
spans = []
for i in ent_i:
    j = min(i + 20, len(x) - 1)
    spans.append(bool(((x.index[i:j + 1].dayofweek >= 5)).any()))
spans = np.array(spans)
print("========== 24時間データ側で、保有窓に土日を含むトレードと含まないトレード")
print(f"  週末を含む   {stats(p[spans])}")
print(f"  週末を含まない {stats(p[~spans])}")

# ---- 平日のみデータで、月曜の窓足が引き金に占める割合
ent_wd = build(wd.loc["2018-01-01":])
mon = np.array([wd.loc['2018-01-01':].index[e[0] + 1].dayofweek == 0 for e in ent_wd])
print(f"\n  平日のみデータの引き金 {len(ent_wd)}本のうち、入口が月曜の足＝{mon.sum()}本 ({mon.mean()*100:.1f}%)")

# 検算: 24時間の 2018-2026 が既知の値を再現
t2, p2 = go(full.loc["2018-01-01":], build(full.loc["2018-01-01":]))
assert 430 <= len(p2) <= 442, len(p2)
assert 0.42 < p2.mean() * 100 < 0.47, p2.mean() * 100
print(f"\nOK: Binance 24時間 2018-2026 が N={len(p2)} 平均{p2.mean()*100:+.3f}% を再現")
