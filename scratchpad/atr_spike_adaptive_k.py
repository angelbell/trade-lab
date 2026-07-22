"""引き金の閾値を「ATR固定倍率」から「直近N本の実体分布に対する順位」へ差し替える。

動機: 今日の測定で、情報を担っているのは射程ではなく【実体】と確定した（m_breakout#atr-spike-reach-vs-body）。
      ならば次に問うべきは「実体の異常さを何で測るか」。現行 body > ATR(14)*2.0 は
      (1) ATRが正しい物差し (2) 倍率がレジーム不変、の2つを暗黙に仮定している。

3つの引き金を同じ執行（損切り=引き金足の安値・ATR×3トレール・fwd20・前日高値>0・土日建て禁止）で比べる:
  FIX  body > ATR(14)[確定] * k                    ← 現行
  RANK body が【直近N本の実体】の上位 (100-P)% にある（ATR不使用・完全適応）
  RANKA body/ATR が【直近N本の body/ATR】の上位 (100-P)% にある（ハイブリッド）
順位は必ず [i-N, i-1] の確定分だけで作る（i 自身を含めない＝先読み禁止）。

判定: 3銘柄（BTC/ETH/USDJPY）で台地がそろい、かつ IS→OOS で崩れないか。
      現行 FIX を明確に上回らないなら、魔法の数字は減らせないと結論して閉じる。
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

FWD, TRAIL = 20, 3.0


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def roll_quantile_prev(x, win, q):
    """[i-win, i-1] の分位。i 自身を含めない（先読み禁止）。"""
    return pd.Series(x).rolling(win, min_periods=win // 2).quantile(q).shift(1).to_numpy()


def prep(path, utc, start):
    d = load_mt5_csv(path)
    if utc:
        idx = d.index.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")
        d = d.set_index(idx)
    d = d.loc[start:]
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    ap = wilder_atr(d).shift(1).to_numpy()
    day = d.index.floor("D")
    _pdh = d["high"].groupby(day).max().shift(1)
    pdh = pd.Series(_pdh.reindex(d.index).to_numpy(), index=d.index).ffill().to_numpy()
    return d, o, h, l, c, ap, pdh


def trig(d, o, h, l, c, ap, pdh, kind, a, b, skip_we, gate=None, lo=None, hi=None):
    body = c - o
    if kind == "FIX":
        m = body > ap * a
    elif kind == "RANK":
        m = body > roll_quantile_prev(body, int(a), b / 100.0)
    elif kind == "RANKA":
        ba = np.where(np.isfinite(ap) & (ap > 0), body / ap, np.nan)
        m = ba > roll_quantile_prev(ba, int(a), b / 100.0)
    m = m & (c > o) & np.isfinite(ap) & (ap > 0)
    s = np.flatnonzero(np.nan_to_num(m, nan=0).astype(bool))
    s = s[s + 1 < len(d)]
    s = s[(c[s] - pdh[s]) / ap[s] > 0.0]
    if skip_we:
        s = s[d.index.dayofweek.to_numpy()[s + 1] < 5]
    if gate is not None:
        s = s[gate[s]]
    if lo is not None:
        yy = d.index.year.to_numpy()[s]
        s = s[(yy >= lo) & (yy <= hi)]
    return s


def run(d, o, l, c, s, cost):
    ent = [(i, o[i + 1], l[i], o[i + 1] + 1000.0 * (o[i + 1] - l[i]), i)
           for i in s if o[i + 1] - l[i] > 0]
    if len(ent) < 15:
        return None
    a = SimpleNamespace(pullback_frac=0.0, fill_win=200, fwd=FWD, cost=0.0, max_pos=1,
                        swap_pct=0.0, tp1_frac=0.0, exec_split=0, trail_atr=TRAIL, trail_n=14)
    t, _ = walk(d, ent, None, a)
    if t is None or len(t) < 15:
        return None
    p = ((t["R"] * t["risk"] - cost) / t["e_px"]).to_numpy()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    w, ls = p[p > 0].sum(), -p[p < 0].sum()
    yr = pd.Series(p).groupby(t["time"].dt.year.values).sum()
    return dict(n=len(p), win=(p > 0).mean() * 100, pf=w / ls if ls > 0 else np.nan,
                mean=p.mean() * 100, tot=p.sum() * 100, dd=dd * 100,
                score=p.sum() / dd if dd > 0 else np.nan, pos=int((yr > 0).sum()), ny=len(yr))


def fmt(r):
    return ("    --    " if r is None else
            f"{r['pf']:4.2f}/{r['n']:4d}/{r['score']:5.1f}")


SYMS = [("BTC", "data/binance_btcusdt_h1.csv", True, "2018-01-01", 15.0, True, None),
        ("ETH", "data/binance_ethusdt_h1.csv", True, "2018-01-01", 2.0, True, None),
        ("USDJPY", "data/vantage_usdjpy_h1.csv", False, "2000-01-01", 0.009, False, "wk30")]

if __name__ == "__main__":
    keep = {}
    for nm, path, utc, st, cost, we, gk in SYMS:
        d, o, h, l, c, ap, pdh = prep(path, utc, st)
        gate = None
        if gk == "wk30":
            wc = d["close"].resample("W").last().dropna()
            gate = (wc > wc.rolling(30).mean()).shift(1).reindex(
                d.index, method="ffill").fillna(False).to_numpy()
        print(f"\n===== {nm}   各セル: PF / 本数 / 総%÷DD%")
        base = run(d, o, l, c, trig(d, o, h, l, c, ap, pdh, "FIX", 2.0, 0, we, gate), cost)
        print(f"  FIX  body>ATR*k")
        print("        " + " ".join(f"k={k:<4}{'':>6}" for k in (1.5, 1.75, 2.0, 2.25, 2.5)))
        print("        " + " ".join(
            f"{fmt(run(d, o, l, c, trig(d, o, h, l, c, ap, pdh, 'FIX', k, 0, we, gate), cost)):>14}"
            for k in (1.5, 1.75, 2.0, 2.25, 2.5)))
        for kind in ("RANK", "RANKA"):
            print(f"  {kind:<5}{'実体の順位（ATR不使用）' if kind == 'RANK' else '実体/ATR の順位'}")
            print(f"  {'N本':>6} " + " ".join(f"{'P='+str(p):>14}" for p in (95, 97, 98, 99, 99.5)))
            for win in (250, 500, 1000, 2000):
                row = [fmt(run(d, o, l, c,
                               trig(d, o, h, l, c, ap, pdh, kind, win, p, we, gate), cost))
                       for p in (95, 97, 98, 99, 99.5)]
                print(f"  {win:>6} " + " ".join(f"{x:>14}" for x in row))
        keep[nm] = base
        print(f"  現行 FIX k=2.0 の詳細: N={base['n']} 勝率{base['win']:.1f}% PF={base['pf']:.2f} "
              f"平均{base['mean']:+.3f}% 総{base['tot']:+.1f}% DD{base['dd']:.1f}% "
              f"総/DD={base['score']:.2f} 黒字年{base['pos']}/{base['ny']}")

    print("\n=== IS(前半)→OOS(後半) 、各銘柄で FIX k2.0 と RANK の最良を突き合わせる")
    for nm, path, utc, st, cost, we, gk in SYMS:
        d, o, h, l, c, ap, pdh = prep(path, utc, st)
        gate = None
        if gk == "wk30":
            wc = d["close"].resample("W").last().dropna()
            gate = (wc > wc.rolling(30).mean()).shift(1).reindex(
                d.index, method="ffill").fillna(False).to_numpy()
        yrs = np.unique(d.index.year.to_numpy())
        mid = int(yrs[len(yrs) // 2])
        for lab, kind, a, b in (("FIX k2.0", "FIX", 2.0, 0),
                                ("RANK 500/P98", "RANK", 500, 98),
                                ("RANK 1000/P99", "RANK", 1000, 99),
                                ("RANKA 500/P98", "RANKA", 500, 98)):
            out = []
            for lo, hi, tag in ((int(yrs[0]), mid - 1, "IS"), (mid, int(yrs[-1]), "OOS")):
                r = run(d, o, l, c, trig(d, o, h, l, c, ap, pdh, kind, a, b, we, gate, lo, hi), cost)
                out.append(f"{tag} PF={r['pf']:.2f}/N={r['n']:3d}/総DD={r['score']:5.2f}"
                           if r else f"{tag} --")
            print(f"  {nm:<7} {lab:<15} " + "  ".join(out))

    assert keep["BTC"]["n"] > 250 and 1.4 < keep["BTC"]["pf"] < 2.1, keep["BTC"]
    print(f"\nOK: BTC の基準 FIX k2.0 が既知帯 (N={keep['BTC']['n']}, PF={keep['BTC']['pf']:.2f})")
