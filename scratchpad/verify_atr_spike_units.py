"""押し目指値ラダーの単位を直した再検定。

measure の報告は R 単位で出ている。押し目を深くすると実現損切り幅が (1-pf) 倍に縮むので
R は 1/(1-pf) 倍に機械的に膨らむ（pf=0.786 なら 4.7倍）。帰無は成行版（フル損切り幅）の
R 母集団からの間引きなので、**違う物差し同士を比べている**。
さらにユーザーはロット0.01固定＝R は元々トレードしていない単位。

ここでは各トレードの損益を「入口価格に対する%」（サイズ規則を含まない尺度不変な単位）に
戻して、pf の梯子と間引き帰無をやり直す。執行は engine の walk() のまま。
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
from src.engine.mirror import invert              # noqa: E402


def wilder_atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def build(d, k, rr):
    """引き金＝実体 > ATR[s-1]×k かつ陽線。stop = 拡大足の安値（A系）。"""
    atr_prev = wilder_atr(d).shift(1).to_numpy()
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    ent = []
    for s in np.flatnonzero(hit):
        if s + 1 >= len(d):
            continue
        e, stop = o[s + 1], l[s]
        if e - stop <= 0:
            continue
        ent.append((s, e, stop, e + rr * (e - stop), s))
    return ent


def run_cell(d, k, rr, pf, fill_win, fwd=20, cost=0.0005, C=None):
    # 🚨 walk() 内部の cost は `cost/risk*e_px` で、反転フレームでは e_px が鏡像価格（実価格の
    # 約3倍）になり過大請求になる（mirror.py の「比率ベースの特徴量は綺麗に鏡像化しない」の実例）。
    # よって walk はコスト0で回し、実価格ベースのコストを外側で引く。
    args = SimpleNamespace(pullback_frac=pf, fill_win=fill_win, fwd=fwd, cost=0.0,
                           max_pos=1, swap_pct=0.0, tp1_frac=0.0, exec_split=0)
    t, _ = walk(d, build(d, k, rr), None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    pnl_px = t["R"] * t["risk"] - cost * e_real          # 往復コストを価格距離で引く
    t = t.assign(pnl_px=pnl_px, pnl_pct=pnl_px / e_real, R_net=pnl_px / t["risk"])
    return t


def stats(t, span):
    p = t["pnl_pct"].to_numpy()
    win, loss = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"N": len(p), "N/年": len(p) / span, "勝率": float((p > 0).mean()),
            "PF": float(win / loss) if loss > 0 else float("inf"),
            "平均%": float(p.mean() * 100), "総%": float(p.sum() * 100), "maxDD%": dd * 100}


def drop_null(t0, q, t_obs, reps=200, seed=11):
    """成行(pf=0)母集団から同じ約定率 q だけランダムに残す。全て同じ価格単位。"""
    rng = np.random.default_rng(seed)
    p0 = t0["pnl_pct"].to_numpy()
    n = max(1, int(round(q * len(p0))))
    means, pfs = [], []
    for _ in range(reps):
        s = rng.choice(p0, size=n, replace=False)
        means.append(s.mean())
        w, l = s[s > 0].sum(), -s[s < 0].sum()
        pfs.append(w / l if l > 0 else np.nan)
    means, pfs = np.array(means), np.array(pfs)
    obs_m = t_obs["pnl_pct"].mean()
    obs_pf = stats(t_obs, 1)["PF"]
    return {"帰無平均%": float(np.median(means) * 100), "平均%ile": float((means < obs_m).mean() * 100),
            "帰無PF": float(np.nanmedian(pfs)), "PF%ile": float((pfs < obs_pf).mean() * 100)}


df = load_mt5_csv("data/vantage_btcusd_h1.csv")
span = (df.index[-1] - df.index[0]).days / 365.25
frames = {"long": (df, None)}
inv = invert(df)
frames["short"] = (inv, 2 * df["high"].max())

PFS = [0.0, 0.25, 0.382, 0.5, 0.618, 0.786]
for side, (d, C) in frames.items():
    for k, rr in ((2.0, 3.0), (1.5, 4.5)):
        print(f"\n===== {side} A系 k={k} RR={rr} fill_win=200 fwd=20 "
              f"（損益＝入口価格に対する%・ロット固定・サイズ規則なし）")
        print(f"{'pf':>6} {'約定率':>7} {'N':>5} {'N/年':>6} {'勝率':>6} {'PF':>6} "
              f"{'平均%':>7} {'総%':>8} {'maxDD%':>7} | {'帰無PF':>6} {'PF%ile':>7} {'平均%ile':>8}")
        base = run_cell(d, k, rr, 0.0, 200, C=C)
        n0 = len(base)
        for pf in PFS:
            t = run_cell(d, k, rr, pf, 200, C=C)
            if t is None:
                continue
            q = len(t) / n0
            s = stats(t, span)
            nl = drop_null(base, q, t) if pf > 0 else {"帰無PF": float("nan"),
                                                       "PF%ile": float("nan"), "平均%ile": float("nan")}
            print(f"{pf:6.3f} {q*100:6.1f}% {s['N']:5d} {s['N/年']:6.1f} {s['勝率']*100:5.1f}% "
                  f"{s['PF']:6.2f} {s['平均%']:7.3f} {s['総%']:8.1f} {s['maxDD%']:7.1f} | "
                  f"{nl['帰無PF']:6.2f} {nl['PF%ile']:6.1f}% {nl['平均%ile']:7.1f}%")

# 検算: R単位では pf が深いほど平均Rが膨らむこと（＝報告が拾っていた効果）を数値で示す
d, C = frames["short"]
ts = run_cell(d, 2.0, 3.0, 0.25, 200, C=C)
td = run_cell(d, 2.0, 3.0, 0.786, 200, C=C)
print(f"\n[検算] ショート k=2.0 RR3 の物差し比較（同じトレード集合を2つの単位で）")
print(f"   平均R     pf=0.25 → {ts['R_net'].mean():+.3f} / pf=0.786 → {td['R_net'].mean():+.3f}")
print(f"   平均価格% pf=0.25 → {ts['pnl_pct'].mean()*100:+.3f}% / pf=0.786 → {td['pnl_pct'].mean()*100:+.3f}%")
print(f"   実現損切り幅の中央値 pf=0.25 → {ts['risk'].median():.1f} / pf=0.786 → {td['risk'].median():.1f}")
# 実現損切り幅は (1-pf) に比例して縮む＝R が機械的に膨らむ経路。3.5倍以上の縮小を数値で固定
assert ts["risk"].median() / td["risk"].median() > 2.5, ts["risk"].median() / td["risk"].median()
# ロング成行の素の期待値は正（BTCの上昇ドリフト＋継続）であること
tl = run_cell(df, 2.0, 3.0, 0.0, 200)
print(f"[検算] ロング成行 k=2.0 RR3: N={len(tl)} 平均価格%={tl['pnl_pct'].mean()*100:+.3f}%")
assert len(tl) > 500, len(tl)
