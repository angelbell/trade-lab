"""A 1-day max hold on btc15m_L lifts the swap-included book 4.84 -> 5.57. But `--fwd` does TWO
things at once, and only one of them is "the time stop works":

  A. it force-closes a trade at 1 day                                  (the time stop itself)
  B. the position slot (max_pos=1) frees up sooner, so 239 MORE signals get taken  (763 -> 1002)

Isolate them:
  A only   the SAME 763 entries, force-closed at bar+cap, slot occupancy IGNORED  (re-walk)
  B only   fwd=500 (no time stop) but max_pos = 2/3/unlimited                     (run(), canonical)
  A + B    what run(fwd=96) produces                                              (the 5.57)

The re-walk must be bit-faithful or it says nothing, so it is checked against the canonical leg at
cap=None first (n / totR / meanR must match). Two traps the first attempt fell into:
  - the pullback-limit's realized RR is NOT 4.5: the target sits at the MARKET entry's 4.5R, and the
    fill is 0.3R closer, so realized RR = (4.5 + 0.3) / (1 - 0.3) = 6.857 (the leg's medRR = 6.86).
  - cost and swap are already account-R at the half-size trades (risk = t.risk / w), so multiplying
    by w again charges w^2.
Run: .venv/bin/python experiments/maxhold_decompose.py
"""
import sys, io, contextlib, warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from rr_with_swap import leg, SIX
from book_spec_fix import book

ROOT = "/home/angelbell/dev/auto-trade"
BTC_PCT_YR = 30.0
PF, RR = 0.3, 4.5
RR_REAL = (RR + PF) / (1.0 - PF)          # 6.857 -- the RR the FILL actually gets


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
    cfg = {**BASE, "gate_kama": 14, "gate_kama_tf": "240min", "pullback_frac": PF, "rr": RR,
           "fill_win": 200, "fwd": 500}

    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d15, SimpleNamespace(**cfg))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    ei = d15.index.get_indexer(t["time"])                       # t["time"] = the FILL bar
    w = np.where(t["e_px"].values > pdh[ei], 1.0, 0.5)
    e = t["e_px"].values
    rk = t["risk"].values                                       # realized risk = fill - stop
    stop, tgt = e - rk, e + RR_REAL * rk
    cost = 15.0 * w / rk                                        # account-R (w applied ONCE)
    swapd = (BTC_PCT_YR / 365.0 / 100.0) * (e / rk) * w         # 口座R / 日
    hi, lo, cl = d15["high"].values, d15["low"].values, d15["close"].values
    ti = pd.DatetimeIndex(t["time"])
    n = len(e)
    # スワップは「暦の日数」で付く。足の本数×15分で数えると、フィードの欠損足のぶん保有を過小評価する
    # （最初の版はこれで totR を +8R 盛っていた）。正典と同じく index の実時間で測る。
    day = d15.index.values.astype("datetime64[s]").astype(np.int64) / 86400.0

    def walk(cap):
        """同じ 763 本の入口。cap 本たったら強制決済（枠の占有は無視＝入口は変えない）。"""
        R = np.empty(n); hold = np.empty(n); why = np.empty(n, object)
        for i in range(n):
            j0 = ei[i]
            lim = min(j0 + (500 if cap is None else min(cap, 500)), len(cl) - 1)
            if lo[j0] <= stop[i]:                              # 約定足で損切り（正典と同じ）
                r, jj, wy = -1.0, j0, "stop"
            else:
                r = None
                for j in range(j0 + 1, lim + 1):
                    if lo[j] <= stop[i]: r, jj, wy = -1.0, j, "stop"; break
                    if hi[j] >= tgt[i]:  r, jj, wy = RR_REAL, j, "target"; break
                if r is None:
                    jj = lim; r = (cl[jj] - e[i]) / rk[i]; wy = "time"
            h = day[jj] - day[j0]
            R[i] = r * w[i] - cost[i] - swapd[i] * h
            hold[i] = h; why[i] = wy
        return pd.Series(R, index=ti), hold, why

    B0 = {k: leg(k)[0] for k in SIX}
    w0 = pd.Series({k: 1 / B0[k].std() for k in SIX})
    w0 = w0 / w0.sum() * 0.03
    r0 = book(B0, SIX)[2]
    canon = B0["btc15m_L"]

    def bk(s, pin):
        L = dict(B0); L["btc15m_L"] = s
        if not pin:
            return book(L, SIX)[2]
        st = max(L[k].index.min() for k in SIX); en = min(L[k].index.max() for k in SIX)
        x = pd.concat([pd.Series(L[k][(L[k].index >= st) & (L[k].index <= en)].values * w0[k],
                                 index=L[k][(L[k].index >= st) & (L[k].index <= en)].index)
                       for k in SIX]).sort_index()
        eq = np.cumprod(1 + x.values); pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / pk).max() * 100
        cg = (eq[-1] ** (365.25 / max((x.index[-1] - x.index[0]).days, 1)) - 1) * 100
        return cg / max(dd, 1e-9)

    pf_ = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())
    Rb, hb, wyb = walk(None)
    print("再現チェック（re-walk が正典のレッグと一致するか）")
    print(f"  正典 leg('btc15m_L') : n={len(canon):>4}  totR={canon.sum():>+7.1f}  meanR={canon.mean():>+.4f}")
    print(f"  re-walk cap=None     : n={len(Rb):>4}  totR={Rb.sum():>+7.1f}  meanR={Rb.mean():>+.4f}"
          f"   最大差 {np.abs(Rb.values - canon.values).max():.4f}\n")

    print(f"スワップ込みの正典ブック = {r0:.2f}\n")
    print("A のみ: 同じ 763 本に、時間ストップだけを当てる（トレードは増やさない）\n")
    print(f"  {'最大保有':<20}{'n':>5}{'PF':>7}{'meanR':>9}{'totR':>8}{'保有平均(日)':>13}"
          f"{'ブック(重み再計算)':>18}{'ブック(重み固定)':>17}")
    print(f"  {'現行（打切り無し）':<20}{len(Rb):>5}{pf_(Rb.values):>7.2f}{Rb.mean():>+9.3f}"
          f"{Rb.sum():>+8.0f}{hb.mean():>13.2f}{bk(Rb, 0):>18.2f}{bk(Rb, 1):>17.2f}  ← 現行")
    keep = {}
    for cap in (48, 96, 192, 300):
        R, h, why = walk(cap)
        print(f"  {f'{cap}本={cap*0.25/24:.1f}日で強制決済':<20}{len(R):>5}{pf_(R.values):>7.2f}"
              f"{R.mean():>+9.3f}{R.sum():>+8.0f}{h.mean():>13.2f}{bk(R, 0):>18.2f}{bk(R, 1):>17.2f}")
        keep[cap] = (R, why)

    print("\n反実仮想の分解（強制決済された本だけ）: 救ったR vs 切ったR")
    for cap, (R, why) in keep.items():
        m = why == "time"
        d = R.values[m] - Rb.values[m]
        sv, ct = d[d > 0], d[d < 0]
        print(f"  {cap}本（{cap*0.25/24:.1f}日）: 強制決済 {m.sum():>3}本   救った {len(sv):>3}本 "
              f"{sv.sum():>+7.1f}R   切った {len(ct):>3}本 {ct.sum():>+7.1f}R   差引 **{d.sum():>+6.1f}R**")

    print("\n法則9の再確認（スワップ込み）: 『h日たっても生きている本』は、持てばいくら稼ぐか")
    print(f"  {'':<22}{'本数':>6}{'そのまま持った時の meanR':>26}")
    print(f"  {'全 763 本':<22}{n:>6}{Rb.mean():>+26.3f}")
    for cap in (48, 96, 192, 300):
        m = keep[cap][1] == "time"
        print(f"  {f'{cap*0.25/24:.1f}日 生存の本だけ':<22}{m.sum():>6}{Rb.values[m].mean():>+26.3f}")

    print("\n\nB のみ: 時間ストップ無し（fwd=500）のまま、同時保有の枠だけを増やす")
    print(f"  {'max_pos':<12}{'n':>6}{'PF':>7}{'meanR':>9}{'totR':>8}"
          f"{'ブック(重み再計算)':>18}{'ブック(重み固定)':>17}")
    for mp in (1, 2, 3, 99):
        with contextlib.redirect_stderr(io.StringIO()):
            tt = run(d15, SimpleNamespace(**{**cfg, "max_pos": mp}))
        ii = d15.index.get_indexer(tt["time"])
        ww = np.where(tt["e_px"].values > pdh[ii], 1.0, 0.5)
        rr2 = tt["risk"].values / ww
        s = pd.Series(tt["R"].values * ww - 15.0 / rr2
                      - (BTC_PCT_YR / 365.0 / 100.0) * (tt["e_px"].values / rr2) * tt["hold"].values,
                      index=pd.DatetimeIndex(tt["time"]))
        print(f"  {mp if mp < 99 else '無制限':<12}{len(s):>6}{pf_(s.values):>7.2f}{s.mean():>+9.3f}"
              f"{s.sum():>+8.0f}{bk(s, 0):>18.2f}{bk(s, 1):>17.2f}"
              + ("  ← 現行" if mp == 1 else ""))


if __name__ == "__main__":
    main()
