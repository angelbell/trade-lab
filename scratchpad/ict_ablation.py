"""ICT v2 の「狩り＋MSS」を分解する（2026-07-14）。
v2 は2つを同時に足したので「フィルタは本物」とは言えても「どちらが本物か」は言えなかった。

3つの軸を独立に切る:
  ゲート1  sweep : ロンドン安値が アジア安値 or 前日安値 を割ったか（流動性の狩り）
  ゲート2  mss   : その後、直前のスイング高値(3本フラクタル)をヒゲで上抜けたか（構造転換）
  定義軸   leg   : 押し目を測る脚 L→H の H をどこで打ち切るか
                   "mss"     = MSS が成立した足まで（v2 の定義。MSS が必要）
                   "lonend"  = ロンドン窓の終わりまで（v1 の定義。ゲートと独立に使える）
  → leg="lonend" に固定すれば、sweep と mss を「純粋なゲート」として ablation できる。

審判は「本物の窓 vs +8h プラセボ窓」の差（＝窓のプレミアム）。
v2 でこれを破ったのは EURUSD ロング・浅い戻り(0.25)・遠い利確(RR4) だけだった。
執行は現実版（ASK基準の指値約定 0.3pip + 手数料のみ）。

Run: .venv/bin/python scratchpad/ict_ablation.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_killzone import load_ny, SYMS
from ict_v2_mss import (prep, window_pos, prev_day_extremes, last_fractal_high,
                        last_fractal_low, walk, stats, MODEL)

PIP = {"eurusd": 1e-4, "gbpusd": 1e-4, "audusd": 1e-4, "usdjpy": 1e-2, "gold": 0.1, "btcusd": 1.0}
FILL_SPREAD_PIPS = 0.3
BUF = 0.1


def build(df, tarr, dates, use_sweep, use_mss, leg, shift=0):
    """ゲートを独立に切れる版の setup builder。"""
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    atr = df["atr14"].values
    pdh, pdl = prev_day_extremes(df, dates)
    A0H, A1H = 19 + shift, 2 + shift          # アジア窓 (前日 A0H:00 〜 当日 A1H:00)
    L0H, L1H = 2 + shift, 7 + shift           # ロンドン窓
    K0H, K1H = 7 + shift, 10 + shift          # キルゾーン
    out = []
    for d in dates:
        day = pd.Timestamp(d)
        a0, a1 = window_pos(tarr, day - pd.Timedelta(days=1) + pd.Timedelta(hours=A0H),
                            day + pd.Timedelta(hours=A1H))
        l0, l1 = window_pos(tarr, day + pd.Timedelta(hours=L0H), day + pd.Timedelta(hours=L1H))
        k0, k1 = window_pos(tarr, day + pd.Timedelta(hours=K0H), day + pd.Timedelta(hours=K1H))
        rec = {"date": d, "long": None, "short": None}
        if (a1 - a0) < 4 or (l1 - l0) < 6 or (k1 - k0) < 2 or not np.isfinite(atr[l1 - 1]):
            out.append(rec); continue
        A = atr[l1 - 1]
        asia_lo, asia_hi = lo[a0:a1].min(), hi[a0:a1].max()
        p_lo, p_hi = pdl.get(d, np.nan), pdh.get(d, np.nan)

        # ---------------- LONG ----------------
        iL = l0 + int(np.argmin(lo[l0:l1])); L = lo[iL]
        ok = True
        if use_sweep:
            ok = ((np.isfinite(asia_lo) and L < asia_lo) or (np.isfinite(p_lo) and L < p_lo))
        jm = None
        if ok and use_mss:
            sh = last_fractal_high(hi, a0, iL)
            if sh is None:
                ok = False
            else:
                lvl = hi[sh]
                for j in range(iL + 1, l1):
                    if hi[j] > lvl:
                        jm = j; break
                ok = jm is not None
        if ok:
            end = (jm + 1) if (leg == "mss" and jm is not None) else l1
            if end <= iL + 1:
                ok = False
            else:
                H = hi[iL:end].max()
                if H - L < 0.25 * A:
                    ok = False
                elif (lo[end:l1] <= L).any():       # KZ 到達前に構造が壊れた
                    ok = False
                else:
                    rec["long"] = dict(L=L, H=H, atr=A, kz=(k0, k1))

        # ---------------- SHORT (鏡像) ----------------
        iH = l0 + int(np.argmax(hi[l0:l1])); Hh = hi[iH]
        ok = True
        if use_sweep:
            ok = ((np.isfinite(asia_hi) and Hh > asia_hi) or (np.isfinite(p_hi) and Hh > p_hi))
        jm = None
        if ok and use_mss:
            sl = last_fractal_low(lo, a0, iH)
            if sl is None:
                ok = False
            else:
                lvl = lo[sl]
                for j in range(iH + 1, l1):
                    if lo[j] < lvl:
                        jm = j; break
                ok = jm is not None
        if ok:
            end = (jm + 1) if (leg == "mss" and jm is not None) else l1
            if end <= iH + 1:
                ok = False
            else:
                Ll = lo[iH:end].min()
                if Hh - Ll < 0.25 * A:
                    ok = False
                elif (hi[end:l1] >= Hh).any():
                    ok = False
                else:
                    rec["short"] = dict(L=Ll, H=Hh, atr=A, kz=(k0, k1))
        out.append(rec)
    return out


CONFIGS = [
    ("1. ゲート無し (v1相当)",        False, False, "lonend"),
    ("2. 狩りのみ",                   True,  False, "lonend"),
    ("3. MSSのみ",                    False, True,  "lonend"),
    ("4. 狩り + MSS",                 True,  True,  "lonend"),
    ("5. MSSのみ / 脚=MSSまで",        False, True,  "mss"),
    ("6. 狩り + MSS / 脚=MSSまで(v2)", True,  True,  "mss"),
]
PARAMS = [(0.25, 4.0, "浅0.25/RR4 (トレンド形・v2の生存セル)"),
          (0.705, 2.0, "深0.705/RR2 (ICT正典)")]


def main():
    D, SPAN = {}, {}
    for name in SYMS:
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
        df, tarr, dates = prep(df)
        D[name] = (df, tarr, dates)
        y = pd.to_datetime(df["broker_dt"]).dt.year.value_counts()
        SPAN[name] = int((y > 5000).sum())

    print("=" * 126)
    print("ABLATION 1: EURUSD ロング（v2 で唯一プラセボ窓を破ったセル）— 何が効いていたのか")
    print("  審判 = 本物の窓 vs +8h プラセボ窓 の差（窓のプレミアム）。本物のフィルタなら premium > 0。")
    print("=" * 126)
    df, tarr, dates = D["eurusd"]
    sp = FILL_SPREAD_PIPS * PIP["eurusd"]; _, cost = MODEL["eurusd"]
    for f, rr, plab in PARAMS:
        print(f"\n--- {plab} ---")
        print(f"  {'config':30s} {'本物の窓':>34s} {'+8h プラセボ':>26s} {'premium':>9s}")
        for lab, us, um, leg in CONFIGS:
            r = stats(walk(df, build(df, tarr, dates, us, um, leg, 0), f, rr, BUF, sp, cost, "long"),
                      SPAN["eurusd"])
            p = stats(walk(df, build(df, tarr, dates, us, um, leg, 8), f, rr, BUF, sp, cost, "long"),
                      SPAN["eurusd"])
            rs = (f"n={r['n']:4d} 年{r['npy']:4.0f} net={r['net']:+.3f} PF={r['pf']:.2f} "
                  f"緑年{r['gy']:3.0f}%") if r else "n<10"
            ps = (f"n={p['n']:4d} net={p['net']:+.3f} PF={p['pf']:.2f}") if p else "n<10"
            pm = f"{r['net'] - p['net']:+9.3f}" if (r and p) else "      n/a"
            print(f"  {lab:30s} {rs:>34s} {ps:>26s} {pm}")

    print("\n" + "=" * 126)
    print("ABLATION 2: 横展開 — 「浅0.25/RR4」で 6銘柄×両サイド。config 3(MSSのみ) と 4(狩り+MSS)")
    print("  premium = 本物の窓の net − +8h プラセボの net。**これが正のセルだけが候補**。")
    print("=" * 126)
    for cfg_i in (2, 3):          # 3.MSSのみ / 4.狩り+MSS
        lab, us, um, leg = CONFIGS[cfg_i]
        print(f"\n--- {lab} ---")
        print(f"  {'sym':7s} {'side':6s} {'n':>5} {'年':>4} {'win%':>5} {'gross':>7} {'net':>7} "
              f"{'PF':>5} {'IS':>7} {'OOS':>7} {'緑年':>5} | {'+8h net':>8} {'premium':>8}")
        for name in ("gold", "eurusd", "gbpusd", "usdjpy", "audusd", "btcusd"):
            dfx, tx, dx = D[name]
            spx = FILL_SPREAD_PIPS * PIP[name]; _, cx = MODEL[name]
            S0 = build(dfx, tx, dx, us, um, leg, 0)
            S8 = build(dfx, tx, dx, us, um, leg, 8)
            for side in ("long", "short"):
                r = stats(walk(dfx, S0, 0.25, 4.0, BUF, spx, cx, side), SPAN[name])
                p = stats(walk(dfx, S8, 0.25, 4.0, BUF, spx, cx, side), SPAN[name])
                if not r:
                    continue
                pn = f"{p['net']:+8.3f}" if p else "     n/a"
                pm = f"{r['net'] - p['net']:+8.3f}" if p else "     n/a"
                print(f"  {name:7s} {side:6s} {r['n']:5d} {r['npy']:4.0f} {r['win']:5.1f} "
                      f"{r['gross']:+7.3f} {r['net']:+7.3f} {r['pf']:5.2f} {r['IS']:+7.1f} "
                      f"{r['OOS']:+7.1f} {r['gy']:4.0f}% | {pn} {pm}")


if __name__ == "__main__":
    main()
