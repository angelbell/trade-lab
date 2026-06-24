"""wave3_lab.py -- ハイトレFX動画「4時間足3波EMA戦略」の機械化＆検証ラボ。

エントリー（ロング。ショートはミラー）:
  1. ゴールデンクロス: 20EMA が 200EMA を上抜け → 「初回押し待ち」
  2. 初回押し: クロス後、価格が 20EMA にタッチした最初の足 = 「基準足」, その高値=トリガー
  3. エントリー: 後続足が基準足高値を【終値（実体）】で上抜け → そのバーをエントリーとする
  4. 1クロス1回のみ。抜け確定前に 200EMA とデッドクロスしたら見送り。

利確（ユーザー確定仕様 2026-06-22）:
  - T1 = 戻り高値 = entryより上の「最寄りの過去スイング高値」 → 半利
  - T2 = 直近高値 = その次（より上）の過去スイング高値          → 残り全利
  - T1までのRR(=（T1-entry）/（entry-stop)) が 1 未満、または T1 が無ければ見送り
  - 半利(T1到達)後、残り半分の損切りは建値(BE)へ移動

損切り（ユーザー確定仕様）:
  - 既定 = 直近安値 = entryより下の「最寄りの過去スイング安値」
  - （裁量: 高値までのRRが3以上なら押し安値=直近の押し目最安値でもよい → --stop pullback）
  - risk下限 = 0.5*ATR フロア（極小riskでRが暴れるのを防ぐ）

★スイング検出は LOCAL PIVOT（左右N本の極値, +right本で確定=因果的）。
  動画の「目立つ高値/安値」= 近くの目立つ極値であって、ATR-ZigZagの大スイング天井ではない
  （k3 ZigZag だと最寄り上値が遠い大天井になり RR9.5 等の異常値が出ていた → これを修正）。

  .venv/bin/python research/wave3_lab.py --csv data/vantage_usdjpy_h1.csv --tf 4h --dump 8
"""
import argparse, os, sys
import numpy as np, pandas as pd, pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_mt5_csv


def resample(df, rule):
    if rule.lower() in ("1h", "h1"):
        return df
    o = {"4h": "4h", "2h": "2h", "8h": "8h", "1d": "1D"}.get(rule.lower(), rule)
    return pd.DataFrame({"open": df["open"].resample(o).first(), "high": df["high"].resample(o).max(),
                         "low": df["low"].resample(o).min(), "close": df["close"].resample(o).last()}).dropna()


def local_pivots(h, l, left, right):
    """LOCAL pivot高値/安値。pivot high@p = high[p] が [p-left, p+right] の厳密最大。
    p+right 本目で確定（それ以前は未確定＝因果的に使えない）。
    返り値: highs/lows = list of (confirm_idx, price)。"""
    n = len(h)
    highs, lows = [], []
    for p in range(left, n - right):
        wh = h[p - left:p + right + 1]
        wl = l[p - left:p + right + 1]
        if h[p] == wh.max() and (h[p] > wh[:left]).all() and (h[p] >= wh[left + 1:]).all():
            highs.append((p + right, h[p]))     # confirmed at p+right
        if l[p] == wl.min() and (l[p] < wl[:left]).all() and (l[p] <= wl[left + 1:]).all():
            lows.append((p + right, l[p]))
    return highs, lows


def wave3_entries(d, side="long"):
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    n = len(c)
    e20 = d["close"].ewm(span=20, adjust=False).mean().values
    e200 = d["close"].ewm(span=200, adjust=False).mean().values
    out = []
    mode = "none"; trig = None; pext = None
    for i in range(201, n - 1):
        if np.isnan(e200[i]):
            continue
        if side == "long":
            gc = e20[i] > e200[i] and e20[i - 1] <= e200[i - 1]
            dc = e20[i] < e200[i] and e20[i - 1] >= e200[i - 1]
        else:
            gc = e20[i] < e200[i] and e20[i - 1] >= e200[i - 1]
            dc = e20[i] > e200[i] and e20[i - 1] <= e200[i - 1]
        if gc:
            mode = "wp"; trig = None; pext = None; continue
        if mode == "none":
            continue
        if dc:
            mode = "none"; continue                          # 抜け前にDC=見送り
        if mode == "wp":
            touch = (l[i] <= e20[i]) if side == "long" else (h[i] >= e20[i])
            if touch:
                trig = (h[i] if side == "long" else l[i])
                pext = (l[i] if side == "long" else h[i])    # 押し目極値(=押し安値/戻り高値)
                mode = "wb"
        elif mode == "wb":
            pext = (min(pext, l[i]) if side == "long" else max(pext, h[i]))
            brk = (c[i] > trig) if side == "long" else (c[i] < trig)
            if brk:
                out.append((i, c[i], pext)); mode = "none"
    return out


def build_trades(d, side, highs, lows, stop_mode, fwd, cost, atr):
    c, h, l = d["close"].values, d["high"].values, d["low"].values
    n = len(c)
    ents = wave3_entries(d, side)
    rows = []
    n_target = 0
    for (i, e, pext) in ents:
        # --- stop: 直近安値(既定) or 押し安値 ---
        if stop_mode == "swing":   # 直近安値 = entryより下の最寄りの確定スイング安値
            below = [pr for (cf, pr) in lows if cf <= i and pr < e]
            if not below:
                continue
            stop = max(below)                                # 最も新しく高い安値(=higher-low)
        else:                       # 押し安値 = 押し目の極値
            stop = pext
        if e - stop < 0.5 * atr[i]:                          # risk フロア
            stop = e - 0.5 * atr[i]
        risk = e - stop
        if risk <= 0:
            continue
        # --- targets: entryより上の確定スイング高値 ---
        above = sorted([pr for (cf, pr) in highs if cf <= i and pr > e])
        if not above:
            continue
        n_target += 1
        T1 = above[0]
        T2 = above[1] if len(above) >= 2 else None
        rr1 = (T1 - e) / risk
        if rr1 < 1.0:                                         # RR≥1 フィルタ
            continue
        # --- sim: T1半利→BE, T2全利 ---
        end = min(i + 1 + fwd, n); half = False; r1 = None; R = None; xj = end - 1
        for j in range(i + 1, end):
            if not half:
                if l[j] <= stop:
                    R = -1.0; xj = j; break
                if h[j] >= T1:
                    r1 = rr1; half = True; xj = j
                    if T2 is None:
                        R = r1; break
            else:
                if l[j] <= e:                                # BE stop on remainder
                    R = 0.5 * r1 + 0.0; xj = j; break
                if h[j] >= T2:
                    R = 0.5 * r1 + 0.5 * ((T2 - e) / risk); xj = j; break
        if R is None:
            R = ((c[xj] - e) / risk) if not half else (0.5 * r1 + 0.5 * ((c[xj] - e) / risk))
        R -= cost / risk * e
        rows.append(dict(time=d.index[i], entry=e, stop=stop, T1=T1,
                         T2=(T2 if T2 else np.nan), rr1=rr1, R=R, year=d.index[i].year))
    return pd.DataFrame(rows), len(ents), n_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--side", default="long", choices=["long", "short"])
    ap.add_argument("--pivot-left", type=int, default=5)
    ap.add_argument("--pivot-right", type=int, default=5)
    ap.add_argument("--stop", default="swing", choices=["swing", "pullback"],
                    help="swing=直近安値(既定) / pullback=押し安値")
    ap.add_argument("--fwd", type=int, default=500)
    ap.add_argument("--cost", type=float, default=0.0005)
    ap.add_argument("--dump", type=int, default=0, help="サンプルトレードをN件表示")
    a = ap.parse_args()

    d = resample(load_mt5_csv(a.csv), a.tf)
    h, l = d["high"].values, d["low"].values
    atr = ta.atr(d["high"], d["low"], d["close"], length=14).values
    highs, lows = local_pivots(h, l, a.pivot_left, a.pivot_right)
    t, ne, nt = build_trades(d, a.side, highs, lows, a.stop, a.fwd, a.cost, atr)

    print(f"\n=== Wave-3 lab: {os.path.basename(a.csv)} {a.tf} {a.side} "
          f"(pivot L{a.pivot_left}/R{a.pivot_right}, stop={a.stop}) ===")
    print(f"  {d.index[0].date()} -> {d.index[-1].date()}  |  pivot高値 {len(highs)}本 / 安値 {len(lows)}本")
    print(f"  エントリー {ne}  → 目標あり {nt} ({nt/max(ne,1)*100:.0f}%)  → RR≥1通過 {len(t)}")
    if len(t) >= 5:
        be = "n/a"
        print(f"  win={ (t.R>0).mean()*100:.0f}%  meanR={t.R.mean():+.3f}  totR={t.R.sum():+.0f}  "
              f"平均RR(T1)={t.rr1.mean():.2f}  中央RR(T1)={t.rr1.median():.2f}")
        print(f"  IS(<2022)={t[t.year<2022].R.mean():+.3f}  OOS(>=2022)={t[t.year>=2022].R.mean():+.3f}")
        print("  per-year totR: " + " ".join(f"{y}:{g.R.sum():+.0f}" for y, g in t.groupby("year")))
    else:
        print(f"  通過{len(t)}本（少なすぎ）")

    if a.dump and len(t):
        print(f"\n  --- サンプルトレード {min(a.dump,len(t))}件（水準を目視確認用） ---")
        print(f"  {'time':>16} {'entry':>9} {'stop':>9} {'T1(戻り)':>9} {'T2(直近)':>9} {'RR1':>5} {'R':>6}")
        for _, r in t.head(a.dump).iterrows():
            print(f"  {str(r.time)[:16]:>16} {r.entry:>9.3f} {r.stop:>9.3f} {r.T1:>9.3f} "
                  f"{r.T2:>9.3f} {r.rr1:>5.2f} {r.R:>+6.2f}")


if __name__ == "__main__":
    main()
