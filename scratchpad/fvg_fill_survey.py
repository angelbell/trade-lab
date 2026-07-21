"""FVGの「埋め」の性質測定（戦略でなく性質。コスト無し。バウンス順序の第1段=率）。

問い: (1) FVGは「ランダムな同距離・同幅の水準」より埋まりやすい/速いのか（距離マッチング対照）。
     (2) 埋まるFVGと埋まらないFVGの、生成時点で既知の差は何か。
     (3) 「TFが高いほど埋まる」は暦時間を揃えて真か。

検出 = 現行仕様（src/engine/size._bullish_fvg_size と同一の3本組定義・size>=0.15ATR、
ATRは確定した candle3 のATR14）。candle3 の確定で生成、計測は次バーから（先読み無し）。
埋めの3水準（bullish）: 近位端 c3.low タッチ / CE=帯50% / 遠位端 c1.high（全埋め）。bearish は鏡像。
対照 = 各FVGにつきランダム3バー、その終値から同じATR距離・同じATR幅の擬似帯（シード固定）。

Run: .venv/bin/python scratchpad/fvg_fill_survey.py [--smoke] 2>/dev/null | tee scratchpad/out_fvg_fill_survey.txt
"""
import argparse, heapq, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_mt5_csv
from breakout_wave import resample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS = {"eurusd": "data/vantage_eurusd_m15.csv", "usdjpy": "data/vantage_usdjpy_m15.csv",
        "gold": "data/vantage_xauusd_m15.csv", "btcusd": "data/vantage_btcusd_m15.csv"}
TFS = ["15min", "60min", "4h"]
MIN_ATR = 0.15
HORIZONS = [1, 5, 20, 90]          # 暦日
N_CTRL = 3
RNG = np.random.default_rng(20260717)
NBOOT = 500


def detect(h, l, a, min_atr):
    """3本組FVG（ベクトル化）。返り値 = [(k=確定バー, 近位端, 遠位端, size_atr)]。bull: l[i+2]>h[i]。"""
    av = a[2:]
    ok = np.isfinite(av) & (av > 0)
    kb = np.arange(2, len(h))
    gap_b = l[2:] - h[:-2]                     # bull: >0 で帯 [h[i], l[i+2]]
    mb = ok & (gap_b > 0) & (gap_b / av >= min_atr)
    bull = list(zip(kb[mb], l[2:][mb], h[:-2][mb], (gap_b / av)[mb]))
    gap_s = l[:-2] - h[2:]                     # bear: >0 で帯 [h[i+2], l[i]]
    ms = ok & (gap_s > 0) & (gap_s / av >= min_atr)
    bear = list(zip(kb[ms], h[2:][ms], l[:-2][ms], (gap_s / av)[ms]))
    return bull, bear


def first_touch(low_or_high, start, levels, is_bull):
    """スイープライン: units[u]=(開始バー, [lv_near, lv_ce, lv_full])。
    bull: low[j] <= lv で到達（大きい水準から確定＝maxヒープ）。bear: high[j] >= lv（minヒープ）。
    返り値 = touched[u][e] = バーindex or -1。"""
    m = len(start)
    order = np.argsort(start, kind="stable")
    res = np.full((m, 3), -1, dtype=np.int64)
    heaps = [[], [], []]
    ptr = 0
    x = low_or_high
    n = len(x)
    sgn = -1.0 if is_bull else 1.0            # bullはmaxヒープ（負値で持つ）
    for j in range(n):
        while ptr < m and start[order[ptr]] <= j:
            u = order[ptr]
            for e in range(3):
                heapq.heappush(heaps[e], (sgn * levels[u][e], u))
            ptr += 1
        xv = x[j]
        for e in range(3):
            hp = heaps[e]
            if is_bull:
                while hp and -hp[0][0] >= xv:
                    _, u = heapq.heappop(hp)
                    if res[u][e] < 0:
                        res[u][e] = j
            else:
                while hp and hp[0][0] <= xv:
                    _, u = heapq.heappop(hp)
                    if res[u][e] < 0:
                        res[u][e] = j
    return res


def build_units(fvgs, c, a, n, is_bull):
    """実FVG＋各3対照の (start, levels, meta) を作る。levels=[近位, CE, 全埋め]。
    対照 = ランダムバー r の終値から、実物と同じATR距離・幅の帯。"""
    starts, levels, meta = [], [], []
    valid = np.where(np.isfinite(a) & (a > 0))[0]
    valid = valid[(valid >= 2) & (valid < n - 1)]
    for (k, top, bottom, sz) in fvgs:
        if k + 1 >= n:
            continue
        ce = 0.5 * (top + bottom)
        starts.append(k + 1); levels.append([top, ce, bottom])
        meta.append(("real", k, sz))
        d_top = (c[k] - top) / a[k] if is_bull else (top - c[k]) / a[k]
        width = abs(top - bottom) / a[k]
        for r in RNG.choice(valid, size=N_CTRL, replace=True):
            if is_bull:
                t2 = c[r] - d_top * a[r]; b2 = t2 - width * a[r]
            else:
                t2 = c[r] + d_top * a[r]; b2 = t2 + width * a[r]
            starts.append(r + 1); levels.append([t2, 0.5 * (t2 + b2), b2])
            meta.append(("ctrl", r, sz))
    return np.array(starts), levels, meta


def rates_at(t2e_days, create_pos, ts, horizon):
    """打ち切り込みの到達率: 地平線ぶんの未来が存在する生成だけ分母に。"""
    end = ts[-1]
    ok = (end - ts[create_pos]) / np.timedelta64(1, "D") >= horizon
    if ok.sum() == 0:
        return np.nan, 0
    hit = (t2e_days[ok] >= 0) & (t2e_days[ok] <= horizon)
    return 100.0 * hit.mean(), int(ok.sum())


def cell(sym, tf, d, min_atr, feat_rows):
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    a = ta.atr(d["high"], d["low"], d["close"], length=14).values
    ts = d.index.values.astype("datetime64[ns]")
    n = len(c)
    er = (pd.Series(c, index=d.index).diff(20).abs()
          / pd.Series(c, index=d.index).diff().abs().rolling(20).sum()).values
    dly = d["close"].resample("1D").last().dropna()
    up = (dly > dly.rolling(150).mean()).shift(1)
    up_bar = up.reindex(d.index, method="ffill").fillna(False).values

    bull, bear = detect(h, l, a, min_atr)
    out = []
    for is_bull, fvgs, tag in ((True, bull, "bull"), (False, bear, "bear")):
        if len(fvgs) < 30:
            print(f"  {sym:<7}{tf:<6}{tag:<5} n={len(fvgs)} 少なすぎskip"); continue
        starts, levels, meta = build_units(fvgs, c, a, n, is_bull)
        res = first_touch(l if is_bull else h, starts, levels, is_bull)
        kinds = np.array([m[0] for m in meta])
        kpos = np.array([m[1] for m in meta])
        t2e = np.full(res.shape, -1.0)
        for e in range(3):
            hitmask = res[:, e] >= 0
            t2e[hitmask, e] = (ts[res[hitmask, e]] - ts[kpos[hitmask]]) / np.timedelta64(1, "D")
        rmask, cmask = kinds == "real", kinds == "ctrl"
        row = dict(sym=sym, tf=tf, dir=tag, n=int(rmask.sum()))
        for hz in HORIZONS:
            for e, en in enumerate(("near", "ce", "full")):
                row[f"{en}@{hz}"], _ = rates_at(t2e[rmask, e], kpos[rmask], ts, hz)
                row[f"c_{en}@{hz}"], _ = rates_at(t2e[cmask, e], kpos[cmask], ts, hz)
        full_hit = t2e[rmask, 2]
        hit20 = full_hit[(full_hit >= 0) & (full_hit <= 20)]
        row["t_med"] = float(np.median(hit20)) if len(hit20) else np.nan
        row["t_q25"] = float(np.percentile(hit20, 25)) if len(hit20) else np.nan
        row["t_q75"] = float(np.percentile(hit20, 75)) if len(hit20) else np.nan
        # ブートストラップCI（生成月クラスタ再抽出）: 20日全埋め率の 実−対照
        mon_r = pd.PeriodIndex(pd.DatetimeIndex(ts[kpos[rmask]]), freq="M")
        mon_c = pd.PeriodIndex(pd.DatetimeIndex(ts[kpos[cmask]]), freq="M")
        umon = mon_r.unique()
        d_r, d_c = t2e[rmask, 2], t2e[cmask, 2]
        p_r, p_c = kpos[rmask], kpos[cmask]
        gr = {m: np.where(mon_r == m)[0] for m in umon}      # 月→index を1度だけ
        gc = {m: np.where(mon_c == m)[0] for m in umon}
        diffs = []
        for _ in range(NBOOT):
            pick = RNG.choice(len(umon), size=len(umon), replace=True)
            mr = np.concatenate([gr[umon[i]] for i in pick])
            mc = np.concatenate([gc.get(umon[i], np.empty(0, dtype=int)) for i in pick])
            a1, _ = rates_at(d_r[mr], p_r[mr], ts, 20)
            a2, _ = rates_at(d_c[mc], p_c[mc], ts, 20)
            if np.isfinite(a1) and np.isfinite(a2):
                diffs.append(a1 - a2)
        row["diff20"] = row["full@20"] - row["c_full@20"]
        row["ci_lo"], row["ci_hi"] = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
        out.append(row)
        # 選別可否テーブル用: 実FVGの特徴と 全埋め@20日
        kr = kpos[rmask]
        okh = (ts[-1] - ts[kr]) / np.timedelta64(1, "D") >= 20
        filled = (t2e[rmask, 2] >= 0) & (t2e[rmask, 2] <= 20)
        sizes = np.array([m[2] for m in meta])[rmask]
        disp = np.sign(c[kr] - c[np.maximum(kr - 20, 0)])
        align = (disp > 0) if is_bull else (disp < 0)
        for i in np.where(okh)[0]:
            feat_rows.append(dict(sym=sym, tf=tf, dir=tag, filled=bool(filled[i]),
                                  size=sizes[i], er=er[kr[i]], trend_up=bool(up_bar[kr[i]]),
                                  hour=pd.Timestamp(ts[kr[i]]).hour, align=bool(align[i])))
    return out


def auc(x, y):
    """yで層別したxのAUC（Mann-Whitney rank法）。"""
    x = np.asarray(x, float); y = np.asarray(y, bool)
    if y.sum() < 20 or (~y).sum() < 20:
        return np.nan
    r = pd.Series(x).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    syms = {"eurusd": SYMS["eurusd"]} if args.smoke else SYMS
    tfs = ["15min"] if args.smoke else TFS

    rows, feat_rows = [], []
    for sym, path in syms.items():
        df = load_mt5_csv(f"{ROOT}/{path}")
        if sym == "usdjpy":
            df = df.loc["2000-01-01":]
        if args.smoke:
            df = df.loc["2024-07-01":]
        for tf in tfs:
            d = resample(df, tf)
            rows.extend(cell(sym, tf, d, MIN_ATR, feat_rows))
    # 感度: 存在のみ（サイズ床なし）を eurusd 15m だけ
    if not args.smoke:
        d = resample(load_mt5_csv(f"{ROOT}/{SYMS['eurusd']}"), "15min")
        rows.extend([{**r, "sym": "eurusd(size>=0)"} for r in cell("eurusd", "15min", d, 0.0, [])])

    print(f"\n===== 埋め率表（実 vs 距離マッチング対照。full=全埋め・ce=帯50%・near=帯タッチ。%）=====")
    print(f"  {'sym':<15}{'tf':<6}{'dir':<5}{'n':>6} | {'full@5d':>8}{'対照':>6}{'full@20d':>9}{'対照':>6}"
          f"{'差@20':>7}{'CI95':>14}{'full@90d':>9} | {'ce@20':>6}{'near@20':>8} | 埋め日数 med[q25,q75]")
    for r in rows:
        ci = f"[{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}]" if np.isfinite(r.get("ci_lo", np.nan)) else "-"
        print(f"  {r['sym']:<15}{r['tf']:<6}{r['dir']:<5}{r['n']:>6} | {r['full@5']:>7.1f}%{r['c_full@5']:>5.1f}%"
              f"{r['full@20']:>8.1f}%{r['c_full@20']:>5.1f}%{r['diff20']:>+7.1f}{ci:>14}{r['full@90']:>8.1f}%"
              f" | {r['ce@20']:>5.1f}%{r['near@20']:>7.1f}% | {r['t_med']:.2f}[{r['t_q25']:.2f},{r['t_q75']:.2f}]日")

    if feat_rows:
        F = pd.DataFrame(feat_rows)
        print(f"\n===== 選別可否: 生成時点の特徴 → 全埋め@20日 の AUC（TF×方向でプール、比較本数=特徴5×セル）=====")
        for (tf, dr), g in F.groupby(["tf", "dir"]):
            a_sz = auc(g["size"], g["filled"]); a_er = auc(g["er"].fillna(0), g["filled"])
            a_tr = auc(g["trend_up"].astype(float), g["filled"])
            a_al = auc(g["align"].astype(float), g["filled"])
            a_hr = auc(g["hour"].astype(float), g["filled"])
            print(f"  {tf:<6}{dr:<5} n={len(g):>6} 埋め率={100*g['filled'].mean():>5.1f}%  "
                  f"AUC: size={a_sz:.3f} ER={a_er:.3f} 日足トレンド={a_tr:.3f} 変位整合={a_al:.3f} 時刻={a_hr:.3f}")
        print("  （AUC 0.5=情報なし。0.45-0.55 は実務上ゼロと読む）")


if __name__ == "__main__":
    main()
