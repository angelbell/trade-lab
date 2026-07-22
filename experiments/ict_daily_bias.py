"""ユーザーの指摘（2026-07-14）:「日足の分析が前提にあるけど、それは？」
前回は「バイアス有り/無しの集計値」を比べただけ＝バイアスが方向を当てているかを直接訊いていない。
正しい訊き方: 日足バイアスが上の日に、ロングはショートより儲かるのか（逆も）。

各日足バイアス定義について、その日の ICT ロング/ショートの素meanR を条件付きで出す:
    lift = [meanR(long|up) - meanR(short|up)] + [meanR(short|dn) - meanR(long|dn)]  / 2
  日足に方向の中身があれば lift > 0。ゼロなら「日足は何も言っていない」。
帰無: 日付をシャッフルしたバイアス（同じ up 比率）500回 の lift 分布と比べる。
Run: .venv/bin/python experiments/ict_daily_bias.py"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from ict_killzone import (load_ny, find_entries, price_and_scan, SYMS,
                          STOPBUF_DEFAULT, RR_DEFAULT, LONDON_HOURS)
from breakout_wave import resample, kama_adaptive

RNG = np.random.default_rng(20260714)


def daily_biases(df):
    """前日までに確定した日足だけから作る（先読み厳禁）。True=強気, False=弱気, NaN=判定不能。"""
    b = df.set_index("broker_dt")[["open", "high", "low", "close"]]
    d = resample(b, "1D")
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    sma = c.rolling(150).mean()
    kama = kama_adaptive(c, 14)
    defs = {
        "D1 前日陽線(動画)":        c > o,
        "D2 日足SMA150 上向き":     sma > sma.shift(1),
        "D3 日足KAMA(14) 上向き":   kama > kama.shift(1),
        "D4 日足HH+HL(構造)":       (h > h.shift(1)) & (l > l.shift(1)),
        "D5 5日モメンタム":          c > c.shift(5),
        "D6 前日終値が日中レンジ上半分": (c - l) > (h - c),
        "D7 前日高値を超えて引け":     c > h.shift(1),
    }
    out = {}
    for k, s in defs.items():
        v = s.astype(float)
        v[s.isna()] = np.nan
        # 「前日までに確定」= その日足の翌ブローカー日の頭で有効になる
        conf = (d.index + pd.Timedelta(days=1)).tz_localize(
            "Europe/Riga", ambiguous="NaT", nonexistent="shift_forward"
        ).tz_convert("America/New_York").tz_localize(None)
        keep = ~conf.isna()
        out[k] = pd.DataFrame({"conf": conf[keep], "v": v.values[keep]}).sort_values("conf")
    return out


def bias_per_day(dates, tl):
    q = pd.DataFrame({"date": dates,
                      "t": [pd.Timestamp(x) + pd.Timedelta(hours=LONDON_HOURS[0]) for x in dates]}
                     ).sort_values("t")
    m = pd.merge_asof(q, tl, left_on="t", right_on="conf", direction="backward")
    return dict(zip(m["date"], m["v"]))


print(f"{'銘柄':8s} {'日足バイアスの定義':26s} {'up日%':>6} "
      f"{'L|up':>7} {'S|up':>7} {'L|dn':>7} {'S|dn':>7} {'lift':>7} {'null中央値':>9} {'%ile':>6}")
for name in ("gold", "eurusd", "gbpusd", "usdjpy", "audusd", "btcusd"):
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
    recs = price_and_scan(df, find_entries(df), STOPBUF_DEFAULT, RR_DEFAULT)
    rows = []
    for r in recs:
        for side in ("long", "short"):
            x = r[side]
            if x.get("valid") == "ok" and x.get("filled"):
                rows.append((r["date"], side, x["hold_R"]))
    T = pd.DataFrame(rows, columns=["date", "side", "R"])
    tls = daily_biases(df)
    for label, tl in tls.items():
        bmap = bias_per_day(sorted(T["date"].unique()), tl)
        T["b"] = T["date"].map(bmap)
        t = T.dropna(subset=["b"])
        up = t["b"] == 1.0
        lu = t[up & (t["side"] == "long")]["R"]
        su = t[up & (t["side"] == "short")]["R"]
        ld = t[~up & (t["side"] == "long")]["R"]
        sd = t[~up & (t["side"] == "short")]["R"]
        if min(len(lu), len(su), len(ld), len(sd)) < 50:
            continue
        lift = ((lu.mean() - su.mean()) + (sd.mean() - ld.mean())) / 2
        # 帰無: 日付とバイアスの対応をシャッフル（up比率は保存）
        days = np.array(sorted(bmap.keys()))
        bvals = np.array([bmap[d] for d in days], dtype=float)
        ok = np.isfinite(bvals)
        draws = []
        Rv = t["R"].values; sv = (t["side"] == "long").values
        dv = t["date"].values
        for _ in range(500):
            sh = bvals.copy()
            sh[ok] = RNG.permutation(bvals[ok])
            m = dict(zip(days, sh))
            bb = np.array([m[d] for d in dv], dtype=float)
            u = bb == 1.0
            a = Rv[u & sv].mean(); b_ = Rv[u & ~sv].mean()
            c_ = Rv[~u & ~sv].mean(); d_ = Rv[~u & sv].mean()
            draws.append(((a - b_) + (c_ - d_)) / 2)
        draws = np.array(draws)
        pc = 100 * (lift > draws).mean()
        print(f"{name:8s} {label:26s} {100*up.mean():5.0f}% "
              f"{lu.mean():+7.3f} {su.mean():+7.3f} {ld.mean():+7.3f} {sd.mean():+7.3f} "
              f"{lift:+7.3f} {np.median(draws):+9.3f} {pc:5.0f}%")
    print()
