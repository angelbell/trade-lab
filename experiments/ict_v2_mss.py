"""ICT v2 — 本体（流動性の狩り + 市場構造の転換 MSS）を入れた忠実版（2026-07-14）。

v1 (`ict_killzone.py`) は「ロンドン窓の最安値 → 0.5ATR 反発 → NYキルゾーンでOTE指値」だった。
これは ICT の中核2つを欠いていた:
  1. 流動性の狩り（sweep）: ロンドンの安値は「アジア安値 or 前日安値を取りに行った」ものでなければならない
  2. 市場構造の転換（MSS）: 狩った後に「直前のスイング高値を上抜ける」ことで反転が確定する
  → MSS を入れると入口の性格が「反発（フェード）」から「継続」に変わる。v1 で見つかった
     「日足バイアスが逆を向いている」は、MSS 欠落の産物かもしれない。ここを再検証する。

機構（ロング。ショートは完全な鏡像）— 全て NY 時刻:
  アジア窓   : 前日 19:00 〜 当日 02:00   （レンジ＝流動性の溜まり場）
  ロンドン窓 : 当日 02:00 〜 07:00
  NYキルゾーン: 当日 07:00 〜 10:00        （＝日本時間 20:00-22:00 付近）
  1. sweep : ロンドン窓の安値 L が {アジア安値 / 前日安値 / どちらか} を下回る
  2. MSS   : L の後、ロンドン窓の終わりまでに、L 直前の「直近スイング高値」(3本フラクタル) を
             {終値 / ヒゲ} で上抜ける。その上抜けまでの最高値を H とする（displacement leg = L→H）
  3. entry : NYキルゾーンで 買い指値 lim = H - f*(H-L)
  4. stop  : L - buf*ATR / target: entry + RR*(entry-stop)（価格で固定）
  5. 無効化: キルゾーン到達前に L を割ったら、その日は無効

執行は最初から現実版:
  ASK基準の指値約定（BIDデータでは low <= lim - 生スプレッド でしか約定しない）＋ コストは手数料のみ
  （ASKで買いBIDで決済する時点でスプレッドは既に損切り幅に埋まっているので、二重計上しない）

Run: .venv/bin/python experiments/ict_v2_mss.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_killzone import load_ny, SYMS
from breakout_wave import resample, kama_adaptive

RNG = np.random.default_rng(20260714)

ASIA = (-1, 19, 0, 2)      # (day offset, start hour) -> (day offset, end hour): 前日19:00 〜 当日02:00
LONDON = (2, 7)            # 当日 02:00 - 07:00 (NY)
KZ = (7, 10)               # 当日 07:00 - 10:00 (NY)
FWD_CAP = 500
ATR_LEN = 14

F_LIST = [0.25, 0.30, 0.50, 0.62, 0.705, 0.79]
RR_LIST = [1.5, 2.0, 3.0, 4.0]
F_CANON, RR_CANON, BUF_CANON = 0.705, 2.0, 0.1

# (fill spread, commission-only cost) — CLAUDE.md の実測値を物理的に分解
MODEL = {
    "gold":   (0.15, 0.06),
    "eurusd": (0.3e-4, 0.6e-4), "gbpusd": (0.3e-4, 0.6e-4), "audusd": (0.3e-4, 0.6e-4),
    "usdjpy": (0.3e-2, 0.6e-2),
    "btcusd": (15.0, 0.0),
}
SWEEPS = ["asia", "pdl", "either"]
MSSDEF = ["close", "wick"]


# ---------------------------------------------------------------------------
def prep(df):
    """ny_wall を datetime64 の昇順配列にし、日ごとの窓を timestamp で切れるようにする。
    （アジア窓は前日19:00から始まるので、ny_date のグループ分けでは切れない）"""
    ny = pd.to_datetime(df["ny_wall"].values)
    df = df.copy()
    df["_t"] = ny
    dates = np.array(sorted(set(pd.DatetimeIndex(ny).normalize())))
    return df, ny.values.astype("datetime64[ns]"), dates


def window_pos(tarr, t0, t1):
    """[t0, t1) に入るバーの位置（tarr は昇順）"""
    a = np.searchsorted(tarr, np.datetime64(t0), "left")
    b = np.searchsorted(tarr, np.datetime64(t1), "left")
    return a, b


def prev_day_extremes(df, dates):
    """NY暦の前日の高値/安値（ICT の "previous day high/low"）"""
    g = df.groupby(df["_t"].dt.normalize()).agg(hi=("high", "max"), lo=("low", "min"))
    hi = g["hi"].reindex(dates).shift(1)
    lo = g["lo"].reindex(dates).shift(1)
    return dict(zip(dates, hi.values)), dict(zip(dates, lo.values))


def last_fractal_high(highs, s, e):
    """[s, e) の中で最後の 3本フラクタル高値の位置（無ければ None）"""
    for k in range(e - 2, s, -1):
        if highs[k] >= highs[k - 1] and highs[k] >= highs[k + 1]:
            return k
    return None


def last_fractal_low(lows, s, e):
    for k in range(e - 2, s, -1):
        if lows[k] <= lows[k - 1] and lows[k] <= lows[k + 1]:
            return k
    return None


def build_setups(df, tarr, dates, sweep, mss):
    """f から独立な部分（sweep 判定 / MSS / displacement leg L->H / KZ のバー範囲）を1回だけ作る。"""
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    atr = df["atr14"].values
    pdh, pdl = prev_day_extremes(df, dates)
    out = []
    stat = {"no_bars": 0, "no_sweep": 0, "no_mss": 0, "broken": 0, "ok": 0}
    for d in dates:
        day = pd.Timestamp(d)
        a0, a1 = window_pos(tarr, day + pd.Timedelta(days=ASIA[0], hours=ASIA[1]),
                            day + pd.Timedelta(hours=ASIA[3]))
        l0, l1 = window_pos(tarr, day + pd.Timedelta(hours=LONDON[0]),
                            day + pd.Timedelta(hours=LONDON[1]))
        k0, k1 = window_pos(tarr, day + pd.Timedelta(hours=KZ[0]),
                            day + pd.Timedelta(hours=KZ[1]))
        rec = {"date": d, "long": None, "short": None}
        if (a1 - a0) < 4 or (l1 - l0) < 6 or (k1 - k0) < 2 or not np.isfinite(atr[l1 - 1]):
            stat["no_bars"] += 1
            out.append(rec); continue
        A = atr[l1 - 1]
        asia_lo, asia_hi = lo[a0:a1].min(), hi[a0:a1].max()
        p_lo, p_hi = pdl.get(d, np.nan), pdh.get(d, np.nan)

        for side in ("long", "short"):
            if side == "long":
                iL = l0 + int(np.argmin(lo[l0:l1])); L = lo[iL]
                tgt_lo = {"asia": asia_lo, "pdl": p_lo,
                          "either": np.nanmax([asia_lo, p_lo])}[sweep]   # 高いほうを取れば「どちらかを割った」
                if sweep == "either":
                    swept = (np.isfinite(asia_lo) and L < asia_lo) or (np.isfinite(p_lo) and L < p_lo)
                else:
                    swept = np.isfinite(tgt_lo) and L < tgt_lo
                if not swept:
                    stat["no_sweep"] += 1; continue
                # MSS: iL 直前のスイング高値を上抜け
                sh = last_fractal_high(hi, a0, iL)
                if sh is None:
                    stat["no_mss"] += 1; continue
                lvl = hi[sh]
                jm = None
                for j in range(iL + 1, l1):
                    if (cl[j] > lvl) if mss == "close" else (hi[j] > lvl):
                        jm = j; break
                if jm is None:
                    stat["no_mss"] += 1; continue
                iH = iL + int(np.argmax(hi[iL:jm + 1])); H = hi[iH]
                if H - L < 0.25 * A:
                    stat["no_mss"] += 1; continue
                if (lo[jm + 1:l1] <= L).any():          # KZ 到達前に構造が壊れた
                    stat["broken"] += 1; continue
                stat["ok"] += 1
                rec["long"] = dict(L=L, H=H, atr=A, kz=(k0, k1), mss_pos=jm)
            else:
                iH = l0 + int(np.argmax(hi[l0:l1])); Hh = hi[iH]
                if sweep == "either":
                    swept = (np.isfinite(asia_hi) and Hh > asia_hi) or (np.isfinite(p_hi) and Hh > p_hi)
                else:
                    tgt_hi = {"asia": asia_hi, "pdl": p_hi}[sweep]
                    swept = np.isfinite(tgt_hi) and Hh > tgt_hi
                if not swept:
                    continue
                sl = last_fractal_low(lo, a0, iH)
                if sl is None:
                    continue
                lvl = lo[sl]
                jm = None
                for j in range(iH + 1, l1):
                    if (cl[j] < lvl) if mss == "close" else (lo[j] < lvl):
                        jm = j; break
                if jm is None:
                    continue
                iL = iH + int(np.argmin(lo[iH:jm + 1])); Ll = lo[iL]
                if Hh - Ll < 0.25 * A:
                    continue
                if (hi[jm + 1:l1] >= Hh).any():
                    continue
                rec["short"] = dict(L=Ll, H=Hh, atr=A, kz=(k0, k1), mss_pos=jm)
        out.append(rec)
    return out, stat


def walk(df, setups, f, rr, buf, spread, cost, side):
    """ASK基準の指値約定 + 前進走査（約定足も損切り判定に含める。同足タイブレークは損切り優先）。"""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    trades = []
    for rec in setups:
        s = rec[side]
        if s is None:
            continue
        L, H, A = s["L"], s["H"], s["atr"]
        k0, k1 = s["kz"]
        if side == "long":
            lim = H - f * (H - L)
            stop = L - buf * A
            if lim <= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if l[p] <= lim - spread:
                    fp = p; break
            if fp is None:
                continue
            entry = min(lim, o[fp] + spread)
            risk = entry - stop
            if risk <= 0:
                continue
            tgt = entry + rr * risk
            R = None
            for p in range(fp, min(fp + FWD_CAP, n)):
                if l[p] <= stop:
                    R = -1.0; break
                if h[p] >= tgt:
                    R = rr; break
            if R is None:
                R = (c[min(fp + FWD_CAP, n) - 1] - entry) / risk
        else:
            lim = L + f * (H - L)
            stop = H + buf * A
            if lim >= stop:
                continue
            fp = None
            for p in range(k0, k1):
                if h[p] >= lim + spread:
                    fp = p; break
            if fp is None:
                continue
            entry = max(lim, o[fp] - spread)
            risk = stop - entry
            if risk <= 0:
                continue
            tgt = entry - rr * risk
            R = None
            for p in range(fp, min(fp + FWD_CAP, n)):
                if h[p] >= stop:
                    R = -1.0; break
                if l[p] <= tgt:
                    R = rr; break
            if R is None:
                R = (entry - c[min(fp + FWD_CAP, n) - 1]) / risk
        trades.append((rec["date"], R - cost / risk, R, risk))
    return trades


def stats(tr, span):
    if len(tr) < 10:
        return None
    net = np.array([t[1] for t in tr]); g = np.array([t[2] for t in tr])
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    cum = np.cumsum(net); dd = float((np.maximum.accumulate(cum) - cum).max())
    yrs = np.array([pd.Timestamp(t[0]).year for t in tr])
    by = pd.Series(net).groupby(yrs).sum()
    half = len(net) // 2
    return dict(n=len(net), npy=len(net) / span, win=100 * (g > 0).mean(),
                gross=g.mean(), net=net.mean(), pf=pos / neg if neg > 0 else np.inf,
                tot=net.sum(), dd=dd, IS=net[:half].sum(), OOS=net[half:].sum(),
                gy=100 * (by > 0).mean(), ny=len(by))


def fmt(s, be):
    if s is None:
        return "  (n<10)"
    return (f"n={s['n']:5d} n/yr={s['npy']:5.1f} win={s['win']:5.1f}(be{be:4.1f}) "
            f"gross={s['gross']:+.3f} net={s['net']:+.3f} PF={s['pf']:5.2f} "
            f"totR={s['tot']:+7.1f} DD={s['dd']:6.1f} IS={s['IS']:+7.1f} OOS={s['OOS']:+7.1f} "
            f"緑年={s['gy']:3.0f}%({s['ny']})")


def main():
    data, spans = {}, {}
    for name in SYMS:
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
        df, tarr, dates = prep(df)
        data[name] = (df, tarr, dates)
        y = pd.to_datetime(df["broker_dt"]).dt.year.value_counts()
        spans[name] = int((y > 5000).sum())      # 実質データのある年数

    print("=" * 130)
    print("TABLE A: 機構バリアント（狩る対象 × MSSの定義）— 何日が生き残るか（gold, 全期間）")
    print("=" * 130)
    df, tarr, dates = data["gold"]
    for sw in SWEEPS:
        for ms in MSSDEF:
            _, st = build_setups(df, tarr, dates, sw, ms)
            tot = len(dates)
            print(f"  sweep={sw:6s} MSS={ms:5s}  全NY日 {tot:5d} → 窓あり {tot-st['no_bars']:5d} "
                  f"→ 狩り成立 {tot-st['no_bars']-st['no_sweep']:5d} → MSS成立 "
                  f"{st['ok']+st['broken']:5d} → KZまで構造維持 {st['ok']:5d} "
                  f"({100*st['ok']/max(tot-st['no_bars'],1):4.1f}% of 窓あり日)")

    print("\n" + "=" * 130)
    print(f"TABLE B: 6機構 × 6銘柄 × 両サイド（ICT正典パラメータ f={F_CANON} RR={RR_CANON} buf={BUF_CANON}）")
    print("        執行=ASK基準の指値約定 + 手数料のみ。RR2 の損益分岐勝率 = 33.3%")
    print("=" * 130)
    cache = {}
    best = []
    for sw in SWEEPS:
        for ms in MSSDEF:
            print(f"\n--- sweep={sw} / MSS={ms} ---")
            for name in ("gold", "eurusd", "gbpusd", "usdjpy", "audusd", "btcusd"):
                df, tarr, dates = data[name]
                key = (name, sw, ms)
                if key not in cache:
                    cache[key] = build_setups(df, tarr, dates, sw, ms)[0]
                sp, cost = MODEL[name]
                for side in ("long", "short"):
                    tr = walk(df, cache[key], F_CANON, RR_CANON, BUF_CANON, sp, cost, side)
                    s = stats(tr, spans[name])
                    print(f"  {name:7s} {side:5s} {fmt(s, 33.3)}")
                    if s and s["n"] >= 50:
                        best.append((s["net"], name, side, sw, ms, s))

    print("\n" + "=" * 130)
    print("TABLE C: 台地スイープ（戻り深さ f × RR）— 最も本数の多い機構 sweep=either/MSS=wick")
    print("        『小さく戻る』(動画, f=0.25-0.30) と ICT の OTE (f=0.62-0.79) を並べる")
    print("=" * 130)
    for name in ("gold", "eurusd", "gbpusd", "btcusd"):
        df, tarr, dates = data[name]
        key = (name, "either", "wick")
        if key not in cache:
            cache[key] = build_setups(df, tarr, dates, "either", "wick")[0]
        sp, cost = MODEL[name]
        for side in ("long", "short"):
            print(f"\n  [{name} / {side}]  (net meanR — 上段) / (n — 下段)")
            hdr = "    f\\RR " + "".join(f"{rr:>10.1f}" for rr in RR_LIST)
            print(hdr)
            for f in F_LIST:
                row, ns = [], []
                for rr in RR_LIST:
                    s = stats(walk(df, cache[key], f, rr, BUF_CANON, sp, cost, side), spans[name])
                    row.append(f"{s['net']:+10.3f}" if s else "       n/a")
                    ns.append(f"{s['n']:10d}" if s else "       n/a")
                print(f"   {f:5.3f} " + "".join(row))
                print(f"         " + "".join(ns))

    print("\n" + "=" * 130)
    print("TABLE D: 日足バイアスは今度こそ噛み合うか（MSS 込みの入口に対する条件付き lift）")
    print("        lift = [meanR(L|強気) - meanR(S|強気)] + [meanR(S|弱気) - meanR(L|弱気)] を2で割る")
    print("        > 0 なら日足に方向の中身がある。v1（MSS無し）では 42セル中24セルがマイナスだった。")
    print("=" * 130)
    for name in ("gold", "eurusd", "gbpusd", "usdjpy", "audusd", "btcusd"):
        df, tarr, dates = data[name]
        key = (name, "either", "wick")
        setups = cache[key]
        sp, cost = MODEL[name]
        rows = []
        for side in ("long", "short"):
            for (d, net, g, risk) in walk(df, setups, F_CANON, RR_CANON, BUF_CANON, sp, cost, side):
                rows.append((d, side, g))
        T = pd.DataFrame(rows, columns=["date", "side", "R"])
        if len(T) < 100:
            print(f"  {name}: n<100 — skip"); continue
        b = df.set_index("broker_dt")[["open", "high", "low", "close"]]
        dd = resample(b, "1D")
        o, h, l, c = dd["open"], dd["high"], dd["low"], dd["close"]
        sma = c.rolling(150).mean(); kama = kama_adaptive(c, 14)
        defs = {"D1 前日陽線": c > o, "D2 SMA150↑": sma > sma.shift(1),
                "D3 KAMA↑": kama > kama.shift(1),
                "D4 日足HH+HL": (h > h.shift(1)) & (l > l.shift(1)),
                "D5 5日モメンタム": c > c.shift(5)}
        conf = (dd.index + pd.Timedelta(days=1)).tz_localize(
            "Europe/Riga", ambiguous="NaT", nonexistent="shift_forward"
        ).tz_convert("America/New_York").tz_localize(None)
        keep = ~conf.isna()
        print(f"\n  --- {name} ---")
        for lab, s_ in defs.items():
            v = s_.astype(float); v[s_.isna()] = np.nan
            tl = pd.DataFrame({"conf": conf[keep], "v": v.values[keep]}).sort_values("conf")
            q = pd.DataFrame({"date": sorted(T["date"].unique())})
            q["t"] = pd.to_datetime(q["date"]) + pd.Timedelta(hours=LONDON[0])
            m = pd.merge_asof(q.sort_values("t"), tl, left_on="t", right_on="conf", direction="backward")
            bmap = dict(zip(m["date"], m["v"]))
            T["b"] = T["date"].map(bmap)
            t = T.dropna(subset=["b"])
            up = t["b"] == 1.0
            cells = {k: t[(up == u) & (t["side"] == sd)]["R"]
                     for k, (u, sd) in {"Lu": (True, "long"), "Su": (True, "short"),
                                        "Ld": (False, "long"), "Sd": (False, "short")}.items()}
            if min(len(x) for x in cells.values()) < 20:
                print(f"    {lab:14s} (層が薄い)"); continue
            lift = ((cells["Lu"].mean() - cells["Su"].mean())
                    + (cells["Sd"].mean() - cells["Ld"].mean())) / 2
            print(f"    {lab:14s} up日{100*up.mean():3.0f}%  "
                  f"L|強気{cells['Lu'].mean():+.3f}(n{len(cells['Lu']):4d}) "
                  f"S|強気{cells['Su'].mean():+.3f}(n{len(cells['Su']):4d}) "
                  f"L|弱気{cells['Ld'].mean():+.3f}(n{len(cells['Ld']):4d}) "
                  f"S|弱気{cells['Sd'].mean():+.3f}(n{len(cells['Sd']):4d})  "
                  f"lift={lift:+.3f}")


if __name__ == "__main__":
    main()
