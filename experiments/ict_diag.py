"""ICT killzone の結果を疑う診断（2026-07-14）。

(1) gold の n/年 が過小報告されている疑い（m15 が 2018 以前スカスカ → span 19年で割っている）
(2) 約定足そのもので損切りになる率（タイトな損切りの実務的インパクト）
(3) コスト/R の分布（実コストに落としたら生き返るか）
(4) ★+8h プラセボ窓（NY15-18時）が本物より強い件の解剖:
    (a) エントリー時刻(15/16/17時)別の meanR   -> ロールオーバー(17時)に集中しているか
    (b) 年別                                    -> 昔のスプレッドが広い時代の産物か
    (c) ASK基準の約定（買い指値は low <= lim - spread でしか約定しない）にすると残るか
    (d) 保有時間                                -> 「その窓のトレード」と呼べるのか
Run: .venv/bin/python experiments/ict_diag.py
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import io, contextlib
import numpy as np
import pandas as pd

from ict_killzone import (load_ny, find_entries, price_and_scan, funnel_table,
                          COST_RT, SYMS, LONDON_HOURS, KZ_HOURS, F_DEFAULT,
                          RR_DEFAULT, STOPBUF_DEFAULT, FWD_CAP)

# 1 pip in price units (Vantage bid data). Used for the ASK-based fill test.
PIP = {"eurusd": 0.0001, "gbpusd": 0.0001, "audusd": 0.0001, "usdjpy": 0.01,
       "gold": 0.1, "btcusd": 1.0}


def bars_per_year(df):
    """真の年数 = 実際にバーがある年数（スカスカの年を数えない）。"""
    y = pd.Series(pd.to_datetime(df["broker_dt"]).dt.year)
    cnt = y.value_counts().sort_index()
    dense = cnt[cnt > 5000]          # 1年フルなら m15 で ~24000 本
    return cnt, dense


def trades_from(df, recs, side):
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    out = []
    for r in recs:
        rec = r[side]
        if not (rec.get("valid") == "ok" and rec.get("filled")):
            continue
        fp = rec["entry_pos"]
        stop = rec["stop"]
        same_bar_stop = (l[fp] <= stop) if side == "long" else (h[fp] >= stop)
        out.append(dict(date=r["date"], pos=fp, entry=rec["entry"], risk=rec["risk"],
                        R=rec["hold_R"], reason=rec["hold_reason"],
                        same_bar_stop=bool(same_bar_stop),
                        hour=pd.Timestamp(df["ny_wall"].values[fp]).hour))
    return pd.DataFrame(out)


def exit_pos_of(df, rec, side):
    """hold-exit がどのバーで終わったかを再走査（保有時間の測定用）。"""
    h, l = df["high"].values, df["low"].values
    fp, stop, tgt = rec["entry_pos"], rec["stop"], rec["tgt"]
    n = len(h)
    for p in range(fp, min(fp + FWD_CAP, n)):
        if side == "long":
            if l[p] <= stop or h[p] >= tgt:
                return p
        else:
            if h[p] >= stop or l[p] <= tgt:
                return p
    return min(fp + FWD_CAP, n - 1)


def ask_fill_entries(df, london_hours, kz_hours, f, spread):
    """ASK基準の約定: 買い指値は low <= lim - spread のときだけ約定（BIDデータ上）。
    売り指値は high >= lim + spread。約定価格は lim（ASK=lim で買えた、と仮定）。"""
    o, h, l = df["open"].values, df["high"].values, df["low"].values
    base = find_entries(df, london_hours, kz_hours, f)
    out = []
    for rec in base:
        new = {"date": rec["date"]}
        for side in ("long", "short"):
            r = dict(rec[side])
            if not r.get("filled"):
                new[side] = r
                continue
            L, H = r["L"], r["H"]
            lim = (H - f * (H - L)) if side == "long" else (L + f * (H - L))
            fp = None
            for p in r["kz_bars"]:
                if side == "long" and l[p] <= lim - spread:
                    fp = p; break
                if side == "short" and h[p] >= lim + spread:
                    fp = p; break
            if fp is None:
                new[side] = {"valid": "ok", "filled": False, "reason": "no_fill_ask"}
            else:
                entry = min(lim, o[p] + spread) if side == "long" else max(lim, o[p] - spread)
                new[side] = dict(r, filled=True, entry_pos=fp, entry=entry)
        out.append(new)
    return out


def stat(tr, cost):
    if not len(tr):
        return None
    g = tr["R"].values
    net = g - cost / tr["risk"].values
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    return dict(n=len(g), win=100 * (g > 0).mean(), gross=g.mean(), net=net.mean(),
                pf=pos / neg if neg > 0 else np.inf, tot=net.sum())


def main():
    data = {}
    for name, path in SYMS.items():
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(path, cut2000=(name == "usdjpy"))
        data[name] = df

    print("=" * 96)
    print("D1. 真の年数（m15 のバー本数が年 5000 本未満の年は「実質データ無し」）")
    print("=" * 96)
    true_span = {}
    for name, df in data.items():
        cnt, dense = bars_per_year(df)
        true_span[name] = len(dense)
        thin = [int(y) for y in cnt.index if cnt[y] <= 5000]
        print(f"  {name:7s} dense years = {len(dense)} ({dense.index[0]}-{dense.index[-1]})  "
              f"thin/empty years = {thin}")

    print("\n" + "=" * 96)
    print("D2/D3. 本物のキルゾーン: 約定足で即損切りになる率 / コスト/R の分布 / n per TRUE year")
    print("=" * 96)
    print(f"  {'sym':7s} {'side':5s} {'n':>5} {'n/yr':>6} {'即死%':>6} {'cost/R med':>11} "
          f"{'cost/R mean':>12} {'gross':>7} {'net(canon)':>11} {'net(real cost)':>14}")
    real_cost = {"gold": 0.25, "btcusd": 15.0, "eurusd": 0.00009, "gbpusd": 0.00009,
                 "audusd": 0.00009, "usdjpy": 0.009}
    for name, df in data.items():
        recs = price_and_scan(df, find_entries(df), STOPBUF_DEFAULT, RR_DEFAULT)
        for side in ("long", "short"):
            tr = trades_from(df, recs, side)
            if not len(tr):
                continue
            cr = COST_RT[name] / tr["risk"].values
            crr = real_cost[name] / tr["risk"].values
            g = tr["R"].values
            print(f"  {name:7s} {side:5s} {len(tr):5d} {len(tr)/true_span[name]:6.1f} "
                  f"{100*tr['same_bar_stop'].mean():6.1f} {np.median(cr):11.3f} {cr.mean():12.3f} "
                  f"{g.mean():+7.3f} {(g-cr).mean():+11.3f} {(g-crr).mean():+14.3f}")

    print("\n" + "=" * 96)
    print("D4. +8h プラセボ窓（London=NY10-15h / KZ=NY15-18h）の解剖")
    print("=" * 96)
    LON8 = (LONDON_HOURS[0] + 8, LONDON_HOURS[1] + 8)
    KZ8 = (KZ_HOURS[0] + 8, KZ_HOURS[1] + 8)

    print("\n(a) エントリー時刻別 meanR（ロールオーバー17時に集中していないか）")
    print(f"  {'sym':7s} {'side':5s} " + " ".join(f"{h:02d}h(n/gross/net)".rjust(22) for h in (15, 16, 17)))
    for name, df in data.items():
        recs8 = price_and_scan(df, find_entries(df, LON8, KZ8, F_DEFAULT), STOPBUF_DEFAULT, RR_DEFAULT)
        for side in ("long", "short"):
            tr = trades_from(df, recs8, side)
            cells = []
            for hh in (15, 16, 17):
                sub = tr[tr["hour"] == hh]
                s = stat(sub, COST_RT[name])
                cells.append("(no trades)".rjust(22) if s is None else
                             f"n{s['n']:4d} {s['gross']:+.3f}/{s['net']:+.3f}".rjust(22))
            print(f"  {name:7s} {side:5s} " + " ".join(cells))

    print("\n(b) FX の +8h ベスト cell (gbpusd/short) — 年別 net meanR")
    df = data["gbpusd"]
    recs8 = price_and_scan(df, find_entries(df, LON8, KZ8, F_DEFAULT), STOPBUF_DEFAULT, RR_DEFAULT)
    tr = trades_from(df, recs8, "short")
    tr["year"] = pd.to_datetime(tr["date"]).dt.year
    tr["net"] = tr["R"] - COST_RT["gbpusd"] / tr["risk"]
    by = tr.groupby("year").agg(n=("net", "size"), net=("net", "mean"), gross=("R", "mean"))
    print("   " + "  ".join(f"{int(y)}:{r.net:+.2f}(n{int(r.n)})" for y, r in by.iterrows()))

    print("\n(c) ASK基準の約定にすると残るか（買い指値は low <= lim - spread のときだけ約定）")
    print("    spread は「往復コスト＝1×spread+手数料」の関係から片側 spread を推定して使用")
    for name in ("gbpusd", "eurusd", "usdjpy", "audusd"):
        df = data[name]
        pip = PIP[name]
        for spread_pips, tag in ((0.0, "spread 0 (現行)"), (1.0, "spread 1.0pip"), (3.0, "spread 3.0pip")):
            sp = spread_pips * pip
            recs = price_and_scan(df, ask_fill_entries(df, LON8, KZ8, F_DEFAULT, sp),
                                  STOPBUF_DEFAULT, RR_DEFAULT)
            tr = trades_from(df, recs, "short")
            s = stat(tr, COST_RT[name])
            if s:
                print(f"  {name:7s} +8h short  {tag:16s} n={s['n']:5d} win={s['win']:5.1f} "
                      f"gross={s['gross']:+.3f} net={s['net']:+.3f} PF={s['pf']:.2f}")

    print("\n(d) 保有時間（+8h の GBPUSD short）: 「その窓のトレード」と呼べるか")
    df = data["gbpusd"]
    recs8 = price_and_scan(df, find_entries(df, LON8, KZ8, F_DEFAULT), STOPBUF_DEFAULT, RR_DEFAULT)
    holds = []
    for r in recs8:
        rec = r["short"]
        if rec.get("valid") == "ok" and rec.get("filled"):
            ep = exit_pos_of(df, rec, "short")
            holds.append((ep - rec["entry_pos"]) * 15 / 60.0)
    holds = np.array(holds)
    print(f"   n={len(holds)}  中央値 {np.median(holds):.1f}h  平均 {holds.mean():.1f}h  "
          f"標準偏差 {holds.std():.1f}h  p90 {np.percentile(holds,90):.1f}h  "
          f"当日中(<=6h)で決着した割合 {100*(holds<=6).mean():.0f}%")

    print("\n(e) 対照: 本物のキルゾーン(NY07-10h)の保有時間 (GBPUSD long)")
    recs0 = price_and_scan(df, find_entries(df), STOPBUF_DEFAULT, RR_DEFAULT)
    holds0 = []
    for r in recs0:
        rec = r["long"]
        if rec.get("valid") == "ok" and rec.get("filled"):
            ep = exit_pos_of(df, rec, "long")
            holds0.append((ep - rec["entry_pos"]) * 15 / 60.0)
    holds0 = np.array(holds0)
    print(f"   n={len(holds0)}  中央値 {np.median(holds0):.1f}h  平均 {holds0.mean():.1f}h  "
          f"p90 {np.percentile(holds0,90):.1f}h  当日中(<=6h) {100*(holds0<=6).mean():.0f}%")

    print("\n(f) ロールオーバー窓のバー実在確認（NY時刻17時台のバー本数と平均値幅/ATR）")
    for name in ("gbpusd", "gold"):
        df = data[name]
        d = df.dropna(subset=["atr14"])
        d = d[d["atr14"] > 0]
        for hh in (15, 16, 17, 18):
            sub = d[d["ny_hour"] == hh]
            if len(sub):
                rr = ((sub["high"] - sub["low"]) / sub["atr14"]).mean()
                print(f"  {name:7s} NY{hh:02d}h  bars={len(sub):6d}  mean(range/ATR)={rr:.3f}")


if __name__ == "__main__":
    main()
