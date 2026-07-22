"""ICT NY killzone (JST 20:00-22:00 = NY 07:00-09:59) mechanization + measurement.

仕様カード（凍結）どおりに実装。引数なしで全部走る:
    .venv/bin/python experiments/ict_killzone.py

出力は stdout に整形して出す（時計自己検査/ファネル/本表/プラトー/帰無1/帰無2）。
既存関数を流用: src.data_loader.load_mt5_csv, breakout_wave.resample/kama_adaptive, pandas_ta.atr。

時計の扱い: Vantage CSV のバー時刻は naive なブローカー時間 (Europe/Riga)。これを
tz_localize("Europe/Riga") -> tz_convert("America/New_York") で NY 現地時刻の
tz-aware Timestamp に直したあと、以降の全ロジックは「NY 壁時計（tz を外した naive）」
だけを使う。tz 変換は最初の一回だけで、以降は単純な naive 比較にする（DST の複雑さを
一箇所に閉じ込めるため）。
"""
import sys
import datetime as dt

sys.path.insert(0, "/home/angelbell/dev/auto-trade")

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv
from breakout_wave import resample, kama_adaptive

RNG_SEED = 20260714
rng = np.random.default_rng(RNG_SEED)

SYMS = {
    "gold":   "data/vantage_xauusd_m15.csv",
    "eurusd": "data/vantage_eurusd_m15.csv",
    "gbpusd": "data/vantage_gbpusd_m15.csv",
    "usdjpy": "data/vantage_usdjpy_m15.csv",
    "audusd": "data/vantage_audusd_m15.csv",
    "btcusd": "data/vantage_btcusd_m15.csv",
}

COST_RT = {
    "gold":   0.60,
    "eurusd": 0.00009,
    "gbpusd": 0.00009,
    "audusd": 0.00009,
    "usdjpy": 0.009,
    "btcusd": 15.0,
}

F_DEFAULT = 0.705
RR_DEFAULT = 2.0
STOPBUF_DEFAULT = 0.1
ATR_LEN = 14
FWD_CAP = 500

LONDON_HOURS = (2, 6)   # NY time, hours 02..06 (=02:00-06:59)
KZ_HOURS = (7, 9)       # NY time, hours 07..09 (=07:00-09:59)
FORCED_EXIT_HOUR = 16   # NY time


# ---------------------------------------------------------------------------
# 1. load + NY wall-clock conversion
# ---------------------------------------------------------------------------
def load_ny(path, cut2000=False):
    df = load_mt5_csv(path)
    if cut2000:
        df = df.loc["2000-01-01":]
    naive_idx = df.index.tz_localize(None)
    riga_idx = naive_idx.tz_localize("Europe/Riga", ambiguous="NaT", nonexistent="shift_forward")
    nat_mask = riga_idx.isna()
    n_nat = int(nat_mask.sum())
    if n_nat:
        df = df.loc[~nat_mask].copy()
        naive_idx = naive_idx[~nat_mask]
        riga_idx = riga_idx[~nat_mask]
    else:
        df = df.copy()
    ny_aware = riga_idx.tz_convert("America/New_York")
    ny_wall = ny_aware.tz_localize(None)  # naive NY local wall-clock, used everywhere after this
    df["broker_dt"] = naive_idx
    df["ny_wall"] = ny_wall
    df["ny_hour"] = ny_wall.hour
    df["ny_date"] = ny_wall.date
    df = df.reset_index(drop=True)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN).values
    return df, n_nat


# ---------------------------------------------------------------------------
# 2. clock self-check
# ---------------------------------------------------------------------------
def clock_check(df, name):
    d = df.dropna(subset=["atr14"])
    d = d[d["atr14"] > 0]
    rng_atr = (d["high"] - d["low"]) / d["atr14"]
    g = rng_atr.groupby(d["ny_hour"]).mean().reindex(range(24))
    print(f"\n[clock self-check] {name}: mean (high-low)/ATR14 by NY hour")
    order = g.sort_values(ascending=False)
    top5 = list(order.index[:5])
    for h in range(24):
        tag = ""
        if 2 <= h <= 5:
            tag += " <- london(02-05)"
        if 7 <= h <= 9:
            tag += " <- NYKZ(07-09)"
        star = " *TOP5*" if h in top5 else ""
        print(f"   {h:02d}:00  {g[h]:.4f}{star}{tag}")
    print(f"   top5 hours = {top5}")
    return g


# ---------------------------------------------------------------------------
# 3. daily bias timeline (broker-calendar daily bars, no-lookahead via merge_asof)
# ---------------------------------------------------------------------------
def build_bias_timeline(df):
    broker_df = df.set_index("broker_dt")[["open", "high", "low", "close"]]
    daily = resample(broker_df, "1D")
    # confirm time = start of the NEXT broker day (safe: never earlier than true close)
    confirm_broker = daily.index + pd.Timedelta(days=1)
    confirm_riga = confirm_broker.tz_localize("Europe/Riga", ambiguous="NaT", nonexistent="shift_forward")
    confirm_ny_wall = confirm_riga.tz_convert("America/New_York").tz_localize(None)

    ret_green = (daily["close"] > daily["open"]).astype(float)
    sma150 = daily["close"].rolling(150).mean()
    sma_up = (sma150 > sma150.shift(1)).astype(float)
    sma_up[sma150.isna() | sma150.shift(1).isna()] = np.nan
    kama14 = kama_adaptive(daily["close"], 14)
    kama_up = (kama14 > kama14.shift(1)).astype(float)
    kama_up[kama14.isna() | kama14.shift(1).isna()] = np.nan

    keep = ~confirm_ny_wall.isna()
    bias_tl = pd.DataFrame({
        "confirm_ny_wall": confirm_ny_wall[keep],
        "ret_green": ret_green.values[keep],
        "sma_up": sma_up.values[keep],
        "kama_up": kama_up.values[keep],
    }).sort_values("confirm_ny_wall").reset_index(drop=True)
    return bias_tl


def per_day_bias(uniq_dates, bias_tl):
    day_starts = pd.DataFrame({
        "date": uniq_dates,
        "win_start": [pd.Timestamp(dd) + pd.Timedelta(hours=LONDON_HOURS[0]) for dd in uniq_dates],
    }).sort_values("win_start")
    merged = pd.merge_asof(day_starts, bias_tl, left_on="win_start",
                            right_on="confirm_ny_wall", direction="backward")
    merged = merged.set_index("date")
    return merged[["ret_green", "sma_up", "kama_up"]]


# ---------------------------------------------------------------------------
# 4. day-level mechanism (long / short), one instrument, one hour-window config
# ---------------------------------------------------------------------------
def day_slices(dates):
    uniq, start = np.unique(dates, return_index=True)
    end = np.append(start[1:], len(dates))
    return uniq, start, end


def forward_scan(start_pos, side, entry, stop, tgt, opens, highs, lows, closes,
                  ny_wall, forced_cutoff):
    n = len(opens)
    end_pos = min(start_pos + FWD_CAP, n)
    risk = (entry - stop) if side == "long" else (stop - entry)
    hold_R = hold_reason = time_R = time_reason = None
    last_pos = start_pos
    for pos in range(start_pos, end_pos):
        last_pos = pos
        lo = lows[pos]; hi = highs[pos]; cl = closes[pos]
        if side == "long":
            hit_stop = lo <= stop
            hit_tgt = hi >= tgt
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= tgt
        if hold_R is None:
            if hit_stop:
                hold_R, hold_reason = -1.0, "stop"
            elif hit_tgt:
                hold_R = (rr_from_tgt(entry, tgt, risk, side))
                hold_reason = "target"
        if time_R is None:
            if hit_stop:
                time_R, time_reason = -1.0, "stop"
            elif hit_tgt:
                time_R = rr_from_tgt(entry, tgt, risk, side)
                time_reason = "target"
            elif ny_wall[pos] >= forced_cutoff:
                time_R = ((cl - entry) if side == "long" else (entry - cl)) / risk
                time_reason = "time_exit"
        if hold_R is not None and time_R is not None:
            break
    if hold_R is None:
        cl = closes[last_pos]
        hold_R = ((cl - entry) if side == "long" else (entry - cl)) / risk
        hold_reason = "cap_expired"
    if time_R is None:
        cl = closes[last_pos]
        time_R = ((cl - entry) if side == "long" else (entry - cl)) / risk
        time_reason = "cap_expired"
    return risk, hold_R, hold_reason, time_R, time_reason


def rr_from_tgt(entry, tgt, risk, side):
    return (tgt - entry) / risk if side == "long" else (entry - tgt) / risk


def find_entries(df, london_hours=LONDON_HOURS, kz_hours=KZ_HOURS, f=F_DEFAULT):
    """Stage 1: mechanism up to the limit fill (independent of stop/target width,
    so it can be cached and re-priced for the f-sweep without redoing the
    London-window/reversal/invalidity search)."""
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values
    atr = df["atr14"].values; hours = df["ny_hour"].values
    dates = df["ny_date"].values

    uniq_dates, start, end = day_slices(dates)
    records = []
    for i, day in enumerate(uniq_dates):
        s, e = start[i], end[i]
        h = hours[s:e]
        lon_mask = (h >= london_hours[0]) & (h <= london_hours[1])
        kz_mask = (h >= kz_hours[0]) & (h <= kz_hours[1])
        idx_range = np.arange(s, e)
        lon_pos = idx_range[lon_mask]
        kz_pos = idx_range[kz_mask]
        rec = {"date": day, "long": {"valid": "no_window_bars", "filled": False},
               "short": {"valid": "no_window_bars", "filled": False}}

        if len(lon_pos) >= 2 and len(kz_pos) >= 1:
            atrval = atr[lon_pos[-1]]
            if np.isnan(atrval) or atrval <= 0:
                rec["long"] = {"valid": "atr_nan", "filled": False}
                rec["short"] = {"valid": "atr_nan", "filled": False}
            else:
                # ---- LONG mechanism ----
                lo_w = lows[lon_pos]
                i_L_rel = int(np.argmin(lo_w))
                if i_L_rel == len(lon_pos) - 1:
                    rec["long"] = {"valid": "extreme_last_bar", "filled": False}
                else:
                    L = lo_w[i_L_rel]
                    after = lon_pos[i_L_rel + 1:]
                    hi_after = highs[after]
                    i_H_rel = int(np.argmax(hi_after))
                    H = hi_after[i_H_rel]
                    i_H_pos = after[i_H_rel]
                    if H - L < 0.5 * atrval:
                        rec["long"] = {"valid": "no_bounce", "filled": False}
                    else:
                        after_H = after[after > i_H_pos]
                        broken = len(after_H) > 0 and bool((lows[after_H] <= L).any())
                        if broken:
                            rec["long"] = {"valid": "broken_before_kz", "filled": False}
                        else:
                            lim = H - f * (H - L)
                            fill_pos = None
                            for p in kz_pos:
                                if lows[p] <= lim:
                                    fill_pos = p
                                    break
                            if fill_pos is None:
                                rec["long"] = {"valid": "ok", "filled": False, "reason": "no_fill"}
                            else:
                                entry = min(lim, opens[fill_pos])
                                rec["long"] = dict(valid="ok", filled=True, entry_pos=fill_pos,
                                                    entry=entry, L=L, H=H, atrval=atrval,
                                                    kz_bars=list(kz_pos))

                # ---- SHORT mechanism (mirror) ----
                hi_w = highs[lon_pos]
                i_H_rel = int(np.argmax(hi_w))
                if i_H_rel == len(lon_pos) - 1:
                    rec["short"] = {"valid": "extreme_last_bar", "filled": False}
                else:
                    H = hi_w[i_H_rel]
                    after = lon_pos[i_H_rel + 1:]
                    lo_after = lows[after]
                    i_L_rel = int(np.argmin(lo_after))
                    L = lo_after[i_L_rel]
                    i_L_pos = after[i_L_rel]
                    if H - L < 0.5 * atrval:
                        rec["short"] = {"valid": "no_bounce", "filled": False}
                    else:
                        after_L = after[after > i_L_pos]
                        broken = len(after_L) > 0 and bool((highs[after_L] >= H).any())
                        if broken:
                            rec["short"] = {"valid": "broken_before_kz", "filled": False}
                        else:
                            lim = L + f * (H - L)
                            fill_pos = None
                            for p in kz_pos:
                                if highs[p] >= lim:
                                    fill_pos = p
                                    break
                            if fill_pos is None:
                                rec["short"] = {"valid": "ok", "filled": False, "reason": "no_fill"}
                            else:
                                entry = max(lim, opens[fill_pos])
                                rec["short"] = dict(valid="ok", filled=True, entry_pos=fill_pos,
                                                     entry=entry, L=L, H=H, atrval=atrval,
                                                     kz_bars=list(kz_pos))
        records.append(rec)
    return records


def price_and_scan(df, records, stopbuf=STOPBUF_DEFAULT, rr=RR_DEFAULT):
    """Stage 2: given cached entries (find_entries), price stop/target for a
    given (stopbuf, rr) and forward-scan for both exit versions. Cheap --
    reused across the RR/stopbuf sweep without redoing find_entries."""
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    ny_wall = df["ny_wall"].values
    out = []
    for rec in records:
        new_rec = {"date": rec["date"]}
        for side in ("long", "short"):
            r = rec[side]
            if not r.get("filled"):
                new_rec[side] = r
                continue
            entry = r["entry"]; L = r["L"]; H = r["H"]; atrval = r["atrval"]
            fill_pos = r["entry_pos"]
            if side == "long":
                stop = L - stopbuf * atrval
                tgt = entry + rr * (entry - stop)
            else:
                stop = H + stopbuf * atrval
                tgt = entry - rr * (stop - entry)
            forced_cutoff = np.datetime64(pd.Timestamp(rec["date"]) + pd.Timedelta(hours=FORCED_EXIT_HOUR))
            risk, hold_R, hold_reason, time_R, time_reason = forward_scan(
                fill_pos, side, entry, stop, tgt, opens, highs, lows, closes, ny_wall, forced_cutoff)
            new_rec[side] = dict(r, stop=stop, tgt=tgt, risk=risk, hold_R=hold_R,
                                  hold_reason=hold_reason, time_R=time_R, time_reason=time_reason)
        out.append(new_rec)
    return out


def run_mechanism(df, london_hours=LONDON_HOURS, kz_hours=KZ_HOURS, f=F_DEFAULT,
                   rr=RR_DEFAULT, stopbuf=STOPBUF_DEFAULT):
    entries = find_entries(df, london_hours, kz_hours, f)
    return price_and_scan(df, entries, stopbuf, rr)


# ---------------------------------------------------------------------------
# 5. funnel / trade assembly / stats
# ---------------------------------------------------------------------------
from collections import Counter


def funnel_table(records, side):
    total = len(records)
    valids = [r[side]["valid"] for r in records]
    c = Counter(valids)
    not_last = total - c.get("no_window_bars", 0) - c.get("atr_nan", 0) - c.get("extreme_last_bar", 0)
    bounce_ok = c.get("ok", 0)
    filled_recs = [r[side] for r in records if r[side].get("filled")]
    filled = len(filled_recs)
    reasons = Counter(r["hold_reason"] for r in filled_recs)
    return dict(total_ny_days=total, valid_london=not_last, bounce_confirmed=bounce_ok,
                filled=filled, win=reasons.get("target", 0), loss=reasons.get("stop", 0),
                expired=reasons.get("cap_expired", 0), breakdown=dict(c))


def assemble_trades(records, bias_df, variant):
    trades = []
    for r in records:
        d = r["date"]
        if variant == "B0L":
            side = "long"
        elif variant == "B0S":
            side = "short"
        else:
            col = {"B1": "ret_green", "B2": "sma_up", "B3": "kama_up"}[variant]
            val = bias_df.loc[d, col] if d in bias_df.index else np.nan
            if pd.isna(val):
                continue
            side = "long" if val == 1 else "short"
        rec = r[side]
        if rec.get("valid") == "ok" and rec.get("filled"):
            trades.append(dict(date=d, side=side, hold_R=rec["hold_R"], hold_reason=rec["hold_reason"],
                                time_R=rec["time_R"], time_reason=rec["time_reason"], risk=rec["risk"]))
    return trades


def compute_stats(trades, cost_rt, years_span, exit_key="hold_R"):
    if not trades:
        return None
    t = pd.DataFrame(trades)
    order = np.argsort(t["date"].values)
    t = t.iloc[order].reset_index(drop=True)
    gross = t[exit_key].values.astype(float)
    risk = t["risk"].values.astype(float)
    net = gross - cost_rt / risk
    n = len(t)

    def pf(x):
        pos = x[x > 0].sum(); neg = -x[x < 0].sum()
        return pos / neg if neg > 0 else np.inf

    cum = np.cumsum(net)
    running_max = np.maximum.accumulate(cum)
    maxDD = float((running_max - cum).max()) if n else 0.0
    half = n // 2
    is_totR = float(net[:half].sum())
    oos_totR = float(net[half:].sum())
    yrs = pd.to_datetime(t["date"]).dt.year
    yearly_net = pd.Series(net).groupby(yrs.values).sum()
    win_year_pct = 100.0 * float((yearly_net > 0).mean()) if len(yearly_net) else np.nan

    return dict(n=n, n_per_year=n / years_span, win_pct=100.0 * float((gross > 0).mean()),
                meanR_gross=float(gross.mean()), meanR_net=float(net.mean()),
                median_net=float(np.median(net)), std_net=float(np.std(net)),
                PF_gross=float(pf(gross)), PF_net=float(pf(net)),
                totR_net=float(net.sum()), maxDD=maxDD, is_totR=is_totR, oos_totR=oos_totR,
                win_year_pct=win_year_pct, n_years=len(yearly_net))


RR_BREAKEVEN = {1.5: 100 / 2.5, 2.0: 100 / 3.0, 3.0: 25.0, 4.0: 20.0, 4.5: 100 / 5.5}


def fmt_stats(s, rr=RR_DEFAULT):
    if s is None:
        return "  (no trades)"
    be = RR_BREAKEVEN.get(rr, 100.0 / (1 + rr))
    return (f"n={s['n']:5d} n/yr={s['n_per_year']:6.1f} win%={s['win_pct']:5.1f} "
            f"(be={be:4.1f}) meanR_g={s['meanR_gross']:+.3f} meanR_n={s['meanR_net']:+.3f} "
            f"med_n={s['median_net']:+.3f} std_n={s['std_net']:.3f} "
            f"PF_g={s['PF_gross']:.2f} PF_n={s['PF_net']:.2f} totR_n={s['totR_net']:+7.1f} "
            f"maxDD={s['maxDD']:6.2f} IS={s['is_totR']:+7.1f} OOS={s['oos_totR']:+7.1f} "
            f"win_yr%={s['win_year_pct']:5.1f}({s['n_years']}yr)")


# ---------------------------------------------------------------------------
# 6. plateau sweep (gold / eurusd only, best bias, canonical hold-exit)
# ---------------------------------------------------------------------------
F_LIST = [0.5, 0.62, 0.705, 0.79, 0.886]
RR_LIST = [1.5, 2.0, 3.0, 4.0]
STOPBUF_LIST = [0.0, 0.1, 0.25]


def plateau_sweep(df, bias_df, variant, cost_rt, years_span):
    rows = []
    for f in F_LIST:
        entries = find_entries(df, f=f)
        for stopbuf in STOPBUF_LIST:
            for rr in RR_LIST:
                recs = price_and_scan(df, entries, stopbuf=stopbuf, rr=rr)
                trades = assemble_trades(recs, bias_df, variant)
                s = compute_stats(trades, cost_rt, years_span, "hold_R")
                rows.append((f, rr, stopbuf, s))
    return rows


# ---------------------------------------------------------------------------
# 7. null 1 -- placebo shifted windows (London+NYKZ shifted together)
# ---------------------------------------------------------------------------
SHIFTS = [0, 4, 8, 12]


def null1_shifted(df, bias_df, cost_rt, years_span):
    out = {}
    for shift in SHIFTS:
        lon = (LONDON_HOURS[0] + shift, LONDON_HOURS[1] + shift)
        kz = (KZ_HOURS[0] + shift, KZ_HOURS[1] + shift)
        recs = run_mechanism(df, london_hours=lon, kz_hours=kz)
        row = {}
        for variant in ("B0L", "B0S"):
            trades = assemble_trades(recs, bias_df, variant)
            row[variant] = compute_stats(trades, cost_rt, years_span, "hold_R")
        out[shift] = row
    return out


# ---------------------------------------------------------------------------
# 8. null 2 -- same-day random-time entry within the NY killzone (500 reps)
# ---------------------------------------------------------------------------
N_REPS = 500


def null2_random_entry(df, records, side, cost_rt, rr=RR_DEFAULT):
    """For every day where `side` actually filled, precompute the canonical
    hold-exit R for EVERY bar in that day's killzone window (fixed risk/rr
    taken from the real trade), then Monte-Carlo resample one bar/day x 500
    reps to get a null distribution of book meanR."""
    opens = df["open"].values; highs = df["high"].values
    lows = df["low"].values; closes = df["close"].values
    ny_wall = df["ny_wall"].values

    per_day_R = []      # list of arrays, one per filled day
    real_meanR_gross = []
    real_meanR_net = []
    for r in records:
        rec = r[side]
        if not (rec.get("valid") == "ok" and rec.get("filled")):
            continue
        risk = rec["risk"]
        kz_bars = rec["kz_bars"]
        forced_cutoff = np.datetime64(pd.Timestamp(r["date"]) + pd.Timedelta(hours=FORCED_EXIT_HOUR))
        outcomes = np.empty(len(kz_bars))
        for j, p in enumerate(kz_bars):
            entry_p = opens[p]
            if side == "long":
                stop_p = entry_p - risk
                tgt_p = entry_p + rr * risk
            else:
                stop_p = entry_p + risk
                tgt_p = entry_p - rr * risk
            _, hold_R, _, _, _ = forward_scan(p, side, entry_p, stop_p, tgt_p,
                                               opens, highs, lows, closes, ny_wall, forced_cutoff)
            outcomes[j] = hold_R
        per_day_R.append(outcomes)
        real_meanR_gross.append(rec["hold_R"])
        real_meanR_net.append(rec["hold_R"] - cost_rt / risk)

    if not per_day_R:
        return None
    real_meanR_gross = float(np.mean(real_meanR_gross))
    real_meanR_net = float(np.mean(real_meanR_net))

    rep_means = np.empty(N_REPS)
    for rep in range(N_REPS):
        picks = np.array([arr[rng.integers(0, len(arr))] for arr in per_day_R])
        rep_means[rep] = picks.mean()
    pctile = 100.0 * float((rep_means < real_meanR_gross).mean())
    return dict(n_days=len(per_day_R), real_meanR_gross=real_meanR_gross,
                real_meanR_net=real_meanR_net, null_mean=float(rep_means.mean()),
                null_std=float(rep_means.std()), null_median=float(np.median(rep_means)),
                pctile=pctile)


# ---------------------------------------------------------------------------
# 9. main
# ---------------------------------------------------------------------------
VARIANTS = ["B0L", "B0S", "B1", "B2", "B3"]
USDJPY_CUT2000 = {"usdjpy": True}


def main():
    print("=" * 100)
    print("ICT NY killzone (JST 20:00-22:00 = NY 07:00-09:59) -- mechanization + measurement")
    print("=" * 100)

    all_data = {}
    for name, path in SYMS.items():
        cut = USDJPY_CUT2000.get(name, False)
        df, n_nat = load_ny(path, cut2000=cut)
        print(f"\n>>> {name}: loaded {len(df)} m15 bars "
              f"[{df['broker_dt'].iloc[0]} .. {df['broker_dt'].iloc[-1]}] (broker time), "
              f"NaT dropped in tz_localize = {n_nat}")
        clock_check(df, name)
        all_data[name] = df

    bias_data = {}
    for name, df in all_data.items():
        bias_tl = build_bias_timeline(df)
        uniq_dates = np.unique(df["ny_date"].values)
        bias_df = per_day_bias(uniq_dates, bias_tl)
        bias_data[name] = bias_df

    default_records = {}
    years_span = {}
    for name, df in all_data.items():
        recs = run_mechanism(df)
        default_records[name] = recs
        d0, d1 = recs[0]["date"], recs[-1]["date"]
        years_span[name] = max((d1 - d0).days / 365.25, 0.5)

    # ---- 1. funnel ----
    print("\n" + "=" * 100)
    print("TABLE 1: FUNNEL (mechanism-level, bias-independent, canonical hold-exit)")
    print("=" * 100)
    for name, recs in default_records.items():
        for side in ("long", "short"):
            fn = funnel_table(recs, side)
            print(f"\n[{name} / {side}]")
            print(f"  total NY days           = {fn['total_ny_days']}")
            print(f"  london window valid     = {fn['valid_london']}  "
                  f"(drop: no_window/atr_nan/extreme_last_bar)")
            print(f"  bounce confirmed (>=0.5ATR, not broken pre-KZ) = {fn['bounce_confirmed']}")
            print(f"  limit filled in KZ      = {fn['filled']}")
            print(f"    -> win(target)  = {fn['win']}")
            print(f"    -> loss(stop)   = {fn['loss']}")
            print(f"    -> expired(cap) = {fn['expired']}")
            print(f"  breakdown of 'valid' reasons: {fn['breakdown']}")

    # ---- 2. main table ----
    print("\n" + "=" * 100)
    print("TABLE 2: MAIN (instrument x bias x exit-version); RR2 breakeven win% = 33.3")
    print("=" * 100)
    main_stats = {}
    for name, df in all_data.items():
        recs = default_records[name]
        bias_df = bias_data[name]
        cost_rt = COST_RT[name]
        ys = years_span[name]
        print(f"\n--- {name} (cost_rt={cost_rt}) ---")
        for variant in VARIANTS:
            trades = assemble_trades(recs, bias_df, variant)
            for exit_key, exit_name in (("hold_R", "HOLD"), ("time_R", "NY16:00")):
                s = compute_stats(trades, cost_rt, ys, exit_key)
                main_stats[(name, variant, exit_name)] = s
                print(f"  {variant:4s} {exit_name:8s} {fmt_stats(s)}")

    # ---- 3. plateau sweep (gold + eurusd, best bias by net PF, n>=30) ----
    print("\n" + "=" * 100)
    print("TABLE 3: PLATEAU SWEEP (gold, eurusd; best bias by net PF among n>=30; canonical hold-exit)")
    print("(NOT re-selecting a new best here -- plateau-vs-spike check only)")
    print("=" * 100)
    best_variant = {}
    for name in ("gold", "eurusd"):
        candidates = []
        for variant in VARIANTS:
            s = main_stats[(name, variant, "HOLD")]
            if s is not None and s["n"] >= 30:
                candidates.append((s["PF_net"], variant, s))
        if not candidates:
            print(f"\n{name}: no variant with n>=30 at default params -- skipping sweep")
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_pf, best_var, best_s = candidates[0]
        best_variant[name] = best_var
        print(f"\n{name}: best bias = {best_var} (default-params PF_net={best_pf:.2f}, n={best_s['n']})")
        rows = plateau_sweep(all_data[name], bias_data[name], best_var, COST_RT[name], years_span[name])
        print(f"  {'f':>6} {'RR':>5} {'stopbuf':>8} {'n':>5} {'n/yr':>6} {'win%':>6} "
              f"{'PF_g':>6} {'PF_n':>6} {'meanR_g':>8} {'meanR_n':>8} {'totR_n':>8}")
        for f, rr, stopbuf, s in rows:
            if s is None:
                print(f"  {f:6.3f} {rr:5.1f} {stopbuf:8.2f}   (no trades)")
                continue
            print(f"  {f:6.3f} {rr:5.1f} {stopbuf:8.2f} {s['n']:5d} {s['n_per_year']:6.1f} "
                  f"{s['win_pct']:6.1f} {s['PF_gross']:6.2f} {s['PF_net']:6.2f} "
                  f"{s['meanR_gross']:+8.3f} {s['meanR_net']:+8.3f} {s['totR_net']:+8.1f}")

    # ---- 4. null 1: placebo shifted windows ----
    print("\n" + "=" * 100)
    print("TABLE 4: NULL-1 PLACEBO (London+NYKZ windows shifted +4h/+8h/+12h together; B0L/B0S, hold-exit)")
    print("=" * 100)
    for name, df in all_data.items():
        n1 = null1_shifted(df, bias_data[name], COST_RT[name], years_span[name])
        print(f"\n--- {name} ---")
        for shift in SHIFTS:
            for variant in ("B0L", "B0S"):
                s = n1[shift][variant]
                tag = "REAL " if shift == 0 else f"+{shift:02d}h"
                print(f"  {tag} {variant:4s} {fmt_stats(s)}")

    # ---- 5. null 2: same-day random-time entry ----
    print("\n" + "=" * 100)
    print("TABLE 5: NULL-2 RANDOM-ENTRY-TIME (same day, same fixed risk/RR, random bar-open in NYKZ; 500 reps)")
    print("=" * 100)
    for name, df in all_data.items():
        recs = default_records[name]
        for side in ("long", "short"):
            n2 = null2_random_entry(df, recs, side, COST_RT[name])
            if n2 is None:
                print(f"\n{name}/{side}: no filled days -- skip")
                continue
            print(f"\n{name}/{side}: n_days={n2['n_days']}  real meanR_gross={n2['real_meanR_gross']:+.3f} "
                  f"(net={n2['real_meanR_net']:+.3f})")
            print(f"  null (500 reps): mean={n2['null_mean']:+.3f} median={n2['null_median']:+.3f} "
                  f"std={n2['null_std']:.3f}  real-vs-null percentile={n2['pctile']:.1f}%ile")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
