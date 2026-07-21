"""Shared measurement harness for the lat2/lat5/lat8 external-data STEP1 screens.

先行の「方向」= log(close[s+H]/close[s])。
先行の「動く量」= 窓内(s, s+H]の絶対単バーリターンの合計 sum(|log(close[i]/close[i-1])|)。
層別ごとに n・平均±標準誤差・中央値・標準偏差を出す。
差/比は月次ブロック・ブートストラップ(1000回)の95%区間を付ける。
"""
import numpy as np
import pandas as pd


def forward_direction(close: pd.Series, H: int) -> pd.Series:
    """log(close[s+H]/close[s]), NaN at the tail where s+H doesn't exist."""
    logc = np.log(close.values.astype(float))
    out = np.full(len(logc), np.nan)
    if H < len(logc):
        out[: len(logc) - H] = logc[H:] - logc[: len(logc) - H]
    return pd.Series(out, index=close.index)


def forward_magnitude(close: pd.Series, H: int) -> pd.Series:
    """sum_{i=1..H} |log(close[s+i]/close[s+i-1])| over the next H bars from s."""
    logc = np.log(close.values.astype(float))
    absret = np.abs(np.diff(logc, prepend=np.nan))  # absret[i] = |log(c[i]/c[i-1])|, absret[0]=nan
    n = len(logc)
    out = np.full(n, np.nan)
    # rolling sum of absret[s+1 .. s+H] = cumsum[s+H] - cumsum[s]
    cs = np.nancumsum(np.nan_to_num(absret, nan=0.0))
    valid = ~np.isnan(absret)
    csvalid = np.cumsum(valid.astype(int))
    for s in range(n - H):
        lo, hi = s, s + H
        if csvalid[hi] - (csvalid[lo] if lo >= 0 else 0) < H:
            # some bars in window missing -> still compute with what's there but flag via NaN if none
            pass
        out[s] = cs[hi] - cs[lo]
    return pd.Series(out, index=close.index)


def layer_table(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """n, mean, se, median, std by group, dropping NaN values."""
    rows = []
    for g, sub in df.groupby(group_col, sort=True):
        v = sub[value_col].dropna().values
        n = len(v)
        if n == 0:
            rows.append({group_col: g, "n": 0, "mean": np.nan, "se": np.nan,
                         "median": np.nan, "std": np.nan})
            continue
        rows.append({
            group_col: g, "n": n,
            "mean": v.mean(), "se": v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
            "median": np.median(v), "std": v.std(ddof=1) if n > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def month_block_bootstrap_diff(df: pd.DataFrame, group_col: str, value_col: str,
                                group_a, group_b, n_boot: int = 1000, seed: int = 0):
    """Monthly-block bootstrap of mean(group_a) - mean(group_b).

    Resamples whole calendar months with replacement (same count as the original span),
    recomputes the group-mean difference each time. Returns (median, p2.5, p97.5, n_boot_used).
    """
    d = df[[group_col, value_col]].dropna(subset=[value_col]).copy()
    d["_m"] = d.index.to_period("M")
    months = d["_m"].unique()
    if len(months) < 3:
        return np.nan, np.nan, np.nan, 0
    by_month = {m: sub for m, sub in d.groupby("_m")}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        chosen = rng.choice(months, size=len(months), replace=True)
        boot = pd.concat([by_month[m] for m in chosen], ignore_index=True)
        ga = boot.loc[boot[group_col] == group_a, value_col]
        gb = boot.loc[boot[group_col] == group_b, value_col]
        if len(ga) == 0 or len(gb) == 0:
            continue
        diffs.append(ga.mean() - gb.mean())
    if not diffs:
        return np.nan, np.nan, np.nan, 0
    diffs = np.array(diffs)
    return float(np.median(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(diffs)


def print_tz_check(sample_jan: pd.Timestamp, sample_jul: pd.Timestamp, label: str = ""):
    """Print the winter/summer UTC offset sanity check (UTC+2 in Jan, UTC+3 in Jul)."""
    jan_riga = sample_jan.tz_localize("UTC").tz_convert("Europe/Riga")
    jul_riga = sample_jul.tz_localize("UTC").tz_convert("Europe/Riga")
    print(f"[tz check{(' ' + label) if label else ''}] "
          f"{sample_jan} UTC -> {jan_riga} (offset {jan_riga.utcoffset()}) "
          f"| {sample_jul} UTC -> {jul_riga} (offset {jul_riga.utcoffset()})")


def utc_to_broker_index(idx_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC-aware timestamps -> broker-time values re-labeled as UTC (to match load_mt5_csv's
    index, which stores broker server time but is tz-tagged UTC). Pattern lifted from
    scratchpad/flow_horizon_test.py (oi.index.tz_convert("Europe/Riga")...)."""
    return idx_utc.tz_convert("Europe/Riga").tz_localize(None).tz_localize("UTC")


def era_split_table(df: pd.DataFrame, group_col: str, value_col: str, split="2022-01-01"):
    """Return two layer_tables, before/after split (broker/UTC-labeled index compare)."""
    before = df[df.index < split]
    after = df[df.index >= split]
    return layer_table(before, group_col, value_col), layer_table(after, group_col, value_col)
