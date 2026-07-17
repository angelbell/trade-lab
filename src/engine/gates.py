"""Regime gates: each returns per-bar arrays aligned to d.index (or None when the
gate is off). Confirmed-bar semantics throughout (shift(1) + ffill) = no lookahead.
Lifted verbatim from breakout_wave.run()."""
from breakout_wave import kama_adaptive


def gate_sma(d, args):
    """Daily(-or gate_tf) SMA regime gate: longs only when the PRIOR completed close
    was above its SMA (optionally also rising). Gap-days dropped so the SMA counts
    real trading days only. Returns (reg, ext_arr); ext_arr is the prior-day
    extension %% above the SMA when --ext-cap is on (skip stretched chases)."""
    reg = None
    ext_arr = None
    if args.daily_sma > 0:
        dc = d["close"].resample(getattr(args, "gate_tf", "1D")).last().dropna()
        sma = dc.rolling(args.daily_sma).mean()
        up = dc > sma
        if args.daily_slope_k > 0:                       # also require the daily SMA rising
            up = up & (sma > sma.shift(args.daily_slope_k))
        up = up.shift(1)
        reg = up.reindex(d.index, method="ffill").fillna(False).values
        if getattr(args, "ext_cap", 0) > 0:              # extension cap: skip entries when the
            ext = (dc - sma) / sma * 100.0               # prior day is >ext_cap% above daily SMA
            ext_arr = ext.shift(1).reindex(d.index, method="ffill").values
    return reg, ext_arr


def gate_kama(d, args):
    """KAMA-rising ENTRY gate on gate_kama_tf (optionally AND a second, slower KAMA
    on gate_kama_tf2 that vetoes bear-rally whipsaws)."""
    kreg = None
    if getattr(args, "gate_kama", 0) > 0:
        dck = d["close"].resample(getattr(args, "gate_kama_tf", "1D")).last().dropna()
        kmg = kama_adaptive(dck, args.gate_kama)
        krise = (kmg > kmg.shift(1)).shift(1)
        kreg = krise.reindex(d.index, method="ffill").fillna(False).values
        gtf2 = getattr(args, "gate_kama_tf2", "")
        if gtf2:
            dck2 = d["close"].resample(gtf2).last().dropna()
            kmg2 = kama_adaptive(dck2, args.gate_kama)
            krise2 = (kmg2 > kmg2.shift(1)).shift(1)
            kreg = kreg & krise2.reindex(d.index, method="ffill").fillna(False).values
    return kreg


def exit_flip(d, args):
    """Regime-flip EXIT (adaptive): per-bar True when the prior completed KAMA on
    exit_kama_tf has turned DOWN — the walker bails a long at that bar's close."""
    against = None
    if getattr(args, "exit_kama", 0) > 0:
        dc2 = d["close"].resample(getattr(args, "exit_kama_tf", "1D")).last().dropna()
        km = kama_adaptive(dc2, args.exit_kama)
        falling = (km < km.shift(1)).shift(1)
        against = falling.reindex(d.index, method="ffill").fillna(False).values
    return against


# ---- ema_pullback (btc_pull) engine gates — lifted verbatim from ema_pullback.run() ----

def ema_htf_gate(d, args):
    """HTF trend gate for the pullback engine: yesterday's completed HTF MA slope
    (shift 1 = no lookahead). Returns (gate_up, gate_dn) or (None, None)."""
    gate_tf = getattr(args, "gate_tf", "")
    gate_type = getattr(args, "gate_type", "ema-slope")
    gate_n = getattr(args, "gate_n", 14)
    gate_up = gate_dn = None
    if gate_tf:
        gc = d["close"].resample(gate_tf).last().dropna()
        if gate_type == "kama-rising":
            from research.regime_adaptive import kama as _kama
            gm = _kama(gc, gate_n)
        elif gate_type == "ema-slope":
            gm = gc.ewm(span=gate_n, adjust=False).mean()
        else:  # sma-slope
            gm = gc.rolling(gate_n).mean()
        rising = (gm > gm.shift(1)).shift(1).reindex(d.index, method="ffill").fillna(0).astype(bool)
        gate_up = rising.values
        gate_dn = (~rising).values
    return gate_up, gate_dn


def ema_exit_ma(d, args):
    """Trend-failure exit line for the pullback engine: close back across this MA
    bails the trade (checked AFTER stop/target — those are intrabar)."""
    exit_ma = None
    if args.exit_sma > 0:
        s = d["close"].rolling(args.exit_sma).mean() if args.exit_ma_type == "sma" \
            else d["close"].ewm(span=args.exit_sma, adjust=False).mean()
        exit_ma = s.values
    return exit_ma
