"""Reporting: the print block of breakout_wave.run(), separated from computation.
summarize() returns the trade table unchanged (it only prints). Lifted verbatim."""
import numpy as np


def summarize(t, rr_real, args):
    if getattr(args, "dump_trades", False):   # clean CSV only (for per-trade slice analysis)
        print("entry_time,R,hold")
        for _, r in t.iterrows():
            print(f"{r['time'].isoformat()},{r['R']:.6f},{r['hold']:.6f}")
        return t
    yrs = sorted(t["y"].unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"] if half else t["R"]
    oosr = t[t["y"] >= half]["R"] if half else t["R"]
    print(f"  n={len(t):>4}  win={(t['R']>0).mean()*100:>3.0f}%  meanR={t['R'].mean():+.2f}  "
          f"totR={t['R'].sum():+6.0f}  | IS={isr.mean():+.2f} OOS={oosr.mean():+.2f}  "
          f"| medRR={np.median(rr_real):.2f}  hold(d) med={t['hold'].median():.1f} max={t['hold'].max():.0f}"
          + (f"  [swap {args.swap_pct}%/d]" if args.swap_pct > 0 else ""))
    # real-money equity curve at constant risk%: the true risk across ALL years (incl chop)
    eq = (1 + args.risk * t["R"]).cumprod()
    dd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    yrs_span = max((t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25, 0.5)
    cagr = (eq.iloc[-1] ** (1 / yrs_span) - 1) * 100
    print(f"  @risk {args.risk*100:.0f}%/trade: return={ (eq.iloc[-1]-1)*100:+.0f}%  "
          f"CAGR={cagr:+.1f}%  maxDD={dd:.1f}%  ret/DD={ (eq.iloc[-1]-1)*100/max(dd,1e-9):.2f}")
    if args.peryear:
        pos = sum(1 for _, g in t.groupby("y") if g["R"].sum() > 0)
        print("       per-year totR: " + " ".join(
            f"{y}:{g['R'].sum():+.0f}(n{len(g)})" for y, g in t.groupby("y"))
            + f"   [{pos}/{t['y'].nunique()} yrs +]")
    return t


def summarize_ema(t, mfe, mae, thr, args):
    """The ema_pullback.run() screen line (MFE/MAE ratio + 1:1 outcome).
    Lifted verbatim; returns the trade table unchanged."""
    mfe, mae = np.array(mfe), np.array(mae)
    ratio = mfe.mean() / mae.mean() if mae.mean() > 0 else float("inf")
    yrs = sorted(t["y"].unique()); half = yrs[len(yrs) // 2] if len(yrs) > 1 else None
    isr = t[t["y"] < half]["R"] if half else t["R"]
    oosr = t[t["y"] >= half]["R"] if half else t["R"]
    tag = "EDGE" if ratio >= 1.2 else "marg" if ratio >= 1.0 else "DEAD"
    print(f"  thr={thr:>4.2f}  n={len(t):>4}  MFE/MAE={ratio:>4.2f}[{tag}]  "
          f"win={(t['R']>0).mean()*100:>3.0f}%  meanR={t['R'].mean():+.2f}  "
          f"totR={t['R'].sum():+6.0f}  | IS={isr.mean():+.2f} OOS={oosr.mean():+.2f}"
          f"  | hold(d) med={t['hold'].median():.1f} max={t['hold'].max():.1f}"
          + (f"  [swap {args.swap_pct}%/d ON]" if args.swap_pct > 0 else ""))
    if args.peryear:
        pos = sum(1 for _, g in t.groupby("y") if g["R"].sum() > 0)
        print("       per-year totR: " + " ".join(
            f"{y}:{g['R'].sum():+.0f}(n{len(g)})" for y, g in t.groupby("y"))
            + f"   [{pos}/{t['y'].nunique()} yrs +]")
    return t
