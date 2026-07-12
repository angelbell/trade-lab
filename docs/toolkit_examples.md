# Toolkit usage examples（コピペ用の検証済み設定 — CLAUDE.md から移設）

```bash
# --- mfe_mae.py: 30-second entry SCREEN before building anything ---
.venv/bin/python mfe_mae.py --csv data/vantage_xauusd_h1.csv --tf 4h --entry breakout --side long
#   read the MFE/MAE ratio: <1.0 dead, >1.2 worth a real test. (--entry swing|meanrev too)

# --- breakout_wave.py: the two validated breakout legs (full history, equity/DD, per-year) ---
.venv/bin/python breakout_wave.py --csv data/vantage_xauusd_h1.csv --tf 1h --pattern B --swing zigzag \
  --zz-k 2 --trend-ema 80 --bo-window 20 --tp-mode rr --rr 3 --fwd 500 \
  --daily-sma 150 --daily-slope-k 10 --risk 0.01 --peryear        # GOLD 1H (gated)
.venv/bin/python breakout_wave.py --csv data/vantage_btcusd_h1.csv --tf 4h --pattern B --swing zigzag \
  --zz-k 2 --trend-ema 80 --tp-mode rr --rr 2 --fwd 300 --risk 0.01 --peryear   # BTC 4H (+KAMA gate)

# --- REGIME-SELECTION tooling (the biggest lever; entry-mining shows ~0 lift to date) ---
.venv/bin/python research/regime_ceiling.py        # how big is deploy-timing? always-on 0.35 -> gate 0.69 -> oracle 1.54
.venv/bin/python research/regime_kama_validate.py  # BTC breakout + daily-KAMA(14)-rising gate: CAGR/DD 0.61->~1.4 (full gauntlet PASS)
.venv/bin/python research/regime_kama_legs.py      # KAMA gate transfers across BREAKOUT legs (gold+BTC); redundant on pullback

# ±1 PLATEAU sweep (no built-in flag -> bash loop). Real edge = neighbors agree.
for k in 1.5 2.0 2.5; do .venv/bin/python breakout_wave.py --csv data/vantage_xauusd_h1.csv --tf 1h \
  --pattern B --swing zigzag --zz-k $k --trend-ema 80 --tp-mode rr --rr 3 --fwd 500 \
  --daily-sma 150 --daily-slope-k 10 2>&1 | grep -E "^  n="; done

# --- ema_pullback.py: the BTC pullback leg (read the thr=0.00 row) ---
.venv/bin/python ema_pullback.py --csv data/vantage_btcusd_h1.csv --tf 4h --side long \
  --ema-fast 20 --ema-slow 80 --slope-k 6 --rr 3 --trend-ma-type sma \
  --entry-trigger close --fill-at-close --no-overlap --fwd 90 --peryear

# --- scalp_lab.py: anti-overfit intraday harness (IS first; TEST stays sealed) ---
.venv/bin/python research/scalp_lab.py bounce --csv data/vantage_xauusd_m5.csv --split is --confirm struct,mom --byyear
.venv/bin/python research/scalp_lab.py orb    --csv data/vantage_xauusd_m5.csv --split is --sweep

# --- portfolio.py: combine the legs into one equity curve + correlations (edit legs in main()) ---
.venv/bin/python research/portfolio.py

# --- gate_passrate.py: which regime gate turns OFF bad years, keeps good ones ---
.venv/bin/python research/gate_passrate.py --csv data/vantage_xauusd_h1.csv

# --- jpy_ema_pullback_orig.py: ALWAYS check the all-signals base, not the capped view ---
.venv/bin/python research/jpy_ema_pullback_orig.py --csv data/vantage_usdjpy_h1.csv --max-per-day 99 --peryear
.venv/bin/python research/jpy_ema_pullback_orig.py --csv data/vantage_usdjpy_m1.csv --resample 5min --start 2026-04-27 --list

# --- COST STRESS: re-run any tool with a harsher round-trip cost before believing it ---
.venv/bin/python ema_pullback.py --csv data/vantage_btcusd_h1.csv --tf 4h --side long --rr 3 \
  --trend-ma-type sma --entry-trigger close --fill-at-close --no-overlap --cost 0.003
```
