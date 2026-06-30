# auto-trade — edge-hunting playbook

This repo is a **trading-strategy research lab**. The user is a discretionary trader
(account in JPY, trades Vantage MT5, validates on TradingView). The job is to find a
**cost-survivable, regime-robust edge** — or to **kill ideas honestly**.

**正確性を最優先する。** 実装・検証で仕様が曖昧な点、疑問点、前提が不確かな点があれば、
**推測で進めず最初に質問すること**（誤った前提のまま大量の作業を積むより、1問の確認が安い）。
同じく CLAUDE.md・docs・コメントの記述も正確に保つ（事実と数字は検証で裏取りしてから書く）。

## ⚠️ PRIME DIRECTIVE — falsify, don't validate on plausibility
The user's documented failure mode: trusting plausible-but-untested rules (and pretty
backtests on cherry-picked windows). **Your job is to try to BREAK every idea before
agreeing**, including the user's own and LLM-authored ones (Gemini's gold momentum and
GPT's USDJPY pullback were both confident narratives that died on data). A good-looking
backtest is a hypothesis, not a result. Never cheerlead a number — stress it first.

## 🧪 YOUR ROLE — keep doing R&D; the user decides when to close
結果が出たら、毎回**データから逆算した改善案を出して前進する**こと。「記録して閉じますか？」と訊いて手を止めない —
**adopt / kill / close の判断はユーザーが行う**。NON-ADOPTED と測定できても、そこで止めず「**なぜ落ちたか＝病巣**」を
特定し、それを直接叩く次の実験を提案・実行する（例：meanR は出るが CAGR/DD が低い → 病巣は entry でなく DD/lumpiness →
no-overlap・頻度・分散バスケット・出口を攻める）。ログ追記や CLAUDE.md 更新は必要なら淡々と行い、研究の手は止めない。
falsify は「殺して終わり」ではなく「殺し方から次の一手を生む」こと。

## Environment / how to run
- `python` is NOT on PATH. Always use `.venv/bin/python`.
- Data loader: `from src.data_loader import load_mt5_csv` (root scripts import directly;
  `research/` scripts already `sys.path.insert` the project root). Keeps MT5 broker-server
  time as the clock so HTF bins align.
- Data (Vantage feed = what the user actually trades; validate on THIS, not the chart feed):
  - `data/vantage_xauusd_h1.csv` (gold H1, 2007→2026) · `vantage_xauusd_m5.csv` (gold M5, ~7yr)
  - `data/vantage_btcusd_h1.csv` (BTC H1, 2017→) · `vantage_usdjpy_h1.csv` (USDJPY H1, 2010→, 16yr)
  - `data/vantage_usdjpy_m1.csv` (USDJPY M1, **only ~97 days** 2026-03→06; M5 = resample, short!)
  - Spans grow as the feed is refreshed (see below); the file is the source of truth, not these notes.
- Resample to higher TF inside scripts via `--tf 4h` / `--resample 5min`.
- **Refresh OHLCV from MT5 (demo) via the `mt5-mcp` bridge** (sibling repo `../mt5-mcp`):
  `bash ../mt5-mcp/scripts/run_backtests.sh` pulls the latest Vantage bars for each runbook
  job into `data/`, then re-runs the book. Data-only: `../mt5-mcp/.venv/bin/python
  ../mt5-mcp/client/export_csv.py --symbol XAUUSD --tf h1`. Requires the bridge up
  (`../mt5-mcp/scripts/run_bridge.sh`) + the MT5 terminal logged in. CSV format is identical to
  `export_history.mq5`, so `load_mt5_csv` is unchanged. Shrink-guarded: it won't replace a
  deep-history file with fewer bars (terminal must have the history cached; `--force` overrides).

## The toolkit (reuse these; don't reinvent)
| script | what it tests | key flags |
|---|---|---|
| `breakout_wave.py` | Elliott Pattern-A/B breakout (wave-3 continuation) | `--tf --pattern --swing zigzag --zz-k --trend-ema --rr --tp-mode rr --daily-sma --daily-slope-k --peryear --risk --start/--end` |
| `ema_pullback.py` | EMA pullback-continuation (confirmed-close entry) | `--tf --side --ema-fast/--ema-slow --trend-ma-type --rr --entry-trigger close --fill-at-close --no-overlap --filter --peryear` (prints a thr-sweep; read the `thr=0.00` row) |
| `mfe_mae.py` | generic entry-edge SCREEN (MFE/MAE ratio) | fast top-of-funnel: <1.0 dead, >1.2 worth deeper test |
| `research/scalp_lab.py` | anti-overfit intraday harness (orb/squeeze/bounce) | `--split is/val` + **SEALED `--unseal` test**, `--sweep`, `--byyear`; pre-register pass criteria in `docs/scalp_research_log.md` |
| `research/portfolio.py` | combine legs into one equity curve + annual-R correlations | edit the legs in `main()` |
| `research/gate_passrate.py` | year-by-year ON% of candidate regime gates | which gate turns OFF bad years, keeps good ones |
| `research/overfit_audit.py` | **MEASURE overfit risk** (Deflated Sharpe + PBO/CSCV + bootstrap-CI/null) | the STANDARD GATE for any new method — number, not vibes; `get_legs()`-based |
| `research/equity_gate.py` | equity-curve META-gate test (deploy on own-equity health) | persistence pre-test (gate-keeper) → causal gate → K-plateau + random-gate %ile. NON-ADOPTED (no trade-R persistence) |
| `research/regime_statedet.py` | price-regime STATE detectors (HMM/Hurst/VR) as deploy gates | bar=KAMA-rising; all fail (re-confirm headroom null). HMM=hand-rolled, IS-fit causal |
| `research/instrument_screen.py` | trend-CHARACTER pre-screen of NEW instruments (yfinance daily) | Hurst/VR/ER/trend% → BO-fit vs PULL-fit split; PRE-SCREEN only (Vantage H1 = arbiter); caches `data/ext_*.csv` |
| `research/bounce_size.py` / `bounce_capture.py` | raw vs capturable MFE off a level | |
| `research/jpy_ema_pullback_orig.py` | mechanization of the user's old TV script | `--max-per-day --day-offset-h --list` (demos the luck-sorter, below) |

Most research tools report: `n, win%, PF, meanR, totR, IS/OOS, maxDD` and `--peryear`.
Cost is modeled (round-trip); raise it to stress-test.

## Usage examples (copy-paste; these are the validated configs)
```bash
# --- mfe_mae.py: 30-second entry SCREEN before building anything ---
.venv/bin/python mfe_mae.py --csv data/vantage_xauusd_h1.csv --tf 4h --entry breakout --side long
#   read the MFE/MAE ratio: <1.0 dead, >1.2 worth a real test. (--entry swing|meanrev too)

# --- breakout_wave.py: the two validated breakout legs (full history, equity/DD, per-year) ---
.venv/bin/python breakout_wave.py --csv data/vantage_xauusd_h1.csv --tf 1h --pattern B --swing zigzag \
  --zz-k 2 --trend-ema 80 --bo-window 20 --tp-mode rr --rr 3 --fwd 500 \
  --daily-sma 150 --daily-slope-k 10 --risk 0.01 --peryear        # GOLD 1H (gated)
.venv/bin/python breakout_wave.py --csv data/vantage_btcusd_h1.csv --tf 4h --pattern B --swing zigzag \
  --zz-k 2 --trend-ema 80 --tp-mode rr --rr 2 --fwd 300 --risk 0.01 --peryear   # BTC 4H (+KAMA gate, below)

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

## 🔪 The falsification checklist (run BEFORE believing any edge)
1. **All-signals base first.** Remove every cap/filter/selection. If the raw signal set
   has no edge, nothing downstream can create one (filters CONCENTRATE an existing edge,
   they don't make one). This is the single most common trap.
2. **Win rate vs RR-breakeven.** breakeven = 1/(1+RR) (RR2.5 → 28.6%). Win rate ≈ breakeven
   ⇒ entries are RANDOM, no matter how pretty the PF on a subset.
3. **IS vs OOS.** IS≈OOS = robust. IS≫OOS = back-loaded / curve-fit / regime luck.
4. **±1 sweep every parameter.** A real edge is a **PLATEAU** (neighbors agree); an overfit
   is a **SPIKE** (lone peak). If a value's neighbors collapse, it's curve-fit.
5. **Per-year spread.** Profit concentrated in ONE trending era = **beta, not edge**. Demand
   green spread across regimes. ("It only works recently" usually = a favorable regime, not skill.)
6. **Cost realism.** Stress realistic spread/slippage. 5m scalps live or die on spread.
7. **Selection rules are luck-sorters.** "1 trade/day" / "max N/day" on small samples fabricate
   PF (sweeping the day-boundary alone swung PF 0.7↔1.8 on a random base). Always compare to base.
8. **Feed-dependence.** Validate on the **feed you'll trade (Vantage)**, not the TV chart feed
   (FXCM/Pepperstone). Tight-stop intraday is hypersensitive to feed wicks.
9. **Beta check.** Long-only in a secular bull (BTC/gold) = beta. Demand it survive short side
   and/or another instrument before calling it an edge.
10. **No lookahead.** HTF via shift/confirm-k-bars-later; next-bar-open fill; intrabar SL/TP.
11. **Log every try (incl. failures)** in `docs/scalp_research_log.md` — the more you try, the
    higher the bar (multiple comparisons). Don't "loop until good results" — that's a p-hacker.

## Edge-hunting workflow
1. Mechanize the rule precisely (faithful to spec; flag suspected bugs but test as-written first).
2. Run FULL history, all-signals, `--peryear`.
3. Apply the checklist above (base → win/RR → IS/OOS → sweep → per-year → cost).
4. If it survives: confirm it's not one-era beta (short side / other instrument).
5. **MEASURE overfit, don't assert it** (`research/overfit_audit.py`): Deflated Sharpe (survives the
   trial-count haircut?), PBO/CSCV (is parameter selection just noise? high PBO ⇒ use plateau defaults,
   never IS-optimized peaks), bootstrap-CI + null-p (is the edge distinguishable from luck, and how
   uncertain is its SIZE?). This is the standard gate — a pass = quantified confidence, NOT proof.
   Necessary, not sufficient: it can't measure regime-change (future-unsampled) — live-forward decides
   that. And it's only honest if EVERY trial is logged (step 11); hidden trials under-count N and hide overfit.
6. Only then: sizing (CAGR/DD, full-history DD × 1.5–2 for live, 1% risk default, never >3%)
   and portfolio combination (low cross-correlation cuts DD only if TOTAL risk held constant).
7. Compare on **CAGR/DD**, not ret/DD (ret/DD is compounding-inflated across CAGRs).

## Structural priors learned (use as starting beliefs, still verify)
- **Timeframe is METHOD- and INSTRUMENT-specific — do NOT generalize one TF verdict across methods.**
  4H is the sweet spot for the PULLBACK family and for BTC (gold 1H pullback loses; BTC 1H ~flat; 1D = too thin).
  BUT gold BREAKOUT survives LOWER: gold_bo is 1H (meanR +0.49, the flagship leg), and gold 15M breakout
  works with an extension-cap + RR4 (validated candidate, 2026-06-24: PBO 0.18 / null p .001 / 7-8yr green;
  pending live-forward). So "1H/15M = noise shreds the edge (meanR→0)" holds for PULLBACK, NOT for gold
  breakout — the standing reminder not to over-generalize a TF kill (it was an over-generalization from the
  pullback result; the surviving book's own gold_bo at 1H already contradicted it).
- **Instrument nature dictates method:** BTC & gold = TREND-followers (long-only beta; pullback/
  breakout continuation work on 4H). USDJPY = managed/mean-reverting (trend-following DEAD both
  sides over 16yr; only a FADE — BB+RSI mean-reversion — worked). Match method to the asset.
- **Regime selection (WHEN to deploy) is the BIGGEST lever** — far bigger than the entry. Same fixed
  gold-breakout entries: CAGR/DD 0.35 (always-on) → 0.69 (mech gate) → 1.54 (oracle). Entry-mining has
  shown ~0 lift to date; regime gating is the live research direction. (`research/regime_*.py`)
- **FIXED regime gates are instrument-SPECIFIC; an ADAPTIVE one can transfer.** A daily-SMA(150)+slope
  gate fixed gold's chop years (removed losers) but HURT BTC (removed winners) — never transfer a fixed
  gate. BUT a daily-**KAMA-rising** gate (efficiency-ratio self-adapts to each asset's vol) transfers
  across the BREAKOUT family: validated on BTC breakout (CAGR/DD 0.61→~1.4, full gauntlet) AND helps gold.
  It's redundant on pullback (which already conditions on trend). Rule: a gate helps only when it supplies
  regime context the strategy LACKS. The mech→oracle headroom beyond this is NOT predictable from price
  features (`regime_headroom.py`) = human-judgment/luck territory. **Regime-mechanism space now walked
  once:** price-feature gates (headroom null) · adaptive-MA (KAMA, the sole survivor) · macro DXY/real-
  yields (redundant w/ price) · the EQUITY-CURVE meta-gate ("deploy only when the strategy's OWN recent
  realized equity is healthy" — orthogonal, doesn't predict the market). **Equity-gate = NON-ADOPTED**
  (`research/equity_gate.py`): its falsifier is trade-R serial-correlation, and the legs lack it (gold_bo
  acf1 +0.11 weak; **btc_bo_kama acf1 −0.16 = anti-persistent**; btc_pull iid) → no plateau beats a random
  same-keep% gate on ≥2 legs; the lone gold streak-K8=1.18>1.05 is a SPIKE (neighbors 0.54–0.83) whose lift
  is OOS-only with IS degraded +0.38→+0.18 = the same regime-luck signature that killed vol-targeting. The
  honest kill came from the persistence pre-test, not from fishing a lucky K. **Price-regime STATE
  detectors (HMM 2-state / Hurst / variance-ratio) = NON-ADOPTED** (`research/regime_statedet.py`,
  2026-06-18): tested as deploy gates vs the proper bar (KAMA-rising, not always-on) on gold 1H AND BTC
  4H breakout — ALL fall well BELOW KAMA (gold: HMM 0.37 / Hurst 0.05–0.37 / VR 0.03–0.17 vs KAMA 0.57;
  BTC: 0.27–0.67 vs KAMA 1.40), most below even always-on, no plateau, KAMA-overlap ≈0.5 (not even
  redundant — differently wrong). The HMM was fit IS-only + applied causally (sanity: 98% regime recovery
  on synthetic), so this is a fair kill: richer latent-state / trend-vs-MR classifiers **re-confirm the
  regime_headroom null** — price-based regime detection captures nothing beyond KAMA's efficiency-ratio.
  **KAMA-rising stays a surviving price-regime gate. BUT "WHEN" is NOT fully exhausted — the right
  GRANULARITY matters: a WEEKLY gate found a separation daily/4h/8h gates do NOT have** (2026-06-21).
  On btc_pull, chop years (2021/23/25, win~23%≈RR3 breakeven) are NOT separable by 4h/daily entry-time
  features (slope/ADX/ER/ext all ≈identical good-vs-bad-year = headroom null) — but the WEEKLY scale
  separates starkly: price<30-week-SMA (early-recovery phase) → btc_pull win49%/meanR+0.81; price>30wMA
  (mature bull chop) → 31%/+0.12, and the bad years are 85% above-MA. Gating to "below 30wMA only":
  meanR +0.37→+0.78, CAGR/DD 0.61→1.65, chop-years -9.5R→+4.1R (all green/flat), and it's ANTI-beta
  (cuts the roaring-bull trades yet improves). Plateau across 15–50wk (sweet spot 25–35), IS≈OOS, DSR@200
  =0.95 PASS, null p=0.000 PASS — but **PBO=0.53 FAIL = regime-concentration** (edge clustered in BTC's
  ~3–4 recovery episodes; CSCV sees config-winner flip by time-block). Read: edge is real (not statistical
  luck) but its SIZE is regime-dependent → live-forward arbitrates; use plateau default 30wk, never an
  IS-optimized length. Crucially it does NOT break diversification — it STRENGTHENS it: gold_bo↔btc_pull
  annual corr -0.04→-0.34, 3-leg book 1.58→2.62 (DD-cut driven; abs return modest at 67 trades).
  **NON-leg-filters re-confirmed dead by the CAGR/DD random-drop null** (same session): stop-width≤2.5xATR
  and ER<0.2 score >90%ile on meanR/OOS but only ~61%ile on CAGR/DD = pure n-trimming sorters (vanish via
  the no-overlap replacement effect; stop-width also fails overfit_audit). The lesson: a within-leg filter
  must beat the CAGR/DD null, not just the meanR null. **"WHEN to tilt" levers have shown no lift to date
  across price-feature, adaptive-MA, macro, equity-meta, dynamic-allocation, state-detector — EXCEPT the cycle-phase
  (weekly long-MA-level) gate, which is the one surviving regime lever beyond KAMA. Open axis remains "WHAT
  to trade" (new entry families / instruments).** Tools: ema_pullback.py (--gate-tf/--gate-type/--gate-n),
  research/overfit_audit_pull.py. (Data: load_mt5_csv now auto-drops feed-glitch bars — local-median ±50%
  spike or high/low>3× — with a stderr warning; caught the 2020-08-10 BTC $295 corruption, zero false-pos
  on gold/USDJPY.)
- **The durable pattern across everything:** trade trends + **confirmed-CLOSE entry** (wait for the
  candle to close across the level; intrabar pokes fail) + **let winners run (RR 2–3 ≫ 1:1)** +
  a regime gate. NOT detectors/filters (ADX, efficiency-ratio, S/R-zone, trendline, double-bottom,
  confluence all scored ~0 lift). The exit and the entry-confirmation are the real levers. A genuine
  multi-touch trendline (the user's hand-drawn "white line", mechanized causally w/ ZigZag pivots +
  tol×ATR touch-count + retest in `research/real_trendline.py`, 2026-06-19) = NON-ADOPTED: its break
  is +0.89 corr with gold_bo (MORE redundant than LuxAlgo's +0.72 — a better line just IS the gold
  breakout) and its mfe/mae (1.24) doesn't even beat random-long (1.27 = pure bull beta). gold_bo
  already embodies "break the ZigZag swing-high on a confirmed close, structural stop at the higher-low".
  (RETEST lead now CLOSED, 2026-06-20: on the standalone trendline a retest beat a random-equal-drop null
  at 99.4%ile CAGR/DD — but the prescribed book test, bolting it onto gold_bo, INVERTED it: gold_bo+retest
  meanR +0.49→+0.27, CAGR/DD 1.09→0.49, BELOW the random-drop null at 0.0–0.5%ile, monotonically worse with
  window = no plateau. Retest SELECTS AGAINST the runners — gold_bo's edge is the strong breaks that never
  look back; the ones that pull back to retest are the weaker breaks. Same "wide-stop=mature breakout=the
  winners, RR-filter is backwards" law as TLB. The trendline's 99.4%ile was detector-specific (filters a
  downtrend's false breaks) and does NOT transfer. `research/gold_bo_retest.py` + `breakout_wave.py --retest`.)
  **Structure-break is structure-break:
  every trendline/break detector re-derives gold_bo; the lever stays exit(RR)+confirmed-close+regime-gate.**
  FINAL line-algo probe (`research/trendline_algos.py`, 2026-06-20): held everything constant, varied ONLY
  the line selector across naive-last2 / RANSAC(max-inlier) / **convex-hull** (the most principled,
  near parameter-free) — ALL are +0.84–0.90 annual-corr with gold_bo and NONE beats random-long mfe/mae
  (1.24 = beta); the smarter the line the MORE redundant (RANSAC +0.90). **Auto-trendline search — no line
  algorithm tested so far escapes the structure-break redundancy** (treat as done unless a genuinely new idea breaks the pattern).
- **WHAT-to-trade axis walked once (2026-06-22): the systematic trend-edge universe screened so far ≈ gold+BTC; "edge"
  and "independence" are an inherent TRADE-OFF.** Screened 6 independent instruments (oil USOUSD, DAX
  GER40, copper, platinum XPTUSD, Russell US2000, silver XAGUSD; Vantage h1 via the mt5-mcp bridge) with
  the REAL gold_bo breakout method at 1h AND a Donchian proxy — same verdict both ways: the instruments
  WITH a breakout edge are all metals (PLAT BO meanR+0.25/OOS+0.28/CAGR-DD0.64, silver, copper) but are
  +0.5–0.8 gold-correlated (redundant) AND back-loaded (IS −0.02 to −0.16 = recent-regime luck); the
  genuinely INDEPENDENT ones (oil corr−0.4, Russell, DAX) have NO trend edge (meanR≤+0.04) even with the
  real method. Mechanism: our method extracts TREND-PERSISTENCE → persistent assets cluster (metals+crypto,
  mutually correlated); decorrelated assets are driven by discrete events (oil=geopolitics/supply,
  indices=earnings/rates, FX=central banks) so technical trend doesn't form. Oil etc. = news-driven =
  discretionary, not mechanizable. Index pullback (NAS100.r H1) also marginal/back-loaded (IS+0.33≫OOS+0.06).
  New-instrument breakout search — nothing both independent and edged found to date (proxy→real two-stage). **Cycle-phase gate is PULLBACK-specific, does
  NOT transfer to breakout legs** (2026-06-22): bolting the weekly-30MA-level gate onto gold_bo/btc_bo cuts
  CAGR/DD (1.09→0.81, 1.53→0.68) — it removes the breakout's best trades (new highs above MA), because a gate
  helps only when it supplies context the strategy lacks IN THE DIRECTION it needs (pullback wants "not
  euphoric"; breakout wants "stretched=strong"). gold has NO tradeable pullback (CAGR/DD 0.02–0.07 all TFs =
  gold is a breakout instrument); BTC is the rare asset with both. Tools: bridge `../mt5-mcp/client/export_csv.py`.
- **The trend CANON's known primitives are now all tested — TSMOM (the last unmechanized one) = NON-ADOPTED**
  (`research/tsmom.py`, 2026-06-22). The proven systematic-trend canon = breakout/Donchian + MA +
  time-series-momentum; the book already embodies the first two (gold_bo, btc_bo_kama, btc_pull). TSMOM
  (long if past-L-mo return>0 else short/flat, monthly rebalance, MOP 2012) was the one primitive left, and
  its DIFFERENT mechanism (slow monthly, holds through trends, can short) made the decisive question one
  number: corr(TSMOM, book). Verdict — it collapses into what the book holds: gold TSMOM is **+0.48–0.63
  corr with gold_bo** (just the gold-trend bet, redundant); BTC long/short TSMOM is low-corr (+0.10) only
  because it's **standalone DEAD** (CAGR/DD −0.05→0.00, the short side bleeds on BTC's bull = "differently
  wrong" noise, same as the HMM-state-detector kill); long-only TSMOM on BTC (0.39) LOSES to BTC buy&hold
  (0.59) = dampened beta, no timing skill; L∈{3,6,9,12} is spiky not a plateau. **The probe-3 "lift" (book
  CAGR/DD 2.91→3.34) was the trap and FAILS the null**: a mean-matched RANDOM noise stream (same monthly
  mean+vol, zero corr) gives median 3.19/3.22 and real TSMOM sits at the 44th/53rd %ile of pure noise — the
  whole gain is the mechanical "add any positive-drift uncorrelated stream → CAGR/DD rises" effect (even a
  zero-mean coin-flip lifts it to 2.74). TSMOM adds ZERO orthogonal information. **Surviving trend-canon
  expressions = breakout(gold_bo) / breakout+adaptive-MA(btc_bo_kama) / MA-pullback(btc_pull); the entry/
  canon axis has no untested primitive left to date.** Open levers stay a DIFFERENT axis: WHEN (weekly cycle-phase, the sole survivor)
  or new entry FAMILIES outside the trend canon.
- **The current surviving book** (details + numbers in the auto-memory `project_auto_trade.md`):
  gold 1H breakout (daily SMA gate+slope, RR3), BTC 4H breakout (RR2, **+ daily-KAMA-rising gate** →
  CAGR/DD ~2x, validated 2026-06), BTC 4H EMA pullback (SMA-trend, RR3, **+ weekly-30MA-level cycle gate
  ≤+10%, adopted 2026-06-21** → single-leg CAGR/DD 0.70→1.97; **3-leg book inv-vol 1.61→2.91**, gold↔pull
  corr −0.05→−0.16; regime-concentration caveat PBO0.53 → live-forward arbitrates, BTC-family ≤parity prudent).
  Pine: `pine/btc_4h_ema_pullback.pine` carries the cycle gate (weekly 30MA, skip if >+10% above). **Best 2-leg = gold-bo +
  BTC-bo+KAMA at INVERSE-VOL weights (gold 0.79% / BTC 1.21%, total risk held at 2%): CAGR/DD
  1.71→1.88** (2026-06-18; numbers bumped 2026-06-19 by the ordering fix below — was 1.65→1.83).
  (`research/portfolio_alloc.py`). Lift is genuine risk-balancing not BTC-beta:
  helps IS *and* OOS, and the recent era is gold-led — overweighting BTC works because KAMA-gated BTC
  is SMOOTHER (lower vol), so it cuts DD (10.6→8.6%), not because it earns more. Caveat: monthly-vol
  UNDERSTATES BTC tail/gap risk → capping BTC≤parity is a defensible live-prudence override. **Vol-
  targeting overlay REJECTED** (improved OOS only by degrading IS = regime-luck, not robust). **Dynamic
  cross-leg allocation REJECTED** (`research/meta_alloc.py`, 2026-06-18): monthly leg-returns have no
  positive persistence (autocorr ≈0/neg) and the trailing-winner leg tends to UNDERperform next month
  (cross-sectional spread negative, esp. 3-leg) = REVERSAL not momentum — so the adopted STATIC inv-vol
  (periodic rebalance harvests that reversal) is correct and leaves nothing on the table; chasing the
  hot leg would be backwards. With both the within-leg equity gate AND cross-leg momentum dead, **this
  asset/strategy set is not temporally streaky → no "WHEN to tilt" dynamic lever has beaten static inv-vol to date; the
  remaining levers are different-axis (new entry families, or price-regime HMM/Hurst).** Pine ports in
  `pine/<asset>_<tf>_*.pine` (breakout Pines carry the KAMA gate: ON for BTC, optional for gold).
  **Validated candidates (NOT yet adopted — pending live-forward; real edge but size uncertain):**
  (a) gold 15M breakout — Pattern B + daily-SMA150 gate + extension-cap 8% + RR4 (`overfit_audit_extcap.py`:
  PBO 0.18 / null p .001 / 7–8yr green; higher-freq, higher-DD sibling of gold_bo 1H; `pine/gold_15m_swing_breakout.pine`).
  ENHANCEMENT (2026-06-25, candidate-on-candidate): skip the 9–15 UTC "dead window" (London AM fix + pre-US-data
  whipsaw) — the dropped trades are near-zero-EV deadweight (n122 meanR +0.04, IS −0.14), and removing them lifts
  CAGR/DD 1.26→2.25 (bootstrap median 1.54), meanR +0.39→+0.57, makes ALL 8 years green (rescues the 2023 chop year),
  IS≈OOS. Beats the CAGR/DD random-drop null at 100%ile and PLATEAUS (drop 9–12/9–15/10–14 all 95–100%ile, not a
  spike). Window-search overfit audit (`research/audit_window.py`): DSR@57-windows = 1.00 (survives the
  trial haircut), null p = 0.000 — BUT PBO via CSCV over the window-configs = 0.41 (better than noise 0.54, NOT robust
  0.2) ⇒ the dead window is REAL but its exact boundary is partly noise: use a plateau window (~9–14), never an
  IS-optimized one, and read the size as ~1.5 not 2.25. Same "real-but-size-uncertain" class as (b)/cycle-gate;
  Pine carries it as an optional session-skip input (default OFF). The earlier finding that 15M internal filters all
  LOSE the random-drop null still holds — this clears it precisely because it drops deadweight, not edge;
  (b) H17-S = gold ORB short-only + daily-SMA80-falling gate (`overfit_audit_h17s.py`: PBO 0.62 = regime-
  concentrated downtrend timer, dormant in bull years, low freq → size small; `pine/gold_1h_orb_short_downtrend.pine`).
  Both real-but-size-uncertain → live-forward decides; size conservatively (gold/BTC family ≤ parity).
  Dead: gold 5m intraday (all families), USDJPY trend-following, every LLM-authored/marketplace
  indicator so far (ML SuperTrend, Kinematic-Physics, LeManChanel, **UT Bot Alerts**, **NWE
  [LuxAlgo]**, **PMax** — all renamed classics, ~0 edge; UT Bot = ATR-trailing-stop flip = Chandelier:
  short side loses both TFs/eras = pure long-gold BETA, gated only reaches 0.90 < breakout 1.05 =
  redundant, `research/ut_bot.py`; Nadaraya-Watson Envelope = slightly-smoother BB whose default mode
  REPAINTS (two-sided kernel = lookahead), causal version is a Gaussian-MA±MAE fade that dies sub-cost
  like BB, `research/nwe_vs_bb.py`; PMax = SuperTrend-on-an-MA = smoother UT Bot, same long-gold-beta,
  gated CAGR/DD only 0.23-0.31 (worse — double-smoothing kills frequency → lumpy → high DD; higher
  per-trade meanR but lowest CAGR/DD = PF≠edge), `research/pmax.py`). **The whole trend-flip family
  (SuperTrend/UT Bot/PMax) = long-gold/BTC BETA extraction; no need to re-test more of them.** Fades
  (NWE, BB+RSI) = BB-class, sub-cost. The lever is never the detector — it's exit(RR)+confirmed-close+
  regime-gate. **Trendlines-with-Breaks [LuxAlgo] = FINAL NON-ADOPTED, CLOSED** (`research/trendline_break.py`
  + `trendline_refine.py`, 2026-06-19). It was the ONE marketplace indicator with a genuine signal: the
  causal diagonal-trendline gold bull-break, with a STRUCTURAL stop (recent swing low, NOT a tight ATR
  stop — that botched it first) + RR2, gave meanR ≈+0.33 and beat random same-exit entries at the ~98th
  pctile = a real ENTRY edge, not beta. But the FINAL 3-probe push (re-test of a dead method ⇒ raised bar)
  killed it on CONVERSION: (P1) a full length×mult×method SURFACE shows meanR>0 almost everywhere (broad
  real edge) but CAGR/DD NEVER plateaus — the ≥0.30 cells are isolated noise-peaks scattered across params
  (len14 was representative, not special); (P2) no exit beats RR2's ~0.30 — structural-trail/KAMA-trail/RR3
  all let DD grow faster than CAGR (the mfe/mae "runner" never converts; lumpiness is the wall); (P3)
  corr(TLB, gold_bo) = +0.72 annual = it's REDUNDANT — the diagonal break detects the SAME thing as our
  horizontal gold breakout under trend, so adding it moves the book CAGR/DD by +0.01 (~nil). NB: a tight
  RR-filter ("skip wide-stop setups") is BACKWARDS (wide-stop=mature breakout=the winners) = luck-sorter.
  **All 6 marketplace indicators are now FINAL-closed; a real-but-tiny entry edge fails adoption unless it
  ALSO plateaus, smooths via exit, and is low-correlation — TLB had the edge but none of the three.**
  NB: gains are backtest + DD-smoothing/trend-beta-timing at root — live-forward is the final arbiter.

## Where things live
- Detailed findings map + what's alive/dead: auto-memory `project_auto_trade.md` (loaded each session).
- Pre-registered hypothesis log: `docs/scalp_research_log.md`. Deep dives: `docs/findings_*.md`.
- How to explore a NEW idea (raw hunch → KILL/LEAD, cheap-screen-first + prompt template):
  `docs/idea_exploration_playbook.md` (front end to the falsification checklist).
- Pine strategies (live/charting): `pine/<asset>_<tf>_*.pine` (e.g. `gold_1h_swing_breakout.pine`).
- Engine split: research = Python (Vantage CSVs); see/alert = TradingView (Pine); live execution =
  Vantage MT5 (manual). Validate on Vantage; TV chart feed ≠ trade feed.
- Data pipeline (WSL→Windows MT5 bridge): sibling repo `../mt5-mcp` auto-refreshes the Vantage
  CSVs and runs the book in one command (`config/runbook.yaml` = the jobs; its README = full spec).
  Add a new strategy to the auto-refresh loop by adding a job there. auto-trade is invoked, never modified.
