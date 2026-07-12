# Structural priors learned（詳細版 — CLAUDE.md から移設）

CLAUDE.md には要約1行リストのみ置き、根拠・経緯・数字はここに保持する。
新しい確定結果は docs/verified_findings.md に追記し、法則が更新されたらここと CLAUDE.md の要約を直す。

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
  (2026-07-12 追記: 26.5年×6ペアの再検証で、FXのトレンドは「政策乖離が持続勾配になった時代」にだけ形成される
  ことを確認。2018-以降のプラスセルは全ペアでドル買い方向に揃う＝単一のドル因子。USDJPY 1hロングのみ3時代連続プラス、
  GBPUSD 4hショートが2/3時代プラス。詳細は verified_findings.md の D2 / funnel / HTF-exit / TF-ladder エントリ。)
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
  (2026-07-12 追記: 出口側も同法則に合流 — USDJPY 1hで「上位足レジサポ利確＋構造フリップ撤退」を entry固定で検証、
  全列が固定RR3に劣後（事前登録KILL）。ブレイクアウトentryの直上はレベル在庫が構造的に密集（最近傍 中央値0.57R）
  するため、レベル利確は勝ち筋を刈る。fixed-RR law はFX・レベルTP・構造トレールまで拡張。)
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
- **Allocation:** best 2-leg = gold-bo + BTC-bo+KAMA at **inverse-vol** (gold 0.79% / BTC 1.21%, total risk 2%):
  CAGR/DD 1.71→1.88 (`portfolio_alloc.py`). Lift = risk-balancing not BTC-beta (helps IS+OOS; KAMA-gated BTC is
  SMOOTHER → cuts DD 10.6→8.6%). Monthly-vol understates BTC tail/gap → cap BTC≤parity as live prudence. **REJECTED:**
  vol-targeting overlay (improves OOS only by degrading IS = regime-luck); dynamic cross-leg allocation (`meta_alloc.py`:
  monthly leg-returns have no positive persistence — trailing-winner UNDERperforms = reversal not momentum → static
  inv-vol already harvests it). Equity-gate + cross-leg momentum both dead ⇒ this set isn't temporally streaky:
  **no "WHEN-to-tilt" dynamic lever has beaten static inv-vol** — remaining levers are different-axis (new entry families).
- **Dead:** gold 5m intraday (all families); USDJPY trend-following; every marketplace/LLM indicator (ML SuperTrend,
  Kinematic-Physics, LeManChanel — renamed classics, ~0 edge). **Trend-flip family** (SuperTrend / **UT Bot** = ATR-trail
  flip = Chandelier / **PMax** = SuperTrend-on-MA) = long-gold/BTC BETA, short side loses, gated < breakout = redundant
  (`ut_bot.py`,`pmax.py`) — don't re-test more. **Fades** (**NWE [LuxAlgo]** = smoother BB, default REPAINTS/lookahead,
  causal = sub-cost fade; BB+RSI) = BB-class, sub-cost (`nwe_vs_bb.py`). **Trendlines-with-Breaks [LuxAlgo]** = FINAL-CLOSED
  (`trendline_break.py`, 2026-06-19): the one genuine signal (diagonal break + STRUCTURAL stop + RR2, meanR ≈+0.33, ~98%ile
  vs random) but failed CONVERSION — CAGR/DD never plateaus (noise-peaks), no exit beats RR2 (lumpiness wall), +0.72 corr
  with gold_bo = redundant (book +0.01). **All 6 marketplace indicators FINAL-closed: a real-but-tiny entry edge fails
  adoption unless it ALSO plateaus + smooths via exit + is low-corr.** A tight RR-filter ("skip wide-stop setups") is
  BACKWARDS (wide-stop = mature breakout = the winners). The lever is never the detector — it's exit(RR)+confirmed-close+
  regime-gate. All gains are backtest + DD-smoothing/beta-timing → live-forward is the final arbiter.
