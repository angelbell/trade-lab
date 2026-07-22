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

---

## 【2026-07-21 移設】CLAUDE.md 反証チェックリスト・構造法則の証拠（逐語）

CLAUDE.md 側は規則の一文だけを持ち、根拠・数字・経緯（＝どう気づいたか）はここに置く。
以下は移設時点の CLAUDE.md の当該2セクションの逐語コピー。規則を疑う時・数字が要る時はここを読む。

## 🔪 The falsification checklist (run BEFORE believing any edge)
🔒 **順序はフックで強制されている**（`.claude/hooks/screen_gate.py`, 2026-07-21 設置）。トレード統計を出す
スクリプトは、**巡行幅(MFE/MAE)を先に測っていないと実行できない**。通し方: `research/screen.py` の
`run_screen(name, df, entries)` を走らせ、スクリプト先頭に `SCREEN = "<name>"` を書く。検算は
**既知の値への数値 assert** を入れる（印字して目視は検算ではない）。番人4本と `research/` 配下は無条件で通る。
🚨 **このリストは棄却基準であって停止条件ではない。項目に当てはまったら、却下する前に必ず分岐せよ。**
**❌を出す時は「この却下が誤りだったら成立していたはずの対立仮説」を1つ書き、安く測れるなら測ってから却下する。
台帳に却下を書くのは、対立仮説を潰した後。**（書いてから疑うと、書いたものを守る側に回る）
2026-07-21の実例: 「初動の向きに賭けると買いだけ勝つ＝ベータ」→ そのまま却下し台帳に記入 → ユーザーの
「ベータがあるなら乗るのが当たり前では」で再検定 → **捨てようとした側（向きを見ずにただ買う）が30倍良かった**。
1. **All-signals base first** — filters CONCENTRATE an edge, they don't create one.
2. **Win rate vs RR-breakeven** (1/(1+RR); RR3→25%). Win≈breakeven ⇒ entries are RANDOM.
3. **IS vs OOS.** IS≫OOS = back-loaded / curve-fit / regime luck.
4. **±1 sweep every parameter.** Real edge = PLATEAU; overfit = lone SPIKE.
5. **Per-year/era spread.** Profit in one era = beta, not edge.
6. **Cost realism** — but judge in order: 素の率×幅→偶然性→コスト→口座寄与。「エッジ無し」と「エッジ有・コスト死」は別ラベル。
7. **Selection rules (caps/1日N回) are luck-sorters** — always compare to base. Within-leg filters must beat the
   **CAGR/DD** random-drop null (not just meanR). **だが random-drop null は必要条件どまり** — それは「同じ価格経路
   の上でランダムに削るよりマシか」しか訊いていない。**巡回ブロック・ブートストラップ（1/3/6/12か月）も必ず通す**
   （「別の月の並びでも成り立つか」）。真の改善はブロックを長くするほど勝率が上がり、経路当てはめは上がらない
   （2026-07-13: 週足ERゲートは random-drop 100%ile → ブロック34〜52%＝コイン投げで死亡）
   **これは leg だけでなく BOOK の CAGR/DD にも適用する** — ブックの月次リターンも単一経路であり、
   12.03 vs 13.26 のような差はブートストラップで初めて意味が付く（2026-07-13 に自分の判定を検算して発覚）。
   🚨 **ただしブートストラップの前に、その CAGR/DD の分母が本物かを見ろ**（下の 8 番）。
7.5. **🚨 inv-vol 重みのブックでは、「R のばらつきを下げる操作」はすべて自動でレバレッジを買う。**
   σ(R) が下がる → 重みが上がる → そのレッグが最良なら CAGR/DD が上がる。**フィルタが効いたのではなく、
   賭け金が増えただけ。** 2026-07-13 にこれで3件を撤回した（損切り/価格フィルタ・PDHソフト倍率・押し目深さ）。
   **必ず「重みを現行に固定した版」と並べて報告する。** 固定して差が消えたら、それはレバレッジ・ダイヤル
   （同じ利得は重みを ×1.1 するだけで得られる ＝ 新しい知見ではない）。
   🔬 **同じ根の兄弟: 「その変数は本当に自分が思っているものか」を最後に確かめる。**
   サイズ倍率で割った量（例: 損切り幅÷価格÷サイズ倍率）は、**サイズ・ルールを暗黙に含む**。
   統計の検定（先読み/null/ブートストラップ/ウォークフォワード）を全部通しても、定義が違えば無意味。
8. **🚨 ブックの maxDD は必ずトレード（or 日次）解像度で測る — 月次資産曲線で測ってはならない。**
   月次に潰すと月内で完結するDDが全部消える。2026-07-13: 6レッグ・ブックの maxDD が **3.62%＝2019-07の単月**
   （CAGR 43.6% ＝ Calmar 12 の非現実値）に化け、CAGR/DD が「最悪の1か月をどれだけ薄められたか」の指標になり、
   その上で下した判定の**順位が全部入れ替わった**（トレード解像度では DD 6.53%・CAGR/DD 6.84）。
   3レッグ時代は月次7.81% vs 日次7.80%で一致していた（月に数本しか建てないため）＝**高頻度レッグ（15分足）を
   足した瞬間に壊れる**。**【修正済み】** `research/portfolio_alloc.py` は `cagr_dd_trades()` で報告するようになった
   （`cagr_dd_monthly()` は警告付きで残置＝低頻度レッグ同士の比較専用）。
   **同じ根（月次に潰すこと）の兄弟バグ: inv-vol を「月次σ」で計算すると低頻度レッグが過大な玉を貰う**
   （「建てない月＝ゼロ」を"低ボラ＝安全"と誤読する）。btc_bo_kama(70本/7年)=1トレード口座1.006% vs
   btc15m_L(758本)=0.231% ＝4.4倍の格差。6レッグ・ブックの重み総当たり: 月次σ逆数(現行)6.84 /
   **トレードRのσ逆数 8.19** / 頻度調整 8.35 / 逆向きダミー4.69（＝機構の確認）。
   **頻度の違うレッグを混ぜる時は、重みをトレードRのσで出す**（詳細 `docs/findings/x_sizing.md`）。
   🚨 **さらに: maxDD の「実測値」を1本の経路から読むな。必ずブートストラップして中央値を使え。**
   2026-07-13: btc15m_A の実測 maxDD 8.5% は、巡回ブロック・ブートストラップ（3000回）の**下から18%の位置**
   ＝運の良い経路だった。**想定値は中央値の 10.2%**（95%点 15.6%）。CAGR/DD は 4.25 ではなく **3.54**。
   ユーザーが「RRごとのDDを教えて」と訊いたことで発覚。**賭け率は必ず中央値のDD（×1.5〜2）から決める。**
9. **Feed-dependence** — validate on Vantage, not the TV chart feed.
10. **Beta check** — long-only in a secular bull = beta; demand short side / another instrument.
    **→ 分岐（ベータは診断であって判定ではない）**: ベータだと分かったら却下せず、(a) **そのベータ自体を素直に取る版**を回す
    （条件付けを外して常時同方向。しばしばこちらが本命）、(b) **時間シェア超過**か（窓の占有時間 vs 取った値幅、対数リターンで）、
    (c) **レジームを割っても残る**か（トレンド上昇/下降で層別。ベータなら下降で消えるか反転する）、(d) **別銘柄で機構が一致する**か。
    (b)(c)(d) を通れば「ベータに乗る」は正解。通らなければ、そこで初めて却下。
11. **🚨 執行モデルは「約定した足そのもの」も損切り判定に含めろ**（2026-07-13, `breakout_wave.py` の押し目指値パスで発覚）。
    指値/ストップ/リテストは**約定バーと損切りバーが同じ足になりうる**。前進走査を `約定足+1` から始めると、
    その足でタダ乗りが発生する。**しかも汚染は「指値と損切りの近さ」とともに増える**（間隔 = (1−押し目)×損切り幅）:
    実測 押し目0.25→3% / 0.30→4% / 0.70→**23%**。これが「押し目を深くすると単調に良くなる」という**偽の発見**を
    生み、ブックを 8.28 → **7.88** に盛っていた。**同足タイブレーク（損切り優先）は、約定足にも適用する。**
12. **No lookahead** — HTF via shift/confirm-later; next-bar-open fill; intrabar SL/TP. **外部データ(UTC)を
    Vantage CSV(ブローカー時刻=EET/EEST=UTC+2/+3)に結合する時は必ずtz変換**（`tz_convert("Europe/Riga")`）。
    素で突き合わせると窓の後半が未来になる（2026-07-12にフロー退出の🟢判定3件がこれで死んだ）。検算＝リターン相関のラグ探索。
13. **Log every try (incl. failures)** — multiple comparisons raise the bar. Don't loop until good results.

Workflow: mechanize faithfully → full history all-signals `--peryear` → checklist → not-one-era-beta →
`overfit_audit.py`（necessary, not sufficient — live-forward decides regime-change）→ sizing (CAGR/DD, DD×1.5–2
for live, 1% risk default, never >3%) → portfolio. Compare on **CAGR/DD**, not ret/DD.

## Structural laws (details & evidence: `docs/structural_priors.md`)
1. TFはmethod×instrument固有 — 1つのTF kill を他methodへ一般化しない。
2. 銘柄の性格がmethodを決める: gold/BTC=トレンド(ロング)、USDJPY=管理相場。FXのトレンドは政策乖離の時代だけ形成され、
   2018-以降のFXプラスセルは全てドル買い方向＝単一ドル因子の疑い（例外: USDJPY 1hロング=3時代プラス、GBPUSD 4hショート）。
3. WHEN（レジーム選択）が最大のレバー。生き残りゲートは KAMA-rising（breakout族）と週足30MAサイクル（pullback専用）のみ。
   固定ゲートは銘柄固有・適応型のみ転移。ゲートは「戦略に欠けている文脈を、必要な方向で」補う時だけ効く。
4. 不変パターン: トレンド＋確定終値エントリー＋勝ちを伸ばす(RR2–3)＋レジームゲート。検出器/フィルタは~0 lift。
   勝ちを切る出口（レベルTP・構造トレール・タイトRRフィルタ）は逆向き — fixed-RR law はFXまで拡張済み。
   🔑🔥 **【出口法則の一般形・2026-07-13 に確定】出口が価値を持つのは「退出価格 > その時点の期待値」のときだけ。**
   **検出器の質の問題ではない。** btc15m_A で「1Hレンジ」検出器を作ったら、**ランダム退出nullを4パーセンタイルで
   通過した＝本当に伸びない本を正確に選べていた**（選ばれた本の"持っていたら"meanR +0.877 vs ランダム +1.989）。
   **それでも同DDで CAGR −0.6〜−10.1pt（全12セル負け）。** 理由: **「伸びない」本は動いていないのだから
   定義上 0R 付近にいる。降りると 0R、持つと +0.88R。** 損切りが −1R で天井を打っているので
   「ダメを避ける」価値は高々1R。ところが**そのダメなやつでさえ 24% は 4.5R まで走り、それが全部を支えている**。
   ∴ **どんなに良い in-hold 検出器を作っても、それを出口に使ってはいけない。**（サイズか、入口か、に使え）
   🔑 **時間ストップも同じ壁**: 勝ちの保有 14.2h vs 負け 4.2h（**負けは速く死ぬ**）。「h時間たっても生きている」本は
   **時間とともに良くなる**（最終R 全部 +1.085 → 6h生存 +1.975 → 24h生存 **+2.332**、利確到達率 35%→**56%**）。
   🔑 **なぜ構造トレールが必ず負けるのか（2026-07-13 に機構を特定）**: 1時間足の押し目安値を割ったトレードは、
   **そのまま持てば平均 +2.649R。ランダムに同数選ぶと +1.038R（2パーセンタイル）**。
   ＝ **構造退出は「悪いトレードを避ける」のではなく「一番伸びるトレードを選んで切っている」。**
   大きく走る前には深い押しが入り、それが構造を割る。**「もう終わりだ」と感じる瞬間が「これから走る」瞬間。**
   🔑 **「持ち続けられない」への正しい対処は、出口ではなく賭け率**（ユーザーの心理的制約は真剣に扱う。
   守れないルールは価値ゼロだから）。**痛み = 賭け率 × R。R の分布はいじらず、賭け率を下げろ。**
   btc15m_A: 賭け率を 1.00%→0.50% にする代償は CAGR/DD 4.25→3.96（−0.29）**だけ**。
   一方、出口を構造トレールにすると**同DDで CAGR −10〜16pt**。
   ∴ **「怖いから小さく張る」はほぼ無料。「怖いから早く降りる」は一番高い買い物。**
   （「RR4.5 を 0.5% で持つ」＞「RR1.5 を 1% で持つ」— 感情的に楽で、しかも儲かる）
5. 構造ブレイク検出器（トレンドライン各種）は全て gold_bo を再導出＝冗長。
6. エッジと独立性はトレードオフ（エッジ有=金属クラスタで冗長、独立=イベント駆動でエッジ無し）。
7. トレンド正典のprimitive（breakout/MA/TSMOM）は全て検証済み。残る軸=新entry族・WHENの粒度・執行。
8. 静的inv-volに勝つ動的配分レバーは未発見。equity-gate・レッグ間モメンタムとも死。
   🚨 **ただし inv-vol 自体に2つの欠陥がある**（2026-07-13 に btc15m_L の分割で発覚）:
   (a) **「1レッグあたり 1/σ」を正規化するので、1つの脚を2つに割ると、その一家の予算配分がほぼ2倍になる**
       （btc15m_L: 0.475% → 0.647%）。**ランダムに割っただけでブックが −1.41 動く**。
       ∴ 分割・統合の比較は必ず「**一家の総重量を固定**」して行う。
   (b) **inv-vol は「ばらつき」で測り「エッジ」を見ない。** σが小さくエッジも無い部分集合に、大きい重みを与える。
       btc15m_L を PDH で割ると σ(A)=3.40 / σ(B)=2.85 で、**稼がない B のほうが重くなる**（A 0.270% vs B 0.323%）。
       一家の中で inv-vol を使うと同DDで **CAGR −25.1pt**。**ブックが「フル/半分」の決め打ちを使っているのは正しい。**
   🔬 **配分の判定は必ず「同じ maxDD にそろえて CAGR で比べる」**（レバレッジを完全に排除できる唯一の方法）。
   CAGR/DD の比較は、σ が下がると重みが上がる経路でレバレッジを買ってしまう（2026-07-13 に4件を撤回）。
   （2026-07-13: 「BTCが直近4週間で走った直後は玉を減らす」が例外候補に見えたが、**月次DD審判のアーティファクト**
   で撤回。トレード解像度の審判では 6.84→6.84 の同値、ブロックを伸ばすとPが50%へ縮む。反証チェックリスト8を参照。
   銘柄レベル（BTC4レッグ全部）への一般化も失敗＝コイン投げ）
9. **トレンドのレッグは「老いない」**（2026-07-13, gold/BTC/FX6ペア×4h/1d/週足×4時代）。残り巡行幅の平均は
   レッグの年齢に依存しない（2.15→2.02で平坦。中央値の低下は検出器バイアスで、帰無も同じだけ下がる）。
   ∴「そろそろ終わる」判断（時間ストップ・サイクル年齢ゲート・伸びたから利確）は全て機構的に無効。
   **遠い固定目標が最適**であることの理由であり、btc15m_L の RR4.0→4.5 の根拠。
   系: **入口の「強さ」は"どこまで伸びるか"を予言しない（＝目標の変数でない）、"機能するか"を予言する（＝サイズの変数）**。
   ⚠️ **ただしこれは per-trade の法則であって、ブックの滑らかさの法則ではない**（2026-07-13 に現役3レッグで検証）。
   RRを伸ばすと meanR/PF は法則どおり上がるが、勝率が下がって資産曲線がゴツゴツになり **CAGR/DD は落ちる**。
   **遠い目標を採るには頻度が要る**（btc15m_L=年200本超なら均されるが、年6〜12本の4H/1Hレッグでは均されない）。
   ∴ gold_bo=RR3 / btc_bo_kama=RR2 / btc_pull=RR3 は既に最良で、動かす余地は無かった。
9b. **法則9は「日足レジーム」の条件付けの下でも生き残った**（2026-07-13, btc15m_A）。ユーザーの裁量ルール
   「日足が下降トレンドのときの短期の上げは、伸ばさず利確する」を機械化 → **観察は正しいが対処法が逆**。
   素の巡行幅は確かに半減する（**MFE中央値 2.69R→1.19R**、P(4.5R到達) 43%→27%）。**だが日足↓の層でも
   最適RRは 4.5〜6.0 のまま**で、近い利確は meanR **−0.07(RR1.0) / −0.05(RR1.5)** ＝ゼロ以下。
   **残った 27% の裾が期待値の全部を担っている。** ∴ **玉を減らせ。利を切るな。**
   正解 = 日足↓のとき **サイズ ×0.75**（同DDで CAGR **+9.3pt**、丘型、両端 negative、ブロック伸長で P 上昇）。
   「建てない」は **−9.7pt**（法則11: ドリフト順方向のロングは見送るな）。
10. **レッグの改善 ≠ ブックの改善。** 統計監査（DSR/PBO）に全通過してもブックで落ちる（2026-07-13, HH4Hサイズ:
   レッグCAGR/DD 1.99→3.02 だがブックは **6.84→6.20**（トレード解像度審判）で却下）。
   **削った"弱い玉"が他レッグとの無相関を担っていた**＝法則6の実例。
   採否は必ずブックのCAGR/DDで裁定する。

10b. **単独運用とブックでは、最適な設定が違う**（2026-07-13, btc15m_A の日足サイズで確定）。
   同じルールが**単独では同DDで CAGR +9.3pt、ブックでは −2.1pt**。理由: 弱いトレード（日足↓の meanR +0.34）は
   単独運用では足を引っ張るだけだが、**ブックでは他レッグとの無相関を担っている**（法則6）。
   ∴ **「単独で回す脚」と「ブックの中の脚」は、別の設定表を持つ。** README に両方を書くこと。
11. **ドリフトと逆向きのレッグは、ハードルを上げる**（2026-07-13, BTC 15分 L/S で確定）。BTCには上昇ドリフトがあり、
   **ロング（ドリフト順方向）は「見送るな、小さく張れ」＋速いゲート（4h）が正解**、
   **ショート（逆方向）は「厳しく切れ」＋遅いゲート（日足）が正解**。
   実測: ショートを4hゲートにすると弱気年の稼ぎは増える（2022: +16→+32R）が**強気年で出血**（2019: +3.1→−0.7R）
   ＝上昇相場の押し目を全部ショートしに行く。前日安値フィルタをソフト化するとブックが単調悪化（8.27→6.17→5.65→5.38）。
   ∴ **同じ機構でも、ドリフトに対する向きでゲートの速さとフィルタの厳しさが反転する。**
   例外は目標(RR)で、こちらは向きに依らず遠いほうが良い（法則9。L/S とも RR4.5）。
