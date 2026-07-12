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
agreeing**, including the user's own and LLM-authored ones. A good-looking backtest is a
hypothesis, not a result. Never cheerlead a number — stress it first.

## 🧪 YOUR ROLE — keep doing R&D; the user decides when to close
結果が出たら、毎回**データから逆算した改善案を出して前進する**こと。「記録して閉じますか？」と訊いて手を止めない —
**adopt / kill / close の判断はユーザーが行う**。NON-ADOPTED と測定できても、そこで止めず「**なぜ落ちたか＝病巣**」を
特定し、それを直接叩く次の実験を提案・実行する。falsify は「殺して終わり」ではなく「殺し方から次の一手を生む」こと。

**進め方の工程表は `/edge-rd` スキル**（アイデア発散=モードA / 実験=モードB。仕様カード凍結→承認→実装→照合→台帳）。
**計測・スクリプト実装は `measure` サブエージェント**（sonnet, `.claude/agents/measure.md`）に委任し、
メインは前提の保持と**返ってきた数字のローカル再実行照合**に専念する。

## Environment / how to run
- `python` is NOT on PATH. Always use `.venv/bin/python`.
- Data loader: `from src.data_loader import load_mt5_csv` (root scripts import directly;
  `research/` scripts already `sys.path.insert` the project root). Keeps MT5 broker-server
  time as the clock so HTF bins align. Auto-drops feed-glitch bars (stderr warning).
- Data (Vantage feed = what the user actually trades; validate on THIS, not the chart feed):
  - gold: `data/vantage_xauusd_{h1,m15,m5,m1}.csv`（h1/m15 は 2007→だが 2017 以前は極端に疎＝実質 2018-）
  - BTC: `vantage_btcusd_{h1,m15,m5}.csv` (2017→) · USDJPY: `{h1,h4,d1,m15,m5,m1}` (h1 2000→26.5yr)
  - FX majors (eurusd/gbpusd/audusd/nzdusd/usdcad): `{m15,h1,h4,d1}` all 2000→2026 (26.5yr)
  - The file is the source of truth for spans, not these notes. Resample inside scripts via `--tf 4h`.
- **LIVE account costs (Vantage RAW/ECN, JPY, limit-order execution; measured 2026-07-02):**
  commission ≈ **$3/lot/side flat** (gold $0.06/oz RT; USDJPY ≈0.9 pip RT; **BTC commission=0**,
  cost = floating spread ~$10–25). Realistic ROUND-TRIP price-distance cost = 1×spread + commission:
  **gold ≈ $0.15–0.35/oz** (backtest canon $0.6 = 2–3× conservative), **BTC ≈ $10–25** (canon $15 ≈ real),
  **FX ≈ 0.9 pip**. Buy-limit fills are ASK-based (on BID data the limit fills only when bid ≤ limit − spread);
  stops slip in fast markets (model separately).
- **Refresh OHLCV from MT5 (demo) via the `mt5-mcp` bridge** (sibling repo `../mt5-mcp`):
  `bash ../mt5-mcp/scripts/run_backtests.sh` = refresh + re-run the book (jobs in `config/runbook.yaml`).
  Data-only: `../mt5-mcp/.venv/bin/python ../mt5-mcp/client/export_csv.py --symbol XAUUSD --tf h1`.
  Requires the bridge up + MT5 terminal logged in. Shrink-guarded. auto-trade is invoked, never modified.

## The toolkit (reuse these; don't reinvent — copy-paste configs in `docs/toolkit_examples.md`)
| script | what it tests |
|---|---|
| `breakout_wave.py` | Elliott Pattern-A/B breakout（gold_bo/btc_bo の本体；--pullback-frac, --retest 等） |
| `ema_pullback.py` | EMA pullback-continuation（btc_pull の本体；--gate-tf 系でサイクルゲート） |
| `mfe_mae.py` | generic entry-edge SCREEN (MFE/MAE ratio)：<1.0 dead, >1.2 worth deeper test |
| `research/edge_harness.py` | **標準evalハーネス — 新signalは必ずこれに通す**（PF/N/リスク/TFラダー/ベータnull/先読み禁止をコードで強制） |
| `research/scalp_lab.py` | anti-overfit intraday harness (orb/squeeze/bounce; IS/VAL/sealed TEST) |
| `research/regime_discriminator.py` | 任意signalの効く場面/効かない場面を IS→OOS＋random-drop null＋年別ON% で見分ける一次スクリーン |
| `research/overfit_audit.py` | **MEASURE overfit risk** (Deflated Sharpe + PBO/CSCV + bootstrap-CI/null) — 採用前の標準ゲート |
| `research/portfolio.py` / `portfolio_alloc.py` | combine legs into one equity curve + annual-R correlations / allocation |
| `research/gate_passrate.py` | year-by-year ON% of candidate regime gates |
| `research/instrument_screen.py` | trend-CHARACTER pre-screen of NEW instruments (PRE-SCREEN only; Vantage H1 = arbiter) |

Most tools report `n, win%, PF, meanR, totR, IS/OOS, maxDD` and `--peryear`. Cost is modeled; raise it to stress-test.

## 🔪 The falsification checklist (run BEFORE believing any edge)
1. **All-signals base first** — filters CONCENTRATE an edge, they don't create one.
2. **Win rate vs RR-breakeven** (1/(1+RR); RR3→25%). Win≈breakeven ⇒ entries are RANDOM.
3. **IS vs OOS.** IS≫OOS = back-loaded / curve-fit / regime luck.
4. **±1 sweep every parameter.** Real edge = PLATEAU; overfit = lone SPIKE.
5. **Per-year/era spread.** Profit in one era = beta, not edge.
6. **Cost realism** — but judge in order: 素の率×幅→偶然性→コスト→口座寄与。「エッジ無し」と「エッジ有・コスト死」は別ラベル。
7. **Selection rules (caps/1日N回) are luck-sorters** — always compare to base. Within-leg filters must beat the
   **CAGR/DD** random-drop null (not just meanR).
8. **Feed-dependence** — validate on Vantage, not the TV chart feed.
9. **Beta check** — long-only in a secular bull = beta; demand short side / another instrument.
10. **No lookahead** — HTF via shift/confirm-later; next-bar-open fill; intrabar SL/TP. **外部データ(UTC)を
    Vantage CSV(ブローカー時刻=EET/EEST=UTC+2/+3)に結合する時は必ずtz変換**（`tz_convert("Europe/Riga")`）。
    素で突き合わせると窓の後半が未来になる（2026-07-12にフロー退出の🟢判定3件がこれで死んだ）。検算＝リターン相関のラグ探索。
11. **Log every try (incl. failures)** — multiple comparisons raise the bar. Don't loop until good results.

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
5. 構造ブレイク検出器（トレンドライン各種）は全て gold_bo を再導出＝冗長。
6. エッジと独立性はトレードオフ（エッジ有=金属クラスタで冗長、独立=イベント駆動でエッジ無し）。
7. トレンド正典のprimitive（breakout/MA/TSMOM）は全て検証済み。残る軸=新entry族・WHENの粒度・執行。
8. 静的inv-volに勝つ動的配分レバーは未発見。equity-gate・レッグ間モメンタムとも死。

## The current surviving book (numbers in auto-memory `project_auto_trade.md`)
- **gold 1H breakout**（daily-SMA150+slopeゲート, RR3）· **BTC 4H breakout**（RR2＋daily-KAMA-rising）·
  **BTC 4H EMA pullback**（SMAトレンド, RR3＋週足30MAサイクルゲート）。3-leg inv-vol CAGR/DD ~2.91、
  caveat PBO 0.53（レジーム集中）→ live-forward 進行中（再提案しない）、BTC族は合計≤parity。
- **Validated candidates（未採用・live-forward待ち・≤parity）:** gold 15M breakout（ext-cap 8%+RR4、
  9–15UTCスキップ強化）· H17-S（gold ORBショート＋daily-SMA80-falling）。
- **Dead の一覧と経緯は `docs/structural_priors.md` と `docs/verified_findings.md`** — 再テスト前に必ず照合。

## Where things live
- **検証済み台帳（回す前に見る・確定したら追記）: `docs/verified_findings.md`＝1行索引、本文は `docs/findings/*.md`。
  検索は `grep -r <語> docs/verified_findings.md docs/findings/`**
- 提案バックログ（機構/検証順/合格基準/死に方）: `docs/proposals.md`（決着分の本文は `proposals_archive.md`）·
  探索の入口: `docs/idea_exploration_playbook.md`
- 事前登録ログ: `docs/scalp_research_log.md`（過去試行は `scalp_research_log_archive.md`）· 深掘り: `docs/findings_*.md` ·
  法則の詳細: `docs/structural_priors.md`
- 工程表スキル: `.claude/skills/edge-rd/SKILL.md` · 計測係: `.claude/agents/measure.md`
- Pine strategies: `pine/<asset>_<tf>_*.pine`（機能コメントのみ、研究履歴は書かない）
- Engine split: research = Python (Vantage CSVs); see/alert = TradingView (Pine); live = Vantage MT5 (manual)。
  Validate on Vantage; TV chart feed ≠ trade feed。データ更新は `../mt5-mcp`（前節）。
