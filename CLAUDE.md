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
  - gold: `data/vantage_xauusd_{h1,m15,m5,m1}.csv`（h1/m15 は 2007→だが 2017 以前は極端に疎＝実質 2018-。
    **gold h1 は必ず `--start 2018-01-01` を付ける**：2026-07-13 の swings_zigzag 修正で疎データ領域にも
    トレードが出るようになり、付けないと IS/OOS が汚染される）
  - 週足（2026-07-13 にブリッジで取得）: `vantage_{eurusd,usdjpy}_w1`(1971→)· `{gbpusd,audusd,nzdusd,usdcad}_w1`
    (1993-94→)· `{xauusd,btcusd}_w1`(2017→)。※EURUSD 1999以前・USDJPY 1973以前は合成/固定相場につき使用禁止。
    銘柄名注意: ターミナル上の gold は **`XAUUSD+`**（`XAUUSD` は存在しない）。
  - BTC: `vantage_btcusd_{h1,m15,m5}.csv` (2017→) · USDJPY: `{h1,h4,d1,m15,m5,m1}` (h1 2000→26.5yr)。
    **USDJPY m5 は 1999 以前が年250本＝日足がm5ラベルで入っている**（gold h1 と同型の罠、2026-07-13 発見）。
    日中の検証では `.loc["2000-01-01":]` 等で切ること。
  - FX majors (eurusd/gbpusd/audusd/nzdusd/usdcad): `{m15,h1,h4,d1}` all 2000→2026 (26.5yr)
  - **指数・その他コモディティ（h1、2026-07-13 に存在を確認。全て gold_bo レシピで検証済み＝全滅）**:
    `nas100.r`(2016→)· `ger40.r`(2015→)· `us2000.r`(2020→)· `xagusd`(2015→、m15 は 2018→)·
    `usousd`原油(2015→)· `xptusd.r`(2022→)。**再テスト前に `docs/verified_findings.md` の
    「レシピの横展開・全滅」を見ること**（4銘柄ともランダム建て帰無を超えない。銀は金と年別R相関 0.81 で冗長）。
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
| `src/engine/` | 上2本の分解エンジン（gates/detect/plan/**walk**/stats、2026-07-17）。**執行ウォーカーは walk.py だけ**＝執行の修正は1箇所。旧 run() は薄い委譲ラッパー（呼び出し側は無変更）。**engine を編集したら `scratchpad/engine_tieback.py`（41構成）と `engine_golden.py check-run` の全PASSが必須**（正典とのビット一致番人）。新規スクリプトの自前ウォーカー実装は禁止＝ここから import |
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
   **頻度の違うレッグを混ぜる時は、重みをトレードRのσで出す**（詳細 `docs/findings/s07_sizing.md`）。
   🚨 **さらに: maxDD の「実測値」を1本の経路から読むな。必ずブートストラップして中央値を使え。**
   2026-07-13: btc15m_A の実測 maxDD 8.5% は、巡回ブロック・ブートストラップ（3000回）の**下から18%の位置**
   ＝運の良い経路だった。**想定値は中央値の 10.2%**（95%点 15.6%）。CAGR/DD は 4.25 ではなく **3.54**。
   ユーザーが「RRごとのDDを教えて」と訊いたことで発覚。**賭け率は必ず中央値のDD（×1.5〜2）から決める。**
9. **Feed-dependence** — validate on Vantage, not the TV chart feed.
10. **Beta check** — long-only in a secular bull = beta; demand short side / another instrument.
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

## The current book — 6 legs, adopted 2026-07-13（全仕様は `README.md`、数字は `project_auto_trade.md`）
審判＝**トレード解像度DD × トレードRσの逆数（inv-vol）· 総リスク3%**。**年206本 / CAGR +61.0% / maxDD 7.74% / CAGR/DD 7.88**（Pine が実際に発注する仕様＝押し目指値の期限200本・S の RR4.5。2026-07-13 の約定足バグ修正後）
（同じ物差しで旧3レッグは 3.03。想定実DD = 11〜15% ＝ backtest×1.5〜2）。Pine は `pine/<asset>_<tf>_*.pine`。

| leg | 銘柄/TF | 機構 | 出口 | ゲート | 年 |
|---|---|---|---|---|---|
| gold_bo | gold 1H | ZigZag(2×ATR) Pattern-B 確定足ブレイク・成行 | RR3 | 日足SMA150↑ | 29 |
| btc_bo_kama | BTC 4H | 同上 | RR2 | 日足KAMA(14)↑ | 8 |
| btc_pull | BTC 4H | EMA20押し目（SMA80トレンド） | RR3 | 週足終値 ≤ 30週MA×1.10 | 10 |
| gold15m | gold 15m | 同ブレイク＋**押し目指値0.25**・ext-cap 8% | RR4 | 日足SMA150↑ | 44 |
| **btc15m_L** | BTC 15m | 同ブレイク＋**押し目指値0.30**・PDHソフト0.5 | **RR4.5** | **4h**-KAMA(14)↑ | 100 |
| btc15m_S | BTC 15m | その鏡像（戻り売り指値0.30・前日安値割れ必須） | **RR4.5** | **日足**KAMA(14)↓ | 12 |

- **btc15m_L がブックの生命線**（抜くと 7.88→3.48）。**btc_bo_kama は CAGR/DD を上げないが DD を下げる**
  （同DDに揃えると CAGR +5.3pt ＝リスク・ダイヤルでは代替不可）。**gold15m はセッションスキップ禁止**（捨てる窓が黒字）。
- **ロング/ショートの非対称は測定で裏付け済み**（構造法則11）: ショートは日足ゲート・PDLハード・フィルタが正しく、
  4hゲート／PDLソフト化はいずれもブックを悪化させる。RRだけ両方 4.5 で揃う。
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
