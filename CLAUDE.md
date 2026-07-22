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
    - **🚨 gold・BTC の m15 は 2018-09-14 まで実質1時間足（24本/日）、以降が本物の15分足（92-96本/日）**
      （2026-07-19 発見・両銘柄とも同一日付＝フィード仕様。gold h1 疎データ罠と同型）。**m15 の日中研究は必ず
      `2018-10-01` 以降に切る**（密化＋ATR14ウォームアップ余裕）。付けないと初期のイベント/バックテストが粗い足で汚染される。
      **gold m5 も同一日付で同じ罠**（2026-07-21 実測: 2018-09-13 まで 23.8本/日 ＝実質1時間足 → 2018-09-15 以降
      274.3本/日）。**gold m5 も `2018-10-01` 以降に切る**。BTC m5 はファイル自体が 2019-01-01 開始なので無関係。
    - **gold m1 は 2026-07-19 に橋(mt5-mcp)で 2019-2026 を再取得済み**（`data/vantage_xauusd_m1.csv`＝267万行・1分刻み。
      旧版は直近7か月200k行だけ＝輸出上限で切れていた、`*.bak_recent200k`）。**ブローカーのm1保持は2019年から**（2018はほぼ無し）。
      **橋の銘柄名は現Demo口座では `XAUUSD`（`XAUUSD+` は旧口座の記述で今は存在しない）**。橋が `IPC send failed` を返す時は
      端末未接続＝**8765を握る古いプロセスをkillして `bash ../mt5-mcp/scripts/run_bridge.sh` で立て直す**（PID特定は
      `powershell.exe -Command "(Get-NetTCPConnection -LocalPort 8765 -State Listen).OwningProcess"` → `taskkill.exe /F /PID`）。
  - 週足（2026-07-13 にブリッジで取得）: `vantage_{eurusd,usdjpy}_w1`(1971→)· `{gbpusd,audusd,nzdusd,usdcad}_w1`
    (1993-94→)· `{xauusd,btcusd}_w1`(2017→)。※EURUSD 1999以前・USDJPY 1973以前は合成/固定相場につき使用禁止。
    銘柄名注意: ターミナル上の gold は **`XAUUSD+`**（`XAUUSD` は存在しない）。
  - BTC: `vantage_btcusd_{h1,m15,m5}.csv` (2017→) · USDJPY: `{h1,h4,d1,m15,m5,m1}` (h1 2000→26.5yr)。
    - **🚨🚨 暗号資産は 2021年まで「平日のみ」の商品だった**（2026-07-21 確認。BTC h1 の曜日別本数＝
      2018-2021: 月〜金 各4,700前後に対し **土98・日351**／2022-2026: 土4,625・日5,443 ＝**2022年から土日も稼働**。
      平日内の時刻分布は均一なので**データ破損ではなく商品仕様の変更**。ETH/XRP/LTC/BCH も同形）。
      **暗号資産の日中研究は `2022-01-01` 以降に切る** — 2021年以前は今はもう存在しない市場で、
      月曜の ATR・前日高値・前方N本が週末をまたぐ。混ぜると水増しになる
      （実測: ATR拡大足レッグの PF 2.12→1.59・1トレード +0.821%→+0.403%）。**再取得しても直らない**（履歴が本当に無い）。
    - **LTC の 2012-2017、ETH/XRP の 2016-2017 は年200〜365本＝日足が h1 ラベルで入っている**（USDJPY m5 と同型）。
    - 暗号資産の追加データ（2026-07-21 に橋で取得、いずれも h1）: `{ethusd,xrpusd,ltcusd,bchusd,trxusd,solusd,
      adausd,dotusd,bnbusd}`。ETH/XRP/LTC/BCH は2018→（濃いのは2022→）、SOL/ADA/DOT は2021→、BNB は2022→。
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
| `src/engine/` | 分解エンジン（gates/detect/plan/**walk**/stats/**size**/**arbiter**/**walk_ict**/mirror、2026-07-17）。**執行ウォーカーは walk.py（breakout系＋walk_ema）と walk_ict.py（ICT系: ASK基準指値・キルゾーン・NY壁時計）、サイズ写像は size.py、同DD裁定は arbiter.py だけ**＝各層の修正は1箇所。旧 run() は薄い委譲ラッパー、旧 `scratchpad/arb_common.py`・`ict_exec.py` は転送シム（呼び出し側は無変更）。**engine を編集したら番人3本の全PASSが必須**: `scratchpad/engine_tieback.py`（41構成）・`engine_golden.py check-run`・`size_tieback.py`（ICT系はアブレーション出力のバイト一致で照合済み）。新規スクリプトの自前ウォーカー/サイズ写像/裁定器の実装は禁止＝ここから import |
| `mfe_mae.py` | generic entry-edge SCREEN (MFE/MAE ratio)：<1.0 dead, >1.2 worth deeper test |
| `research/edge_harness.py` | **標準evalハーネス — 新signalは必ずこれに通す**（PF/N/リスク/TFラダー/ベータnull/先読み禁止をコードで強制） |
| `research/scalp_lab.py` | anti-overfit intraday harness (orb/squeeze/bounce; IS/VAL/sealed TEST) |
| `research/regime_discriminator.py` | 任意signalの効く場面/効かない場面を IS→OOS＋random-drop null＋年別ON% で見分ける一次スクリーン |
| `research/overfit_audit.py` | **MEASURE overfit risk** (Deflated Sharpe + PBO/CSCV + bootstrap-CI/null) — 採用前の標準ゲート |
| `research/portfolio.py` / `portfolio_alloc.py` | combine legs into one equity curve + annual-R correlations / allocation |
| `research/book.py` | **構成の正典パイプライン（2026-07-17）**: 採用6レッグを運用仕様（fill_win200/S=RR4.5/ネットコスト/PDHソフト/PDLハード）で構築し、採用審判（トレード解像度DD×トレードRσ逆数・総3%）で裁定。`get_book_legs()`/`book()`/`w_trade()`。アンカー=206本/年・CAGR+61.0%・maxDD7.74%・**CAGR/DD7.88**。番人=`scratchpad/book_tieback.py`（凍結証拠スクリプトと配列一致12検査）。**構成の裁定を伴う実験は今後これを import する（book_integration 等の手書き再構築は禁止）** |
| `research/gate_passrate.py` | year-by-year ON% of candidate regime gates |
| `research/instrument_screen.py` | trend-CHARACTER pre-screen of NEW instruments (PRE-SCREEN only; Vantage H1 = arbiter) |
| `research/instrument_character.py` | 銘柄の素質分解装置（7軸: ドリフト/VR・Hurst/周期/季節/集中度/ボラ/分布 ＋ method-fitタグを13銘柄横並び）。法則2を測定化・自己検証付き。**日次VR単体でmethod-fitを切らない**（全トレンド銘柄がgrind-up＝日次平均回帰なので"fade"誤判定する）。詳細 `docs/findings/x_instruments.md` |

Most tools report `n, win%, PF, meanR, totR, IS/OOS, maxDD` and `--peryear`. Cost is modeled; raise it to stress-test.

## 🔪 The falsification checklist (run BEFORE believing any edge)
🔒 **順序はフックで強制**（`.claude/hooks/screen_gate.py`）。トレード統計を出すスクリプトは、巡行幅(MFE/MAE)を
先に測っていないと実行できない。通し方＝`research/screen.py` の `run_screen(name, df, entries)` を走らせ、
スクリプト先頭に `SCREEN = "<name>"` を書く。検算は**既知の値への数値 assert**（印字して目視は検算ではない）。
🚨 **これは棄却基準であって停止条件ではない。** ❌を出す時は「この却下が誤りだったら成立していたはずの対立仮説」を
1つ書き、安く測れるなら測ってから却下する。**台帳に書くのは対立仮説を潰した後**（書いてから疑うと、書いたものを
守る側に回る。実例＝ベータとして却下 → 再検定で捨てた側が30倍良かった）。

1. 素の全信号をまず見る — フィルタはエッジを濃縮するだけで、無からは作らない。
2. 勝率 vs RR分岐点 (1/(1+RR); RR3→25%)。分岐点付近＝入口はランダム。
3. IS≫OOS = 後半偏り・曲線当てはめ・レジームの運。
4. 全パラメータを±1スイープ。真のエッジ＝台地、過剰当てはめ＝単独の尖り。
5. 年別・時代別の広がり。1つの時代だけの利益はベータであってエッジではない。
6. コスト現実性。判定順＝素の率×幅→偶然性→コスト→口座寄与（「エッジ無し」と「エッジ有・コスト死」は別ラベル）。
7. 選択ルール（上限・1日N回）は運の選別器 — 必ず素の母集団と比べる。レッグ内フィルタは CAGR/DD の
   ランダム間引き帰無を超えること。**ただしそれは必要条件どまり**（同じ価格経路の上での話しか訊いていない）。
   **巡回ブロック・ブートストラップ（1/3/6/12か月）も必ず通す** — 真の改善はブロックを長くするほど勝率が上がり、
   経路当てはめは上がらない。**レッグだけでなく構成の CAGR/DD にも適用する。**
7.5. 🚨 **inv-vol 重みの構成では「R のばらつきを下げる操作」はすべて自動でレバレッジを買う。**
   フィルタが効いたのではなく賭け金が増えただけ。**必ず「重みを現行に固定した版」と並べて報告する** —
   固定して差が消えたら、それは新しい知見ではなくレバレッジ・ダイヤル。
   🔬 兄弟: サイズ倍率で割った量（損切り幅÷価格÷サイズ倍率 等）は**サイズ・ルールを暗黙に含む**。
   統計の検定を全部通しても、変数の定義が違えば無意味。
8. 🚨 **構成の maxDD はトレード（or 日次）解像度で測る — 月次資産曲線で測ってはならない。**
   月次に潰すと月内で完結するDDが全部消え、その上で下した判定の順位が全部入れ替わる（低頻度レッグ同士では
   一致するので、**高頻度レッグを足した瞬間に壊れる**）。`portfolio_alloc.py` は `cagr_dd_trades()` で報告する。
   同じ根の兄弟: **inv-vol を月次σで出すと低頻度レッグが過大な玉を貰う**（建てない月をゼロ＝低ボラと誤読）
   → **頻度の違うレッグを混ぜる時は、重みをトレードRのσで出す。**
   さらに **maxDD の実測値を1本の経路から読むな** — ブートストラップの中央値を使う（実測は運の良い経路でありうる）。
   **賭け率は中央値のDD×1.5〜2 から決める。**
9. フィード依存 — Vantage で検証する（TVチャート・フィードではない）。
10. ベータ点検 — 上昇相場のロングのみはベータ。**ただしベータは診断であって判定ではない**: 分かったら却下せず
    (a) **そのベータを素直に取る版**（条件付けを外して常時同方向。しばしばこちらが本命）、(b) 時間シェア超過か、
    (c) レジームを割っても残るか、(d) 別銘柄で機構が一致するか。(b)(c)(d) を通れば「乗る」が正解。
11. 🚨 **執行モデルは「約定した足そのもの」も損切り判定に含めろ。** 指値・ストップ・リテストは約定バーと損切りバーが
    同じ足になりうる。走査を `約定足+1` から始めるとタダ乗りが発生し、**汚染は指値と損切りの近さとともに増える**
    （押し目0.25→3% / 0.70→23%）＝「押し目を深くすると単調に良くなる」という偽の発見を生む。
    **同足タイブレーク（損切り優先）を、約定足にも適用する。**
12. 先読み禁止 — HTFはshift/確定後、次足始値で約定、足内で SL/TP。**外部データ(UTC)を Vantage CSV
    (ブローカー時刻=EET/EEST)に結合する時は必ず `tz_convert("Europe/Riga")`**（素で突き合わせると窓の後半が未来になる）。
13. 失敗も含め全試行を記録する — 多重比較はハードルを上げる。良い結果が出るまで回さない。

Workflow: mechanize faithfully → full history all-signals `--peryear` → checklist → not-one-era-beta →
`overfit_audit.py`（necessary, not sufficient — live-forward decides regime-change）→ sizing (CAGR/DD, DD×1.5–2
for live, 1% risk default, never >3%) → portfolio. Compare on **CAGR/DD**, not ret/DD.

## Structural laws（規則のみ。根拠・数字・経緯は `docs/structural_priors.md`）
1. TFはmethod×instrument固有 — 1つのTF kill を他methodへ一般化しない。
2. 銘柄の性格がmethodを決める: gold/BTC=トレンド(ロング)、USDJPY=管理相場。FXのトレンドは政策乖離の時代だけ形成され、
   2018-以降のFXプラスセルは全てドル買い方向＝単一ドル因子の疑い（例外: USDJPY 1hロング、GBPUSD 4hショート）。
3. WHEN（レジーム選択）が最大のレバー。生き残ったゲートは KAMA-rising（breakout族）と週足30MAサイクル（pullback専用）。
   固定ゲートは銘柄固有・適応型のみ転移。**ゲートは「戦略に欠けている文脈を、必要な方向で」補う時だけ効く。**
4. 不変パターン: トレンド＋確定足エントリー＋勝ちを伸ばす(RR2–3)＋レジームゲート。検出器/フィルタは lift ≈0。
   勝ちを切る出口（レベルTP・構造トレール・タイトRRフィルタ）は逆向き（FXまで拡張済み）。
   🔑🔥 **出口法則の一般形: 出口が価値を持つのは「退出価格 > その時点の期待値」のときだけ。検出器の質の問題ではない。**
   「伸びない」本は動いていない＝定義上0R付近にいるので、避けて浮くのは高々1R。ところが**その層でさえ24%は
   4.5Rまで走り、そこが全部を支えている**。∴ **どんなに良い保有中の検出器も、出口に使ってはいけない**
   （サイズか入口に使え）。時間ストップも同じ壁＝**生き残っている本は時間とともに良くなる**。
   構造トレールが必ず負ける機構＝**「悪い本を避ける」のではなく「一番伸びる本を選んで切っている」**
   （大きく走る前に深い押しが入り、それが構造を割る＝「もう終わりだ」と感じる瞬間が「これから走る」瞬間）。
   🔑 **「持ち続けられない」への対処は、出口ではなく賭け率**（守れないルールは価値ゼロなので心理的制約は真剣に扱う）。
   **痛み＝賭け率×R。Rの分布はいじらず、賭け率を下げろ。** 賭け率を半分にする代償は CAGR/DD −0.29 だけ、
   出口を構造トレールにすると同DDで CAGR −10〜16pt。∴ **「怖いから小さく張る」はほぼ無料、
   「怖いから早く降りる」は一番高い買い物**（RR4.5を0.5%で持つ ＞ RR1.5を1%で持つ）。
5. 構造ブレイク検出器（トレンドライン各種）は全て gold_bo を再導出＝冗長。
6. エッジと独立性はトレードオフ（エッジ有=金属クラスタで冗長、独立=イベント駆動でエッジ無し）。
7. トレンド正典のprimitive（breakout/MA/TSMOM）は全て検証済み。残る軸=新entry族・WHENの粒度・執行。
8. 静的inv-volに勝つ動的配分レバーは未発見（equity-gate・レッグ間モメンタムとも死）。
   🚨 **ただし inv-vol 自体に2つの欠陥がある**: (a) 正規化のせいで**1つの脚を2つに割ると、その一家の予算が
   ほぼ2倍になる**（ランダムに割るだけで構成が −1.41 動く）→ 分割・統合の比較は**一家の総重量を固定**して行う。
   (b) **ばらつきで測りエッジを見ない**ので、σが小さくエッジも無い部分集合に大きい重みを与える
   （一家の中で使うと同DDで CAGR −25.1pt）＝**構成が「フル/半分」の決め打ちを使っているのは正しい。**
   🔬 **配分の判定は「同じ maxDD にそろえて CAGR で比べる」**（レバレッジを排除できる形）。CAGR/DD の比較は、
   σが下がると重みが上がる経路でレバレッジを買ってしまう。
9. **トレンドのレッグは「老いない」** — 残り巡行幅の平均はレッグの年齢に依存しない（2.15→2.02＝平坦。
   中央値の低下は検出器バイアスで、帰無も同じだけ下がる）。∴「そろそろ終わる」判断（時間ストップ・サイクル年齢
   ゲート・伸びたから利確）は機構的に無効＝**遠い固定目標が最適**であることの裏付け。
   系: **入口の「強さ」は"どこまで伸びるか"を予言しない（＝目標の変数ではない）、"機能するか"を予言する（＝サイズの変数）。**
   ⚠️ **ただしこれは1トレードの法則であって、構成の滑らかさの法則ではない。** RRを伸ばすと meanR/PF は法則どおり
   上がるが、勝率が下がって資産曲線がゴツゴツになり CAGR/DD は落ちる。**遠い目標を採るには頻度が要る**
   （年200本超なら均されるが、年6〜12本の4H/1Hレッグでは均されない）＝現行の RR3/RR2/RR3 は既に最良。
9b. **法則9は日足レジームの条件付けの下でも生き残る。** 「日足が下降のときの短期の上げは伸ばさず利確」は
   **観察は正しいが対処法が逆**。素の巡行幅は確かに半減する（MFE中央値 2.69R→1.19R）が、**その層でも最適RRは
   4.5〜6.0 のまま**で、近い利確は meanR ゼロ以下。**残った27%の裾が期待値の全部を担っている。**
   ∴ **玉を減らせ。利を切るな**（サイズ×0.75 が同DDで CAGR +9.3pt、「建てない」は −9.7pt）。
10. **レッグの改善 ≠ 構成の改善。** 統計監査（DSR/PBO）に全通過しても構成で落ちる。
   **削った"弱い玉"が他レッグとの無相関を担っていた**＝法則6の実例。採否は必ず構成の CAGR/DD で裁定する。
10b. **単独運用と構成では、最適な設定が違う**（同じルールが単独 +9.3pt / 構成 −2.1pt）。弱いトレードは
   単独では足を引っ張るだけだが、構成では無相関を担っている。∴ **「単独で回す脚」と「構成の中の脚」は
   別の設定表を持つ。** README に両方を書くこと。
11. **ドリフトと逆向きのレッグは、ハードルを上げる。** BTCには上昇ドリフトがあり、**ロング（順方向）は
   「見送るな、小さく張れ」＋速いゲート（4h）**、**ショート（逆方向）は「厳しく切れ」＋遅いゲート（日足）**が正解。
   ショートを4hゲートにすると弱気年は増えるが強気年で出血し（上昇相場の押し目を全部売りに行く）、
   前日安値フィルタをソフト化すると構成が単調悪化する。∴ **同じ機構でも、ドリフトに対する向きで
   ゲートの速さとフィルタの厳しさが反転する。** 例外は目標(RR)で、向きに依らず遠いほうが良い（法則9）。

## 検証を通過した構成 — 6レッグ（2026-07-13）（全仕様は `README.md`、数字は `project_auto_trade.md`）
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

- **btc15m_L が構成の生命線**（抜くと 7.88→3.48）。**btc_bo_kama は CAGR/DD を上げないが DD を下げる**
  （同DDに揃えると CAGR +5.3pt ＝リスク・ダイヤルでは代替不可）。**gold15m はセッションスキップ禁止**（捨てる窓が黒字）。
- **ロング/ショートの非対称は測定で裏付け済み**（構造法則11）: ショートは日足ゲート・PDLハード・フィルタが正しく、
  4hゲート／PDLソフト化はいずれも構成を悪化させる。RRだけ両方 4.5 で揃う。
- **Dead の一覧と経緯は `docs/structural_priors.md` と `docs/verified_findings.md`** — 再テスト前に必ず照合。

## Where things live
- **検証済み台帳（回す前に見る・確定したら追記）: `docs/verified_findings.md`＝1行索引、本文は `docs/findings/*.md`。
  検索は `grep -r <語> docs/verified_findings.md docs/findings/`**
  - 本文は2軸（2026-07-21 再編）: **`m_*` = 手法別の台帳**（breakout / pullback / bounce / fade / ict /
    vwap_mtf / event / indicators ＋ book）· **`x_*` = 手法をまたぐ層**（entries / exits / gates / sizing /
    instruments / conventions）。**手法固有か、移転する知見か**で置き場を決める。
  - 🔒 **索引の1行は 300バイト上限**（全角100字）。超える分は本文へ `### <スラグ>` で置き、索引は
    「記号＋主張＋機構の一言＋(日付, ファイル#スラグ)」に留める。**本文は重くてよい／索引だけ軽く保つ。**
- 提案バックログ（機構/検証順/合格基準/死に方）: `docs/proposals.md`（決着分の本文は `proposals_archive.md`）·
  探索の入口: `docs/idea_exploration_playbook.md`
- 事前登録ログ: `docs/scalp_research_log.md`（過去試行は `scalp_research_log_archive.md`）·
  **構造法則の根拠・数字・経緯: `docs/structural_priors.md`**（CLAUDE.md は規則だけを持つ）
- 工程表スキル: `.claude/skills/edge-rd/SKILL.md` · 計測係: `.claude/agents/measure.md`
- Pine strategies: `pine/<asset>_<tf>_*.pine`（機能コメントのみ、研究履歴は書かない）
- Engine split: research = Python (Vantage CSVs); see/alert = TradingView (Pine); live = Vantage MT5 (manual)。
  Validate on Vantage; TV chart feed ≠ trade feed。データ更新は `../mt5-mcp`（前節）。
