# 仕様カード12 — nas100 指数プルバック（装置由来の新単体レッグ・第一歩）

## 背景
instrument_character 装置が nas100 を「トレンド・ロング/グラインド型（押し目）」と判定
（年率ドリフト+18.1%・t=2.6・VR20=0.78・集中度21%）。保留リード [[project-index-pullback-followup]]
「指数は buy-the-dip、追うなら ema_pullback（breakoutでない）を Vantage H1 で」と一致。
gold_bo レシピ（breakout）の指数横展開は全滅済み（docs/verified_findings.md「レシピの横展開・全滅」）＝
**道具違い（breakout→pullback）が唯一の残存理由**。単体軸で評価・ブック裁定なし [[feedback-standalone-focus]]。

## データ
`data/vantage_nas100.r_h1.csv`（2016→・h1のみ）。TFラダーは resample: **1h/2h/4h/8h/1d**
[[feedback-default-tf-set]]（15m/5mはデータ無し＝正直にそう書く）。

## 第一歩A — 可否確認（口座制約・ユーザーの stop$ 問題）
- ema_pullback の全シグナルで stop距離（指数ポイント）の分布: 中央値・p25/p75/p90・年別中央値、TF別。
- **契約仕様はCSVから不明**なので、$換算は「0.01ロットの $/ポイント = v」をパラメタに
  v ∈ {0.01, 0.1, 1.0} の表で提示（Vantage の指数CFDの典型仕様が不明な旨を明記。最終判定はユーザーのMT5確認）。
- ついでに現在価格帯（~2.5万pt級）も出す（notionalの規模感）。

## 第一歩B — 素の測定（all-signals base first・falsify checklist 1）
- `ema_pullback.py` を流用（自前ウォーカー実装禁止・CLAUDE.mdの規約）。long のみ（装置: ドリフト順方向）。
  パラメタは btc_pull の既定（PB プリセット）を出発点に、**チューニングしない**（素の全シグナルが先）。
- TFラダー各TFで: n, n/年, win%, PF, meanR, totR, IS/OOS, --peryear（年別）。
- **ベータnull（最重要の反証）**: 2016→はほぼ一本調子の強気＝ロング必勝時代。同レジーム・同数ランダム建て
  （同保有期間）の null 分布に対する percentile を必ず出す。**素の PF>1 はベータで出るので、null超えだけが情報**。
- コスト: 指数のスプレッドは不明→ 0 / 2pt / 5pt の3段で感度（実測は後段）。コストは入口に混ぜない
  [[feedback-cost-after-edge]]（素の率→偶然性→コストの順）。

## 判定
- ベータnullを超えるTFが在る → 次段（ゲート・押し目深さ・cost実測）へ。
- 全TFでnull以下 → 「指数プルバックもベータの上澄み＝装置のタグは"性格"であって"エッジ"でない」と記録して閉じる
  （タグの読み方の教訓として価値あり）。
- stop$ が v=現実値で口座に対し過大なら、エッジ有無に関わらず「取れない」でクローズ（gold4Hと同じ棚）。

## 死に方（予想）
- ベータnull不通過が本命（2016→の指数ロングは何をやってもPF>1に見える）。
- 通過しても2016→8.5年＝1時代サンプルで時代分散が測れない（正直にそう書く）。
- us2000(2020→)は更に短いので今回は nas100 のみ（横展開は生き残ってから）。
scratchpad/nas100_pullback.py。実行 .venv/bin/python。--smoke は直近2年。
