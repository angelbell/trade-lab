# 仕様カード10 — gold15m 単体改善 A/B/C 同時（単体軸で裁定・ブック裁定しない）

## 前提（評価軸の転換）
[[feedback-standalone-focus]]: ブックCAGR/DDで裁定しない。単体運用の軸で:
勝率・PF・N/年・per-trade EV・**中央値ブートDDのCAGR/DD**（単一経路DD禁止・CLAUDE.md#8）・
**資金倍率分布**（[[feedback-low-tf-gamble-up]] N·PF·f＝期間別口座倍率の中央値/分位）・IS/OOS。
gold15mは狭stopで口座制約フリー（[[project-gold-out-of-budget]]）。固定0.01ロット＝サイズ写像しない[[feedback-fixed-lot-no-sizing]]。

## 土台（照合ゲート必須）
gold15m を research/book.py get_book_legs() L94-98 と厳密一致で再構築。`scratchpad/strength_gateslope_generalize.py`
の build_gold15m（照合ゲート3本=book.get_book_legs()['gold15m']と配列一致 PASS済み）を流用。R=book定義(0.3/risk netコスト込み)。
baseline: n≈325 / 46本年 / win24.3% / PF1.64 / meanR+0.517。

## 共通の単体指標（各構成で出す）
n, n/年, 勝率, PF, meanR, totR, IS/OOS。@1%固定リスクで: CAGR, **maxDD=巡回ブロックブートストラップ(1/3/6/12mo,3000回)の
中央値DD**（単一経路でなく）, CAGR/中央値DD。**資金倍率分布**: 1年窓の口座倍率(固定1%)の中央値/p25/p75（N·PF·f の実効）。

## A. 入口: 構造アンカー vs frac0.25 vs market
- frac0.25(現行)・market(frac0)・**構造アンカー(H1レベルへの戻り指値, param無, 台帳で単体最良IS/OOS+0.48/+0.45)**を比較。
  構造アンカーは `scratchpad/pullback_struct.py`（台帳記載）を流用。3者を上の単体指標で並べる。
- 判定: アンカーが 0.25 を単体軸（特に資金倍率分布とIS/OOS均衡）で上回るか。同等なら「param消えるが数字同じ＝簡素化止まり」。

## B. 選別: stop_atr で take/skip（単体はPF/勝率を選べる）
- base leg のトレードを stop_atr(=risk/ATR14[確定足i]) で上位 X% のみ採用（X=100/80/60/40）。
  entries↔確定足i は strength 系スクリプトの手順を流用。固定ベットの取捨（サイズにしない）。
- 各Xで単体指標。**頻度が落ちるので資金倍率分布で「N減 vs PF増」を裁定**（PF上げてもN·f が落ちれば口座倍率は悪化）。
- null: 上位X%の meanR/PF が **ランダム除去null＋巡回ブロックブートストラップ**を超えるか（法則7）。超えなければ選別は運。
- 予想死因: 見送り側も+EV（強度研究で確認済み）＝濃縮止まりで資金倍率むしろ悪化。

## C. RR: 単体最適の再検（資金倍率レンズ）
- base leg(入口frac0.25固定)で RR ∈ {3,4,5,6} を掃引。各RRで単体指標＋資金倍率分布。
- 判定: 丘型プラトーか単発スパイクか。法則9(遠い目標)と頻度不足のトレードオフ。RR4付近が既に最良か、別RRに資金倍率ピークがあるか。
- 予想死因: 法則9どおり遠いほどper-trade良いが年44本で資金曲線ゴツゴツ＝RR4付近が最良で動く余地無し。

## 出力・判定
A/B/C それぞれ表で、baseline(現行=frac0.25/RR4/全採用)を必ず併記して差分を読む。
「単体で現行を上回る構成が在るか」を明示。無ければ「現行が単体でも最良＝改善余地小」と正直に。
scratchpad/gold15m_standalone_improve.py。実行 .venv/bin/python。--smoke は短期間サブセット。
## 注意
- 資金倍率は固定1%で計算するが、これは比較のためのスケールでありサイズ写像の採用ではない（ユーザーは固定ロット）。
- 全構成で照合ゲート(baseがbookと一致)を先に通す。掃引の各セルは base と同じ再構築経路で。
