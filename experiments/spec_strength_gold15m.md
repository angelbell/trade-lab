# 仕様カード7 — 合成強度旗(stop_atr+atr_pctile)を gold15m へ移植（同型レッグ検定）

## 背景
btc15m_L で合成旗（stop_atr=risk/ATR＋atr_pctile=ATR trailing500分位, 等重みランク平均）のトップ20%が
PF2.35/+0.94R vs 残りPF1.36/+0.33R(ギャップ+0.62R・ブロックCI全0超)。他BTCレッグ(btc15m_S/btc_bo_kama/btc_pull)へは
移植せず＝型が違った（ショート/4H成行/別エンジン）。gold15m は btc15m_L と**同型**（15m・押し目指値・ロング・
確定足breakout）なので効く見込みが相対的に高い。gold15m は狭stopで口座制約フリー（[[project-gold-out-of-budget]]）。

## 対象（gold15m 1本）
research/book.py get_book_legs() L94-98 と厳密一致で再構築。土台 `scratchpad/strength_gateslope_generalize.py` の
gold15m 経路（entries直呼び再構築＋トレード↔確定足i対応、照合ゲート3本=book.get_book_legs()['gold15m']と配列一致
PASS済み）を import 流用（再発明禁止）。n≈325（2019-05〜2026-05, 約44本/年）。R は gold15m の book 定義
（0.3/risk コスト込み netR）。

## 強度候補（確定足iで・no-lookahead・gold 15m足のATRで）
- stop_atr = tL.risk / ATR(14)[i]（ATRは g15=15m gold）。
- atr_pctile = ATR(14)[i] の trailing 500本 percentile。
- combo2 = (rank_pct(stop_atr)+rank_pct(atr_pctile))/2。
- 参考 combo3 = 上記＋日足SMA150の傾き rank（gold15mのゲート指標。カード3では単体フラットだったが合成での寄与を見る）。

## 測り方（btc15m_L と同一・n=325でトップ20%が65本＝5分位も可）
- 5分位表 n/win%/PF/meanR/totR、単調性、Spearman(combo2 vs R)。
- **トップ20% vs 残り**の meanR ギャップ＋巡回ブロックbootstrap(1/3/6/12mo,3000回)95%CI＋年別 top−rest。
- combo3 も同様に併記。R は gold15m book netR。報告は PF・N・meanR 併記。

## 判定
- トップ20%旗が PF/ギャップとも btc15m_L と同傾向・ブロックCI下限0超・年別符号が揃う
  → 合成旗は「15m・押し目指値・ロング・breakout」型で銘柄横断＝btc15m_L と gold15m の両方で使える強度旗（法則1の
  "同型内では転移"の実例）。gold15m は口座制約フリーなので実運用にも乗る。
- フラット/反転 → 同型でも銘柄固有＝btc15m_L 専用に留まる。その場合 stop_atr/atr_pctile のどちらが崩れたかを
  分解し、gold特有の性格（金属＝トレンドは滑らか・ボラ構造がBTCと違う）で機構を説明。

## 死に方（予想）
- gold は BTC よりトレンドが滑らか＝ATR構造の情報量が薄く atr_pctile が効きにくい可能性。
- gold15m は押し目 frac 0.25(btc15m_L 0.30)＝実効riskの縮み方が違い stop_atr の意味がわずかにずれる可能性。
- ext-cap 8% で既に伸びすぎブレイクを除外済み＝stop_atr の上側情報が一部食われている可能性。
