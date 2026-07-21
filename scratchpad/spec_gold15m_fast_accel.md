# 仕様カード4 — gold15m に "速い" トレンド加速を当てる（加速仮説の適速リトライ）

## 背景
btc15m_L で kama_slope(4H KAMA(14) の傾きの急さ) が弱いが本物の強度勾配（Spearman +0.088・ブロックCI全0超・
時代隔離通過）。だが横展開（カード3）では gold15m に「自分のゲート＝日足SMA150(150本平均=鈍い)」の傾きを当てて消えた。
病巣＝指標の速さ。∴ 今回は btc15m_L で効いた信号を**そのまま移植**し、gold にも "速い" 加速を当てて、
「効いたのは 4Hトレンド加速というプリミティブ（銘柄横断）か、btc15m_L 脚固有か」を切り分ける。

## 対象レッグ（gold15m 1本）
research/book.py get_book_legs() L94-98 と厳密一致で再構築。**土台は `scratchpad/strength_gateslope_generalize.py`
の gold15m 経路**（entries直呼び再構築＋トレード↔確定足i対応、照合ゲート3本=book.get_book_legs()['gold15m']と
配列一致 PASS済み）をそのまま import/複製して使う（再発明禁止）。確定足i配列は取得済み。

## 強度候補（3つ・単体で・すべて確定足iで評価・no-lookahead）
1. h4_kama_slope（本命・直接移植）: gold の 4H(240min) KAMA(14) の1本あたり傾き (KAMA[i]-KAMA[i-1])/KAMA[i-1]。
   btc15m_L の勝ち信号を銘柄だけ替えて完全移植。HTFは既存 gate_kama_tf="240min" と同じ足付け・shift/confirm-later。
   ※ここが本題: これが gold15m で効けばプリミティブ横断、消えれば btc15m_L 脚固有が確定。
2. atr_expansion（別系統の速い加速）: 確定足iでの ATR(14)[i] / ATR(14)[i-20]（直近ボラ / 20本前ボラ）。
   ボラ拡大＝トレンド加速の代理。15m足のATRで計算（確定足までの過去のみ）。
3. h1_ema_slope（中間速度）: gold の 1H EMA(20) の1本あたり傾き。日足SMA150 と 4H の中間の速さ。
   日足SMA150(鈍・カード3で死)と 4H(速)の間で、効きが速さに単調かを見るための中間点。

## 測り方（カード2/3と同一）
各候補: 5分位で n/win%/PF/meanR/totR、単調性、Spearman(変数vs実現R)、
巡回ブロック・ブートストラップ(1/3/6/12か月,各1000回)95%CI、ランダム除去null percentile、
時代ベータ隔離(年別Q5-Q1 meanR、probe_kama_slope_era.py 同型)。報告はPF・N・meanR併記、金額換算不要。
scratchpad/strength_gold15m_fastaccel.py。実行 .venv/bin/python。

## 判定
- h4_kama_slope が gold15m でも Spearman正・ブロックCI下限0超・時代分散（btc15m_L と同傾向）
  → 「4Hトレンド加速」は横断の強度プリミティブ（銘柄別に"4H"の速さで測ればよい）＝カード3の否定を上書き。
- h4_kama_slope フラット/反転 → kama_slope は btc15m_L 脚固有が確定（法則1）、加速仮説は銘柄横断しない。
  その場合 atr_expansion / h1_ema_slope のどちらかが効けば「gold は別の速い加速指標が要る」を記録。

## 死に方（予想）
- h4_kama_slope: gold は BTC よりトレンドが滑らかで 4H 加速の情報量が薄い可能性（消えても不思議でない）。
- atr_expansion: ボラ拡大は"動く量"の予測子で方向でない（vol_z と同型）＝meanR フラットの疑い（law: 出来高と同じ罠）。
- h1_ema_slope: 中間速度でどっちつかず＝弱い勾配 or ノイズ。速さ単調性が出るかだけ見る。
