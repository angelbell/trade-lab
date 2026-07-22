# 仕様カード6 — 合成強度旗(stop_atr+atr_pctile)を他BTCレッグへ移植（横断プリミティブ検定）

## 背景
btc15m_L で「エントリー質＝実効リスク」2軸 stop_atr(=risk/ATR, 損切りがATR対比で広い＝有意な構造)＋
atr_pctile(ATRのtrailing500分位, 高ボラ=非チョップ) が互いに直交して生存、合成ランク平均のトップ20%旗で
PF2.35/+0.94R vs 残りPF1.36/+0.33R(ギャップ+0.62R・ブロックCI全0超)。kama_slopeは合成のトップ並べ替え補助。
問い: この2軸(＋任意でkama_slope)がBTCの他レッグでも効くか＝BTC横断の強度旗か、btc15m_L固有か。
※BTCは口座制約フリー([[project-gold-out-of-budget]])なので全レッグ運用対象。

## 対象3レッグ（research/book.py get_book_legs と厳密一致・各レッグ照合ゲート必須）
1. btc15m_S (L107-112, 15m ショート鏡像): 土台 `scratchpad/strength_gateslope_generalize.py` の btc15m_S 経路
   （照合ゲートPASS済み）を流用。ATRは d15(15m)。n≈100。
2. btc_bo_kama (L86-89, 4H 成行ブレイク): b4=resample(btc h1,"4h")→run(rr2,fwd300)→kama_gate_btc。
   pullback_frac=0＝成行なのでエントリー足=確定足i（押し目待ち無し＝btc15m系より対応付けが単純）。ATRは b4(4h)。n≈70。
3. btc_pull (L90-92, 4H EMA押し目・別エンジン walk_ema): run_pb(b4,"long")→cycle_gate_pull。
   walk_ema は signal足約定＝エントリー足既知。ATRは b4(4h)。**engine が違うので entries 対応は walk_ema の
   出力(time)で突き合わせる**。照合ゲート=自作R系列が book.get_book_legs()['btc_pull'] と時刻・値一致。

## 強度候補（確定足/エントリー足で・no-lookahead・レッグ自身のTFで）
- stop_atr = leg.risk / ATR(14)[entry]（ATRはそのレッグのTF＝btc15m_Sは15m, 4H系は4h）。
- atr_pctile = ATR(14)[entry] の trailing 500本 percentile（同TF）。
- 合成 = (rank_pct(stop_atr)+rank_pct(atr_pctile))/2。参考でkama_slope込み3軸版も出す(そのレッグのゲートTF傾き)。

## 測り方（母数で解像度を変える・過剰有意化を避ける）
- btc15m_S(n≈100): トップ20% vs 残り の meanR ギャップ＋巡回ブロックbootstrap(1/3/6/12mo,3000回)95%CI＋
  年別Q(トップ)−(ボトム)。5分位表も出すが薄いので参考。
- btc_bo_kama/btc_pull(n≈70): **トップ1/3 vs 残り2/3** の meanR ギャップ＋ブロックbootstrap＋年別。
  5分位はn薄すぎ＝作らない。n/年を明記し「薄い」と正直に。
- 各レッグ: n/win%/PF/meanR、合成スコアのSpearman(対R)も併記。R は各レッグの book 定義(netコスト込み)。

## 判定
- 2軸合成のトップ旗が btc15m_S で PF/ギャップとも btc15m_L と同傾向・ブロックCI0超 → 少なくとも15m系で横断。
- btc_bo_kama/btc_pull でも同符号（薄くても年別で符号が揃う）→ BTC横断プリミティブ候補。
- フラット/反転 → btc15m_L(または15m系)固有。どのレッグで効きどこで消えるかを機構で説明（TF依存か・ショート/ロング
  非対称[[構造法則11]]か・エンジン依存か）。

## 死に方（予想）
- btc15m_S: ショートはドリフト逆行で薄く(年12本)、高ボラ(atr_pctile高)は下落局面＝ショート順風で効く可能性、
  or 高ボラ=底で反発食らう可能性。符号は測定任せ。
- btc_bo_kama/btc_pull: 4H・年約9本でブロックbootstrapのCIが常に0をまたぐ＝「測定不能」に終わる公算大。
  その場合は正直に「薄くて判定不能」と書き、無理に有意化しない。
- atr_pctile が全レッグで効くなら「高ボラ局面のブレイク/押し目が良い」＝銘柄・TF非依存のBTC地合い変数の可能性。
