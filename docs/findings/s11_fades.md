## 11. 反転/フェード探索（mean-reversion）
- ❌ **単体トリガーのフェード（RSI70/80・BB2σ・extMA2ATR・連続足）＝H1で無エッジ**。出口RR1でも MA20回帰でも、USDJPY/gold/BTCの全TFで PF<1中心。**行き過ぎは反転でなく継続（モメンタム）になる**。`reversal_search.py` `reversal_tf_ladder.py`
- 🟡→❌寄り **高TF（8h/1d）深乖離 LONG フェード（BB-L/extMA-L, 出口=MA20回帰）＝ベータnull(anyDip)を突破**: USDJPY 8h PF1.22(n1111,~70/yr)/1d 1.61、gold 8h extMA 1.30(n782)、BTC 8h BB 1.10。anyDip≒1.0を行き過ぎ要求で1.1-1.6へ＝深乖離ほど戻る＝実在の反転成分。`reversal_beta_null.py`
  - **だが致命的留保:** (1) **ロング限定**（ショート・フェードは全TFでPF<1＝上昇相場の逆風）＝反転はベータの上に乗る。(2) **年別で一era集中**：USDJPY extMA-L 8hは2012-14(アベノミクス)でPF1.8-5.0、**2022-24(現在の円安局面)はPF0.83-0.95＝死**。falsify#5(一era beta)。(3) **USDJPY163円=40年ぶり極値=レジーム転換リスク最大、直近3年効かず＝前向きdeploy根拠ゼロ**。集計PFは2012-14の遺産。
  - 🔑 *教訓:* ベータnull(anyDip)突破でも**年別era集中＋現regime健在性**を必ず見る。バックテストはレジーム転換を見られない（人/macroの知見が要る）。

