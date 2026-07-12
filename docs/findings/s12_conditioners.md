## 12. 入口の条件付け（WHEN-conditioners on breakout）
- ❌ **ボラ収縮ブレイク（squeeze breakout＝BB幅が直近100本の下位1/4＝収縮明けにDonchianブレイク）= NON-ADOPTED, n間引き**（2026-07-01, edge_harness初通し）。
  per-trade meanR は上がる（gold 1d +0.21→+0.32, BTC 1d +0.21→+0.47）が、**CAGR/DDの random-drop null を超えない**：gold 4h 27%/1d 55%, BTC 4h **11%**/1d 89%（4hは害、1dも閾値90未満・n5本/年）。
  ＝「フィルタはmeanR nullでなくCAGR/DD nullを超えねば」（stop幅/ERと同じ罠）。IS偏重(gold4h IS+0.36/OOS-0.29)も警告。**条件付け検出器は≒0 liftの法則を再確認。**
  *教訓:* 新フィルタは必ず `random_drop_null` でCAGR/DDを見る。meanR上昇は n減のばらつき artifact のことが多い。
  - **v2（箱ブレイク版＝収縮した箱の上下限を確定足ブレイク＋RRをデータから, 2026-07-01）も素ブレイクに全TF/全RRで負け**（gold1d/RR4 素+3.44 vs sq+0.73、BTC4h/RR4 素+4.05 vs sq+0.28、勝セル0）。
    **機構＝決定的:** スクイーズ（収縮）要求は「既に動いてるトレンド継続ブレイク」を除外する＝エッジの本体を削る。収縮明けは"溜め"でなく"prior-trend無しのレンジ"＝騙し/チョップ。**retest-kill/「強いブレイクは戻らない」と同じ法則。スクイーズ・ブレイクはトレンド資産では構造的に不適**（実装でなく前提がエッジと衝突）。棲むならレンジ/平均回帰の地合い・銘柄、or "騙しをフェード"側（別の入口族・未検証）。

