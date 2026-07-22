# 仕様カード — btc15m_L のトレードを「強度スコア」で層別し、EV単調性を測る

## 目的（なぜ）
ユーザーは固定ロットではなく **signal強度を見て裁量でサイズを変える**（強→大、弱→小）。
∴ 変数の価値 = 「強度が上がるほど per-trade EV(meanR/PF/totR) が単調に上がる」勾配情報。
フラットなら強度スコアとして無価値（＝ただの取捨フィルターに降格）。自動サイズ写像は組まない。

## 対象レッグ（1本だけ）
btc15m_L（研究仕様は research/book.py get_book_legs L100-105 と厳密一致させる）:
- d15 = resample(load_mt5_csv data/vantage_btcusd_m15.csv .loc["2018-10-01":], "15min")
- tL = run(d15, BASE + gate_kama=14, gate_kama_tf="240min", pullback_frac=0.3, rr=4.5, fill_win=200)
- WL,_ = pdh_soft(d15, tL);  netR = (tL.R - 15.0/tL.risk) * WL   （PDHソフト0.5・ネットコスト$15込み）
- **照合ゲート（必須・先に通す）**: 自作の tL 由来 netR 系列が book.get_book_legs()["btc15m_L"] と
  時刻・値で配列一致することを確認してから層別に進む。一致しなければ層別の数字は信用しない。

## 強度候補（3つ・単体で）
1. base_bars: tL の既存列（i - i_origin = 押し目指値が埋まるまでの待ち本数）。深い/浅い押しの代理。
2. risk_frac: tL.risk / tL.e_px（損切り幅の価格相対＝breakout の幅・タイトさ）。
3. vol_z: 確定足(breakout バー i)の 15m tick_volume の z-score。
   - **no-lookahead 厳守**: 約定足(fill bar, tL.time)ではなく breakout 確定足 i の出来高を使う。
     押し目指値レッグは fill が i より最大200本後なので、fill足の出来高は決定時に未知＝先読み。
   - i の復元: detect 側の entries=(i,e,stop,tgt,i_origin) を同じ引数で取得し、トレードと対応付ける。
     対応が難しければ、確定足時刻 = tL.time から base_bars 本ぶん遡った 15m バー、で近似してよいが
     **その近似が i と一致するか数本サンプルで目視確認し、ズレるなら entries 経由に切替**。
   - z-score = (vol[i] - rolling_mean) / rolling_std、窓は確定足までの過去のみ（例: 直近96本=1日）。
     tick_volume はローダで "volume" にリネームされている点に注意。

## 測り方（各候補ごと）
- トレードを候補変数で 5 分位（qcut, X.rank(method="first") で同値処理）に切る。
- 各分位について報告: n, win%, PF, meanR, totR。**分位1→5 で meanR/PF が単調に上がるか**を明示。
- 単調性の検定:
  - Spearman 相関（変数 vs 実現R）と、その巡回ブロック・ブートストラップ（1/3/6/12か月, 各1000回）95%CI。
  - ランダム除去null（必要条件）: 上位分位と同数をランダム抽出した時の meanR 分布に対する percentile。
- **強度スコアとしての結論**: 単調で、ブロックを伸ばしても勾配が残る → 裁量サイジングに根拠あり。
  上位/下位で meanR が反転 or フラット、ブロック伸長でP→50% → 経路当てはめ＝無価値。

## 報告フォーマット
候補3つ × 分位表（n/win/PF/meanR/totR）＋ Spearman(CI)＋null percentile。
0.01ロット換算の金額は不要（強度勾配の有無だけが論点）。CLAUDE.md の PF・N・meanR 併記規約に従う。

## 死に方（予想）
- base_bars: 押し目が深い(待ち長い)ほど良く見えるが、約定足バグ族（間隔が縮むと同足タダ乗り）の
  残り香が出る可能性 → fill_bar_stopped 済みの現エンジンなら消えているはず。要確認。
- risk_frac: タイトな損切りほど RR 実現しやすく meanR 上がるが、これは RR 固定の力学であって強度でない疑い。
- vol_z: STEP1 では「動く量」の予測子だった。動く量↑は勝ち負け両方を増やすので meanR は上がらない
  可能性が高い（方向情報ではない）。その場合「強度」失格を明記。
