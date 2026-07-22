# 仕様カード18 — gold イベント・スキャルプ「確認5分の大きさ上位だけ取る」×決済5/10/15/20/30分

## 目的
カード17で gold は follow-through が本物（Spearman +0.28, null 98.7%ile）だが全事象平均はコスト死。
**確認5分の動きが大きいイベントだけ取れば net がコストを抜けるか**を、閾値スイープ＋同条件nullで検定。
[[feedback-cost-after-edge]][[feedback-edge-size-threshold]]。可否ルール（take/skip）を出す。強度サイジングは出さない。

## 流用（車輪の再発明禁止）
- `scratchpad/event_scalp.py` の関数（scalp_metrics・build_scalp_table・null_scalp_table・bootstrap系）と
  `fomc_event_study.py`（price_before・candidate_dates・atr14・M15_START）を import。**新規は「確認サイズでの層別」だけ**。

## 対象
- **gold m5 のみ**（`vantage_xauusd_m5.csv`, 2018-10-01以降）。イベント=`data/ext_fomc_dates.csv`（読み取り専用）。
- BTC は順張り死（カード17）＝今回は対象外。フェード側は別カード。

## 機構（カード17と同一・先読み厳禁）
- t0=リリース、P0=直前確定足終値。**確認窓 w_c=5分固定**。P_entry=close_{t0+5分}。d=sign(P_entry−P0)、d=0スキップ。
- **確認サイズ C = |P_entry − P0| ÷ ATR14(m5, t0直前)**（ATR正規化＝時代/ボラをまたいで比較可能に）。生$も併記。
- 建て=P_entry 成行。**決済 H ∈ {5,10,15,20,30}分**（ユーザー指示: 5だけでなく10/15も見る）。g=d·(P_exit−P_entry)。
- コスト: gold 往復 $0.30（保守 $0.60も）。net = g − cost。

## 層別・スイープ（本体）
1. **確認サイズ C の閾値スイープ**: C の分位で上位集合を作る（take 上位 100/70/50/33/25/20%）。各集合×各Hで
   n / gross中央値($・ATR比) / win% / **net_mean・net_median・P(net>0)・年tot相当**（中央値・p25・p75・std併記
   [[feedback-prob-report-std-median]]）。**閾値を上げるほど net が単調改善して頭打ちになるか（丘型=本物）／一点スパイク（まぐれ）か**を見る。
2. 生$の絶対閾値でも1回（C_atr だけでなく「$Xoz以上動いた日」でも同傾向か）。多重比較は増やさない。

## null（同条件・最重要）
- **同ブローカー時刻・非FOMC平日ランダム**（3000回相当）に**同じ確認サイズ閾値を適用**して、
  「大きく動いたランダム日」の net 分布を作る。real-conditioned が null-conditioned の何%ileか（net_mean/win%）を各閾値×Hで。
  ＝**「大きく動いた日」の中で、FOMC のほうがランダムより follow-through が強いか**を分離（単なる大値幅選択でない証明）。
- 併せて random-drop（全事象から同数ランダム抽出）とも比べ、「大きいのを選ぶ」ことの寄与を切り分ける。

## 検定の締め（本物/まぐれ）
- 丘型プラトー（閾値↑で net 単調改善→頭打ち）か。**巡回ブロック・ブートストラップ(1/3/6/12mo)**で
  best閾値×H の net が0超のCIを持つか（薄いnで会合1-2個に振れないか）。
- IS/OOS（前半/後半・年別）で符号一致か。**閾値×Hのグリッド探索＝多重比較**をBonferroni的に注記し、
  best セルの数字は割り引く。

## 報告
- gold: 閾値×H の net 表（中央値/std/n/P(net>0)/null%ile）。丘型かスパイクか。best セルの可否ルール
  （「確認5分が Catr≥X（≈$Y/oz）動いたら d方向に建て、H分保有、コスト後 +Z/oz・年N本」）またはコスト死の確定。
- 病巣（届かない/丘型でない/nullと同等/時代不一致）を特定。

## 死に方（予想）
- 閾値↑で net は上がるが、それは「大きく動いた日を選んだ」だけで null-conditioned も同じだけ上がる（＝FOMC寄与ゼロ）。
- 丘型にならず一点スパイク（best セルだけ+、隣は−）＝多重比較の運。
- n=60→上位1/3で20・上位1/4で15＝薄い。ブロックCIで0またぎが本命。
- 生き残るとしても「年数本・net +数十cent/oz」＝口座寄与が薄く自動化前提（[[feedback-edge-size-threshold]]）。

scratchpad/event_scalp_cond.py。実行 .venv/bin/python。--smoke は直近2年。数字は必ずローカル実行して返す。
