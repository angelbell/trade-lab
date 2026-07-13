# auto-trade

トレード戦略の研究ラボ。**コストを払っても生き残り、レジームが変わっても壊れないエッジ**を探し、
だめなものは**正直に殺す**ためのリポジトリ。

- **研究** = Python（Vantage の CSV）· **可視化/アラート** = TradingView（Pine）· **執行** = Vantage MT5（手動）
- 検証は必ず**実際に取引しているフィード（Vantage）**で行う。TradingView のチャート・フィードは使わない。
- 進め方・戒め・構造法則は `CLAUDE.md`。検証済みの台帳は `docs/verified_findings.md`（1行索引）＋ `docs/findings/*.md`。

---

## 採用ブック（2026-07-13 時点・6レッグ）

**審判**: トレード解像度のドローダウン × **トレードRの標準偏差の逆数**で重み付け（inv-vol）· **総リスク 3%**

| leg | 銘柄 / TF | 機構 | 入口 | 損切 | 利確 | ゲート | Pine |
|---|---|---|---|---|---|---|---|
| `gold_bo` | XAUUSD+ / 1H | ZigZag(2×ATR) Pattern-B ブレイク | 確定足で H1 抜け → **次足成行** | L2（構造） | **RR3** | 日足 SMA150 ↑ | `gold_1h_swing_breakout_zigzag.pine` |
| `btc_bo_kama` | BTCUSD / 4H | 同上 | 同上 | L2 | **RR2** | 日足 KAMA(14) ↑ | `btc_4h_swing_breakout_zigzag.pine` |
| `btc_pull` | BTCUSD / 4H | EMA20 への押し目（SMA80 トレンド） | 確定足 | 最小 0.5×ATR | **RR3** | 週足終値 ≤ 30週MA×1.10 | `btc_4h_ema_pullback.pine` |
| `gold15m` | XAUUSD+ / 15m | 同ブレイク＋ **ext-cap 8%** | **押し目指値 0.25**（200本で失効） | L2 | **RR4** | 日足 SMA150 ↑ | `gold_15m_swing_breakout_zigzag.pine` |
| **`btc15m_L`** | BTCUSD / 15m | 同ブレイク＋ **PDHソフトサイズ 0.5** | **押し目指値 0.30** | L2 | **RR4.5** | **4時間足** KAMA(14) ↑ | `btc_15m_swing_breakout_zigzag.pine` |
| `btc15m_S` | BTCUSD / 15m | 上の鏡像・**前日安値割れ必須** | **戻り売り指値 0.30** | H2 | **RR4.0** | **日足** KAMA(14) ↓ | `btc_15m_swing_breakdown_zigzag.pine` |

**押し目指値** = ブレイクの終値 `e` に飛び乗らず、`lim = e - frac×(e - 損切)` に指値を置く。
損切と利確は **`e` を基準に固定したまま**なので、実効RRが膨らむ。
押しが来ずに利確へ走り抜けたブレイクは**見送る**（＝取り逃し＝逆選択の税。これも損益に計上済み）。

### 成績（2019-05 → 2026-05、7.0年 · コスト込み）

| | 年間本数 | CAGR | maxDD | **CAGR/DD** |
|---|---|---|---|---|
| **6レッグ・ブック** | **203** | **+63.4%** | **7.66%** | **8.27** |
| 旧3レッグ（gold_bo + btc_bo_kama + btc_pull） | 47 | +29.9% | 9.85% | 3.03 |

**年別（口座%）— 8年中8年プラス**

| 2019 | 2020 | 2021 | **2022** | 2023 | 2024 | 2025 | 2026(途中) |
|---|---|---|---|---|---|---|---|
| +34.9% | +67.2% | +23.8% | **+25.4%** | +44.9% | +74.5% | +62.0% | +19.7% |

2022年は **BTC が −64%** の年。**2025年は BTC が −7.6%** の年（それでも +62.0%）。

### レッグ別

| leg | 重み（1トレードあたり口座%） | 年間本数 | 勝率 | PF | meanR | 最長連敗 |
|---|---|---|---|---|---|---|
| gold_bo | 0.547% | 29 | 41.0% | 1.74 | +0.492 | 7 |
| btc_bo_kama | 0.716% | 8 | **52.6%** | 2.16 | +0.549 | **3** |
| btc_pull | 0.551% | 10 | 47.8% | 2.57 | +0.820 | 4 |
| gold15m | 0.376% | 44 | 25.6% | 1.74 | +0.585 | 11 |
| **btc15m_L** | 0.472% | **100** | 22.8% | 1.86 | +0.445 | **17** |
| btc15m_S | 0.337% | 13 | 32.6% | 2.36 | +0.985 | 7 |

**それぞれの役割**

- **`btc15m_L` がブックの生命線。** 抜くと CAGR/DD が **8.27 → 3.62** に落ちる。頻度（年100本）が統計を効かせている。
- **`btc_bo_kama` は稼がないが、DD を下げる。** 抜いても CAGR/DD は 8.21 とほぼ同じだが、DD は 7.66% → 9.95% に悪化。
  5レッグの総リスクを 2.28% に絞って**同じ DD に揃える**と CAGR は 58.1%（6レッグは 63.4%）＝ **同じドローダウンで +5.3 ポイント高い。
  リスク・ダイヤルでは代替できない、本物の分散。** 機構＝ブックで唯一 勝率>50%・最長連敗3回・R のばらつき最小・
  **ブックの最悪 5% の日に建てている率が 22%**（gold_bo は 55%）＝**そもそも悪い日に市場にいない**。
- **`btc15m_S` は弱気年の中和役。** 2022年は L が −6.6R、S が +16.0R。
- **`btc_pull` は別機構**（押し目＋週足サイクルゲート）なので、ブレイク族と冗長にならない。

---

## 使い方

`python` は PATH にない。**必ず `.venv/bin/python`**。

```bash
# ブックの正典レポート（現役レッグ）
.venv/bin/python research/portfolio_kama.py
.venv/bin/python research/portfolio_alloc.py      # 重み付けスキームの比較

# gold_bo を単体で再現（n=208 / meanR +0.49 が出れば一致）
.venv/bin/python breakout_wave.py --csv data/vantage_xauusd_h1.csv --tf 1h \
    --pattern B --tp-mode rr --rr 3 --trend-ema 80 --fwd 500 \
    --daily-sma 150 --daily-slope-k 10 --start 2018-01-01 --peryear

# btc15m_L を単体で再現（n=759 / meanR +0.59 が出れば一致）
.venv/bin/python breakout_wave.py --csv data/vantage_btcusd_m15.csv --tf 15min \
    --pattern B --tp-mode rr --rr 4.5 --trend-ema 80 --fwd 500 \
    --gate-kama 14 --gate-kama-tf 240min \
    --pullback-frac 0.30 --fill-win 200 --cost 0 --start 2018-10-01 --peryear

# 新しいシグナルは必ず標準ハーネスに通す（PF/N/リスク/TFラダー/先読み禁止を強制）
.venv/bin/python research/edge_harness.py --help

# 採用前の統計監査（Deflated Sharpe + PBO/CSCV + ブートストラップ）
.venv/bin/python research/overfit_audit.py

# MT5 から OHLCV を更新（隣の ../mt5-mcp ブリッジ経由。ブリッジ + MT5 ログイン必須）
bash ../mt5-mcp/scripts/run_backtests.sh
```

### 主なスクリプト

| ファイル | 役割 |
|---|---|
| `breakout_wave.py` | ブレイクアウト本体（gold_bo / btc_bo_kama / gold15m / btc15m_L / btc15m_S の全部） |
| `ema_pullback.py` | EMA 押し目本体（btc_pull） |
| `research/portfolio_kama.py` | 現役レッグの定義（`get_legs()`）＋ ブック集計 |
| `research/portfolio_alloc.py` | 重み付けスキームの比較（**CAGR/DD はトレード解像度**、`inv_vol_TRADE` が正典） |
| `research/edge_harness.py` | 新シグナルの標準評価ハーネス |
| `research/overfit_audit.py` | 過学習リスクの計測（DSR / PBO / ブートストラップ） |
| `research/regime_discriminator.py` | 「どこで効くか」の一次スクリーン |
| `src/data_loader.py` | Vantage CSV ローダー（ブローカー時刻を維持。異常足の自動除去、`GOLD_H1_START`） |

### データの落とし穴

- **gold H1 は 2018年以前が実質日足**（年 250〜300本 vs 以降 5900本）。ブックを組む経路では
  `GOLD_H1_START` で必ず切る（`src/data_loader.py`）。
- **Vantage の時計はブローカー時刻（EET/EEST = UTC+2/+3）。** 外部データ（UTC）と結合するときは
  必ずタイムゾーン変換する。これを怠って過去に判定を3件落とした。
- ターミナル上の gold のシンボル名は **`XAUUSD+`**（`XAUUSD` は存在しない）。

### コスト（実測 2026-07-02、Vantage RAW/ECN）

| 銘柄 | 実際の往復コスト | バックテストの想定 |
|---|---|---|
| gold | $0.15〜0.35 / oz | $0.3（15分足）· 価格の 0.1%（1時間足）= **6〜13倍の保守側** |
| BTC | $10〜25（コミッション 0） | **$15** ≈ 実勢 |
| FX | 0.9 pip | — |

⚠️ **`--cost` は「価格に対する比率」**（既定 0.001 = 0.1%）。15分足レッグの $15 / $0.3 は**絶対額**なので、
CLI では表現できず、ブック側で `R -= cost / risk` として後から引いている（BTC は $3千〜$10万まで動くため、
比率では正しくモデルできない）。上の CLI 例で `--cost 0` としているのはそのため。

---

## 正直に書いておくこと

- **想定実ドローダウンは 11〜15%**（backtest の 7.66% × 1.5〜2）。バックテストの DD をそのまま信じない。
- **`btc15m_L` / `btc15m_S` / `gold15m` は 2026-07-13 に採用したばかり**で、実弾での前進検証はこれから。
- **年203本 = 週に約4回**、しかも 15分足の3本は**指値注文**を伴う。**裁量で目視追尾するのは相当きつい。**
  アラート（TradingView）＋ MT5 での指値発注で回せるが、**15分足の確定を待って発注する規律**が要る。
  取りこぼすとバックテストとずれる。ここは自動化の余地がある。
- **`btc15m_L` は本物の弱気相場（2022年、BTC −64%）では出血する**（PF 0.86）。入口ゲートでは直せないことを確認済み。
  ショート脚が中和しているが、それに依存している。
- ロング/ショートでゲートTF・RR が非対称（L=4時間足/RR4.5、S=日足/RR4.0）。**これは測定結果であって設計思想ではない。**
  ショート脚に 4時間足ゲート・RR4.5 を試した結果はまだ無い。
- 2026-07-13 に**ブックを測る機械のバグを4つ**見つけた（月次DD審判 / 月次σ重み / gold の疎データ漏れ / gold15m の仕様違い）。
  **それ以前のブックの数字は、すべてその上で読んでいた。** 経緯は `docs/findings/s08_audit.md`。
