# 仕様カード16 — CPI/NFP リリース日カレンダーの取得（決定論カノン・イベントスタディ流用用）

## 目的
仕様カード15（event_kinetics.py）を CPI/NFP に流用するためのイベント台帳を作る。**取得のみ・分析しない**。
FOMC を federalreserve.gov から取ったのと同じ要領で、CPI/NFP を BLS 公式から決定論的に取る（Wayback不使用）。

## 対象・出典（公式カノン）
- **NFP = Employment Situation（雇用統計・非農業部門雇用者数）**。発表 **08:30 ET**。
- **CPI = Consumer Price Index（消費者物価指数）**。発表 **08:30 ET**。
- 出典: BLS 年次リリーススケジュール `https://www.bls.gov/schedule/news_release/YYYY_sched.htm`
  （YYYY=2018..2026。全リリースが日付付きで載る＝決定論）。または各系列のスケジュールページ
  `.../schedule/news_release/empsit.htm`・`.../cpi.htm`（当年＋将来）＋過去年アーカイブ。
  取得できない年は正直にそう書き、取れた範囲で作る。
- 期間: **2018-01 〜 2026 現在**（FOMC台帳と揃える）。

## tz変換（先読み罠・FOMCと同一手法を流用）
- 08:30 ET は **America/New_York のローカル時刻**（DSTでEDT=UTC-4 / EST=UTC-5 が変わる）。**固定オフセット禁止**。
- `scratchpad/fomc_event_study.py` / `context_time_fomc.py` にある **FOMCで検算済みのtz変換メソッドをそのまま流用**
  （14:00ET→UTC→Europe/Riga を 08:30ET に置換）。ブローカー時刻 = Europe/Riga（EEST=UTC+3 / EET=UTC+2）。

## 出力（ext_fomc_dates.csv と同一スキーマ）
- `data/ext_nfp_dates.csv` … 列 `kind,dt_utc,dt_broker`（kind=NFP、dt_broker=発表時刻のブローカー時刻）。
- `data/ext_cpi_dates.csv` … 同上（kind=CPI）。
- 1イベント1行。時刻は 08:30 ET を各日付の実DSTで変換した値。

## 検算（正確性最優先・必須）
- **内部整合**: NFP はほぼ「毎月第1金曜」・CPI は「毎月中旬の平日」。月次1件のカデンスから外れる行・重複・欠落を洗い、
  理由（政府閉鎖による遅延等）が説明できるものだけ残す。年×12件に対する充足率を報告。
- **既知アンカーとの突き合わせ**: 広く知られた数件（例 NFP 2020-05-08＝4月分・雇用大幅減の回、
  CPI 2022-07-13＝9.1%の回 等）で日付一致を確認。ズレたら停止して報告。
- **dt_utc↔dt_broker** の差が各行で +2h か +3h（Rigaの夏冬）になっているかを確認。

## 報告
- 各ファイルの行数・期間・年別件数・充足率。既知アンカー突き合わせ結果。取れなかった年/欠落と理由。
- **分析・バックテストはしない**（それは仕様カード15の担当）。取得と検算だけ。
実行 `.venv/bin/python`。取得したデータは必ずローカルで検算して数字を返す。
