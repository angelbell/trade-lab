# 仕様カード19 — CPI/NFPをFREDで取得し、news事象を合算してgoldスキャルプの時代安定性を検定

## 背景（役割）
反証は道具・目的は「効く方法を育てる」。カード18でgoldの「確認5分が大きい→d方向にH分保有」がコスト後+・丘型・
follow-through本物（null98-100%ile）と判明。唯一の弱点は**サンプル薄（年8会合でIS/OOS振れ・上位セルはnull不成立）**。
∴ **news事象を増やして時代安定性を出す**。CPI+NFPで年8→30本規模に。

## 第一部 — CPI/NFP発表日カレンダーをFREDで取得
- FRED release/dates API（キーは環境変数 `FRED_API_KEY` で渡す。ソースにも文書にも書かない）。
  NFP=Employment Situation `release_id=50`、CPI=Consumer Price Index `release_id=10`。
  `https://api.stlouisfed.org/fred/release/dates?release_id=..&api_key=..&file_type=json&realtime_start=2018-01-01&realtime_end=2026-12-31&include_release_dates_with_no_data=false&sort_order=asc&limit=1000`
- **返り値は年12本より数件多い**（年次改定・併載系列で重複日）。**主系列の月次発表だけ残す**:
  NFP=毎月ほぼ第1金曜1本／CPI=毎月中旬平日1本。月内の余分な日（改定等）は落とし、月次1本に正規化。落とした行と理由を報告。
- 発表時刻は**固定 08:30 ET**（FREDは日付のみ→各日付に08:30 ETを当てる）。**tz変換はFOMC検算済み手法を流用**
  （`fomc_event_study.py`/`context_time_fomc.py` の ET→UTC→Europe/Riga を 08:30 に置換。America/New_York・DST可変・固定オフセット禁止）。
- 出力: `data/ext_nfp_dates.csv`・`data/ext_cpi_dates.csv`、列 `kind,dt_utc,dt_broker`（ext_fomc_dates.csv と同一スキーマ）。
- **検算必須**: (a)月次カデンス・NFP第1金曜/CPI中旬 (b)既知アンカー NFP 2020-05-08(4月分)・CPI 2022-07-13(9.1%回) で日付一致（ズレたら停止報告） (c)各行 dt_utc↔dt_broker 差が+2h/+3h(Riga夏冬)。年別件数・充足率を報告。

## 第二部 — 合算してgoldスキャルプ再走（カード18の機構をそのまま）
- `scratchpad/event_scalp_cond.py` を **--events で台帳差し替え**（無ければ複数台帳の合算に対応する最小改修。ウォーカー等は無改変）。
- 事象セット4通り: (1)CPI単独 (2)NFP単独 (3)FOMC+CPI+NFP合算(union・重複日dedupe) (4)参考FOMC単独(カード18再掲=tie-back)。
- 各セットで **確認サイズC_atr スイープ(上位100/70/50/33/25%) × 決済H∈{5,10,15}分**、コスト gold往復$0.30(保守$0.60):
  n / gross中央値($・ATR比) / win% / **net_mean・net_median・P(net>0)・年tot相当**（中央値/p25/p75/std）。
- **同条件null**（同ブローカー時刻・非該当平日ランダム3000回に同じC閾値を適用）で real が何%ileか。
  ＝「大きく動いた日の中で news がランダムより強いか」を各セットで。
- **時代安定性（本命の課題）**: IS/OOS（前半/後半）と年別を各セットで。**合算(3)で n が増えた分、上位50%(C≥2ATR付近)×H=5 が
  IS/OOSとも+か・年別の符号一貫か・巡回ブロック(1/3/6/12mo)ブートストラップで net の0超CIを持つか**。ここが「育った」かの判定。
- **事象種別の異質性**: NFP vs CPI vs FOMC で follow-through/最適C閾値/最適Hが違うか（種別で継続の効き方が違えば種別ゲートの芽）。
- 参考: BTC でも合算(3)だけ1回（カード17でBTC順張りは死＝確認だけ、深追いしない）。

## 報告
- カレンダー2ファイルの検算結果。事象セット×スイープの net 表。合算での可否ルール
  （「確認5分が C≥X ATR 動いたら d方向に建てH分保有＝コスト後 +Z/oz・年N本、IS/OOS±・null%ile」）。
- 「育ったか」の平文判定: サンプルが増えて時代安定したか／まだ薄いか／種別で効きが違うか。多重比較(セット×frac×H)をBonferroni注記。

## 死に方（予想）
- CPI/NFP時間帯(15:30 Riga)はFOMC時間帯(20-21時)と流動性が違い、goldの続伸がFOMCほど出ない可能性（時間帯効果）。
- 合算でnが増えてもIS/OOSが揃わない＝時代でなく個別大事象依存だった、が判明する可能性。
- 種別で符号が割れる（例CPIは続伸・NFPは反転）＝合算は薄まる。その場合は効く種別だけ残す。

scratchpad/news_scalp_combined.py（またはevent_scalp_cond.pyの--events拡張）。実行 .venv/bin/python。--smoke対応。
数字は必ずローカル実行して返す。ext_fomc_dates.csv は読み取り専用（触らない）。
