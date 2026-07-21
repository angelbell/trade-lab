"""news_scalp_combined.py -- spec_news_scalp_combined.md (仕様カード19)

カード18で gold の「確認5分が大きい -> d方向にH分保有」がコスト後+・丘型・follow-through本物
(null 98-100%ile) と判明。唯一の弱点はサンプル薄(FOMC=年8会合)。CPI/NFPを足して年30本規模に
しても「育つ」(=時代安定)かを検定する。

第一部: FRED release/dates API から NFP(release_id=50)/CPI(release_id=10) の発表日カレンダーを
取得し、月次主系列だけに正規化して data/ext_nfp_dates.csv・data/ext_cpi_dates.csv に書き出す
(ext_fomc_dates.csv と同一スキーマ: kind,dt_utc,dt_broker)。tz変換は
scratchpad/context_time_fomc.py の et14_to_utc_broker と同じ手法(ET固定時刻->tz_localize->
tz_convert(UTC/Europe-Riga))を 08:30 ET に置き換えて流用する(手法自体は同スクリプトの
validate_tz_method で既存48行のFOMC日付に対して再検算し、完全一致することを確認する)。

第二部: scratchpad/event_scalp_cond.py / event_scalp.py の関数を無改変でimportし、
事象セット(1)CPI単独 (2)NFP単独 (3)FOMC+CPI+NFP合算 (4)FOMC単独(tie-back) について
確認サイズ(C_atr)スイープ x 決済H{5,10,15}分・同条件null・IS/OOS・年別・巡回ブロック
ブートストラップを走らせる。自前のイベント抽出・価格取得・出口計算・ウォーカーは一切書かない
(scalp_metrics/build_scalp_table/null_scalp_table/is_oos_table/annual_table/sweep_table/
threshold_subset/block_bootstrap_ci はすべて import)。

実行:
  .venv/bin/python scratchpad/news_scalp_combined.py --smoke 2>&1 | tee scratchpad/out_news_scalp_combined_smoke.txt
  .venv/bin/python scratchpad/news_scalp_combined.py 2>&1 | tee scratchpad/out_news_scalp_combined.txt
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.data_loader import load_mt5_csv  # noqa: E402
from event_scalp import (  # noqa: E402
    build_scalp_table, null_scalp_table, is_oos_table, annual_table, follow_through_table,
    GOLD_M5_START, BTC_M5_START, COST_ROUNDTRIP, SEED, NULL_DRAWS_TARGET,
)
from event_scalp_cond import (  # noqa: E402
    threshold_subset, sweep_table, block_bootstrap_ci, B_BOOT, BLOCK_MONTHS,
)
import context_time_fomc as ctf  # noqa: E402  -- et14_to_utc_broker / validate_tz_method (tz手法の検算済み再利用元)

# 🔑 鍵はソースに書かない（公開リポジトリなので履歴に残ると取り消せない）。
#    使うときは環境変数で渡す:  FRED_API_KEY=xxxx .venv/bin/python scratchpad/news_scalp_combined.py
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
REALTIME_START = "2018-01-01"
REALTIME_END = "2026-12-31"

DATA_FOMC = f"{ROOT}/data/ext_fomc_dates.csv"
DATA_NFP = f"{ROOT}/data/ext_nfp_dates.csv"
DATA_CPI = f"{ROOT}/data/ext_cpi_dates.csv"

W_C = 5                                        # 確認窓5分固定(カード17/18と同一)
HSET = [5, 10, 15]                             # カード19指定
FRACS = [1.00, 0.70, 0.50, 0.33, 0.25]         # カード19指定(上位100/70/50/33/25%)
COST_BASE = COST_ROUNDTRIP["GOLD"]["base"]     # $0.30/oz
COST_ALT = COST_ROUNDTRIP["GOLD"]["alt"][0]    # $0.60/oz (保守)
PRIMARY_FRAC = 0.50                            # 時代安定性の本命セル(C>=2ATR付近)
PRIMARY_H = 5

# ============================================================================
# 第一部: CPI/NFPカレンダー(FRED)
# ============================================================================

ANCHOR_NFP = "2020-05-08"   # 2020年4月分, 事前確認済みアンカー
ANCHOR_CPI = "2022-07-13"   # 2022年6月分(9.1%回), 事前確認済みアンカー


def fetch_release_dates(release_id):
    url = ("https://api.stlouisfed.org/fred/release/dates"
           f"?release_id={release_id}&api_key={FRED_API_KEY}&file_type=json"
           f"&realtime_start={REALTIME_START}&realtime_end={REALTIME_END}"
           "&include_release_dates_with_no_data=false&sort_order=asc&limit=1000")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    js = resp.json()
    dates = pd.to_datetime(sorted(x["date"] for x in js["release_dates"]))
    return dates, js["count"]


def dedupe_nfp(dates):
    """NFP(Employment Situation)は必ず金曜発表。月内に複数日があれば金曜を残し、それ以外
    (改定/他系列の巻き添え)を落とす。金曜が0/2以上ならその月最も早い日を残す(想定外パターンの
    フォールバック、実データでは発生しない)。"""
    df = pd.DataFrame({"date": dates})
    df["ym"] = df["date"].dt.to_period("M")
    kept, dropped = [], []
    for ym, grp in df.groupby("ym"):
        if len(grp) == 1:
            kept.append(grp["date"].iloc[0])
            continue
        fridays = grp[grp["date"].dt.day_name() == "Friday"]
        keep = fridays["date"].iloc[0] if len(fridays) == 1 else grp["date"].min()
        kept.append(keep)
        for d in grp["date"]:
            if d != keep:
                dropped.append((str(ym), d, keep,
                                 "金曜でない重複(改定/巻き添え系列と判定、金曜のみ残す)"))
    return sorted(kept), dropped


def dedupe_cpi(dates):
    """CPI(Consumer Price Index)は2019-2024の各2月に月内2回の発表日が出る(1月分データの
    本報告の直前2-4日に別の巻き添え/予告系列が同release_idに載る)。ALFRED実データ
    (series=CPIAUCSL, realtime_start)で検算した結果、月内で"遅い方"の日付が実際に当月データが
    初出現する日(=本当のCPI発表日)であると6年分すべてで確認済み(2019/2020/2021/2022/2023/2024)。
    ゆえに月内で最も遅い日付を残し、早い方を落とす。"""
    df = pd.DataFrame({"date": dates})
    df["ym"] = df["date"].dt.to_period("M")
    kept, dropped = [], []
    for ym, grp in df.groupby("ym"):
        if len(grp) == 1:
            kept.append(grp["date"].iloc[0])
            continue
        keep = grp["date"].max()
        kept.append(keep)
        for d in grp["date"]:
            if d != keep:
                dropped.append((str(ym), d, keep,
                                 "月内で早い方の重複(ALFRED実データ検算で当月データの初出現は"
                                 "遅い方の日付=本当のCPI発表日と確認済み。早い方は別系列/予告)"))
    return sorted(kept), dropped


def et_hm_to_utc_broker(date_strs, hour, minute):
    """context_time_fomc.et14_to_utc_broker と同一手法(カレンダー日+固定ET時刻->
    tz_localize(America/New_York)->tz_convert(UTC/Europe-Riga))を任意の時:分に一般化したもの。
    DSTはtzライブラリが自動処理、固定オフセット手書きは禁止(仕様カード指示)。"""
    et = pd.DatetimeIndex(pd.to_datetime(list(date_strs))) + pd.Timedelta(hours=hour, minutes=minute)
    et = et.tz_localize("America/New_York")
    dt_utc = et.tz_convert("UTC").tz_localize(None)
    dt_broker = et.tz_convert("Europe/Riga").tz_localize(None)
    return dt_utc, dt_broker


def build_calendar(kind):
    release_id = {"NFP": 50, "CPI": 10}[kind]
    dedupe_fn = {"NFP": dedupe_nfp, "CPI": dedupe_cpi}[kind]
    anchor = {"NFP": ANCHOR_NFP, "CPI": ANCHOR_CPI}[kind]

    raw_dates, raw_count = fetch_release_dates(release_id)
    kept, dropped = dedupe_fn(raw_dates)
    print(f"\n[{kind}] FRED release_id={release_id}: API count={raw_count} (取得{len(raw_dates)}件) "
          f"-> 月次主系列に正規化後={len(kept)}件 (落とした{len(dropped)}件)")
    for ym, d, keep, reason in dropped:
        print(f"    drop {d.date()} (同月{ym}内、残したのは {keep.date()})  理由: {reason}")

    dt_utc, dt_broker = et_hm_to_utc_broker([d.strftime("%Y-%m-%d") for d in kept], 8, 30)
    out = (pd.DataFrame({"kind": kind, "dt_utc": dt_utc, "dt_broker": dt_broker})
           .sort_values("dt_utc").reset_index(drop=True))

    # ---- 検算 ----
    anchor_ts = pd.Timestamp(anchor)
    anchor_ok = (out["dt_utc"].dt.normalize() == anchor_ts).any()
    print(f"  [アンカー検算] {kind} {anchor} が最終カレンダーに存在するか -> "
          f"{'PASS' if anchor_ok else 'FAIL(停止して報告すべき事態)'}")

    diff_h = (out["dt_broker"] - out["dt_utc"]).dt.total_seconds() / 3600
    offsets = sorted(diff_h.round(6).unique())
    offsets_ok = set(np.round(offsets)) <= {2.0, 3.0}
    print(f"  [tzオフセット検算] dt_broker-dt_utc の一意な値: {offsets}h -> "
          f"{'PASS(+2h/+3h=Riga冬/夏)' if offsets_ok else 'FAIL'}")

    by_year = out["dt_utc"].dt.year.value_counts().sort_index()
    print(f"  [年別件数] (期待値: 年12本前後、2026は年半ばまでのため部分年)")
    for y, c in by_year.items():
        note = "  <-部分年(2026年進行中)" if y == pd.Timestamp.utcnow().year else ""
        fill = f"{c/12*100:.0f}%" if y != pd.Timestamp.utcnow().year else "n/a"
        print(f"    {y}: {c}回  充足率(対12)={fill}{note}")

    dow_check = out["dt_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.day_name()
    if kind == "NFP":
        pct_fri = (dow_check == "Friday").mean() * 100
        print(f"  [カデンス検算] ET暦日の曜日分布(第1金曜想定): Friday={pct_fri:.1f}%  "
              f"(非金曜は祝日/シャットダウン等でのスライドと想定、全件を却下はしない)")
    return out


def build_all_calendars():
    fomc = pd.read_csv(DATA_FOMC, parse_dates=["dt_utc", "dt_broker"])
    print(f"[FOMC] 既存(読み取り専用) {DATA_FOMC}: {len(fomc)}行")
    ok_utc, ok_broker = ctf.validate_tz_method(fomc)
    print(f"  [tz変換メソッド検算(流用元)] 既存FOMC48+行に 14:00ET->UTC/Riga を再適用: "
          f"dt_utc一致={ok_utc} dt_broker一致={ok_broker} -> "
          f"{'PASS(このメソッドを08:30ETに置き換えて信用してよい)' if ok_utc and ok_broker else 'FAIL'}")

    nfp = build_calendar("NFP")
    nfp.to_csv(DATA_NFP, index=False)
    print(f"  -> {DATA_NFP} に書き出し ({len(nfp)}行)")

    cpi = build_calendar("CPI")
    cpi.to_csv(DATA_CPI, index=False)
    print(f"  -> {DATA_CPI} に書き出し ({len(cpi)}行)")

    return fomc, nfp, cpi


# ============================================================================
# 第二部: 事象セット x スイープ
# ============================================================================

def load_broker_events(df_or_path):
    """event_scalp_cond.py の main() と同一の変換(dt_broker を tz_localize('UTC') で
    タグ付けするだけ -- 値自体はブローカー壁時計のまま、算術のためのタグ)。"""
    if isinstance(df_or_path, str):
        ev = pd.read_csv(df_or_path, parse_dates=["dt_utc", "dt_broker"])
    else:
        ev = df_or_path.copy()
    ev["dt_broker"] = ev["dt_broker"].dt.tz_localize("UTC")
    return ev


def build_event_sets(fomc, nfp, cpi):
    f = load_broker_events(fomc)["dt_broker"]
    n = load_broker_events(nfp)["dt_broker"]
    c = load_broker_events(cpi)["dt_broker"]

    union_all = pd.concat([f, n, c])
    union_dedup = pd.Series(sorted(union_all.drop_duplicates()))
    n_exact_dupe_ts = len(union_all) - len(union_dedup)

    days_f = set(f.dt.normalize())
    days_n = set(n.dt.normalize())
    days_c = set(c.dt.normalize())
    same_day_overlaps = len((days_f & days_n) | (days_f & days_c) | (days_n & days_c))

    sets = {
        "CPI": sorted(c),
        "NFP": sorted(n),
        "FOMC+CPI+NFP": list(union_dedup),
        "FOMC": sorted(f),
    }
    print(f"\n[事象セット] CPI={len(sets['CPI'])}  NFP={len(sets['NFP'])}  FOMC={len(sets['FOMC'])}  "
          f"合算(3種類・重複タイムスタンプdedupe後)={len(sets['FOMC+CPI+NFP'])}  "
          f"(厳密同時刻の重複除去数={n_exact_dupe_ts}, 同一暦日で時刻違いの重複日数={same_day_overlaps}"
          f"[同日でも時刻が違うため各々独立イベントとしてそのまま採用、除去しない])")
    return sets


def run_sweep_for_set(name, events_all, df, span_label, deep=True):
    real_full = build_scalp_table(df, events_all, W_C, HSET, f"{span_label}-{name}")
    if real_full.empty or len(real_full) < 5:
        print(f"\n[{name}] too few usable events (n={len(real_full)}) -- skip")
        return None
    span_years = (real_full["t0"].max() - real_full["t0"].min()).days / 365.25
    print(f"\n{'='*100}\n事象セット: {name}  usable n={len(real_full)}  span={span_years:.2f}y  "
          f"({len(real_full)/span_years:.2f}件/年)\n{'='*100}")

    null_full = null_scalp_table(df, events_all, W_C, HSET, f"{span_label}-{name}",
                                  draws_target=NULL_DRAWS_TARGET)
    print(f"  null pool: {len(null_full)} draws")

    sw, subsets = sweep_table(real_full, null_full, "confirm_move_atr", FRACS, HSET,
                               COST_BASE, span_years, deep=deep)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"\n--- {name}: C_atrスイープ x H, cost_base=${COST_BASE}/oz ---")
    print(sw.round(4).to_string(index=False))

    print(f"\n--- {name}: 同スイープ, cost_alt=${COST_ALT}/oz (保守) -- net_mean/median点推定のみ ---")
    for frac in FRACS:
        sub_real, _, thr = subsets[frac]
        for h in HSET:
            gcol = f"g_{h}"
            g = sub_real[gcol].dropna()
            if len(g) == 0:
                continue
            net = g - COST_ALT
            print(f"  frac={frac:.2f} thr={thr:.4f} H={h:>2} n={len(g):>3}  "
                  f"net_mean={net.mean():+.4f}  net_median={net.median():+.4f}  "
                  f"P(net>0)={((net>0).mean()*100):.1f}%")

    ft = follow_through_table(real_full, HSET)
    print(f"\n--- {name}: follow-through Spearman(confirm_move, g_H) ---")
    print(ft.round(4).to_string(index=False))

    return dict(real=real_full, null=null_full, sweep=sw, subsets=subsets,
                span=span_years, ft=ft)


def deep_dive_stability(name, res, frac=PRIMARY_FRAC, h=PRIMARY_H):
    if res is None:
        return
    if frac not in res["subsets"]:
        print(f"\n[{name}] frac={frac} 未計算 -- deep-dive省略")
        return
    sub_real, sub_null, thr = res["subsets"][frac]
    gcol = f"g_{h}"
    if gcol not in sub_real.columns or sub_real[gcol].dropna().empty:
        print(f"\n[{name}] frac={frac} H={h} でn=0 -- deep-dive省略")
        return

    print(f"\n{'#'*100}\n時代安定性 deep-dive: {name}  frac={frac:.2f} (C_atr>={thr:.4f})  H={h}min  "
          f"n={len(sub_real[gcol].dropna())}\n{'#'*100}")

    print(f"\n--- IS/OOS(前半/後半) ---")
    ist, span_desc = is_oos_table(sub_real, HSET, COST_BASE)
    print(f"  {span_desc}")
    print(ist.round(4).to_string(index=False))

    print(f"\n--- 年別breakdown @ H={h}min (net, cost_base=${COST_BASE}) ---")
    at = annual_table(sub_real, h, COST_BASE)
    print(at.round(4).to_string(index=False))

    print(f"\n--- 巡回ブロックブートストラップ(暦月ブロック, resample-with-replacement, B={B_BOOT}) ---")
    for bm in BLOCK_MONTHS:
        r = block_bootstrap_ci(sub_real, gcol, COST_BASE, bm)
        if r is None:
            print(f"  block={bm:>2}mo: n<4 events, skipped")
        elif "note" in r:
            print(f"  block={bm:>2}mo: {r['note']} (n_blocks={r['n_blocks']})")
        else:
            print(f"  block={bm:>2}mo: n_blocks={r['n_blocks']:>3} n_events={r['n_events']:>3}  "
                  f"net_median={r['median']:+.4f}  [p5={r['p5']:+.4f}, p95={r['p95']:+.4f}]  "
                  f"std={r['std']:.4f}  P(net<=0)={r['P_le_0']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="last 2 years of events only")
    ap.add_argument("--skip-fetch", action="store_true",
                     help="re-use existing data/ext_nfp_dates.csv / ext_cpi_dates.csv instead of re-hitting FRED")
    args = ap.parse_args()

    print(f"{'='*100}\n第一部: CPI/NFPカレンダー構築(FRED)\n{'='*100}")
    if args.skip_fetch and os.path.exists(DATA_NFP) and os.path.exists(DATA_CPI):
        fomc = pd.read_csv(DATA_FOMC, parse_dates=["dt_utc", "dt_broker"])
        nfp = pd.read_csv(DATA_NFP, parse_dates=["dt_utc", "dt_broker"])
        cpi = pd.read_csv(DATA_CPI, parse_dates=["dt_utc", "dt_broker"])
        print("[--skip-fetch] 既存CSVを再利用(FRED再取得なし)")
    else:
        fomc, nfp, cpi = build_all_calendars()

    print(f"\n{'='*100}\n第二部: 事象セット x スイープ (gold m5)\n{'='*100}")
    sets = build_event_sets(fomc, nfp, cpi)

    if args.smoke:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=730)
        cutoff = pd.Timestamp(cutoff, tz="UTC")
        for k in sets:
            before = len(sets[k])
            sets[k] = [e for e in sets[k] if e >= cutoff]
            print(f"[SMOKE] {k}: {before} -> {len(sets[k])} events since {cutoff.date()}")

    df_gold = load_mt5_csv("data/vantage_xauusd_m5.csv")
    df_gold = df_gold.loc[GOLD_M5_START:]
    print(f"\nGOLD m5 data: {len(df_gold)} bars, span {df_gold.index.min()} .. {df_gold.index.max()}  "
          f"(density boundary GOLD_M5_START={GOLD_M5_START})")

    order = ["CPI", "NFP", "FOMC+CPI+NFP", "FOMC"]
    results = {}
    for name in order:
        results[name] = run_sweep_for_set(name, sets[name], df_gold, "GOLD", deep=True)

    print(f"\n\n{'='*100}\n時代安定性(本命) -- frac={PRIMARY_FRAC:.2f}(C_atr上位50%付近) x H={PRIMARY_H}min を"
          f"4事象セットで比較\n{'='*100}")
    for name in order:
        deep_dive_stability(name, results[name])

    n_cells_gold = len(order) * len(FRACS) * len(HSET)
    print(f"\n[多重比較注記] gold側で走査したセル数 = {len(order)}事象セット x {len(FRACS)}frac x "
          f"{len(HSET)}H = {n_cells_gold}セル。単発の事前登録検定に対しては ~{n_cells_gold}倍で"
          f"割り引いて読むこと(Bonferroni目安)。")

    # ---------------- 参考: BTC で合算(3)を1回だけ ----------------
    print(f"\n\n{'='*100}\n参考: BTC で 合算(FOMC+CPI+NFP) を1回だけ(カード17でBTC順張りは死=確認のみ、深追いしない)\n{'='*100}")
    df_btc = load_mt5_csv("data/vantage_btcusd_m5.csv")
    df_btc = df_btc.loc[BTC_M5_START:]
    print(f"BTC m5 data: {len(df_btc)} bars, span {df_btc.index.min()} .. {df_btc.index.max()}")
    events_btc = sets["FOMC+CPI+NFP"]
    real_btc = build_scalp_table(df_btc, events_btc, W_C, HSET, "BTC-FOMC+CPI+NFP")
    if not real_btc.empty and len(real_btc) >= 5:
        span_btc = (real_btc["t0"].max() - real_btc["t0"].min()).days / 365.25
        null_btc = null_scalp_table(df_btc, events_btc, W_C, HSET, "BTC-FOMC+CPI+NFP",
                                     draws_target=NULL_DRAWS_TARGET)
        cost_btc = COST_ROUNDTRIP["BTC"]["base"]
        sw_btc, _ = sweep_table(real_btc, null_btc, "confirm_move_atr", FRACS, HSET, cost_btc,
                                 span_btc, deep=True)
        print(f"usable n={len(real_btc)} span={span_btc:.2f}y  cost_base=${cost_btc}")
        print(sw_btc.round(4).to_string(index=False))
        n_cells_btc = len(FRACS) * len(HSET)
        print(f"[多重比較注記] BTC側は {n_cells_btc}セル(この1セットのみ、深掘りなし)")
    else:
        print(f"BTC 合算イベント usable n={len(real_btc)} -- 不足のためスキップ")

    print(f"\n\n{'='*100}\nDONE\n{'='*100}")
    print(f"\n実行コマンド: .venv/bin/python scratchpad/news_scalp_combined.py{' --smoke' if args.smoke else ''}")


if __name__ == "__main__":
    main()
