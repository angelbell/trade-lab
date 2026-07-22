"""ICT 忠実版 — Draw on Liquidity: 外部流動性プール検出（未タップ・確定済みのみ、先読み厳禁）。

背骨: 正典の Draw on Liquidity＝「次の磁石（未タップ流動性プール）」を、日足/週足の確定スイング高安から
検出し、EURUSD m15 の各セットアップの**約定時刻(fill_dt)**時点で「確定済み・未タップ」なものだけを
参照する（進行中バー・未確定pivotは使わない）。

検出ロジック:
  1. スイング = フラクタル pivot(2,2)（中央足 i が [i-2,i+2] の中で最高値/最安値、同値可）。
     確定 = pivot足の2本後の足(i+2)が閉じた時点＝そのタイムスタンプは bar[i+3] の open
     （MT5 CSV はバーを open 時刻でラベルするため、i+2 が完全に閉じたと保証できる最初の瞬間）。
     系列末尾で i+3 が存在しない pivot は「未確定」として捨てる（サンプル末尾の未来参照を防ぐ）。
  2. タップ判定は EURUSD m15 系列（セットアップと同じ価格系列）で行う：confirm_ts 以降で
     価格が buy-side なら high>=level、sell-side なら low<=level に達した最初の m15 バーの時刻を
     tap_ts として記録（一度だけ・グローバルに計算）。
     先読みでない理由: 「fill_dt 時点で untapped か」の判定は
         confirm_ts <= fill_dt  AND  (tap_ts is None OR tap_ts > fill_dt)
     という条件だけを使う。tap_ts の値そのものが fill_dt より未来にあっても、それは
     「fill_dt までにタップが起きていない」という事実（=fill_dt までのデータだけで判定可能な命題）
     を確認しているに過ぎない。tap_ts の具体的な値を出口/入口の計算に使うことは無い
     （untapped_mask の外では一切参照しない）。
     同足タイブレーク: tap_ts == fill_dt（同じm15足で入口約定とタップが同時に起きる場合）は
     「タップ済み」扱い（tap_ts > fill_dt が False になる＝保守側）。

self-test: .venv/bin/python experiments/ict_draw_on_liq_pools.py
  - 決定性: build_pools() を同じ入力で2回呼び、全配列が bit-identical であることを確認。
  - 先読み健全性: 全 confirm_ts が pivot足自身の時刻より後であること、
    全 tap_ts が対応する confirm_ts 以降であることを assert。
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv

DAILY_PATH = "data/vantage_eurusd_d1.csv"
WEEKLY_PATH = "data/vantage_eurusd_w1.csv"
WEEKLY_CUTOFF = "2000-01-01"   # CLAUDE.md: EURUSD 1999年以前は合成/固定相場につき使用禁止（保守的に2000年で切る）


def load_htf_naive(path, cutoff=None):
    """Vantage CSV → ブローカーnaive時刻の日足/週足 df（load_mt5_csv を流用、tz情報だけ剥がす）。"""
    df = load_mt5_csv(path)
    df = df.copy()
    df.index = df.index.tz_localize(None)
    if cutoff is not None:
        df = df.loc[cutoff:]
    return df


def fractal_pivots(idx_ts, high, low):
    """pivot(2,2)。戻り値 = (buy_idx, buy_level, buy_confirm_ts), (sell_idx, sell_level, sell_confirm_ts)
    confirm_ts = idx_ts[i+3]（i+2 が閉じたと保証できる最初の瞬間）。i+3 が範囲外の pivot は捨てる。"""
    n = len(high)
    b_idx, b_lvl, b_ct = [], [], []
    s_idx, s_lvl, s_ct = [], [], []
    for i in range(2, n - 2):
        if i + 3 >= n:
            continue
        if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
            b_idx.append(i); b_lvl.append(high[i]); b_ct.append(idx_ts[i+3])
        if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
            s_idx.append(i); s_lvl.append(low[i]); s_ct.append(idx_ts[i+3])
    return (np.array(b_idx), np.array(b_lvl), np.array(b_ct, dtype="datetime64[ns]"),
            np.array(s_idx), np.array(s_lvl), np.array(s_ct, dtype="datetime64[ns]"))


def first_tap(m15_ts, m15_high, m15_low, side, level, confirm_ts):
    """confirm_ts 以降で最初にレベルへ達した m15 バーの時刻（無ければ pd.NaT）。"""
    start = np.searchsorted(m15_ts, confirm_ts, side="left")
    if start >= len(m15_ts):
        return np.datetime64("NaT")
    if side == "buy":
        mask = m15_high[start:] >= level
    else:
        mask = m15_low[start:] <= level
    if not mask.any():
        return np.datetime64("NaT")
    return m15_ts[start + int(mask.argmax())]


def build_side_pools(idx_arr, lvl_arr, ct_arr, side, m15_ts, m15_high, m15_low):
    tap = np.array([first_tap(m15_ts, m15_high, m15_low, side, lvl_arr[k], ct_arr[k])
                    for k in range(len(lvl_arr))], dtype="datetime64[ns]")
    return dict(idx=idx_arr, level=lvl_arr, confirm=ct_arr, tap=tap)


def build_pools(m15_ts, m15_high, m15_low, daily_df=None, weekly_df=None):
    """全プールテーブルを構築。daily_df/weekly_df は load_htf_naive() の戻り値（省略時は自前でロード）。
    戻り値 = dict(daily_buy, daily_sell, weekly_buy, weekly_sell) 、各 dict(idx, level, confirm, tap)。"""
    if daily_df is None:
        daily_df = load_htf_naive(DAILY_PATH)
    if weekly_df is None:
        weekly_df = load_htf_naive(WEEKLY_PATH, cutoff=WEEKLY_CUTOFF)

    d_ts = daily_df.index.values.astype("datetime64[ns]")
    d_hi, d_lo = daily_df["high"].values, daily_df["low"].values
    w_ts = weekly_df.index.values.astype("datetime64[ns]")
    w_hi, w_lo = weekly_df["high"].values, weekly_df["low"].values

    db_i, db_l, db_c, ds_i, ds_l, ds_c = fractal_pivots(d_ts, d_hi, d_lo)
    wb_i, wb_l, wb_c, ws_i, ws_l, ws_c = fractal_pivots(w_ts, w_hi, w_lo)

    return dict(
        daily_buy=build_side_pools(db_i, db_l, db_c, "buy", m15_ts, m15_high, m15_low),
        daily_sell=build_side_pools(ds_i, ds_l, ds_c, "sell", m15_ts, m15_high, m15_low),
        weekly_buy=build_side_pools(wb_i, wb_l, wb_c, "buy", m15_ts, m15_high, m15_low),
        weekly_sell=build_side_pools(ws_i, ws_l, ws_c, "sell", m15_ts, m15_high, m15_low),
    )


def untapped_mask(confirm_ts, tap_ts, fill_dt):
    """fill_dt 時点で確定済み・未タップの真偽配列。fill_dt: np.datetime64。"""
    return (confirm_ts <= fill_dt) & (np.isnat(tap_ts) | (tap_ts > fill_dt))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import io, contextlib
    print("Draw on Liquidity プール検出: 自己検査")
    with contextlib.redirect_stderr(io.StringIO()):
        from ict_population import load_prepped
    df, tarr, dates, span = load_prepped("eurusd")
    m15_ts = df["broker_dt"].values.astype("datetime64[ns]")
    m15_high, m15_low = df["high"].values, df["low"].values

    daily_df = load_htf_naive(DAILY_PATH)
    weekly_df = load_htf_naive(WEEKLY_PATH, cutoff=WEEKLY_CUTOFF)
    print(f"  daily bars={len(daily_df)} ({daily_df.index[0]} .. {daily_df.index[-1]})")
    print(f"  weekly bars={len(weekly_df)} ({weekly_df.index[0]} .. {weekly_df.index[-1]}) [cutoff={WEEKLY_CUTOFF}]")

    pools1 = build_pools(m15_ts, m15_high, m15_low, daily_df, weekly_df)
    for k, v in pools1.items():
        n_untapped = int(np.isnat(v["tap"]).sum())
        print(f"  {k:12s} n_pool={len(v['level']):5d}  never-tapped-in-sample={n_untapped:5d}")

    # 1) 決定性: 同じ入力で2回呼び bit-identical
    pools2 = build_pools(m15_ts, m15_high, m15_low, daily_df, weekly_df)
    ident = True
    for k in pools1:
        for f in ("idx", "level", "confirm"):
            if not np.array_equal(pools1[k][f], pools2[k][f]):
                ident = False; print(f"  !! 不一致: {k}.{f}")
        t1, t2 = pools1[k]["tap"], pools2[k]["tap"]
        if not (np.array_equal(np.isnat(t1), np.isnat(t2)) and
                np.array_equal(t1[~np.isnat(t1)], t2[~np.isnat(t2)])):
            ident = False; print(f"  !! 不一致: {k}.tap")
    print(f"  決定性(2回呼び bit-identical): {'OK' if ident else 'NG'}")

    # 2) 先読み健全性
    ok = True
    for k, v in pools1.items():
        htf_ts = (daily_df.index.values.astype("datetime64[ns]") if k.startswith("daily")
                 else weekly_df.index.values.astype("datetime64[ns]"))
        # confirm_ts はどの pivot 足の時刻よりも後（idx+3 の時刻なので idx 自身の時刻より真に後）
        pivot_ts = htf_ts[v["idx"]]
        if not (v["confirm"] > pivot_ts).all():
            ok = False; print(f"  !! 先読み疑い: {k} の confirm_ts が pivot足以前")
        tapped = ~np.isnat(v["tap"])
        if tapped.any() and not (v["tap"][tapped] >= v["confirm"][tapped]).all():
            ok = False; print(f"  !! 先読み疑い: {k} の tap_ts が confirm_ts より前")
    print(f"  先読み健全性チェック: {'OK' if ok else 'NG'}")

    # サンプル: 直近の未タップ日足buyプールを何件か表示
    v = pools1["daily_buy"]
    untapped = np.isnat(v["tap"])
    print(f"\n  直近5件の未タップ日足buyプール（サンプル: confirm/level）:")
    idxs = np.where(untapped)[0]
    for j in idxs[np.argsort(v["confirm"][idxs])[-5:]]:
        print(f"    confirm={pd.Timestamp(v['confirm'][j])}  level={v['level'][j]:.5f}")
