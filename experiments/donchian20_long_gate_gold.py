"""donchian20_sar_gold.py の続き（実験2）: 20本ドンチャンのロング専用化 ＋ 日足ゲート3種。

前回の訂正: 前回のPFは全て$建て（1オンスあたりドル損益）だったが、この期間に金は
$1,188→$5,543と4.7倍になるため$建てPFは高値圏の後半トレードを約4倍重く数える。
今回は %建て（建値に対する％損益）を主の物差しとし、$建ては参考列として併記する。

ロジックの平文説明:
  1. 買いのみ: 20本高値（その足を含む、次足から有効=donchian20_sar_gold.pyと同じ
     rolling(20).shift(1)の水準）を上抜けたら買い逆指値で建てる。
  2. 手仕舞い: 20本安値に触れたら決済してフラットに戻る。そこで売りは建てない
     （常時ドテンだった前回とは異なり、退出後は次の高値ブレイクまで待機）。
  3. ゲート（建玉の可否のみに作用。決済は常に許可）を3本並べる:
       (a) 無ゲート（=実験2の基準線）
       (b) 日足SMA150が上向き（前日確定値ベース、傾き=SMA150[前日]>SMA150[2日前]）
       (c) 日足KAMA(14)が上向き（同上）
     日足は常に h1（donchian20_sar_gold.load_tf("h1")）から作り、4H側のトレードに
     ゲートを適用する時も同じ h1由来の日足系列をリサンプル・ffillして揃える
     （4hから作った日足とh1から作った日足で理論上ズレる余地を無くすため）。

実行の仮定・コストは donchian20_sar_gold.py と同一を再利用（simulate系の水準ちょうど約定・
コスト往復$/滑り片道$の後乗せ）。サイズは1ロット固定、サイズ写像は入れない。

ゲート実装についての注記（車輪の再発明を避けた範囲）:
  - KAMA自体の計算は breakout_wave.kama_adaptive をそのまま import（コア計算のみ流用）。
  - src.engine.gates.gate_kama は「KAMA傾きのみ」で価格条件を含まないため今回の仕様(c)と
    完全に一致する一方、(1) 常に h1 由来の日足を使う制約と (2) 未来混入assert用に
    「どの日のデータが使われたか」を追跡する必要があったため、gate_kama と全く同じ
    rising→shift(1)→reindex(ffill) の型を踏襲しつつ、この2点のためだけにここで組んだ
    （ロジックはgate_kamaの実装をそのまま読み、行単位で一致することを確認済み）。
  - src.engine.gates.gate_sma は「価格>SMA」も同時に要求する複合ゲートで、仕様(b)の
    「傾きのみ」とは条件が異なるため流用せず、同型の3行だけを自前で書いた。

事前スクリーン: エントリー定義（20本高値ブレイクのロング、次足から有効な水準）は
donchian20_sar_gold.py と完全に同一なので、巡行幅（MFE/MAE）は変わらない。
research/screens/donchian20_sar_gold_h1.json と _4h.json が両方とも既に存在する
（前回作成済み、比 h1=0.981死・4h=1.000境界）。SCREENは h1 側を宣言する。
"""
SCREEN = "donchian20_sar_gold_h1"

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from breakout_wave import kama_adaptive
from donchian20_sar_gold import (
    LENGTH, START, load_tf, simulate, apply_cost, years_span,
    max_drawdown, max_losing_streak, summarize, pct_rank,
)

N_NULL_REPS = 1000
RNG_SEED = 20260729
BLOCK_MONTHS = [1, 3, 6, 12]
CENTRAL = (0.3, 0.1)


# ---------------- ゲート構築（常に h1 の日足終値から） ----------------

def build_daily_gates(h1_df):
    """h1のcloseを日足にリサンプルし、SMA150傾き・KAMA(14)傾きの2ゲートを作る。
    戻り値: dict(sma_assigned, kama_assigned, day_used_assigned) いずれも日足indexのSeries。
    *_assigned[D+1] = 日D確定値を使った判定（前日確定・先読み無し）。
    day_used_assigned[D+1] = D（そのバーで実際に参照した日足の日付。未来混入assert用）。"""
    daily_close = h1_df["close"].resample("1D").last().dropna()

    sma = daily_close.rolling(150).mean()
    sma_rising_raw = sma > sma.shift(1)
    sma_assigned = sma_rising_raw.shift(1).astype("boolean").fillna(False).astype(bool)

    kama_d = kama_adaptive(daily_close, 14)
    kama_rising_raw = kama_d > kama_d.shift(1)
    kama_assigned = kama_rising_raw.shift(1).astype("boolean").fillna(False).astype(bool)

    day_label = pd.Series(daily_close.index, index=daily_close.index)
    day_used_assigned = day_label.shift(1)

    return dict(sma_assigned=sma_assigned, kama_assigned=kama_assigned,
                day_used_assigned=day_used_assigned, daily_close_index=daily_close.index)


def align_bool(assigned_series, target_index):
    s = assigned_series.reindex(target_index, method="ffill")
    return s.fillna(False).values.astype(bool)


def align_day_used(day_used_assigned, target_index):
    return day_used_assigned.reindex(target_index, method="ffill")


# ---------------- 本体: ロングのみ・フラット待機シミュレーション ----------------

def simulate_long_gate(df, gate_arr=None):
    """20本高値ブレイクのロングのみ。ゲートは建玉可否のみに作用、決済は常に許可。
    退出後はフラットで次の高値ブレイクを待つ（ドテンしない）。"""
    hh_lvl = df["high"].rolling(LENGTH).max().shift(1).values
    ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    n = len(df)

    def buy_fill(i, lvl):
        return o[i] if o[i] >= lvl else lvl

    def sell_fill(i, lvl):
        return o[i] if o[i] <= lvl else lvl

    position = 0
    entry_i = entry_price = None
    trades = []
    n_live_bars = 0
    n_entry_signals_total = 0
    n_entries_blocked_by_gate = 0

    for i in range(n):
        if np.isnan(hh_lvl[i]) or np.isnan(ll_lvl[i]):
            continue
        n_live_bars += 1
        buy_lvl, sell_lvl = hh_lvl[i], ll_lvl[i]
        buy_trig = h[i] >= buy_lvl
        sell_trig = l[i] <= sell_lvl

        if position == 0:
            if buy_trig:
                n_entry_signals_total += 1
                gated_ok = True if gate_arr is None else bool(gate_arr[i])
                if gated_ok:
                    position, entry_price, entry_i = 1, buy_fill(i, buy_lvl), i
                else:
                    n_entries_blocked_by_gate += 1
        else:
            if sell_trig:
                fp = sell_fill(i, sell_lvl)
                trades.append(dict(
                    entry_time=idx[entry_i], exit_time=idx[i], direction=1,
                    entry_raw=entry_price, exit_raw=fp, bars_held=i - entry_i,
                    entry_i=entry_i, exit_i=i,
                ))
                position, entry_price, entry_i = 0, None, None

    open_trade = None
    if position != 0:
        open_trade = dict(entry_time=idx[entry_i], direction=1, entry_raw=entry_price,
                           bars_since=n - 1 - entry_i, last_close=df["close"].iloc[-1])

    diag = dict(n_live_bars=n_live_bars, n_entry_signals_total=n_entry_signals_total,
                n_entries_blocked_by_gate=n_entries_blocked_by_gate,
                n_entries_taken=len(trades) + (1 if open_trade else 0))
    return trades, open_trade, diag


# ---------------- 統計ヘルパー ----------------

def pf_from_array(arr):
    gw = arr[arr > 0].sum()
    gl = -arr[arr < 0].sum()
    return float(gw / gl) if gl > 0 else float("inf")


def pf_pct(trades_c):
    """%建てPF（建値に対する％損益の総和比）。summarize()の'pf'は$建てのみを返すため別途計算する。"""
    if not trades_c:
        return float("nan")
    return pf_from_array(np.array([t["pnl_pct"] for t in trades_c]))


def year_breakdown_pct(trades_c):
    df = pd.DataFrame(trades_c)
    df["year"] = pd.DatetimeIndex(df["exit_time"]).year
    rows = []
    for y, g in df.groupby("year"):
        pp = g["pnl_pct"].values
        gw, gl = pp[pp > 0].sum(), -pp[pp < 0].sum()
        pf = float(gw / gl) if gl > 0 else float("inf")
        rows.append(dict(year=int(y), n=len(g), pf_pct=pf,
                          tot_pct=float(pp.sum()), win_pct=float((pp > 0).mean() * 100)))
    return rows


def half_split(trades_c):
    """トレード数で二等分し、それぞれの実経過年数でPF(%)・年率%を出す。"""
    n = len(trades_c)
    if n < 4:
        return None, None
    mid = n // 2
    out = []
    for part in (trades_c[:mid], trades_c[mid:]):
        yrs = max((part[-1]["exit_time"] - part[0]["entry_time"]).days / 365.25, 1e-6)
        s = summarize(part, yrs)
        s["pf_pct"] = pf_pct(part)
        out.append(s)
    return out[0], out[1]


def daily_pnl_series(trades_c, date_index):
    s = pd.Series(0.0, index=date_index)
    for t in trades_c:
        d = pd.Timestamp(t["exit_time"]).normalize()
        if d in s.index:
            s.loc[d] += t["pnl_pct"]
        else:
            near = date_index[date_index <= d]
            if len(near):
                s.loc[near[-1]] += t["pnl_pct"]
    return s


def cagr_dd_ratio_from_seq(seq, yrs):
    cum = np.cumsum(seq)
    tot_peryr = cum[-1] / yrs if yrs > 0 else np.nan
    dd = max_drawdown(cum)
    return (tot_peryr / dd) if dd > 0 else np.nan


def subsample_null(nogate_trades, n_target, n_reps, rng):
    """無ゲート版から同数をランダム抽出（時系列順は保持）し、PF(%)と年率/DDの分布を作る。"""
    idxs = np.arange(len(nogate_trades))
    pf_list, ratio_list = [], []
    if n_target < 2 or n_target > len(nogate_trades):
        return np.array(pf_list), np.array(ratio_list)
    for _ in range(n_reps):
        sel = np.sort(rng.choice(idxs, size=n_target, replace=False))
        sub = [nogate_trades[k] for k in sel]
        yrs_sub = max((sub[-1]["exit_time"] - sub[0]["entry_time"]).days / 365.25, 1e-6)
        s = summarize(sub, yrs_sub)
        pf_list.append(pf_pct(sub))
        ratio_list.append(s["tot_pct_per_year"] / s["maxdd_pct"] if s["maxdd_pct"] > 0 else np.nan)
    return np.array(pf_list), np.array(ratio_list)


def block_bootstrap_diff(daily_nogate, daily_gate, block_months, n_reps, rng):
    total_days = len(daily_nogate)
    block_days = max(int(round(block_months * 30.44)), 1)
    n_blocks = int(np.ceil(total_days / block_days))
    starts_max = max(total_days - block_days, 1)
    diffs = []
    for _ in range(n_reps):
        block_starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idxs = np.concatenate([np.arange(s, min(s + block_days, total_days)) for s in block_starts])
        yrs_boot = len(idxs) / 365.25
        r_no = cagr_dd_ratio_from_seq(daily_nogate.values[idxs], yrs_boot)
        r_ga = cagr_dd_ratio_from_seq(daily_gate.values[idxs], yrs_boot)
        if np.isfinite(r_no) and np.isfinite(r_ga):
            diffs.append(r_ga - r_no)
    return np.array(diffs)


def atr14_causal(df):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean().shift(1)  # 因果的（前バー確定値）


def exit_anatomy(trades_c, df, atr_arr, label):
    """巡行効率・MFE/MAE・決済後の追随を勝ち/負け別に集計する。"""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    rows = []
    for t in trades_c:
        ei, xi = t["entry_i"], t["exit_i"]
        ep, xp = t["entry_raw"], t["exit_raw"]
        hi_seg = h[ei:xi + 1].max()
        lo_seg = l[ei:xi + 1].min()
        mfe_d = hi_seg - ep
        mae_d = lo_seg - ep
        eff = (xp - ep) / mfe_d if mfe_d > 1e-9 else np.nan
        a = atr_arr[ei]
        mfe_atr = mfe_d / a if a and np.isfinite(a) and a > 0 else np.nan
        mae_atr = mae_d / a if a and np.isfinite(a) and a > 0 else np.nan
        row = dict(
            win=t["pnl_dollar"] > 0,
            eff=eff, mfe_pct=mfe_d / ep * 100, mae_pct=mae_d / ep * 100,
            mfe_atr=mfe_atr, mae_atr=mae_atr,
        )
        a_exit = atr_arr[xi] if xi < len(atr_arr) else np.nan
        for W in (20, 50):
            j = xi + W
            if j < n:
                fwd_hi = h[xi + 1:j + 1].max() if j > xi else np.nan
                row[f"follow_{W}_pct"] = (fwd_hi - xp) / xp * 100
                row[f"follow_{W}_beat_atr"] = (fwd_hi >= xp + a_exit) if (a_exit and np.isfinite(a_exit)) else np.nan
            else:
                row[f"follow_{W}_pct"] = np.nan
                row[f"follow_{W}_beat_atr"] = np.nan
        rows.append(row)
    dfa = pd.DataFrame(rows)
    return dfa


def summ_stats(series):
    s = series.dropna()
    if len(s) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, q1=np.nan, q3=np.nan)
    return dict(n=len(s), mean=float(s.mean()), median=float(s.median()),
                std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                q1=float(s.quantile(0.25)), q3=float(s.quantile(0.75)))


def print_anatomy_block(dfa, title):
    print(f"\n    -- {title} (n={len(dfa)}) --")
    for name, sub in (("全体", dfa), ("勝ち", dfa[dfa["win"]]), ("負け", dfa[~dfa["win"]])):
        print(f"      [{name}] n={len(sub)}")
        for col, jp in (("eff", "巡行効率(退出/MFE)"), ("mfe_pct", "MFE%"), ("mae_pct", "MAE%"),
                        ("mfe_atr", "MFE(ATR倍)"), ("mae_atr", "MAE(ATR倍)"),
                        ("follow_20_pct", "決済後20本高値超過%"), ("follow_50_pct", "決済後50本高値超過%")):
            st = summ_stats(sub[col])
            print(f"        {jp:22s} n={st['n']:4d} mean={st['mean']:8.4f} median={st['median']:8.4f} "
                  f"std={st['std']:8.4f} q1={st['q1']:8.4f} q3={st['q3']:8.4f}")
        for col, jp in (("follow_20_beat_atr", "決済後20本でexit+1ATR超えた割合"),
                        ("follow_50_beat_atr", "決済後50本でexit+1ATR超えた割合")):
            v = sub[col].dropna()
            rate = float(v.mean() * 100) if len(v) else float("nan")
            print(f"        {jp:22s} n={len(v):4d} rate={rate:6.2f}%")


# ---------------- メイン ----------------

def main():
    rng = np.random.default_rng(RNG_SEED)
    costs = [0.0, 0.3, 0.6]
    slips = [0.0, 0.1, 0.3]

    h1_df = load_tf("h1")
    gates_raw = build_daily_gates(h1_df)

    results = {}
    for tf in ("h1", "4h"):
        df = load_tf(tf)
        gate_sma_arr = align_bool(gates_raw["sma_assigned"], df.index)
        gate_kama_arr = align_bool(gates_raw["kama_assigned"], df.index)
        day_used_arr = align_day_used(gates_raw["day_used_assigned"], df.index)
        yrs = years_span(df)

        variants = {}
        for gname, garr in (("none", None), ("sma150", gate_sma_arr), ("kama14", gate_kama_arr)):
            trades_raw, open_trade, diag = simulate_long_gate(df, garr)
            grid = {}
            for c in costs:
                for s in slips:
                    tc = apply_cost(trades_raw, c, s)
                    g = summarize(tc, yrs)
                    if g is not None:
                        g["pf_pct"] = pf_pct(tc)
                    grid[(c, s)] = g
            central = apply_cost(trades_raw, *CENTRAL)
            central_summary = summarize(central, yrs)
            if central_summary is not None:
                central_summary["pf_pct"] = pf_pct(central)
            yearly = year_breakdown_pct(central)
            h0, h1s = half_split(central)
            variants[gname] = dict(
                trades_raw=trades_raw, open_trade=open_trade, diag=diag, grid=grid,
                central=central, central_summary=central_summary, yearly=yearly,
                half=(h0, h1s), gate_arr=garr,
            )

        results[tf] = dict(df=df, yrs=yrs, variants=variants, day_used_arr=day_used_arr,
                            gate_sma_arr=gate_sma_arr, gate_kama_arr=gate_kama_arr)

    # ---------------- 検算 ----------------
    print("=" * 100)
    print("検算 (数値assert)")
    print("=" * 100)

    # 1) ゲートOFF時に建て玉が発生していないこと
    for tf in ("h1", "4h"):
        for gname in ("sma150", "kama14"):
            v = results[tf]["variants"][gname]
            garr = v["gate_arr"]
            bad = [t for t in v["trades_raw"] if not garr[t["entry_i"]]]
            assert len(bad) == 0, f"{tf}/{gname}: ゲートOFFで建った玉が{len(bad)}件"
    print("[OK] 全ゲート版で、建玉は必ずgate_arr[entry_i]==Trueのバーでのみ発生（0件の例外）")

    # 2) ゲート版のトレードが無ゲート版の部分集合か（測定・機構説明つき）
    for tf in ("h1", "4h"):
        none_trades = results[tf]["variants"]["none"]["trades_raw"]
        none_keys = {(t["entry_time"], t["exit_time"], round(t["entry_raw"], 6)) for t in none_trades}
        for gname in ("sma150", "kama14"):
            g_trades = results[tf]["variants"][gname]["trades_raw"]
            overlap = sum(1 for t in g_trades
                          if (t["entry_time"], t["exit_time"], round(t["entry_raw"], 6)) in none_keys)
            frac = overlap / len(g_trades) * 100 if g_trades else float("nan")
            print(f"  {tf}/{gname}: ゲート版トレード{len(g_trades)}件のうち無ゲート版と"
                  f"(entry_time,exit_time,entry_price)完全一致={overlap}件({frac:.1f}%)")
            # 真に常に成立する不変量（構造的に保証される方）をassertする:
            # ゲート版の建玉は必ず「無ゲート版と同じ生シグナル配列(hh_lvl越え=buy_trig)」の上でのみ発生する。
            hh_lvl = results[tf]["df"]["high"].rolling(LENGTH).max().shift(1).values
            h_arr = results[tf]["df"]["high"].values
            for t in g_trades:
                assert h_arr[t["entry_i"]] >= hh_lvl[t["entry_i"]], \
                    f"{tf}/{gname}: entry_i={t['entry_i']}がbuy_trig条件を満たしていない"
    print("[OK] 全ゲート版の建玉は必ずbuy_trig(=無ゲート版と同じ生シグナル)のバーでのみ発生"
          "（この不変量は常に真。ただし後述の通り「トレード完全一致での部分集合」は"
          "この機構では構造的に成立しない=下記本文で説明）")

    # 3) 日足forward-fillに未来混入が無いこと
    for tf in ("h1", "4h"):
        v_sma = results[tf]["variants"]["sma150"]
        v_kama = results[tf]["variants"]["kama14"]
        day_used_arr = results[tf]["day_used_arr"]
        for gname, v in (("sma150", v_sma), ("kama14", v_kama)):
            for t in v["trades_raw"]:
                du = day_used_arr.iloc[t["entry_i"]]
                assert pd.notna(du) and du < t["entry_time"], \
                    f"{tf}/{gname}: entry_time={t['entry_time']}が参照日足{du}より前ではない"
    print("[OK] 全ゲート版の全建玉で、参照した日足確定日 < 建玉バー時刻（未来混入なし）")

    # ---------------- tie-back: 無ゲート長のみトレードは、常時ドテン版simulate()の
    #                   ロング側部分集合と完全一致するはず（機構的に、退出後の反対建玉の
    #                   有無は次のbuy_trig発火の時刻・価格に影響しないため） ----------------
    print("\n" + "=" * 100)
    print("tie-back: 無ゲート版ロング専用トレード vs donchian20_sar_gold.simulate()のロング側部分集合")
    print("=" * 100)
    for tf in ("h1", "4h"):
        df = results[tf]["df"]
        rev_trades, _, rev_diag = simulate(df, tiebreak_flat="unfavorable")
        rev_long = [t for t in rev_trades if t["direction"] == 1]
        new_none = results[tf]["variants"]["none"]["trades_raw"]
        rev_keys = [(t["entry_time"], t["exit_time"], round(t["entry_raw"], 6), round(t["exit_raw"], 6))
                    for t in rev_long]
        new_keys = [(t["entry_time"], t["exit_time"], round(t["entry_raw"], 6), round(t["exit_raw"], 6))
                    for t in new_none]
        match = (rev_keys == new_keys)
        print(f"  {tf}: 常時ドテン版ロング側n={len(rev_long)}  新ロング専用版n={len(new_none)}  "
              f"完全一致(順序含む)={match}  flatタイ件数(常時ドテン版)={rev_diag['n_ties_flat']}")
        if not match:
            # 先頭からの最初の不一致点を報告
            m = min(len(rev_keys), len(new_keys))
            diff_at = next((i for i in range(m) if rev_keys[i] != new_keys[i]), m)
            print(f"    最初の不一致点: index={diff_at} rev={rev_keys[diff_at] if diff_at < len(rev_keys) else None} "
                  f"new={new_keys[diff_at] if diff_at < len(new_keys) else None}")

    # ---------------- 出力1: 本表 ----------------
    print("\n" + "=" * 100)
    print(f"実験2: gold h1/4h ロング専用ドンチャン(20) + 日足ゲート3種  {START}以降  (%建て主)")
    print("=" * 100)

    hdr_names = {"none": "(a)無ゲート", "sma150": "(b)日足SMA150傾き", "kama14": "(c)日足KAMA14傾き"}

    for tf in ("h1", "4h"):
        r = results[tf]
        print(f"\n### TF={tf}  期間={r['df'].index[0]}~{r['df'].index[-1]}  ({r['yrs']:.2f}年)")
        for gname in ("none", "sma150", "kama14"):
            v = r["variants"][gname]
            cs = v["central_summary"]
            d = v["diag"]
            print(f"\n  --- {hdr_names[gname]} --- "
                  f"(生候補{d['n_entry_signals_total']}件, ゲートで見送り{d['n_entries_blocked_by_gate']}件, "
                  f"建玉{d['n_entries_taken']}件)")
            if cs is None:
                print("    トレード無し")
                continue
            print(f"  {'cost':>6}{'slip':>6}{'n/年':>8}{'win%':>7}{'PF%':>8}{'PF$':>8}{'mean%':>8}"
                  f"{'中央値%':>9}{'std%':>7}{'年率%':>9}{'maxDD%':>8}{'年率/DD':>9}{'連敗':>5}{'平均本数':>9}")
            for c, s in ((0.0, 0.0), (0.3, 0.1), (0.6, 0.3)):
                g = v["grid"][(c, s)]
                if g is None:
                    continue
                ratio = g["tot_pct_per_year"] / g["maxdd_pct"] if g["maxdd_pct"] > 0 else float("nan")
                print(f"  {c:>6.1f}{s:>6.1f}{g['n_per_year']:>8.1f}{g['win_pct']:>7.1f}"
                      f"{g['pf_pct']:>8.3f}{g['pf']:>8.3f}{g['mean_pct']:>8.3f}"
                      f"{g['median_pct']:>9.3f}{g['std_pct']:>7.3f}"
                      f"{g['tot_pct_per_year']:>9.2f}{g['maxdd_pct']:>8.2f}{ratio:>9.2f}"
                      f"{g['max_losing_streak']:>5d}{g['avg_bars_held']:>9.1f}")
            # 参考: 完全グリッド9点も出す（$建て併記）
            print(f"    参考フルグリッド(9点, $建て併記):")
            for c in costs:
                for s in slips:
                    g = v["grid"][(c, s)]
                    if g is None:
                        continue
                    print(f"      cost={c:.1f} slip={s:.1f}: PF%={g['pf_pct']:.3f} n/年={g['n_per_year']:.1f} "
                          f"年率%={g['tot_pct_per_year']:.2f} maxDD%={g['maxdd_pct']:.2f} "
                          f"| 参考$建て PF$={g['pf']:.3f} meanR$={g['mean_dollar']:.4f} totR$={g['tot_dollar']:.2f} "
                          f"maxDD$={g['maxdd_dollar']:.2f}")

    # ---------------- 出力2: ゲートの合否 ----------------
    print("\n" + "=" * 100)
    print("出力2: ゲートの合否")
    print("=" * 100)

    for tf in ("h1", "4h"):
        r = results[tf]
        df = r["df"]
        none_central = r["variants"]["none"]["central"]
        date_index = pd.date_range(df.index[0].normalize(), df.index[-1].normalize(), freq="D")
        daily_none = daily_pnl_series(none_central, date_index)

        for gname in ("sma150", "kama14"):
            v = r["variants"][gname]
            cs = v["central_summary"]
            if cs is None:
                print(f"\n  {tf}/{gname}: トレード無し（スキップ）")
                continue
            print(f"\n  --- {tf} / {hdr_names[gname]} (cost=0.3/slip=0.1) ---")
            ratio_gate = cs["tot_pct_per_year"] / cs["maxdd_pct"] if cs["maxdd_pct"] > 0 else float("nan")
            print(f"    実測: n={cs['n']} PF%={cs['pf_pct']:.4f} (参考PF$={cs['pf']:.4f}) "
                  f"年率%={cs['tot_pct_per_year']:.2f} "
                  f"maxDD%={cs['maxdd_pct']:.2f} 年率/DD={ratio_gate:.3f}")

            # 同数ランダム間引きの帰無
            pf_null, ratio_null = subsample_null(none_central, cs["n"], N_NULL_REPS, rng)
            if len(pf_null):
                print(f"    [同数間引き帰無 n={N_NULL_REPS}] PF%: 中央値={np.median(pf_null):.4f} "
                      f"std={pf_null.std(ddof=1):.4f} 実測分位={pct_rank(cs['pf_pct'], pf_null):.1f}%ile")
                print(f"                              年率/DD: 中央値={np.nanmedian(ratio_null):.3f} "
                      f"std={np.nanstd(ratio_null, ddof=1):.3f} "
                      f"実測分位={pct_rank(ratio_gate, ratio_null):.1f}%ile")
            else:
                print("    [同数間引き帰無] 対象数不足でスキップ")

            # 巡回ブロック・ブートストラップ
            daily_gate = daily_pnl_series(v["central"], date_index)
            print(f"    [巡回ブロックブートストラップ n={N_NULL_REPS}/ブロック長] "
                  f"差=(ゲート版年率/DD − 無ゲート版年率/DD)")
            for bm in BLOCK_MONTHS:
                diffs = block_bootstrap_diff(daily_none, daily_gate, bm, N_NULL_REPS, rng)
                if len(diffs) == 0:
                    print(f"      ブロック{bm:>2d}か月: 有効サンプル無し")
                    continue
                pos_frac = float((diffs > 0).mean() * 100)
                print(f"      ブロック{bm:>2d}か月: 差>0の割合={pos_frac:5.1f}%  "
                      f"中央値={np.median(diffs):7.3f} std={diffs.std(ddof=1):7.3f} "
                      f"(有効{len(diffs)}/{N_NULL_REPS})")

            # 年別ゲートON%
            garr = v["gate_arr"]
            gate_s = pd.Series(garr, index=df.index)
            on_daily = gate_s.resample("1D").first().dropna()
            on_by_year = on_daily.groupby(on_daily.index.year).mean() * 100
            print(f"    [年別ゲートON%] " + " / ".join(f"{y}:{p:.0f}%" for y, p in on_by_year.items()))

            # 前半/後半
            h0, h1s = v["half"]
            if h0 and h1s:
                print(f"    [前半/後半] 前半: n={h0['n']} PF%={h0['pf_pct']:.3f} 年率%={h0['tot_pct_per_year']:.2f}"
                      f" / 後半: n={h1s['n']} PF%={h1s['pf_pct']:.3f} 年率%={h1s['tot_pct_per_year']:.2f}")
            else:
                print("    [前半/後半] トレード数不足でスキップ")

            # 年別 本数・PF%・年率%
            print(f"    [年別内訳]")
            for row in v["yearly"]:
                print(f"      {row['year']}: n={row['n']:>3d} PF%={row['pf_pct']:>7.3f} "
                      f"win%={row['win_pct']:>5.1f} 年率寄与%={row['tot_pct']:>7.2f}")

        # 無ゲート版自身の前半/後半・年別も参考に出す
        v0 = r["variants"]["none"]
        h0, h1s = v0["half"]
        print(f"\n  --- {tf} / (a)無ゲート 前半/後半・年別(参考) ---")
        if h0 and h1s:
            print(f"    [前半/後半] 前半: n={h0['n']} PF%={h0['pf_pct']:.3f} 年率%={h0['tot_pct_per_year']:.2f}"
                  f" / 後半: n={h1s['n']} PF%={h1s['pf_pct']:.3f} 年率%={h1s['tot_pct_per_year']:.2f}")
        for row in v0["yearly"]:
            print(f"      {row['year']}: n={row['n']:>3d} PF%={row['pf_pct']:>7.3f} "
                  f"win%={row['win_pct']:>5.1f} 年率寄与%={row['tot_pct']:>7.2f}")

    # ---------------- 出力3: 出口の解剖 ----------------
    print("\n" + "=" * 100)
    print("出力3: 出口の解剖 (中心コスト0.3/0.1、ゲート(b)(c)のうち良かったほう + 無ゲート)")
    print("=" * 100)

    for tf in ("h1", "4h"):
        r = results[tf]
        df = r["df"]
        atr_arr = atr14_causal(df).values
        cs_sma = r["variants"]["sma150"]["central_summary"]
        cs_kama = r["variants"]["kama14"]["central_summary"]

        def ratio_of(cs):
            if cs is None or cs["maxdd_pct"] <= 0:
                return -np.inf
            return cs["tot_pct_per_year"] / cs["maxdd_pct"]

        better_name = "sma150" if ratio_of(cs_sma) >= ratio_of(cs_kama) else "kama14"
        print(f"\n### TF={tf}  良かったほう(年率/DD基準) = {hdr_names[better_name]}"
              f"  (sma150 年率/DD={ratio_of(cs_sma):.3f} vs kama14 年率/DD={ratio_of(cs_kama):.3f})")

        for gname in ("none", better_name):
            trades_c = r["variants"][gname]["central"]
            if not trades_c:
                print(f"    {hdr_names[gname]}: トレード無し")
                continue
            dfa = exit_anatomy(trades_c, df, atr_arr, gname)
            print_anatomy_block(dfa, f"{tf} / {hdr_names[gname]}")

    print("\n" + "=" * 100)
    print("完了")
    print("=" * 100)


if __name__ == "__main__":
    main()
