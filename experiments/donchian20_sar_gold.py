"""TradingView組み込み "Price Channel Strategy"（20本ドンチャン・常時ドテン）の忠実な機械化。

Pine原文:
    length = 20
    hh = ta.highest(high, length)
    ll = ta.lowest(low, length)
    if (not na(close[length]))
        strategy.entry("PChLE", strategy.long,  stop=hh)
        strategy.entry("PChSE", strategy.short, stop=ll)

対象: gold m15 / h1 / 4h(h1リサンプル)、2018-10-01以降（m15の実質1H罠・h1の疎データ罠を
両方避けるため。仕様カードで指定された共通開始日）。

ロジックの平文説明:
  各確定足で直近20本(その足を含む)の高値の最大=hh, 安値の最小=llを計算する。この水準は
  「次の足から」有効な逆指値として機能する（先読み禁止・毎足置き直し）。買い逆指値(hh)か
  売り逆指値(ll)のどちらかに触れたら、常にドテン(反対建玉を閉じて同じ約定価格で新規建て)する。
  固定の損切り・利確は無く、出口は常に反対側の逆指値のみ＝口座は常にポジションを持つ。
  20本の助走が完了するまで発注しない。

執行モデル:
  - 約定価格は水準ちょうど（滑り無し版）。始値が既に水準を超えていれば始値で約定（ギャップ）。
  - ポジションを持っている間、意味を持つ注文は「現在の建玉と反対方向」の逆指値だけである
    （同方向の注文は既に同方向のポジションを持っているため実質ノーオペ＝TradingViewの
    デフォルトpyramiding=1により同方向の追加建玉はブロックされる）。したがって
    「同一足でhh・llの両方に触れた」場合の保守的タイブレークが実際に出力へ影響しうるのは
    ポジションが flat（最初のトレード）のときだけである。この構造的事実は本体のコードで
    診断カウンタとして検証し、レポートに実測値を出す。
  - コストは往復 $cost/oz を各トレードの決済時に一括控除。滑りは片道 $slip/oz とし、
    建玉のエントリー・エグジットそれぞれの約定で不利方向に効かせる
    （買い約定は+slip、売り約定は-slip。1トレードにつき往復で2回分＝2*slip）。
  - 水準(hh/ll)は当日以降の価格経路とは無関係に過去の高安だけで決まるため、コスト・滑りの
    設定を変えてもトレードの発生タイミング・方向系列は変化しない。よって生シミュレーションは
    1回だけ行い、コスト・滑りは後段でトレードごとの価格に加算する後処理として適用する。

事前スクリーン: donchian20_sar_gold_screen.py が research/screens/donchian20_sar_gold_{tf}.json
を作成済み（巡行幅比 m15=0.94死・h1=0.98死・4h=1.00境界。詳細はレポート本文）。
"""
SCREEN = "donchian20_sar_gold_h1"

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv

LENGTH = 20
START = "2018-10-01"
N_NULL_REPS = 1000
RNG_SEED = 20260729

TF_BAR_MIN = {"m15": 15, "h1": 60, "4h": 240}


def resample_4h(df):
    o = {
        "open": df["open"].resample("4h").first(),
        "high": df["high"].resample("4h").max(),
        "low": df["low"].resample("4h").min(),
        "close": df["close"].resample("4h").last(),
    }
    return pd.DataFrame(o).dropna()


def load_tf(tf):
    if tf == "m15":
        df = load_mt5_csv("data/vantage_xauusd_m15.csv")
    elif tf == "h1":
        df = load_mt5_csv("data/vantage_xauusd_h1.csv")
    elif tf == "4h":
        df = load_mt5_csv("data/vantage_xauusd_h1.csv")
        df = resample_4h(df)
    else:
        raise ValueError(tf)
    return df.loc[START:]


def simulate(df, tiebreak_flat="unfavorable"):
    """ノーコストの生シミュレーション。

    戻り値:
      trades: list of dict(entry_time, exit_time, direction, entry_raw, exit_raw,
                            bars_held, both_touched_on_exit)
      open_trade: 期末に未決済のまま残ったポジション（統計には含めない）か None
      diag: dict(n_ties_flat, n_both_touched_bars, n_live_bars)
    """
    hh_lvl = df["high"].rolling(LENGTH).max().shift(1).values
    ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    n = len(df)

    position = 0
    entry_i = None
    entry_price = None
    trades = []
    n_ties_flat = 0
    n_both_touched_bars = 0
    n_live_bars = 0

    def buy_fill(i, lvl):
        return o[i] if o[i] >= lvl else lvl

    def sell_fill(i, lvl):
        return o[i] if o[i] <= lvl else lvl

    for i in range(n):
        if np.isnan(hh_lvl[i]) or np.isnan(ll_lvl[i]):
            continue
        n_live_bars += 1
        buy_lvl, sell_lvl = hh_lvl[i], ll_lvl[i]
        buy_trig = h[i] >= buy_lvl
        sell_trig = l[i] <= sell_lvl
        if buy_trig and sell_trig:
            n_both_touched_bars += 1

        if position == 0:
            if buy_trig and sell_trig:
                n_ties_flat += 1
                chosen = "sell" if tiebreak_flat == "unfavorable" else "buy"
            elif buy_trig:
                chosen = "buy"
            elif sell_trig:
                chosen = "sell"
            else:
                chosen = None
            if chosen == "buy":
                position, entry_price, entry_i = 1, buy_fill(i, buy_lvl), i
            elif chosen == "sell":
                position, entry_price, entry_i = -1, sell_fill(i, sell_lvl), i
        else:
            opposite_trig = sell_trig if position == 1 else buy_trig
            if opposite_trig:
                fp = sell_fill(i, sell_lvl) if position == 1 else buy_fill(i, buy_lvl)
                trades.append(dict(
                    entry_time=idx[entry_i], exit_time=idx[i], direction=position,
                    entry_raw=entry_price, exit_raw=fp, bars_held=i - entry_i,
                    both_touched_on_exit=bool(buy_trig and sell_trig),
                ))
                position, entry_price, entry_i = -position, fp, i

    open_trade = None
    if position != 0:
        open_trade = dict(entry_time=idx[entry_i], direction=position,
                           entry_raw=entry_price, bars_since=n - 1 - entry_i,
                           last_close=df["close"].iloc[-1])

    diag = dict(n_ties_flat=n_ties_flat, n_both_touched_bars=n_both_touched_bars,
                n_live_bars=n_live_bars)
    return trades, open_trade, diag


def apply_cost(trades, cost_rt, slip_side):
    """トレードごとに cost_rt($往復)・slip_side($片道)を適用しpnlを計算する。"""
    out = []
    for t in trades:
        raw_pnl = (t["exit_raw"] - t["entry_raw"]) * t["direction"]
        net_pnl = raw_pnl - 2 * slip_side - cost_rt
        out.append(dict(t, pnl_dollar=net_pnl, pnl_pct=net_pnl / t["entry_raw"] * 100))
    return out


def years_span(df):
    return (df.index[-1] - df.index[0]).days / 365.25


def max_drawdown(cum_series):
    """累積系列(list/array)からトレード解像度の最大DDを返す（ピーク−谷、非負値）。"""
    arr = np.asarray(cum_series, dtype=float)
    if len(arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = peak - arr
    return float(dd.max())


def max_losing_streak(pnls):
    m = cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return m


def summarize(trades_c, yrs):
    n = len(trades_c)
    if n == 0:
        return None
    pnl_d = np.array([t["pnl_dollar"] for t in trades_c])
    pnl_p = np.array([t["pnl_pct"] for t in trades_c])
    wins = pnl_d > 0
    gross_win = pnl_d[pnl_d > 0].sum()
    gross_loss = -pnl_d[pnl_d < 0].sum()
    pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    cum_d = np.cumsum(pnl_d)
    cum_p = np.cumsum(pnl_p)
    return dict(
        n=n, n_per_year=n / yrs, win_pct=float(wins.mean() * 100), pf=pf,
        mean_dollar=float(pnl_d.mean()), mean_pct=float(pnl_p.mean()),
        median_dollar=float(np.median(pnl_d)), median_pct=float(np.median(pnl_p)),
        std_dollar=float(pnl_d.std(ddof=1)) if n > 1 else 0.0,
        std_pct=float(pnl_p.std(ddof=1)) if n > 1 else 0.0,
        tot_dollar=float(pnl_d.sum()), tot_pct=float(pnl_p.sum()),
        tot_pct_per_year=float(pnl_p.sum() / yrs),
        maxdd_dollar=max_drawdown(cum_d), maxdd_pct=max_drawdown(cum_p),
        avg_bars_held=float(np.mean([t["bars_held"] for t in trades_c])),
        max_losing_streak=max_losing_streak(pnl_d),
    )


def year_breakdown(trades_c):
    df = pd.DataFrame(trades_c)
    df["year"] = pd.DatetimeIndex(df["exit_time"]).year
    rows = []
    for y, g in df.groupby("year"):
        pnl_d = g["pnl_dollar"].values
        gw, gl = pnl_d[pnl_d > 0].sum(), -pnl_d[pnl_d < 0].sum()
        pf = float(gw / gl) if gl > 0 else float("inf")
        rows.append(dict(year=int(y), n=len(g), pf=pf,
                          tot_dollar=float(pnl_d.sum()),
                          win_pct=float((pnl_d > 0).mean() * 100)))
    return rows


def side_breakdown(trades_c):
    out = {}
    for name, sign in (("long", 1), ("short", -1)):
        sub = [t for t in trades_c if t["direction"] == sign]
        if not sub:
            out[name] = None
            continue
        pnl_d = np.array([t["pnl_dollar"] for t in sub])
        gw, gl = pnl_d[pnl_d > 0].sum(), -pnl_d[pnl_d < 0].sum()
        pf = float(gw / gl) if gl > 0 else float("inf")
        out[name] = dict(n=len(sub), pf=pf, win_pct=float((pnl_d > 0).mean() * 100),
                          tot_dollar=float(pnl_d.sum()), mean_dollar=float(pnl_d.mean()))
    return out


def null_test(df, n_trades, bars_held_pool, cost_rt, slip_side, n_reps, rng):
    """同数・同じ保有本数プールからのブートストラップで時点・方向ランダムな建てを1000回。"""
    close = df["close"].values
    n_bars = len(close)
    max_bh = int(max(bars_held_pool)) if len(bars_held_pool) else 1
    valid_start_hi = n_bars - max_bh - 1
    pf_list, tot_list = [], []
    for _ in range(n_reps):
        bh = rng.choice(bars_held_pool, size=n_trades, replace=True)
        starts = rng.integers(0, max(valid_start_hi, 1), size=n_trades)
        starts = np.minimum(starts, n_bars - bh - 1)
        starts = np.maximum(starts, 0)
        ends = starts + bh
        directions = rng.choice([1, -1], size=n_trades)
        entry_px = close[starts]
        exit_px = close[ends]
        raw = (exit_px - entry_px) * directions
        net = raw - 2 * slip_side - cost_rt
        gw, gl = net[net > 0].sum(), -net[net < 0].sum()
        pf = float(gw / gl) if gl > 0 else float("inf")
        pf_list.append(pf)
        tot_list.append(float(net.sum()))
    return np.array(pf_list), np.array(tot_list)


def pct_rank(value, dist):
    dist = np.asarray(dist)
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0 or not np.isfinite(value):
        return float("nan")
    return float((dist < value).mean() * 100)


def main():
    rng = np.random.default_rng(RNG_SEED)
    costs = [0.0, 0.3, 0.6]
    slips = [0.0, 0.1, 0.3]

    all_results = {}
    for tf in ("m15", "h1", "4h"):
        df = load_tf(tf)
        trades_raw, open_trade, diag = simulate(df, tiebreak_flat="unfavorable")
        trades_raw_fav, _, diag_fav = simulate(df, tiebreak_flat="favorable")
        yrs = years_span(df)

        both_touched_trades = sum(1 for t in trades_raw if t["both_touched_on_exit"])
        n_trades_raw = len(trades_raw)

        grid = {}
        for c in costs:
            for s in slips:
                tc = apply_cost(trades_raw, c, s)
                grid[(c, s)] = summarize(tc, yrs)

        # central combo central=(0.3,0.1) 用の詳細(年別・side別・null)
        central = apply_cost(trades_raw, 0.3, 0.1)
        yearly = year_breakdown(central)
        sides = side_breakdown(central)
        bars_pool = np.array([t["bars_held"] for t in trades_raw])
        pf_null, tot_null = null_test(df, len(central), bars_pool, 0.3, 0.1, N_NULL_REPS, rng)
        central_summary = summarize(central, yrs)

        # 有利側優先 tiebreak の central PF（1行比較用）
        central_fav = apply_cost(trades_raw_fav, 0.3, 0.1)
        central_fav_summary = summarize(central_fav, yrs)

        all_results[tf] = dict(
            df=df, yrs=yrs, trades_raw=trades_raw, open_trade=open_trade, diag=diag,
            grid=grid, central=central, central_summary=central_summary,
            central_fav_summary=central_fav_summary,
            yearly=yearly, sides=sides, pf_null=pf_null, tot_null=tot_null,
            both_touched_trades=both_touched_trades, n_trades_raw=n_trades_raw,
        )

    # ---------------- 出力 ----------------
    print("=" * 100)
    print("Donchian(20) 常時ドテン (Price Channel Strategy) — gold m15/h1/4h, "
          f"{START}以降")
    print("=" * 100)

    for tf in ("m15", "h1", "4h"):
        r = all_results[tf]
        print(f"\n### TF={tf}  期間={r['df'].index[0]}~{r['df'].index[-1]} "
              f"({r['yrs']:.2f}年)  生トレード数={r['n_trades_raw']}")
        ot = r["open_trade"]
        if ot:
            unreal = (ot["last_close"] - ot["entry_raw"]) * ot["direction"]
            print(f"  期末未決済: dir={ot['direction']} entry={ot['entry_raw']:.3f} "
                  f"経過{ot['bars_since']}本 含み損益(生)=${unreal:.3f}/oz（統計には非算入）")

        d = r["diag"]
        pct_both_bars = d["n_both_touched_bars"] / d["n_live_bars"] * 100
        pct_both_trades = r["both_touched_trades"] / r["n_trades_raw"] * 100 if r["n_trades_raw"] else 0.0
        print(f"  同一足でhh・llの両方に触れたバー: {d['n_both_touched_bars']}/{d['n_live_bars']}本"
              f"({pct_both_bars:.2f}%)。うち実際の建玉決済足で発生: "
              f"{r['both_touched_trades']}/{r['n_trades_raw']}トレード({pct_both_trades:.2f}%)")
        print(f"  flat状態での真のタイ(両水準が初回同時発火)={d['n_ties_flat']}件"
              f"（ポジション保有中は反対方向の注文しか意味を持たないため、"
              f"タイブレークが結果に影響しうるのは理論上ここだけ）")
        cs = r["central_summary"]; cfs = r["central_fav_summary"]
        same = (cs is not None and cfs is not None and
                abs(cs["tot_dollar"] - cfs["tot_dollar"]) < 1e-9)
        print(f"  参考: 有利側優先tiebreak版のPF(cost0.3/slip0.1)={cfs['pf']:.4f} "
              f"vs 不利側優先(保守)={cs['pf']:.4f} "
              f"({'完全一致(タイ0件のため無影響)' if same else '差あり'})")

        print(f"\n  --- コスト×滑り グリッド (往復$/oz × 片道$/oz) ---")
        header = f"  {'cost':>6}{'slip':>6}{'n/年':>8}{'win%':>7}{'PF':>8}{'meanR$':>9}{'meanR%':>8}" \
                 f"{'中央値$':>9}{'中央値%':>8}{'std$':>8}{'std%':>7}{'totR/年%':>10}{'maxDD$':>9}{'maxDD%':>8}{'連敗':>5}"
        print(header)
        for c in costs:
            for s in slips:
                g = r["grid"][(c, s)]
                if g is None:
                    continue
                print(f"  {c:>6.1f}{s:>6.1f}{g['n_per_year']:>8.1f}{g['win_pct']:>7.1f}"
                      f"{g['pf']:>8.3f}{g['mean_dollar']:>9.3f}{g['mean_pct']:>8.3f}"
                      f"{g['median_dollar']:>9.3f}{g['median_pct']:>8.3f}"
                      f"{g['std_dollar']:>8.3f}{g['std_pct']:>7.3f}"
                      f"{g['tot_pct_per_year']:>10.3f}{g['maxdd_dollar']:>9.2f}"
                      f"{g['maxdd_pct']:>8.2f}{g['max_losing_streak']:>5d}")

        print(f"\n  --- 年別 (cost=$0.3/slip=$0.1) ---")
        for row in r["yearly"]:
            print(f"    {row['year']}: n={row['n']:>4d} PF={row['pf']:>7.3f} "
                  f"win%={row['win_pct']:>5.1f} totR=${row['tot_dollar']:>9.2f}")

        print(f"\n  --- ロング/ショート内訳 (cost=$0.3/slip=$0.1) ---")
        for name in ("long", "short"):
            s = r["sides"][name]
            if s is None:
                print(f"    {name}: トレード無し")
                continue
            print(f"    {name}: n={s['n']:>4d} PF={s['pf']:>7.3f} win%={s['win_pct']:>5.1f} "
                  f"totR=${s['tot_dollar']:>9.2f} meanR=${s['mean_dollar']:>7.4f}")

        print(f"\n  --- 帰無検定 (同数={cs['n']}・同じ保有本数プール・時点/方向ランダム・"
              f"cost=$0.3/slip=$0.1・{N_NULL_REPS}回) ---")
        pf_null, tot_null = r["pf_null"], r["tot_null"]
        pf_finite = pf_null[np.isfinite(pf_null)]
        print(f"    帰無PF: 中央値={np.median(pf_finite):.4f} std={pf_finite.std(ddof=1):.4f} "
              f"実測PF={cs['pf']:.4f}の分位={pct_rank(cs['pf'], pf_null):.1f}%ile")
        print(f"    帰無totR$: 中央値={np.median(tot_null):.2f} std={tot_null.std(ddof=1):.2f} "
              f"実測totR$={cs['tot_dollar']:.2f}の分位={pct_rank(cs['tot_dollar'], tot_null):.1f}%ile")

    print("\n" + "=" * 100)
    print("巡行幅スクリーン結果（別スクリプトdonchian20_sar_gold_screen.pyで測定済み）:")
    print("  m15: ratio=0.935(死) h1: ratio=0.981(死) 4h: ratio=1.000(境界)")
    print("=" * 100)


if __name__ == "__main__":
    main()
