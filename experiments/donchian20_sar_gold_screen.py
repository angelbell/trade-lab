"""巡行幅(MFE/MAE)の一次スクリーン — donchian20_sar_gold.py の本体より先に実行する。

対象: TradingView組み込み "Price Channel Strategy"（20本ドンチャン・常時ドテン）の忠実な機械化。
この戦略には固定出口(損切り/利確)が無く反対側の逆指値でしか閉じないため、mfe_mae.py の
--entry 5択には乗らない。research/screen.py の run_screen(name, df, entries) を直接使い、
「エントリー時点から前方N本」を窓としてMFE/MAE比を測る。Nは各TFの平均保有本数の概算
（このスクリプト自身が一度ノーコストでSARをシミュレートして概算する）。

出力: research/screens/donchian20_sar_gold_{m15,h1,4h}.json
このファイルはトレード成績の要約語を書かない「測定専用」スクリプトとして screen_gate フックの
対象にならないようにする（本体スクリプト側で SCREEN 宣言を使う）。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from src.data_loader import load_mt5_csv
from research.screen import run_screen

LENGTH = 20
START = "2018-10-01"


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


def raw_entries(df):
    """20本ドンチャンの常時ドテンをノーコストでシミュレートし、
    (entry_time, direction, entry_price_raw, bars_held) のリストを返す。
    段取りは donchian20_sar_gold.py の simulate() と同一（先読み無し・
    水準はshift(1)で確定後・始値ギャップは始値約定・同足タイは position==0の
    最初の1回のみ発生しうる — 詳細はコメント参照）。
    """
    hh_lvl = df["high"].rolling(LENGTH).max().shift(1).values
    ll_lvl = df["low"].rolling(LENGTH).min().shift(1).values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    idx = df.index

    position = 0
    entry_i = None
    entry_price = None
    out = []  # (entry_time, direction, entry_price_raw, bars_held)
    n_ties = 0

    for i in range(len(df)):
        if np.isnan(hh_lvl[i]) or np.isnan(ll_lvl[i]):
            continue
        buy_lvl, sell_lvl = hh_lvl[i], ll_lvl[i]
        buy_trig = h[i] >= buy_lvl
        sell_trig = l[i] <= sell_lvl

        def buy_fill():
            return o[i] if o[i] >= buy_lvl else buy_lvl

        def sell_fill():
            return o[i] if o[i] <= sell_lvl else sell_lvl

        if position == 0:
            if buy_trig and sell_trig:
                n_ties += 1
                chosen = "sell"  # 保守側tiebreak（flat時の規約。本体スクリプト参照）
            elif buy_trig:
                chosen = "buy"
            elif sell_trig:
                chosen = "sell"
            else:
                chosen = None
            if chosen == "buy":
                position, entry_price, entry_i = 1, buy_fill(), i
            elif chosen == "sell":
                position, entry_price, entry_i = -1, sell_fill(), i
        else:
            opposite_trig = sell_trig if position == 1 else buy_trig
            if opposite_trig:
                fp = sell_fill() if position == 1 else buy_fill()
                out.append((idx[entry_i], position, entry_price, i - entry_i))
                position, entry_price, entry_i = -position, fp, i

    return out, n_ties


def main():
    for tf, bar_min in (("m15", 15), ("h1", 60), ("4h", 240)):
        df = load_tf(tf)
        entries, n_ties = raw_entries(df)
        bars_held = np.array([e[3] for e in entries])
        avg_bars = float(np.mean(bars_held)) if len(bars_held) else 20.0
        window_min = int(round(avg_bars * bar_min))
        screen_entries = [(t, d, pe, None) for (t, d, pe, _bh) in entries]
        name = f"donchian20_sar_gold_{tf}"
        res = run_screen(name, df, screen_entries, windows=[window_min], quiet=True)
        w = res["windows"].get(str(window_min), {})
        print(f"[{tf}] n_entries={len(entries)} avg_bars_held={avg_bars:.1f} "
              f"window_min={window_min} n_ties(flat時)={n_ties}")
        print(f"[{tf}] ratio={w.get('ratio_median')} verdict={w.get('verdict')} "
              f"mfe_med%={w.get('mfe_median_pct')} mae_med%={w.get('mae_median_pct')}")


if __name__ == "__main__":
    main()
