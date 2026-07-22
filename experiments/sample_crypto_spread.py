"""Vantage デモ口座の暗号資産スプレッドを一定間隔で採取して CSV に落とす（読み取りのみ）。

拡大足レッグのコスト仮定（0.05〜0.20%）が実勢と合っているかを確かめるため。
1回の採取では時間帯の偏りが取れないので、間隔をあけて回し続ける。
出力: experiments/crypto_spread_samples.csv（追記）
"""
import sys
import os
import time
import datetime as dt

sys.path.insert(0, os.path.abspath("../mt5-mcp"))
from client.mt5api import Bridge          # noqa: E402

SYMS = ["BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD", "DOTUSD", "XRPUSD",
        "LTCUSD", "BCHUSD", "BNBUSD", "TRXUSD"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_spread_samples.csv")


def one_round(b):
    rows = []
    for s in SYMS:
        try:
            t = b.tick(s)
            mid = (t["bid"] + t["ask"]) / 2
            if mid <= 0:
                continue
            rows.append((t["time"], s, t["bid"], t["ask"], t["spread"],
                         t["spread"] / mid * 100))
        except Exception:                                   # noqa: BLE001
            continue
    return rows


if __name__ == "__main__":
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    gap = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    b = Bridge()
    for s in SYMS:                       # 気配値表示に載せる（rates 呼び出しで選択される）
        try:
            b.rates(s, "h1", count=2)
        except Exception:                                   # noqa: BLE001
            pass
    new = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if new:
            f.write("srv_time,symbol,bid,ask,spread,spread_pct\n")
        for i in range(n_rounds):
            for r in one_round(b):
                f.write(",".join(str(x) for x in r) + "\n")
            f.flush()
            if i < n_rounds - 1:
                time.sleep(gap)
    print(f"{dt.datetime.now():%H:%M:%S} 採取 {n_rounds} 回ぶんを {OUT} に追記")
