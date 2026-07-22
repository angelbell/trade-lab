"""A4: all-major-symbol cost prescreen -- rt cost / median 15m ATR, no backtest needed.
Spread = last tick (Saturday: FX/metals frozen at Friday close = indicative; crypto live).
Commission approx: FX $6/lot rt (0.6 pip USD-quote / ~0.9 JPY-quote), gold $0.06/oz,
silver $0.0012/oz, crypto/indices/oil 0 (spread-only). Benchmarks: gold/BTC = the band
where legs survive; silver 16%+ = the measured graveyard."""
import json, urllib.request, urllib.parse, statistics, sys

URL = "http://172.17.144.1:8765"
def get(path, **kw):
    q = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in kw.items())
    with urllib.request.urlopen(f"{URL}{path}?{q}", timeout=30) as r:
        return json.load(r)

SYMS = [  # (symbol, class)
 ("EURUSD+","fx"),("GBPUSD+","fx"),("USDJPY+","fxjpy"),("AUDUSD+","fx"),("NZDUSD+","fx"),
 ("USDCAD+","fx"),("USDCHF+","fx"),("EURJPY+","fxjpy"),("GBPJPY+","fxjpy"),("AUDJPY+","fxjpy"),
 ("EURGBP+","fx"),
 ("XAUUSD+","gold"),("XAGUSD","silver"),("XPTUSD.r","plat"),("XCUUSD.crp","copper"),
 ("BTCUSD","crypto"),("ETHUSD","crypto"),
 ("NAS100.r","index"),("GER40.r","index"),("US2000.r","index"),("USOUSD","oil"),
]
COMM = {"fx": 0.00006, "fxjpy": 0.009, "gold": 0.06, "silver": 0.0012,
        "plat": 0.06, "copper": 0.0, "crypto": 0.0, "index": 0.0, "oil": 0.0}

rows = []
for sym, cls in SYMS:
    try:
        t = get("/tick", symbol=sym)
        r = get("/rates", symbol=sym, timeframe="m15", count=6000)
        bars = r["rates"]
        hs = [b["high"] for b in bars]; ls = [b["low"] for b in bars]; cs = [b["close"] for b in bars]
        trs = []
        for i in range(1, len(cs)):
            trs.append(max(hs[i]-ls[i], abs(hs[i]-cs[i-1]), abs(ls[i]-cs[i-1])))
        # ATR14 series -> median of the last ~2 months
        atr = []
        a = sum(trs[:14])/14
        for x in trs[14:]:
            a = (a*13 + x)/14
            atr.append(a)
        atr_med = statistics.median(atr[-4000:])
        cost = t["spread"] + COMM[cls]
        rows.append((sym, cls, cs[-1], t["spread"], COMM[cls], atr_med, cost/atr_med*100))
    except Exception as e:
        print(f"  {sym}: ERR {e}", file=sys.stderr)

rows.sort(key=lambda x: x[6])
print(f"{'symbol':<12}{'class':<8}{'price':>10}{'spread':>9}{'comm':>8}{'ATR15m':>9}{'cost/ATR%':>10}")
for sym, cls, px, sp, cm, am, ratio in rows:
    flag = "  <= gold帯" if ratio <= 6 else ("  (BTC帯)" if ratio <= 10 else ("  x" if ratio > 14 else ""))
    print(f"{sym:<12}{cls:<8}{px:>10.4g}{sp:>9.4g}{cm:>8.4g}{am:>9.4g}{ratio:>9.1f}%{flag}")
print("\nNOTE: 土曜のためFX/金属/指数のspreadは金曜クローズ値（広めに出やすい・参考値）。"
      "月曜にロガー実測で確定させる。crypto は live。")
