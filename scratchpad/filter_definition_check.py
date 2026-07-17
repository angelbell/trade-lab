"""Which variable is the filter actually cutting on?

btc15m_L carries a PDH soft-size rule: a break that happens INSIDE the previous day's range is taken
at HALF size. In R terms that halves both the win and the loss, so the price distance that costs one
full account-R is DOUBLE the structural stop. The variable I swept was
        risk_per_R / price   =   (stop_distance / w) / price          [w = 1.0 or 0.5]
which quietly doubles the reading for every soft-sized trade -- so "stop > 2% of price" was also
cutting soft-sized trades whose real stop was only 1%.

That is not the rule I am about to describe to the user. Separate them:
  A. TRUE stop/price      = stop_distance / price          <- the clean, chart-readable rule
  B. risk-per-R / price   = (stop_distance / w) / price    <- what I actually measured
  C. just drop the soft-sized trades entirely              <- the confound, on its own
If only B works, the "filter" is entangled with the PDH rule and must be described as such.
Run: .venv/bin/python scratchpad/filter_definition_check.py
"""
import sys, io, contextlib, warnings; warnings.filterwarnings("ignore")
from types import SimpleNamespace
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from radar_gate_race import BASE
from wide_stop_stress import raw_legs, SIX
from book_spec_fix import w_trade

ROOT = "/home/angelbell/dev/auto-trade"


def book_of(legs):
    w = w_trade(legs, SIX)
    st = max(legs[k].index.min() for k in SIX)
    en = min(legs[k].index.max() for k in SIX)
    s = pd.concat([pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                             index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
                   for k in SIX]).sort_index()
    eq = np.cumprod(1 + s.values); pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    cagr = (eq[-1] ** (365.25 / max((s.index[-1] - s.index[0]).days, 1)) - 1) * 100
    return cagr / max(dd, 1e-9)


def main():
    L = raw_legs()
    legs0 = {k: v[0] for k, v in L.items()}
    r0 = book_of(legs0)
    with contextlib.redirect_stderr(io.StringIO()):
        d15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_btcusd_m15.csv").loc["2018-10-01":], "15min")
        t = run(d15, SimpleNamespace(**{**BASE, "gate_kama": 14, "gate_kama_tf": "240min",
                                        "pullback_frac": 0.3, "rr": 4.5, "fill_win": 200}))
    pdh = d15["high"].resample("1D").max().dropna().shift(1).reindex(d15.index, method="ffill").values
    w = np.where(t["e_px"].values > pdh[d15.index.get_indexer(t["time"])], 1.0, 0.5)
    soft = w < 1.0                                     # 前日高値の中でのブレイク = 半分サイズ
    R = L["btc15m_L"][0]
    A = 100 * t["risk"].values / t["e_px"].values      # 素直な 損切り幅 / 価格
    B = 100 * (t["risk"].values / w) / t["e_px"].values  # 私が掃引した変数（1R当たりの値幅 / 価格）
    pf = lambda x: x[x > 0].sum() / abs(x[x <= 0].sum())

    print(f"現行ブック = {r0:.2f}   btc15m_L n={len(R)}  うち PDHソフト（半分サイズ）= "
          f"{soft.sum()} 本 ({100*soft.mean():.0f}%)\n")
    print(f"  A. 素直な 損切り/価格:      中央値 {np.median(A):.2f}%  90%点 {np.percentile(A,90):.2f}%")
    print(f"  B. 1R当たりの値幅/価格:     中央値 {np.median(B):.2f}%  90%点 {np.percentile(B,90):.2f}%")
    print(f"  → 2%閾値で切ると: A では {(A>2.0).sum()}本, B では {(B>2.0).sum()}本 が対象")
    print(f"     B で切られる {(B>2.0).sum()}本 のうち **PDHソフトは {(soft & (B>2.0)).sum()}本 "
          f"({100*(soft & (B>2.0)).mean() / max((B>2.0).mean(),1e-9):.0f}%)**\n")

    print("  どの定義が効くのか（ブックで裁定）")
    print(f"  {'ルール':<44}{'n':>6}{'年本数':>8}{'PF':>7}{'totR/年':>9}{'ブック':>9}{'差':>8}")
    yrs = (R.index[-1] - R.index[0]).days / 365.25
    print(f"  {'全部（現行）':<44}{len(R):>6}{len(R)/yrs:>8.0f}{pf(R.values):>7.2f}"
          f"{R.sum()/yrs:>+9.1f}{r0:>9.2f}{0.0:>+8.2f}")
    arms = {}
    for thr in (1.5, 2.0, 2.5, 3.0):
        arms[f"A. 損切り/価格 <= {thr}%（素直な定義）"] = A <= thr
    for thr in (1.5, 2.0, 2.5, 3.0):
        arms[f"B. 1R当たりの値幅/価格 <= {thr}%（掃引した定義）"] = B <= thr
    arms["C. PDHソフトのトレードを全部捨てる"] = ~soft
    for tag, m in arms.items():
        lg = dict(legs0); lg["btc15m_L"] = R[m]
        rb = book_of(lg)
        print(f"  {tag:<44}{m.sum():>6}{m.sum()/yrs:>8.0f}{pf(R.values[m]):>7.2f}"
              f"{R.values[m].sum()/yrs:>+9.1f}{rb:>9.2f}{rb-r0:>+8.2f}"
              + ("  ★" if rb > r0 + 0.05 else ""))

    print("\n  A を PDHソフト/通常 に分けて見る（混入しているなら片方だけに効くはず）")
    print(f"  {'':<30}{'n':>6}{'PF':>7}{'meanR':>9}   （損切り/価格 が 2% 超のトレードだけ）")
    big = A > 2.0
    for tag, m in (("通常サイズ", ~soft & big), ("PDHソフト（半分）", soft & big)):
        if m.sum() < 5:
            print(f"  {tag:<30}{m.sum():>6}  少なすぎ"); continue
        print(f"  {tag:<30}{m.sum():>6}{pf(R.values[m]):>7.2f}{R.values[m].mean():>+9.3f}")
    print(f"  {'':<30}{'':>6}{'':>7}{'':>9}   （2% 以下のトレードだけ）")
    for tag, m in (("通常サイズ", ~soft & ~big), ("PDHソフト（半分）", soft & ~big)):
        print(f"  {tag:<30}{m.sum():>6}{pf(R.values[m]):>7.2f}{R.values[m].mean():>+9.3f}")


if __name__ == "__main__":
    main()
