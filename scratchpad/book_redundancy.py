"""Are any two legs the SAME BET wearing different clothes?

The user's concern, and it is a fair one on inspection alone:
  - 4 of the 6 legs are BTC.
  - 4 of the 6 legs are the SAME detector (ZigZag Pattern-B breakout): gold_bo, btc_bo_kama,
    btc15m_L, btc15m_S. Structural law 5 already says structure-break detectors all re-derive
    gold_bo. If two legs are the same bet, the book's "diversification" is an illusion and the
    inv-vol weights are spreading risk across a single factor.

Three ways of asking the same question, because correlation alone is not enough for a book of
lumpy, sparse, non-overlapping trade streams:
  T1  RETURN correlation -- monthly and daily. The classic. But two legs that rarely trade in the
      same month can look uncorrelated purely from sparsity.
  T2  CO-TIMING -- of the days each leg trades, what fraction does the other also trade? And when
      both trade on the same day, do their R's agree in sign? Two legs firing on the same day in
      the same direction are the same bet regardless of what the monthly correlation says.
  T3  MARGINAL VALUE -- for each PAIR, does holding both beat holding the better one alone
      (trade-resolution CAGR/DD, weights re-derived on trade-level sigma at 3%)? This is the only
      question that actually matters: redundancy is only a problem if it costs the book.
Run: .venv/bin/python scratchpad/book_redundancy.py
"""
import sys, io, contextlib, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
from btc_family_ext_throttle import build_base
from book_leave_one_out import cdd

NEW = ["gold_bo", "btc_bo_kama", "btc_pull", "gold15m", "btc15m_L", "btc15m_S"]
SHORT = {"gold_bo": "gold_bo", "btc_bo_kama": "btc_bo", "btc_pull": "btc_pull",
         "gold15m": "gold15m", "btc15m_L": "b15_L", "btc15m_S": "b15_S"}


def w_trade(legs, basket, budget=0.03):
    """inv-vol on TRADE-level R sigma (the corrected weighting)."""
    sig = pd.Series({k: legs[k].std() for k in basket})
    w = (1.0 / sig); return w / w.sum() * budget


def book(legs, basket):
    w = w_trade(legs, basket)
    # common window across the basket
    st = max(legs[k].index.min() for k in basket)
    en = min(legs[k].index.max() for k in basket)
    parts = [pd.Series(legs[k][(legs[k].index >= st) & (legs[k].index <= en)].values * w[k],
                       index=legs[k][(legs[k].index >= st) & (legs[k].index <= en)].index)
             for k in basket]
    s = pd.concat(parts).sort_index()
    return cdd(s.values, (s.index[-1] - s.index[0]).days)


def main():
    with contextlib.redirect_stderr(io.StringIO()):
        legs = build_base()

    st = max(legs[k].index.min() for k in NEW)
    en = min(legs[k].index.max() for k in NEW)
    L = {k: legs[k][(legs[k].index >= st) & (legs[k].index <= en)] for k in NEW}
    print(f"common window {st.date()} -> {en.date()}   trades: "
          + "  ".join(f"{SHORT[k]}={len(L[k])}" for k in NEW))

    mon = pd.DataFrame({k: v.groupby(v.index.to_period("M")).sum() for k, v in L.items()}).fillna(0.0)
    day = pd.DataFrame({k: v.groupby(v.index.floor("D")).sum() for k, v in L.items()}).fillna(0.0)

    for tag, M in (("T1a  MONTHLY R correlation", mon), ("T1b  DAILY R correlation", day)):
        print(f"\n{tag}")
        C = M.corr()
        print("          " + "".join(f"{SHORT[k]:>9}" for k in NEW))
        for k in NEW:
            print(f"{SHORT[k]:<10}" + "".join(
                f"{C.loc[k, j]:>+9.2f}" if j != k else f"{'-':>9}" for j in NEW))

    print("\nT2  CO-TIMING -- rows: of THIS leg's trading days, what % does the column leg also trade?")
    days = {k: set(v.index.floor("D")) for k, v in L.items()}
    print("          " + "".join(f"{SHORT[k]:>9}" for k in NEW))
    for k in NEW:
        row = []
        for j in NEW:
            row.append("-" if j == k else f"{100*len(days[k] & days[j])/max(len(days[k]),1):.0f}%")
        print(f"{SHORT[k]:<10}" + "".join(f"{x:>9}" for x in row))

    print("\n     on the days BOTH trade: how often do their daily R's share a sign?  (50% = independent)")
    print(f"  {'pair':<22}{'co-days':>9}{'same sign':>11}{'corr on co-days':>17}")
    for a, b in itertools.combinations(NEW, 2):
        co = sorted(days[a] & days[b])
        if len(co) < 10:
            print(f"  {SHORT[a]+' x '+SHORT[b]:<22}{len(co):>9}{'(too few)':>11}")
            continue
        x, y = day.loc[co, a], day.loc[co, b]
        same = np.mean(np.sign(x) == np.sign(y)) * 100
        print(f"  {SHORT[a]+' x '+SHORT[b]:<22}{len(co):>9}{same:>10.0f}%{x.corr(y):>+17.2f}")

    print("\nT3  MARGINAL VALUE of each PAIR (trade-resolution CAGR/DD, trade-sigma weights, 3%)")
    print(f"  {'pair':<22}{'A alone':>9}{'B alone':>9}{'A+B':>8}{'lift vs better':>16}")
    solo = {k: book(legs, [k])[2] for k in NEW}
    for a, b in itertools.combinations(NEW, 2):
        both = book(legs, [a, b])[2]
        better = max(solo[a], solo[b])
        flag = "  <-- pair adds nothing" if both <= better else ""
        print(f"  {SHORT[a]+' x '+SHORT[b]:<22}{solo[a]:>9.2f}{solo[b]:>9.2f}{both:>8.2f}"
              f"{both - better:>+16.2f}{flag}")

    print(f"\n  full 6-leg book (trade-sigma weights): CAGR/DD = {book(legs, NEW)[2]:.2f}")


if __name__ == "__main__":
    main()
