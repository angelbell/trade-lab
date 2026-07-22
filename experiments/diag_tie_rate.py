import sys
import numpy as np
sys.path.insert(0, ".")
from src.data_loader import load_mt5_csv
from experiments.scalp_null_bracket import entry_positions_for_tf, WINDOW_NS
import numba


@numba.njit(cache=True)
def scan_diag(open_, high, low, close, time_ns, entry_pos, T, S, side, window_ns):
    n_e = entry_pos.shape[0]
    n = open_.shape[0]
    resolved_bar0 = 0
    tie_count = 0
    resolved_count = 0
    dist_bars = np.zeros(n_e, dtype=np.int64)
    for k in range(n_e):
        pos = entry_pos[k]
        e = open_[pos]
        t_entry = time_ns[pos]
        deadline = t_entry + window_ns
        if side == 1:
            tp = e + T
            sl = e - S
        else:
            tp = e - T
            sl = e + S
        j = pos
        while j < n and time_ns[j] <= deadline:
            hh = high[j]
            ll = low[j]
            if side == 1:
                sl_t = ll <= sl
                tp_t = hh >= tp
            else:
                sl_t = hh >= sl
                tp_t = ll <= tp
            if sl_t or tp_t:
                resolved_count += 1
                dist_bars[k] = j - pos
                if j == pos:
                    resolved_bar0 += 1
                if sl_t and tp_t:
                    tie_count += 1
                break
            j += 1
    return resolved_count, resolved_bar0, tie_count, dist_bars


if __name__ == "__main__":
    df = load_mt5_csv("data/vantage_usdjpy_m5.csv")
    df5 = df.loc["1999-01-01":]
    open_ = df5["open"].values.astype(np.float64)
    high = df5["high"].values.astype(np.float64)
    low = df5["low"].values.astype(np.float64)
    close = df5["close"].values.astype(np.float64)
    time_ns = df5.index.values.astype("int64")
    entry_pos = entry_positions_for_tf(df5, "5min")
    T, RR = 0.75, 0.7
    S = T / RR
    for side, sv in [("long", 1), ("short", -1)]:
        rc, rb0, tc, db = scan_diag(open_, high, low, close, time_ns, entry_pos, T, S, sv, WINDOW_NS)
        n = len(entry_pos)
        print(f"--- side={side} T={T} S={S:.4f} ---")
        print(f"resolved: {rc}/{n} = {rc/n*100:.1f}%   (timeout {100-rc/n*100:.1f}%)")
        print(f"resolved on entry bar itself (dist=0): {rb0} = {rb0/rc*100:.2f}% of resolved")
        print(f"ties (both TP & SL touched in the resolving bar): {tc} = {tc/rc*100:.3f}% of resolved")
        nz = db[db > 0]
        print(f"median bars-to-resolution (excl dist=0): {np.median(nz):.1f}   "
              f"p90={np.percentile(nz,90):.1f}  max={nz.max()}")
