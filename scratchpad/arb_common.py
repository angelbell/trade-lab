"""Paired-bootstrap arbiter. Shared by every "is this book change real?" script.

Why this exists: the equal-maxDD comparison de-levers each arm until its bootstrapped-median maxDD
hits a target, then reads its CAGR. Done naively, every boot_dd() call draws a FRESH set of blocks,
so the bisection converges to a slightly different scale each time and the CAGR it reports carries
+/-2-3 CAGR pt of pure Monte-Carlo noise. Today's weighting comparison was reading 1-3pt differences
off that -- and it showed: the SAME equal-weight portfolio scored -1.1pt and +1.6pt in two columns of
one table, and the sigma baseline moved 41.9% -> 45.1% between two runs.

The fix is common random numbers. Draw the block layout ONCE and reuse the identical layout for every
arm and every bisection step. Then boot_dd is a deterministic function of its input stream, the
bisection is exact, and two arms are compared on the SAME resampled histories -- which also makes the
comparison paired, so the noise that remains largely cancels.

The trade stream is resampled by MONTH (circular block bootstrap), never collapsed to monthly returns
-- maxDD is always read at trade resolution (CLAUDE checklist 8).
"""
import numpy as np
import pandas as pd

BUDGET = 0.03


def cd(v, days):
    """トレード列 -> (CAGR%, maxDD%)。maxDD はトレード解像度（月次に潰さない）。"""
    eq = np.cumprod(1.0 + v)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100.0
    cagr = (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100.0
    return cagr, dd


class Boot:
    """ある1本のトレード列の『月の並び』を固定し、全アームで同じ並びを使い回す。

    月の集合と各月の本数はアームによって変わりうる（例: 1日で切ると本数が増える）。そこで
    「月ラベルの並び」を1度だけ引き、各アームは自分のトレードをその並びで連結する。こうすると
    2つのアームは *同じ歴史の並べ替え* の上で比較される（対比較）。
    """

    def __init__(self, months, nb=1000, k=3, seed=20260714):
        self.months = list(months)
        nm = len(self.months)
        nblk = int(np.ceil(nm / k))
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, nm, size=(nb, nblk))
        # 各行 = 月インデックスの並び（巡回ブロック）
        self.layout = [np.concatenate([[(b + j) % nm for j in range(k)] for b in row])
                       for row in starts]

    def dd_median(self, s):
        mk = s.index.to_period("M")
        by = {m: s.values[mk == m] for m in self.months}
        days = max((s.index[-1] - s.index[0]).days, 1)
        n = len(s)
        out = np.empty(len(self.layout))
        for i, seq in enumerate(self.layout):
            v = np.concatenate([by[self.months[j]] for j in seq])[:n]
            out[i] = cd(v, days)[1]
        return float(np.median(out))

    def ratios(self, s):
        """全ブートストラップ経路の CAGR/DD（アーム同士を対で比べるため、経路ごとに返す）。"""
        mk = s.index.to_period("M")
        by = {m: s.values[mk == m] for m in self.months}
        days = max((s.index[-1] - s.index[0]).days, 1)
        n = len(s)
        out = np.empty(len(self.layout))
        for i, seq in enumerate(self.layout):
            v = np.concatenate([by[self.months[j]] for j in seq])[:n]
            c, d = cd(v, days)
            out[i] = c / max(d, 1e-9)
        return out

    def equal_dd_cagr(self, s, D0, iters=24):
        """maxDD（ブートストラップ中央値）が D0 になるまで倍率を振り、その時の実測 CAGR。"""
        lo, hi = 0.05, 6.0
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            if self.dd_median(s * m) > D0:
                hi = m
            else:
                lo = m
        return cd((s * lo).values, max((s.index[-1] - s.index[0]).days, 1))[0], lo


def months_union(*streams):
    """複数アームの月ラベルの和集合。アームごとに月の集合が違っても（1日で切ると本数が増える等）、
    同じ台紙の上で並べ替えられるようにする。トレードが1本も無い月は空として連結される。"""
    m = set()
    for s in streams:
        m |= set(s.index.to_period("M"))
    return sorted(m)
