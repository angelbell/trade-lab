"""The book sizes its legs by 1/sigma(trade R). Two things are wrong with that, and they point in
opposite directions, so the net effect has to be measured rather than argued:

  1. sigma charges the RIGHT tail as risk. btc15m_L has skew +1.84 (loss floored at -1R, wins run to
     +6.8R), so any rule that amputates winners is REWARDED with a bigger weight -- the sizing rule
     pays a bounty on exactly the behaviour structural law 4 says is the most expensive thing you can
     do. Downside-only measures (semi / loss-RMS / CVaR) and path measures (ulcer) don't have this hole.
  2. path measures have their OWN hole: a leg's equity curve indexed by ITS OWN trades is short when
     the leg is slow. btc_bo_kama fires 70 times in 7.7 years, so its trade-indexed drawdown is
     mechanically shallow, and 1/ulcer hands it a huge weight. That is the same frequency bug that
     monthly-sigma had (CLAUDE law 8) wearing a different hat.

Full-sample ulcer weighting scored +10.6 CAGR pt at equal drawdown, which is a bigger lever than any
exit question. But EVERY scheme here is fitted on the whole history = lookahead. A weight you could
not have known is not a weight you can trade. So:

  IS   weights from the full sample                    (what was measured before -- lookahead)
  WF   weights recomputed each Jan 1 from PRIOR trades (what you could actually have run)

plus the controls that catch the two holes above:
  ulcer_d   ulcer on a CALENDAR (daily) equity grid -- every leg gets the same path length, so if
            ulcer's advantage is really the frequency bug, it dies here
  freq      1/(sigma * sqrt(trades per year))   -- deliberately frequency-tilted, as a contrast
  equal     flat 0.5% per leg                   -- the do-nothing null
  ulcer_p   weights PROPORTIONAL to ulcer (not inverse) = the reversed dummy; must be clearly worse

Everything is judged the only way that cannot be gamed: de-lever each scheme to the SAME bootstrapped
MEDIAN maxDD (never a single path's DD) and compare CAGR. Total risk budget 3% throughout.
Run: .venv/bin/python experiments/arbiter_walkforward.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
from rr_with_swap import leg, SIX

RNG = np.random.default_rng(20260714)
BUDGET = 0.03
SCHEMES = ["sigma", "semi", "dnrisk", "cvar", "ulcer", "ulcer_d", "freq", "equal", "ulcer_p"]


def _ulcer(eq):
    pk = np.maximum.accumulate(eq)
    return np.sqrt((((pk - eq) / pk) ** 2).mean()) * 100


def risk_of(s, how):
    """s = そのレッグの R 系列（entry time index）。小さいほど『安全』＝重みが大きくなる量を返す。"""
    v = s.values
    if len(v) < 5:
        return 1e9
    if how in ("sigma", "freq"):
        r = v.std()
        if how == "freq":
            yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
            r *= np.sqrt(len(v) / yrs)          # 頻度が高いほど『重い』＝重みを削る（対照用）
        return r
    if how == "semi":
        d = v[v < v.mean()] - v.mean()
        return np.sqrt((d ** 2).mean()) if len(d) else 1e-9
    if how == "dnrisk":
        d = v[v < 0]
        return np.sqrt((d ** 2).mean()) if len(d) else 1e-9
    if how == "cvar":
        return abs(np.mean(np.sort(v)[:max(1, int(0.10 * len(v)))]))
    if how in ("ulcer", "ulcer_p"):
        return max(_ulcer(np.cumprod(1 + v * 0.01)), 1e-9)      # トレード軸（＝頻度バイアスあり）
    if how == "ulcer_d":
        d = pd.Series(v, index=s.index).resample("1D").sum().fillna(0.0)   # 暦軸＝全レッグ同じ長さ
        return max(_ulcer(np.cumprod(1 + d.values * 0.01)), 1e-9)
    if how == "equal":
        return 1.0
    raise ValueError(how)


def weights(L, how, budget=BUDGET):
    r = pd.Series({k: risk_of(L[k], how) for k in SIX})
    raw = r if how == "ulcer_p" else 1.0 / r          # ulcer_p だけ逆向き（ダミー）
    return raw / raw.sum() * budget


def cd(v, days):
    eq = np.cumprod(1 + v); pk = np.maximum.accumulate(eq)
    return (eq[-1] ** (365.25 / max(days, 1)) - 1) * 100, ((pk - eq) / pk).max() * 100


def boot_dd(s, nb=200, k=3):
    mk = s.index.to_period("M"); months = sorted(mk.unique())
    by = {m: s.values[mk == m] for m in months}
    nm = len(months); nblk = int(np.ceil(nm / k)); days = (s.index[-1] - s.index[0]).days
    return float(np.median([cd(np.concatenate([by[months[(b + j) % nm]]
                                               for b in RNG.integers(0, nm, nblk)
                                               for j in range(k)])[:len(s)], days)[1]
                            for _ in range(nb)]))


def equal_dd_cagr(s, D0):
    lo, hi = 0.10, 5.0
    for _ in range(16):
        m = (lo + hi) / 2
        if boot_dd(s * m) > D0:
            hi = m
        else:
            lo = m
    c = cd((s * lo).values, (s.index[-1] - s.index[0]).days)[0]
    return c, lo


def main():
    B = {k: leg(k)[0] for k in SIX}
    st = max(B[k].index.min() for k in SIX); en = min(B[k].index.max() for k in SIX)
    B = {k: B[k][(B[k].index >= st) & (B[k].index <= en)] for k in SIX}
    yrs = sorted({y for k in SIX for y in B[k].index.year})

    def mix(wser_by_year):
        """年ごとの重みでレッグを混ぜ、entry time 順の1本のトレード列にする。"""
        parts = []
        for k in SIX:
            s = B[k]
            wv = np.array([wser_by_year[y][k] for y in s.index.year])
            parts.append(pd.Series(s.values * wv, index=s.index))
        return pd.concat(parts).sort_index()

    # IS = 全期間の重み（先読み）  /  WF = 各年1月1日に、その年より前のトレードだけで計算した重み
    W_IS, W_WF, first_wf = {}, {}, yrs[0] + 2
    for how in SCHEMES:
        w_all = weights(B, how)
        W_IS[how] = {y: w_all for y in yrs}
        byyear = {}
        for y in yrs:
            past = {k: B[k][B[k].index.year < y] for k in SIX}
            byyear[y] = weights(past, how) if y >= first_wf and min(len(past[k]) for k in SIX) >= 5 \
                else weights(B, "equal")            # 履歴が足りない最初の2年は等分（先読みしない）
        W_WF[how] = byyear

    print(f"レッグ（スワップ込み）: {st.date()} 〜 {en.date()}")
    print(f"  {'leg':<14}{'n':>5}{'年間':>6}{'meanR':>8}{'σ':>6}{'半偏差':>7}{'損失RMS':>8}"
          f"{'CVaR10':>8}{'ulcer(トレード軸)':>17}{'ulcer(暦軸)':>12}")
    for k in SIX:
        yn = len(B[k]) / max((en - st).days / 365.25, 1e-9)
        print(f"  {k:<14}{len(B[k]):>5}{yn:>6.0f}{B[k].mean():>+8.3f}"
              f"{risk_of(B[k],'sigma'):>6.2f}{risk_of(B[k],'semi'):>7.2f}{risk_of(B[k],'dnrisk'):>8.2f}"
              f"{risk_of(B[k],'cvar'):>8.2f}{risk_of(B[k],'ulcer'):>17.2f}{risk_of(B[k],'ulcer_d'):>12.2f}")

    D0 = boot_dd(mix(W_IS["sigma"]))
    print(f"\n基準 maxDD = {D0:.2f}%（現行σ重み・巡回ブロック3か月・中央値）")
    print(f"全ての行をこの maxDD にそろえて CAGR で比べる。WF = {first_wf}年以降だけ重みを推定"
          f"（それ以前は等分＝先読みしない）\n")
    print(f"  {'重みの決め方':<12}" + "".join(f"{k.replace('btc','b').replace('gold','g'):>9}" for k in SIX)
          + f"{'IS CAGR':>10}{'IS 差':>8}{'WF CAGR':>10}{'WF 差':>8}")

    base_is = base_wf = None
    rows = {}
    for how in SCHEMES:
        s_is, s_wf = mix(W_IS[how]), mix(W_WF[how])
        c_is, _ = equal_dd_cagr(s_is, D0)
        c_wf, _ = equal_dd_cagr(s_wf, D0)
        if base_is is None:
            base_is, base_wf = c_is, c_wf
        w = W_IS[how][yrs[-1]]
        rows[how] = (s_wf, c_wf)
        print(f"  {how:<12}" + "".join(f"{100*w[k]:>8.2f}%" for k in SIX)
              + f"{c_is:>+9.1f}%{c_is-base_is:>+7.1f}pt{c_wf:>+9.1f}%{c_wf-base_wf:>+7.1f}pt"
              + ("  ← 現行" if how == "sigma" else ""))
    print("  ※ 表示している重みは IS（全期間）のもの。WF は毎年組み替わる。")

    print("\n巡回ブロック・ブートストラップ（WF の列で、σ重みに勝つ確率。本物ならブロックを長くするほど上がる）")
    print(f"  {'重み':<12}{'1か月':>8}{'3か月':>8}{'6か月':>8}{'12か月':>8}")
    s0 = rows["sigma"][0]
    for how in SCHEMES:
        if how == "sigma":
            continue
        s1 = rows[how][0]
        out = []
        for k in (1, 3, 6, 12):
            for s, store in ((s0, "a"), (s1, "b")):
                mk = s.index.to_period("M"); months = sorted(mk.unique())
                by = {m: s.values[mk == m] for m in months}
                nm = len(months); nblk = int(np.ceil(nm / k)); days = (s.index[-1] - s.index[0]).days
                r = np.array([(lambda c, d: c / max(d, 1e-9))(
                    *cd(np.concatenate([by[months[(b + j) % nm]] for b in RNG.integers(0, nm, nblk)
                                        for j in range(k)])[:len(s)], days)) for _ in range(800)])
                if store == "a":
                    ra = r
                else:
                    rb = r
            out.append(100 * np.mean(rb > ra))
        print(f"  {how:<12}" + "".join(f"{o:>7.0f}%" for o in out))


if __name__ == "__main__":
    main()
