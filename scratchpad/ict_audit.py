"""ICT 再設計 — フェーズ4: 多重比較の割引（統計の締め・最後に置く）。

背骨: 試行回数は事後にしか数えられないので、ここで初めて全変種を数えて割り引く。
USDJPY discount-long はまさにここ（DSR@108=0.56）で落ちた ―― 符号頑健・全時代プラス・
ブック独立でも、試行数の運を引くと有意性が残らない ＝ 機械レッグ不可・裁量読みは可。

道具（全て CLAUDE.md チェックリスト由来。random-drop は必要条件どまりなので単独では使わない）:
  placebo_premium    : 本物の窓 vs +4/8/12h 偽窓（薄い時間帯の偽約定＝流動性の空白の指紋を暴く）
  random_drop_null   : 同じ本数をランダムに残した totR/DD 分布との %ile（運の選別機との比較）
  block_boot         : 巡回ブロック・ブートストラップ 1/3/6/12か月（別の月の並びでも totR>0 か）
  overfit_grid       : 設定グリッドの月次リターン行列 → PBO via CSCV + Deflated Sharpe（psr/sr0）
  book_corr          : 現役6脚との年別R相関（法則6: 独立なら価値／冗長なら焼き直し）

自己検査（USDJPY discount-long 監査の台帳アンカーを再現）:
    .venv/bin/python scratchpad/ict_audit.py
"""
import sys, io, itertools, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, RR_CANON, walk, sc
from ict_population import canonical_setups, trade_pool, load_prepped
from ict_gates import pd_frame, join_days, gate_discount_long
from research.overfit_audit import psr, sr0, cscv

RNG = np.random.default_rng(20260715)
POSCOLS = ["pos10", "pos20", "pos40"]
RRS = [3.0, 4.0, 5.0]
FRACS = [0.20, 0.25, 0.30]
BANDS = [0.0, 0.10, 0.20, 0.30]


def long_net_map(df, S, name, f, rr):
    sp, cost = MODEL[name]
    return {d: net for (d, net, g, risk) in walk(df, S, f, rr, BUF, sp, cost, "long")}


# ---------------------------------------------------------------------------
def placebo_premium(df, tarr, dates, name, side, span, shifts=(0, 4, 8, 12),
                    gate=None, poscol="pos10", band=0.20, f=F_CANON, rr=RR_CANON,
                    use_fvg=False, fvg_min_atr=0.0, lim_fn=None,
                    use_liq=False, liq_ns=(20, 40), tgt_fn=None):
    """本物の窓 vs 偽窓の net。gate=None は素の片側、gate="discount" は discount ロング。
    use_fvg/fvg_min_atr は項目4（MSS+FVG displacement）のablation用パススルー。
    lim_fn: walk() と同じ入口アンカー上書き（FVG近位端タップablation用、省略時は従来のf固定リトレース）。
    use_liq/liq_ns/tgt_fn: 優先3（外部流動性ターゲット）用パススルー。tgt_fn を使う時は
    use_liq=True も渡す必要がある（各shiftの窓で流動性レベルを窓内で再計算するため）。"""
    sp, cost = MODEL[name]
    P = pd_frame(df) if gate else None
    out = {}
    for sh in shifts:
        S = canonical_setups(df, tarr, dates, sh, use_fvg=use_fvg, fvg_min_atr=fvg_min_atr,
                             use_liq=use_liq, liq_ns=liq_ns)
        if gate == "discount":
            L = {d: net for (d, net, g, risk) in walk(df, S, f, rr, BUF, sp, cost, side, lim_fn=lim_fn, tgt_fn=tgt_fn)}
            J = join_days(sorted(L), P)
            tr = gate_discount_long(L, J, poscol, band)
        else:
            tr = [(d, net) for (d, net, g, risk) in walk(df, S, f, rr, BUF, sp, cost, side, lim_fn=lim_fn, tgt_fn=tgt_fn)]
        out[sh] = sc(tr)
    return out


def random_drop_null(base, k, nrep=2000):
    """base から k 本をランダムに残した totR/DD 分布（運の選別機との比較）。"""
    net = np.array([t[1] for t in base]); out = []
    for _ in range(nrep):
        x = net[np.sort(RNG.choice(len(net), k, replace=False))]; cum = np.cumsum(x)
        dd = (np.maximum.accumulate(cum) - cum).max()
        out.append(x.sum() / dd if dd > 0 else np.inf)
    return np.array(out)


def block_boot(tr, months, nrep=2000):
    """巡回ブロック・ブートストラップ: 月をブロック単位で再標本し totR>0 の割合。"""
    s = pd.Series([t[1] for t in tr], index=pd.to_datetime([t[0] for t in tr])).sort_index()
    g = [x.values for _, x in s.groupby(s.index.to_period("M"))]
    nb = max(1, len(g) // months)
    bl = [np.concatenate(g[i * months:(i + 1) * months]) for i in range(nb)
          if len(g[i * months:(i + 1) * months])]
    bl = [b for b in bl if len(b)]
    if len(bl) < 4:
        return np.nan
    return 100.0 * sum(1 for _ in range(nrep)
                       if np.concatenate([bl[i] for i in RNG.integers(0, len(bl), len(bl))]).sum() > 0) / nrep


def overfit_grid(df, S, name):
    """discount-long の設定グリッド(4band×3L×3RR×3frac=108) → 月次リターン行列 M・SR分散・flagship。"""
    P = pd_frame(df)
    cols, srs, flagship = {}, [], None
    cid = 0
    for f, rr in itertools.product(FRACS, RRS):
        L = long_net_map(df, S, name, f, rr)
        J = join_days(sorted(L), P)
        posmap = {d: J.loc[d] for d in J.index}
        for band, poscol in itertools.product(BANDS, POSCOLS):
            disc = [(d, L[d]) for d in J.index
                    if d in L and not pd.isna(posmap[d][poscol]) and posmap[d][poscol] < 0.5 - band]
            if len(disc) < 20:
                continue
            r = np.array([x[1] for x in disc])
            s = pd.Series(r, index=pd.to_datetime([x[0] for x in disc]))
            cols[f"c{cid}"] = s.groupby(s.index.to_period("M")).sum()
            srs.append(r.mean() / r.std(ddof=1))
            if abs(f - 0.25) < 1e-9 and abs(rr - 4.0) < 1e-9 and band == 0.0 and poscol == "pos10":
                flagship = disc
            cid += 1
    M = pd.concat(cols, axis=1).fillna(0.0).values
    return M, float(np.var(srs)), flagship


def book_corr(disc, rr=4.5):
    """discount-long の年別R と 現役6脚の年別R の相関。"""
    ann = pd.Series([x[1] for x in disc],
                    index=pd.to_datetime([x[0] for x in disc])).groupby(lambda t: t.year).sum()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from book_deployed_spec import build as book_build, SIX   # import 時に重い出力→抑制
        legs = book_build(200, rr)
    out = {}
    for k in SIX:
        la = legs[k].groupby(lambda t: t.year).sum()
        j, l = ann.align(la, join="inner")
        if len(j) >= 4:
            out[k] = (float(j.corr(l)), len(j))
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("フェーズ4 自己検査: USDJPY discount-long 監査の台帳アンカーを再現")
    print("  台帳: PBO=0.40(ノイズ較正0.34) / SR-tr=0.099 t=2.12 skew+1.16 / DSR@108=0.56(≪0.95) / ブック独立")
    name = "usdjpy"
    df, tarr, dates, span = load_prepped(name)
    S = canonical_setups(df, tarr, dates, 0)
    M, Vsr, flagship = overfit_grid(df, S, name)

    pbo, oos_med, ploss = cscv(M)
    pbo_n, _, _ = cscv(RNG.standard_normal((M.shape[0], M.shape[1])))
    print(f"\n1. PBO via CSCV: grid={M.shape[1]}cfg × {M.shape[0]}mo  "
          f"PBO={pbo:.2f}  IS-best平均OOS-Sharpe={oos_med:+.2f}  P(OOS損)={ploss:.2f}  "
          f"[ノイズ健全性 PBO={pbo_n:.2f}]")

    r = np.array([x[1] for x in flagship])
    p0, sr, g1, g4 = psr(r, 0.0)
    tstat = sr * np.sqrt(len(r))
    Ns = [1, 25, 50, 100, 108, 200]
    dsrs = [psr(r, sr0(N, Vsr))[0] for N in Ns]
    print(f"\n2. Deflated Sharpe (flagship=band0.0/pos10/RR4/押し目0.25): "
          f"n={len(r)} SR/tr={sr:.3f} t={tstat:.2f} skew={g1:.2f} kurt={g4:.1f} V_SR={Vsr:.4f}")
    print("   " + "  ".join(f"DSR@{N}={d:.2f}" for N, d in zip(Ns, dsrs)))

    print("\n3. ブック6脚との年別R相関（独立なら価値）:")
    for k, (c, nyr) in book_corr(flagship).items():
        print(f"   {k:14s} {c:+.2f}  ({nyr}年)")
