"""BTC h1 ATR拡大足レッグ: レジームゲート追加の検定（第3段・仕様凍結カード）。

前段（verify_atr_spike_units.py / verify_atr_spike_beta.py）で確定した4本の基準セル
（引き金・入口・出口は凍結）に、engine のゲート（gate_kama / gate_sma）を4種類（4h/日足 ×
KAMA/SMA）足し、ゲート無しと並べる。損益は「入口価格に対する%」（ロット0.01固定＝R では
判定しない）。帰無は「ゲートは必ずトレードを間引く」＝同じ通過率 q でランダム間引きした
場合の PF/平均% 分布に対する%ile。

本題は構造法則11の反証可能な検定: ロング×{4h,日足} と ショート×{4h,日足} の4象限を
KAMA ゲートで必ず埋め、「順方向=速いゲート／逆方向=遅いゲート」の予言が成立するか。

Run:
  .venv/bin/python scratchpad/atr_spike_btc_h1_gates.py --smoke 2>&1 | tee scratchpad/out_atr_spike_gates_smoke.txt
  .venv/bin/python scratchpad/atr_spike_btc_h1_gates.py 2>&1 | tee scratchpad/out_atr_spike_gates.txt
"""
SCREEN = "atr_spike_btc_h1"

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_loader import load_mt5_csv          # noqa: E402
from src.engine.walk import walk                  # noqa: E402
from src.engine.mirror import invert               # noqa: E402
from src.engine.gates import gate_kama, gate_sma    # noqa: E402
from breakout_wave import kama_adaptive             # noqa: E402


# ---------------------------------------------------------------- 引き金・入口・出口（凍結）

def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def trigger_idx(d, k):
    atr_prev = wilder_atr(d).shift(1).to_numpy()
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    hit = (c - o > atr_prev * k) & (c > o) & np.isfinite(atr_prev)
    return np.flatnonzero(hit)


def build_entries(d, k, rr, gate_arr=None):
    """A系: stop=拡大足の安値, entry=翌足始値, target=entry+rr*risk。gate_arr があれば
    トリガー足で True の物だけ残す（gate_kama/gate_sma は shift(1)+ffill 済み＝先読み無し）。"""
    trigs = trigger_idx(d, k)
    if gate_arr is not None:
        trigs = trigs[gate_arr[trigs]]
    o, l = d["open"].to_numpy(), d["low"].to_numpy()
    ent = []
    for s in trigs:
        if s + 1 >= len(d):
            continue
        e, stop = o[s + 1], l[s]
        if e - stop <= 0:
            continue
        ent.append((s, e, stop, e + rr * (e - stop), s))
    return ent


def run_cell(d, entries, pf, fill_win=200, fwd=20, cost=0.0005, C=None, cost_mode="pct", cost_fixed=25.0):
    if not entries:
        return None
    args = SimpleNamespace(pullback_frac=pf, fill_win=fill_win, fwd=fwd, cost=0.0,
                           max_pos=1, swap_pct=0.0, tp1_frac=0.0, exec_split=0)
    t, _ = walk(d, entries, None, args)
    if t is None or len(t) == 0:
        return None
    e_real = (C - t["e_px"]) if C is not None else t["e_px"]
    if cost_mode == "pct":
        pnl_px = t["R"] * t["risk"] - cost * e_real
    else:
        pnl_px = t["R"] * t["risk"] - cost_fixed
    t = t.assign(pnl_px=pnl_px, pnl_pct=pnl_px / e_real)
    return t


def stats(t, span):
    p = t["pnl_pct"].to_numpy()
    win, loss = p[p > 0].sum(), -p[p < 0].sum()
    eq = np.cumsum(p)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"N": len(p), "N/年": len(p) / span, "勝率": float((p > 0).mean()) * 100,
            "PF": float(win / loss) if loss > 0 else float("inf"),
            "平均%": float(p.mean() * 100), "総%": float(p.sum() * 100), "maxDD%": dd * 100}


def pf_of(p):
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return float(w / l) if l > 0 else float("nan")


def drop_null(base_t, q, obs_t, reps=200, seed=11):
    """ゲート無し母集団(base_t)から同じ通過率 q だけランダムに残す（200回）。
    ゲートの PF/平均% がその分布の何%ileかを返す。"""
    rng = np.random.default_rng(seed)
    p0 = base_t["pnl_pct"].to_numpy()
    n = max(1, int(round(q * len(p0))))
    n = min(n, len(p0))
    means, pfs = [], []
    for _ in range(reps):
        s = rng.choice(p0, size=n, replace=False)
        means.append(s.mean())
        pfs.append(pf_of(s))
    means, pfs = np.array(means), np.array(pfs)
    obs_m = obs_t["pnl_pct"].mean()
    obs_pf = pf_of(obs_t["pnl_pct"].to_numpy())
    return {"帰無PF中央値": float(np.nanmedian(pfs)), "PF%ile": float((pfs < obs_pf).mean() * 100),
            "帰無平均%中央値": float(np.nanmedian(means) * 100),
            "平均%ile": float((means < obs_m).mean() * 100)}


# ---------------------------------------------------------------- ゲート構成

def gate_configs():
    return [
        ("none", None, None),
        ("kama_4h", "kama", dict(gate_kama=14, gate_kama_tf="4h", gate_kama_tf2="")),
        ("kama_1d", "kama", dict(gate_kama=14, gate_kama_tf="1D", gate_kama_tf2="")),
        ("sma_1d150", "sma", dict(daily_sma=150, gate_tf="1D", daily_slope_k=0, ext_cap=0)),
        ("sma_4h150", "sma", dict(daily_sma=150, gate_tf="4h", daily_slope_k=0, ext_cap=0)),
    ]


def compute_gate_arrays(d):
    out = {}
    for name, kind, params in gate_configs():
        if kind is None:
            out[name] = None
        elif kind == "kama":
            out[name] = gate_kama(d, SimpleNamespace(**params))
        else:
            reg, _ = gate_sma(d, SimpleNamespace(**params))
            out[name] = reg
    return out


# ---------------------------------------------------------------- 検算: ミラー対称性

def verify_kama_mirror(df, inv, C, period=14, tf="1D"):
    dck = df["close"].resample(tf).last().dropna()
    kmg = kama_adaptive(dck, period)
    dck_inv = inv["close"].resample(tf).last().dropna()
    kmg_inv = kama_adaptive(dck_inv, period)
    mask = kmg.notna() & kmg_inv.notna()
    diff = (kmg_inv[mask] - (C - kmg[mask])).abs()
    assert diff.max() < 1e-6, f"KAMA鏡像対称性が崩れている: max diff={diff.max()}"

    krise = kmg > kmg.shift(1)
    krise_inv = kmg_inv > kmg_inv.shift(1)
    tie = np.isclose(kmg, kmg.shift(1), rtol=0, atol=1e-6)  # 浮動小数の誤差もタイ扱いで除外
    # kmg.shift(1) が NaN の初回シード行は比較不能（False扱いされてしまう）ので除外
    m2 = mask & kmg.shift(1).notna() & kmg_inv.shift(1).notna() & ~pd.Series(tie, index=kmg.index).fillna(False)
    ok = (krise[m2] == ~krise_inv[m2])
    assert ok.all(), f"反転フレームgate_kamaが実フレームの否定と一致しない: {(~ok).sum()}/{len(ok)} 件不一致"
    print(f"[検算OK] KAMA({period},{tf}) 鏡像対称性: 数値一致(誤差<1e-6, n={mask.sum()}), "
          f"符号反転一致(タイ{int(tie.sum())}件除きn={m2.sum()}, 不一致0件)")


# ---------------------------------------------------------------- 巡回ブロック・ブートストラップ

def block_bootstrap(t, k_months_list, n_boot=1000, seed=20260721):
    s = t.set_index(pd.DatetimeIndex(t["time"]))["pnl_pct"]
    months = sorted(s.index.to_period("M").unique())
    nm = len(months)
    by_month = {m: s[s.index.to_period("M") == m] for m in months}
    rng = np.random.default_rng(seed)
    out = {}
    for k_months in k_months_list:
        nblk = int(np.ceil(nm / k_months))
        wins, pfs, means = [], [], []
        for _ in range(n_boot):
            starts = rng.integers(0, nm, size=nblk)
            seq = np.concatenate([[(st + j) % nm for j in range(k_months)] for st in starts])
            samp = pd.concat([by_month[months[j]] for j in seq])
            if len(samp) < 5:
                continue
            v = samp.values
            wins.append(float((v > 0).mean() * 100))
            pfs.append(pf_of(v))
            means.append(float(v.mean() * 100))
        wins, pfs, means = np.array(wins), np.array(pfs), np.array(means)
        out[k_months] = {
            "勝率中央値": float(np.median(wins)), "勝率CI": (float(np.percentile(wins, 2.5)), float(np.percentile(wins, 97.5))),
            "PF中央値": float(np.nanmedian(pfs)), "PFCI": (float(np.nanpercentile(pfs, 2.5)), float(np.nanpercentile(pfs, 97.5))),
            "平均%中央値": float(np.median(means)), "draw": len(wins)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="部分データ(直近2年)で通し確認のみ")
    a = ap.parse_args()

    df = load_mt5_csv(os.path.join(ROOT, "data/vantage_btcusd_h1.csv"))
    if a.smoke:
        df = df.loc[df.index[-1] - pd.Timedelta(days=730):]
    span = (df.index[-1] - df.index[0]).days / 365.25
    C = 2 * df["high"].max()
    inv = invert(df)

    print(f"データ: {df.index[0]} 〜 {df.index[-1]}  span={span:.2f}年  N(bars)={len(df)}  smoke={a.smoke}")

    # ---- 検算1: ミラー対称性（KAMA、4h/1D 両方で）
    verify_kama_mirror(df, inv, C, 14, "1D")
    verify_kama_mirror(df, inv, C, 14, "4h")

    # ---- 検算2: 基準セルの再現（フルデータのみ厳密。smokeでは緩める）
    if not a.smoke:
        tl = run_cell(df, build_entries(df, 2.0, 3.0), 0.0, C=None)
        assert len(tl) == 556, f"ロング成行 k=2.0 RR3 の N が一致しない: {len(tl)}"
        assert abs(tl["pnl_pct"].mean() * 100 - 0.393) < 0.005, tl["pnl_pct"].mean() * 100
        assert abs(pf_of(tl["pnl_pct"].to_numpy()) - 1.45) < 0.02, pf_of(tl["pnl_pct"].to_numpy())
        ts = run_cell(inv, build_entries(inv, 2.0, 3.0), 0.5, C=C)
        assert len(ts) == 457, f"ショート指値0.5 k=2.0 RR3 の N が一致しない: {len(ts)}"
        assert abs(ts["pnl_pct"].mean() * 100 - 0.192) < 0.005, ts["pnl_pct"].mean() * 100
        assert abs(pf_of(ts["pnl_pct"].to_numpy()) - 1.27) < 0.02, pf_of(ts["pnl_pct"].to_numpy())
        print(f"[検算OK] 基準セル再現: ロング成行 N={len(tl)} 平均={tl['pnl_pct'].mean()*100:+.3f}% "
              f"PF={pf_of(tl['pnl_pct'].to_numpy()):.2f} / "
              f"ショート指値0.5 N={len(ts)} 平均={ts['pnl_pct'].mean()*100:+.3f}% "
              f"PF={pf_of(ts['pnl_pct'].to_numpy()):.2f}")

    gates_long = compute_gate_arrays(df)
    gates_short = compute_gate_arrays(inv)

    # ---- 検算3: ON% が 0% でも 100% でもないこと（配線ミス検出）
    for name, arr in gates_long.items():
        if arr is None:
            continue
        onp = arr.mean() * 100
        assert 0.0 < onp < 100.0, f"long {name} のON%が異常: {onp}"
    for name, arr in gates_short.items():
        if arr is None:
            continue
        onp = arr.mean() * 100
        assert 0.0 < onp < 100.0, f"short {name} のON%が異常: {onp}"
    print("[検算OK] 全ゲートのON%が0%/100%ではない（配線ミス無し）")

    # ---- 年別ゲートON%（H1バー解像度）
    print("\n=== ゲートON%（年別、H1バー基準） ===")
    for side, gates in (("long", gates_long), ("short", gates_short)):
        for name, arr in gates.items():
            if arr is None:
                continue
            s = pd.Series(arr, index=df.index)
            yr = s.groupby(s.index.year).mean() * 100
            print(f"  {side:5s} {name:10s}: " + " ".join(f"{y}:{v:4.0f}%" for y, v in yr.items()))

    # ---- メイン格子
    K_LIST = [1.5, 2.0, 2.5]
    RR_LIST = [2.0, 3.0, 4.5]
    SIDES = [("long", df, None, [0.0], gates_long), ("short", inv, C, [0.382, 0.5], gates_short)]
    GATES = gate_configs()
    COST = 0.0005

    rows = []
    print("\n=== メイン格子（コスト割合0.0005・fill_win=200・fwd=20） ===")
    header = (f"{'方向':5s} {'k':>4s} {'pf':>5s} {'RR':>5s} {'ゲート':10s} | {'ON%':>6s} {'N':>5s} "
              f"{'N/年':>6s} {'勝率':>6s} {'PF':>6s} {'平均%':>8s} {'総%':>8s} {'maxDD%':>7s} | "
              f"{'帰無PF':>7s} {'PF%ile':>7s} {'帰無平均%':>9s} {'平均%ile':>8s}")
    print(header)
    for side, d, Cx, pfs, gates in SIDES:
        for k in K_LIST:
            for rr in RR_LIST:
                for pf in pfs:
                    base_ent = build_entries(d, k, rr, None)
                    base_t = run_cell(d, base_ent, pf, C=Cx, cost=COST)
                    if base_t is None:
                        continue
                    for gname, _, _ in GATES:
                        garr = gates[gname]
                        if gname == "none":
                            t = base_t
                            onp = 100.0
                            nl = {"帰無PF中央値": float("nan"), "PF%ile": float("nan"),
                                  "帰無平均%中央値": float("nan"), "平均%ile": float("nan")}
                        else:
                            ent = build_entries(d, k, rr, garr)
                            t = run_cell(d, ent, pf, C=Cx, cost=COST)
                            if t is None or len(t) == 0:
                                continue
                            onp = len(t) / len(base_t) * 100
                            q = len(t) / len(base_t)
                            nl = drop_null(base_t, q, t)
                        s = stats(t, span)
                        row = dict(side=side, k=k, pf=pf, rr=rr, gate=gname, onp=onp, **s, **nl)
                        rows.append(row)
                        print(f"{side:5s} {k:4.1f} {pf:5.3f} {rr:5.1f} {gname:10s} | {onp:5.1f}% "
                              f"{s['N']:5d} {s['N/年']:6.1f} {s['勝率']:5.1f}% {s['PF']:6.2f} "
                              f"{s['平均%']:8.3f} {s['総%']:8.1f} {s['maxDD%']:7.1f} | "
                              f"{nl['帰無PF中央値']:7.2f} {nl['PF%ile']:6.1f}% "
                              f"{nl['帰無平均%中央値']:9.3f} {nl['平均%ile']:7.1f}%")

    gdf = pd.DataFrame(rows)

    # ---- 4象限まとめ（法則11の検定・KAMAゲート、基準セルのk/RR）
    print("\n" + "=" * 90)
    print("4象限まとめ: 法則11（順方向=速いゲート4h／逆方向=遅いゲート日足）の検定 — KAMAゲート")
    print("=" * 90)
    BASE_CELLS = [("long", 2.0, 3.0, 0.0), ("long", 1.5, 4.5, 0.0),
                  ("short", 2.0, 3.0, 0.5), ("short", 1.5, 4.5, 0.382)]
    for side, k, rr, pf in BASE_CELLS:
        print(f"\n--- {side} k={k} RR={rr} pf={pf} ---")
        for gname in ("none", "kama_4h", "kama_1d"):
            sub = gdf[(gdf.side == side) & (gdf.k == k) & (gdf.rr == rr) & (gdf.pf == pf) & (gdf.gate == gname)]
            if len(sub) == 0:
                print(f"    {gname:10s}: データ無し")
                continue
            r = sub.iloc[0]
            print(f"    {gname:10s}: ON%={r.onp:5.1f}%  N={r.N:5d}  N/年={r['N/年']:6.1f}  勝率={r['勝率']:5.1f}%  "
                  f"PF={r.PF:5.2f}  平均%={r['平均%']:+7.3f}  PF%ile={r['PF%ile']:6.1f}  平均%ile={r['平均%ile']:6.1f}")

    # ---- 帰無を明確に超えたセルの抽出（PF%ile>=95 かつ 平均%ile>=95、gate!=none）
    gated = gdf[gdf.gate != "none"].copy()
    cleared = gated[(gated["PF%ile"] >= 95) & (gated["平均%ile"] >= 95)].sort_values("PF", ascending=False)
    print("\n" + "=" * 90)
    print(f"帰無(200回間引き)を PF・平均% とも %ile>=95 で超えたセル: {len(cleared)}/{len(gated)}")
    print("=" * 90)
    if len(cleared) > 0:
        print(cleared[["side", "k", "pf", "rr", "gate", "onp", "N", "N/年", "勝率", "PF", "平均%", "PF%ile", "平均%ile"]]
              .to_string(index=False))
    else:
        print("  該当なし（ゲートは全て、同じ通過率のランダム間引きと統計的に区別できない）")

    # ---- 最良セルの深掘り（cleared があればその最上位、無ければ gate!=none 全体でPF最大）
    if len(cleared) > 0:
        best = cleared.iloc[0]
    else:
        best = gated.sort_values("PF", ascending=False).iloc[0]
    print("\n" + "=" * 90)
    print(f"最良セル深掘り: {best.side} k={best.k} RR={best.rr} pf={best.pf} ゲート={best.gate}")
    print("=" * 90)
    bside = df if best.side == "long" else inv
    bC = None if best.side == "long" else C
    bgarr = (gates_long if best.side == "long" else gates_short)[best.gate]
    bent = build_entries(bside, best.k, best.rr, bgarr)
    bt = run_cell(bside, bent, best.pf, C=bC, cost=COST)
    bt = bt.assign(y=pd.DatetimeIndex(bt["time"]).year)
    yr = bt.groupby("y")["pnl_pct"].agg(N="size", PF=lambda x: pf_of(x.to_numpy()), 平均pct=lambda x: x.mean() * 100)
    print("年別(比例コスト0.0005):")
    print(yr.to_string())

    print("\n巡回ブロック・ブートストラップ（比例コスト、1/3/6/12か月）:")
    bb = block_bootstrap(bt, [1, 3, 6, 12])
    for kmo, v in bb.items():
        print(f"  ブロック{kmo:>2}か月: 勝率中央値={v['勝率中央値']:.1f}% CI=[{v['勝率CI'][0]:.1f},{v['勝率CI'][1]:.1f}]  "
              f"PF中央値={v['PF中央値']:.2f} CI=[{v['PFCI'][0]:.2f},{v['PFCI'][1]:.2f}]  "
              f"平均%中央値={v['平均%中央値']:+.3f}%  (有効draw={v['draw']}/1000)")

    print("\n固定$25コスト版:")
    bt25 = run_cell(bside, bent, best.pf, C=bC, cost_mode="fixed", cost_fixed=25.0)
    bt25 = bt25.assign(y=pd.DatetimeIndex(bt25["time"]).year)
    s25 = stats(bt25, span)
    print(f"  全期間: N={s25['N']} N/年={s25['N/年']:.1f} 勝率={s25['勝率']:.1f}% PF={s25['PF']:.2f} "
          f"平均%={s25['平均%']:+.3f} maxDD%={s25['maxDD%']:.1f}")
    for y in (2018, 2021, 2025):
        sub = bt25[bt25.y == y]
        if len(sub) == 0:
            print(f"  {y}: データ無し")
            continue
        p = sub["pnl_pct"].to_numpy()
        print(f"  {y}: N={len(sub)} 勝率={float((p>0).mean())*100:.1f}% PF={pf_of(p):.2f} 平均%={p.mean()*100:+.3f}")

    # ---- 台地/尖り点検: best の隣接セル(k, RR, ゲートtf)を並べる
    print("\n台地/尖り点検（best 周辺のセル、比例コスト）:")
    neigh = gdf[(gdf.side == best.side) & (gdf.pf == best.pf) &
                (gdf.k.isin([x for x in K_LIST])) & (gdf.rr.isin(RR_LIST)) &
                (gdf.gate.isin(["none", "kama_4h", "kama_1d", "sma_1d150", "sma_4h150"]))]
    neigh = neigh[(neigh.k.between(best.k - 0.5, best.k + 0.5)) & (neigh.rr.between(min(RR_LIST), max(RR_LIST)))]
    print(neigh[["k", "rr", "gate", "onp", "N", "PF", "平均%", "PF%ile", "平均%ile"]]
          .sort_values(["gate", "k", "rr"]).to_string(index=False))

    gdf.to_csv(os.path.join(ROOT, "scratchpad/atr_spike_btc_h1_gates_grid.csv"), index=False)
    print(f"\n全格子を保存: scratchpad/atr_spike_btc_h1_gates_grid.csv ({len(gdf)}行)")


if __name__ == "__main__":
    main()
