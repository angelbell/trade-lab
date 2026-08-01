"""仕様カード experiments/spec_level_tier_1h4h.md の実装。

上位足レベルの強度3層（ブレイク終値が {前日高値, 直近確定4Hスイング高値} の何本の上か）を、
BTC 15m 以外の5セル（gold 1H / BTC 1H / BTC 4H / USDJPY 1H / USDJPY 4H）へ移せるかの検定。
固定ロット（0.01床・非複利・R等ウェイト）の裁量手張りが前提なので、審判は CAGR/DD ではなく
**総R ÷ maxDD(R)**。腕は3つ: 全部張る / neither を張らない / both だけ張る。

流用（車輪の再発明禁止）:
  - breakout_wave.{run, resample, swings_zigzag} … 入口・ウォーカーは engine のものをそのまま使う
  - research/regime_gate_lab.CFG … gold_bo と同じ ZigZag(2×ATR)/Pattern-B/L2 の設定辞書
  - stack_size_btc15mL.hh4h_series と同一アルゴリズム（swings_zigzag → confirm 後 shift(1) → ffill）を
    任意の入口TFへ一般化（--tieback で既知値に対して数値 assert する）
  - research/screen.run_screen … 巡行幅（MFE/MAE）を先に測るフックの通し方

実行:
  .venv/bin/python experiments/level_tier_1h4h.py --tieback   # 層実装の検算のみ（先にこれを通す）
  .venv/bin/python experiments/level_tier_1h4h.py --smoke     # 軽い通し（null 200回）
  .venv/bin/python experiments/level_tier_1h4h.py             # 本番

⚠️ このスクリプトはデータの無い環境で書かれており、価格CSVに対しては未実行です。
   初回は必ず --tieback → --smoke → 本番の順で回してください。
"""
SCREEN = "level_tier_1h4h"

import sys
import io
import contextlib
import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src.data_loader import load_mt5_csv                      # noqa: E402
from breakout_wave import run, resample, swings_zigzag        # noqa: E402
from research.regime_gate_lab import CFG                      # noqa: E402
from research.screen import run_screen                        # noqa: E402

NULL_ITERS = 2000
NULL_SEED = 20260801
BOOT_NB = 2000
BLOCK_MONTHS = (1, 3, 6, 12)

# 対象5セル。cost_kind: "pct"=価格比率 / "abs"=絶対額（いずれも R 単位へ /risk して引く）
CELLS = [
    dict(name="gold 1H",   csv="vantage_xauusd_h1.csv", tf="1h", start="2018-01-01",
         cost_kind="pct", cost=0.001),
    dict(name="BTC 1H",    csv="vantage_btcusd_h1.csv", tf="1h", start="2022-01-01",
         cost_kind="abs", cost=15.0),
    dict(name="BTC 4H",    csv="vantage_btcusd_h1.csv", tf="4h", start="2022-01-01",
         cost_kind="abs", cost=15.0),
    dict(name="USDJPY 1H", csv="vantage_usdjpy_h1.csv", tf="1h", start="2000-01-01",
         cost_kind="abs", cost=0.009),   # 0.9pip（JPY建て 1pip = 0.01）
    dict(name="USDJPY 4H", csv="vantage_usdjpy_h1.csv", tf="4h", start="2000-01-01",
         cost_kind="abs", cost=0.009),
]

TF_MIN = {"1h": 60, "4h": 240, "15min": 15}


# ------------------------------------------------------------------ 層の定義
def pdh_series(d):
    """前日高値。日足の確定値を shift(1) して入口TFへ ffill 展開（先読み無し）。"""
    return d["high"].resample("1D").max().dropna().shift(1).reindex(d.index, method="ffill").values


def hh4h_series(d):
    """直近の確定4Hスイング高値。stack_size_btc15mL.hh4h_series と同一アルゴリズム
    （4hへ集約 → ATR(14) → swings_zigzag(k=2) → 確定足に置いて ffill → shift(1) → 入口TFへ展開）。"""
    h4 = d.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = ta.atr(h4["high"], h4["low"], h4["close"], 14).values
    sw = swings_zigzag(h4["high"].values, h4["low"].values, a4, 2.0)
    s = pd.Series(np.nan, index=h4.index)
    for (ci, pi, px, kind) in sw:
        if kind == +1:
            s.iloc[ci] = px
    return s.ffill().shift(1).reindex(d.index, method="ffill").values


def tier_masks(d, t, ii):
    """both / one / neither の3マスク。e_px（ブレイク終値＝約定価格）を2本のレベルと比べる。"""
    pdh, hh4 = pdh_series(d), hh4h_series(d)
    e_px = t["e_px"].values
    above_pdh = np.where(np.isfinite(pdh[ii]), e_px > pdh[ii], False)
    above_hh4 = np.where(np.isfinite(hh4[ii]), e_px > hh4[ii], False)
    both = above_pdh & above_hh4
    neither = (~above_pdh) & (~above_hh4)
    one = ~both & ~neither
    return both, one, neither


# ------------------------------------------------------------------ 統計
def tier_stats(R, mask, years):
    r = R[mask]
    if len(r) == 0:
        return dict(n=0, per_yr=0.0, win=np.nan, pf=np.nan, mean=np.nan, med=np.nan,
                    sd=np.nan, tot=0.0)
    w, l = r[r > 0], r[r <= 0]
    return dict(n=len(r), per_yr=len(r) / years,
                win=100.0 * len(w) / len(r),
                pf=(w.sum() / abs(l.sum())) if len(l) and l.sum() != 0 else np.inf,
                mean=r.mean(), med=np.median(r), sd=r.std(ddof=1) if len(r) > 1 else np.nan,
                tot=r.sum())


def dd_R(R):
    """固定ロット（非複利・R等ウェイト）の maxDD を R 単位で。"""
    if len(R) == 0:
        return np.nan
    cum = np.cumsum(R)
    return float(np.max(np.maximum.accumulate(cum) - cum))


def arm_stats(R):
    if len(R) == 0:
        return dict(n=0, tot=np.nan, dd=np.nan, ratio=np.nan)
    tot, dd = float(np.sum(R)), dd_R(R)
    return dict(n=len(R), tot=tot, dd=dd, ratio=(tot / dd) if dd and dd > 0 else np.inf)


# ------------------------------------------------------------------ 帰無
def null_gradient(R, n_b, n_o, n_n, real_grad, iters, seed):
    """同じサイズの無作為3分割で meanR(both)-meanR(neither) の勾配を作り、実測の%ileを返す。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(R))
    out = np.empty(iters)
    for i in range(iters):
        p = rng.permutation(idx)
        b, n = R[p[:n_b]], R[p[n_b + n_o:n_b + n_o + n_n]]
        out[i] = b.mean() - n.mean()
    return 100.0 * float(np.mean(out < real_grad)), out


def null_drop(R, keep_n, real_ratio, iters, seed):
    """同じ本数だけランダムに間引いた母集団の 総R÷maxDD(R) を作り、実測の%ileを返す。
    ⚠️ 物差しは meanR でも PF でもなく 総R÷maxDD(R)（法則7の間引き帰無）。"""
    rng = np.random.default_rng(seed + 1)
    idx = np.arange(len(R))
    out = np.empty(iters)
    for i in range(iters):
        k = np.sort(rng.choice(idx, size=keep_n, replace=False))   # 時系列順は保つ
        s = arm_stats(R[k])
        out[i] = s["ratio"]
    return 100.0 * float(np.mean(out < real_ratio)), out


def block_boot(times, R, mask, k_months, nb, seed):
    """巡回ブロック・ブートストラップ。P(both群の meanR > 全体の meanR)。
    ブロックは k か月のカレンダー区画で切り、区画を復元抽出して並べ直す。"""
    rng = np.random.default_rng(seed + k_months)
    t0 = times.min()
    per = ((times.year - t0.year) * 12 + (times.month - t0.month)) // k_months
    blocks = [np.where(per == p)[0] for p in np.unique(per)]
    blocks = [b for b in blocks if len(b)]
    if len(blocks) < 2:
        return np.nan
    n_need = len(R)
    hit = 0
    for _ in range(nb):
        take, tot = [], 0
        while tot < n_need:
            b = blocks[rng.integers(len(blocks))]
            take.append(b)
            tot += len(b)
        s = np.concatenate(take)[:n_need]
        rs, ms = R[s], mask[s]
        if ms.sum() < 3:
            continue
        if rs[ms].mean() > rs.mean():
            hit += 1
    return 100.0 * hit / nb


# ------------------------------------------------------------------ セル1本
def build_cell(cell):
    path = ROOT / "data" / cell["csv"]
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。ブリッジ（../mt5-mcp）で取得してから回すこと。")
    with contextlib.redirect_stderr(io.StringIO()):
        d = resample(load_mt5_csv(str(path)).loc[cell["start"]:], cell["tf"])
    args = SimpleNamespace(**{**CFG, "csv": "x", "tf": cell["tf"], "rr": 3.0, "fwd": 500,
                              "cost": 0.0, "daily_sma": 0, "daily_slope_k": 0})
    with contextlib.redirect_stderr(io.StringIO()):
        t = run(d, args)
    if len(t) == 0:
        raise RuntimeError(f"{cell['name']}: トレードが0本。データ範囲か引数を疑うこと。")
    ii = d.index.get_indexer(pd.DatetimeIndex(t["time"]))
    return d, t, ii


def net_R(t, cell):
    """素R からコストを R 単位で引く。gold=価格の0.1%、BTC=$15、USDJPY=0.9pip。"""
    gross = t["R"].values.astype(float)
    c = cell["cost"] * t["e_px"].values if cell["cost_kind"] == "pct" else cell["cost"]
    return gross, gross - c / t["risk"].values


def screen_first(d, t, cell):
    """フック（screen_gate.py）が要求する巡行幅の先行測定。入口TFに合わせた窓で測る。"""
    w = TF_MIN[cell["tf"]]
    entries = [(pd.Timestamp(tm), +1, float(px), float(px - rk))
               for tm, px, rk in zip(t["time"], t["e_px"], t["risk"])]
    with contextlib.redirect_stderr(io.StringIO()):
        return run_screen(f"{SCREEN}_{cell['name'].replace(' ', '_')}", d, entries,
                          windows=(w, w * 5, w * 20), quiet=True)


def report_cell(cell, iters, boot_nb):
    d, t, ii = build_cell(cell)
    times = pd.DatetimeIndex(t["time"])
    years = max((times[-1] - times[0]).days / 365.25, 0.5)
    gross, net = net_R(t, cell)
    both, one, neither = tier_masks(d, t, ii)

    sc = screen_first(d, t, cell)
    print(f"\n{'=' * 104}\n■ {cell['name']}  n={len(t)}  {times[0].date()}→{times[-1].date()}"
          f"  ({years:.1f}年)  コスト={cell['cost']}{'(比率)' if cell['cost_kind'] == 'pct' else '(絶対)'}")
    if sc and "windows" in sc:
        for w, v in sc["windows"].items():
            if isinstance(v, dict) and "mfe_mae" in v:
                print(f"   巡行幅スクリーン W={w}: MFE/MAE={v['mfe_mae']:.2f}")

    print(f"\n  {'層':<9}{'n':>6}{'n/年':>7}{'勝率':>8}{'PF':>7}{'平均R':>9}{'中央R':>9}"
          f"{'σ(R)':>8}{'総R':>9}   （素 / ネット）")
    for lbl, m in (("both", both), ("one", one), ("neither", neither)):
        sg, sn = tier_stats(gross, m, years), tier_stats(net, m, years)
        thin = "  ← N不足" if sg["n"] < 50 else ""
        print(f"  {lbl:<9}{sg['n']:>6}{sg['per_yr']:>7.1f}{sg['win']:>7.1f}%{sg['pf']:>7.2f}"
              f"{sg['mean']:>+9.3f}{sg['med']:>+9.3f}{sg['sd']:>8.2f}{sg['tot']:>+9.1f}{thin}")
        print(f"  {'':<9}{'':>6}{'':>7}{sn['win']:>7.1f}%{sn['pf']:>7.2f}"
              f"{sn['mean']:>+9.3f}{sn['med']:>+9.3f}{sn['sd']:>8.2f}{sn['tot']:>+9.1f}   ネット")

    # --- 検定(a) 単調性（素R。サイズ前・コスト前で機構を見る）
    sb, so, sn_ = (tier_stats(gross, m, years) for m in (both, one, neither))
    mono = (sb["mean"] > so["mean"] > sn_["mean"]) if sb["n"] and so["n"] and sn_["n"] else False
    grad = sb["mean"] - sn_["mean"] if sb["n"] and sn_["n"] else np.nan
    pct_a = np.nan
    if sb["n"] and sn_["n"]:
        pct_a, _ = null_gradient(gross, sb["n"], so["n"], sn_["n"], grad, iters, NULL_SEED)
    print(f"\n  (a) 単調性 both>one>neither: {'YES' if mono else 'NO'}"
          f"   勾配(both−neither)={grad:+.3f}   無作為3分割null: {pct_a:.1f}%ile"
          f"   {'PASS' if (mono and pct_a >= 95) else 'FAIL'}")

    # --- 検定(b) skip腕（ネットR。実際に手にする損益で）
    arms = {"1.全部張る": net, "2.neitherを張らない": net[~neither], "3.bothだけ": net[both]}
    print(f"\n  {'腕':<22}{'n':>6}{'n/年':>7}{'総R':>9}{'maxDD(R)':>11}{'総R÷DD':>10}")
    res = {}
    for lbl, r in arms.items():
        s = arm_stats(r)
        res[lbl] = s
        print(f"  {lbl:<22}{s['n']:>6}{s['n'] / years:>7.1f}{s['tot']:>+9.1f}"
              f"{s['dd']:>11.1f}{s['ratio']:>10.2f}")
    pct_b = np.nan
    keep = int((~neither).sum())
    if 0 < keep < len(net):
        pct_b, _ = null_drop(net, keep, res["2.neitherを張らない"]["ratio"], iters, NULL_SEED)
    print(f"  (b) skip腕の 総R÷DD をランダム間引き{keep}本と比較: {pct_b:.1f}%ile"
          f"   {'PASS' if pct_b >= 95 else 'FAIL'}")

    # --- 巡回ブロック・ブートストラップ
    bb = {k: block_boot(times, gross, both, k, boot_nb, NULL_SEED) for k in BLOCK_MONTHS}
    print("  巡回ブロック P(both群 meanR > 全体): "
          + " / ".join(f"{k}mo {v:.0f}%" for k, v in bb.items()))

    # --- 年別の層別本数（時代の偏りを見る）
    yr = pd.DataFrame({"y": times.year, "b": both, "o": one, "n": neither})
    g = yr.groupby("y")[["b", "o", "n"]].sum()
    print("  年別 both/one/neither: " + "  ".join(f"{y}:{r.b}/{r.o}/{r.n}" for y, r in g.iterrows()))

    thin_any = min(sb["n"], so["n"], sn_["n"]) < 50
    verdict = "N不足（未検証）" if thin_any else ("PASS" if (mono and pct_a >= 95 and pct_b >= 95) else "FAIL")
    print(f"\n  → {cell['name']}: {verdict}")
    return verdict


# ------------------------------------------------------------------ 検算
def tieback():
    """層の実装が正しいことを、btc15m_L の既知値に対する数値 assert で固定する。
    既知（docs/findings/x_sizing.md#fixed-lot-tier, 2026-07-24）:
      both n=135 meanR +1.307 PF 3.05 / one n=188 +0.948 2.31 / neither n=440 +0.177 1.21"""
    from stack_size_btc15mL import build_population
    d15, t, ii = build_population()
    assert len(t) == 763, f"母体 tie-back 失敗: n={len(t)} (既知 763)"
    both, one, neither = tier_masks(d15, t, ii)
    R = t["R"].values.astype(float)
    exp = {"both": (135, 1.307, 3.05), "one": (188, 0.948, 2.31), "neither": (440, 0.177, 1.21)}
    print(f"\n{'層':<9}{'n':>6}{'平均R':>9}{'PF':>7}   （既知値）")
    for lbl, m in (("both", both), ("one", one), ("neither", neither)):
        s = tier_stats(R, m, 1.0)
        n_e, mr_e, pf_e = exp[lbl]
        print(f"{lbl:<9}{s['n']:>6}{s['mean']:>+9.3f}{s['pf']:>7.2f}   （{n_e} / {mr_e:+.3f} / {pf_e:.2f}）")
        assert s["n"] == n_e, f"{lbl}: n={s['n']} 既知={n_e}"
        assert abs(s["mean"] - mr_e) < 0.01, f"{lbl}: meanR={s['mean']:+.3f} 既知={mr_e:+.3f}"
        assert abs(s["pf"] - pf_e) < 0.05, f"{lbl}: PF={s['pf']:.2f} 既知={pf_e:.2f}"
    print("\n✅ 検算OK: 層の実装は既知の btc15m_L 3層と一致。5セルへ広げてよい。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tieback", action="store_true", help="層実装の検算のみ（最初にこれを通す）")
    ap.add_argument("--smoke", action="store_true", help="null 200回の軽い通し")
    a = ap.parse_args()

    if a.tieback:
        tieback()
        return

    iters = 200 if a.smoke else NULL_ITERS
    nb = 200 if a.smoke else BOOT_NB
    print("#" * 104)
    print("上位足レベル強度の二値旗（前日高値 × 直近確定4Hスイング高値）— 1H/4H 5セル")
    print("入口=ZigZag(2×ATR) Pattern-B 確定足ブレイク・ロングのみ / RR3固定 / ゲート無し（素の全信号）")
    print("審判=固定ロット（非複利・R等ウェイト）の 総R ÷ maxDD(R)")
    print("事前登録: (a)単調性かつ無作為3分割null≥95%ile かつ (b)skip腕がランダム間引きnull≥95%ile")
    print("          5セル中3セル以上PASSで🟢 / 1〜2セルなら🟡 / 0セルで❌ / 層n<50は「N不足」")
    print("#" * 104)

    out = {}
    for cell in CELLS:
        try:
            out[cell["name"]] = report_cell(cell, iters, nb)
        except Exception as e:                                    # noqa: BLE001
            out[cell["name"]] = f"実行不能: {type(e).__name__}: {e}"
            print(f"\n■ {cell['name']}: 実行不能 — {type(e).__name__}: {e}")

    print(f"\n{'#' * 104}\n判定まとめ")
    for k, v in out.items():
        print(f"  {k:<12} {v}")
    n_pass = sum(1 for v in out.values() if v == "PASS")
    print(f"\n  PASS {n_pass}/5 → " + ("🟢" if n_pass >= 3 else "🟡" if n_pass >= 1 else "❌"))
    print("  ※ 事前登録どおり、走った後にこの線を緩めないこと。")


if __name__ == "__main__":
    main()
