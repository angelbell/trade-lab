"""ICT の日足バイアスを *出典どおり* に機械化する = premium/discount（平均回帰極性）(2026-07-15)。

これまでの私の棄権テスト（ict_abstain / ict_fx_abstain）はバイアスを「トレンドの強さ」
(ER/KAMA傾き/MA乖離/実体比) で測っていた。だが文字起こし(youtube-docs)を読むと、ICT の日足
バイアスはそれではない:
  Ep35: ディーリング・レンジ 安値→高値 に fib、50=equilibrium、discount に落ちるのを待つ
  Ep16: アルゴは discount→premium を求める。discount array から *買う*、上の buy-side を狙う
  Ep12: 片方向にしか枠づけできない = high prob。どっちにも行ける日 = ダメ
  Ep19: muddy でチョップ = smart money が座らない相場（held/manipulated）= 触るな
→ (1) 方向は premium/discount で決まる（discount→ロング）＝ トレンド追随とは *逆の極性*
   (2) equilibrium 付近 = どっちつかず = 棄権
   (3) 直近日足に displacement があり muddy でない

母集団: v2 生存形 = 狩り(sweep)+MSS + 浅0.25 + RR4 + NYキルゾーン、ASK指値約定(0.3pip+手数料のみ)。
キルゾーンのセットアップは *それ自体に向き* を持つ（ロング=安値を狩って上へMSS / ショート=鏡像）。
日足ゲートは、そのセットアップの向きが premium/discount 極性と *一致する時だけ* 通す。

ゲートA（ICT の本命 = premium/discount 極性）:
   ロング設定 → 日足が discount（rngpos < 0.5-band）の時だけ通す
   ショート設定 → 日足が premium（rngpos > 0.5+band）の時だけ通す
   equilibrium 帯（|rngpos-0.5|<band）= 棄権。 L∈{10,20,40}日, band∈{0.0,0.1,0.2}
   ★対照として「逆極性」(トレンド追随: discount→ショート)も出す＝機構の符号確認
ゲートB（muddy 棄権・サイド固定 = ユーザーの「不明な日は入るな」）:
   セットアップの向きは保持。直近 k 日足に displacement が無い（重なり合っている）日を棄権。

審判(CLAUDE.md ch.7): ランダム間引き帰無の totR/DD %ile / 閾値台地 / 時代別 / プラセボ窓(+8h)。

Run: .venv/bin/python experiments/ict_pd_bias.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, BUF
from ict_abstain import trade_pool, join_days, sc, random_drop_null
from breakout_wave import resample

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
LOOKBACKS = [10, 20, 40]
BANDS = [0.0, 0.10, 0.20]
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]


def pd_frame(df):
    """日足の premium/discount 位置と displacement を、前日までに確定した形で返す。"""
    b = df.set_index("broker_dt")[["open", "high", "low", "close"]]
    d = resample(b, "1D")
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    D = pd.DataFrame(index=d.index)
    for L in LOOKBACKS:
        hi = h.rolling(L).max()
        lo = l.rolling(L).min()
        D[f"pos{L}"] = ((c - lo) / (hi - lo)).replace([np.inf, -np.inf], np.nan)
    # displacement: 直近 k 本の「行進度」= (union range) / (Σ individual range)。
    #   1 に近い = 一方向に変位（clean）/ 小さい = 重なり合ってチョップ（muddy）
    for k in (3, 5):
        rng = (h - l)
        union = h.rolling(k).max() - l.rolling(k).min()
        D[f"march{k}"] = (union / rng.rolling(k).sum()).replace([np.inf, -np.inf], np.nan)
    conf = (D.index + pd.Timedelta(days=1)).tz_localize(
        "Europe/Riga", ambiguous="NaT", nonexistent="shift_forward"
    ).tz_convert("America/New_York").tz_localize(None)
    D["conf"] = conf
    return D[~D["conf"].isna()].sort_values("conf").reset_index(drop=True)


def eras_of(tr):
    o = []
    for a, b in ERAS:
        v = [x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b]
        o.append(f"{sum(v):+6.0f}" if v else "   n/a")
    return " ".join(o)


def gate_pd(pool, J, poscol, band, invert=False):
    """ゲートA: セットアップの向きが premium/discount 極性と一致する日だけ通す。
    invert=True は逆極性（トレンド追随: discount→ショート）＝ 符号確認の対照。"""
    lo_thr, hi_thr = 0.5 - band, 0.5 + band
    out = []
    for d, row in J.iterrows():
        pos = row[poscol]
        if pd.isna(pos):
            continue
        disc = pos < lo_thr        # discount
        prem = pos > hi_thr        # premium
        long_ok = prem if invert else disc
        short_ok = disc if invert else prem
        if long_ok and d in pool["long"]:
            out.append((d, pool["long"][d]))
        if short_ok and d in pool["short"]:
            out.append((d, pool["short"][d]))
    return out


def base_bothsides(pool, J):
    out = []
    for d in J.index:
        if d in pool["long"]:
            out.append((d, pool["long"][d]))
        if d in pool["short"]:
            out.append((d, pool["short"][d]))
    return out


def gate_muddy(pool, J, side, marchcol, q):
    """ゲートB: セットアップの向き(side)を保持し、muddy(下位 q の march)な日を棄権。"""
    thr = J[marchcol].quantile(q) if q > 0 else -np.inf
    out = []
    for d, row in J.iterrows():
        if d not in pool[side]:
            continue
        m = row[marchcol]
        if q > 0 and (pd.isna(m) or m < thr):
            continue
        out.append((d, pool[side][d]))
    return out


def line(lab, s, base_for_null=None):
    if s is None:
        return
    extra = ""
    if base_for_null is not None:
        nul = random_drop_null(base_for_null, s["n"])
        pc = 100 * (s["rdd"] > nul).mean()
        extra = f" {np.median(nul):8.2f} {pc:4.0f}%{' *' if pc >= 90 else ''}"
    print(f"  {lab:26s} {s['n']:5d} {s['net']:+7.3f} {s['pf']:5.2f} "
          f"{s['tot']:+7.1f} {s['dd']:6.1f} {s['rdd']:8.2f}{extra}")


def main():
    print("ICT 日足バイアス = premium/discount（平均回帰極性）。母集団: 狩り+MSS/浅0.25/RR4/NYキルゾーン")
    print("審判: ランダム間引き帰無 totR/DD %ile · 逆極性は符号確認の対照")
    for name in ("eurusd", "gbpusd", "gold", "btcusd"):
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(SYMS[name], cut2000=(name == "usdjpy"))
        df, tarr, dates = prep(df)
        span = int((pd.to_datetime(df["broker_dt"]).dt.year.value_counts() > 5000).sum())
        S0 = build(df, tarr, dates, True, True, "mss", 0)
        S8 = build(df, tarr, dates, True, True, "mss", 8)
        pool0 = trade_pool(df, S0, name)
        pool8 = trade_pool(df, S8, name)
        P = pd_frame(df)
        alld = sorted(set(list(pool0["long"]) + list(pool0["short"])))
        J = join_days(alld, P)
        J8 = join_days(sorted(set(list(pool8["long"]) + list(pool8["short"]))), P)

        base = base_bothsides(pool0, J)
        b = sc(base)
        if b is None:
            continue
        print("\n" + "=" * 120)
        print(f"=== {name} ===  ({span}年)   base=両サイド全設定")
        print(f"  {'条件':26s} {'n':>5} {'net':>7} {'PF':>5} {'totR':>7} {'DD':>6} "
              f"{'totR/DD':>8} {'null中央':>8} {'%ile':>5}  時代別 totR")
        line("(ゲートなし＝ベース)", b)
        print(f"  {'':26s} {'':5s} {'':7s} {'':5s} {'':7s} {'':6s} {'':8s} {'':8s} {'':5s}  {eras_of(base)}")

        print("  --- ゲートA: premium/discount 極性（ICT 本命: discount→ロング） ---")
        best = None
        for L in LOOKBACKS:
            for band in BANDS:
                tr = gate_pd(pool0, J, f"pos{L}", band)
                s = sc(tr)
                if s is None:
                    continue
                lab = f"L={L} band={band:.2f}"
                line(lab, s, base_for_null=base)
                if best is None or s["rdd"] > best[1]["rdd"]:
                    best = (lab, s, tr, L, band)
        # 逆極性の対照（同じ最良 L/band で符号確認）
        if best is not None:
            _, _, _, L, band = best
            inv = gate_pd(pool0, J, f"pos{L}", band, invert=True)
            print("  --- 逆極性の対照（トレンド追随: discount→ショート）＝ 符号確認 ---")
            line(f"逆 L={L} band={band:.2f}", sc(inv), base_for_null=base)

        print("  --- ゲートB: muddy 棄権（サイド固定・ユーザーの「不明な日は入るな」） ---")
        for side in ("long", "short"):
            sb = sc([(d, pool0[side][d]) for d in J.index if d in pool0[side]])
            if sb is None:
                continue
            print(f"    [{side}] ベース: n={sb['n']} totR/DD={sb['rdd']:.2f}")
            for k in (3, 5):
                for q in (0.35, 0.5, 0.65):
                    tr = gate_muddy(pool0, J, side, f"march{k}", q)
                    s = sc(tr)
                    base_side = [(d, pool0[side][d]) for d in J.index if d in pool0[side]]
                    line(f"    march{k} 棄権{int(q*100)}%", s, base_for_null=base_side)

        # プラセボ窓（最良ゲートA 設定を +8h でも引けるか）
        if best is not None:
            lab, s, tr, L, band = best
            z = sc(gate_pd(pool8, J8, f"pos{L}", band))
            print("  --- プラセボ窓(+8h)：最良ゲートA設定 ---")
            if z is not None:
                print(f"    本物 {lab}: n={s['n']} net={s['net']:+.3f} totR/DD={s['rdd']:.2f}"
                      f"  |  +8h: n={z['n']} net={z['net']:+.3f} totR/DD={z['rdd']:.2f}"
                      f"  premium={s['net']-z['net']:+.3f}")
            print(f"    最良ゲートA 時代別: {eras_of(tr)}")


if __name__ == "__main__":
    main()
