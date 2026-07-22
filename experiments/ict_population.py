"""ICT 再設計 — フェーズ1: 素の入口母集団（エンジンを裸で測る）。

背骨: フィルタは濃縮しかしない（法則1）。素母集団がプラセボ窓に勝てなければ、上に何を積んでも
無意味。だからゲート・棄権より先に、ここで --peryear / IS-OOS / プラセボ窓(+4/8/12h) を通す。

機構（ロング。ショートは完全な鏡像）— 全て NY 壁時計:
  1. sweep : ロンドン窓の安値 L が「アジア安値 or 前日安値」を割る（流動性の狩り）
  2. MSS   : L の後、L 直前の3本フラクタル高値を上抜ける（構造転換）。上抜けまでの最高値 = H
  3. entry : NYキルゾーンで買い指値 lim = H - f*(H-L)（浅い押し目 f=0.25）
  4. stop  : L - buf*ATR / target: entry + RR*(entry-stop)（RR4 の遠い固定目標）
  5. 無効化: キルゾーン到達前に L を割ったら無効

use_sweep/use_mss/leg/shift を引数に持つので、フェーズ4 の ablation・プラセボ窓でそのまま使える。
**方向は long/short を別々に集計する（絶対に混ぜない）**＝両サイド平均は今回2回私を騙した病巣。

自己検査（EURUSD ロング旗艦 n≈1148/PF1.17/totR-DD2.56 とプラセボ窓プレミアムを再現）:
    .venv/bin/python experiments/ict_population.py
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/experiments")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import (SYMS, MODEL, PIP, CUT2000, BUF, F_CANON, RR_CANON,
                      ASIA_HOURS, LONDON_HOURS, KZ_HOURS,
                      load_ny, prep, span_years, window_pos, walk, stats)


def prev_day_extremes(df, dates):
    """NY暦の前日の高値/安値（ICT の "previous day high/low"）。"""
    g = df.groupby(df["_t"].dt.normalize()).agg(hi=("high", "max"), lo=("low", "min"))
    hi = g["hi"].reindex(dates).shift(1)
    lo = g["lo"].reindex(dates).shift(1)
    return dict(zip(dates, hi.values)), dict(zip(dates, lo.values))


def last_fractal_high(highs, s, e):
    """[s, e) の中で最後の 3本フラクタル高値の位置（無ければ None）。"""
    for k in range(e - 2, s, -1):
        if highs[k] >= highs[k - 1] and highs[k] >= highs[k + 1]:
            return k
    return None


def last_fractal_low(lows, s, e):
    for k in range(e - 2, s, -1):
        if lows[k] <= lows[k - 1] and lows[k] <= lows[k + 1]:
            return k
    return None


def recent_swing_high(hi, end, n):
    """[end-n, end) 内の3本フラクタル高値(確定済み、k+1<end)のうち最大＝"直近N本のスイング高値"。
    ICT 優先3（外部流動性ターゲット）用。無ければ None。"""
    s = max(1, end - n)
    best = None
    for k in range(s, end - 1):
        if hi[k] >= hi[k - 1] and hi[k] >= hi[k + 1]:
            if best is None or hi[k] > best:
                best = hi[k]
    return best


def recent_swing_low(lo, end, n):
    """recent_swing_high の鏡像（安値側）。"""
    s = max(1, end - n)
    best = None
    for k in range(s, end - 1):
        if lo[k] <= lo[k - 1] and lo[k] <= lo[k + 1]:
            if best is None or lo[k] < best:
                best = lo[k]
    return best


def bullish_fvg_size(hi, lo, atr_val, s, e, min_atr):
    """[s, e] 区間内（bar index、両端含む）の 3本1組 bullish FVG（quant-audit.md 準拠）を走査。
    candle1=i, candle2=i+1, candle3=i+2。candle3.low > candle1.high の帯 [candle1.high, candle3.low]。
    size=(candle3.low-candle1.high)/atr_val が min_atr 以上のうち最大を返す（無ければ None,None）。
    生成タイムスタンプ(=candle3確定時)で固定。後埋めは判定に使わない（存在＝生成時点のギャップの有無）。
    戻り値 = (best_size, (lo_edge, hi_edge)) — lo_edge=candle1.high（深い/全埋め）, hi_edge=candle3.low（浅い）。"""
    best = None
    edges = None
    for i in range(s, e - 1):
        c1_hi, c3_lo = hi[i], lo[i + 2]
        if c3_lo > c1_hi:
            size = (c3_lo - c1_hi) / atr_val
            if size >= min_atr and (best is None or size > best):
                best = size
                edges = (c1_hi, c3_lo)
    return best, edges


def bearish_fvg_size(hi, lo, atr_val, s, e, min_atr):
    """bearish FVG: candle1.low > candle3.high の帯 [candle3.high, candle1.low]。
    戻り値 = (best_size, (lo_edge, hi_edge)) — lo_edge=candle3.high（浅い、鏡像）, hi_edge=candle1.low（深い、鏡像）。"""
    best = None
    edges = None
    for i in range(s, e - 1):
        c1_lo, c3_hi = lo[i], hi[i + 2]
        if c1_lo > c3_hi:
            size = (c1_lo - c3_hi) / atr_val
            if size >= min_atr and (best is None or size > best):
                best = size
                edges = (c3_hi, c1_lo)
    return best, edges


def build(df, tarr, dates, use_sweep=True, use_mss=True, leg="mss", shift=0,
          use_fvg=False, fvg_min_atr=0.0, use_liq=False, liq_ns=(20, 40)):
    """ゲート（狩り/MSS/FVG）を独立に切れる setup builder。leg="mss"=v2定義（MSS足まで）/ "lonend"=v1定義。
    shift>0 は窓を +Nh ずらした偽キルゾーン（プラセボ）。
    use_fvg: MSS 認定（直近3本フラクタルの上抜け/下抜け）に、ブレイク脚内（抜かれたフラクタル足〜
    ブレイク足）に閾値以上の FVG が存在することを AND 条件で追加する（quant-audit.md 項目4）。
    use_mss=False の時は無効（ブレイク脚が未定義のため）。
    use_liq: ICT優先3（外部流動性ターゲット）用に、rec[side] へ流動性レベルを付与する
    （PDH/PDL, asiaH/asiaL, swingH/swingL(N∈liq_ns)）。全て MSS確定足(jm)以前＝先読み無し。
    entry/stop/母集団は不変（この引数は付与のみで判定ロジックに影響しない）。"""
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    atr = df["atr14"].values
    pdh, pdl = prev_day_extremes(df, dates)
    A0H, A1H = ASIA_HOURS[0] + shift, ASIA_HOURS[1] + shift
    L0H, L1H = LONDON_HOURS[0] + shift, LONDON_HOURS[1] + shift
    K0H, K1H = KZ_HOURS[0] + shift, KZ_HOURS[1] + shift
    out = []
    for d in dates:
        day = pd.Timestamp(d)
        a0, a1 = window_pos(tarr, day - pd.Timedelta(days=1) + pd.Timedelta(hours=A0H),
                            day + pd.Timedelta(hours=A1H))
        l0, l1 = window_pos(tarr, day + pd.Timedelta(hours=L0H), day + pd.Timedelta(hours=L1H))
        k0, k1 = window_pos(tarr, day + pd.Timedelta(hours=K0H), day + pd.Timedelta(hours=K1H))
        rec = {"date": d, "long": None, "short": None}
        if (a1 - a0) < 4 or (l1 - l0) < 6 or (k1 - k0) < 2 or not np.isfinite(atr[l1 - 1]):
            out.append(rec); continue
        A = atr[l1 - 1]
        asia_lo, asia_hi = lo[a0:a1].min(), hi[a0:a1].max()
        p_lo, p_hi = pdl.get(d, np.nan), pdh.get(d, np.nan)

        # ---------------- LONG ----------------
        iL = l0 + int(np.argmin(lo[l0:l1])); L = lo[iL]
        ok = True
        if use_sweep:
            ok = ((np.isfinite(asia_lo) and L < asia_lo) or (np.isfinite(p_lo) and L < p_lo))
        jm = None
        fvg_edges = None
        if ok and use_mss:
            sh = last_fractal_high(hi, a0, iL)
            if sh is None:
                ok = False
            else:
                lvl = hi[sh]
                for j in range(iL + 1, l1):
                    if hi[j] > lvl:
                        jm = j; break
                ok = jm is not None
                if ok and use_fvg:
                    sz, fvg_edges = bullish_fvg_size(hi, lo, A, sh, jm, fvg_min_atr)
                    ok = sz is not None
        if ok:
            end = (jm + 1) if (leg == "mss" and jm is not None) else l1
            if end <= iL + 1:
                ok = False
            else:
                H = hi[iL:end].max()
                if H - L < 0.25 * A:
                    ok = False
                elif (lo[end:l1] <= L).any():
                    ok = False
                else:
                    rec["long"] = dict(L=L, H=H, atr=A, kz=(k0, k1))
                    if fvg_edges is not None:
                        rec["long"]["fvg_lo"], rec["long"]["fvg_hi"] = fvg_edges
                    if use_liq:
                        anchor_end = jm if jm is not None else end
                        rec["long"]["pdh"] = p_hi
                        rec["long"]["asiaH"] = asia_hi
                        for nn in liq_ns:
                            rec["long"][f"swingH{nn}"] = recent_swing_high(hi, anchor_end, nn)

        # ---------------- SHORT (鏡像) ----------------
        iH = l0 + int(np.argmax(hi[l0:l1])); Hh = hi[iH]
        ok = True
        if use_sweep:
            ok = ((np.isfinite(asia_hi) and Hh > asia_hi) or (np.isfinite(p_hi) and Hh > p_hi))
        jm = None
        fvg_edges = None
        if ok and use_mss:
            sl = last_fractal_low(lo, a0, iH)
            if sl is None:
                ok = False
            else:
                lvl = lo[sl]
                for j in range(iH + 1, l1):
                    if lo[j] < lvl:
                        jm = j; break
                ok = jm is not None
                if ok and use_fvg:
                    sz, fvg_edges = bearish_fvg_size(hi, lo, A, sl, jm, fvg_min_atr)
                    ok = sz is not None
        if ok:
            end = (jm + 1) if (leg == "mss" and jm is not None) else l1
            if end <= iH + 1:
                ok = False
            else:
                Ll = lo[iH:end].min()
                if Hh - Ll < 0.25 * A:
                    ok = False
                elif (hi[end:l1] >= Hh).any():
                    ok = False
                else:
                    rec["short"] = dict(L=Ll, H=Hh, atr=A, kz=(k0, k1))
                    if fvg_edges is not None:
                        rec["short"]["fvg_lo"], rec["short"]["fvg_hi"] = fvg_edges
                    if use_liq:
                        anchor_end = jm if jm is not None else end
                        rec["short"]["pdl"] = p_lo
                        rec["short"]["asiaL"] = asia_lo
                        for nn in liq_ns:
                            rec["short"][f"swingL{nn}"] = recent_swing_low(lo, anchor_end, nn)
        out.append(rec)
    return out


def canonical_setups(df, tarr, dates, shift=0, use_fvg=False, fvg_min_atr=0.0,
                     use_liq=False, liq_ns=(20, 40)):
    """本命母集団 = 狩り + MSS + 脚=MSSまで。use_fvg=True で MSS に FVG-displacement条件を追加。
    use_liq=True で流動性レベル（PDH/PDL, asiaH/asiaL, swingH/L(N)）を rec に付与（優先3用）。"""
    return build(df, tarr, dates, True, True, "mss", shift, use_fvg=use_fvg, fvg_min_atr=fvg_min_atr,
                 use_liq=use_liq, liq_ns=liq_ns)


def trade_pool(df, setups, name, f=F_CANON, rr=RR_CANON):
    """両サイドの全トレードを日付キーで持つ（サイド選択はフェーズ2の日足ゲートが決める）。"""
    sp, cost = MODEL[name]
    sp = sp   # MODEL の spread は既に価格単位（PIP 済み）
    pool = {"long": {}, "short": {}}
    for side in ("long", "short"):
        for (d, net, g, risk) in walk(df, setups, f, rr, BUF, sp, cost, side):
            pool[side][d] = net
    return pool


def side_list(df, setups, name, side, f=F_CANON, rr=RR_CANON):
    """片側の (date, net) リスト（方向を混ぜずにフェーズ1の裸母集団を測る用）。"""
    sp, cost = MODEL[name]
    return [(d, net) for (d, net, g, risk) in walk(df, setups, f, rr, BUF, sp, cost, side)]


def load_prepped(name):
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        df, _ = load_ny(SYMS[name], cut2000=(name in CUT2000))
    df, tarr, dates = prep(df)
    return df, tarr, dates, span_years(df)


if __name__ == "__main__":
    print("フェーズ1 自己検査: EURUSD ロング旗艦（狩り+MSS/浅0.25/RR4）と +8h プラセボ窓のプレミアム")
    print("  台帳アンカー: n=1148・PF 1.17・net +0.134・totR/DD 2.56、プラセボ窓プレミアム net ≈ +0.237")
    df, tarr, dates, span = load_prepped("eurusd")
    S0 = canonical_setups(df, tarr, dates, 0)
    S8 = canonical_setups(df, tarr, dates, 8)
    sp, cost = MODEL["eurusd"]
    r = stats(walk(df, S0, F_CANON, RR_CANON, BUF, sp, cost, "long"), span)
    p = stats(walk(df, S8, F_CANON, RR_CANON, BUF, sp, cost, "long"), span)
    print(f"  本物の窓  : n={r['n']} 年{r['npy']:.0f} win%={r['win']:.1f} net={r['net']:+.3f} "
          f"PF={r['pf']:.2f} totR/DD={r['rdd']:.2f} IS={r['IS']:+.0f} OOS={r['OOS']:+.0f} 緑年{r['gy']:.0f}%")
    print(f"  +8h偽窓   : n={p['n']} net={p['net']:+.3f} PF={p['pf']:.2f}")
    print(f"  窓プレミアム(net差) = {r['net'] - p['net']:+.3f}")
