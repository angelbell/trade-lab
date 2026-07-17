"""ICT 忠実化・優先2 — DXY master bias + クロスペアSMT を方向ゲート本流に配線（2026-07-15）。

忠実性監査 A-3 の解消：FX の ICT 方向判断の背骨（DXY/SMT）が既存のショート脚に未配線だった
（`ict_smt.py` の相方ペア scaffold のみ）。本スクリプトは `docs/... smt-dxy-fx.md` の疑似コードを
そのまま実装し、死んでいるショート（sweep+MSS+FVG のショートは全銘柄 PF<1）が蘇生するか、
SMT/DXYが超加法的に働くかを測る。母集団・執行・コストは v4 固定、動かすのは「どの日に通すか」だけ。

凍結仕様（背景ファイルどおり）:
  DXY = data/vantage_usdx.r_m5.csv（同一Vantageブローカー時刻・UTC変換不要、m5→m15リサンプル）。
  スイング=MSSと同じ3本フラクタル相当として sweep_frame の 1本遅れ・ロンドン窓定義を流用
  （＝相方ペアscaffoldが既に凍結している「観測窓=ロンドン窓・確定=窓終了時」の定義をDXYにも適用）。
  bearish SMT: bearishSMT_pair = own.buy_swept AND NOT partner.buy_swept
               bearishSMT_dxy  = own.buy_swept AND NOT dxy.sell_swept        (EUR/GBP、DXY負相関)
               shortAllowed    = bearishSMT_pair OR bearishSMT_dxy
  USDJPY（ドキュメント原文どおり、逆相関で反転）:
               shortAllowed = usdjpyMadeLowerLow AND (NOT dxyMadeLowerLow OR NOT eurgbpMadeHigherHigh)
    ⚠️ この式は母集団(v4)のショート自己条件（own.buy_swept＝高値更新→MSS下）と矛盾する
       （'usdjpyMadeLowerLow' は逆方向）。まず文字通り実装し、結果を見てからフラグを立てる
       （measure契約: 仕様のバグを疑っても書かれた通りに実装してから報告）。
  DXY draw オーバーレイ: EUR/GBPショート=DXY draw=UP（discount, pos<0.5-band）の日だけ。
               USDJPYショート=DXY draw=DOWN（premium, pos>0.5+band）の日だけ。
               draw の日足位置は既存 pd_frame/join_days をDXYのdfに対して再利用（poscol=pos10, band=0.20固定）。
  ablation: (a) long-only基準 → (b) 無ゲートshort=死の再現 → (c) SMT単体(pair OR dxy) →
            (d) DXY draw単体 → (e) 両方(SMT AND draw)。各段 ゼロ/現実/保守(spread+2pip)の3コスト。
  ロング合流: EURUSDロング base → +EUR/GBP bullish SMT（own.sell_swept AND NOT partner.sell_swept）。

Run: .venv/bin/python scratchpad/ict_dxy_smt.py [--smoke] 2>&1 | tee scratchpad/out_ict_dxy_smt.txt
"""
import sys, argparse
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.data_loader import load_mt5_csv
from breakout_wave import resample as bw_resample

from ict_exec import MODEL, PIP, BUF, F_CANON, RR_CANON, walk, sc, prep
from ict_population import canonical_setups, trade_pool, load_prepped
from ict_gates import sweep_frame, smt_short_gate, pd_frame, join_days
from ict_audit import random_drop_null, block_boot

ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]
DXY_PATH = "data/vantage_usdx.r_m5.csv"
POSCOL, BAND = "pos10", 0.20     # DXY draw の日足lookback/バンド（既存 discount-long 台帳の flagship と同一値に固定）


def eras_of(tr):
    return " ".join(f"{sum(x[1] for x in tr if a <= pd.Timestamp(x[0]).year <= b):+6.0f}"
                    if any(a <= pd.Timestamp(x[0]).year <= b for x in tr) else "   n/a"
                    for a, b in ERAS)


# --------------------------------------------------------------------------- DXY ロード
def load_dxy_prepped(smoke=False):
    """DXY m5 → m15 リサンプル → 他銘柄と同一の NY壁時計パイプライン（ict_exec.load_ny と同型）。"""
    raw = load_mt5_csv(DXY_PATH)
    if smoke:
        raw = raw.loc[:"2022-01-01"]
    r15 = bw_resample(raw, "15min")
    naive_idx = r15.index.tz_localize(None)
    riga_idx = naive_idx.tz_localize("Europe/Riga", ambiguous="NaT", nonexistent="shift_forward")
    nat_mask = riga_idx.isna()
    if nat_mask.any():
        r15 = r15.loc[~nat_mask].copy()
        naive_idx = naive_idx[~nat_mask]; riga_idx = riga_idx[~nat_mask]
    else:
        r15 = r15.copy()
    ny_wall = riga_idx.tz_convert("America/New_York").tz_localize(None)
    r15["broker_dt"] = naive_idx
    r15["ny_wall"] = ny_wall
    r15 = r15.reset_index(drop=True)
    r15["atr14"] = ta.atr(r15["high"], r15["low"], r15["close"], length=14).values
    df, tarr, dates = prep(r15)
    return df, tarr, dates


# --------------------------------------------------------------------------- SMT/DXYゲート
def smt_full_gate_negcorr(short_pool, partner_sweep, dxy_sweep):
    """EUR/GBP型（DXY負相関）: shortAllowed = pairSMT OR dxySMT。own.buy_sweptは母集団で既に真。"""
    out = []
    for d, net in short_pool.items():
        sp_ = partner_sweep.get(d); sd = dxy_sweep.get(d)
        pair_ok = (sp_ is not None) and (not sp_[0])
        dxy_ok = (sd is not None) and (not sd[1])
        if pair_ok or dxy_ok:
            out.append((d, net))
    return out


def smt_dxy_gate_usdjpy_literal(short_pool, own_sweep, dxy_sweep, partner_sweep):
    """USDJPY型：仕様書の式を一字一句（own条件が母集団の自己条件と逆向きのため、期待どおり空になり得る）。
    usdjpyMadeLowerLow AND (NOT dxyMadeLowerLow OR NOT eurgbpMadeHigherHigh)"""
    out = []
    for d, net in short_pool.items():
        so = own_sweep.get(d); sd = dxy_sweep.get(d); sp_ = partner_sweep.get(d)
        if so is None or sd is None or sp_ is None:
            continue
        _, own_sell = so
        _, dxy_sell = sd
        partner_buy, _ = sp_
        if own_sell and (not dxy_sell or not partner_buy):
            out.append((d, net))
    return out


def gate_dxy_draw(pool_side, J_dxy, poscol, band, direction):
    """draw=up ⟺ DXYがdiscount(pos<0.5-band)、draw=down ⟺ premium(pos>0.5+band)。"""
    out = []
    for d, net in pool_side.items():
        if d not in J_dxy.index:
            continue
        pos = J_dxy.loc[d, poscol]
        if pd.isna(pos):
            continue
        if direction == "up" and pos < 0.5 - band:
            out.append((d, net))
        elif direction == "down" and pos > 0.5 + band:
            out.append((d, net))
    return out


def combine_and(list_c, list_d):
    dd = dict(list_d)
    return [(d, net) for d, net in list_c if d in dd]


def smt_bullish_gate(long_pool, partner_sweep):
    """ロング合流（EUR/GBP正相関）: own.sell_swept(母集団で既に真) AND NOT partner.sell_swept。"""
    out = []
    for d, net in long_pool.items():
        sw = partner_sweep.get(d)
        if sw is None:
            continue
        _, sell_swept = sw
        if not sell_swept:
            out.append((d, net))
    return out


# --------------------------------------------------------------------------- コスト3段
def cost_tiers(name):
    sp, cost = MODEL[name]
    pip = PIP[name]
    return {"zero": (0.0, 0.0), "realistic": (sp, cost), "conservative": (sp + 2 * pip, cost)}


def pools_by_tier(df, S, name, side):
    """同じ setups から3コスト段の {date: net} を作る（spreadは約定条件にも効くので walk を3回回す）。"""
    out = {}
    for tier, (sp, cost) in cost_tiers(name).items():
        out[tier] = {d: net for (d, net, g, risk) in walk(df, S, F_CANON, RR_CANON, BUF, sp, cost, side)}
    return out


def report_cell(label, tr, base_short_for_null=None):
    s = sc(tr, minn=15)
    if s is None:
        return f"{label:34s}  n<15 ({len(tr)}本)"
    win = 100 * np.mean([1 for d, n in tr]) if False else None
    row = f"{label:34s}  n={s['n']:5d}  PF={s['pf']:5.2f}  net={s['net']:+7.3f}  totR/DD={s['rdd']:6.2f}"
    if base_short_for_null is not None and s['n'] >= 15:
        nul = random_drop_null(base_short_for_null, s['n'])
        pc = 100 * (nul < s['rdd']).mean()
        row += f"  間引%ile={pc:4.0f}%"
    return row


# =========================================================================== メイン
def main(smoke=False):
    print("=" * 130)
    print("STEP 0: 検算アンカー — 無ゲートshortの死を再現 + 相方ペアscaffoldが ict_smt.py と一致するか")
    print("=" * 130)
    PAIRS = [("eurusd", "gbpusd"), ("gbpusd", "eurusd"), ("audusd", "nzdusd"), ("nzdusd", "audusd")]
    cache = {}
    names_needed = sorted(set([p for pair in PAIRS for p in pair] + ["usdjpy"]))
    for name in names_needed:
        df, tarr, dates, span = load_prepped(name)
        S0 = canonical_setups(df, tarr, dates, 0)
        pool0 = trade_pool(df, S0, name)
        sw0 = sweep_frame(df, tarr, dates, 0)
        cache[name] = dict(df=df, tarr=tarr, dates=dates, span=span, S0=S0, pool0=pool0, sw0=sw0)

    match_ok = True
    for X, Y in PAIRS:
        cx, cy = cache[X], cache[Y]
        short_pool = cx["pool0"]["short"]
        base_short = [(d, short_pool[d]) for d in short_pool]
        smt = smt_short_gate(short_pool, cy["sw0"])
        bS, s = sc(base_short), sc(smt)
        print(f"  {X}({Y}): baseS n={bS['n']} rdd={bS['rdd']:+.2f} PF={bS['pf']:.2f}  |  "
              f"pairSMT n={s['n']} rdd={s['rdd']:+.2f} PF={s['pf']:.2f}")
        if bS['pf'] >= 1.0:
            match_ok = False
    print(f"  -> 全 baseS が PF<1（死の再現）: {'OK' if match_ok else 'NG（要確認）'}")
    print("  -> pairSMT の数値は ict_smt.py の自己検査出力（scratchpad/out_ict_smt_selfcheck.txt）と手動照合すること")

    # USDJPY 単独の base short（PAIRS に無いので明示測定）
    cu = cache["usdjpy"]
    us_short = cu["pool0"]["short"]
    us_base = [(d, us_short[d]) for d in us_short]
    us_b = sc(us_base)
    print(f"  usdjpy: baseS n={us_b['n']} rdd={us_b['rdd']:+.2f} PF={us_b['pf']:.2f} net={us_b['net']:+.3f}")

    # --------------------------------------------------------------------- DXY ロード
    print("\n" + "=" * 130)
    print("STEP 1: DXY データロード + sweep_frame/draw構築")
    print("=" * 130)
    dxy_df, dxy_tarr, dxy_dates = load_dxy_prepped(smoke=smoke)
    print(f"  DXY m15 (m5→resample): bars={len(dxy_df)}  span={pd.Timestamp(dxy_df['broker_dt'].min())}"
          f" .. {pd.Timestamp(dxy_df['broker_dt'].max())}")
    dxy_sweep = sweep_frame(dxy_df, dxy_tarr, dxy_dates, 0)
    dxy_pd = pd_frame(dxy_df)

    # --------------------------------------------------------------------- ablation a-e (主軸)
    PRIMARY = [
        ("eurusd", "gbpusd", "negcorr", "up"),
        ("gbpusd", "eurusd", "negcorr", "up"),
        ("usdjpy", None, "usdjpy_literal", "down"),
    ]
    print("\n" + "=" * 130)
    print("STEP 2: ablation (a)long基準 (b)無ゲートshort (c)SMT単体 (d)DXYdraw単体 (e)両方 — 3コスト段")
    print("        (poscol=%s, band=%.2f 固定。DXYデータ span=%s〜)" %
          (POSCOL, BAND, pd.Timestamp(dxy_df["broker_dt"].min()).date()))
    print("=" * 130)

    for name, partner, kind, draw_dir in PRIMARY:
        df, tarr, dates, span = load_prepped(name)
        S0 = canonical_setups(df, tarr, dates, 0)
        own_sweep = sweep_frame(df, tarr, dates, 0)
        partner_sweep = cache[partner]["sw0"] if partner else None
        J_dxy = join_days(sorted(set(dates)), dxy_pd)

        tiers_long = pools_by_tier(df, S0, name, "long")
        tiers_short = pools_by_tier(df, S0, name, "short")

        print(f"\n--- {name.upper()} (相方={partner}) ---")
        for tier in ("zero", "realistic", "conservative"):
            long_pool = tiers_long[tier]
            short_pool = tiers_short[tier]
            base_long = [(d, long_pool[d]) for d in long_pool]
            base_short = [(d, short_pool[d]) for d in short_pool]

            if kind == "negcorr":
                c_gate = smt_full_gate_negcorr(short_pool, partner_sweep, dxy_sweep)
            else:
                c_gate = smt_dxy_gate_usdjpy_literal(short_pool, own_sweep, dxy_sweep, partner_sweep_all_eurgbp(cache, dates))

            d_gate = gate_dxy_draw(short_pool, J_dxy, POSCOL, BAND, draw_dir)
            e_gate = combine_and(c_gate, d_gate)

            print(f"  [{tier:12s}] " + report_cell("(a)long-base", base_long))
            print(f"  [{tier:12s}] " + report_cell("(b)無ゲートshort", base_short))
            print(f"  [{tier:12s}] " + report_cell("(c)SMT単体", c_gate, base_short))
            print(f"  [{tier:12s}] " + report_cell("(d)DXYdraw単体", d_gate, base_short))
            print(f"  [{tier:12s}] " + report_cell("(e)SMT AND draw", e_gate, base_short))

        # ブロック・ブートストラップ + 時代別（realistic コストのみ）
        short_pool_r = tiers_short["realistic"]
        base_short_r = [(d, short_pool_r[d]) for d in short_pool_r]
        if kind == "negcorr":
            c_r = smt_full_gate_negcorr(short_pool_r, partner_sweep, dxy_sweep)
        else:
            c_r = smt_dxy_gate_usdjpy_literal(short_pool_r, own_sweep, dxy_sweep, partner_sweep_all_eurgbp(cache, dates))
        d_r = gate_dxy_draw(short_pool_r, J_dxy, POSCOL, BAND, draw_dir)
        e_r = combine_and(c_r, d_r)
        for label, tr in (("(b)無ゲート", base_short_r), ("(c)SMT単体", c_r), ("(d)DXYdraw単体", d_r), ("(e)両方", e_r)):
            if len(tr) < 20:
                print(f"    {label:14s} n<20 ({len(tr)}本) — ブロック/時代別スキップ")
                continue
            bbs = "/".join(f"{block_boot(tr, m):.0f}" for m in (1, 3, 6, 12))
            print(f"    {label:14s} n={len(tr):4d} ブロック1/3/6/12={bbs:>15}%  時代別={eras_of(tr)}")

    # --------------------------------------------------------------------- プラセボ窓 (+4/+8/+12h)
    print("\n" + "=" * 130)
    print("STEP 3: プラセボ窓（本物の窓 vs +4/+8/+12h偽窓、realisticコスト・生存候補=(e)のみ）")
    print("=" * 130)
    for name, partner, kind, draw_dir in PRIMARY:
        row = [f"{name:8s}"]
        for sh in (0, 4, 8, 12):
            df, tarr, dates, span = load_prepped(name)
            Ssh = canonical_setups(df, tarr, dates, sh)
            sp, cost = MODEL[name]
            short_pool_sh = {d: net for (d, net, g, risk) in walk(df, Ssh, F_CANON, RR_CANON, BUF, sp, cost, "short")}
            own_sweep_sh = sweep_frame(df, tarr, dates, sh)
            dxy_sweep_sh = sweep_frame(dxy_df, dxy_tarr, dxy_dates, sh)
            J_dxy_sh = join_days(sorted(set(dates)), dxy_pd)
            partner_sweep_sh = sweep_frame(cache[partner]["df"], cache[partner]["tarr"], cache[partner]["dates"], sh) if partner else None
            if kind == "negcorr":
                c_sh = smt_full_gate_negcorr(short_pool_sh, partner_sweep_sh, dxy_sweep_sh)
            else:
                eurgbp_sh = partner_sweep_all_eurgbp(cache, dates, sh)
                c_sh = smt_dxy_gate_usdjpy_literal(short_pool_sh, own_sweep_sh, dxy_sweep_sh, eurgbp_sh)
            d_sh = gate_dxy_draw(short_pool_sh, J_dxy_sh, POSCOL, BAND, draw_dir)
            e_sh = combine_and(c_sh, d_sh)
            s = sc(e_sh, minn=10)
            row.append(f"+{sh:>2}h: " + (f"n={s['n']:4d} net={s['net']:+.3f}" if s else f"n={len(e_sh):3d} n<10"))
        print("  " + "  |  ".join(row))

    # --------------------------------------------------------------------- ロング合流
    print("\n" + "=" * 130)
    print("STEP 4: ロング合流 — EURUSDロング base → +EUR/GBP bullish SMT（追加コスト0）")
    print("=" * 130)
    ce = cache["eurusd"]
    long_pool = ce["pool0"]["long"]
    base_long = [(d, long_pool[d]) for d in long_pool]
    bull = smt_bullish_gate(long_pool, cache["gbpusd"]["sw0"])
    bL, bg = sc(base_long), sc(bull)
    print(f"  base EURUSDロング     : n={bL['n']} PF={bL['pf']:.2f} net={bL['net']:+.3f} totR/DD={bL['rdd']:.2f}")
    if bg:
        nul = random_drop_null(base_long, bg['n'])
        pc = 100 * (nul < bg['rdd']).mean()
        bbs = "/".join(f"{block_boot(bull, m):.0f}" for m in (1, 3, 6, 12))
        print(f"  +EUR/GBP bullishSMT   : n={bg['n']} PF={bg['pf']:.2f} net={bg['net']:+.3f} totR/DD={bg['rdd']:.2f}"
              f"  間引%ile={pc:.0f}%  ブロック1/3/6/12={bbs}")
        print(f"  時代別 base : {eras_of(base_long)}")
        print(f"  時代別 +SMT : {eras_of(bull)}")
    else:
        print(f"  +EUR/GBP bullishSMT   : n<15 ({len(bull)}本)")


def partner_sweep_all_eurgbp(cache, dates, shift=0):
    """USDJPY用: 'eu/gbp made higher high' を EURUSD/GBPUSD いずれかの buy_swept の OR として近似
    （ドキュメントは "eu/gbp made higher high" と単数扱いで書いており、EUR/GBPは正相関で同方向に
    動くことが多いため、どちらかが高値更新すれば "eu/gbp陣営が高値を取った" とみなすOR結合とした。
    これも仕様の曖昧点＝報告でフラグを立てる）。"""
    eu = cache["eurusd"]; gb = cache["gbpusd"]
    eu_sw = eu["sw0"] if shift == 0 else sweep_frame(eu["df"], eu["tarr"], eu["dates"], shift)
    gb_sw = gb["sw0"] if shift == 0 else sweep_frame(gb["df"], gb["tarr"], gb["dates"], shift)
    out = {}
    all_days = set(eu_sw) | set(gb_sw)
    for d in all_days:
        e = eu_sw.get(d); g = gb_sw.get(d)
        e_buy = e[0] if e is not None else None
        g_buy = g[0] if g is not None else None
        if e_buy is None and g_buy is None:
            out[d] = None
        else:
            out[d] = (bool(e_buy) or bool(g_buy), False)   # sell_swept 未使用（ダミー False）
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
