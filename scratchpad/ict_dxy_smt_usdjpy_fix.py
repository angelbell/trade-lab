"""ICT 忠実化・優先2 — USDJPY DXY-SMT の訂正版（コーディネーター指示・2026-07-15）。

バグ: 正典の式 `usdjpyMadeLowerLow AND (...)` は v4 母集団のショート自己条件
（own.buy_swept＝高値を狩ってMSS下）と逆向きだった（`ict_dxy_smt.py` の smt_dxy_gate_usdjpy_literal
は文字通り実装した結果 n=2〜3 本でほぼ空、というフラグをそのまま裏付けた）。

訂正（母集団は書き換えない。own.buy_swept は既に short_pool の構築条件＝自明に真）:
  DXY版    : usdjpy_shortAllowed = own.buy_swept AND (NOT dxy.buy_swept)
             （DXYはUSDJPYと正相関 → ドルが同じ窓で高値更新を確認しない＝ダイバージェンス）
  EUR/GBP版: usdjpy_shortAllowed = own.buy_swept AND (NOT eurgbp.sell_swept)
             （EUR/GBPはDXY・USDJPYと逆相関 → ドル安側が安値更新を拒否＝ドル高が本物でない）
  eurgbp_combo_sweep: 「eu/gbp」を単数扱いする仕様書の曖昧さは ict_dxy_smt.py と同じ処理
             （EURUSD/GBPUSD の buy_swept/sell_swept を OR で合成）を踏襲。
  DXY draw (d) は既存のまま（USDJPY ショート = DXY premium/draw下 の日だけ）。

対象は USDJPY のみ。realistic/conservative の2コスト段、n/PF/net/totR-DD/間引%ile/
ブロック1-3-6-12/時代別/プラセボ窓(+4/8/12h)。他銘柄・ロング合流は既に決着済みにつき再走しない。

Run: .venv/bin/python scratchpad/ict_dxy_smt_usdjpy_fix.py 2>&1 | tee scratchpad/out_ict_dxy_smt_usdjpy_fix.txt
"""
import sys
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ict_exec import MODEL, BUF, F_CANON, RR_CANON, walk, sc
from ict_population import canonical_setups, trade_pool, load_prepped
from ict_gates import sweep_frame, pd_frame, join_days
from ict_audit import random_drop_null, block_boot
from ict_dxy_smt import (load_dxy_prepped, gate_dxy_draw, cost_tiers, pools_by_tier,
                          eras_of, report_cell, combine_and, POSCOL, BAND)

NAME = "usdjpy"


def eurgbp_combo_sweep(shift=0):
    """EUR/GBP を「相方」として OR 合成（仕様書の "eu/gbp" 単数表記の解釈は ict_dxy_smt.py と同一）。
    buy=どちらかが高値更新（買方向流動性の掃除） / sell=どちらかが安値更新。"""
    dfe, te, de, _ = load_prepped("eurusd")
    dfg, tg, dg, _ = load_prepped("gbpusd")
    eu_sw = sweep_frame(dfe, te, de, shift)
    gb_sw = sweep_frame(dfg, tg, dg, shift)
    out = {}
    for d in set(eu_sw) | set(gb_sw):
        e = eu_sw.get(d); g = gb_sw.get(d)
        if e is None and g is None:
            out[d] = None; continue
        e_buy, e_sell = e if e is not None else (None, None)
        g_buy, g_sell = g if g is not None else (None, None)
        buy = (bool(e_buy) if e_buy is not None else False) or (bool(g_buy) if g_buy is not None else False)
        sell = (bool(e_sell) if e_sell is not None else False) or (bool(g_sell) if g_sell is not None else False)
        out[d] = (buy, sell)
    return out


def smt_gate_usdjpy_dxy(short_pool, own_sweep, dxy_sweep):
    """訂正版DXY: own.buy_swept(母集団で既に真) AND NOT dxy.buy_swept。"""
    out = []
    for d, net in short_pool.items():
        so = own_sweep.get(d); sd = dxy_sweep.get(d)
        if so is None or sd is None:
            continue
        own_buy, _ = so
        dxy_buy, _ = sd
        if own_buy and not dxy_buy:
            out.append((d, net))
    return out


def smt_gate_usdjpy_eurgbp(short_pool, own_sweep, eurgbp_sweep):
    """訂正版EUR/GBP: own.buy_swept AND NOT eurgbp.sell_swept。"""
    out = []
    for d, net in short_pool.items():
        so = own_sweep.get(d); sp_ = eurgbp_sweep.get(d)
        if so is None or sp_ is None:
            continue
        own_buy, _ = so
        _, eurgbp_sell = sp_
        if own_buy and not eurgbp_sell:
            out.append((d, net))
    return out


def build_gates(short_pool, own_sweep, dxy_sweep, eurgbp_sweep, J_dxy):
    c_dxy = smt_gate_usdjpy_dxy(short_pool, own_sweep, dxy_sweep)
    c_eg = smt_gate_usdjpy_eurgbp(short_pool, own_sweep, eurgbp_sweep)
    d_gate = gate_dxy_draw(short_pool, J_dxy, POSCOL, BAND, "down")
    e_dxy = combine_and(c_dxy, d_gate)
    e_eg = combine_and(c_eg, d_gate)
    return c_dxy, c_eg, d_gate, e_dxy, e_eg


def main():
    df, tarr, dates, span = load_prepped(NAME)
    S0 = canonical_setups(df, tarr, dates, 0)
    own_sweep = sweep_frame(df, tarr, dates, 0)

    dxy_df, dxy_tarr, dxy_dates = load_dxy_prepped(smoke=False)
    dxy_sweep = sweep_frame(dxy_df, dxy_tarr, dxy_dates, 0)
    dxy_pd = pd_frame(dxy_df)
    eurgbp_sweep = eurgbp_combo_sweep(0)
    J_dxy = join_days(sorted(set(dates)), dxy_pd)

    print("=" * 130)
    print("USDJPY DXY-SMT 訂正版 ablation ―― own.buy_swept AND NOT dxy.buy_swept（DXY版）/")
    print("                                  own.buy_swept AND NOT eurgbp.sell_swept（EURGBP版）")
    print(f"  DXYデータ span = {pd.Timestamp(dxy_df['broker_dt'].min()).date()} 〜 "
          f"{pd.Timestamp(dxy_df['broker_dt'].max()).date()}")
    print("=" * 130)

    tiers_short = pools_by_tier(df, S0, NAME, "short")
    results_realistic = {}
    for tier in ("realistic", "conservative"):
        short_pool = tiers_short[tier]
        base_short = [(d, short_pool[d]) for d in short_pool]
        c_dxy, c_eg, d_gate, e_dxy, e_eg = build_gates(short_pool, own_sweep, dxy_sweep, eurgbp_sweep, J_dxy)
        if tier == "realistic":
            results_realistic = dict(base=base_short, c_dxy=c_dxy, c_eg=c_eg, d=d_gate, e_dxy=e_dxy, e_eg=e_eg)
        print(f"\n[{tier}]")
        print("  " + report_cell("(b)無ゲートshort", base_short))
        print("  " + report_cell("(c)SMT単体 DXY版", c_dxy, base_short))
        print("  " + report_cell("(c)SMT単体 EURGBP版", c_eg, base_short))
        print("  " + report_cell("(d)DXYdraw単体", d_gate, base_short))
        print("  " + report_cell("(e)両方 DXY版", e_dxy, base_short))
        print("  " + report_cell("(e)両方 EURGBP版", e_eg, base_short))

    print("\n--- ブロック・ブートストラップ 1/3/6/12か月 + 時代別（realisticコスト） ---")
    for label, tr in (("(b)無ゲート", results_realistic["base"]),
                      ("(c)DXY版", results_realistic["c_dxy"]),
                      ("(c)EURGBP版", results_realistic["c_eg"]),
                      ("(d)DXYdraw", results_realistic["d"]),
                      ("(e)DXY版", results_realistic["e_dxy"]),
                      ("(e)EURGBP版", results_realistic["e_eg"])):
        if len(tr) < 20:
            print(f"  {label:16s} n<20 ({len(tr)}本) — ブロック/時代別スキップ")
            continue
        bbs = "/".join(f"{block_boot(tr, m):.0f}" for m in (1, 3, 6, 12))
        print(f"  {label:16s} n={len(tr):4d} ブロック1/3/6/12={bbs:>15}%  時代別={eras_of(tr)}")

    print("\n--- プラセボ窓（本物 vs +4/+8/+12h、realisticコスト） ---")
    sp, cost = MODEL[NAME]
    for label_kind in ("c_dxy", "c_eg", "e_dxy", "e_eg"):
        row = [f"{label_kind:10s}"]
        for sh in (0, 4, 8, 12):
            dfsh, tsh, dsh, _ = load_prepped(NAME)
            Ssh = canonical_setups(dfsh, tsh, dsh, sh)
            short_pool_sh = {d: net for (d, net, g, risk) in walk(dfsh, Ssh, F_CANON, RR_CANON, BUF, sp, cost, "short")}
            own_sweep_sh = sweep_frame(dfsh, tsh, dsh, sh)
            dxy_sweep_sh = sweep_frame(dxy_df, dxy_tarr, dxy_dates, sh)
            eurgbp_sweep_sh = eurgbp_combo_sweep(sh)
            J_dxy_sh = join_days(sorted(set(dsh)), dxy_pd)
            c_dxy_sh, c_eg_sh, d_sh, e_dxy_sh, e_eg_sh = build_gates(
                short_pool_sh, own_sweep_sh, dxy_sweep_sh, eurgbp_sweep_sh, J_dxy_sh)
            tr = {"c_dxy": c_dxy_sh, "c_eg": c_eg_sh, "e_dxy": e_dxy_sh, "e_eg": e_eg_sh}[label_kind]
            s = sc(tr, minn=10)
            row.append(f"+{sh:>2}h: " + (f"n={s['n']:4d} net={s['net']:+.3f}" if s else f"n={len(tr):3d} n<10"))
        print("  " + "  |  ".join(row))


if __name__ == "__main__":
    main()
