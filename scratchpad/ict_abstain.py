"""ICT ステップ1 の正しい機械化 =「迷う日は何もしない」（2026-07-15）。

これまでの私の実装は、日足バイアスを「毎日どちらかに賭ける」形でしか測っていなかった。
ユーザーが動画で聞いたルールは **abstention（棄権）** である:
   日足が明確に上 → ロングのみ / 明確に下 → ショートのみ / **曖昧なら何もしない**
「どちらに動くか」を当てる必要はなく、「今日は触るな」が当たれば良い＝WHEN（レジーム選択）の問題。

母集団: v2 の生存形 = 狩り(sweep) + MSS + 浅い押し目 f=0.25 + RR4 + NYキルゾーン、
        執行は現実版（ASK基準の指値約定 0.3pip + 手数料のみ）。

方向の定義 (3): 日足KAMA(14)の向き / 日足SMA150の向き / 前日の陰陽
明確さのダイヤル (5): 閾値を上げるほど棄権が増える
   E1 日足ER(10)                      … 方向感の強さそのもの
   E2 |日足KAMAの傾き| / 日足ATR        … 横ばいか
   E3 |終値 - SMA150| / 日足ATR        … MAに絡んでいる = チョップ
   E4 前日足の実体比 |C-O| / (H-L)      … 十字線 = 迷い
   E5 終値の直近20日レンジ内の位置（中央からの距離）

審判（CLAUDE.md チェックリスト7: 選別ルールは運の選別機）:
   同じ本数をランダムに間引いた帰無と、**totR/DD** で比較（meanR では不可）。
   さらに閾値を上げるほど単調に良くなる「台地」か、一点だけの「スパイク」かを見る。
   時代別（2000-08 / 2009-16 / 2017-20 / 2021-26）も必ず出す。

Run: .venv/bin/python scratchpad/ict_abstain.py
"""
import sys, io, contextlib
sys.path.insert(0, "/home/angelbell/dev/auto-trade")
sys.path.insert(0, "/home/angelbell/dev/auto-trade/scratchpad")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pandas_ta as ta

from ict_killzone import load_ny, SYMS
from ict_v2_mss import prep, walk, MODEL
from ict_ablation import build, PIP, FILL_SPREAD_PIPS, BUF
from breakout_wave import resample, kama_adaptive

RNG = np.random.default_rng(20260715)
F, RR = 0.25, 4.0
QUANTS = [0.0, 0.2, 0.35, 0.5, 0.65]     # 棄権の閾値（下位 q を「曖昧」として捨てる）
ERAS = [(2000, 2008), (2009, 2016), (2017, 2020), (2021, 2026)]


def daily_frame(df):
    b = df.set_index("broker_dt")[["open", "high", "low", "close"]]
    d = resample(b, "1D")
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    atr = ta.atr(h, l, c, 14)
    kama = kama_adaptive(c, 14)
    sma = c.rolling(150).mean()
    ch = c.diff(10).abs()
    vol = c.diff().abs().rolling(10).sum()
    er = (ch / vol).replace([np.inf, -np.inf], np.nan)
    rng20 = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min())

    D = pd.DataFrame(index=d.index)
    # 方向 (True=強気)
    D["dir_kama"] = kama > kama.shift(1)
    D["dir_sma"] = sma > sma.shift(1)
    D["dir_body"] = c > o
    # 明確さ（大きいほど明確）
    D["E1_er"] = er
    D["E2_kslope"] = (kama - kama.shift(1)).abs() / atr
    D["E3_madist"] = (c - sma).abs() / atr
    D["E4_body"] = (c - o).abs() / (h - l).replace(0, np.nan)
    D["E5_rngpos"] = (rng20 - 0.5).abs() * 2
    # 前日までに確定 -> 翌ブローカー日の頭で有効
    conf = (D.index + pd.Timedelta(days=1)).tz_localize(
        "Europe/Riga", ambiguous="NaT", nonexistent="shift_forward"
    ).tz_convert("America/New_York").tz_localize(None)
    D["conf"] = conf
    return D[~D["conf"].isna()].sort_values("conf").reset_index(drop=True)


def join_days(dates, D):
    q = pd.DataFrame({"date": list(dates)})
    q["t"] = pd.to_datetime(q["date"]) + pd.Timedelta(hours=2)   # ロンドン窓の開始
    m = pd.merge_asof(q.sort_values("t"), D, left_on="t", right_on="conf", direction="backward")
    return m.set_index("date")


def trade_pool(df, setups, name):
    """両サイドの全トレードを日付キーで持つ（サイド選択は後で日足が決める）。"""
    sp = FILL_SPREAD_PIPS * PIP[name]; _, cost = MODEL[name]
    pool = {"long": {}, "short": {}}
    for side in ("long", "short"):
        for (d, net, g, risk) in walk(df, setups, F, RR, BUF, sp, cost, side):
            pool[side][d] = net
    return pool


def assemble(pool, J, dircol, clarcol, q):
    """日足が明確なときだけ、その方向に建てる。曖昧(clarity が下位 q)なら棄権。"""
    cl = J[clarcol]
    thr = cl.quantile(q) if q > 0 else -np.inf
    out = []
    for d, row in J.iterrows():
        v = row[dircol]
        if pd.isna(v):
            continue
        c = row[clarcol]
        if q > 0 and (pd.isna(c) or c < thr):
            continue                                   # 迷う日 -> 何もしない
        side = "long" if bool(v) else "short"
        if d in pool[side]:
            out.append((d, pool[side][d]))
    return out


def sc(tr):
    if len(tr) < 20:
        return None
    net = np.array([t[1] for t in tr])
    cum = np.cumsum(net)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    pos, neg = net[net > 0].sum(), -net[net < 0].sum()
    return dict(n=len(net), net=net.mean(), pf=pos / neg if neg > 0 else np.inf,
                tot=net.sum(), dd=dd, rdd=net.sum() / dd if dd > 0 else np.inf)


def random_drop_null(base, k, nrep=500):
    """base から k 本をランダムに残した totR/DD の分布（＝運の選別機との比較）"""
    net = np.array([t[1] for t in base])
    out = []
    for _ in range(nrep):
        idx = np.sort(RNG.choice(len(net), size=k, replace=False))
        x = net[idx]; cum = np.cumsum(x)
        dd = (np.maximum.accumulate(cum) - cum).max()
        out.append(x.sum() / dd if dd > 0 else np.inf)
    return np.array(out)


def main():
    for name in ("eurusd", "gbpusd", "gold", "btcusd"):
        with contextlib.redirect_stderr(io.StringIO()):
            df, _ = load_ny(SYMS[name])
        df, tarr, dates = prep(df)
        S = build(df, tarr, dates, True, True, "mss", 0)      # 狩り + MSS
        pool = trade_pool(df, S, name)
        D = daily_frame(df)
        J = join_days(sorted(set(list(pool["long"].keys()) + list(pool["short"].keys()))), D)

        print("\n" + "=" * 122)
        print(f"=== {name} ===  母集団: 狩り+MSS / 浅0.25 / RR4 / NYキルゾーン / ASK約定0.3pip")
        print("=" * 122)
        for dircol, dlab in (("dir_kama", "日足KAMA↑"), ("dir_sma", "日足SMA150↑"), ("dir_body", "前日陽線")):
            print(f"\n--- 方向 = {dlab} ---")
            print(f"  {'明確さの定義':18s} {'棄権':>5} {'n':>5} {'年':>4} {'net':>7} {'PF':>5} "
                  f"{'totR':>7} {'DD':>6} {'totR/DD':>8} {'null中央':>8} {'%ile':>5}  時代別 totR")
            base = assemble(pool, J, dircol, "E1_er", 0.0)
            b = sc(base)
            if b is None:
                print("   (母集団が薄い)"); continue
            eras_b = []
            for a1, a2 in ERAS:
                v = [x[1] for x in base if a1 <= pd.Timestamp(x[0]).year <= a2]
                eras_b.append(f"{sum(v):+6.0f}" if v else "   n/a")
            print(f"  {'(棄権なし＝ベース)':18s} {'0%':>5} {b['n']:5d} {b['n']/26.5:4.0f} "
                  f"{b['net']:+7.3f} {b['pf']:5.2f} {b['tot']:+7.1f} {b['dd']:6.1f} {b['rdd']:8.2f} "
                  f"{'':8s} {'':5s}  {' '.join(eras_b)}")
            for clarcol, clab in (("E1_er", "E1 日足ER(10)"), ("E2_kslope", "E2 KAMA傾き/ATR"),
                                  ("E3_madist", "E3 SMA150乖離/ATR"), ("E4_body", "E4 前日実体比"),
                                  ("E5_rngpos", "E5 20日レンジ位置")):
                for q in QUANTS[1:]:
                    tr = assemble(pool, J, dircol, clarcol, q)
                    s = sc(tr)
                    if s is None:
                        continue
                    nul = random_drop_null(base, s["n"])
                    pc = 100 * (s["rdd"] > nul).mean()
                    eras = []
                    for a1, a2 in ERAS:
                        v = [x[1] for x in tr if a1 <= pd.Timestamp(x[0]).year <= a2]
                        eras.append(f"{sum(v):+6.0f}" if v else "   n/a")
                    star = " *" if pc >= 90 else ""
                    print(f"  {clab:18s} {int(q*100):4d}% {s['n']:5d} {s['n']/26.5:4.0f} "
                          f"{s['net']:+7.3f} {s['pf']:5.2f} {s['tot']:+7.1f} {s['dd']:6.1f} "
                          f"{s['rdd']:8.2f} {np.median(nul):8.2f} {pc:4.0f}%  {' '.join(eras)}{star}")


if __name__ == "__main__":
    main()
