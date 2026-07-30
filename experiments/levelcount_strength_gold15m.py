"""gold15m BO の強度ゲージ検定 —「エントリー価格より上に残っている期間高値の数」。

======================================================================================
前提（コンテキストを消しても読めるよう全部ここに書く）
======================================================================================
仮説: ブレイクで建てた時、頭上に残っている「期間高値」の数と、その本の伸びに関係があるか。
      素朴な読みは「少ない＝抵抗を食い終わっている＝良い」だが、**下記のとおり gold では
      その素朴版は既に否定されており、実測は逆U字（中間が最良）**。検定する仮説は
      「本数は単調な強度勾配になるか」で、素朴版が正しいことを期待して撃つのではない。

🚨 事前確率は否定側。撃つ前に必ずこの節を読むこと（2026-07-30 に台帳と照合した結果）。

  1. **gold では「頭上に抵抗が多いと失敗しやすい」は既に否定されている。**
     `docs/findings/m_pullback.md`（2026-07-04）: gold の PDH族は全形態クローズ済み。
     gold canon + PDHソフト0.5 は totR/yr 22.6→15.1 の劣後で、年別分解の機構は
     **「gold はレンジ内ブレイクが利益の本体」**（2022 レンジ内+33.4 vs 新値圏+0.4／
     2024 39.9 vs 19.7／2025 27.0 vs 2.5）。BTC と真逆で、**PDH/PDL族＝BTC専用で確定**。
     機構: BTC は新値圏の空中戦で走る、gold は階段状（レンジ→ブレイク→レンジ）に刻む。
  2. **本数で測っても単調ではなく逆U字だった**（Donchian20 gold 1H/4H, 2026-07-30。
     詳細は `docs/findings/m_breakout.md#overhead-level-count-gold`）:
       - 日足スイング高値の本数 … 0本(最高値圏) +0.074 / 1本 +0.361 / 2本 +0.337 /
         3-4本 +0.331 / 5+本 +0.074、Spearman +0.00 ＝**上が空でも詰まってもダメ**
       - 期間高値の数(前日/前週/前月) … 1H: 1本 +0.193 / **2本 +0.304** / 3本 +0.098、
         4H: **1本(n=35) +1.037** / 2本(n=133) +0.139 ＝**頂点の位置が1Hと4Hでずれる**
       - 一致しているのは「3本＝全部が頭上＝最悪」の一点だけ
       - ATR正規化距離・TF整合の二値化(1H R差 -0.195・10.2%ile)はいずれも棄却
  3. ∴ **本スクリプトは「新しいエッジを探す」実験ではない。**「gold は中程度の頭上供給を
     好む」という**既知の機構に分解能を足す**実験として撃つ。台帳の gold 検定は PDH の
     二値・ソフト二値だけで、**階段（何本頭上に残っているか）は測っていない**——逆U字が
     本物なら二値では原理的に見えない形なので、そこだけが非冗長な残余。
  4. gold15m を選ぶ理由は本数。gold_bo 1H = 29本/年、Donchian 4H = 22本/年 では
     バケットを切ると各層が n=26〜35 になり検定にならない。**gold15m は 44本/年**。

使い方の建付け: **フィルタではなく強度勾配**として検定する（ユーザーは 0.01 固定ではなく、
      強度を見て手でサイズを変える）。∴ 合格条件は「バケットを切ると meanR/PF/totR が
      単調に上がる」こと。ONE-OFF の高いバケットが1つあるだけでは不合格。

======================================================================================
設計上の注意（ここを外すと偽陽性が出る）
======================================================================================
1. 🚨 15m はエントリーが日内で固まる。1本単位で順列帰無を回すと有意に見えすぎる。
   → **帰無は日ブロックで振る**（日ごとのRベクトルを日どうしで入れ替え、バケット標識は固定）。
   → ブロック・ブートストラップも 1/3/6/12 か月で回す（真の改善はブロックを長くすると上がる）。
2. 期間高値は必ず**確定した前期間**を使う（当日の高値を使うと先読み）。1時間・日・週・月の
   4段すべて resample→shift(1)→asof で 15m 足に貼る。
3. gold15m レッグは既に ext_cap 8%（前日終値が日足SMA150 から8%以上乖離したら建てない）を
   持っている。「上に期間高値が少ない＝伸び切っている」と ext は交絡しうるので、必ず相関を出す。
4. レッグ本体は research/book.py の gold15m と同一仕様を import して作る（手書き再構築は禁止）。
   仕様: 15m(m5から生成, 2018-09-14以降)・Pattern-B確定足ブレイク・押し目指値 0.25・
   fill_win 200・RR4・日足SMA150↑・ext_cap 8.0・ネットコスト $0.3/risk。

実行: .venv/bin/python experiments/levelcount_strength_gold15m.py
"""
SCREEN = "levelcount_strength_gold15m"

import os
import sys
import contextlib
import io
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.data_loader import load_mt5_csv
from breakout_wave import run, resample
from src.engine.presets import BASE
from research.screen import run_screen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG_SEED = 20260730
N_PERM = 2000
BLOCK_MONTHS = [1, 3, 6, 12]
N_BOOT = 1000

# 期間高値の階段（15m 用。前日/前週/前月では 15m には粗すぎるので 1時間を足した4段）
LADDER = [("直前1時間高値", "1h"), ("前日高値", "1D"), ("前週高値", "1W"), ("前月高値", "1ME")]


# ----------------------------------------------------------------------
# レッグ構築（research/book.py の gold15m と同一仕様）
# ----------------------------------------------------------------------

def build_gold15m():
    with contextlib.redirect_stderr(io.StringIO()):
        g15 = resample(load_mt5_csv(f"{ROOT}/data/vantage_xauusd_m5.csv").loc["2018-09-14":], "15min")
        t = run(g15, SimpleNamespace(**{**BASE, "daily_sma": 150, "daily_slope_k": 10,
                                        "ext_cap": 8.0, "pullback_frac": 0.25, "fill_win": 200}))
    t = t.copy()
    t["Rnet"] = t["R"].values - 0.3 / t["risk"].values      # ネットコスト $0.3
    t["time"] = pd.DatetimeIndex(t["time"])
    return g15, t


# ----------------------------------------------------------------------
# 頭上に残っている期間高値の本数（0〜4）— 全段 shift(1) で確定期間のみ
# ----------------------------------------------------------------------

def overhead_counts(df15, entry_times, entry_px):
    cols = {}
    for label, rule in LADDER:
        prev_high = df15["high"].resample(rule).max().shift(1).dropna()
        lvl = prev_high.reindex(df15.index, method="ffill").reindex(entry_times).values
        cols[label] = lvl
    lv = pd.DataFrame(cols, index=entry_times)
    above = lv.gt(pd.Series(entry_px, index=entry_times), axis=0)
    return above.sum(axis=1).values, lv


# ----------------------------------------------------------------------
# バケット統計
# ----------------------------------------------------------------------

def pf(a):
    gw, gl = a[a > 0].sum(), -a[a < 0].sum()
    return float(gw / gl) if gl > 0 else float("inf")


def bucket_table(R, cnt, label):
    print(f"\n--- {label} ---")
    print("  上の期間高値  n     n/年   勝率    meanR    PF      totR")
    rows = []
    for c in range(len(LADDER) + 1):
        m = cnt == c
        if m.sum() == 0:
            print(f"  {c}本         (該当なし)")
            continue
        r = R[m]
        rows.append((c, r.mean()))
        print(f"  {c}本      {m.sum():5d}  {m.sum()/YRS:6.1f}  {(r>0).mean()*100:5.1f}%  "
              f"{r.mean():+7.3f}  {pf(r):6.2f}  {r.sum():+8.1f}")
    if len(rows) >= 3:
        cs = np.array([x[0] for x in rows], float)
        ms = np.array([x[1] for x in rows], float)
        rho = float(pd.Series(cs).corr(pd.Series(ms), method="spearman"))
        print(f"  バケット順位とmeanRのSpearman = {rho:+.2f}  "
              f"(単調に下がる＝仮説どおりなら負)")
    return rows


# ----------------------------------------------------------------------
# 日ブロック順列帰無（バケット標識は固定、日ごとのRベクトルを日どうしで入れ替える）
# ----------------------------------------------------------------------

def day_block_permutation(R, cnt, times, rng, stat_fn):
    days = pd.DatetimeIndex(times).normalize()
    uniq = pd.unique(days)
    groups = [np.where(days == d)[0] for d in uniq]
    obs = stat_fn(R, cnt)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        order = rng.permutation(len(groups))
        Rp = np.empty_like(R)
        pos = 0
        for gi in order:
            g = groups[gi]
            Rp[pos:pos + len(g)] = R[g]
            pos += len(g)
        null[k] = stat_fn(Rp, cnt)
    pct = float((null < obs).mean() * 100)
    return obs, null, pct


def stat_lowhigh(R, cnt):
    """「上が少ない層」−「上が多い層」の meanR 差。仮説が正なら正。"""
    lo = R[cnt <= 1]
    hi = R[cnt >= 3]
    if len(lo) == 0 or len(hi) == 0:
        return np.nan
    return float(lo.mean() - hi.mean())


# ----------------------------------------------------------------------
# 月ブロック・ブートストラップ（強度傾斜サイズ vs 均等サイズ、同じ経路上で比較）
# ----------------------------------------------------------------------

def cagr_dd(vals, days, risk):
    eq = np.cumprod(1 + risk * vals)
    pk = np.maximum.accumulate(eq)
    dd = ((pk - eq) / pk).max() * 100
    if dd <= 0 or days <= 0:
        return np.nan, np.nan, np.nan
    cagr = (eq[-1] ** (365.25 / days) - 1) * 100
    return cagr, dd, cagr / dd


def block_boot_ratio(R_flat, R_grad, times, rng):
    """1/3/6/12か月ブロックで「傾斜版が均等版の CAGR/DD を上回る割合」。
    真の改善はブロックを長くするほど上がる（経路当てはめは上がらない）。"""
    t = pd.DatetimeIndex(times)
    out = {}
    for bm in BLOCK_MONTHS:
        span_days = (t[-1] - t[0]).days
        blk = max(int(round(bm * 30.44)), 1)
        nb = int(np.ceil(span_days / blk))
        wins = []
        starts = pd.date_range(t[0], t[-1] - pd.Timedelta(days=blk), freq="D")
        if len(starts) < 2:
            continue
        for _ in range(N_BOOT):
            idxs = []
            for s in rng.choice(len(starts), size=nb):
                a = starts[s]
                b = a + pd.Timedelta(days=blk)
                idxs.append(np.where((t >= a) & (t < b))[0])
            sel = np.concatenate(idxs) if idxs else np.array([], int)
            if len(sel) < 20:
                continue
            d = blk * nb
            _, _, r0 = cagr_dd(R_flat[sel], d, 0.01)
            _, _, r1 = cagr_dd(R_grad[sel], d, 0.01)
            if np.isfinite(r0) and np.isfinite(r1):
                wins.append(r1 > r0)
        if wins:
            out[bm] = float(np.mean(wins) * 100)
    return out


# ----------------------------------------------------------------------

def main():
    global YRS
    rng = np.random.default_rng(RNG_SEED)
    df15, t = build_gold15m()
    YRS = (t["time"].iloc[-1] - t["time"].iloc[0]).days / 365.25
    R = t["Rnet"].values
    times = t["time"].values

    # 🔒 巡行幅スクリーン（フックの要求。トレード統計の前に必ず通す）
    entries = [(pd.Timestamp(tm), 1, float(px), float(px - rk))
               for tm, px, rk in zip(t["time"], t["e_px"], t["risk"])]
    run_screen(SCREEN, df15, entries, quiet=True)

    cnt, lv = overhead_counts(df15, pd.DatetimeIndex(t["time"]), t["e_px"].values)

    print("=" * 96)
    print(f"gold15m BO / 上に残る期間高値の本数（0〜4）  n={len(t)} ({len(t)/YRS:.1f}本/年) "
          f"期間 {t['time'].iloc[0].date()}→{t['time'].iloc[-1].date()} ({YRS:.2f}年)")
    print(f"素の成績: meanR={R.mean():+.3f} PF={pf(R):.2f} 勝率={(R>0).mean()*100:.1f}% "
          f"totR={R.sum():+.1f}")
    print("=" * 96)

    bucket_table(R, cnt, "全期間")

    # 年別
    print("\n--- 年別（バケット別 meanR / n） ---")
    yr = pd.DatetimeIndex(t["time"]).year
    for y in sorted(set(yr)):
        m = yr == y
        cells = []
        for c in range(len(LADDER) + 1):
            mm = m & (cnt == c)
            cells.append(f"{c}本:{R[mm].mean():+.2f}(n{mm.sum()})" if mm.sum() else f"{c}本:  -   ")
        print(f"  {y}: " + "  ".join(cells))

    # 日ブロック順列帰無
    obs, null, pctile = day_block_permutation(R, cnt, times, rng, stat_lowhigh)
    print(f"\n--- 日ブロック順列帰無（上0-1本 − 上3-4本 の meanR 差） ---")
    print(f"  実測差 = {obs:+.3f}   帰無中央値 = {np.median(null):+.3f}  "
          f"σ = {null.std():.3f}   実測の位置 = {pctile:.1f}%ile")
    print(f"  合格の目安: 95%ile 以上。50%前後なら勾配は無い。")

    # 強度傾斜サイズ vs 均等サイズ（同じ経路上）
    w = np.clip(len(LADDER) - cnt, 0, None).astype(float)   # 上が少ないほど大きい
    w = w / w.mean() if w.mean() > 0 else np.ones_like(w)
    R_grad, R_flat = R * w, R
    for lab, arr in (("均等", R_flat), ("傾斜", R_grad)):
        c, d, r = cagr_dd(arr, (pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).days, 0.01)
        print(f"\n  {lab}サイズ: CAGR={c:+.1f}% maxDD={d:.2f}% CAGR/DD={r:.2f} "
              f"(トレード解像度・risk1%)")
    print("  ⚠️ 傾斜はRのばらつきを変えるので、同じ maxDD に揃えないと比較にならない")
    print("     （CAGR/DD の改善はレバレッジ購入でありうる＝法則7.5）")

    bb = block_boot_ratio(R_flat, R_grad, times, rng)
    print(f"\n--- 月ブロック・ブートストラップ（傾斜が均等の CAGR/DD を上回る割合） ---")
    for bm in BLOCK_MONTHS:
        if bm in bb:
            print(f"  {bm:2d}か月ブロック: {bb[bm]:5.1f}%")
    print("  合格の形: ブロックを長くするほど上がる。下がる/横ばいなら経路当てはめ。")

    # 交絡点検
    print(f"\n--- 交絡（本数と他の変数の相関） ---")
    conf = pd.DataFrame({
        "上の本数": cnt,
        "risk($)": t["risk"].values,
        "保有日数": t["hold"].values,
        "建値": t["e_px"].values,
    })
    print(conf.corr(method="spearman")["上の本数"].round(3).to_string())
    print("\n  ⚠️ gold15m は ext_cap 8% を既に持つ。『上に高値が少ない＝伸び切っている』と")
    print("     交絡していないかは、上の建値・risk との相関で見る（強い相関が出たら別変数）。")

    print(f"\n[判定の書き方] 合格 = (1) バケットが単調 (2) 順列帰無 95%ile 以上")
    print(f"              (3) 月ブロックがブロック長とともに上昇 (4) 交絡が弱い。")
    print(f"              1つでも欠けたら NON-ADOPTED として台帳に落とし、")
    print(f"              『なぜ落ちたか＝病巣』を書いてから次の実験を設計する。")


if __name__ == "__main__":
    main()
