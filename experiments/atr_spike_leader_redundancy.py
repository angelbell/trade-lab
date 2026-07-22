"""「BTC同時」は新しい変数か、それとも既知の量（自分の実体の大きさ・時間帯・ボラ）の言い換えか。

由来: アルト9銘柄プールで BTC同時 +0.913% vs BTC静か +0.032%（帰無%ile 100・符号一致8/9・5年とも同符号）。
      2×2 で「広がり」説は倒れた（BTC静か×アルト2+ が最悪の −0.208%）。

残る対立仮説（記憶の規律: 新変数はまず単体の情報測定→冗長性→レッグ層別）:
  H1 自分の実体が大きいだけ  → 自分の body/ATR を3分位に切り、各層で差が残るか
  H2 時間帯の言い換え        → 時刻帯(4区分)で切り、各層で差が残るか
  H3 ボラ・レジームの言い換え → 自分の ATR/価格 を3分位に切り、各層で差が残るか
どれかで差が消えるなら言い換え。全層で残るなら独立した情報。
"""
SCREEN = "atr_spike_btc_h1"

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.atr_spike_btc_leader import load, spikes, leg, st, dropnull, ALTS   # noqa: E402


def wilder_atr(d, n=14):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


if __name__ == "__main__":
    sB = spikes(load("btcusd"))
    sB1 = (sB | sB.shift(1)).fillna(False)

    recs = []
    for sym in ALTS:
        d = load(sym)
        t = leg(d)
        if t is None:
            continue
        ap = wilder_atr(d).shift(1)
        body = (d["close"] - d["open"]) / ap          # 引き金足の実体（ATR単位）
        vol = ap / d["close"]                          # ボラ水準
        tt = t["time"]
        recs.append(pd.DataFrame({
            "sym": sym, "pct": t["pct"].to_numpy(), "y": tt.dt.year.values,
            "hour": tt.dt.hour.values,
            "body": body.reindex(tt).to_numpy(),
            "vol": vol.reindex(tt).to_numpy(),
            "btc": sB1.reindex(tt).fillna(False).to_numpy()}))
    R = pd.concat(recs, ignore_index=True).dropna(subset=["body", "vol"])
    print(f"母集団 N={len(R)}  BTC同時 {R['btc'].mean()*100:.0f}%")

    def cmp(sub, lab):
        a = sub.loc[sub["btc"], "pct"].to_numpy()
        b = sub.loc[~sub["btc"], "pct"].to_numpy()
        if len(a) < 20 or len(b) < 20:
            print(f"  {lab:<24} 本数不足 ({len(a)}/{len(b)})")
            return
        dn = dropnull(sub["pct"].to_numpy(), a)
        print(f"  {lab:<24} 同時 N={len(a):4d} PF={st(a)[2]:5.2f} {st(a)[3]:+.3f}%  |  "
              f"静か N={len(b):4d} PF={st(b)[2]:5.2f} {st(b)[3]:+.3f}%  |  "
              f"差={st(a)[3]-st(b)[3]:+.3f}%  帰無%ile={dn[0]:5.1f}")

    print("\n=== H1 自分の実体の大きさで層別（body/ATR の3分位）")
    q = R["body"].quantile([1/3, 2/3]).to_numpy()
    for lo, hi, lab in ((-np.inf, q[0], f"小 (<{q[0]:.2f}ATR)"),
                        (q[0], q[1], f"中 ({q[0]:.2f}-{q[1]:.2f})"),
                        (q[1], np.inf, f"大 (>{q[1]:.2f}ATR)")):
        cmp(R[(R["body"] >= lo) & (R["body"] < hi)], lab)
    print(f"  参考: 実体の平均 同時={R.loc[R['btc'],'body'].mean():.2f}ATR "
          f"静か={R.loc[~R['btc'],'body'].mean():.2f}ATR")

    print("\n=== H2 時間帯で層別（ブローカー時刻）")
    for lo, hi, lab in ((0, 6, "0-5時"), (6, 12, "6-11時"),
                        (12, 18, "12-17時"), (18, 24, "18-23時")):
        cmp(R[(R["hour"] >= lo) & (R["hour"] < hi)], lab)
    print(f"  参考: 同時の時刻中央値={R.loc[R['btc'],'hour'].median():.0f}時 "
          f"静か={R.loc[~R['btc'],'hour'].median():.0f}時")

    print("\n=== H3 ボラ水準で層別（ATR/価格 の3分位）")
    qv = R["vol"].quantile([1/3, 2/3]).to_numpy()
    for lo, hi, lab in ((-np.inf, qv[0], "低ボラ"), (qv[0], qv[1], "中ボラ"),
                        (qv[1], np.inf, "高ボラ")):
        cmp(R[(R["vol"] >= lo) & (R["vol"] < hi)], lab)

    print("\n=== 銘柄別（既に測ったが、層別の後に再掲）")
    for sym in ALTS:
        cmp(R[R["sym"] == sym], sym)

    print("\n=== 実務: ETH レッグ単独に「BTC同時」を課す（年別）")
    E = R[R["sym"] == "ethusd"]
    print(f"  {'年':>6} {'全体':>20} {'BTC同時のみ':>24}")
    for y in sorted(E["y"].unique()):
        a = E[E["y"] == y]
        b = a[a["btc"]]
        print(f"  {y:>6} N={len(a):3d} {a['pct'].mean()*100:+.3f}%   "
              f"N={len(b):3d} {b['pct'].mean()*100:+.3f}%" if len(b) else f"  {y:>6} --")

    n_layers_ok = 0
    for col, qs in (("body", q), ("vol", qv)):
        for lo, hi in ((-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)):
            s = R[(R[col] >= lo) & (R[col] < hi)]
            a, b = s.loc[s["btc"], "pct"], s.loc[~s["btc"], "pct"]
            if len(a) >= 20 and len(b) >= 20 and a.mean() > b.mean():
                n_layers_ok += 1
    assert len(R) > 1000, len(R)
    print(f"\nOK: 実体3層×ボラ3層のうち {n_layers_ok}/6 層で「BTC同時」が上回る（N={len(R)}）")
