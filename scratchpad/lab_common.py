"""共通の単体測定ハーネス（案1/3/4/7/9 で共用・仕様カード準拠）。

これは測定専用のユーティリティであり、トレードルールではない。
- forward_direction / forward_move は「未来」を使うが、それは予測対象（アウトカム）としてで
  あって、予測変数 X の計算には一切使わない（X は必ずバー s で確定した情報のみを使う）。
- 五分位/四分位/カテゴリの境界は必ず全期間で1回だけ切る（測定用途であり、ウォークフォワード
  最適化ではないため、期毎に切り直すと「安定性の目視」という前後分割表の目的自体が壊れる）。
- 月次ブロック・ブートストラップは「暦月」を復元抽出する（trade-count blockではない。
  research/overfit_audit.py の block_resample はトレード本数ブロックであり月ブロックではない
  ため、月ブロックはこのファイルに新規実装した）。
"""
import numpy as np
import pandas as pd


def forward_direction(close: pd.Series, H: int) -> pd.Series:
    """先行方向 = log(close[s+H]/close[s])。アウトカム専用（Xの計算には使わない）。"""
    lc = np.log(close)
    return lc.shift(-H) - lc


def forward_move(close: pd.Series, H: int) -> pd.Series:
    """先行の動く量 = (s, s+H] の15分足log|リターン|の合計。アウトカム専用。"""
    ar = np.log(close).diff().abs()
    return ar.rolling(H).sum().shift(-H)


def quantile_labels(X: pd.Series, q: int) -> pd.Series:
    """全期間の分位点で切った固定ラベル（1..q）。測定専用でトレードルールではない。"""
    lab = pd.qcut(X.rank(method="first"), q, labels=False, duplicates="drop")
    return lab + 1


def summarize_group(dir_: pd.Series, move_: pd.Series, move_all_mean: float) -> dict:
    d = dir_.dropna()
    m = move_.dropna()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, mean=d.mean() if n else np.nan, se=se,
                median=d.median() if n else np.nan,
                std=d.std(ddof=1) if n > 1 else np.nan,
                move_mean=m.mean() if len(m) else np.nan,
                move_ratio=(m.mean() / move_all_mean) if (move_all_mean and len(m)) else np.nan)


_COLS = ["n", "mean", "se", "median", "std", "move_mean", "move_ratio"]


def build_table(labels: pd.Series, dir_: pd.Series, move_: pd.Series) -> pd.DataFrame:
    move_all_mean = move_.dropna().mean()
    rows = []
    for lab in sorted(pd.unique(labels.dropna())):
        idx = labels == lab
        row = summarize_group(dir_[idx], move_[idx], move_all_mean)
        row["group"] = lab
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_COLS, index=pd.Index([], name="group"))
    return pd.DataFrame(rows).set_index("group")


def month_block_bootstrap(labels: pd.Series, dir_: pd.Series, move_: pd.Series,
                           top_label, bot_label, n_boot: int = 1000, seed: int = 0) -> dict:
    """暦月を復元抽出（block bootstrap, n_boot回）して、top_label群 - bot_label群の
    方向差、および move比 (top/bot) の分布を返す。分位境界(top/bot の定義)は全期間固定。"""
    df = pd.DataFrame({"label": labels, "dir": dir_, "move": move_}).dropna(subset=["label"])
    idx = df.index.tz_convert(None) if getattr(df.index, "tz", None) is not None else df.index
    df["month"] = idx.to_period("M")
    months = df["month"].unique()
    by_month = {m: g for m, g in df.groupby("month")}
    rng = np.random.default_rng(seed)
    diffs = np.full(n_boot, np.nan)
    ratios = np.full(n_boot, np.nan)
    for b in range(n_boot):
        draw = rng.choice(months, size=len(months), replace=True)
        rs = pd.concat([by_month[m] for m in draw], ignore_index=False)
        top = rs.loc[rs.label == top_label, "dir"]
        bot = rs.loc[rs.label == bot_label, "dir"]
        topm = rs.loc[rs.label == top_label, "move"]
        botm = rs.loc[rs.label == bot_label, "move"]
        if len(top) == 0 or len(bot) == 0:
            continue
        diffs[b] = top.mean() - bot.mean()
        bm = botm.mean()
        ratios[b] = (topm.mean() / bm) if (bm and not np.isnan(bm)) else np.nan
    return dict(
        n_boot=n_boot, n_months=len(months),
        diff_median=np.nanmedian(diffs), diff_lo=np.nanpercentile(diffs, 2.5),
        diff_hi=np.nanpercentile(diffs, 97.5),
        ratio_median=np.nanmedian(ratios), ratio_lo=np.nanpercentile(ratios, 2.5),
        ratio_hi=np.nanpercentile(ratios, 97.5),
    )


def split_tables(labels: pd.Series, dir_: pd.Series, move_: pd.Series, split_date: str):
    """全期間で固定したラベルのまま、split_date前後で summarize（安定性の目視用）。"""
    pre = labels.index < split_date
    post = ~pre
    t_pre = build_table(labels[pre], dir_[pre], move_[pre])
    t_post = build_table(labels[post], dir_[post], move_[post])
    return t_pre, t_post


def fmt_table(df: pd.DataFrame, h_label: str) -> str:
    lines = [f"  {h_label:<8}{'n':>8}{'mean±se':>18}{'median':>10}{'std':>10}"
             f"{'move_mean':>12}{'move/全体':>10}"]
    for g, r in df.iterrows():
        lines.append(f"  {str(g):<8}{r['n']:>8.0f}"
                      f"{r['mean']:>+10.5f}±{r['se']:<6.5f}"
                      f"{r['median']:>+10.5f}{r['std']:>10.5f}"
                      f"{r['move_mean']:>12.5f}{r['move_ratio']:>10.3f}")
    return "\n".join(lines)
