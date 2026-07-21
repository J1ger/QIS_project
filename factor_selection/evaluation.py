"""IC/IR、统计显著性和分层收益检验。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_forward_returns(
    data: pd.DataFrame,
    period: int = 1,
    periods: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Add forward returns and optionally prevent labels crossing split boundaries."""

    frame = data.sort_values(["symbol", "date"]).copy()
    grouped = frame.groupby("symbol")
    frame["forward_return"] = (
        grouped["close"].shift(-period) / frame["close"] - 1
    )
    frame["forward_return_date"] = grouped["date"].shift(-period)
    if periods:
        allowed = pd.Series(False, index=frame.index)
        for start_text, end_text in periods.values():
            start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
            allowed |= (
                frame["date"].between(start, end)
                & frame["forward_return_date"].between(start, end)
            )
        frame.loc[~allowed, "forward_return"] = np.nan
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """使用秩变换计算 Spearman 相关系数。"""

    valid = left.notna() & right.notna()
    if valid.sum() < 5:
        return float("nan")
    left_rank = left[valid].rank()
    right_rank = right[valid].rank()
    if left_rank.std(ddof=0) <= 1e-12 or right_rank.std(ddof=0) <= 1e-12:
        return float("nan")
    return float(left_rank.corr(right_rank))


def calculate_ic_series(
    data: pd.DataFrame,
    factor: str,
    return_column: str = "forward_return",
) -> pd.Series:
    """计算因子日度 Rank IC 序列。"""

    return data.groupby("date").apply(
        lambda group: _spearman_correlation(group[factor], group[return_column]),
        include_groups=False,
    ).rename(factor)


def calculate_quantile_returns(
    data: pd.DataFrame,
    factor: str,
    quantiles: int = 5,
    return_column: str = "forward_return",
) -> pd.DataFrame:
    """计算因子分位数组合的日均远期收益。"""

    subset = (
        data[["date", factor, return_column]]
        .dropna()
        .reset_index(drop=True)
    )

    def assign_quantile(values: pd.Series) -> pd.Series:
        ranks = values.rank(method="first")
        try:
            return pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.Series(np.nan, index=values.index)

    subset["quantile"] = subset.groupby("date")[factor].transform(assign_quantile)
    return (
        subset.groupby(["date", "quantile"], as_index=False)[return_column]
        .mean()
        .rename(columns={return_column: "mean_return"})
    )


def evaluate_factors(
    data: pd.DataFrame,
    factors: list[str],
    quantiles: int = 5,
    min_observations: int = 30,
    return_column: str = "forward_return",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """批量输出因子摘要、IC 时间序列和分层收益。"""

    summaries: list[dict[str, float | str | int]] = []
    ic_frames: list[pd.Series] = []
    quantile_frames: list[pd.DataFrame] = []
    for factor in factors:
        ic = calculate_ic_series(data, factor, return_column=return_column).dropna()
        if len(ic) < min_observations:
            continue
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1))
        ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")
        t_stat = ic_mean / (ic_std / math.sqrt(len(ic))) if ic_std > 0 else float("nan")
        quantile_result = calculate_quantile_returns(
            data,
            factor,
            quantiles,
            return_column=return_column,
        )
        top = quantile_result[quantile_result["quantile"] == quantiles]["mean_return"].mean()
        bottom = quantile_result[quantile_result["quantile"] == 1]["mean_return"].mean()
        quantile_means = quantile_result.groupby("quantile")["mean_return"].mean()
        monotonicity = (
            float(pd.Series(quantile_means.index).corr(pd.Series(quantile_means.values), method="spearman"))
            if len(quantile_means) >= 3
            else float("nan")
        )
        available = data[[factor, return_column]].dropna()
        eligible_rows = int(data[return_column].notna().sum())
        coverage = len(available) / eligible_rows if eligible_rows > 0 else 0.0
        summaries.append(
            {
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "ic_positive_ratio": float((ic > 0).mean()),
                "t_stat": t_stat,
                "observations": int(len(ic)),
                "top_bottom_spread": float(top - bottom),
                "quantile_monotonicity": monotonicity,
                "coverage_ratio": float(coverage),
                "missing_rate": float(1.0 - coverage),
            }
        )
        ic_frames.append(ic)
        quantile_result["factor"] = factor
        quantile_frames.append(quantile_result)

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values("ic_mean", key=lambda s: s.abs(), ascending=False)
    ic_data = pd.concat(ic_frames, axis=1, sort=False) if ic_frames else pd.DataFrame()
    quantile_data = (
        pd.concat(quantile_frames, ignore_index=True) if quantile_frames else pd.DataFrame()
    )
    return summary.reset_index(drop=True), ic_data, quantile_data


def add_residual_forward_returns(
    data: pd.DataFrame,
    market_beta_column: str = "portfolio_market_beta_60",
    industry_column: str = "industry",
    size_column: str = "portfolio_log_size",
    minimum_cross_section: int = 20,
) -> pd.DataFrame:
    """Residualize forward returns on point-in-time market, industry, and size exposures.

    The regression is run independently for each signal date.  Exposures are
    observed on that date and the resulting label is used only by the caller's
    train/validation evaluation, so no information is carried across dates.
    """

    frame = data.copy()
    frame["residual_forward_return"] = np.nan
    required = {
        "forward_return",
        market_beta_column,
        industry_column,
        size_column,
    }
    if not required.issubset(frame.columns):
        return frame

    for _, group in frame.groupby("date", sort=False):
        valid = group[
            [
                "forward_return",
                market_beta_column,
                industry_column,
                size_column,
            ]
        ].replace([np.inf, -np.inf], np.nan)
        mask = valid.notna().all(axis=1)
        if int(mask.sum()) < minimum_cross_section:
            continue
        sample = valid.loc[mask]
        continuous = sample[[market_beta_column, size_column]].astype(float)
        continuous = continuous - continuous.mean()
        scales = continuous.std(ddof=0).replace(0.0, 1.0)
        continuous = continuous / scales
        industries = pd.get_dummies(
            sample[industry_column].astype(str),
            prefix="industry",
            drop_first=True,
            dtype=float,
        )
        design = pd.concat([continuous, industries], axis=1)
        design.insert(0, "intercept", 1.0)
        matrix = design.to_numpy(dtype=float)
        target = sample["forward_return"].to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ coefficients
        frame.loc[sample.index, "residual_forward_return"] = residual
    return frame


def evaluate_residual_factors(
    data: pd.DataFrame,
    factors: list[str],
    quantiles: int = 5,
    min_observations: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate factors against market/industry/size residual forward returns."""

    summary, ic_data, quantile_data = evaluate_factors(
        data,
        factors,
        quantiles=quantiles,
        min_observations=min_observations,
        return_column="residual_forward_return",
    )
    if not summary.empty:
        summary = summary.rename(
            columns={
                column: f"residual_{column}"
                for column in summary.columns
                if column != "factor"
            }
        )
    return summary, ic_data, quantile_data


def evaluate_factor_by_group(
    data: pd.DataFrame,
    factors: list[str],
    group_column: str,
    minimum_rows: int = 20,
) -> pd.DataFrame:
    """按行业或市场环境计算分组 Rank IC。"""

    records: list[dict[str, float | str | int]] = []
    for group_name, group_data in data.groupby(group_column):
        for factor in factors:
            values = group_data.groupby("date").apply(
                lambda cross_section: _spearman_correlation(
                    cross_section[factor], cross_section["forward_return"]
                ),
                include_groups=False,
            ).dropna()
            if len(values) < minimum_rows:
                continue
            records.append(
                {
                    "group_type": group_column,
                    "group": str(group_name),
                    "factor": factor,
                    "ic_mean": float(values.mean()),
                    "ic_ir": float(values.mean() / values.std(ddof=1))
                    if values.std(ddof=1) > 0
                    else float("nan"),
                    "observations": int(len(values)),
                }
            )
    return pd.DataFrame(records)


def add_market_regimes(
    data: pd.DataFrame,
    trend_window: int = 60,
    volatility_window: int = 60,
    minimum_periods: int = 30,
    volatility_threshold_window: int = 252,
) -> pd.DataFrame:
    """Classify dates using a benchmark return when available and trailing data."""

    frame = data.copy()
    return_column = "benchmark_return" if "benchmark_return" in frame else "market_return"
    market = (
        frame[["date", return_column]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")[return_column]
    )
    trend = (
        (1.0 + market)
        .rolling(trend_window, min_periods=minimum_periods)
        .apply(np.prod, raw=True)
        - 1.0
    )
    volatility = market.rolling(
        volatility_window, min_periods=minimum_periods
    ).std(ddof=1)
    volatility_threshold = volatility.shift(1).rolling(
        volatility_threshold_window, min_periods=minimum_periods
    ).median()
    volatility_threshold = volatility_threshold.fillna(
        volatility.shift(1).expanding(min_periods=minimum_periods).median()
    )
    regime = pd.Series(index=market.index, dtype="object")
    regime[(trend >= 0) & (volatility <= volatility_threshold)] = "上涨-低波"
    regime[(trend >= 0) & (volatility > volatility_threshold)] = "上涨-高波"
    regime[(trend < 0) & (volatility <= volatility_threshold)] = "下跌-低波"
    regime[(trend < 0) & (volatility > volatility_threshold)] = "下跌-高波"
    frame["market_regime"] = frame["date"].map(regime).fillna("预热期")
    return frame
