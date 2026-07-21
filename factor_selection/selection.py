"""因子正交化、相关性诊断和稳定筛选。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_GROUPS: dict[str, str] = {
    "momentum_5": "momentum",
    "momentum_20": "momentum",
    "momentum_60": "momentum",
    "reversal_5": "reversal",
    "ma_gap_5": "moving_average",
    "ma_gap_20": "moving_average",
    "ma_gap_60": "moving_average",
    "rsi_14": "technical_structure",
    "price_position_20": "technical_structure",
    "volatility_20": "volatility",
    "volatility_60": "volatility",
    "volume_ratio_20": "volume_price",
    "price_volume_divergence_20": "volume_price",
    "volume_price_corr_20": "volume_price",
    "turnover_mean_20": "liquidity",
    "turnover_stability_20": "liquidity",
    "illiquidity_20": "liquidity",
    "max_return_20": "tail_return",
    "min_return_20": "tail_return",
    "market_beta_60": "market_exposure",
    "earnings_yield": "valuation",
    "book_to_price": "valuation",
    "sales_to_price": "valuation",
    "cashflow_yield": "valuation",
    "dividend_yield": "valuation",
    "roe": "profitability",
    "roa": "profitability",
    "gross_margin": "profitability",
    "asset_turnover": "profitability",
    "accrual_ratio": "earnings_quality",
    "cash_conversion_ratio": "earnings_quality",
    "current_ratio": "balance_sheet",
    "leverage": "balance_sheet",
    "profit_growth_60": "growth",
    "revenue_growth_60": "growth",
    "northbound_5": "fund_flow",
    "northbound_20": "fund_flow",
    "sentiment_5": "sentiment",
    "sentiment_20": "sentiment",
    "pmi_change_20": "macro",
    "log_size": "size",
}


REDUNDANCY_CLUSTERS: dict[str, str] = {
    "momentum": "price_trend",
    "reversal": "price_trend",
    "moving_average": "price_trend",
}


def _factor_family(factor: str) -> str:
    """Return the explicit economic-logic group for a factor."""

    return FACTOR_GROUPS.get(factor, factor)


def factor_correlation(data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """计算各日期截面相关矩阵的时间均值。"""

    matrices = []
    for _, group in data.groupby("date"):
        if len(group) < 5:
            continue
        ranked = group[factors].rank()
        matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
        valid_factors = [
            factor
            for factor in factors
            if ranked[factor].count() >= 2
            and float(ranked[factor].std(ddof=0)) > 1e-12
        ]
        if valid_factors:
            matrix.loc[valid_factors, valid_factors] = ranked[valid_factors].corr()
        matrices.append(matrix)
    if not matrices:
        return pd.DataFrame(index=factors, columns=factors, dtype=float)
    return (
        pd.concat(matrices)
        .groupby(level=0, sort=False)
        .mean()
        .reindex(index=factors, columns=factors)
    )


def ridge_feature_importance(
    data: pd.DataFrame,
    factors: list[str],
    regularization: float = 1.0,
    max_rows: int = 50_000,
) -> pd.DataFrame:
    """使用岭回归估计因子的联合收益解释能力。

    该实现仅依赖 NumPy。输入因子已完成截面标准化，系数绝对值可作为同量纲的
    特征重要性参考。为控制内存，按固定间隔抽取最多 `max_rows` 行。
    """

    columns = factors + ["forward_return"]
    sample = data[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > max_rows:
        step = max(1, len(sample) // max_rows)
        sample = sample.iloc[::step].head(max_rows)
    if sample.empty or not factors:
        return pd.DataFrame(columns=["factor", "ridge_coefficient", "importance"])

    features = sample[factors].to_numpy(dtype=float)
    target = sample["forward_return"].to_numpy(dtype=float)
    target = target - target.mean()
    identity = np.eye(features.shape[1])
    try:
        coefficients = np.linalg.solve(
            features.T @ features + regularization * identity,
            features.T @ target,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(
            features.T @ features + regularization * identity
        ) @ features.T @ target
    importance = np.abs(coefficients)
    total = float(importance.sum())
    normalized = importance / total if total > 0 else importance
    return pd.DataFrame(
        {
            "factor": factors,
            "ridge_coefficient": coefficients,
            "importance": normalized,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


def orthogonalize_factors(data: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """使用逐日 Gram-Schmidt 残差法降低共线性。"""

    frame = data.copy()

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        result = group[factors].copy()
        basis: list[np.ndarray] = []
        for factor in factors:
            values = result[factor].fillna(0).to_numpy(dtype=float)
            residual = values.copy()
            for vector in basis:
                denominator = float(vector @ vector)
                if denominator > 1e-12:
                    residual -= vector * float(residual @ vector) / denominator
            deviation = residual.std()
            if deviation > 0:
                residual = (residual - residual.mean()) / deviation
            result[factor] = residual
            basis.append(residual)
        return result

    transformed = frame.groupby("date", group_keys=False).apply(
        transform, include_groups=False
    )
    if isinstance(transformed.index, pd.MultiIndex):
        transformed.index = transformed.index.droplevel(0)
    frame[factors] = transformed.reindex(frame.index)
    return frame


def select_factors(
    summary: pd.DataFrame,
    correlation: pd.DataFrame,
    ridge_importance: pd.DataFrame | None = None,
    max_correlation: float = 0.75,
    max_factors: int = 12,
    max_factors_per_family: int = 2,
) -> tuple[list[str], dict[str, float]]:
    """Select factors while preserving the original two-value interface."""

    selected, weights, _ = select_factors_detailed(
        summary=summary,
        correlation=correlation,
        ridge_importance=ridge_importance,
        max_correlation=max_correlation,
        max_factors=max_factors,
        max_factors_per_family=max_factors_per_family,
        minimum_coverage=0.0,
    )
    return selected, weights


def select_factors_detailed(
    summary: pd.DataFrame,
    correlation: pd.DataFrame,
    ridge_importance: pd.DataFrame | None = None,
    validation_summary: pd.DataFrame | None = None,
    regime_summary: pd.DataFrame | None = None,
    residual_summary: pd.DataFrame | None = None,
    validation_residual_summary: pd.DataFrame | None = None,
    max_correlation: float = 0.65,
    max_factors: int = 12,
    max_factors_per_family: int = 2,
    minimum_coverage: float = 0.6,
    score_weights: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, float], pd.DataFrame]:
    """Select factors from training metrics with validation-only stability gates."""

    if summary.empty:
        return [], {}, pd.DataFrame()

    metric_columns = [
        "factor",
        "ic_mean",
        "ic_ir",
        "ic_positive_ratio",
        "t_stat",
        "top_bottom_spread",
        "quantile_monotonicity",
        "coverage_ratio",
        "missing_rate",
    ]
    ranking = summary[[column for column in metric_columns if column in summary]].copy()
    for column in metric_columns[1:]:
        if column not in ranking:
            ranking[column] = 0.0
    ranking["factor_group"] = ranking["factor"].map(_factor_family)
    ranking["ic_score"] = ranking["ic_mean"].abs()
    ranking["stability_score"] = (
        ranking["ic_ir"].abs().fillna(0.0)
        + 0.1 * ranking["t_stat"].abs().fillna(0.0)
        + ranking["quantile_monotonicity"].abs().fillna(0.0)
    )
    if residual_summary is not None and not residual_summary.empty:
        ranking = ranking.merge(residual_summary, on="factor", how="left")
        ranking["residual_score"] = (
            ranking["residual_ic_mean"].abs().fillna(0.0)
            + ranking["residual_ic_ir"].abs().fillna(0.0)
            + ranking["residual_quantile_monotonicity"].abs().fillna(0.0)
        )
    else:
        ranking["residual_score"] = 0.0
    if validation_summary is not None and not validation_summary.empty:
        validation = validation_summary[
            [column for column in metric_columns if column in validation_summary]
        ].copy()
        validation = validation.add_prefix("validation_").rename(
            columns={"validation_factor": "factor"}
        )
        ranking = ranking.merge(validation, on="factor", how="left")
        ranking["validation_direction_consistent"] = (
            np.sign(ranking["ic_mean"])
            == np.sign(ranking["validation_ic_mean"].fillna(0.0))
        )
        ranking["validation_score"] = (
            ranking["validation_ic_ir"].abs().fillna(0.0)
            + ranking["validation_quantile_monotonicity"].abs().fillna(0.0)
        )
    else:
        ranking["validation_direction_consistent"] = True
        ranking["validation_score"] = 0.0
    if (
        validation_residual_summary is not None
        and not validation_residual_summary.empty
    ):
        validation_residual = validation_residual_summary.add_prefix(
            "validation_"
        ).rename(columns={"validation_factor": "factor"})
        ranking = ranking.merge(validation_residual, on="factor", how="left")
        ranking["validation_residual_direction_consistent"] = (
            np.sign(ranking.get("residual_ic_mean", 0.0))
            == np.sign(ranking["validation_residual_ic_mean"].fillna(0.0))
        )
        ranking["validation_residual_score"] = (
            ranking["validation_residual_ic_ir"].abs().fillna(0.0)
            + ranking["validation_residual_quantile_monotonicity"].abs().fillna(0.0)
        )
    else:
        ranking["validation_residual_direction_consistent"] = True
        ranking["validation_residual_score"] = 0.0
    if regime_summary is not None and not regime_summary.empty:
        signs = regime_summary.copy()
        signs["same_direction"] = signs["ic_mean"].apply(np.sign).eq(
            signs["factor"].map(summary.set_index("factor")["ic_mean"]).apply(np.sign)
        )
        regime_stability = signs.groupby("factor")["same_direction"].mean()
        ranking["regime_stability"] = ranking["factor"].map(regime_stability).fillna(0.0)
    else:
        ranking["regime_stability"] = 0.0
    if ridge_importance is not None and not ridge_importance.empty:
        ranking = ranking.merge(
            ridge_importance[["factor", "importance"]], on="factor", how="left"
        )
        ranking["importance"] = ranking["importance"].fillna(0)
    else:
        ranking["importance"] = 0.0
    for column in (
        "ic_score",
        "stability_score",
        "importance",
        "validation_score",
        "residual_score",
        "validation_residual_score",
        "regime_stability",
    ):
        total = float(ranking[column].sum())
        if total > 0:
            ranking[column] /= total
    weights_config = {
        "ic": 0.20,
        "stability": 0.15,
        "ridge": 0.10,
        "validation": 0.20,
        "regime": 0.10,
        "residual": 0.20,
        "validation_residual": 0.05,
    }
    if score_weights:
        weights_config.update(
            {key: float(value) for key, value in score_weights.items()}
        )
    total_score_weight = sum(max(value, 0.0) for value in weights_config.values())
    if total_score_weight <= 0:
        raise ValueError("selection score weights must contain a positive value")
    weights_config = {
        key: max(value, 0.0) / total_score_weight
        for key, value in weights_config.items()
    }
    ranking["combined_score"] = (
        weights_config["ic"] * ranking["ic_score"]
        + weights_config["stability"] * ranking["stability_score"]
        + weights_config["ridge"] * ranking["importance"]
        + weights_config["validation"] * ranking["validation_score"]
        + weights_config["regime"] * ranking["regime_stability"]
        + weights_config["residual"] * ranking["residual_score"]
        + weights_config["validation_residual"]
        * ranking["validation_residual_score"]
    )
    ranking = ranking.sort_values("combined_score", ascending=False)
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    cluster_counts: dict[str, int] = {}
    rejection_reasons: dict[str, str] = {}
    for row in ranking.itertuples(index=False):
        factor = row.factor
        family = _factor_family(factor)
        cluster = REDUNDANCY_CLUSTERS.get(family, family)
        if float(getattr(row, "coverage_ratio", 0.0)) < minimum_coverage:
            rejection_reasons[factor] = "insufficient_training_coverage"
            continue
        if validation_summary is not None and not bool(row.validation_direction_consistent):
            rejection_reasons[factor] = "validation_direction_unstable"
            continue
        if family_counts.get(family, 0) >= max_factors_per_family:
            rejection_reasons[factor] = "factor_group_limit"
            continue
        if cluster == "price_trend" and cluster_counts.get(cluster, 0) >= max_factors_per_family:
            rejection_reasons[factor] = "similar_logic_cluster_limit"
            continue
        correlated = []
        for existing in selected:
            value = correlation.loc[factor, existing]
            if pd.notna(value) and abs(float(value)) >= max_correlation:
                correlated.append(existing)
        if not correlated:
            selected.append(factor)
            family_counts[family] = family_counts.get(family, 0) + 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        else:
            rejection_reasons[factor] = "high_correlation_with:" + ",".join(correlated)
        if len(selected) >= max_factors:
            break
    if not selected:
        ranking["selected"] = False
        ranking["selection_reason"] = ranking["factor"].map(rejection_reasons).fillna(
            "not_selected"
        )
        return [], {}, ranking.reset_index(drop=True)

    selected_ranking = ranking.set_index("factor").reindex(selected)
    ordinary_direction = summary.set_index("factor")["ic_mean"].reindex(selected)
    if residual_summary is not None and not residual_summary.empty:
        residual_direction = residual_summary.set_index("factor")[
            "residual_ic_mean"
        ].reindex(selected)
        direction_values = residual_direction.where(
            residual_direction.abs() > 1e-12,
            ordinary_direction,
        )
        ranking["direction_source"] = np.where(
            ranking["factor"].isin(residual_summary["factor"]),
            "residual_ic_mean",
            "ordinary_ic_mean",
        )
    else:
        direction_values = ordinary_direction
        ranking["direction_source"] = "ordinary_ic_mean"
    directions = np.sign(direction_values.fillna(0.0))
    directions = directions.replace(0.0, 1.0)
    score_total = float(selected_ranking["combined_score"].sum())
    if score_total == 0:
        weights = {factor: 1 / len(selected) for factor in selected}
    else:
        weights = {
            factor: float(
                directions.loc[factor]
                * selected_ranking.loc[factor, "combined_score"]
                / score_total
            )
            for factor in selected
        }
    ranking["selected"] = ranking["factor"].isin(selected)
    ranking["selection_reason"] = np.where(
        ranking["selected"],
        "selected_for_stable_nonredundant_signal",
        ranking["factor"].map(rejection_reasons).fillna("ranking_below_cutoff"),
    )
    ranking["final_weight"] = ranking["factor"].map(weights).fillna(0.0)
    return selected, weights, ranking.reset_index(drop=True)
