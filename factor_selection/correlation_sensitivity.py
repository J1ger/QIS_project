"""Correlation-threshold sensitivity analysis without test-period tuning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from selection import (
    FACTOR_GROUPS,
    orthogonalize_factors,
    select_factors_detailed,
)


# 第四阶段的策略回测与绩效模块尚未接入。保留这些名称是为了维持
# 敏感性分析函数的接口；完整回测接入前，入口函数会给出明确错误。
BacktestConfig = Any
MultiFactorBacktester: Any = None
build_composite_score: Any = None
calculate_performance: Any = None
calculate_benchmark_attribution: Any = None
BACKTEST_INTEGRATION_AVAILABLE = False
BACKTEST_UNAVAILABLE_MESSAGE = (
    "完整相关性阈值敏感性测试需要第四阶段的策略构建、回测和绩效分析模块；"
    "当前 factor_selection 项目尚未接入，请在后续集成回测系统后运行。"
)


def _require_backtest_integration() -> None:
    if not BACKTEST_INTEGRATION_AVAILABLE:
        raise RuntimeError(BACKTEST_UNAVAILABLE_MESSAGE)


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "validation_information_ratio": 0.25,
    "validation_annual_alpha": 0.20,
    "validation_sharpe": 0.15,
    "validation_max_drawdown": 0.10,
    "selected_factor_average_abs_correlation": 0.10,
    "mean_jaccard_similarity": 0.10,
    "validation_turnover": 0.05,
    "validation_trading_cost": 0.05,
}


def _factor_group(factor: str) -> str:
    return FACTOR_GROUPS.get(factor, factor)


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def _json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _selection_input_fingerprint(
    summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    factors: list[str],
    selection_options: dict[str, Any],
) -> str:
    """Hash all threshold-invariant experiment inputs for audit purposes."""

    payload = {
        "factors": list(factors),
        "summary": pd.util.hash_pandas_object(
            summary.sort_index(axis=1), index=True
        ).astype(str).tolist(),
        "validation_summary": pd.util.hash_pandas_object(
            validation_summary.sort_index(axis=1), index=True
        ).astype(str).tolist(),
        "selection_options": selection_options,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return set Jaccard similarity, treating two empty sets as identical."""

    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def factor_weight_l1_distance(
    left: dict[str, float], right: dict[str, float]
) -> float:
    """Return L1 distance over the union of factor-weight keys."""

    factors = set(left) | set(right)
    return float(sum(abs(float(left.get(factor, 0.0)) - float(right.get(factor, 0.0))) for factor in factors))


def _selected_pair_correlation(
    correlation: pd.DataFrame, selected: list[str]
) -> tuple[float, float, int]:
    if len(selected) < 2:
        return 0.0, 0.0, 0
    values: list[float] = []
    nan_pairs = 0
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            value = correlation.loc[left, right]
            if pd.isna(value):
                nan_pairs += 1
            else:
                values.append(abs(float(value)))
    if not values:
        return float("nan"), float("nan"), nan_pairs
    return float(np.mean(values)), float(np.max(values)), nan_pairs


def _rejection_count(selection: pd.DataFrame, patterns: tuple[str, ...]) -> int:
    if selection.empty or "selection_reason" not in selection:
        return 0
    reasons = selection["selection_reason"].fillna("").astype(str)
    return int(reasons.apply(lambda value: any(pattern in value for pattern in patterns)).sum())


def _validation_metrics(
    validation_data: pd.DataFrame,
    selected: list[str],
    weights: dict[str, float],
    backtest_config: BacktestConfig,
    benchmark: pd.DataFrame,
    annual_risk_free_rate: float,
) -> tuple[dict[str, float], Any]:
    _require_backtest_integration()
    if not selected:
        return {}, None
    scored = build_composite_score(
        orthogonalize_factors(validation_data, selected), weights
    )
    result = MultiFactorBacktester(backtest_config).run(scored)
    metrics = calculate_performance(
        result.daily,
        backtest_config.annualization,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    attribution = calculate_benchmark_attribution(
        result.daily,
        benchmark,
        annualization=backtest_config.annualization,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    if not attribution.empty:
        metrics.update(attribution.iloc[0].to_dict())
    return metrics, result


def build_factor_pool_comparisons(
    experiments: dict[float, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build pairwise stability rows, Jaccard/weight matrices, and group counts."""

    thresholds = sorted(experiments)
    pairwise_rows: list[dict[str, Any]] = []
    jaccard = pd.DataFrame(index=thresholds, columns=thresholds, dtype=float)
    weight_distance = pd.DataFrame(index=thresholds, columns=thresholds, dtype=float)
    groups = sorted(
        {
            _factor_group(factor)
            for experiment in experiments.values()
            for factor in experiment["selected"]
        }
    )
    group_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        selected = experiments[threshold]["selected"]
        group_rows.append(
            {
                "threshold": threshold,
                **{
                    group: sum(_factor_group(factor) == group for factor in selected)
                    for group in groups
                },
            }
        )
    group_counts = pd.DataFrame(group_rows)
    for left in thresholds:
        left_selected = set(experiments[left]["selected"])
        left_weights = experiments[left]["weights"]
        left_directions = {
            factor: int(np.sign(weight)) for factor, weight in left_weights.items()
        }
        left_groups = {
            group: sum(_factor_group(factor) == group for factor in left_selected)
            for group in groups
        }
        for right in thresholds:
            right_selected = set(experiments[right]["selected"])
            right_weights = experiments[right]["weights"]
            right_directions = {
                factor: int(np.sign(weight)) for factor, weight in right_weights.items()
            }
            intersection = left_selected & right_selected
            union = left_selected | right_selected
            similarity = jaccard_similarity(left_selected, right_selected)
            distance = factor_weight_l1_distance(left_weights, right_weights)
            jaccard.loc[left, right] = similarity
            weight_distance.loc[left, right] = distance
            group_changes = {
                group: sum(_factor_group(factor) == group for factor in right_selected)
                - left_groups[group]
                for group in groups
            }
            pairwise_rows.append(
                {
                    "threshold_left": left,
                    "threshold_right": right,
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "jaccard_similarity": similarity,
                    "added_factor_count": len(right_selected - left_selected),
                    "removed_factor_count": len(left_selected - right_selected),
                    "direction_change_count": sum(
                        left_directions.get(factor) != right_directions.get(factor)
                        for factor in intersection
                    ),
                    "weight_l1_distance": distance,
                    "factor_group_count_changes": json.dumps(
                        group_changes, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    jaccard.index.name = "threshold"
    weight_distance.index.name = "threshold"
    jaccard.columns = [_threshold_key(value) for value in thresholds]
    weight_distance.columns = [_threshold_key(value) for value in thresholds]
    return pd.DataFrame(pairwise_rows), jaccard, weight_distance, group_counts


def _robust_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ranked = values.rank(method="average", pct=True, ascending=higher_is_better)
    return ranked.fillna(0.0)


def select_correlation_threshold(
    sensitivity: pd.DataFrame,
    default_threshold: float = 0.65,
    selection_metric: str = "composite_score",
    score_weights: dict[str, float] | None = None,
    close_score_tolerance: float = 0.03,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    """Choose a threshold from train/validation diagnostics only."""

    if sensitivity.empty:
        raise ValueError("correlation sensitivity produced no threshold experiment")
    table = sensitivity.copy()
    valid = table["status"].eq("ok") & table["selected_factor_count"].gt(0)
    if not valid.any():
        reasons = table[["threshold", "status", "failure_reason"]].to_dict("records")
        raise ValueError(f"all correlation thresholds are infeasible: {reasons}")
    weights = DEFAULT_SCORE_WEIGHTS.copy()
    if score_weights:
        weights.update({key: float(value) for key, value in score_weights.items()})
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("correlation threshold score weights must be non-negative and non-zero")
    total = float(sum(weights.values()))
    weights = {key: value / total for key, value in weights.items()}
    directions = {
        "validation_information_ratio": True,
        "validation_annual_alpha": True,
        "validation_sharpe": True,
        "validation_max_drawdown": False,
        "selected_factor_average_abs_correlation": False,
        "mean_jaccard_similarity": True,
        "validation_turnover": False,
        "validation_trading_cost": False,
    }
    table["composite_score"] = 0.0
    for metric, weight in weights.items():
        if metric not in table:
            raise ValueError(f"unknown correlation threshold score metric: {metric}")
        source = table[metric].abs() if metric == "validation_max_drawdown" else table[metric]
        rank_column = f"rank_{metric}"
        table[rank_column] = _robust_rank(source.where(valid), directions[metric])
        table["composite_score"] += weight * table[rank_column]
    table.loc[~valid, "composite_score"] = np.nan
    if selection_metric not in table:
        raise ValueError(f"unknown correlation threshold selection metric: {selection_metric}")
    selection_values = pd.to_numeric(table[selection_metric], errors="coerce")
    if selection_metric == "validation_max_drawdown":
        selection_values = selection_values.abs()
    selection_values = selection_values.where(valid)
    if selection_values.notna().sum() == 0:
        raise ValueError(f"selection metric {selection_metric} has no valid value")
    higher_is_better = selection_metric not in {
        "validation_max_drawdown",
        "selected_factor_average_abs_correlation",
        "validation_turnover",
        "validation_trading_cost",
    }
    if higher_is_better:
        best_value = float(selection_values.max())
        candidates = table[valid & (selection_values >= best_value - close_score_tolerance)]
    else:
        best_value = float(selection_values.min())
        candidates = table[valid & (selection_values <= best_value + close_score_tolerance)]
    candidates = candidates.assign(
        _default_distance=(candidates["threshold"] - default_threshold).abs()
    ).sort_values(
        [
            "mean_jaccard_similarity",
            "selected_factor_average_abs_correlation",
            "validation_turnover",
            "validation_trading_cost",
            "_default_distance",
        ],
        ascending=[False, True, True, True, True],
        na_position="last",
    )
    selected_threshold = float(candidates.iloc[0]["threshold"])
    table["is_selected_threshold"] = table["threshold"].eq(selected_threshold)
    evidence_columns = [
        "threshold",
        selection_metric,
        "composite_score",
        "mean_jaccard_similarity",
        "selected_factor_average_abs_correlation",
        "validation_turnover",
        "validation_trading_cost",
    ]
    evidence = {
        "selection_metric": selection_metric,
        "selection_metric_best_value": best_value,
        "close_score_tolerance": close_score_tolerance,
        "tie_break_priority": [
            "factor_pool_stability",
            "lower_remaining_correlation",
            "lower_turnover_and_cost",
            "distance_to_default_threshold",
        ],
        "selected_row": table.loc[
            table["threshold"].eq(selected_threshold), evidence_columns
        ].iloc[0].to_dict(),
        "normalized_score_weights": weights,
    }
    return selected_threshold, table, evidence


def run_correlation_threshold_sensitivity(
    *,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    factor_names: list[str],
    train_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    train_correlation: pd.DataFrame,
    train_ridge: pd.DataFrame,
    train_regime: pd.DataFrame,
    train_residual_summary: pd.DataFrame,
    validation_residual_summary: pd.DataFrame,
    thresholds: list[float],
    selection_options: dict[str, Any],
    backtest_config: BacktestConfig,
    validation_benchmark: pd.DataFrame,
    annual_risk_free_rate: float = 0.0,
    default_threshold: float = 0.65,
    selection_metric: str = "composite_score",
    threshold_score_weights: dict[str, float] | None = None,
    close_score_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Run threshold experiments using only train and validation information."""

    _require_backtest_integration()
    grid = sorted({float(value) for value in thresholds})
    if not grid or any(not 0.0 < value < 1.0 for value in grid):
        raise ValueError("correlation threshold grid must contain values between 0 and 1")
    fingerprint = _selection_input_fingerprint(
        train_summary, validation_summary, factor_names, selection_options
    )
    experiments: dict[float, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    for threshold in grid:
        selected, weights, detailed = select_factors_detailed(
            train_summary,
            train_correlation,
            ridge_importance=train_ridge,
            validation_summary=validation_summary,
            regime_summary=train_regime,
            residual_summary=train_residual_summary,
            validation_residual_summary=validation_residual_summary,
            max_correlation=threshold,
            **selection_options,
        )
        metrics: dict[str, float] = {}
        result = None
        status = "ok"
        failure_reason = ""
        if selected:
            try:
                metrics, result = _validation_metrics(
                    validation_data,
                    selected,
                    weights,
                    backtest_config,
                    validation_benchmark,
                    annual_risk_free_rate,
                )
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                status = "validation_backtest_failed"
                failure_reason = str(exc)
        else:
            status = "empty_factor_pool"
            failure_reason = "all candidates were rejected by configured selection constraints"
        average_correlation, maximum_correlation, nan_pair_count = (
            _selected_pair_correlation(train_correlation, selected)
        )
        selected_rows = detailed[detailed.get("selected", False)].copy()
        experiments[threshold] = {
            "selected": selected,
            "weights": weights,
            "detailed": detailed,
            "validation_result": result,
        }
        row = {
            "threshold": threshold,
            "status": status,
            "failure_reason": failure_reason,
            "experiment_input_fingerprint": fingerprint,
            "selected_factor_count": len(selected),
            "selected_factors": _json_list(selected),
            "selected_factor_groups": _json_list(
                sorted({_factor_group(factor) for factor in selected})
            ),
            "rejected_by_correlation_count": _rejection_count(
                detailed, ("high_correlation_with:",)
            ),
            "rejected_by_family_limit_count": _rejection_count(
                detailed, ("factor_group_limit", "similar_logic_cluster_limit")
            ),
            "rejected_by_coverage_count": _rejection_count(
                detailed, ("insufficient_training_coverage",)
            ),
            "selected_factor_average_abs_correlation": average_correlation,
            "selected_factor_max_abs_correlation": maximum_correlation,
            "selected_factor_nan_correlation_pair_count": nan_pair_count,
            "selected_factor_average_ic": _safe_mean(selected_rows.get("ic_mean", pd.Series(dtype=float))),
            "selected_factor_average_abs_ic": _safe_mean(selected_rows.get("ic_mean", pd.Series(dtype=float)).abs()),
            "selected_factor_average_icir": _safe_mean(selected_rows.get("ic_ir", pd.Series(dtype=float))),
            "selected_factor_average_residual_icir": _safe_mean(selected_rows.get("residual_ic_ir", pd.Series(dtype=float))),
            "selected_factor_average_validation_icir": _safe_mean(selected_rows.get("validation_ic_ir", pd.Series(dtype=float))),
            "selected_factor_average_monotonicity": _safe_mean(selected_rows.get("quantile_monotonicity", pd.Series(dtype=float))),
            "selected_factor_average_coverage": _safe_mean(selected_rows.get("coverage_ratio", pd.Series(dtype=float))),
            "ridge_importance_sum": float(pd.to_numeric(selected_rows.get("importance", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
            "validation_annual_return": float(metrics.get("annual_return", np.nan)),
            "validation_annual_alpha": float(metrics.get("annual_alpha", np.nan)),
            "validation_sharpe": float(metrics.get("sharpe_ratio", np.nan)),
            "validation_information_ratio": float(metrics.get("information_ratio", np.nan)),
            "validation_max_drawdown": float(metrics.get("max_drawdown", np.nan)),
            "validation_beta": float(metrics.get("market_beta", np.nan)),
            "validation_r_squared": float(metrics.get("r_squared", np.nan)),
            "validation_turnover": float(metrics.get("annual_turnover", np.nan)),
            "validation_trading_cost": float(metrics.get("total_trading_cost", np.nan)),
            "validation_failed_order_ratio": float(metrics.get("failed_order_ratio", np.nan)),
            "validation_constraint_degraded_count": int(
                result.constraint_log["success"].eq(False).sum()
                if result is not None and not result.constraint_log.empty
                else 0
            ),
        }
        summary_rows.append(row)
        detailed_index = detailed.set_index("factor") if not detailed.empty else pd.DataFrame()
        for factor in factor_names:
            detail = detailed_index.loc[factor] if factor in detailed_index.index else None
            membership_rows.append(
                {
                    "threshold": threshold,
                    "factor": factor,
                    "factor_group": _factor_group(factor),
                    "selected": factor in selected,
                    "weight": float(weights.get(factor, 0.0)),
                    "direction": int(np.sign(weights.get(factor, 0.0))),
                    "direction_source": (
                        str(detail.get("direction_source", "")) if detail is not None else ""
                    ),
                    "selection_reason": (
                        str(detail.get("selection_reason", "not_evaluated"))
                        if detail is not None
                        else "not_evaluated"
                    ),
                    "experiment_input_fingerprint": fingerprint,
                }
            )
    pairwise, jaccard, weight_distance, group_counts = build_factor_pool_comparisons(
        experiments
    )
    sensitivity = pd.DataFrame(summary_rows)
    mean_stability = pairwise[
        pairwise["threshold_left"].ne(pairwise["threshold_right"])
    ].groupby("threshold_left")["jaccard_similarity"].mean()
    sensitivity["mean_jaccard_similarity"] = sensitivity["threshold"].map(
        mean_stability
    ).fillna(1.0)
    sensitivity["previous_threshold"] = sensitivity["threshold"].shift(1)
    adjacent = pairwise.merge(
        sensitivity[["threshold", "previous_threshold"]],
        left_on=["threshold_right", "threshold_left"],
        right_on=["threshold", "previous_threshold"],
        how="inner",
    ).set_index("threshold_right")
    for source, target in (
        ("intersection_count", "adjacent_intersection_count"),
        ("union_count", "adjacent_union_count"),
        ("jaccard_similarity", "adjacent_jaccard_similarity"),
        ("added_factor_count", "adjacent_added_factor_count"),
        ("removed_factor_count", "adjacent_removed_factor_count"),
        ("direction_change_count", "adjacent_direction_change_count"),
        ("weight_l1_distance", "adjacent_weight_l1_distance"),
        ("factor_group_count_changes", "adjacent_factor_group_count_changes"),
    ):
        sensitivity[target] = sensitivity["threshold"].map(adjacent[source])
    selected_threshold, sensitivity, evidence = select_correlation_threshold(
        sensitivity,
        default_threshold=default_threshold,
        selection_metric=selection_metric,
        score_weights=threshold_score_weights,
        close_score_tolerance=close_score_tolerance,
    )
    membership = pd.DataFrame(membership_rows)
    frequency = membership.groupby("factor")["selected"].mean()
    membership["selection_frequency"] = membership["factor"].map(frequency)
    return {
        "enabled": True,
        "selected_threshold": selected_threshold,
        "selection_evidence": evidence,
        "sensitivity": sensitivity,
        "membership": membership,
        "pairwise": pairwise,
        "jaccard_matrix": jaccard,
        "weight_distance": weight_distance,
        "group_counts": group_counts,
        "experiments": experiments,
        "test_thresholds_evaluated": [],
    }
