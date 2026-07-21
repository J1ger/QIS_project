from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlation_sensitivity import (
    BACKTEST_INTEGRATION_AVAILABLE,
    factor_weight_l1_distance,
    jaccard_similarity,
    run_correlation_threshold_sensitivity,
)
from walk_forward import generate_walk_forward_windows, validate_walk_forward_windows
from weights import build_rolling_composite_score


def test_sensitivity_helpers_work_without_backtest() -> None:
    assert BACKTEST_INTEGRATION_AVAILABLE is False
    assert np.isclose(jaccard_similarity({"a", "b"}, {"b", "c"}), 1 / 3)
    assert np.isclose(
        factor_weight_l1_distance(
            {"a": 0.6, "b": 0.4},
            {"a": 0.5, "c": 0.5},
        ),
        1.0,
    )


def test_full_sensitivity_reports_missing_backtest_integration() -> None:
    empty = pd.DataFrame()
    with pytest.raises(RuntimeError, match="尚未接入"):
        run_correlation_threshold_sensitivity(
            train_data=empty,
            validation_data=empty,
            factor_names=[],
            train_summary=empty,
            validation_summary=empty,
            train_correlation=empty,
            train_ridge=empty,
            train_regime=empty,
            train_residual_summary=empty,
            validation_residual_summary=empty,
            thresholds=[0.65],
            selection_options={},
            backtest_config=None,
            validation_benchmark=empty,
        )


def test_walk_forward_windows_are_ordered_and_non_overlapping() -> None:
    windows = generate_walk_forward_windows(
        "2015-01-01",
        "2022-12-31",
        train_years=3,
        validation_months=6,
        test_months=6,
        step_months=6,
    )
    validate_walk_forward_windows(windows)
    assert not windows.empty
    assert (windows["train_end"] < windows["validation_start"]).all()
    assert (windows["validation_end"] < windows["test_start"]).all()


def test_rolling_composite_score_uses_latest_effective_weights() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["000001", "000002"],
            "factor_a": [1.0, 2.0],
            "factor_b": [2.0, 1.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "factor": ["factor_a", "factor_b"],
            "weight": [0.6, -0.4],
        }
    )

    result = build_rolling_composite_score(data, schedule)

    assert np.allclose(result["composite_score"], [-0.2, 0.8])
    assert result["factor_weight_date"].eq(pd.Timestamp("2024-01-01")).all()
