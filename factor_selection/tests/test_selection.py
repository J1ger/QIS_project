from __future__ import annotations

import numpy as np
import pandas as pd

from selection import factor_correlation, select_factors_detailed


def _summary() -> pd.DataFrame:
    factors = ["momentum_5", "momentum_20", "momentum_60", "roe", "roa"]
    return pd.DataFrame(
        {
            "factor": factors,
            "ic_mean": [0.05, 0.04, 0.03, 0.02, 0.01],
            "ic_ir": [1.0, 0.9, 0.8, 0.7, 0.6],
            "ic_positive_ratio": [0.7] * 5,
            "t_stat": [3.0, 2.8, 2.5, 2.0, 1.8],
            "top_bottom_spread": [0.01] * 5,
            "quantile_monotonicity": [0.9] * 5,
            "coverage_ratio": [1.0, 1.0, 1.0, 1.0, 0.2],
            "missing_rate": [0.0, 0.0, 0.0, 0.0, 0.8],
        }
    )


def test_correlation_matrix_matches_manual_example() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 5 + ["2024-01-03"] * 5),
            "factor_a": list(range(5)) + list(range(5)),
            "factor_b": list(range(5)) + list(range(5)),
            "factor_c": list(reversed(range(5))) * 2,
        }
    )

    correlation = factor_correlation(frame, ["factor_a", "factor_b", "factor_c"])

    assert np.isclose(correlation.loc["factor_a", "factor_b"], 1.0)
    assert np.isclose(correlation.loc["factor_a", "factor_c"], -1.0)


def test_selection_enforces_correlation_family_and_coverage_limits() -> None:
    factors = _summary()["factor"].tolist()
    correlation = pd.DataFrame(np.eye(len(factors)), index=factors, columns=factors)
    correlation.loc["momentum_5", "momentum_20"] = 0.9
    correlation.loc["momentum_20", "momentum_5"] = 0.9

    selected, weights, detailed = select_factors_detailed(
        _summary(),
        correlation,
        max_correlation=0.65,
        max_factors=5,
        max_factors_per_family=2,
        minimum_coverage=0.6,
    )

    assert not {"momentum_5", "momentum_20"}.issubset(selected)
    assert sum(name.startswith("momentum") for name in selected) <= 2
    assert "roa" not in selected
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert set(detailed["factor"]) == set(factors)
