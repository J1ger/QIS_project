from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import add_forward_returns, evaluate_factors


def test_forward_returns_do_not_cross_period_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
            ),
            "symbol": ["000001"] * 4,
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )
    periods = {
        "train": ["2020-01-01", "2020-01-02"],
        "validation": ["2020-01-03", "2020-01-03"],
        "test": ["2020-01-06", "2020-01-06"],
    }

    result = add_forward_returns(frame, period=1, periods=periods)

    assert np.isclose(result.loc[0, "forward_return"], 0.1)
    assert result.loc[1:, "forward_return"].isna().all()


def test_rank_ic_and_quantile_evaluation_with_known_signal() -> None:
    dates = pd.bdate_range("2024-01-02", periods=12)
    rows = []
    for date in dates:
        for rank in range(10):
            rows.append(
                {
                    "date": date,
                    "symbol": f"{rank:06d}",
                    "known_factor": float(rank),
                    "forward_return": float(rank) / 1000.0,
                }
            )
    frame = pd.DataFrame(rows)

    summary, ic_series, quantile_returns = evaluate_factors(
        frame,
        ["known_factor"],
        quantiles=5,
        min_observations=5,
    )

    assert len(summary) == 1
    assert np.isclose(summary.loc[0, "ic_mean"], 1.0)
    assert np.isclose(ic_series["known_factor"].dropna(), 1.0).all()
    quantile_means = quantile_returns.groupby("quantile")["mean_return"].mean()
    assert quantile_means.is_monotonic_increasing
