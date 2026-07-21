"""Leakage-safe rolling train/validation/test window construction."""

from __future__ import annotations

import pandas as pd


WINDOW_COLUMNS = [
    "window_id",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
]


def generate_walk_forward_windows(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_years: int,
    validation_months: int,
    test_months: int,
    step_months: int,
) -> pd.DataFrame:
    """Generate rolling windows whose test segments never overlap."""

    if min(train_years, validation_months, test_months, step_months) <= 0:
        raise ValueError("walk-forward lengths must all be positive")
    if step_months < test_months:
        raise ValueError("step_months must be at least test_months")
    sample_start = pd.Timestamp(start_date).normalize()
    sample_end = pd.Timestamp(end_date).normalize()
    records: list[dict[str, object]] = []
    train_start = sample_start
    window_number = 1
    while True:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = (
            validation_start
            + pd.DateOffset(months=validation_months)
            - pd.Timedelta(days=1)
        )
        test_start = validation_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > sample_end:
            break
        records.append(
            {
                "window_id": f"WF{window_number:02d}",
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_start = train_start + pd.DateOffset(months=step_months)
        window_number += 1
    windows = pd.DataFrame(records, columns=WINDOW_COLUMNS)
    validate_walk_forward_windows(windows)
    return windows


def validate_walk_forward_windows(windows: pd.DataFrame) -> None:
    """Reject boundary overlap, invalid ordering, or overlapping test windows."""

    if windows.empty:
        return
    missing = set(WINDOW_COLUMNS).difference(windows.columns)
    if missing:
        raise ValueError(f"walk-forward windows missing columns: {sorted(missing)}")
    ordered = windows.copy()
    for column in WINDOW_COLUMNS[1:]:
        ordered[column] = pd.to_datetime(ordered[column])
    invalid = ~(
        (ordered["train_start"] <= ordered["train_end"])
        & (ordered["train_end"] < ordered["validation_start"])
        & (ordered["validation_start"] <= ordered["validation_end"])
        & (ordered["validation_end"] < ordered["test_start"])
        & (ordered["test_start"] <= ordered["test_end"])
    )
    if invalid.any():
        raise ValueError("walk-forward window boundaries are invalid or overlapping")
    tests = ordered.sort_values("test_start")
    if len(tests) > 1 and not (
        tests["test_end"].iloc[:-1].to_numpy()
        < tests["test_start"].iloc[1:].to_numpy()
    ).all():
        raise ValueError("walk-forward test windows must not overlap")
