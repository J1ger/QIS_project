"""Point-in-time rolling factor-weight estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import calculate_ic_series


def _rebalance_dates(dates: pd.Series, frequency: str) -> list[pd.Timestamp]:
    unique = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    if frequency == "monthly":
        key = unique.dt.to_period("M")
    elif frequency == "quarterly":
        key = unique.dt.to_period("Q")
    else:
        raise ValueError("weight frequency must be 'monthly' or 'quarterly'")
    return list(unique.groupby(key).first())


def _normalize_with_cap(raw: pd.Series, maximum: float) -> pd.Series | None:
    """Normalize absolute weights to one without violating the configured cap."""

    values = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    active = values.abs() > 1e-12
    if int(active.sum()) * maximum < 1.0 - 1e-12:
        return None
    signs = np.sign(values)
    magnitudes = values.abs()
    result = pd.Series(0.0, index=values.index)
    free = active.copy()
    remaining = 1.0
    while free.any() and remaining > 1e-12:
        free_values = magnitudes[free]
        allocation = (
            free_values / free_values.sum() * remaining
            if float(free_values.sum()) > 0
            else pd.Series(remaining / int(free.sum()), index=free_values.index)
        )
        over = allocation > maximum
        if not over.any():
            result.loc[allocation.index] = allocation
            break
        capped_index = allocation[over].index
        result.loc[capped_index] = maximum
        free.loc[capped_index] = False
        remaining = 1.0 - float(result.sum())
    return result * signs


def estimate_rolling_factor_weights(
    data: pd.DataFrame,
    factors: list[str],
    method: str = "icir",
    frequency: str = "monthly",
    lookback_days: int = 756,
    minimum_history_days: int = 252,
    minimum_factor_observations: int | None = None,
    maximum_absolute_weight: float = 0.25,
    exponential_decay: float | None = None,
    smoothing: float = 0.5,
    maximum_weight_change: float | None = None,
) -> pd.DataFrame:
    """Estimate weights using only IC observations strictly before each date."""

    if method not in {"equal", "ic", "icir"}:
        raise ValueError("weight method must be equal, ic, or icir")
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must be between 0 and 1")
    if maximum_absolute_weight <= 0:
        raise ValueError("maximum_absolute_weight must be positive")
    if maximum_weight_change is not None and maximum_weight_change <= 0:
        raise ValueError("maximum_weight_change must be positive when configured")
    ic_data = pd.concat(
        [calculate_ic_series(data, factor) for factor in factors], axis=1
    ).sort_index()
    schedule: list[dict[str, object]] = []
    previous: pd.Series | None = None
    for effective_date in _rebalance_dates(data["date"], frequency):
        history = ic_data.loc[ic_data.index < effective_date].tail(lookback_days)
        if len(history) < minimum_history_days:
            continue
        if method == "equal":
            raw = pd.Series(1.0, index=factors, dtype=float)
        else:
            if exponential_decay is not None and 0.0 < exponential_decay < 1.0:
                ages = np.arange(len(history) - 1, -1, -1, dtype=float)
                decay_weights = pd.Series(exponential_decay**ages, index=history.index)
                means = history.mul(decay_weights, axis=0).sum() / history.notna().mul(
                    decay_weights, axis=0
                ).sum().replace(0, np.nan)
            else:
                means = history.mean()
            raw = means if method == "ic" else means / history.std(ddof=1).replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        minimum_factor_count = (
            minimum_history_days
            if minimum_factor_observations is None
            else minimum_factor_observations
        )
        raw.loc[history.notna().sum() < minimum_factor_count] = 0.0
        if float(raw.abs().sum()) <= 1e-12:
            continue
        weights = _normalize_with_cap(raw, maximum_absolute_weight)
        if weights is None:
            continue
        if previous is not None and smoothing > 0:
            weights = smoothing * previous + (1.0 - smoothing) * weights
            normalized = _normalize_with_cap(weights, maximum_absolute_weight)
            if normalized is None:
                continue
            weights = normalized
        if previous is not None and maximum_weight_change is not None:
            change = (weights - previous).clip(
                lower=-maximum_weight_change,
                upper=maximum_weight_change,
            )
            weights = previous + change
            normalized = _normalize_with_cap(weights, maximum_absolute_weight)
            if normalized is None:
                continue
            weights = normalized
        previous = weights
        for factor, weight in weights.items():
            schedule.append(
                {
                    "effective_date": effective_date,
                    "factor": factor,
                    "weight": float(weight),
                    "history_start": history.index.min(),
                    "history_end": history.index.max(),
                    "history_observations": int(history[factor].notna().sum()),
                    "method": method,
                }
            )
    return pd.DataFrame(schedule)


def build_rolling_composite_score(
    data: pd.DataFrame,
    weight_schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the latest available point-in-time weights to each cross-section."""

    frame = data.copy().sort_values(["date", "symbol"])
    frame["composite_score"] = np.nan
    frame["factor_weight_date"] = pd.NaT
    if weight_schedule.empty:
        return frame
    pivot = weight_schedule.pivot(
        index="effective_date", columns="factor", values="weight"
    ).sort_index()
    for effective_date, weights in pivot.iterrows():
        next_dates = pivot.index[pivot.index > effective_date]
        next_date = next_dates.min() if len(next_dates) else None
        mask = frame["date"] >= effective_date
        if next_date is not None:
            mask &= frame["date"] < next_date
        factors = [factor for factor in weights.dropna().index if factor in frame]
        if not factors:
            continue
        frame.loc[mask, "composite_score"] = frame.loc[mask, factors].fillna(0.0).mul(
            weights[factors], axis=1
        ).sum(axis=1)
        frame.loc[mask, "factor_weight_date"] = effective_date
    return frame
