"""Dataset loading, provenance and leakage-safe feature construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig

DRIVER_FEATURES = (
    "co2",
    "co2_growth_pct",
    "ghg_temperature_contribution",
    "primary_energy",
    "renewables_share",
    "fossil_share",
)


@dataclass
class SupervisedDataset:
    horizon_months: int
    frame: pd.DataFrame
    feature_groups: dict[str, list[str]]
    train_mask: pd.Series
    validation_mask: pd.Series
    test_mask: pd.Series

    def split(self, mask: pd.Series, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        return self.frame.loc[mask, features], self.frame.loc[mask, "target"]


def _rolling_slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _rolling_ar1(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values) < 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _monthly_driver(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Expand annual data monthly and lag it one year for reporting latency."""
    annual = series.dropna().copy()
    annual.index = pd.to_datetime(annual.index.astype(str) + "-01-01")
    return annual.reindex(index, method="ffill").shift(12)


def build_feature_frame(config: ExperimentConfig) -> pd.DataFrame:
    monthly = pd.read_csv(config.monthly_path)
    required = {"Source", "Year", "Mean"}
    if not required.issubset(monthly.columns):
        raise ValueError(f"Berkeley monthly data must contain {sorted(required)}")
    monthly = monthly.loc[monthly["Source"] == "GCAG"].copy()
    monthly["date"] = pd.to_datetime(monthly["Year"], errors="raise")
    monthly = monthly.loc[
        (monthly["date"] >= config.start_date) & (monthly["date"] < config.end_date)
    ].sort_values("date")
    if monthly["date"].duplicated().any():
        raise ValueError("duplicate Berkeley monthly dates")
    temperature = monthly.set_index("date")["Mean"].astype(float)
    frame = pd.DataFrame({"temperature": temperature})

    for lag in (1, 2, 3, 6, 12, 24):
        frame[f"temperature_lag_{lag}"] = temperature.shift(lag)
    for window in (6, 12, 24, 60):
        history = temperature.shift(1).rolling(window)
        frame[f"rolling_mean_{window}"] = history.mean()
        frame[f"rolling_std_{window}"] = history.std()
        frame[f"rolling_slope_{window}"] = history.apply(_rolling_slope, raw=True)
        frame[f"ar1_{window}"] = history.apply(_rolling_ar1, raw=True)

    month = frame.index.month
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["time_years"] = np.arange(len(frame), dtype=float) / 12

    co2 = pd.read_csv(config.co2_path)
    co2 = co2.loc[co2["country"] == "World"].set_index("year")
    energy = pd.read_csv(config.energy_path)
    energy = energy.loc[energy["country"] == "World"].set_index("year")
    sources = {
        "co2": co2["co2"],
        "co2_growth_pct": co2["co2_growth_prct"],
        "ghg_temperature_contribution": co2["temperature_change_from_ghg"],
        "primary_energy": energy["primary_energy_consumption"],
        "renewables_share": energy["renewables_share_energy"],
        "fossil_share": energy["fossil_share_energy"],
    }
    for name, series in sources.items():
        frame[name] = _monthly_driver(series, frame.index)
    return frame


def build_supervised(config: ExperimentConfig, horizon_months: int) -> SupervisedDataset:
    if horizon_months < 1:
        raise ValueError("horizon_months must be positive")
    features = build_feature_frame(config)
    temperature = features["temperature"]
    future = pd.concat(
        [temperature.shift(-step) for step in range(1, horizon_months + 1)], axis=1
    ).mean(axis=1)
    frame = features.copy()
    frame["future_mean_temperature"] = future
    frame["reference_temperature"] = frame["rolling_mean_12"]
    frame["surge_c"] = frame["future_mean_temperature"] - frame["reference_temperature"]
    frame["target"] = frame["surge_c"] >= config.surge_margin_c
    frame["target_end"] = frame.index + pd.offsets.MonthBegin(horizon_months)

    all_features = list(features.columns)
    frame = frame.dropna(subset=[*all_features, "future_mean_temperature"]).copy()
    train_mask = frame["target_end"] < pd.Timestamp(config.train_end)
    validation_mask = (frame["target_end"] >= pd.Timestamp(config.train_end)) & (
        frame["target_end"] < pd.Timestamp(config.validation_end)
    )
    test_mask = frame["target_end"] >= pd.Timestamp(config.validation_end)
    if min(int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum())) == 0:
        raise ValueError("temporal split produced an empty partition")

    seasonal = ["temperature", "temperature_lag_12", "month_sin", "month_cos"]
    history = [name for name in all_features if name not in DRIVER_FEATURES]
    return SupervisedDataset(
        horizon_months=horizon_months,
        frame=frame,
        feature_groups={"seasonal": seasonal, "history": history, "full": all_features},
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annual_consistency_report(config: ExperimentConfig) -> dict[str, float | int]:
    """Reconcile the monthly GCAG series against the independent annual snapshot."""
    monthly = pd.read_csv(config.monthly_path)
    monthly = monthly.loc[monthly["Source"] == "GCAG"].copy()
    monthly["date"] = pd.to_datetime(monthly["Year"], errors="raise")
    monthly["year"] = monthly["date"].dt.year
    monthly_annual = monthly.groupby("year")["Mean"].mean().rename("monthly_aggregation")

    annual = pd.read_csv(config.annual_path)
    annual_series = (
        annual.loc[annual["Source"] == "GCAG"].set_index("Year")["Mean"].rename("annual_file")
    )
    comparison = pd.concat([monthly_annual, annual_series], axis=1).dropna()
    comparison = comparison.loc[
        (comparison.index >= pd.Timestamp(config.start_date).year)
        & (comparison.index < pd.Timestamp(config.end_date).year)
    ]
    if comparison.empty:
        raise ValueError("monthly and annual GCAG files have no overlapping experiment years")
    absolute_error = (comparison["monthly_aggregation"] - comparison["annual_file"]).abs()
    return {
        "overlap_years": len(comparison),
        "mean_absolute_difference_c": float(absolute_error.mean()),
        "max_absolute_difference_c": float(absolute_error.max()),
        "correlation": float(comparison.corr().iloc[0, 1]),
    }


def dataset_manifest(config: ExperimentConfig) -> dict[str, object]:
    paths = [config.monthly_path, config.annual_path, config.co2_path, config.energy_path]
    records = []
    for path in paths:
        rows = sum(1 for _ in path.open("rb")) - 1
        records.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "rows": rows,
                "sha256": sha256_file(path),
            }
        )
    return {
        "sources": records,
        "feature_policy": "past-only rolling features; annual drivers lagged 12 months",
        "temperature_source": "GCAG series in DataHub global-temp monthly snapshot",
        "annual_temperature_consistency": annual_consistency_report(config),
    }
