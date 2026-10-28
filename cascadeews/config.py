"""Experiment configuration with explicit temporal boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("results/validated")
    artifact_dir: Path = Path("artifacts")
    start_date: str = "1980-01-01"
    end_date: str = "2025-01-01"
    train_end: str = "2015-01-01"
    validation_end: str = "2020-01-01"
    horizons: tuple[int, ...] = (3, 6, 8, 12)
    surge_margin_c: float = 0.10
    random_seed: int = 42
    bootstrap_samples: int = 500
    block_months: int = 12

    @property
    def monthly_path(self) -> Path:
        return self.data_dir / "berkeley_monthly.csv"

    @property
    def annual_path(self) -> Path:
        return self.data_dir / "berkeley_annual.csv"

    @property
    def co2_path(self) -> Path:
        return self.data_dir / "owid_co2.csv"

    @property
    def energy_path(self) -> Path:
        return self.data_dir / "owid_energy.csv"
