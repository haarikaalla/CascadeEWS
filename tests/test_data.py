from pathlib import Path

import pandas as pd

from cascadeews.config import ExperimentConfig
from cascadeews.data import DRIVER_FEATURES, build_supervised, dataset_manifest


def test_real_datasets_are_present_and_fingerprinted():
    manifest = dataset_manifest(ExperimentConfig())
    sources = manifest["sources"]
    assert len(sources) == 4
    assert all(item["rows"] > 100 for item in sources)
    assert all(len(item["sha256"]) == 64 for item in sources)


def test_temporal_partitions_are_purged_by_target_end():
    config = ExperimentConfig()
    dataset = build_supervised(config, 6)
    frame = dataset.frame
    assert frame.loc[dataset.train_mask, "target_end"].max() < pd.Timestamp(config.train_end)
    assert frame.loc[dataset.validation_mask, "target_end"].min() >= pd.Timestamp(config.train_end)
    assert frame.loc[dataset.validation_mask, "target_end"].max() < pd.Timestamp(
        config.validation_end
    )
    assert frame.loc[dataset.test_mask, "target_end"].min() >= pd.Timestamp(config.validation_end)


def test_features_use_lagged_drivers_and_past_only_statistics():
    dataset = build_supervised(ExperimentConfig(), 3)
    assert set(DRIVER_FEATURES).issubset(dataset.feature_groups["full"])
    assert not set(DRIVER_FEATURES).intersection(dataset.feature_groups["history"])
    forbidden = {"target", "target_end", "future_mean_temperature", "surge_c"}
    assert not forbidden.intersection(dataset.feature_groups["full"])
    assert dataset.frame[dataset.feature_groups["full"]].notna().all().all()


def test_config_paths_are_repository_relative():
    config = ExperimentConfig()
    assert config.monthly_path == Path("data/raw/berkeley_monthly.csv")


def test_default_warning_horizons_include_eight_to_twelve_month_range():
    assert ExperimentConfig().horizons == (3, 6, 8, 12)
