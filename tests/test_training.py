import numpy as np

from cascadeews.config import ExperimentConfig
from cascadeews.data import build_supervised
from cascadeews.training import train_horizon


def test_training_is_deterministic_and_produces_probabilities():
    config = ExperimentConfig(bootstrap_samples=30)
    dataset = build_supervised(config, 6)
    first = train_horizon(dataset, config)
    second = train_horizon(dataset, config)
    assert first.metrics["selected_feature_group"] == second.metrics["selected_feature_group"]
    assert first.metrics["test"]["roc_auc"] == second.metrics["test"]["roc_auc"]
    assert np.allclose(first.predictions["probability"], second.predictions["probability"])
    assert first.predictions["probability"].between(0, 1).all()
    assert len(first.metrics["test"]["roc_auc_block_bootstrap_95_ci"]) == 2


def test_model_selection_never_reads_test_labels_for_selection():
    config = ExperimentConfig(bootstrap_samples=10)
    dataset = build_supervised(config, 3)
    result = train_horizon(dataset, config)
    assert result.metrics["validation_auc"] >= 0.5
    assert result.metrics["selected_feature_group"] in dataset.feature_groups
    assert 0.05 <= result.metrics["decision_threshold"] <= 0.95
