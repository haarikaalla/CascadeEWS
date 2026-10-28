from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from cascadeews.config import ExperimentConfig
from cascadeews.gnn import GRAPH_FEATURES, signal_adjacency, train_real_data_gnn


def test_signal_graph_is_symmetric_connected_and_contains_amoc():
    adjacency = signal_adjacency()
    assert adjacency.shape == (len(GRAPH_FEATURES), len(GRAPH_FEATURES))
    assert np.array_equal(adjacency, adjacency.T)
    assert (adjacency.sum(axis=1) > 0).all()
    assert "amoc_strength" in GRAPH_FEATURES


def test_real_data_gnn_trains_and_writes_auditable_outputs(tmp_path):
    config = replace(ExperimentConfig(), horizons=(6,))
    output = tmp_path / "gnn"
    artifacts = tmp_path / "artifacts"
    report = train_real_data_gnn(config, output_dir=output, artifact_dir=artifacts)
    result = report["horizons"]["6"]
    assert 0 <= result["test"]["roc_auc"] <= 1
    assert result["test"]["samples"] > 0
    assert len(result["mean_node_attention"]) == len(GRAPH_FEATURES)
    assert (output / "metrics.json").exists()
    assert (output / "predictions.csv").exists()
    assert (output / "MODEL_CARD.md").exists()
    assert (artifacts / "real_data_gnn_6m.pt").exists()
