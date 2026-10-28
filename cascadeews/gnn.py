"""Chronologically validated GNN over observed climate and energy signal nodes."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import ExperimentConfig
from .data import build_supervised, dataset_manifest
from .training import _choose_sensitivity_threshold, _choose_threshold

GRAPH_FEATURES = (
    "temperature",
    "temperature_lag_1",
    "temperature_lag_3",
    "temperature_lag_6",
    "temperature_lag_12",
    "rolling_mean_12",
    "rolling_std_12",
    "rolling_slope_12",
    "ar1_12",
    "co2",
    "co2_growth_pct",
    "ghg_temperature_contribution",
    "primary_energy",
    "renewables_share",
    "fossil_share",
    "amoc_strength",
    "month_sin",
    "month_cos",
    "time_years",
)


def signal_adjacency() -> np.ndarray:
    """Return the explicit observed-signal graph used by the GNN."""
    count = len(GRAPH_FEATURES)
    adjacency = np.zeros((count, count), dtype=np.float32)
    index = {name: position for position, name in enumerate(GRAPH_FEATURES)}

    def connect(left: str, right: str) -> None:
        adjacency[index[left], index[right]] = 1.0
        adjacency[index[right], index[left]] = 1.0

    temperature_nodes = GRAPH_FEATURES[:9]
    for left, right in itertools.pairwise(temperature_nodes):
        connect(left, right)
    for node in temperature_nodes[1:]:
        connect("temperature", node)
    connect("co2", "co2_growth_pct")
    connect("co2", "ghg_temperature_contribution")
    connect("ghg_temperature_contribution", "temperature")
    connect("primary_energy", "co2")
    connect("primary_energy", "fossil_share")
    connect("primary_energy", "renewables_share")
    connect("fossil_share", "co2")
    connect("renewables_share", "co2_growth_pct")
    connect("amoc_strength", "temperature")
    connect("amoc_strength", "rolling_mean_12")
    connect("amoc_strength", "ghg_temperature_contribution")
    connect("month_sin", "month_cos")
    connect("month_sin", "temperature")
    connect("month_cos", "temperature")
    connect("time_years", "temperature")
    connect("time_years", "co2")
    return adjacency


def normalized_adjacency() -> torch.Tensor:
    adjacency = signal_adjacency() + np.eye(len(GRAPH_FEATURES), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.diag(np.power(degree, -0.5))
    return torch.tensor(inverse_sqrt @ adjacency @ inverse_sqrt, dtype=torch.float32)


class GraphLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden, hidden)
        self.neighbor_projection = nn.Linear(hidden, hidden, bias=False)
        self.normalization = nn.LayerNorm(hidden)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjh->bih", adjacency, nodes)
        update = self.self_projection(nodes) + self.neighbor_projection(neighbors)
        return torch.relu(self.normalization(update))


class RealDataSignalGNN(nn.Module):
    """Small graph classifier whose nodes are observed real-world signals."""

    def __init__(self, node_count: int, hidden: int = 16, dropout: float = 0.10) -> None:
        super().__init__()
        self.value_encoder = nn.Linear(1, hidden)
        self.node_embedding = nn.Embedding(node_count, hidden)
        self.layers = nn.ModuleList([GraphLayer(hidden), GraphLayer(hidden)])
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(
        self, values: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_ids = torch.arange(values.shape[1], device=values.device)
        nodes = self.value_encoder(values.unsqueeze(-1)) + self.node_embedding(node_ids)
        for layer in self.layers:
            nodes = layer(nodes, adjacency)
        weights = torch.softmax(self.attention(nodes).squeeze(-1), dim=1)
        pooled = torch.sum(nodes * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1), weights


def _as_tensor(frame: pd.DataFrame, scaler: StandardScaler) -> torch.Tensor:
    values = scaler.transform(frame.loc[:, GRAPH_FEATURES])
    return torch.tensor(values, dtype=torch.float32)


def _attach_amoc(dataset: Any, config: ExperimentConfig) -> None:
    amoc = pd.read_csv(config.amoc_path, comment="#")
    required = {"Year", "Month", "RAPID (12 month mean) (Sv)"}
    if not required.issubset(amoc.columns):
        raise ValueError(f"RAPID AMOC data must contain {sorted(required)}")
    amoc["date"] = pd.to_datetime(
        {"year": amoc["Year"], "month": amoc["Month"], "day": 1}, errors="raise"
    )
    series = amoc.set_index("date")["RAPID (12 month mean) (Sv)"].rename("amoc_strength")
    dataset.frame["amoc_strength"] = series.reindex(dataset.frame.index)
    observed = dataset.frame["amoc_strength"].notna()
    dataset.train_mask &= observed
    dataset.validation_mask &= observed
    dataset.test_mask &= observed
    if (
        min(
            int(dataset.train_mask.sum()),
            int(dataset.validation_mask.sum()),
            int(dataset.test_mask.sum()),
        )
        == 0
    ):
        raise ValueError("AMOC alignment produced an empty temporal partition")


def _train_horizon(
    dataset: Any, config: ExperimentConfig
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    _attach_amoc(dataset, config)
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.set_num_threads(1)
    scaler = StandardScaler().fit(dataset.frame.loc[dataset.train_mask, GRAPH_FEATURES])
    x_train = _as_tensor(dataset.frame.loc[dataset.train_mask], scaler)
    x_validation = _as_tensor(dataset.frame.loc[dataset.validation_mask], scaler)
    x_test = _as_tensor(dataset.frame.loc[dataset.test_mask], scaler)
    y_train = torch.tensor(
        dataset.frame.loc[dataset.train_mask, "target"].to_numpy(dtype=np.float32)
    )
    y_validation = dataset.frame.loc[dataset.validation_mask, "target"].to_numpy(dtype=int)
    y_test = dataset.frame.loc[dataset.test_mask, "target"].to_numpy(dtype=int)
    adjacency = normalized_adjacency()
    model = RealDataSignalGNN(len(GRAPH_FEATURES))
    positive = max(float(y_train.sum()), 1.0)
    negative = max(float(len(y_train) - y_train.sum()), 1.0)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    best_auc = -np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience = 60
    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x_train, adjacency)
        loss = loss_function(logits, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits, _ = model(x_validation, adjacency)
            validation_probability = torch.sigmoid(validation_logits).numpy()
        auc = float(roc_auc_score(y_validation, validation_probability))
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= 80 and epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logits, _ = model(x_validation, adjacency)
        test_logits, attention = model(x_test, adjacency)
        validation_probability = torch.sigmoid(validation_logits).numpy()
        test_probability = torch.sigmoid(test_logits).numpy()
    balanced_threshold = _choose_threshold(pd.Series(y_validation), validation_probability)
    sensitivity_threshold = _choose_sensitivity_threshold(
        pd.Series(y_validation), validation_probability
    )
    balanced_prediction = test_probability >= balanced_threshold
    sensitivity_prediction = test_probability >= sensitivity_threshold
    frame = dataset.frame.loc[dataset.test_mask]
    predictions = pd.DataFrame(
        {
            "origin_date": frame.index,
            "target_end": frame["target_end"].to_numpy(),
            "horizon_months": dataset.horizon_months,
            "actual_alert": y_test,
            "probability": test_probability,
            "predicted_alert": balanced_prediction.astype(int),
            "sensitivity_alert": sensitivity_prediction.astype(int),
        }
    )
    mean_attention = attention.mean(dim=0).numpy()
    attention_records = sorted(
        [
            {"node": node, "mean_attention": float(value)}
            for node, value in zip(GRAPH_FEATURES, mean_attention, strict=True)
        ],
        key=lambda item: item["mean_attention"],
        reverse=True,
    )
    metrics = {
        "horizon_months": dataset.horizon_months,
        "validation_auc": best_auc,
        "best_epoch": best_epoch,
        "balanced_threshold": balanced_threshold,
        "high_sensitivity_threshold": sensitivity_threshold,
        "test": {
            "roc_auc": float(roc_auc_score(y_test, test_probability)),
            "average_precision": float(average_precision_score(y_test, test_probability)),
            "f1": float(f1_score(y_test, balanced_prediction, zero_division=0)),
            "precision": float(precision_score(y_test, balanced_prediction, zero_division=0)),
            "recall": float(recall_score(y_test, balanced_prediction, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_test, balanced_prediction, labels=[False, True]
            ).tolist(),
            "high_sensitivity": {
                "f1": float(f1_score(y_test, sensitivity_prediction, zero_division=0)),
                "precision": float(
                    precision_score(y_test, sensitivity_prediction, zero_division=0)
                ),
                "recall": float(recall_score(y_test, sensitivity_prediction, zero_division=0)),
            },
            "samples": len(y_test),
        },
        "mean_node_attention": attention_records,
    }
    artifact = {
        "state_dict": model.state_dict(),
        "scaler": scaler,
        "graph_features": GRAPH_FEATURES,
        "adjacency": signal_adjacency(),
        "horizon_months": dataset.horizon_months,
        "balanced_threshold": balanced_threshold,
        "high_sensitivity_threshold": sensitivity_threshold,
    }
    return metrics, predictions, artifact


def train_real_data_gnn(
    config: ExperimentConfig | None = None,
    *,
    output_dir: Path = Path("results/gnn"),
    artifact_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model": "two-layer attention-pooled signal GNN",
        "claim_scope": (
            "global temperature-surge prediction with observed AMOC context; "
            "not AMOC-collapse or tipping-point prediction"
        ),
        "graph": {
            "nodes": list(GRAPH_FEATURES),
            "undirected_edges": int(signal_adjacency().sum() // 2),
            "node_semantics": "observed temperature, CO2, energy and RAPID AMOC signals",
        },
        "dataset_manifest": dataset_manifest(config),
        "horizons": {},
    }
    prediction_frames = []
    for horizon in config.horizons:
        dataset = build_supervised(config, horizon)
        metrics, predictions, artifact = _train_horizon(dataset, config)
        report["horizons"][str(horizon)] = metrics
        prediction_frames.append(predictions)
        torch.save(artifact, artifact_dir / f"real_data_gnn_{horizon}m.pt")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    _write_model_card(report, output_dir / "MODEL_CARD.md")
    return report


def _write_model_card(report: dict[str, Any], output: Path) -> None:
    rows = []
    for horizon, result in report["horizons"].items():
        test = result["test"]
        rows.append(
            f"| {horizon} months | {test['roc_auc']:.3f} | "
            f"{test['average_precision']:.3f} | {test['f1']:.3f} | "
            f"{test['precision']:.3f} | {test['recall']:.3f} |"
        )
    content = f"""# CascadeEWS real-data signal GNN model card

## Scope

This two-layer graph neural network predicts the repository's defined global
temperature-surge target. Its {len(GRAPH_FEATURES)} graph nodes are observed, past-only
temperature, CO₂, energy and RAPID 26.5°N AMOC signals from the five committed datasets.
AMOC is a contextual observation node; the model does **not** predict AMOC collapse or
physical tipping cascades.

## Chronological evaluation

| Horizon | ROC-AUC | Average precision | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Training target windows end before 2015, validation windows cover 2015–2019, and the
2020+ period is used for evaluation. Validation ROC-AUC controls early stopping;
validation labels select both operating thresholds.

## Architecture

- {len(report["graph"]["nodes"])} real-signal nodes and
  {report["graph"]["undirected_edges"]} explicit domain edges
- learned node identities plus standardized scalar observations
- two normalized message-passing layers
- attention pooling and binary classification head
- class-weighted loss and validation-based early stopping

## Limitations

The graph represents relationships among measured signals, not geography or physical
climate elements. RAPID observations begin in 2004, shortening the available GNN
training history. Annual OWID signals are lagged and repeated monthly. Attention is a
model diagnostic, not causal attribution. Independent AMOC target definitions, spatial
data and event labels are required before making AMOC-collapse or tipping-cascade claims.
"""
    output.write_text(content)


def gnn_summary(report: dict[str, Any]) -> str:
    lines = ["CascadeEWS real-data signal GNN", "=" * 35]
    for horizon, result in report["horizons"].items():
        test = result["test"]
        lines.append(
            f"{horizon:>2} months | AUC {test['roc_auc']:.3f} | "
            f"AP {test['average_precision']:.3f} | F1 {test['f1']:.3f}"
        )
    return "\n".join(lines)
