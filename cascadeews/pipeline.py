"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import ExperimentConfig
from .data import build_supervised, dataset_manifest
from .reporting import render_evaluation, write_model_card
from .training import train_horizon


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "protocol": {
            "task": "future warming-surge early warning",
            "surge_margin_c": config.surge_margin_c,
            "train_target_end_before": config.train_end,
            "validation_target_end_before": config.validation_end,
            "test_target_end_from": config.validation_end,
            "annual_driver_lag_months": 12,
            "selection_metric": "validation ROC-AUC",
            "balanced_threshold_metric": "validation F1",
            "high_sensitivity_threshold_policy": (
                "maximum validation precision subject to at least 0.90 validation recall"
            ),
        },
        "dataset_manifest": dataset_manifest(config),
        "horizons": {},
    }
    prediction_frames: list[pd.DataFrame] = []
    for horizon in config.horizons:
        dataset = build_supervised(config, horizon)
        result = train_horizon(dataset, config)
        result.metrics["features"] = result.features
        result.metrics["top_coefficients"] = result.coefficients[:15]
        metrics["horizons"][str(horizon)] = result.metrics
        prediction_frames.append(result.predictions)
        joblib.dump(
            {
                "model": result.model,
                "features": result.features,
                "horizon_months": horizon,
                "decision_threshold": result.metrics["decision_threshold"],
                "surge_margin_c": config.surge_margin_c,
            },
            config.artifact_dir / f"cascadeews_{horizon}m.joblib",
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_path = config.output_dir / "metrics.json"
    predictions_path = config.output_dir / "predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    predictions.to_csv(predictions_path, index=False)
    render_evaluation(metrics, predictions, config.output_dir / "evaluation.png")
    write_model_card(metrics, config.output_dir / "MODEL_CARD.md")
    return metrics


def summary(metrics: dict[str, Any]) -> str:
    lines = ["CascadeEWS validated held-out results", "=" * 43]
    for horizon, result in metrics["horizons"].items():
        test = result["test"]
        line = (
            f"{horizon:>2} months | AUC {test['roc_auc']:.3f} | "
            f"AP {test['average_precision']:.3f} | F1 {test['f1']:.3f} | "
            f"model {result['selected_feature_group']}"
        )
        if int(horizon) >= 8:
            sensitive = test["high_sensitivity"]
            line += (
                f" | sensitivity-mode F1 {sensitive['f1']:.3f} "
                f"recall {sensitive['recall']:.3f}"
            )
        lines.append(line)
    return "\n".join(lines)


def locate_results(root: Path = Path(".")) -> Path:
    return root / "results" / "validated" / "metrics.json"
