"""Validation-selected early-warning model training and honest test evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import SupervisedDataset

REGULARIZATION_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass
class HorizonResult:
    horizon_months: int
    model: Pipeline
    features: list[str]
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    coefficients: list[dict[str, float | str]]


def _model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _safe_auc(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else 0.5


def _choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        f1_score(y_true, probability >= threshold, zero_division=0) for threshold in thresholds
    ]
    best = max(scores)
    candidates = [float(t) for t, score in zip(thresholds, scores, strict=True) if score == best]
    return min(candidates, key=lambda value: abs(value - 0.5))


def _block_bootstrap_auc(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    samples: int,
    block_months: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    blocks = [
        np.arange(start, min(start + block_months, len(y_true)))
        for start in range(0, len(y_true), block_months)
    ]
    estimates: list[float] = []
    for _ in range(samples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])[: len(y_true)]
        if len(np.unique(y_true[indices])) == 2:
            estimates.append(float(roc_auc_score(y_true[indices], probability[indices])))
    return estimates


def train_horizon(dataset: SupervisedDataset, config: ExperimentConfig) -> HorizonResult:
    candidates: dict[str, dict[str, float]] = {}
    best: tuple[float, int, float, str, Pipeline, np.ndarray] | None = None
    for group_name, feature_names in dataset.feature_groups.items():
        x_train, y_train = dataset.split(dataset.train_mask, feature_names)
        x_validation, y_validation = dataset.split(dataset.validation_mask, feature_names)
        group_best: tuple[float, float, Pipeline, np.ndarray] | None = None
        for c_value in REGULARIZATION_GRID:
            model = _model(c_value, config.random_seed)
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_validation)[:, 1]
            auc = _safe_auc(y_validation, probability)
            if group_best is None or auc > group_best[0]:
                group_best = (auc, c_value, model, probability)
        assert group_best is not None
        auc, c_value, model, validation_probability = group_best
        candidates[group_name] = {"validation_auc": auc, "c": c_value}
        candidate = (auc, -len(feature_names), -c_value, group_name, model, validation_probability)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None
    validation_auc, _, negative_c, group_name, model, validation_probability = best
    feature_names = dataset.feature_groups[group_name]
    x_validation, y_validation = dataset.split(dataset.validation_mask, feature_names)
    del x_validation
    threshold = _choose_threshold(y_validation, validation_probability)
    x_test, y_test = dataset.split(dataset.test_mask, feature_names)
    test_probability = model.predict_proba(x_test)[:, 1]
    test_prediction = test_probability >= threshold
    auc = _safe_auc(y_test, test_probability)
    bootstrap = _block_bootstrap_auc(
        y_test.to_numpy(dtype=int),
        test_probability,
        samples=config.bootstrap_samples,
        block_months=config.block_months,
        seed=config.random_seed + dataset.horizon_months,
    )
    ci = np.percentile(bootstrap, [2.5, 97.5]).tolist() if bootstrap else [auc, auc]
    matrix = confusion_matrix(y_test, test_prediction, labels=[False, True]).tolist()
    frame = dataset.frame.loc[dataset.test_mask].copy()
    predictions = pd.DataFrame(
        {
            "origin_date": frame.index,
            "target_end": frame["target_end"].to_numpy(),
            "horizon_months": dataset.horizon_months,
            "surge_c": frame["surge_c"].to_numpy(),
            "actual_alert": y_test.to_numpy(dtype=int),
            "probability": test_probability,
            "predicted_alert": test_prediction.astype(int),
        }
    )
    classifier = model.named_steps["classifier"]
    coefficients = sorted(
        [
            {"feature": name, "coefficient": float(value)}
            for name, value in zip(feature_names, classifier.coef_[0], strict=True)
        ],
        key=lambda item: abs(float(item["coefficient"])),
        reverse=True,
    )
    metrics: dict[str, Any] = {
        "horizon_months": dataset.horizon_months,
        "selected_feature_group": group_name,
        "selected_c": -negative_c,
        "decision_threshold": threshold,
        "validation_auc": validation_auc,
        "test": {
            "roc_auc": auc,
            "roc_auc_block_bootstrap_95_ci": ci,
            "average_precision": float(average_precision_score(y_test, test_probability)),
            "f1": float(f1_score(y_test, test_prediction, zero_division=0)),
            "precision": float(precision_score(y_test, test_prediction, zero_division=0)),
            "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
            "brier_score": float(brier_score_loss(y_test, test_probability)),
            "positive_rate": float(y_test.mean()),
            "confusion_matrix": matrix,
            "samples": len(y_test),
        },
        "partitions": {
            "train": int(dataset.train_mask.sum()),
            "validation": int(dataset.validation_mask.sum()),
            "test": int(dataset.test_mask.sum()),
        },
        "candidate_selection": candidates,
    }
    return HorizonResult(
        horizon_months=dataset.horizon_months,
        model=model,
        features=feature_names,
        predictions=predictions,
        metrics=metrics,
        coefficients=coefficients,
    )
