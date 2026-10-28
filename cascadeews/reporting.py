"""Validated experiment figures and model-card generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_evaluation(metrics: dict[str, Any], predictions: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = {3: "#2563eb", 6: "#7c3aed", 8: "#ea580c", 12: "#dc2626"}
    for horizon, group in predictions.groupby("horizon_months"):
        fpr, tpr, _ = roc_curve(group["actual_alert"], group["probability"])
        auc = metrics["horizons"][str(horizon)]["test"]["roc_auc"]
        axes[0, 0].plot(fpr, tpr, color=colors[int(horizon)], label=f"{horizon}m AUC={auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="#64748b")
    axes[0, 0].set(
        title="Held-out ROC curves", xlabel="False-positive rate", ylabel="True-positive rate"
    )
    axes[0, 0].legend()

    six = predictions.loc[predictions["horizon_months"] == 6]
    axes[0, 1].plot(
        six["target_end"], six["probability"], color=colors[6], label="Alert probability"
    )
    axes[0, 1].scatter(
        six.loc[six["actual_alert"] == 1, "target_end"],
        six.loc[six["actual_alert"] == 1, "probability"],
        color="#dc2626",
        s=22,
        label="Observed surge",
        zorder=3,
    )
    threshold = metrics["horizons"]["6"]["decision_threshold"]
    axes[0, 1].axhline(
        threshold, linestyle="--", color="#0f172a", label=f"Threshold {threshold:.2f}"
    )
    axes[0, 1].set(title="Six-month early-warning timeline", ylabel="Probability")
    axes[0, 1].legend(fontsize=8)

    horizon_labels = sorted(metrics["horizons"], key=int)
    aucs = [metrics["horizons"][value]["test"]["roc_auc"] for value in horizon_labels]
    aps = [metrics["horizons"][value]["test"]["average_precision"] for value in horizon_labels]
    x = np.arange(len(horizon_labels))
    axes[1, 0].bar(x - 0.18, aucs, width=0.36, label="ROC-AUC", color="#2563eb")
    axes[1, 0].bar(x + 0.18, aps, width=0.36, label="Average precision", color="#14b8a6")
    axes[1, 0].set_xticks(x, [f"{value} months" for value in horizon_labels])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set(title="Performance by warning horizon", ylabel="Score")
    axes[1, 0].legend()

    coefficient_items = metrics["horizons"]["6"]["top_coefficients"][:10]
    labels = [str(item["feature"]) for item in reversed(coefficient_items)]
    values = [float(item["coefficient"]) for item in reversed(coefficient_items)]
    axes[1, 1].barh(
        labels, values, color=["#dc2626" if value < 0 else "#16a34a" for value in values]
    )
    axes[1, 1].axvline(0, color="#0f172a", linewidth=0.8)
    axes[1, 1].set(title="Six-month standardized coefficients", xlabel="Log-odds contribution")
    fig.suptitle(
        "CascadeEWS — validated real-data early-warning evaluation", fontsize=16, fontweight="bold"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_model_card(metrics: dict[str, Any], output: Path) -> None:
    six = metrics["horizons"]["6"]
    test = six["test"]
    ci = test["roc_auc_block_bootstrap_95_ci"]
    long_horizon_rows = []
    for horizon in ("8", "12"):
        if horizon in metrics["horizons"]:
            sensitive = metrics["horizons"][horizon]["test"]["high_sensitivity"]
            long_horizon_rows.append(
                f"| {horizon} months | {sensitive['precision']:.3f} | "
                f"{sensitive['recall']:.3f} | {sensitive['f1']:.3f} |"
            )
    long_horizon_table = "\n".join(long_horizon_rows) or "| Not evaluated | — | — | — |"
    content = f"""# CascadeEWS validated model card

## Intended use

Research and portfolio demonstration of a leakage-safe early-warning pipeline.
The validated model estimates whether the future mean global temperature anomaly
will exceed the preceding 12-month mean by at least
{metrics["protocol"]["surge_margin_c"]:.2f} °C. It is **not** an operational
climate-disaster alarm and must not be used for public-safety decisions.

## Data and split

- GCAG monthly global temperature anomaly, 1980–2024.
- OWID CO₂ and energy indicators, lagged 12 months before use.
- Train target windows end before 2015; validation ends before 2020; test target
  windows end in 2020 or later.
- Model family and regularization are selected only on validation ROC-AUC.
- The balanced threshold is selected only on validation F1.
- The high-sensitivity threshold maximizes validation precision subject to at least
  90% validation recall.

## Primary held-out result

- Horizon: 6 months
- ROC-AUC: **{test["roc_auc"]:.3f}**
- 95% 12-month block-bootstrap interval: **{ci[0]:.3f}–{ci[1]:.3f}**
- Average precision: **{test["average_precision"]:.3f}**
- F1: **{test["f1"]:.3f}**
- Precision / recall: **{test["precision"]:.3f} / {test["recall"]:.3f}**
- Test samples: {test["samples"]}

## Long-horizon high-sensitivity operating point

| Horizon | Precision | Recall | F1 |
|---:|---:|---:|---:|
{long_horizon_table}

These are alternative thresholds on the same validation-selected models. They are
reported to make the missed-event/false-alarm tradeoff visible, not as proof of
operational climate-cascade detection.

## Cascade scenario layer

The optional six-element cascade engine exposes Stable, Watch, Warning and Critical
stages across 0–52 scenario weeks for AMOC, Greenland, Arctic sea ice, Amazon, ENSO and
West Antarctica. Its normalized pressure, coupling matrix and stage thresholds are
analyst-controlled assumptions. Scenario transition weeks are **not forecasts** of
physical tipping events and are excluded from the validated metrics above.

## Limitations

- Global monthly signals are not evidence of local tipping points.
- The dataset is small and temporally autocorrelated; confidence intervals use
  block bootstrap, but results still require independent out-of-sample study.
- OWID annual drivers are repeated monthly and intentionally lagged; they are
  contextual signals, not causal attribution.
- Candidate models are interpretable logistic classifiers. The repository's GNN
  remains an experimental research path until evaluated under this same protocol.

## Reproduce

```bash
pip install -e ".[dev]"
cascadeews train
pytest
```
"""
    output.write_text(content)


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
