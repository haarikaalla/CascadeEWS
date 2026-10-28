"""Transparent multi-element climate cascade scenario engine.

This module is a stress-testing tool, not an AMOC forecast. Its assumptions are
explicit so scenario outputs cannot be confused with the validated surge model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ELEMENTS = (
    "AMOC",
    "Greenland ice sheet",
    "Arctic sea ice",
    "Amazon rainforest",
    "ENSO",
    "West Antarctic ice sheet",
)

# Relative response to a user-supplied global stress level. These are scenario
# assumptions, not fitted physical parameters.
SENSITIVITY = np.array([0.82, 0.74, 0.88, 0.68, 0.62, 0.70], dtype=float)

# Directed source -> destination coupling strengths for an illustrative stress test.
COUPLING = np.array(
    [
        [0.00, 0.20, 0.30, 0.24, 0.18, 0.12],
        [0.36, 0.00, 0.42, 0.08, 0.08, 0.18],
        [0.24, 0.30, 0.00, 0.12, 0.16, 0.14],
        [0.18, 0.05, 0.10, 0.00, 0.32, 0.12],
        [0.16, 0.06, 0.12, 0.34, 0.00, 0.22],
        [0.22, 0.18, 0.16, 0.10, 0.18, 0.00],
    ],
    dtype=float,
)

STAGE_ORDER = ("Stable", "Watch", "Warning", "Critical")


def risk_stage(risk: float) -> str:
    if risk < 0.25:
        return "Stable"
    if risk < 0.50:
        return "Watch"
    if risk < 0.75:
        return "Warning"
    return "Critical"


def simulate_cascade(
    global_pressure: float,
    *,
    weeks: int = 52,
    spread_rate: float = 0.08,
    recovery_rate: float = 0.03,
) -> pd.DataFrame:
    """Run an illustrative multi-element cascade scenario.

    ``global_pressure`` is a 0–1 stress-test input selected by the analyst. It is
    deliberately not inferred from AMOC observations, because none are in the repo.
    """
    if not 0 <= global_pressure <= 1:
        raise ValueError("global_pressure must be between 0 and 1")
    if weeks < 1:
        raise ValueError("weeks must be positive")
    if spread_rate < 0 or recovery_rate < 0:
        raise ValueError("rates must be non-negative")

    baseline = global_pressure * SENSITIVITY
    risk = np.clip(baseline * 0.55, 0, 1)
    records: list[dict[str, Any]] = []
    incoming_scale = np.maximum(COUPLING.sum(axis=0), 1.0)
    for week in range(weeks + 1):
        for element, value in zip(ELEMENTS, risk, strict=True):
            records.append(
                {
                    "week": week,
                    "element": element,
                    "risk": float(value),
                    "stage": risk_stage(float(value)),
                }
            )
        propagated = (COUPLING.T @ risk) / incoming_scale
        forcing = 0.10 * (baseline - risk)
        spread = spread_rate * propagated * (1 - risk)
        recovery = recovery_rate * risk * (1 - global_pressure)
        risk = np.clip(risk + forcing + spread - recovery, 0, 1)
    return pd.DataFrame(records)


def scenario_summary(frame: pd.DataFrame, global_pressure: float) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    for element, group in frame.groupby("element", sort=False):
        warning = group.loc[group["risk"] >= 0.50, "week"]
        critical = group.loc[group["risk"] >= 0.75, "week"]
        peak_row = group.loc[group["risk"].idxmax()]
        elements.append(
            {
                "element": element,
                "peak_risk": float(peak_row["risk"]),
                "peak_stage": risk_stage(float(peak_row["risk"])),
                "first_warning_week": int(warning.iloc[0]) if not warning.empty else None,
                "first_critical_week": int(critical.iloc[0]) if not critical.empty else None,
            }
        )
    return {
        "scenario_type": "illustrative stress test; not an operational forecast",
        "global_pressure": global_pressure,
        "weeks": int(frame["week"].max()),
        "stage_thresholds": {"Stable": 0.0, "Watch": 0.25, "Warning": 0.50, "Critical": 0.75},
        "elements": elements,
    }


def write_scenario(
    output_dir: Path,
    *,
    global_pressure: float = 0.60,
    weeks: int = 52,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = simulate_cascade(global_pressure, weeks=weeks)
    summary = scenario_summary(frame, global_pressure)
    frame.to_csv(output_dir / "cascade_stages.csv", index=False)
    (output_dir / "cascade_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for element, group in frame.groupby("element", sort=False):
        axes[0].plot(group["week"], group["risk"], label=element)
    for boundary, label in ((0.25, "Watch"), (0.50, "Warning"), (0.75, "Critical")):
        axes[0].axhline(boundary, linestyle="--", linewidth=0.8, label=label)
    axes[0].set(
        title=f"Illustrative cascade scenario · pressure {global_pressure:.2f}",
        xlabel="Scenario week",
        ylabel="Normalized scenario risk",
        ylim=(0, 1),
    )
    axes[0].legend(fontsize=7, ncol=2)

    heatmap = frame.pivot(index="element", columns="week", values="risk").loc[list(ELEMENTS)]
    image = axes[1].imshow(heatmap, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set(
        title="Stage evolution across connected elements",
        xlabel="Scenario week",
        ylabel="Climate element",
    )
    axes[1].set_yticks(np.arange(len(ELEMENTS)), ELEMENTS)
    fig.colorbar(image, ax=axes[1], label="Scenario risk")
    fig.suptitle("CascadeEWS multi-element stress test · not an AMOC forecast", fontweight="bold")
    fig.savefig(output_dir / "cascade_scenario.png", dpi=180)
    plt.close(fig)
    return summary
