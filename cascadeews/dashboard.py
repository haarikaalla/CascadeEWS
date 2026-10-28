"""Streamlit dashboard for the committed, validated experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional UI
        raise SystemExit('Install the dashboard with: pip install -e ".[dashboard]"') from exc

    from .cascade import scenario_summary, simulate_cascade
    from .reporting import load_metrics

    root = Path("results/validated")
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        st.error("Validated outputs are missing. Run `cascadeews train` first.")
        return
    metrics = load_metrics(metrics_path)
    predictions = pd.read_csv(predictions_path, parse_dates=["origin_date", "target_end"])
    st.set_page_config(page_title="CascadeEWS", page_icon="🌍", layout="wide")
    st.title("CascadeEWS · validated climate early-warning lab")
    st.caption("Held-out results from GCAG global temperature + OWID data · research use only")
    horizons = sorted(int(value) for value in metrics["horizons"])
    default_index = horizons.index(6) if 6 in horizons else 0
    horizon = st.selectbox(
        "Warning horizon",
        horizons,
        index=default_index,
        format_func=lambda value: f"{value} months",
    )
    result = metrics["horizons"][str(horizon)]
    test = result["test"]
    operating_mode = st.radio(
        "Operating mode",
        ["Balanced", "High sensitivity"],
        horizontal=True,
        help=(
            "Balanced maximizes validation F1. High sensitivity maximizes validation "
            "precision while requiring at least 90% validation recall."
        ),
    )
    mode_metrics = test if operating_mode == "Balanced" else test["high_sensitivity"]
    threshold_key = (
        "decision_threshold" if operating_mode == "Balanced" else "high_sensitivity_threshold"
    )
    alert_column = "predicted_alert" if operating_mode == "Balanced" else "sensitivity_alert"
    columns = st.columns(6)
    columns[0].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    columns[1].metric("Average precision", f"{test['average_precision']:.3f}")
    columns[2].metric("F1", f"{mode_metrics['f1']:.3f}")
    columns[3].metric("Precision", f"{mode_metrics['precision']:.3f}")
    columns[4].metric("Recall", f"{mode_metrics['recall']:.3f}")
    columns[5].metric("Test windows", test["samples"])
    ci = test["roc_auc_block_bootstrap_95_ci"]
    st.info(
        f"Selected on validation data: `{result['selected_feature_group']}` features, "
        f"{operating_mode.lower()} threshold {result[threshold_key]:.3f}. "
        f"Block-bootstrap ROC-AUC interval: {ci[0]:.3f}–{ci[1]:.3f}."
    )
    overview, timeline, comparison, cascade_tab, provenance = st.tabs(
        [
            "Overview",
            "Warning timeline",
            "Model comparison",
            "Cascade scenario",
            "Data & governance",
        ]
    )
    selected = predictions.loc[predictions["horizon_months"] == horizon].set_index("target_end")
    with overview:
        st.subheader("What the model estimates")
        st.write(
            "The probability that the mean temperature anomaly over the selected future "
            "window will be at least 0.10 °C above the previous 12-month mean."
        )
        st.subheader("Most influential standardized features")
        coefficients = pd.DataFrame(result["top_coefficients"][:12]).set_index("feature")
        st.bar_chart(coefficients)
        st.caption("Coefficient magnitude is model influence, not causal importance.")
    with timeline:
        st.subheader("Locked 2020+ test timeline")
        st.line_chart(selected[["probability", "actual_alert"]])
        st.dataframe(
            selected[["origin_date", "probability", alert_column, "actual_alert", "surge_c"]]
            .rename(columns={alert_column: "active_mode_alert"})
            .sort_index(ascending=False),
            width="stretch",
        )
    with comparison:
        rows = []
        for value, horizon_result in metrics["horizons"].items():
            horizon_test = horizon_result["test"]
            rows.append(
                {
                    "horizon_months": int(value),
                    "features": horizon_result["selected_feature_group"],
                    "roc_auc": horizon_test["roc_auc"],
                    "average_precision": horizon_test["average_precision"],
                    "f1": horizon_test["f1"],
                    "precision": horizon_test["precision"],
                    "recall": horizon_test["recall"],
                }
            )
        comparison_frame = pd.DataFrame(rows).set_index("horizon_months")
        st.dataframe(comparison_frame.style.format(precision=3), width="stretch")
        st.bar_chart(comparison_frame[["roc_auc", "average_precision", "f1"]])
        st.subheader("Validation candidate selection")
        candidate_frame = pd.DataFrame(result["candidate_selection"]).T
        st.dataframe(candidate_frame.style.format(precision=3), width="stretch")
    with cascade_tab:
        st.warning(
            "Scenario analysis only: this graph is not an AMOC forecast and its coupling "
            "strengths are explicit stress-test assumptions, not fitted physical parameters."
        )
        scenario_source = st.radio(
            "Scenario input",
            ["Analyst stress test", "Latest selected-model probability"],
            horizontal=True,
        )
        if scenario_source == "Analyst stress test":
            pressure = st.slider(
                "Illustrative global stress level",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
            )
        else:
            pressure = float(selected.iloc[-1]["probability"])
            st.metric("Scenario pressure from latest evaluation window", f"{pressure:.3f}")
        scenario = simulate_cascade(pressure, weeks=52)
        scenario_wide = scenario.pivot(index="week", columns="element", values="risk")
        st.line_chart(scenario_wide)
        summary = scenario_summary(scenario, pressure)
        st.subheader("Stable → Watch → Warning → Critical stage transitions")
        st.dataframe(pd.DataFrame(summary["elements"]), hide_index=True, width="stretch")
        st.caption(
            "Elements: AMOC, Greenland ice sheet, Arctic sea ice, Amazon rainforest, "
            "ENSO and West Antarctic ice sheet. A missing week means the scenario never "
            "crossed that stage threshold."
        )
    with provenance:
        st.subheader("Evaluation protocol")
        st.json(metrics["protocol"])
        st.subheader("Pinned input fingerprints")
        st.dataframe(pd.DataFrame(metrics["dataset_manifest"]["sources"]), hide_index=True)
        with st.expander("Model card and limitations", expanded=True):
            st.markdown((root / "MODEL_CARD.md").read_text())
