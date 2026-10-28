"""Streamlit dashboard for the committed, validated experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional UI
        raise SystemExit('Install the dashboard with: pip install -e ".[dashboard]"') from exc

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
    horizon = st.selectbox(
        "Warning horizon", [3, 6, 12], index=1, format_func=lambda value: f"{value} months"
    )
    result = metrics["horizons"][str(horizon)]
    test = result["test"]
    columns = st.columns(6)
    columns[0].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    columns[1].metric("Average precision", f"{test['average_precision']:.3f}")
    columns[2].metric("F1", f"{test['f1']:.3f}")
    columns[3].metric("Precision", f"{test['precision']:.3f}")
    columns[4].metric("Recall", f"{test['recall']:.3f}")
    columns[5].metric("Test windows", test["samples"])
    ci = test["roc_auc_block_bootstrap_95_ci"]
    st.info(
        f"Selected on validation data: `{result['selected_feature_group']}` features, "
        f"decision threshold {result['decision_threshold']:.3f}. "
        f"Block-bootstrap ROC-AUC interval: {ci[0]:.3f}–{ci[1]:.3f}."
    )
    overview, timeline, comparison, provenance = st.tabs(
        ["Overview", "Warning timeline", "Model comparison", "Data & governance"]
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
            selected[
                ["origin_date", "probability", "predicted_alert", "actual_alert", "surge_c"]
            ].sort_index(ascending=False),
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
    with provenance:
        st.subheader("Evaluation protocol")
        st.json(metrics["protocol"])
        st.subheader("Pinned input fingerprints")
        st.dataframe(pd.DataFrame(metrics["dataset_manifest"]["sources"]), hide_index=True)
        with st.expander("Model card and limitations", expanded=True):
            st.markdown((root / "MODEL_CARD.md").read_text())
