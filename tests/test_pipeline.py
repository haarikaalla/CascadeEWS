from dataclasses import replace

from cascadeews.config import ExperimentConfig
from cascadeews.pipeline import run_experiment


def test_pipeline_writes_machine_readable_outputs(tmp_path):
    config = replace(
        ExperimentConfig(),
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "artifacts",
        horizons=(6,),
        bootstrap_samples=10,
    )
    metrics = run_experiment(config)
    assert metrics["horizons"]["6"]["test"]["samples"] > 0
    assert (config.output_dir / "metrics.json").exists()
    assert (config.output_dir / "predictions.csv").exists()
    assert (config.output_dir / "evaluation.png").exists()
    assert (config.output_dir / "MODEL_CARD.md").exists()
    assert (config.artifact_dir / "cascadeews_6m.joblib").exists()
