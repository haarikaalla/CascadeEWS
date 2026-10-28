"""Backward-compatible entry point for the validated pipeline."""

from cascadeews.pipeline import run_experiment, summary

if __name__ == "__main__":
    print(summary(run_experiment()))
