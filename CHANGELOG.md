# Changelog

## 1.2.0

- Integrated a deterministic six-element cascade stress-test engine covering AMOC,
  Greenland, Arctic sea ice, Amazon, ENSO and West Antarctica.
- Added explicit Stable, Watch, Warning and Critical stages over a configurable
  0–52-week scenario timeline.
- Added a cascade dashboard tab, model-probability or analyst-defined scenario input,
  CLI command, machine-readable stage outputs, visualization and unit tests.
- Kept this layer explicitly separate from the validated surge predictor: cascade
  stages are transparent scenario assumptions, not observed AMOC forecasts.

## 1.1.0

- Added an independently evaluated 8-month warning horizon alongside the existing
  3, 6, and 12-month forecasts.
- Made dashboard horizon options data-driven so future validated horizons appear
  without hard-coded UI changes.
- Kept the production path interpretable and validation-selected rather than adding
  model complexity without measurable validation benefit.

## 1.0.0

- Added a leakage-safe real-data pipeline for 3, 6, and 12-month warning horizons.
- Added chronological selection, held-out evaluation, block-bootstrap confidence
  intervals, interpretable coefficients, and immutable data fingerprints.
- Added a dashboard, CLI, model card, data card, reproducible outputs, package
  metadata, tests, coverage enforcement, packaging checks, and CI reproduction.
- Added weekly Dependabot updates for Python and GitHub Actions.
- Replaced unsupported headline claims with results generated from committed data.
