# CascadeEWS validated model card

## Intended use

Research and portfolio demonstration of a leakage-safe early-warning pipeline.
The validated model estimates whether the future mean global temperature anomaly
will exceed the preceding 12-month mean by at least
0.10 °C. It is **not** an operational
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
- ROC-AUC: **0.897**
- 95% 12-month block-bootstrap interval: **0.793–0.976**
- Average precision: **0.771**
- F1: **0.727**
- Precision / recall: **0.640 / 0.842**
- Test samples: 65

## Long-horizon high-sensitivity operating point

| Horizon | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 8 months | 0.579 | 0.524 | 0.550 |
| 12 months | 0.895 | 0.654 | 0.756 |

These are alternative thresholds on the same validation-selected models. They are
reported to make the missed-event/false-alarm tradeoff visible, not as proof of
operational climate-cascade detection.

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
