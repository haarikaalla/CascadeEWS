# CascadeEWS real-data signal GNN model card

## Scope

This two-layer graph neural network predicts the repository's defined global
temperature-surge target. Its 19 graph nodes are observed, past-only
temperature, CO₂, energy and RAPID 26.5°N AMOC signals from the five committed datasets.
AMOC is a contextual observation node; the model does **not** predict AMOC collapse or
physical tipping cascades.

## Chronological evaluation

| Horizon | ROC-AUC | Average precision | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 3 months | 0.466 | 0.173 | 0.000 | 0.000 | 0.000 |
| 6 months | 0.245 | 0.121 | 0.125 | 0.077 | 0.333 |
| 8 months | 0.840 | 0.457 | 0.000 | 0.000 | 0.000 |
| 12 months | 0.541 | 0.245 | 0.000 | 0.000 | 0.000 |

Training target windows end before 2015, validation windows cover 2015–2019, and the
2020+ period is used for evaluation. Validation ROC-AUC controls early stopping;
validation labels select both operating thresholds.

## Architecture

- 19 real-signal nodes and
  31 explicit domain edges
- learned node identities plus standardized scalar observations
- two normalized message-passing layers
- attention pooling and binary classification head
- class-weighted loss and validation-based early stopping

## Limitations

The graph represents relationships among measured signals, not geography or physical
climate elements. RAPID observations begin in 2004, shortening the available GNN
training history. Annual OWID signals are lagged and repeated monthly. Attention is a
model diagnostic, not causal attribution. Independent AMOC target definitions, spatial
data and event labels are required before making AMOC-collapse or tipping-cascade claims.
