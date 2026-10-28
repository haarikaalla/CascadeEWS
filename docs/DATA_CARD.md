# CascadeEWS data card

## Dataset purpose

The committed snapshots support a reproducible research demonstration of near-term
global temperature-surge warning. They are not a complete climate-observation archive
and are not suitable on their own for operational or local hazard decisions.

## Sources

| Local file | Upstream project | Selected fields |
|---|---|---|
| `data/raw/berkeley_monthly.csv` | [DataHub global temperature](https://datahub.io/core/global-temp) | `GCAG` monthly global temperature anomaly |
| `data/raw/berkeley_annual.csv` | [DataHub global temperature](https://datahub.io/core/global-temp) | Independent annual reconciliation of the monthly temperature signal |
| `data/raw/owid_co2.csv` | [Our World in Data CO₂ dataset](https://github.com/owid/co2-data) | World CO₂, CO₂ growth and GHG temperature contribution |
| `data/raw/owid_energy.csv` | [Our World in Data energy dataset](https://github.com/owid/energy-data) | World primary energy, renewable share and fossil share |
| `data/raw/amoc_rapid_12_month_mean.csv` | [Met Office AMOC dashboard](https://climate.metoffice.cloud/amoc.html) / RAPID-MOCHA-WBTS | RAPID 26.5°N observed 12-month mean AMOC strength in Sverdrups |

The two temperature filenames are historical repository names. The pipeline inspects
the `Source` column and uses the GCAG series; it does not infer provenance from the
filename.

The AMOC CSV is the dashboard-derived RAPID v2023.1 series cited by DOI
`10.5285/223b34a3-2dc5-c945-e063-7086abc0f274`. The Met Office publishes its dashboard
CSV files under the Open Government Licence. It is used as a contextual GNN node, not
as an AMOC-collapse label.

## Time and transformation policy

- Forecast origins cover 1980–2024 where required features and labels are available.
- Temperature lags and rolling statistics use only data known at the forecast origin.
- Annual OWID variables are forward-filled monthly, then lagged by 12 months to
  approximate publication availability.
- Train target windows end before 2015, validation windows before 2020, and test
  windows in 2020 or later.
- Every run records source file byte size, row count, and SHA-256 fingerprint in
  `results/validated/metrics.json`.
- The annual GCAG file is independently reconciled against averages calculated from
  the monthly GCAG file. Across the 45 experiment years, correlation exceeds 0.999 and
  the maximum absolute difference is below 0.001 °C.

## Target definition

For horizon `h`, an alert is positive when the mean anomaly across the next `h` months
is at least 0.10 °C above the mean of the preceding 12 months. This is a
project-specific warming-surge definition, not a standardized climate-tipping label.

## Known limitations

- Annual variables repeated monthly do not become monthly observations.
- Climate time series are autocorrelated and the evaluated sample is small.
- Global averages hide regional variation and extremes.
- Observational associations cannot establish causal climate mechanisms.
- Upstream revisions may change results; fingerprints make changes detectable.
- The AMOC record begins in 2004 and therefore substantially reduces the available GNN
  training and evaluation history.

## Redistribution

The repository's MIT license covers project code, not third-party dataset ownership.
Review each upstream dataset's current license and attribution rules.
