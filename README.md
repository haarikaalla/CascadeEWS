# CascadeEWS 🌍⚡
## AI-Driven Early Warning and Cascading Climate Instability Intelligence Framework

---

## Paper Title
**"CascadeEWS: A Spatiotemporal Graph Neural Network for Multi-Tipping Cascade
Early Warning Signals in Climate Systems"**



---

## One Command

```bash
python run.py
```

---

## Step-by-Step Instructions

### Step 1 — Install Python + VSCode (if not done)
- Python 3.9+: https://python.org/downloads
- VSCode: https://code.visualstudio.com
- Open VSCode → File → Open Folder → select CascadeEWS_Q1

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the project
```bash
# All at once (recommended):
python run.py

# OR step by step:
python step1_data.py      # ~5 min — downloads 4 real datasets
python step2_graph.py     # ~1 min — validates graph + EDA figure
python step3_train.py     # ~30-60 min CPU — trains model
python step4_evaluate.py  # ~5 min — 8 paper figures + metrics
```

---

## Real Datasets (Auto-Downloaded, No Account Needed)

| # | Dataset | Source | Citation | Size |
|---|---------|--------|----------|------|
| D1 | Berkeley Earth Monthly Temp | github.com/datasets/global-temp | Rohde & Hausfather (2020) Earth Syst. Sci. Data Q1 IF=11.4 | 81KB |
| D2 | Berkeley Earth Annual Temp | github.com/datasets/global-temp | Same | 15KB |
| D3 | OWID CO2 + Climate Indicators | github.com/owid/co2-data | Friedlingstein et al. (2023) Earth Syst. Sci. Data Q1 IF=11.4 | 14MB |
| D4 | OWID Energy Data | github.com/owid/energy-data | Ritchie et al. (2022) Nature Energy Q1 IF=67.7 | 7MB |


---

## Architecture 
```
Real Climate Data (4 datasets)
        ↓
Spatiotemporal Graph
(36×72 grid = 2592 nodes, spatial + teleconnection edges)
        ↓
3-layer GAT (Graph Attention Network)     ← spatial encoding
        ↓
2-layer GRU                               ← temporal dynamics
        ↓
Physics-Informed Constraint               ← CSD theory compliance [NOVEL]
        ↓
Multi-head outputs:
  ├── Regression: global anomaly score
  ├── Classification: Normal/Moderate/Extreme alert
  ├── Per-node risk map
  └── Uncertainty (MC Dropout)            ← epistemic uncertainty [NOVEL]
        ↓
Novel Metrics:
  ├── CPI (Cascade Propagation Index)     ← [NOVEL FORMULA]
  └── EWS (Early Warning Signal score)   ← [NOVEL]
        ↓
Cascade Simulator (SIR-model on graph)   ← [NOVEL]
```

---

## Novel Contributions for Paper

1. **CPI formula** — Cascade Propagation Index combining bifurcation theory + graph topology
2. **Physics-informed training** — CSD theory as a differentiable penalty term
3. **MC Dropout uncertainty** — first climate EWS system with epistemic uncertainty maps
4. **Multi-dataset fusion** — Berkeley Earth + OWID CO2 + OWID Energy jointly
5. **Cascade simulator** — SIR-model propagation on climate teleconnection graph

---

## Output Files

```
results/
├── metrics.txt                         ← MAE, RMSE, R², AUC-ROC, F1, Uncertainty
├── checkpoints/cascadeews_best.pt      ← Trained model
└── figures/
    ├── fig0_data_overview.png          ← EDA: Berkeley Earth + anomaly maps
    ├── fig1_training_curves.png        ← Train/val loss + breakdown
    ├── fig2_prediction_uncertainty.png ← Regression + MC dropout uncertainty
    ├── fig3_risk_heatmaps.png          ← Global risk, uncertainty, CPI, EWS maps
    ├── fig4_ews_timeline.png           ← Alert timeline + EWS scores
    ├── fig5_cascade_simulation.png     ← 24-month cascade propagation
    ├── fig6_cpi_analysis.png           ← CPI temporal + spatial analysis
    ├── fig7_ablation.png               ← Component ablation study
    └── fig8_paper_summary.png          ← Combined summary figure
```

---

## GitHub Upload

```bash
git init
git add .
git commit -m "CascadeEWS: Q1 Climate Early Warning System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CascadeEWS.git
git push -u origin main
```

Then on GitHub:
- Add description: "AI Early Warning System for Climate Tipping Cascades"
- Add topics: `climate-ai` `graph-neural-network` `early-warning` `pytorch` `Q1-paper`

---

## Related Q1 Papers (cite these)

1. Scheffer et al. (2009) *Nature* — Early warning signals for critical transitions
2. Boers (2021) *Nature Climate Change* — AMOC tipping warning signals
3. Lam et al. (2023) *Science* — GraphCast weather forecasting
4. Cachay et al. (2021) *NeurIPS* — GNN for El Niño forecasting
5. Rocha et al. (2018) *Science* — Cascading regime shifts
6. Rohde & Hausfather (2020) *Earth Syst. Sci. Data* — Berkeley Earth data
7. Friedlingstein et al. (2023) *Earth Syst. Sci. Data* — OWID CO2 data

---

## Citation

```bibtex
@article{cascadeews2026,
  title={CascadeEWS: A Spatiotemporal Graph Neural Network for
         Multi-Tipping Cascade Early Warning Signals in Climate Systems},
  author={Your Name},
  journal={Geophysical Research Letters},
  year={2026},
  doi={...},
  url={https://github.com/YOUR_USERNAME/CascadeEWS}
}
```
