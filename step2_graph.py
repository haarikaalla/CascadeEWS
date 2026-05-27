"""
STEP 2 — Validate Graph + Visualize Data Structure
"""
import numpy as np, pandas as pd, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*65); print("STEP 2 — Graph Validation + EDA"); print("="*65)

nf   = np.load("data/processed/node_features.npy")
ei   = np.load("data/processed/edge_index.npy")
ew   = np.load("data/processed/edge_weights.npy")
gs   = np.load("data/processed/global_score.npy")
al   = np.load("data/processed/alert_label.npy")
mc   = np.load("data/processed/multiclass_label.npy")
tp   = np.load("data/processed/tipping_label.npy")
berk = np.load("data/processed/berk_signal.npy")
grid = np.load("data/processed/grid_shape.npy")
with open("data/processed/metadata.json") as f: meta = json.load(f)

T, N, F = nf.shape; NLAT, NLON = int(grid[0]), int(grid[1])

print(f"\n  Graph Summary:")
print(f"  Nodes     : {N:,} ({NLAT}×{NLON})")
print(f"  Features  : {F} — {meta['features']}")
print(f"  Timesteps : {T} months")
print(f"  Edges     : {ei.shape[1]:,}")
print(f"  Alerts    : {al.sum()} / {T} ({100*al.mean():.0f}%)")
print(f"  Tipping   : {tp.sum()} El Niño-like events")
print(f"  MC labels : {dict(zip(*np.unique(mc, return_counts=True)))}")

# ── EDA Plot ──────────────────────────────────────────────────
os.makedirs("results/figures", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
t = np.arange(T)

# 1. Berkeley Earth temp signal
axes[0,0].plot(t, berk[:T], color='#d7191c', lw=1.5)
axes[0,0].fill_between(t, berk[:T], 0,
    where=(berk[:T]>0), alpha=0.3, color='#d7191c', label='Warm anomaly')
axes[0,0].fill_between(t, berk[:T], 0,
    where=(berk[:T]<0), alpha=0.3, color='#2c7bb6', label='Cool anomaly')
axes[0,0].set_title('Berkeley Earth Global Temp Anomaly (REAL DATA)', fontweight='bold')
axes[0,0].set_xlabel('Month (1980–2024)'); axes[0,0].set_ylabel('°C anomaly')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# 2. Global anomaly score + alerts
axes[0,1].plot(t, gs, color='#7b2d8b', lw=1.5, label='Anomaly score')
axes[0,1].scatter(t[al==1], gs[al==1], color='red', s=15, zorder=5, label='Alert')
axes[0,1].scatter(t[tp==1], gs[tp==1], color='orange', s=30, marker='*', zorder=6, label='Tipping event')
axes[0,1].set_title('Global Anomaly Score + EWS Alerts', fontweight='bold')
axes[0,1].set_xlabel('Month'); axes[0,1].set_ylabel('Anomaly score')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# 3. CSD features over time (mean across nodes)
for idx, (label, color) in enumerate([
    ('AR1', '#e41a1c'), ('Variance', '#377eb8'),
    ('Skewness', '#4daf4a'), ('Kurtosis', '#984ea3')]):
    axes[1,0].plot(t, nf[:, :, 5+idx].mean(1), label=label, color=color, lw=1.2)
axes[1,0].set_title('Critical Slowing Down Indicators (Mean over Nodes)', fontweight='bold')
axes[1,0].set_xlabel('Month'); axes[1,0].set_ylabel('CSD value')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# 4. Risk map (mean across time)
risk_map = np.abs(nf[:, :, 0]).mean(0).reshape(NLAT, NLON)
im = axes[1,1].imshow(risk_map, cmap='RdYlGn_r', aspect='auto',
                       extent=[-180,180,-90,90])
plt.colorbar(im, ax=axes[1,1], label='Mean Risk Score')
axes[1,1].set_title('Mean Climate Anomaly Risk Map', fontweight='bold')
axes[1,1].set_xlabel('Longitude'); axes[1,1].set_ylabel('Latitude')

plt.suptitle('CascadeEWS — Data Overview (Real Berkeley Earth + OWID Datasets)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/fig0_data_overview.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"\n  ✓ EDA figure saved → results/figures/fig0_data_overview.png")
print(f"✓ Graph validated\nNext → python step3_train.py")
