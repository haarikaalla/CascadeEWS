"""
CascadeEWS — Complete Pipeline
================================
ONE COMMAND: python run.py

Runs all 4 steps automatically.
"""
import subprocess, sys, os

print("=" * 65)
print("CascadeEWS — AI Climate Instability Intelligence Framework")
print("=" * 65)

# Install dependencies
print("\nInstalling dependencies...")
pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "torch",
        "matplotlib", "tqdm", "requests", "xarray", "netCDF4"]
subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs + ["-q"])
print("✓ Dependencies ready\n")

os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

steps = [
    ("step1_data.py",    "Downloading real datasets + building graph"),
    ("step2_graph.py",   "Validating graph + EDA"),
    ("step3_train.py",   "Training GAT+GRU model with physics constraints"),
    ("step4_evaluate.py","Evaluating + generating 8 paper figures"),
]

for script, desc in steps:
    print(f"\n{'='*65}\n  {desc}\n{'='*65}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"\n✗ {script} failed — check error above")
        sys.exit(1)
    print(f"✓ {desc} complete")

print("\n" + "="*65)
print("✓ CASCADEEWS COMPLETE")
print("  Figures : results/figures/ (8 figures)")
print("  Metrics : results/metrics.txt")
print("  Model   : results/checkpoints/cascadeews_best.pt")
print("="*65)
