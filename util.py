"""
Just a script I'm rewriting to look into the data as needed
"""

import numpy as np

data = np.load("data/ml_data.npz")
X_input = data["X_input"]
y_obs = data["y_obs"]
preds = data["preds"]

models = [
    "canesm5",
    "cola-rsmas-ccsm4",
    "cola-rsmas-cesm1",
    "gem5p2-nemo",
    "gfdl-spear",
    "nasa-geoss2s",
    "ncep-cfsv2",
]

print("=== SHAPES ===")
print(f"X_input: {X_input.shape}")
print(f"y_obs:   {y_obs.shape}")
print(f"preds:   {preds.shape}")

print("\n=== NaNs ===")
print(f"X_input: {np.isnan(X_input).sum()}")
print(f"y_obs:   {np.isnan(y_obs).sum()}")
for i, m in enumerate(models):
    print(f"  {m}: {np.isnan(preds[:, i]).sum()}")

print("\n=== y_obs stats ===")
print(f"  min:  {y_obs.min():.4f}")
print(f"  max:  {y_obs.max():.4f}")
print(f"  mean: {y_obs.mean():.4f}")

print("\n=== preds stats per model ===")
for i, m in enumerate(models):
    col = preds[:, i]
    print(f"  {m}: min={col.min():.4f} max={col.max():.4f} mean={col.mean():.4f}")

print("\n=== sanity check: equal weight == mean ===")
equal_weight = preds.mean(axis=1)
manual_mean = np.stack([preds[:, i] for i in range(7)], axis=1).mean(axis=1)
print(f"  Match: {np.allclose(equal_weight, manual_mean)}")
