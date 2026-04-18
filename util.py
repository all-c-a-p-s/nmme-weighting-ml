import numpy as np

d = np.load("data/ml_data.npz")
y = d["y_obs"]  # (N,)
p = d["preds"]  # (N, 7)

models = [
    "canesm5",
    "cola-rsmas-ccsm4",
    "cola-rsmas-cesm1",
    "gem5p2-nemo",
    "gfdl-spear",
    "nasa-geoss2s",
    "ncep-cfsv2",
]

T = 357
NY = 179
NX = 360

# ── per-gridpoint correlations ────────────────────────────────────────────────
# reshape to (T, NY, NX) for obs and (T, NY, NX, 7) for preds
y3 = y.reshape(T, NY, NX)
p4 = p.reshape(T, NY, NX, 7)

# demean over time axis
y_dm = y3 - y3.mean(axis=0, keepdims=True)  # (T, NY, NX)
p_dm = p4 - p4.mean(axis=0, keepdims=True)  # (T, NY, NX, 7)

num = (y_dm[:, :, :, None] * p_dm).sum(axis=0)  # (NY, NX, 7)
denom = np.sqrt((y_dm**2).sum(axis=0)[:, :, None]) * np.sqrt(
    (p_dm**2).sum(axis=0)
)  # (NY, NX, 7)

with np.errstate(invalid="ignore"):
    corr = num / denom  # (NY, NX, 7)  — NaN where variance is 0

print("PER-GRIDPOINT CORRELATIONS WITH OBS  (mean over globe)")
print(f"  {'model':<22}  mean_r   std_r   frac>0   frac>0.2")
for i, name in enumerate(models):
    c = corr[:, :, i].ravel()
    valid = c[np.isfinite(c)]
    print(
        f"  {name:<22}  {valid.mean():+.4f}  {valid.std():.4f}"
        f"   {(valid > 0).mean():.3f}    {(valid > 0.2).mean():.3f}"
    )

# sanity: correlations should be in [-1, 1]
assert np.nanmax(np.abs(corr)) <= 1.0 + 1e-5, "correlation out of range!"
print("\n✓ all correlations in [-1, 1]")

# ── global (pooled) correlations ──────────────────────────────────────────────
print("\nGLOBAL POOLED CORRELATION WITH OBS")
for i, name in enumerate(models):
    r = np.corrcoef(y, p[:, i])[0, 1]
    print(f"  {name:<22}  r = {r:+.4f}")

# ── inter-model correlations ──────────────────────────────────────────────────
print("\nINTER-MODEL CORRELATION MATRIX  (pooled)")
C = np.corrcoef(p.T)  # (7, 7)
header = "  " + "".join(f"{m[:6]:>8}" for m in models)
print(header)
for i, name in enumerate(models):
    row = "  " + f"{name[:12]:<12}" + "".join(f"{C[i, j]:>8.3f}" for j in range(7))
    print(row)

