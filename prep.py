import xarray as xr
import numpy as np

obs = xr.open_dataset("data/era5_ppt.nc")["tp"]
pred = xr.open_dataset("data/master.nc")["prec"]

models = [
    "canesm5",
    "cola-rsmas-ccsm4",
    "cola-rsmas-cesm1",
    "gem5p2-nemo",
    "gfdl-spear",
    "nasa-geoss2s",
    "ncep-cfsv2",
]

T = 360
Ys = pred.Y.values[1:-1]  # ignore poles bc gfdl-spear has NaNs there
Xs = pred.X.values
months = np.arange(T) % 12

obs_np = obs.values[:, 1:-1, :]
pred_np = pred.sel(model=models).values[:, 1:-1, :, :]

t_idx, y_idx, x_idx = np.meshgrid(
    np.arange(T), np.arange(179), np.arange(360), indexing="ij"
)
t_idx = t_idx.ravel()
y_idx = y_idx.ravel()
x_idx = x_idx.ravel()

m = months[t_idx]
y = Ys[y_idx]
x = Xs[x_idx]

X_input = np.stack(
    [
        np.sin(2 * np.pi * m / 12),
        np.cos(2 * np.pi * m / 12),
        np.sin(2 * np.pi * x / 360),
        np.cos(2 * np.pi * x / 360),
        np.sin(np.pi * (y + 90) / 180),
        np.cos(np.pi * (y + 90) / 180),
    ],
    axis=1,
).astype(np.float32)

y_obs = obs_np.ravel().astype(np.float32)
preds = pred_np.reshape(-1, 7).astype(np.float32)

np.savez("data/ml_data.npz", X_input=X_input, y_obs=y_obs, preds=preds)
print(f"saved {len(y_obs)} samples")

