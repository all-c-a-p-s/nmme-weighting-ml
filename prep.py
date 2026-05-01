"""
master.nc + era5_ppt.nc -> .npz file for ML
"""

import xarray as xr
import numpy as np
from util import AOI_ONLY, Y_MIN, Y_MAX, X_MIN, X_MAX

obs = xr.open_dataset("data/era5_ppt.nc")["tp"]
master = xr.open_dataset("data/master.nc")
pred = master["prec_mean"]
gini = master["prec_gini"]

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
all_Ys = pred.Y.values[1:-1]  # ignore poles bc gfdl-spear has NaNs there
all_Xs = pred.X.values

if AOI_ONLY:
    y_mask = (all_Ys >= Y_MIN) & (all_Ys <= Y_MAX)
    x_mask = (all_Xs >= X_MIN) & (all_Xs <= X_MAX)
    Ys = all_Ys[y_mask]
    Xs = all_Xs[x_mask]
    y_slice = slice(
        np.where(y_mask)[0][0] + 1, np.where(y_mask)[0][-1] + 2
    )  # +1 offset for pole removal
    x_slice = slice(np.where(x_mask)[0][0], np.where(x_mask)[0][-1] + 1)
else:
    Ys = all_Ys
    Xs = all_Xs
    y_slice = slice(1, -1)
    x_slice = slice(None)

months = (np.arange(T) + 3) % 12

obs_np = obs.values[3:, y_slice, x_slice]
pred_np = pred.sel(model=models).values[:-3, y_slice, x_slice, :]
gini_np = gini.sel(model=models).values[:-3, y_slice, x_slice, :]
gini_np = np.nan_to_num(gini_np)

n_y = len(Ys)
n_x = len(Xs)

t_idx, y_idx, x_idx = np.meshgrid(
    np.arange(T), np.arange(n_y), np.arange(n_x), indexing="ij"
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
ginis = gini_np.reshape(-1, 7).astype(np.float32)

X_input = np.hstack([X_input, ginis])

out_path = "data/ml_data_aoi.npz" if AOI_ONLY else "data/ml_data.npz"
np.savez(
    out_path, X_input=X_input, y_obs=y_obs, preds=preds, n_y=n_y, n_x=n_x, Ys=Ys, Xs=Xs
)
print(f"saved {len(y_obs)} samples to {out_path}")

