"""
Calculate:
- MAE
- RMSE
- N-ile accuracy (accuracy for N %ile groups) per grid cell
- ACC per grid cell
"""

import xarray as xr
import numpy as np

ds = xr.open_dataset("data/test_predictions.nc")

AOI_ONLY = True

forecast = (
    ds["forecast"].sel(X=slice(7, 50), Y=slice(-18, 16)) if AOI_ONLY else ds["forecast"]
)
obs = ds["obs"].sel(X=slice(7, 50), Y=slice(-18, 16)) if AOI_ONLY else ds["obs"]
baseline = (
    ds["baseline"].sel(X=slice(7, 50), Y=slice(-18, 16)) if AOI_ONLY else ds["basline"]
)

N = 3


def nile_accuracy(forecast, obs, n):
    quantiles = np.linspace(0, 1, n + 1)
    o = obs.values
    f = forecast.values
    bins = np.quantile(o, quantiles, axis=0)
    interior = bins[1:-1]
    o_bin = (o[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
    f_bin = (f[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
    acc = (o_bin == f_bin).mean(axis=0)
    return xr.DataArray(acc, dims=["Y", "X"], coords={"Y": obs.Y, "X": obs.X})


def evaluate(pred, obs, label):
    mae = np.abs(pred - obs).mean("time")
    rmse = np.sqrt(((pred - obs) ** 2).mean("time"))
    nile_acc = nile_accuracy(pred, obs, N)
    obs_anom = obs - obs.mean("time")
    pred_anom = pred - pred.mean("time")
    acc = (obs_anom * pred_anom).mean("time") / (
        obs_anom.std("time") * pred_anom.std("time")
    )
    print(f"\n--- {label} ---")
    print(f"MAE:        {mae.mean().values:.4f}")
    print(f"RMSE:       {rmse.mean().values:.4f}")
    print(f"{N}-ile acc: {nile_acc.mean().values:.4f}")
    print(f"ACC:        {acc.mean().values:.4f}")


evaluate(forecast, obs, "Senate")
evaluate(baseline, obs, "Baseline (ensemble)")
