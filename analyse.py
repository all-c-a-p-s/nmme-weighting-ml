"""
Calculate:
- MAE
- RMSE
- N-ile accuracy (accuracy for N %ile groups) per grid cell, month
- ACC per grid cell, month
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

Y_MIN, Y_MAX, X_MIN, X_MAX = -18, 16, 7, 50
PRED_PATH = "data/test_predictions.nc"
ds = xr.open_dataset(PRED_PATH)


LEAD = 3
TEST_INIT_START = 300
N_TEST_RAW = 60
N_TEST_VALID = N_TEST_RAW - LEAD

ds = ds.isel(time=slice(0, N_TEST_VALID))

VERIF_MONTHS = (np.arange(TEST_INIT_START, TEST_INIT_START + N_TEST_VALID) + LEAD) % 12

forecast_global = ds["forecast"].assign_coords(month=("time", VERIF_MONTHS))
obs_global = ds["obs"].assign_coords(month=("time", VERIF_MONTHS))
baseline_global = ds["baseline"].assign_coords(month=("time", VERIF_MONTHS))

forecast_aoi = forecast_global.sel(X=slice(X_MIN, X_MAX), Y=slice(Y_MIN, Y_MAX))
obs_aoi = obs_global.sel(X=slice(X_MIN, X_MAX), Y=slice(Y_MIN, Y_MAX))
baseline_aoi = baseline_global.sel(X=slice(X_MIN, X_MAX), Y=slice(Y_MIN, Y_MAX))

N = 3


def nile_accuracy(forecast, obs, n):
    quantiles = np.linspace(0, 1, n + 1)
    o = obs.values
    f = forecast.values
    months = obs.month.values

    o_bin = np.empty_like(o, dtype=int)
    f_bin = np.empty_like(f, dtype=int)

    for m in range(12):
        idx = np.where(months == m)[0]
        if len(idx) == 0:
            continue
        o_m = o[idx]
        f_m = f[idx]
        bins = np.quantile(o_m, quantiles, axis=0)
        interior = bins[1:-1]
        o_bin[idx] = (o_m[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
        f_bin[idx] = (f_m[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)

    acc = (o_bin == f_bin).mean(axis=0)
    return xr.DataArray(acc, dims=["Y", "X"], coords={"Y": obs.Y, "X": obs.X})


def evaluate(pred, obs, label):
    mae = np.abs(pred - obs).mean("time")
    rmse = np.sqrt(((pred - obs) ** 2).mean("time"))
    nile_acc = nile_accuracy(pred, obs, N)

    obs_clim = obs.groupby("month").mean("time")
    pred_clim = pred.groupby("month").mean("time")
    obs_anom = obs.groupby("month") - obs_clim
    pred_anom = pred.groupby("month") - pred_clim

    acc = (obs_anom * pred_anom).mean("time") / (
        obs_anom.std("time") * pred_anom.std("time")
    )
    print(f"\n--- {label} ---")
    print(f"MAE:        {mae.mean().values:.4f}")
    print(f"RMSE:       {rmse.mean().values:.4f}")
    print(f"{N}-ile acc: {nile_acc.mean().values:.4f}")
    print(f"ACC:        {acc.mean().values:.4f}")
    return {"MAE": mae, "RMSE": rmse, f"{N}-ile Acc": nile_acc, "ACC": acc}


def plot_metrics(metrics, label, filename):
    cmaps = {
        "MAE": ("YlOrRd", None, None),
        "RMSE": ("YlOrRd", None, None),
        f"{N}-ile Acc": ("RdYlGn", 0, 1),
        "ACC": ("RdYlGn", -1, 1),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(label, fontsize=13)
    for ax, (name, da) in zip(axes.flat, metrics.items()):
        cmap, vmin, vmax = cmaps[name]
        vals = da.values
        im = ax.pcolormesh(
            da.X,
            da.Y,
            vals,
            cmap=cmap,
            vmin=vmin if vmin is not None else np.nanpercentile(vals, 2),
            vmax=vmax if vmax is not None else np.nanpercentile(vals, 98),
        )
        ax.set_title(name)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def plot_diff_metrics(fc_metrics, bl_metrics, label, filename):
    # flip signs so that its consistent that smaller value = win for us
    diffs = {
        "MAE": bl_metrics["MAE"] - fc_metrics["MAE"],
        "RMS": bl_metrics["RMSE"] - fc_metrics["RMSE"],
        f"{N}-ile Acc": fc_metrics[f"{N}-ile Acc"] - bl_metrics[f"{N}-ile Acc"],
        "ACC": fc_metrics["ACC"] - bl_metrics["ACC"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(label, fontsize=13)
    for ax, (name, da) in zip(axes.flat, diffs.items()):
        vals = da.values
        lim = np.nanpercentile(np.abs(vals), 98)
        im = ax.pcolormesh(da.X, da.Y, vals, cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_title(name)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


fc_metrics_global = evaluate(forecast_global, obs_global, "Forecast (global)")
bl_metrics_global = evaluate(baseline_global, obs_global, "Baseline (global)")
plot_metrics(fc_metrics_global, "Forecast (global)", "plots/forecast_global.png")
plot_metrics(bl_metrics_global, "Baseline (global)", "plots/baseline_global.png")

fc_metrics_aoi = evaluate(forecast_aoi, obs_aoi, "Forecast (AOI)")
bl_metrics_aoi = evaluate(baseline_aoi, obs_aoi, "Baseline (AOI)")
plot_metrics(fc_metrics_aoi, "Forecast (AOI)", "plots/forecast_aoi.png")
plot_metrics(bl_metrics_aoi, "Baseline (AOI)", "plots/baseline_aoi.png")

plot_diff_metrics(
    fc_metrics_global,
    bl_metrics_global,
    "Forecast vs Baseline (global)",
    "plots/diff_global.png",
)
plot_diff_metrics(
    fc_metrics_aoi, bl_metrics_aoi, "Forecast vs Baseline (AOI)", "plots/diff_aoi.png"
)
