"""
For each NMME model, produce 12 monthly 2x2 metric maps
(MAE, RMSE, ACC, tercile accuracy) — all computed per grid cell, per month.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

LEAD = 3
N_TIME = 360
N_VALID = N_TIME - LEAD
N_QUANTILES = 3

MODELS = [
    "canesm5",
    "cola-rsmas-ccsm4",
    "cola-rsmas-cesm1",
    "gem5p2-nemo",
    "gfdl-spear",
    "nasa-geoss2s",
    "ncep-cfsv2",
]

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

obs_all = xr.open_dataset("data/era5_ppt.nc")["tp"]
master = xr.open_dataset("data/master.nc")
pred_all = master["prec_mean"]

pred_aligned = pred_all.isel(time=slice(0, N_VALID))
obs_aligned = obs_all.isel(time=slice(LEAD, N_TIME))

verif_months = (np.arange(N_VALID) + LEAD) % 12
pred_aligned = pred_aligned.assign_coords(month=("time", verif_months))
obs_aligned = obs_aligned.assign_coords(month=("time", verif_months))


def monthly_metrics(pred, obs, month_val):
    """Return MAE, RMSE, ACC, tercile-acc for one calendar month."""
    mask = pred.month == month_val
    p = pred.isel(time=mask)
    o = obs.isel(time=mask)

    mae = np.abs(p - o).mean("time")
    rmse = np.sqrt(((p - o) ** 2).mean("time"))

    p_anom = p - p.mean("time")
    o_anom = o - o.mean("time")
    acc = (p_anom * o_anom).mean("time") / (p_anom.std("time") * o_anom.std("time"))

    quantiles = np.linspace(0, 1, N_QUANTILES + 1)
    ov = o.values
    fv = p.values
    bins = np.quantile(ov, quantiles, axis=0)
    interior = bins[1:-1]
    o_bin = (ov[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
    f_bin = (fv[np.newaxis] > interior[:, np.newaxis]).sum(axis=0)
    qacc_vals = (o_bin == f_bin).mean(axis=0)
    qacc = xr.DataArray(qacc_vals, dims=["Y", "X"], coords={"Y": o.Y, "X": o.X})

    return {
        "MAE": mae,
        "RMSE": rmse,
        f"{N_QUANTILES}-ile Acc": qacc,
        "ACC": acc,
    }


def plot_metrics(metrics, title, filepath):
    cmaps = {
        "MAE": ("YlOrRd", None, None),
        "RMSE": ("YlOrRd", None, None),
        f"{N_QUANTILES}-ile Acc": ("RdYlGn", 0, 1),
        "ACC": ("RdYlGn", -1, 1),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=13)
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
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


out_root = "plots/per_model"

for mdl in MODELS:
    mdl_dir = os.path.join(out_root, mdl)
    os.makedirs(mdl_dir, exist_ok=True)

    mdl_pred = pred_aligned.sel(model=mdl)

    for m in range(12):
        metrics = monthly_metrics(mdl_pred, obs_aligned, m)
        tag = f"{mdl} — {MONTH_NAMES[m]}"
        fpath = os.path.join(mdl_dir, f"month_{m + 1:02d}.png")
        plot_metrics(metrics, tag, fpath)
        print(f"  saved {fpath}")

    print(f"  {mdl} complete")

print("\nAll done.")

