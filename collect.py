from pathlib import Path
import re
import pandas as pd
import xarray as xr

IN_DIR = Path("preds/precipitation_emos_v01")
OUT = Path("data/emos_2016-01_to_2020-09.nc")

START = "2016-01-01"
END = "2020-09-01"


def get_ic_date(path: Path) -> pd.Timestamp:
    m = re.search(r"ic-(\d{4}-\d{2}-\d{2})", path.name)
    if not m:
        raise ValueError(f"Could not parse init date from {path.name}")
    return pd.Timestamp(m.group(1))


pieces = []

for path in sorted(
    IN_DIR.glob("nmme_ensemble_precipitation_mon_ic-*_leads-7_c-afr_emos-v01.nc")
):
    ic = get_ic_date(path)

    ds = xr.open_dataset(path)

    ds = ds.sortby("Y")

    da = ds["pred_mean"].sel(L=3.0)

    time = ic + pd.DateOffset(months=3)

    da = da.expand_dims(time=[time])
    da = da.rename("forecast")

    pieces.append(da)

forecast = xr.concat(pieces, dim="time")
forecast = forecast.sortby("time")

forecast = forecast.sel(time=slice(START, END))

time_index = pd.Index(forecast.time.values)
keep = ~time_index.duplicated(keep="first")
forecast = forecast.isel(time=keep)

out = xr.Dataset({"forecast": forecast})

ref = xr.open_dataset("data/test_predictions_aoi.nc")

ref = ref.sortby("Y")

out["obs"] = (
    ref["obs"]
    .isel(time=slice(0, len(out.time)))
    .assign_coords(
        time=out.time,
        Y=out.Y,
        X=out.X,
    )
)

out["baseline"] = (
    ref["baseline"]
    .isel(time=slice(0, len(out.time)))
    .assign_coords(
        time=out.time,
        Y=out.Y,
        X=out.X,
    )
)

out = out.transpose("time", "Y", "X")

out.to_netcdf(OUT)

print(ref)
print(out)
print(f"saved to {OUT}")
