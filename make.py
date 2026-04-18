"""
Separate NetCDF files -> master.nc
"""

import xarray as xr

models = [
    "canesm5",
    "cola-rsmas-ccsm4",
    "cola-rsmas-cesm1",
    "gem5p2-nemo",
    "gfdl-spear",
    "nasa-geoss2s",
    "ncep-cfsv2",
]


res = []


for m in models:
    ds = xr.open_dataset("data/" + m + ".nc")
    ds = ds.assign_coords(model=m)
    res.append(ds)


combined = xr.concat(res, dim="model").transpose("time", "Y", "X", "model")
combined.to_netcdf("data/master.nc")
