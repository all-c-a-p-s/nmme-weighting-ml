"""
Just a script I'm rewriting to look into the data as needed
"""

import xarray as xr

ds = xr.open_dataset("sample.nc")
print(ds)

v = ds["prec"].sel(L=3.0, X=0.0, Y=0.0).mean()
print(v)
