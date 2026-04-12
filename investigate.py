import xarray as xr

ds = xr.open_dataset("data/master.nc")["prec"]
gfdl = ds.sel(model="gfdl-spear")
print(gfdl.sel(Y=-90.0).mean())
print(gfdl.sel(Y=90.0).mean())
