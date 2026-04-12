"""
Download era5 precipitation data and save necessary parts to file
"""

import xarray as xr
import numpy as np
import os

cmd = "gcloud storage cp -r gs://clim_data_reg_useast1/era5/monthly_means/total_precipitation/ ."

os.system(cmd)


pref = "total_precipitation/era5_total-precipitation_mon_"
suf = "-01.nc"

u = []

for yr in range(1991, 2021):
    for mnth in range(1, 13):
        file = pref + str(yr) + "-" + str(mnth).zfill(2) + suf
        ds = xr.open_dataset(file)

        new_lat = np.arange(-90, 91, 1)
        new_lon = np.arange(0, 360, 1)

        da = ds["tp"].rename({"latitude": "Y", "longitude": "X"})

        v = da.interp(Y=new_lat, X=new_lon) * 1000
        u.append(v)


da = xr.concat(u, dim="time")

path = "data/era5_ppt.nc"
da.to_netcdf(path)

os.system("rm -rf total_precipitation")
