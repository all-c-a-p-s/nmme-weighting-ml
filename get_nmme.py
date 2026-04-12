"""
Get all model data and save just what we need to .nc files
"""

import xarray as xr
import os

models = [
    "canesm5",
    "cola-rsmas-ccsm4",
    "cola-rsmas-cesm1",
    "gem5p2-nemo",
    "gfdl-spear",
    "nasa-geoss2s",
    "ncep-cfsv2",
]


cpref = "gcloud storage cp -r gs://clim_data_reg_useast1/nmme/monthly/"
csuf = "/precipitation/ ."


def once(mdl):
    get_cmd = cpref + mdl + csuf
    os.system(get_cmd)

    pref = "precipitation/nmme_" + mdl + "_precipitation_mon_ic-"
    suf = "-01_leads-7.nc"
    suf2 = "-01_leads-6.nc"

    u = []

    for yr in range(1991, 2021):
        for mnth in range(1, 13):
            try:
                file = pref + str(yr) + "-" + str(mnth).zfill(2) + suf
                data = xr.open_dataset(file)
            except FileNotFoundError:
                file = pref + str(yr) + "-" + str(mnth).zfill(2) + suf2
                data = xr.open_dataset(file)

            v = data["prec"].sel(M=1.0, L=3.0)
            u.append(v)

    da = xr.concat(u, dim="time")

    path = "data/" + mdl + ".nc"

    da.to_netcdf(path)
    os.system("rm -rf precipitation")


for m in models:
    once(m)
    print("done model " + m)
