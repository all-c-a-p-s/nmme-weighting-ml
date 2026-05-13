import xarray as xr

ds1 = xr.open_dataset(
    "preds/precipitation_emos_v01/nmme_ensemble_precipitation_mon_ic-2020-12-01_leads-7_c-afr_emos-v01.nc"
)

ds2 = xr.open_dataset("data/test_predictions_aoi.nc")

print(ds1)
print(ds2)
