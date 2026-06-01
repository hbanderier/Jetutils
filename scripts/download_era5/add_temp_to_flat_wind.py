from pathlib import Path
import xarray as xr
from jetstream_hugo.data import standardize, flatten_by
from jetstream_hugo.definitions import DATADIR, YEARS, KAPPA, compute


ar_full_37_1h = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token="anon"),
)
ar_full_37_1h = ar_full_37_1h.sel(
    time=slice(ar_full_37_1h.attrs["valid_time_start"], ar_full_37_1h.attrs["valid_time_stop"])
)
da_temp = standardize(ar_full_37_1h["temperature"])

odirs = {
    f"{level}_{freq}": Path(f"{DATADIR}/ERA5/plev/{level}_wind/{freq}")
    for level in ["low", "high"] for freq in ["6H", "dailymean"]
}
for year in YEARS:
    opaths = {key: odir.joinpath(f"{year}.nc") for key, odir in odirs.items()}
    if all([opath.is_file() for opath in opaths.values()]):
        continue
    pass
    ds = xr.open_mfdataset(f"/storage/workspaces/giub_meteo_impacts/ci01/ERA5/plev/?_to_del/6H/{year}??.nc")
    ds = standardize(ds)
    ds = compute(ds.sel(lev=[175, 200, 225, 250, 300, 350, 700, 850]), progress_flag=True)
    this_temp = compute(da_temp.sel(**ds.coords), progress=True)
    this_temp = this_temp * (1000 / this_temp.lev) ** KAPPA
    ds["theta"] = this_temp
    high = ds.sel(lev=[175, 200, 225, 250, 300, 350])
    low = ds.sel(lev=[700, 850])
    del ds
    high = flatten_by(high, "s")
    high_daily = high.resample(time="1D").mean()
    high.to_netcdf(opaths["high_6H"])
    high_daily.to_netcdf(opaths["high_dailymean"])
    
    low = flatten_by(low, "s")
    low_daily = low.resample(time="1D").mean()
    low.to_netcdf(opaths["low_6H"])
    low_daily.to_netcdf(opaths["low_dailymean"])
