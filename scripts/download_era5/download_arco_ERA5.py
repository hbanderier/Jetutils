from pathlib import Path
import numpy as np
import xarray as xr
from jetstream_hugo.data import standardize
from jetstream_hugo.definitions import DATADIR, compute, YEARS
import argparse

parser = argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("year")
parser.add_argument("varname")
args = parser.parse_args()
year = int(args.year)
varname = args.varname

var_map = {
    "t2m": ["2m_temperature", np.mean],
    "tp": ["total_precipitation", np.sum],
}
long_name = var_map[varname][0]
func = var_map[varname][1]

ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token="anon"),
)
ar_full_37_1h = ds.sel(
    time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"])
)

base_ds = standardize(ar_full_37_1h[long_name].chunk("auto"))
base_ds = (
    base_ds
    .sel(
        lat=base_ds.lat >= 0,
        time=np.isin(base_ds.time.dt.year, YEARS)
    )
    .isel(lon=slice(None, None, 2), lat=slice(None, None, 2))
)

base_path_1 = Path(f"{DATADIR}/ERA5/surf/{varname}/6H")
base_path_1.mkdir(exist_ok=True, parents=True)
base_path_2 = Path(f"{DATADIR}/ERA5/surf/{varname}/daily{func.__name__}")
base_path_2.mkdir(exist_ok=True, parents=True)

six_hourly = base_ds.sel(time=base_ds.time.dt.year == year).resample(time="6h").reduce(func)
opath_1 = base_path_1.joinpath(f"{year}.nc")
opath_2 = base_path_2.joinpath(f"{year}.nc")
if not opath_1.is_file() or not opath_2.is_file():
    six_hourly = compute(six_hourly, progress_flag=True)
    six_hourly.to_netcdf(opath_1)
if not opath_2.is_file():
    daily = six_hourly.resample(time="1d").reduce(func)
    daily.to_netcdf(opath_2)
print(f"Completed {year}")
