from typing import Tuple
from itertools import pairwise, product
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from jetstream_hugo.definitions import DATADIR, KAPPA, compute
from jetstream_hugo.data import standardize, flatten_by, extract
import numpy as np 
import xarray as xr
import argparse
import traceback
import intake
import intake_esm


varname_to_search_dict = {
    "high_wind": ["U", "V", "T"],
    "mid_wind": ["U500", "V500"],
    "PRECL": "PRECL",
    "TS": "TS",
}
levels_dict = {
    "high_wind": list(range(14, 18)),
    "mid_wind": "all",
    "PRECL": "all",
    "TS": "all",
}
experiment_dict = {
    "past": "historical",
    "future": "ssp370",
}
years = {
    "past": np.arange(1970, 2010),
    "future": np.arange(2060, 2100),
}
for key, val in yearbounds_dict.items():
    yearbounds_dict[key][-1] = val[-1] - 5

parser=argparse.ArgumentParser(description="sample argument parser")
parser.add_argument("period", choices=["past", "future"])
parser.add_argument("variable", choices=list(varname_to_search_dict))
parser.add_argument("n_workers", default=10, type=int)
args=parser.parse_args()
period = args.period
variable = args.variable   
n_workers = args.n_workers   

url = 'https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/intake-catalogs/aws-cesm2-le.json'
component = "atm"
experiment = experiment_dict[period]
yearbounds = yearbounds_dict[period]
varname_to_search = varname_to_search_dict[variable]
levels = levels_dict.get(variable, None)
frequency = "daily"
forcing_variant = "cmip6"
season = None
minlon = -180
maxlon = 180
minlat = 0
maxlat = 90
members = "all"
reduce_da = True
Path(f"{DATADIR}/CESM2/{variable}/{period}").mkdir(exist_ok=True)

indexers = [component, experiment, frequency, forcing_variant]
indexers = [np.atleast_1d(indexer) for indexer in indexers]
indexers = [[str(idx_) for idx_ in indexer] for indexer in indexers]
indexers = list(product(*indexers))
ensemble_keys = [".".join(indexer) for indexer in indexers]

catalog = intake.open_esm_datastore(url)
catalog_subset = catalog.search(
    variable=varname_to_search,
    component=component,
    experiment=experiment,
    frequency=frequency,
    forcing_variant=forcing_variant,
)
dsets = catalog_subset.to_dataset_dict(
    xarray_open_kwargs={"consolidated": False},
    storage_options={"anon": True}
)
ds = dsets[ensemble_keys[0]]
ds = standardize(ds)
members = np.unique(ds.member.values)
def downloader(ds_: xr.Dataset, varname: str | list, period: str, member: str, bounds: Tuple[int]):
    opath = Path(f"{DATADIR}/CESM2/{variable}/{period}/{member}.{bounds[0]}-{bounds[1]}.nc")
    if opath.is_file():
        return f"{member} {bounds[0]}-{bounds[1]} already exists"
    ds = extract(
        ds_,
        period=bounds,
        season=season,
        minlon=minlon,
        maxlon=maxlon,
        minlat=minlat,
        maxlat=maxlat,
        levels=levels_dict[varname],
        members=member,
    )
    if varname in ["high_wind", "mid_wind"]:
        ds["s"] = ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds = compute(ds, progress_flag=True, n_workers=1)
    if varname == "high_wind":
        ds = flatten_by(ds, "s")
        ds["theta"] = (1000 / ds["lev"]) ** KAPPA * ds["t"]
        ds = ds.drop_vars("t")
    for varname in (list(ds.data_vars) + ["lat", "lon"]):
        ds[varname] = ds[varname].astype(np.float32)
    ds = ds.to_netcdf(opath)
    return f"Retrieved {member} {bounds[0]}-{bounds[1]}"
    
bounds = [(y0, y1) for y0, y1 in pairwise(yearbounds_dict[period])]
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = [
        executor.submit(downloader, ds, variable, period, member, bounds_) for member in members for bounds_ in bounds
    ]
    for f in as_completed(futures):
        try:
            print(f.result())
        except Exception:
            print(traceback.format_exc())
            print("could not retrieve")                
