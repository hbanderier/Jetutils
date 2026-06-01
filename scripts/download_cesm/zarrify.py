from jetutils.data import standardize
from jetutils.definitions import DATADIR
from tqdm import tqdm
from pathlib import Path
import numpy as np
import xarray as xr


basepath = Path(f"{DATADIR}/CESM2/high_wind/ssp370")
paths = list(basepath.glob("*-*.nc"))
names = [path.stem.split("-") for path in paths]
members = [name[0] for name in names]
years = [name[1] for name in names]
for i, member in enumerate(tqdm(np.unique(members))):
    ds = standardize(xr.open_mfdataset(basepath.joinpath(f"{member}-*.nc").as_posix()))
    ds_temp = standardize(
        xr.open_mfdataset(basepath.joinpath(f"raw/{member}-*.nc").as_posix())
    )
    ds["t"] = (
        ds_temp["t"]
        .assign_coords(time=ds.time.values)
        .sel(lat=ds["lat"].values, lon=ds["lon"].values)
        .sel(lev=ds["lev"])
        .reset_coords("lev", drop=True)
    )
    ds["member"] = ds["member"].astype("<U15")
    ds = ds.expand_dims("member").copy(deep=True)
    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)
    kwargs = {"mode": "w"} if i == 0 else {"mode": "a", "append_dim": "member"}
    ds.to_zarr(basepath.joinpath("ds.zarr"), **kwargs)
