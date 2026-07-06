from jetutils.data import standardize
from tqdm import trange
from pathlib import Path
from jetutils.definitions import compute, DATADIR
import xarray as xr


Basepath = Path(DATADIR, "ERA5/plev/uv/6H")
ozarr = Basepath.joinpath("full.zarr")
for year in trange(1959, 2025):
    yearstr = str(year).zfill(4)
    ds = standardize(
        xr.open_mfdataset(
            [Basepath.joinpath(f"{yearstr}{str(month).zfill(2)}.nc") for month in range(1, 13)]
        )
    )
    ds = compute(ds, progress_flag=False)
    ds = (
        ds
        .chunk({"time": -1, "lev": 1, "lon": -1, "lat": -1})
        .drop_encoding()
        )
    if year == 1959:
        kwargs = {"mode": "w"}
    else:
        kwargs = {"mode": "a", "append_dim": "time"}
    kwargs = kwargs | {"consolidated": False}
    ds.to_zarr(ozarr, **kwargs)