from jetutils.data import standardize, extract
from pathlib import Path
import xarray as xr
from jetutils.definitions import DATADIR, KAPPA
from jetutils.jet_finding import gaussian_smooth_func, do_everything
from functools import partial

kwargs = dict(
    n_coarsen=1,
    base_s_thresh=0.55,
    alignment_thresh=0.6,
    int_thresh_factor=0.6,
    hole_size=6,
    smooth_func=partial(gaussian_smooth_func, sigma_lon=2, sigma_lat=0.8),
)

both_jets = {}
both_paths = {}
for run in ["historical", "ssp370"]:
    path = Path(DATADIR, "CESM2/high_wind", run)
    ds = xr.open_dataset(path.joinpath("ds.zarr"))
    ds = standardize(ds, do_chunk=True)
    ds["theta"] = ds["t"] * (1000 / ds["lev"]) ** KAPPA
    ds = extract(
        ds, minlon=-80, maxlon=40, minlat=15, maxlat=80
    )
    jets, ph_jets, props, props_full = do_everything(ds, path.joinpath("results/1"), feature_names=("s", "theta"), n_init=3, **kwargs)
    both_paths[run] = path
    both_jets[run] = ph_jets