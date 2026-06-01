from functools import partial
from pathlib import Path

import xarray as xr
from jetutils.definitions import RESULTS, DATADIR
from jetutils.data import extract
from jetutils.derived_quantities import compute_norm_derivative
from jetutils.geospatial import create_jet_relative_dataset
from jetutils.jet_finding import do_everything, gaussian_smooth_func

ds = xr.open_dataset(
    f"{DATADIR}/ERA5/plev/high_wind/6H/results/8/da.nc"
)
ds = extract(
    ds, minlon=-80, maxlon=40, minlat=15, maxlat=80
)
ds["sigma"] = compute_norm_derivative(ds, "s")
times = ds["time"].values
path_fine = Path(RESULTS, "fine")
path_coarse = Path(RESULTS, "coarse")
path_coarse_bc = Path(RESULTS, "coarse_bc")
path_coarse_bc_s = Path(RESULTS, "coarse_bc_s")

fine_kwargs = {
    "smooth_func": partial(gaussian_smooth_func, sigma_lon=5, sigma_lat=1.3),
    "n_coarsen": 1,
    "hole_size": 8,
}
coarse_kwargs = {
    "smooth_func": partial(gaussian_smooth_func, sigma_lon=1, sigma_lat=0.8),
    "n_coarsen": 3,
    "hole_size": 5,
}
coarse_bc_kwargs = {
    "smooth_func": partial(gaussian_smooth_func, sigma_lon=1, sigma_lat=0.8),
    "n_coarsen": 3,
    "hole_size": 5,
    "do_bias_correct": True,
}
coarse_bc_s_kwargs = {
    "smooth_func": partial(gaussian_smooth_func, sigma_lon=1, sigma_lat=0.8),
    "n_coarsen": 3,
    "hole_size": 5,
    "do_smooth_spline": True,
    "do_bias_correct": True,
}
jets_coarse, _ , _, props_coarse = do_everything(ds, path_coarse, **coarse_kwargs)
jets_coarse_bc, _, _, props_coarse_bc = do_everything(ds, path_coarse_bc, **coarse_bc_kwargs)
jets_coarse_bc_s, _, _, props_coarse_bc_s = do_everything(ds, path_coarse_bc_s, **coarse_bc_s_kwargs)
jets_fine, _, _, props_fine = do_everything(ds, path_fine, **fine_kwargs)
