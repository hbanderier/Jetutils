import os
import platform
import pickle as pkl
from pathlib import Path
from nptyping import NDArray
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

pf = platform.platform()
if pf.find("cray") >= 0:
    NODE = "DAINT"
    DATADIR = "/scratch/snx3000/hbanderi/data/persistent"
    N_WORKERS = 16
    MEMORY_LIMIT = "4GiB"
elif platform.node()[:4] == "clim":
    NODE = "CLIM"
    DATADIR = "/scratch2/hugo"
    N_WORKERS = 8
    MEMORY_LIMIT = "4GiB"
elif pf.find("el7") >= 0:  # find better later
    NODE = "UBELIX"
    DATADIR = "/storage/scratch/users/hb22g102"
    os.environ["CDO"] = "/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo"
    N_WORKERS = 16
    MEMORY_LIMIT = "5GiB"
else:
    NODE = "LOCAL"
    N_WORKERS = 8
    DATADIR = "../data"
    MEMORY_LIMIT = "2GiB"

COMPUTE_KWARGS = {
    'n_workers': N_WORKERS,
    'memory_limit': MEMORY_LIMIT,
}

CLIMSTOR = "/mnt/climstor/ecmwf/era5/raw"
DEFAULT_VARNAME = "__xarray_dataarray_variable__"

DATERANGEPL = pd.date_range("19590101", "20211231")
YEARSPL = np.unique(DATERANGEPL.year)
DATERANGEPL_SUMMER = DATERANGEPL[np.isin(DATERANGEPL.month, [6, 7, 8])]

DATERANGEPL_EXT = pd.date_range("19400101", "20221231")
YEARSPL_EXT = np.unique(DATERANGEPL_EXT.year)
DATERANGEPL_EXT_SUMMER = DATERANGEPL_EXT[np.isin(DATERANGEPL_EXT.month, [6, 7, 8])]

DATERANGEML = pd.date_range("19770101", "20211231")

WINDBINS = np.arange(0, 25, 0.5)
LATBINS = np.arange(15, 75.1, 0.5)
LONBINS = np.arange(-90, 30, 1)
DEPBINS = np.arange(-25, 25.1, 0.5)

ZOO = [
    "Lat",
    "Int",
    "Shar",
    "Lats",
    "Latn",
    "Tilt",
    "Lon",
    "Lonw",
    "Lone",
    "Dep",
    "Mea",
]

REGIONS = ["S-W", "West", "S-E", "North", "East", "N-E"]

SMALLNAME = {
    "Geopotential": "z",
    "Wind": "s",
    "Temperature": "t",
    "Precipitation": "tp",
}  # Wind speed

PRETTIER_VARNAME = {
    'mean_lon': 'Avg. Longitude',
    'mean_lat': 'Avg. Latitude',
    'Lon': 'Lon. of max. speed',
    'Lat': 'Lat. of max. speed',
    'Spe': 'Max. speed',
    'lon_ext': 'Extent in lon.',
    'lat_ext': 'Extent in lat.',
    'tilt': 'Tilt',
    'sinuosity': 'Sinuosity',
    'int_over_europe': 'Int. speed over Europe',
    'int_all': 'Integrated speed',
    'persistence': 'Persistence',
}

LATEXY_VARNAME = {
    'mean_lon': '$\overline{\lambda}$',
    'mean_lat': '$\overline{\phi}$',
    'Lon': '$\lambda_{s^*}$',
    'Lat': '$\phi_{s^*}$',
    'Spe': '$s^*$',
    'lon_ext': '$\Delta \lambda$',
    'lat_ext': '$\Delta \phi$',
    'tilt': r'$\overline{\frac{\mathrm{d}\phi}{\mathrm{d}\lambda}}$',
    'sinuosity': r'$R^2$',
    'int_over_europe': '$\int_{\mathrm{Eur.}} s \mathrm{d}\lambda$',
    'int_all': '$\int s \mathrm{d}\lambda$',
    'persistence': '$\Delta t$',
}

RADIUS = 6.371e6  # m
OMEGA = 7.2921e-5  # rad.s-1
KAPPA = 0.2854
R_SPECIFIC_AIR = 287.0500676


def degcos(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    return np.sin(x / 180 * np.pi)


def load_pickle(filename: str | Path) -> Any:
    with open(filename, "rb") as handle:
        to_ret = pkl.load(handle)
    return to_ret


def save_pickle(to_save: Any, filename: str | Path) -> None:
    with open(filename, "wb") as handle:
        pkl.dump(to_save, handle)


def case_insensitive_equal(str1: str, str2: str) -> bool:
    """case-insensitive string equality check

    Args:
        str1 (str): first string
        str2 (str): second string

    Returns:
        bool: case insensitive string equality
    """
    return str1.casefold() == str2.casefold()


def infer_sym(to_plot: Any) -> bool:
    max = np.amax(to_plot)
    min = np.amin(to_plot)
    sym = (np.sign(max) == -np.sign(min)) and (
        np.abs(np.log10(np.abs(max)) - np.log10(np.abs(min))) <= 1
    )
    try:
        return sym.item()
    except AttributeError:
        return sym


def labels_to_mask(labels: xr.DataArray | NDArray) -> NDArray:
    if isinstance(labels, xr.DataArray):
        labels = labels.values
    unique_labels = np.unique(labels)
    return labels[:, None] == unique_labels[None, :]


def get_region(da: xr.DataArray | xr.Dataset) -> tuple:
    try:
        return (
            da.lon.min().item(),
            da.lon.max().item(),
            da.lat.min().item(),
            da.lat.max().item(),
        )
    except AttributeError:
        return (
            da.longitude.min().item(),
            da.longitude.max().item(),
            da.latitude.min().item(),
            da.latitude.max().item(),
        )