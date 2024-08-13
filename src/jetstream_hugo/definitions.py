import os
import platform
import pickle as pkl
from pathlib import Path
from nptyping import NDArray
from typing import Any, Callable, ClassVar, Dict, Optional
from itertools import groupby
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
import xarray as xr

np.set_printoptions(precision=5, suppress=True)
os.environ["PATH"] += os.pathsep + "/storage/homefs/hb22g102/latex/bin/x86_64-linux/"

pf = platform.platform()
if pf.find("cray") >= 0:
    NODE = "DAINT"
    DATADIR = "/scratch/snx3000/hbanderi/data/persistent"
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = "8GiB"
elif platform.node()[:4] == "clim":
    NODE = "CLIM"
    DATADIR = "/scratch2/hugo"
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = "4GiB"
elif (pf.find("el7") >= 0) or (pf.find("el9") >= 0):  # find better later
    NODE = "UBELIX"
    DATADIR = "/storage/workspaces/giub_meteo_impacts/ci01"
    os.environ["CDO"] = "/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo"
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = int(os.environ.get("SLURM_MEM_PER_NODE", "150000")) // N_WORKERS
    MEMORY_LIMIT = f"{MEMORY_LIMIT // 1000}GB"
else:
    NODE = "LOCAL"
    N_WORKERS = 8
    DATADIR = "../data"
    MEMORY_LIMIT = "2GB"

COMPUTE_KWARGS = {
    "processes": True,
    "threads_per_worker": 1,
    "n_workers": N_WORKERS,
    "memory_limit": MEMORY_LIMIT,
}

CLIMSTOR = "/mnt/climstor/ecmwf/era5/raw"
FIGURES = "/storage/homefs/hb22g102/persistent-extremes-era5/Figures"
RESULTS = "/storage/homefs/hb22g102/persistent-extremes-era5/Results"
DEFAULT_VARNAME = "__xarray_dataarray_variable__"

DATERANGE = pd.date_range("19590101", "20221231")
TIMERANGE = pd.date_range("19590101", "20230101", freq="6h", inclusive="left")
YEARS = np.unique(DATERANGE.year)
DATERANGE_SUMMER = DATERANGE[np.isin(DATERANGE.month, [6, 7, 8])]

SMALLNAME = {
    "Geopotential": "z",
    "Wind": "s",
    "Temperature": "t",
    "Precipitation": "tp",
}  # Wind speed

SHORTHAND = {
    "subtropical": "STJ",
    "polar": "EDJ",
}

PRETTIER_VARNAME = {
    "mean_lon": "Avg. longitude",
    "mean_lat": "Avg. latitude",
    "mean_lev": "Avg. p. level",
    "mean_spe": "Avg. speed",
    "mean_P": "Avg. p. level",
    "mean_theta": "Avg. theta level",
    "lon_star": "Lon. of max. speed",
    "lat_star": "Lat. of max. speed",
    "spe_star": "Max. speed",
    "lon_ext": "Extent in lon.",
    "lat_ext": "Extent in lat.",
    "tilt": "Tilt",
    "waviness1": "Linear waviness",
    "waviness2": "Flat waviness",
    "wavinessR16": "R16 waviness",
    "wavinessDC16": "DC16 waviness",
    "wavinessFV15": "FV15 waviness",
    "width": "Width",
    "int": "Integrated speed",
    "int_low": "Intd. speed low level",
    "int_over_europe": "Intd. speed over Eur.",
    "persistence": "Jet lifetime",
    "exists": "Exists",
    "int_ratio": "Ratio low / high ints",
    "com_speed": "Speed of COM",
    "double_jet_index": "Double jet index",
}

UNITS = {
    "mean_lon": r"$~^{\circ} \mathrm{E}$",
    "mean_lat": r"$~^{\circ} \mathrm{N}$",
    "mean_lev": r"$\mathrm{hPa}$",
    "mean_spe": r"$\mathrm{m} \cdot \mathrm{s}^{-1}$",
    "lon_star": r"$~^{\circ} \mathrm{E}$",
    "lat_star": r"$~^{\circ} \mathrm{N}$",
    "spe_star": r"$\mathrm{m} \cdot \mathrm{s}^{-1}$",
    "lon_ext": r"$~^{\circ} \mathrm{E}$",
    "lat_ext": r"$~^{\circ} \mathrm{N}$",
    "tilt": r"$~^{\circ} \mathrm{N} / ~^{\circ} \mathrm{E}$",
    "waviness1": r"$~^{\circ} \mathrm{N} / ~^{\circ} \mathrm{E}$",
    "waviness2": r"$~^{\circ} \mathrm{N}$",
    "wavinessR16": r"$~^{\circ} \mathrm{N} / ~^{\circ} \mathrm{E}$",
    "wavinessDC16": "$~$",
    "wavinessFV15": "$~$",
    "width": r"$\mathrm{m}$",
    "int": r"$\mathrm{m}^2 \cdot \mathrm{s}^{-1}$",
    "int_low": r"$\mathrm{m}^2 \cdot \mathrm{s}^{-1}$",
    "int_over_europe": r"$\mathrm{m}^2 \cdot \mathrm{s}^{-1}$",
    "persistence": r"$\mathrm{day}$",
    "exists": r"$~$",
    "com_speed": r"$\mathrm{m} \cdot \mathrm{s}^{-1}$",
    "double_jet_index": "$~$",
}

DEFAULT_VALUES = {
    "mean_lon": 0,
    "mean_lat": 45,
    "mean_lev": 250,
    "mean_spe": 0,
    "lon_star": 0,
    "lat_star": 45,
    "spe_star": 0,
    "lon_ext": 0,
    "lat_ext": 0,
    "tilt": 0,
    "waviness1": 0,
    "waviness2": 0,
    "wavinessR16": 0,
    "wavinessDC16": 0,
    "wavinessFV15": 0,
    "width": 0,
    "int": 0,
    "int_low": 0,
    "int_over_europe": 0,
    "persistence": 1,
    "exists": 0,
    "com_speed": 0,
    "double_jet_index": 0,
}

LATEXY_VARNAME = {
    "mean_lon": "$\overline{\lambda}$",
    "mean_lat": "$\overline{\phi}$",
    "mean_lev": "$\overline{p}$",
    "mean_spe": "$\overline{U}$",
    "lon_star": "$\lambda_{s^*}$",
    "lat_star": "$\phi_{s^*}$",
    "spe_star": "$s^*$",
    "lon_ext": "$\Delta \lambda$",
    "lat_ext": "$\Delta \phi$",
    "tilt": r"$\overline{\frac{\mathrm{d}\phi}{\mathrm{d}\lambda}}$",
    "waviness1": "$s_1$",
    "waviness2": "$s_2$",
    "wavinessR16": "$s_3$",
    "wavinessDC16": "$s_4$",
    "wavinessFV15": "$s_5$",
    "width": "$w$",
    "int": "$\int s \mathrm{d}\lambda$",
    "int_low": r"$\int_{700\text{ hPa}} s \mathrm{d}\lambda$",
    "int_over_europe": "$\int_{\mathrm{Eur.}} s \mathrm{d}\lambda$",
    "persistence": "$\Delta t$",
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


def to_zero_one(X):
    Xmin = np.nanmin(X, axis=0)
    Xmax = np.nanmax(X, axis=0)

    return (X - Xmin[None, :]) / (Xmax - Xmin)[None, :], Xmin, Xmax


def revert_zero_one(X, Xmin, Xmax):
    return Xmin[None, :] + (Xmax - Xmin)[None, :] * X


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


def infer_direction(to_plot: Any) -> int:
    max_ = np.nanmax(to_plot)
    min_ = np.nanmin(to_plot)
    try:
        max_ = max_.item()
        min_ = min_.item()
    except AttributeError:
        pass
    sym = np.sign(max_) == -np.sign(min_)
    sym = sym and np.abs(np.log10(np.abs(max_)) - np.log10(np.abs(min_))) <= 2
    if sym:
        return 0
    return 1 if np.abs(max_) > np.abs(min_) else -1


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


def slice_1d(da: xr.DataArray | xr.Dataset, indexers: dict, dim: str = "points"):
    return da.interp(
        {key: xr.DataArray(indexer, dims=dim) for key, indexer in indexers.items()},
        method="linear",
        kwargs=dict(fill_value=None),
    )


def slice_from_df(
    da: xr.DataArray | xr.Dataset, indexer: pd.DataFrame, dim: str = "point"
) -> xr.DataArray | xr.Dataset:
    cols = [col for col in ["lev", "lon", "lat"] if col in indexer and col in da.dims]
    indexer = {col: xr.DataArray(indexer[col].to_numpy(), dims=dim) for col in cols}
    return da.loc[indexer]


def first_elements(arr: NDArray, n_elements: int, sort: bool = False) -> NDArray:
    ndim = arr.ndim
    if ndim > 1 and sort:
        print("sorting output not supported for arrays with ndim > 1")
        sort = False
        raise RuntimeWarning
    idxs = np.argpartition(arr.ravel(), n_elements)[:n_elements]
    if ndim > 1:
        return np.unravel_index(idxs, arr.shape)
    if sort:
        return idxs[np.argsort(arr[idxs])]
    return idxs


def last_elements(arr: NDArray, n_elements: int, sort: bool = False) -> NDArray:
    arr = np.nan_to_num(arr, posinf=0)
    ndim = arr.ndim
    if ndim > 1 and sort:
        print("sorting output not supported for arrays with ndim > 1")
        sort = False
        raise RuntimeWarning
    idxs = np.argpartition(arr.ravel(), -n_elements)[-n_elements:]
    if ndim > 1:
        return np.unravel_index(idxs, arr.shape)
    if sort:
        return idxs[np.argsort(arr[idxs])]
    return idxs


def coarsen_da(
    da: xr.Dataset | xr.DataArray, target_dx: float
) -> xr.Dataset | xr.DataArray:
    dx = (da.lon[1] - da.lon[0]).item()
    coarsening = int(np.round(target_dx / dx))
    return da.coarsen({"lon": coarsening, "lat": coarsening}, boundary="trim").mean()


def get_runs(mask, cyclic: bool = True):
    start = 0
    runs = []
    if cyclic:
        for key, run in groupby(np.tile(mask, 2)):
            if start >= len(mask):
                break
            length = sum(1 for _ in run)
            runs.append((key, start, start + length - 1))
            start += length
        return runs
    for key, run in groupby(mask):
        length = sum(1 for _ in run)
        runs.append((key, start, start + length - 1))
        start += length
    return runs


def get_runs_fill_holes(mask, cyclic: bool = True, hole_size: int = 8):
    runs = get_runs(mask, cyclic=cyclic)
    for run in runs:
        key, start, end = run
        leng = end - start + 1
        if key or leng > hole_size:  # I want negative short spans
            continue
        if start == 0 and (not mask[-1] or not cyclic):
            continue
        if end == len(mask) - 1 and (not mask[0] or not cyclic):
            continue
        end_ = min(len(mask), end + 1)
        mask[start:end_] = ~mask[start:end_]
    runs = get_runs(mask, cyclic=cyclic)
    indices = []
    for run in runs:
        key, start, end = run
        leng = end - start + 1
        if leng > 10 and key:
            indices.append(np.arange(start, end + 1) % len(mask))
    if len(indices) == 0:
        _, start, end = max(runs, key=lambda x: (x[2] - x[1]) * int(x[0]))
        indices.append(np.arange(start, end + 1) % len(mask))
    return indices


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()
