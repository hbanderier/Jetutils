from ast import main
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence
from itertools import groupby
from dataclasses import dataclass, field
import time
import datetime

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from dask.diagnostics import ProgressBar  # if no client
from dask.distributed import progress  # if client

np.set_printoptions(precision=5, suppress=True)

if Path("/scratch/snx3000").is_dir():
    NODE = "DAINT"
    DATADIR = "/scratch/snx3000/hbanderi/data/persistent"
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = "8GiB"
elif Path("/scratch2/hugo").is_dir():
    NODE = "CLIM"
    DATADIR = "/scratch2/hugo"
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = "4GiB"
elif Path("/gws/nopw/j04/aopp").is_dir():
    NODE = "JASMIN"
    DATADIR = "/gws/nopw/j04/aopp/hbanderi/data"
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = int(os.environ.get("SLURM_MEM_PER_NODE", "60000")) // N_WORKERS
    MEMORY_LIMIT = f"{MEMORY_LIMIT // 1000}GB"
    FIGURES = "/home/users/hbanderi/Henrik_data/Figures"
    RESULTS = "/home/users/hbanderi/Henrik_data/Figures"
elif Path("/storage/workspaces/giub_meteo_impacts/ci01").is_dir():
    NODE = "UBELIX"
    DATADIR = "/storage/workspaces/giub_meteo_impacts/ci01"
    os.environ["CDO"] = "/storage/homefs/hb22g102/mambaforge/envs/env11/bin/cdo"
    os.environ["PATH"] += (
        os.pathsep + "/storage/homefs/hb22g102/latex/bin/x86_64-linux/"
    )
    N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    MEMORY_LIMIT = int(os.environ.get("SLURM_MEM_PER_NODE", "150000")) // N_WORKERS
    MEMORY_LIMIT = f"{MEMORY_LIMIT // 1000}GB"
    FIGURES = "/storage/homefs/hb22g102/persistent-extremes-era5/Figures"
    RESULTS = "/storage/homefs/hb22g102/persistent-extremes-era5/Results"
else:
    NODE = "LOCAL"
    N_WORKERS = 8
    DATADIR = "../data"
    MEMORY_LIMIT = "2GB"
    FIGURES = "/Users/bandelol/Documents/code_local/local_figs"
    RESULTS = "/Users/bandelol/Documents/code_local/data/results"
COMPUTE_KWARGS = {
    "processes": True,
    "threads_per_worker": 1,
    "n_workers": N_WORKERS,
    "memory_limit": MEMORY_LIMIT,
}

CLIMSTOR = "/mnt/climstor/ecmwf/era5/raw"
DEFAULT_VARNAME = "__xarray_dataarray_variable__"

DATERANGE = pd.date_range("19590101", "20221231")
TIMERANGE = pd.date_range("19590101", "20230101", freq="6h", inclusive="left")
LEAPYEAR = pd.date_range("19600101", "19601231")
JJADOYS = LEAPYEAR[np.isin(LEAPYEAR.month, [6, 7, 8])].dayofyear
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
    "mean_s": "Avg. speed",
    "mean_P": "Avg. p. level",
    "mean_theta": "Avg. theta level",
    "lon_star": "Lon. of max. speed",
    "lat_star": "Lat. of max. speed",
    "s_star": "Max. speed",
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
    "persistence": "Lifetime",
    "njets": r"\# Jets",
    "int_ratio": "Ratio low / high ints",
    "com_speed": "COM speed",
    "double_jet_index": "Double jet index",
}

UNITS = {
    "mean_lon": r"$~^{\circ} \mathrm{E}$",
    "mean_lat": r"$~^{\circ} \mathrm{N}$",
    "mean_lev": r"$\mathrm{hPa}$",
    "mean_s": r"$\mathrm{m} \cdot \mathrm{s}^{-1}$",
    "lon_star": r"$~^{\circ} \mathrm{E}$",
    "lat_star": r"$~^{\circ} \mathrm{N}$",
    "s_star": r"$\mathrm{m} \cdot \mathrm{s}^{-1}$",
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
    "njets": r"$~$",
    "com_speed": r"$\mathrm{m} \cdot \mathrm{s}^{-1}$",
    "double_jet_index": "$~$",
}

DEFAULT_VALUES = {
    "mean_lon": 0,
    "mean_lat": 45,
    "mean_lev": 250,
    "mean_s": 0,
    "lon_star": 0,
    "lat_star": 45,
    "s_star": 0,
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
    "njets": 0,
    "com_speed": 0,
    "double_jet_index": 0,
}

LATEXY_VARNAME = {
    "mean_lon": r"$\overline{\lambda}$",
    "mean_lat": r"$\overline{\phi}$",
    "mean_lev": r"$\overline{p}$",
    "mean_s": r"$\overline{U}$",
    "lon_star": r"$\lambda_{s^*}$",
    "lat_star": r"$\phi_{s^*}$",
    "s_star": r"$s^*$",
    "lon_ext": r"$\Delta \lambda$",
    "lat_ext": r"$\Delta \phi$",
    "tilt": r"$\overline{\frac{\mathrm{d}\phi}{\mathrm{d}\lambda}}$",
    "waviness1": r"$s_1$",
    "waviness2": r"$s_2$",
    "wavinessR16": r"$s_3$",
    "wavinessDC16": r"$s_4$",
    "wavinessFV15": r"$s_5$",
    "width": "$w$",
    "int": r"$\int s \mathrm{d}\lambda$",
    "int_low": r"$\int_{700\text{ hPa}} s \mathrm{d}\lambda$",
    "int_over_europe": r"$\int_{\mathrm{Eur.}} s \mathrm{d}\lambda$",
    "persistence": r"$\Delta t$",
}

SEASONS = {
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

RADIUS = 6.371e6  # m
OMEGA = 7.2921e-5  # rad.s-1
KAPPA = 0.2854
R_SPECIFIC_AIR = 287.0500676


def degcos(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    return np.sin(x / 180 * np.pi)


def save_pickle(to_save: Any, filename: str | Path) -> None:
    with open(filename, "wb") as handle:
        pkl.dump(to_save, handle)


def load_pickle(filename: str | Path) -> Any:
    with open(filename, "rb") as handle:
        to_ret = pkl.load(handle)
    return to_ret


def to_zero_one(X):
    def expr(col):
        return (pl.col(col) - pl.col(col).min()) / (
            pl.col(col).max() - pl.col(col).min()
        )

    if isinstance(X, pl.DataFrame):
        Xmin = X.min()
        Xmax = X.max()
        X = X.with_columns(expr(col) for col in X.columns)
        return X, Xmin, Xmax
    Xmin = np.nanmin(X, axis=0)
    Xmax = np.nanmax(X, axis=0)
    try:
        X = (X - Xmin[None, :]) / (Xmax - Xmin)[None, :]
    except IndexError:
        X = (X - Xmin) / (Xmax - Xmin)
    return X, Xmin, Xmax


def revert_zero_one(X, Xmin, Xmax):
    def expr(col):
        return Xmin[0, col] + (Xmax[0, col] - Xmin[0, col]) * pl.col(col)

    if isinstance(X, pl.DataFrame):
        X = X.with_columns(expr(col).alias(col) for col in X.columns)
        return X
    try:
        X = Xmin[None, :] + (Xmax - Xmin)[None, :] * X
    except IndexError:
        X = Xmin + (Xmax - Xmin) * X
    return X


def normalize(X):
    def expr(col):
        return (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()

    if isinstance(X, pl.DataFrame):
        meanX = X.mean()
        stdX = X.std()
        X = X.with_columns(expr(col) for col in X.columns)
        return X, meanX, stdX
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0)
    try:
        X = (X - meanX[None, :]) / stdX[None, :]
    except IndexError:
        X = (X - meanX) / stdX
    return X, meanX, stdX


def revert_normalize(X, meanX, stdX):
    def expr(col):
        return meanX[0, col] + stdX[0, col] * pl.col(col)

    if isinstance(X, pl.DataFrame):
        X = X.with_columns(expr(col).alias(col) for col in X.columns)
        return X
    try:
        X = X * stdX[None, :] + meanX[None, :]
    except IndexError:
        X = X * stdX + meanX
    return X


def xarray_to_polars(da: xr.DataArray | xr.Dataset):
    return pl.from_pandas(da.to_dataframe().reset_index())


def polars_to_xarray(df: pl.DataFrame, index_columns: Sequence[str]):
    ds = xr.Dataset.from_dataframe(df.to_pandas().set_index(index_columns))
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        ds = ds[data_vars[0]]
    return ds


def get_index_columns(
    df,
    potentials: tuple = (
        "member",
        "time",
        "cluster",
        "jet ID",
        "spell",
        "relative_index",
        "relative_time",
    ),
):
    index_columns = [ic for ic in potentials if ic in df.columns]
    return index_columns


def extract_season_from_df(
    df: pl.DataFrame,
    season: list | str | tuple | int | None = None,
) -> pl.DataFrame:
    if season is None:
        return df
    if isinstance(season, str):
        season = SEASONS[season]
    if isinstance(season, int):
        season = [season]
    return df.filter(pl.col("time").dt.month().is_in(season))


def case_insensitive_equal(str1: str, str2: str) -> bool:
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


def labels_to_mask(labels: xr.DataArray | np.ndarray) -> np.ndarray:
    if isinstance(labels, xr.DataArray):
        labels = labels.values
    unique_labels = np.unique(labels)
    return labels[..., None] == unique_labels[None, :]


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


def first_elements(arr: np.ndarray, n_elements: int, sort: bool = False) -> np.ndarray:
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


def last_elements(arr: np.ndarray, n_elements: int, sort: bool = False) -> np.ndarray:
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


def explode_rle(df):
    return df.with_columns(
        index=(pl.int_ranges(pl.col("start"), pl.col("start") + pl.col("len"))).cast(
            pl.List(pl.UInt32)
        )
    ).explode("index")


def do_rle(
    df: pl.DataFrame, group_by: Sequence[str] | Sequence[pl.Expr]
) -> pl.DataFrame:
    conditional = (
        df.group_by(group_by, maintain_order=True)
        .agg(
            pl.col("condition").rle().alias("rle"),
        )
        .explode("rle")
        .unnest("rle")
    )
    conditional = (
        conditional.group_by(group_by, maintain_order=True)
        .agg(
            len=pl.col("len"),
            start=pl.lit(0).append(
                pl.col("len").cum_sum().slice(0, pl.col("len").len() - 1)
            ),
            value=pl.col("value"),
        )
        .explode(["len", "start", "value"])
    )
    return conditional


def do_rle_fill_hole(
    df: pl.DataFrame,
    condition_expr: pl.Expr,
    group_by: str | Sequence[str] | None = None,
    hole_size: int | datetime.timedelta = 4,
    unwrap: bool = False,
) -> pl.DataFrame:
    if isinstance(hole_size, datetime.timedelta):
        if "time" not in df.columns or (group_by is not None and "time" in group_by):
            raise ValueError
        times = df["time"].unique().bottom_k(2).sort()
        dt = times[1] - times[0]
        hole_size = int(hole_size / dt)  # I am not responsible for rounding errors
    to_drop = []
    if group_by is None:
        if "contour" in df.columns:
            group_by = get_index_columns(
                df, ("member", "time", "cluster", "contour", "spell", "relative_index")
            )
        else:
            group_by = []
            group_by.extend(get_index_columns(df, ["member", "cluster"]))
            if "time" in df.columns:
                unique_months = df["time"].dt.month().unique().sort()
                n_months = unique_months.shape[0]
                if n_months < 12:
                    index_jump = (unique_months.diff().fill_null(1) > 1).arg_max()
                    indices = (np.arange(n_months) + index_jump) % n_months
                    dmonth = 13 - unique_months[int(indices[0])] 
                    dmonth = dmonth if index_jump != 0 else 0
                    df = df.with_columns(pl.col("time").dt.offset_by(f"{dmonth}mo"))
                    df = df.with_columns(year=pl.col("time").dt.year())
                    group_by.append("year")
                orig_time = df[["time", *group_by]].clone()
                orig_time = orig_time.with_columns(
                    year=pl.col("time").dt.year(),
                )
    if not isinstance(group_by, Sequence):
        group_by = [group_by]
    
    if len(group_by) == 0:
        df = df.with_columns(dummy=1)
        group_by.append("dummy")
        to_drop.append("dummy")
    df = (
        df.group_by(group_by, maintain_order=True)
        .agg(
            condition_expr.alias("condition"),
            index=pl.int_range(0, condition_expr.len()).cast(pl.UInt32),
        )
        .explode("condition", "index")
    )
    holes_to_fill = do_rle(df, group_by=group_by)
    holes_to_fill = holes_to_fill.filter(
        pl.col("len") <= hole_size, pl.col("value").not_(), pl.col("start") > 0
    )
    holes_to_fill = (
        explode_rle(holes_to_fill)
        .with_columns(condition=pl.lit(True))
        .drop("len", "start", "value")
    )
    df = df.join(holes_to_fill, on=[*group_by, "index"], how="left")
    df = df.with_columns(
        condition=pl.when(pl.col("condition_right").is_not_null())
        .then(pl.col("condition_right"))
        .otherwise(pl.col("condition"))
    ).drop("condition_right", "index")
    df = do_rle(df, group_by=group_by)
    
    if not unwrap and "year" not in group_by:
        return df.drop(*to_drop)
    
    if not unwrap and "year" in group_by:
        start_idx = orig_time.group_by(*group_by, maintain_order=True).len("start_idx")
        group_by.remove("year")
        if len(group_by) == 0:
            start_idx = start_idx.with_columns(pl.col("start_idx").cum_sum() - pl.col("start_idx").get(0))
        else:
            start_idx = start_idx.with_columns(
                start_idx.group_by(
                    group_by, maintain_order=True
                ).agg(
                    pl.col("start_idx").cum_sum() - pl.col("start_idx").get(0)
                )["start_idx"].explode()
            )
        df = df.join(start_idx, on=["year", *group_by])
        df = df.with_columns(start=pl.col("start") + pl.col("start_idx")).drop("year", "start_idx")
        return df
    
    df = df.filter("value")
    to_drop.extend(["len", "start", "value"])
    df = explode_rle(df)
    if "year" not in group_by:
        return df.drop(to_drop)
    orig_time = (
        orig_time
        .group_by("year")
        .agg(
            pl.col("time").dt.offset_by(f"{-dmonth}mo"), index=pl.int_range(0, pl.col("time").len()).cast(pl.UInt32)
        )
        .explode("time", "index")
    )
    return df.join(orig_time, on=["year", "index"]).sort(*group_by)


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


def compute(obj, progress_flag: bool = False, **kwargs):
    kwargs = COMPUTE_KWARGS | kwargs
    try:
        client  # in globals # type: ignore # noqa: F821
    except NameError:
        try:
            if progress_flag:
                with ProgressBar():
                    return obj.compute(**kwargs)
            else:
                return obj.compute(**kwargs)
        except AttributeError:
            return obj
    try:
        if progress_flag:
            obj = client.gather(client.persist(obj))  # type: ignore # noqa: F821
            progress(obj, notebook=False)
            return obj
        else:
            return client.compute(obj)  # type: ignore # noqa: F821
    except AttributeError:
        return obj


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
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

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
