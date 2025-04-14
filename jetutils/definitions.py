# coding: utf-8
"""
This file contains commonly used definitions of paths and compute options, gotten from the file `$HOME/.jetutils.ini` if it exists, otherwise guessed.

It also contains all the constants to do physics, the common timeranges, the full names of jet variables as well as their units, default values and LaTeX symbols.

Finally, it contains a few functions that are useful all over.
"""
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence
from itertools import groupby
from dataclasses import dataclass, field
import time
import datetime
import configparser
from importlib import resources as impresources

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from dask.diagnostics import ProgressBar  # to use without a specified dask client
from dask.distributed import progress  # to use with a specified dask client


if "DATADIR" not in globals():
    np.set_printoptions(precision=5, suppress=True)

    # Try to find a .jetutils.ini
    config = configparser.ConfigParser()
    path_default_config = impresources.files("jetutils").joinpath("config.ini")
    path_override_config = Path.home().joinpath(".jetutils.ini")
    config.read([path_default_config, path_override_config])

    if path_override_config.is_file():
        print("Found config override file at ", path_override_config)
    else:
        print("No config override found at ", path_override_config, "Guessing everything")

    # For what's not found, guesses and prints the guesses
    DATADIR = config.get("PATHS", "DATADIR")
    FIGURES = config.get("PATHS", "FIGURES")
    RESULTS = config.get("PATHS", "RESULTS")
    N_WORKERS = config.get("COMPUTE", "N_WORKERS")
    MEMORY_LIMIT = config.get("COMPUTE", "MEMORY_LIMIT")

    if DATADIR == "guess":
        DATADIR = Path.cwd().joinpath("data")
        print("Guessed DATADIR : ", DATADIR)
    if FIGURES == "guess":
        FIGURES = Path.cwd().joinpath("figures")
        print("Guessed FIGURES : ", FIGURES)
    if RESULTS == "guess":
        RESULTS = Path.cwd().joinpath("results")
        print("Guessed RESULTS : ", RESULTS)
    if N_WORKERS == "guess":
        if "SLURM_NTASKS" not in os.environ and "SLURM_CPUS_ON_NODE" not in os.environ:
            N_WORKERS = os.cpu_count()
        else:
            guess_1 = os.environ.get("SLURM_NTASKS", 1)
            guess_2 = os.environ.get("SLURM_CPUS_ON_NODE", 1)
            N_WORKERS = max(guess_1, guess_2)
        print("Guessed N_WORKERS : ", N_WORKERS)
    N_WORKERS = int(N_WORKERS)
    if MEMORY_LIMIT == "guess":
        MEMORY_LIMIT = os.environ.get("SLURM_MEM_PER_NODE", "8000")
        print("Guessed MEMORY_LIMIT : ", MEMORY_LIMIT)
    MEMORY_LIMIT = int(MEMORY_LIMIT) // N_WORKERS
    MEMORY_LIMIT = f"{MEMORY_LIMIT / 1000}GiB"

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
    }

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
        "mean_theta": 300,
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
        "is_polar": True,
        "n_jets": 0,
        "flag": 0,
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
    """
    Cosine of an angle expressed in degrees

    Parameters
    ----------
    x : float
        Angle in degrees

    Returns
    -------
    float
        Cosine result
    """
    return np.cos(x / 180 * np.pi)


def degsin(x: float) -> float:
    """
    Sine of an angle expressed in degrees

    Parameters
    ----------
    x : float
        Angle in degrees

    Returns
    -------
    float
        Sine results
    """
    return np.sin(x / 180 * np.pi)


def save_pickle(to_save: Any, filename: str | Path) -> None:
    """
    Save a pickleable object to file

    Parameters
    ----------
    to_save : Any
        Pickleable
    filename : str | Path
        path, it's better if it ends in `.pkl`
    """
    with open(filename, "wb") as handle:
        pkl.dump(to_save, handle)


def load_pickle(filename: str | Path) -> Any:
    """
    Save a pickleable object to file

    Parameters
    ----------
    filename : str | Path
        path, it's better if it ends in `.pkl`
        
    Returns
    -------
    Any
        Pickled object
    """
    with open(filename, "rb") as handle:
        to_ret = pkl.load(handle)
    return to_ret


def to_zero_one(X: np.ndarray | pl.DataFrame):
    """
    Normalizes an arbitrary polars DataFrame or numpy Array to the range [0, 1] along one axis. The 0 axis if numpy, the columns if polars. Returns the original minimum and maximum to be able to revert.

    Parameters
    ----------
    X : np.ndarray | pl.DataFrame
        Input array

    Returns
    -------
    X : same as input
        Input normalised to the range [0, 1]
    
    Xmin : same as input, with one fewer dimension
        Original minimum of the data, used to revert this function
        
    Xmax : same as input, with one fewer dimension
        Original maximum of the data, used to revert this function
    """    
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
    """
    Reverts the function to_zero_one().
    """
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
    """
    Normalizes an arbitrary polars DataFrame or numpy Array to a standard normal along one axis. The 0 axis if numpy, the columns if polars. Returns the original minimum and maximum to be able to revert.

    Parameters
    ----------
    X : np.ndarray | pl.DataFrame
        Input array

    Returns
    -------
    X : same as input
        Input normalised to a standard normal
    
    meanX : same as input, with one fewer dimension
        Original minimum of the data, used to revert this function
        
    stdX : same as input, with one fewer dimension
        Original maximum of the data, used to revert this function
    """    
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
    """
    Reverts the function normalize().
    """
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
    """
    Turns a xarray Dataset or DataArray into a polars DataFrame.
    """
    if "time" in da.dims and da["time"].dtype == np.dtype("object"):
        da["time"] = da.indexes["time"].to_datetimeindex(time_unit="us")
    df = da.to_dataframe().reset_index(allow_duplicates=True)
    df = df.loc[:, ~df.columns.duplicated()] # weird but easiest to handle multiindex unwrapping
    return pl.from_pandas(df)


def polars_to_xarray(df: pl.DataFrame, index_columns: Sequence[str]):
    """
    Turns a polars DataFrame into a xarray DataArray if possible, a Dataset otherwise. Which columns of `df` will be dimensions of the xarray output cannot be inferred from `df` and have to be passed as `index_columns`.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input array
        
    index_columns : list[str]
        Which columns of `df` to use as dimensions for the xarray object
        
    Returns
    -------
    da : xr.DataArray or xr.Dataset
        Data transformed in to a xarray object. If `df` had only index columns and one other column (inferred to be the data), `da` will be turned into a DataArray. If there are several other columns, then it stays a Dataset.
    """
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
        "sample_index",
        "inside_index",
    ),
):
    """
    Finds columns in `df` that represent an index imformation more than a data information in the context of this package.

    Parameters
    ----------
    df : pl.DataFrame
        Any DataFrame
    potentials : tuple, optional
        Potential names of column indices, by default ( "member", "time", "cluster", "jet ID", "spell", "relative_index", "relative_time", "sample_index", "inside_index", )

    Returns
    -------
    list
        list of columns in `potentials` that are columns in `df`.
    """
    index_columns = [ic for ic in potentials if ic in df.columns]
    return index_columns


def extract_season_from_df(
    df: pl.DataFrame,
    season: list | str | tuple | int | None = None,
) -> pl.DataFrame:
    """
    Subsets a DataFrame containing a `"time"` column to a given season.
    """
    if season is None:
        return df
    if isinstance(season, str):
        season = SEASONS[season]
    if isinstance(season, int):
        season = [season]
    return df.filter(pl.col("time").dt.month().is_in(season))


def case_insensitive_equal(str1: str, str2: str) -> bool:
    """
    Returns whether two strings are equal if all letters are lowercased.
    
    Examples
    --------
    >>> case_insensitive_equal("AbC", "aBc")
    True
    """
    return str1.casefold() == str2.casefold()


def infer_direction(to_plot: Any) -> int:
    """
    Infers the direction of an arbitrary array.

    Parameters
    ----------
    to_plot : Any
        Array or list of arrays

    Returns
    -------
    int
        -1 if the data is mostly negative, +1 if it is mostly positive and 0 if the data is symmetric
    """
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


def labels_to_mask(labels: xr.DataArray | np.ndarray, as_da: bool = False) -> np.ndarray:
    """
    Turns an array of labels into a mask

    Parameters
    ----------
    labels : xr.DataArray | np.ndarray
        Array of labels.
    as_da : bool, optional
        If `labels` is a DataArray and `as_da` is True, then turns the output into a DataArray, by default False

    Returns
    -------
    xr.DataArray | np.ndarray of shape (*labels.shape, n_unique_labels)
        Boolean mask, of the same shape as labels plus one dimension / axis at position 0. If turned into a DataArray, that new dimension is named "cluster".
        
    Examples
    --------
    >>> labels_to_mask([1, 3, 2, 1])
    array([[True, False, False],
       [False, False, True],
       [False, True, False],
       [True, False, False]])
    """
    if isinstance(labels, np.ndarray):
        as_da = False
    else:
        coords = labels.coords.copy()
        labels = labels.values
    unique_labels = np.unique(labels)
    mask = labels[..., None] == unique_labels[None, :]
    if not as_da:
        return mask
    coords = coords.assign({"cluster": unique_labels})
    mask = xr.DataArray(mask, coords=coords)
    return mask


def get_region(da: xr.DataArray | xr.Dataset) -> tuple:
    """
    Extracts the lon-lat region spanned by an xarray object containing the `"lon"` and `"lat"` dimensions.

    Parameters
    ----------
    da : xr.DataArray | xr.Dataset
        Xarray object

    Returns
    -------
    minlon: float
        minimum longitude
    
    maxlon: float
        maximum longitude
        
    minlat: float
        minimum latitude
        
    maxlat: float
        maximum latitude
    """
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
    """
    Gets a *(N - n + 1)* dimensional slice from a *N* dimensional Xarray object using Xarray's advanced indexing, by passing *n* indexers in a dict.

    Parameters
    ----------
    da : xr.DataArray | xr.Dataset
        Xarray object
    indexers : dict
        Dictionnary whose keys must be dimensions of `da` and values are arrays of values along this dimension, and of the correct `dtype`. Each array must be of the same length. Indexers is to be interpreted as the coordinates of points onto which we wish to interpolate `da`. 
    dim : str, optional
        Name of the newly created dimension in the output, that will be of the same length as all of the (equally sized) arrays in `indexers`. By default "points".

    Returns
    -------
    da_slice : same as `da`
        Input DataArray interpolated on the points specified by `indexers`. It retains all the dimension that are in `da` but not as keys of `indexers`. It has lost all the dimensions named in `indexers` and gained a new dimension named `dim` and of the same length as all the arrays in `indexers`.
        
    References
    ----------
    https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
    """
    return da.interp(
        {key: xr.DataArray(indexer, dims=dim) for key, indexer in indexers.items()},
        method="linear",
        kwargs=dict(fill_value=None),
    )


def first_elements(arr: np.ndarray, n_elements: int, sort: bool = False) -> np.ndarray:
    """
    Get the smallest `n_elements` of `arr`, along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        Any array
    n_elements : int
        Number of elements to return along each axis
    sort : bool, optional
        Sort the output, only valid for 1D `arr`, by default False

    Returns
    -------
    np.ndarray

    Raises
    ------
    RuntimeWarning
        If `sort=True` and `arr.ndim > 1` because it's ambiguous.
    """
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
    """
    Get the largest `n_elements` of `arr`, along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        Any array
    n_elements : int
        Number of elements to return along each axis
    sort : bool, optional
        Sort the output, only valid for 1D `arr`, by default False

    Returns
    -------
    np.ndarray

    Raises
    ------
    RuntimeWarning
        If `sort=True` and `arr.ndim > 1` because it's ambiguous.
    """
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
    """
    Thin wrapper around `da.coarsen()` to coarsen as close as possible to a target *dx*
    """
    dx = (da.lon[1] - da.lon[0]).item()
    coarsening = int(np.round(target_dx / dx))
    return da.coarsen({"lon": coarsening, "lat": coarsening}, boundary="trim").mean()


def _explode_rle(df):
    return df.with_columns(
        index=(pl.int_ranges(pl.col("start"), pl.col("start") + pl.col("len"))).cast(
            pl.List(pl.UInt32)
        )
    ).explode("index")


def _do_rle(
    df: pl.DataFrame, group_by: Sequence[str] | Sequence[pl.Expr] | str | pl.Expr
) -> pl.DataFrame:
    if isinstance(group_by, str | pl.Expr):
        group_by = [group_by]
    conditional = (
        df.group_by(group_by, maintain_order=True)
        .agg(
            pl.col("condition").rle().alias("rle"),
        )
        .explode("rle")
        .unnest("rle")
        .group_by(group_by, maintain_order=True)
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
    group_by: Sequence[str] | Sequence[pl.Expr] | str | pl.Expr | None = None,
    hole_size: int | datetime.timedelta = 4,
    unwrap: bool = False,
) -> pl.DataFrame:
    """
    Wraps around polars' `pl.Expr.rle()` to find runs of identical values, potentially interrupted by a different value, as long as this interruption is shorter than `hole_size`. 
    
    It can do it for the whose DataFrame or in groups specified by `group_by`.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    condition_expr : pl.Expr
        Expression that evaluates to True or False from one or several columns of `df`
    group_by : Sequence[str] | Sequence[pl.Expr] | str | pl.Expr, optional
        Columns to group by, by default None
    hole_size : int | datetime.timedelta, optional
        Maximum authorized size of holes than can be in a run without interrupting it, by default 4
    unwrap : bool, optional
        If False, returns the whole data as a modified run length encoded DataFrame. If True, returns the True runs exploded. By default False

    Returns
    -------
    pl.DataFrame
        Modified-run-length-encoded input, or exploded True runs. 

    Raises
    ------
    ValueError
        If `hole_size` is specified as a `datetime.timedelta` but there is no `"time"`, or `"time"` is in `group_by`.
    """
    if isinstance(group_by, str | pl.Expr):
        group_by = [group_by]
    if isinstance(hole_size, datetime.timedelta):
        if "time" not in df.columns or (group_by is not None and "time" in group_by):
            raise ValueError
        times = df["time"].unique().bottom_k(2).sort()
        dt = times[1] - times[0]
        hole_size = int(hole_size / dt)  # I am not responsible for rounding errors
    to_drop = []
    if not group_by:
        if "contour" in df.columns:
            group_by = get_index_columns(
                df, ("member", "time", "cluster", "contour", "spell", "relative_index", "relative_time")
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

    if not group_by:
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
    holes_to_fill = _do_rle(df, group_by=group_by)
    holes_to_fill = holes_to_fill.filter(
        pl.col("len") <= hole_size, pl.col("value").not_(), pl.col("start") > 0
    )
    holes_to_fill = (
        _explode_rle(holes_to_fill)
        .with_columns(condition=pl.lit(True))
        .drop("len", "start", "value")
    )
    df = df.join(holes_to_fill, on=[*group_by, "index"], how="left")
    df = df.with_columns(
        condition=pl.when(pl.col("condition_right").is_not_null())
        .then(pl.col("condition_right"))
        .otherwise(pl.col("condition"))
    ).drop("condition_right", "index")
    df = _do_rle(df, group_by=group_by)

    if not unwrap and "year" not in group_by:
        return df.drop(*to_drop)

    if not unwrap and "year" in group_by:
        start_idx = orig_time.group_by(*group_by, maintain_order=True).len("start_idx")
        group_by.remove("year")
        if len(group_by) == 0:
            start_idx = start_idx.with_columns(
                pl.col("start_idx").cum_sum() - pl.col("start_idx").get(0)
            )
        else:
            start_idx = start_idx.with_columns(
                start_idx.group_by(group_by, maintain_order=True)
                .agg(pl.col("start_idx").cum_sum() - pl.col("start_idx").get(0))[
                    "start_idx"
                ]
                .explode()
            )
        df = df.join(start_idx, on=["year", *group_by])
        df = df.with_columns(start=pl.col("start") + pl.col("start_idx")).drop(
            "year", "start_idx"
        )
        return df

    df = df.filter("value")
    to_drop.extend(["len", "start", "value"])
    df = _explode_rle(df)
    if "year" not in group_by:
        return df.drop(to_drop)
    orig_time = (
        orig_time.group_by("year")
        .agg(
            pl.col("time").dt.offset_by(f"{-dmonth}mo"),
            index=pl.int_range(0, pl.col("time").len()).cast(pl.UInt32),
        )
        .explode("time", "index")
    )
    return df.join(orig_time, on=["year", "index"]).sort(*group_by)


# Obsolete
def get_runs(mask, cyclic: bool = True): 
    """
    Obsolete basic implementaion of the Run Length Encoding algorithm using `itertools.groupby`. 
    
    With the `cyclic` argument on, runs are allowed to wrap around the end of the list to its start. For instance, list `[True, True, False, ..., False, True, True]` will have a `True` run going from indices `-2` to `1` included if `cyclic=True`.
    """
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


# Obsolete
def get_runs_fill_holes(mask, cyclic: bool = True, hole_size: int = 8):
    """
    Obsolete algorithm to get potentially interrupted runs of `True` values. The runs can be uninterrupted like the basic algorithm, or interrupted by `False` values if the run of `False` values within the run of `True` values is shorter than `hole_size`.
    
    The algorithm first performs RLE using `get_runs`, then fills the short `False` runs with `True` and applies `get_runs` a second time on the modified input.
    """
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
    """
    Computes a Dask object. If a dask client named `client` exists in the globals, uses it.

    Parameters
    ----------
    obj : Any
        Dask object to force compute
    progress_flag : bool, optional
        Whether to show a progress bar, by default False
    kwargs
        Keyword arguments passed to `obj.compute()` if no client exists

    Returns
    -------
    obj : Any
        Computed object. 
    """
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
    """
    This is stolen from a gist somewhere I don't remember. Nice context manager timer.
    
    Raises
    ------
    TimerError
    
    Examples
    --------
    >>> with Timer():
    ...    do_something_long()
    "elapsed time: 5.3s"
    ```
    """
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
