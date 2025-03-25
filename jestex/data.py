# coding: utf-8
import warnings
from os.path import commonpath

from typing import Optional, Mapping, Sequence, Tuple, Literal
from itertools import product
from pathlib import Path

import cf_xarray
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from tqdm import tqdm

from .definitions import (
    TIMERANGE,
    DEFAULT_VARNAME,
    DATADIR,
    YEARS,
    SEASONS,
    get_region,
    save_pickle,
    load_pickle,
    compute,
    get_index_columns,
)


def to_netcdf(da: xr.Dataset | xr.DataArray, path: Path | str, **kwargs):
    """
    Wrapper around `da.to_netcdf()` that handles multiindex.

    :param da: Object to sace
    :type da: xr.Dataset | xr.DataArray
    :param path: Path
    :type path: Path | str
    """
    try:
        da.to_netcdf(path, **kwargs)
    except (NotImplementedError, ValueError) as e:
        try:
            import cf_xarray
        except ModuleNotFoundError:
            raise e
        if isinstance(da, xr.DataArray):
            da = da.to_dataset()
        da = cf_xarray.encode_multi_index_as_compress(da)
        da.to_netcdf(path, **kwargs)


def open_dataset(path: Path | str, **kwargs) -> xr.Dataset:
    """
    Wrapper around `xr.open_dataset` that handles multiindex

    :param path: Path to file
    :type path: Path | str
    :return: Dataset
    :rtype: xr.Dataset
    """
    ds = xr.open_dataset(path, **kwargs)
    try:
        import cf_xarray
    except ModuleNotFoundError:
        return ds
    try:
        return cf_xarray.decode_compress_to_multi_index(ds)
    except ValueError:
        return ds


def open_dataarray(path: Path | str, **kwargs) -> xr.DataArray:
    """
    Wrapper around xr.open_dataarray to handle multiindex.

    :param path: Path to file
    :type path: Path | str
    :return: Dataset at file
    :rtype: xr.DataArray
    """
    ds = open_dataset(path, **kwargs)
    if len(ds.data_vars) != 1:
        raise ValueError("More than one data var in ds!")
    (da,) = ds.data_vars.values()
    return da


def _open_many_da_wrapper(
    filename: Path | list[Path], varname: str | None = None
) -> xr.DataArray | xr.Dataset:
    """
    Reimplementation of of `xr.open_mfdataset` with a handlful of special rules. Tries to extract a DataArray if possible, guessing the variable name if `varname=None`. This function calls `standardize()`

    :param filename: Path or list of Paths to file(s)
    :type filename: Path | list[Path]
    :param varname: Gives the function the correct varname instead of letting it guess, defaults to None
    :type varname: str | None, optional
    :return: Dataarray if possible, dataset if several data variables are found.
    :rtype: xr.DataArray | xr.Dataset
    """
    if isinstance(filename, list) and len(filename) == 1:
        filename = filename[0]
    if isinstance(filename, list):
        da = []
        for fn in filename:
            stem = Path(fn).stem
            try:
                int(stem)
            except ValueError:  
                # exceptions are raised by clim.nc, which we do not want anyways
                continue
            da_ = open_dataset(fn, chunks="auto")
            if len(stem) == 6:
                yearmask = da_.time.dt.year.values == int(stem[:4])
                monthmask = da_.time.dt.month.values == int(stem[4:])
                da_ = da_.sel(time=yearmask & monthmask)
            else:
                yearmask = da_.time.dt.year.values == int(stem)
                da_ = da_.sel(time=yearmask)
            da.append(da_)
        da = xr.concat(da, dim="time")
        da = da.unify_chunks()
    else:
        da = open_dataset(filename, chunks="auto")
        da = da.unify_chunks()
    da = standardize(da)
    if varname is None or len(da.data_vars) > 1:
        return da
    for potential in [
        varname,
        varname.split("_")[0],
        "dummy",
        DEFAULT_VARNAME,
        list(da.data_vars)[0],
    ]:
        try:
            da = da[potential].rename(varname)
            break
        except KeyError:
            pass
    return da


def get_land_mask() -> xr.DataArray:
    """
    Gets the land mask if it's at the standard location

    :return: Land mask, a 2d dataarray at 0.5 degreees resolution, gridded like the standardized data.
    :rtype: xr.DataArray
    """
    mask = open_dataarray(f"{DATADIR}/ERA5/grid_info/land_sea.nc")
    mask = (
        mask.squeeze()
        .rename(longitude="lon", latitude="lat")
        .reset_coords("time", drop=True)
    )
    return mask.astype(bool)


def determine_file_structure(path: Path) -> str:
    """
    Determines if files in the folder pointed by `path` are monthly, yearly, or all-in-one.

    :param path: Path to a folder containing data
    :type path: Path
    :raises RuntimeError: _description_
    :return: One of "one_file", "yearly" or "monthly" depending on the files found in `path`.
    :rtype: str
    """
    if path.joinpath("full.nc").is_file():
        return "one_file"
    if any([path.joinpath(f"{year}.nc").is_file() for year in YEARS]):
        return "yearly"
    if any([path.joinpath(f"{year}01.nc").is_file() for year in YEARS]):
        return "monthly"
    print("Could not determine file structure")
    raise RuntimeError


def determine_sample_dims(da: xr.DataArray) -> Mapping:
    """
    Returns which dimensions, among "member", "time" or "megatime", are present in `da`, along with the coordinates as indices.

    :param da: Xarray object from which to extract the sample dimensions and coordinates
    :type da: xr.DataArray
    :return: Dictionary whose keys are dimension names and values are the corresponding indices.
    :rtype: Mapping
    """
    included = ["member", "time", "megatime"]
    return {key: da[key].to_index() for key in da.dims if key in included}


def determine_feature_dims(da: xr.DataArray) -> Mapping:
    """
    Returns all dimensions, except for "member", "time", "cluster" or "megatime", are present in `da`, along with the coordinates as indices.

    :param da: Xarray object from which to extract the feature dimensions and coordinates
    :type da: xr.DataArray
    :return: Dictionary whose keys are dimension names and values are the corresponding indices.
    :rtype: Mapping
    """
    excluded = ["member", "time", "cluster", "megatime"]
    return {key: da[key].to_index() for key in da.dims if key not in excluded}


def data_path(
    dataset: str,
    level_type: Literal["plev"]
    | Literal["thetalev"]
    | Literal["surf"]
    | Literal["2PVU"]
    | None = None,
    varname: str | None = None,
    resolution: str | None = None,
    clim_type: str | None = None,
    clim_smoothing: Mapping | None = None,
    smoothing: Mapping | None = None,
    for_compute_anomaly: bool = False,
) -> Path | Tuple[Path, Path, Path]:
    """
    Constructs a path from various, mostly optional, path elements. Note that the name of the path elements refer to the data structure used by the author, but only their order matter, since this function essentially does
    ```python
    return Path(DATADIR, dataset, level_type, varname, resolution, clim_type + unpack_smooth_map(clim_smoothing), smoothing)
    ```

    :param dataset: name of the dataset, typically "ERA5" or "CESM2"
    :type dataset: str
    :param level_type: Level type, defaults to None
    :type level_type: Literal[&quot;plev&quot;] | Literal[&quot;thetalev&quot;] | Literal[&quot;surf&quot;] | Literal[&quot;2PVU&quot;] | None, optional
    :param varname: Name of the variable, or name of the group of variables like "high_wind", defaults to None
    :type varname: str | None, optional
    :param resolution: Time resolution, typically "6H" or "dailymean", defaults to None
    :type resolution: str | None, optional
    :param clim_type: type of climatology, like "dayofyear", defaults to None
    :type clim_type: str | None, optional
    :param clim_smoothing: time-smoothing of the climatology as a 1-key mapping whose key if the `clim_type` and the value is a tuple (smoothing_type, window_size), see `smooth()`, defaults to None
    :type clim_smoothing: Mapping | None, optional
    :param smoothing: smoothing of the anomalies as a mapping whose keys are dimension names and values are tuples (smoothing_type, window_size), see `smooth()`, defaults to None
    :type smoothing: Mapping | None, optional
    :param for_compute_anomaly: is this function called by `compute_anomaly` or not, defaults to False
    :type for_compute_anomaly: bool, optional
    :raises TypeError: _description_
    :raises FileNotFoundError: _description_
    :return: path to the folder containing the data fitting the description
    :rtype: Path | Tuple[Path, Path, Path]
    """
    if clim_type is None and for_compute_anomaly:
        clim_type = "none"
    elif clim_type is None:
        clim_type = ""

    if clim_smoothing is None:
        clim_smoothing = {}

    if smoothing is None:
        smoothing = {}

    if clim_type == "" and len(clim_smoothing) != 0:
        print("Cannot define clim_smoothing if clim is None")
        raise TypeError

    elements = (DATADIR, dataset, level_type, varname, resolution)
    path = Path(*[element for element in elements if element is not None])

    unpacked = unpack_smooth_map(clim_smoothing)
    underscore = "_" if unpacked != "" else ""

    clim_path = path.joinpath(clim_type + underscore + unpacked)
    anom_path = clim_path.joinpath(unpack_smooth_map(smoothing))
    if not anom_path.is_dir() and not for_compute_anomaly:
        if not clim_type == "":
            print(
                "Folder does not exist. Try running compute_all_smoothed_anomalies before"
            )
            raise FileNotFoundError
        anom_path = clim_path
    elif for_compute_anomaly:
        anom_path.mkdir(exist_ok=True, parents=True)
    if for_compute_anomaly:
        return path, clim_path, anom_path
    return anom_path


def standardize(da):
    """
    Applies a bunch of different rules to standardize Xarray objects. Names of variables, dimensions, coordinates and indices are modified as specified by the `standard_dict` defined at the start of the function. The longitudes are forced to go from -180 to +180 - dx, the latitudes are forced to be in increasing order. Finally, the data is coerced into a dask array.

    :param da: Xarray object to standardize
    :type da: xr.DataArray or xr.Dataset
    :return: standardized Xarray object
    :rtype: same as input
    """
    standard_dict = {
        "valid_time": "time",
        "time_counter": "time",
        "longitude": "lon",
        "lon_um_atmos_grid_uv": "lon",
        "latitude": "lat",
        "lat_um_atmos_grid_uv": "lat",
        "level": "lev",
        "plev": "lev",
        "pres": "lev",
        "pressure_level": "lev",
        "um_atmos_PLEV19": "lev",
        "member_id": "member",
        "U": "u",
        "U500": "u",
        "u_component_of_wind": "u",
        "m01s30i201_3": "u",
        "V": "v",
        "V500": "v",
        "v_component_of_wind": "v",
        "m01s30i202_3": "v",
        "T": "t",
        "t2m": "t",
        "2m_temperature": "t",
        "m01s30i204_3": "t",
        "pt": "theta",
        "PRECL": "tp",
        "total_precipitation": "tp",
        "Z3": "z",
    }
    if isinstance(da, xr.Dataset):
        for key, value in standard_dict.items():
            if key in da:
                da = da.rename({key: value})
            else:
                pass
    elif isinstance(da, xr.DataArray):
        for key, value in standard_dict.items():
            if key in da.coords:
                da = da.rename({key: value})
            else:
                pass
    for to_del in ["number", "expver"]:
        try:
            da = da.reset_coords(to_del, drop=True)
        except ValueError:
            pass
    if "time" in da.dims:
        inityear = da["time"].dt.year.values[0]
        if inityear < 1800:
            new_time_range = pd.date_range(
                f"{inityear + 1968}0101",
                end=None,
                freq="6h",
                inclusive="left",
                periods=da.time.shape[0],
            )
            da["time"] = new_time_range
    if (da.lon.max() > 180) and (da.lon.min() >= 0):
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby("lon")
    if np.diff(da.lat.values)[0] < 0:
        da = da.reindex(lat=da.lat[::-1])
    if isinstance(da, xr.Dataset):
        for var in da.data_vars:
            if "chunksizes" in da[var].encoding and da[var].chunks is None:
                chunks = da[var].encoding["chunksizes"]
                chunks = chunks if chunks is not None else "auto"
                da[var] = da[var].chunk(chunks)
            if da[var].dtype == np.float64:
                da[var] = da[var].astype(np.float32)
            elif da[var].dtype == np.int64:
                da[var] = da[var].astype(np.int32)
    else:
        if "chunksizes" in da.encoding and da.chunks is None:
            chunks = da.encoding["chunksizes"]
            chunks = chunks if chunks is not None else "auto"
            da = da.chunk(chunks)
        if da.dtype == np.float64:
            da = da.astype(np.float32)
        elif da.dtype == np.int64:
            da = da.astype(np.int32)
    return da.unify_chunks()


def extract_period(
    da: xr.DataArray | xr.Dataset,
    period: list | tuple | Literal["all"] | int | str = "all",
):
    """
    Extracts a period, specified by a list of years or a tuple or year bounds, or "all".

    :param da: Xarray object that has a "time" dimension from which to extract a period
    :type da: xr.DataArray or xr.Dataset
    :param period: period specified, expressed as list of years, the string "all" or a 2-tuple specifying the first and last year (both included) to extract, defaults to "all"
    :type period: list | tuple | Literal[&quot;all&quot;] | int | str, optional
    :return da:
    :rtype:
    """
    if period == "all":
        return da
    if isinstance(period, tuple):
        if len(period) == 2:
            period = np.arange(period[0], period[1] + 1).tolist()
        elif len(period) == 3:
            period = np.arange(period[0], period[1] + 1, period[2]).tolist()
    elif isinstance(period, int):
        period = [period]
    return da.isel(time=np.isin(da.time.dt.year, period))


def unpack_levels(levels: int | str | tuple | list) -> Tuple[list, list]:
    """
    uh

    :param levels: _description_
    :type levels: int | str | tuple | list
    :return: _description_
    :rtype: Tuple[list, list]
    """
    if isinstance(levels, int | str | tuple | np.int64 | np.int32):
        levels = [levels]
    to_sort = []
    for level in levels:
        to_sort.append(
            float(level)
            if isinstance(level, int | str | np.int64 | np.int32)
            else level[0]
        )
    levels = [levels[i] for i in np.argsort(to_sort)]
    level_names = []
    for level in levels:
        if isinstance(level, tuple | list):
            level_names.append(f"{level[0]}-{len(level)}-{level[-1]}")
        else:
            level_names.append(str(level))
    return levels, level_names


def extract_levels(da: xr.DataArray, levels: int | str | list | tuple | Literal["all"]):
    if levels == "all":
        if da.lev.values.size == 1:
            try:
                return da.squeeze().reset_index("lev").reset_coords("lev", drop=True)
            except ValueError:
                return da.squeeze().reset_coords("lev", drop=True)
        return da.squeeze()

    levels, level_names = unpack_levels(levels)
    da.attrs["orig_lev"] = levels
    if not any([isinstance(level, tuple) for level in levels]):
        try:
            da = da.isel(lev=levels)
        except (ValueError, IndexError):
            da = da.sel(lev=levels)
        if len(levels) == 1 or da.lev.values.size <= 1:
            return da.squeeze().reset_coords("lev", drop=True)
        return da

    newcoords = {dim: da.coords[dim] for dim in ["time", "lat", "lon"]}
    if "lev" in da.coords:
        newcoords = newcoords | {"lev": level_names}
    shape = [len(coord) for coord in newcoords.values()]
    da2 = xr.DataArray(np.zeros(shape), coords=newcoords)
    for level, level_name in zip(levels, level_names):
        if isinstance(level, tuple):
            val = da.sel(lev=level).mean(dim="lev").values
        else:
            val = da.sel(lev=level).values
        da2.loc[:, :, :, level_name] = val

    return da2.squeeze()


def extract_season(
    da: xr.DataArray | xr.Dataset, season: list | str | tuple
) -> xr.DataArray | xr.Dataset:
    if isinstance(season, list):
        da = da.isel(time=np.isin(da.time.dt.month, season))
    elif isinstance(season, str):
        if season in ["DJF", "MAM", "JJA", "SON"]:
            da = da.isel(time=da.time.dt.season == season)
        else:
            print(f"Wrong season specifier : {season} is not a valid xarray season")
            raise ValueError
    elif isinstance(season, tuple):
        if season[1] > season[0]:
            time_mask = (da.time.dt.dayofyear >= season[0]) & (
                da.time.dt.dayofyear <= season[1]
            )
        else:  # over winter
            time_mask = (da.time.dt.dayofyear >= season[1]) | (
                da.time.dt.dayofyear >= season[1]
            )
        da = da.sel(time=time_mask)
    return da


def extract_region(
    da: xr.DataArray | xr.Dataset,
    minlon: Optional[int | float] = None,
    maxlon: Optional[int | float] = None,
    minlat: Optional[int | float] = None,
    maxlat: Optional[int | float] = None,
) -> xr.DataArray | xr.Dataset:
    if all([bound is not None for bound in [minlon, maxlon, minlat, maxlat]]):
        da = da.sel(lon=slice(minlon, maxlon), lat=slice(minlat, maxlat))
    return da


def extract(
    da: xr.DataArray | xr.Dataset,
    period: list | tuple | Literal["all"] | int | str = "all",
    season: list | str | tuple | None = None,
    minlon: Optional[int | float] = None,
    maxlon: Optional[int | float] = None,
    minlat: Optional[int | float] = None,
    maxlat: Optional[int | float] = None,
    levels: int | str | list | tuple | Literal["all"] = "all",
    members: str | list | Literal["all"] = "all",
) -> xr.DataArray | xr.Dataset:
    """
    Applies all the extract_something functions.

    :param da: _description_
    :type da: xr.DataArray | xr.Dataset
    :param period: _description_, defaults to "all"
    :type period: list | tuple | Literal[&quot;all&quot;] | int | str, optional
    :param season: _description_, defaults to None
    :type season: list | str | tuple | None, optional
    :param minlon: _description_, defaults to None
    :type minlon: Optional[int  |  float], optional
    :param maxlon: _description_, defaults to None
    :type maxlon: Optional[int  |  float], optional
    :param minlat: _description_, defaults to None
    :type minlat: Optional[int  |  float], optional
    :param maxlat: _description_, defaults to None
    :type maxlat: Optional[int  |  float], optional
    :param levels: _description_, defaults to "all"
    :type levels: int | str | list | tuple | Literal[&quot;all&quot;], optional
    :param members: _description_, defaults to "all"
    :type members: str | list | Literal[&quot;all&quot;], optional
    :return: _description_
    :rtype: xr.DataArray | xr.Dataset
    """
    da = standardize(da)

    da = extract_period(da, period)

    if season is not None:
        da = extract_season(da, season)

    if "member" in da.dims and members != "all":
        try:
            da = da.isel(member=members)
        except (ValueError, TypeError):
            da = da.sel(member=members)

    da = extract_region(da, minlon, maxlon, minlat, maxlat)

    if "lev" in da.dims or "lev" in da.coords:
        da = extract_levels(da, levels)

    return da


def determine_period(path: Path):
    """
    Determines the full list of years spanned by the data in a given data folder.

    :param path: Path to a folder containing many .nc files, whose names are of either YYYYMM.nc or YYYY.nc format.
    :type path: Path
    :return: list of years spanned by the data in the folder.
    :rtype: list
    """
    yearlist = []
    for f in path.glob("*.nc"):
        try:
            yearlist.append(int(f.stem[:4]))
        except ValueError:  
            # exceptions are raised by clim.nc, which we do not want anyways
            continue
    return np.unique(yearlist).tolist()


def open_da(
    dataset: str,
    level_type: Literal["plev"]
    | Literal["thetalev"]
    | Literal["surf"]
    | Literal["2PVU"]
    | None = None,
    varname: str | None = None,
    resolution: str | None = None,
    period: list | tuple | Literal["all"] | int | str = "all",
    season: list | str | tuple | None = None,
    minlon: Optional[int | float] = None,
    maxlon: Optional[int | float] = None,
    minlat: Optional[int | float] = None,
    maxlat: Optional[int | float] = None,
    levels: int | str | list | tuple | Literal["all"] = "all",
    clim_type: str | None = None,
    clim_smoothing: Mapping | None = None,
    smoothing: Mapping | None = None,
) -> xr.DataArray:
    """
    Applies `data_path()`, `_open_many_da_wrapper()`, and `extract()` one after the other.

    :param dataset: _description_
    :type dataset: str
    :param level_type: _description_, defaults to None
    :type level_type: Literal[&quot;plev&quot;] | Literal[&quot;thetalev&quot;] | Literal[&quot;surf&quot;] | Literal[&quot;2PVU&quot;] | None, optional
    :param varname: _description_, defaults to None
    :type varname: str | None, optional
    :param resolution: _description_, defaults to None
    :type resolution: str | None, optional
    :param period: _description_, defaults to "all"
    :type period: list | tuple | Literal[&quot;all&quot;] | int | str, optional
    :param season: _description_, defaults to None
    :type season: list | str | tuple | None, optional
    :param minlon: _description_, defaults to None
    :type minlon: Optional[int  |  float], optional
    :param maxlon: _description_, defaults to None
    :type maxlon: Optional[int  |  float], optional
    :param minlat: _description_, defaults to None
    :type minlat: Optional[int  |  float], optional
    :param maxlat: _description_, defaults to None
    :type maxlat: Optional[int  |  float], optional
    :param levels: _description_, defaults to "all"
    :type levels: int | str | list | tuple | Literal[&quot;all&quot;], optional
    :param clim_type: _description_, defaults to None
    :type clim_type: str | None, optional
    :param clim_smoothing: _description_, defaults to None
    :type clim_smoothing: Mapping | None, optional
    :param smoothing: _description_, defaults to None
    :type smoothing: Mapping | None, optional
    :return: _description_
    :rtype: xr.DataArray
    """
    if isinstance(varname, Tuple):
        varname_in_path, varname_in_file = varname
    else:
        varname_in_path, varname_in_file = varname, varname
    path = data_path(
        dataset,
        level_type,
        varname_in_path,
        resolution,
        clim_type,
        clim_smoothing,
        smoothing,
        False,
    )
    file_structure = determine_file_structure(path)

    if isinstance(period, tuple):
        period = np.arange(int(period[0]), int(period[1] + 1)).tolist()
    elif isinstance(period, list):
        period = np.asarray(period).astype(int).tolist()
    elif period == "all":
        period = determine_period(path)
    elif isinstance(period, int | str):
        period = [int(period)]

    files_to_load = []

    if file_structure == "one_file":
        files_to_load = [path.joinpath("full.nc")]
    elif file_structure == "yearly":
        files_to_load = [path.joinpath(f"{year}.nc") for year in period]
    elif file_structure == "monthly":
        if season is None:
            files_to_load = [
                path.joinpath(f"{year}{str(month).zfill(2)}.nc")
                for month in range(1, 13)
                for year in period
            ]
        else:
            if isinstance(season, str):
                monthlist = SEASONS[season]
            elif isinstance(season, list):
                monthlist = np.atleast_1d(season)
            elif isinstance(season, tuple):
                sample_tr = TIMERANGE[TIMERANGE.year == period[0]]
                monthlist = np.unique(
                    sample_tr[
                        np.isin(sample_tr.dayofyear, np.arange(season[0], season[1]))
                    ].month
                )
            files_to_load = [
                path.joinpath(f"{year}{str(month).zfill(2)}.nc")
                for month in monthlist
                for year in period
            ]

    files_to_load = [fn for fn in files_to_load if fn.is_file()]

    da = _open_many_da_wrapper(files_to_load, varname_in_file)
    da = extract(
        da,
        period=period,
        season=season,
        minlon=minlon,
        maxlon=maxlon,
        minlat=minlat,
        maxlat=maxlat,
        levels=levels,
    )
    return da


def unpack_smooth_map(smooth_map: Mapping | Sequence) -> str:
    """
    Creates a unique string out of a smoothing map, useful for path creation
    Example:
    ```python
    unpack_smooth_map({"dayofyear": ("win", 10)})
    >>> "doywin10"
    
    unpack_smooth_map({"time": ("win", 10), "lon": ("fft", 10)})
    >>> "timewin10_lonfft10"
    ```

    :param smooth_map: _description_
    :type smooth_map: Mapping | Sequence
    :return: _description_
    :rtype: str
    """
    strlist = []
    for dim, value in smooth_map.items():
        if dim == "detrended":
            if smooth_map["detrended"]:
                strlist.append("detrended")
            continue
        smooth_type, winsize = value
        if dim == "dayofyear":
            dim = "doy"
        if isinstance(winsize, float):
            winsize = f"{winsize:.2f}"
        elif isinstance(winsize, int):
            winsize = str(winsize)
        strlist.append("".join((dim, smooth_type, winsize)))
    return "_".join(strlist)


def pad_wrap(da: xr.DataArray, dim: str) -> bool:
    """
    Checks whether we need to wrap-pad the data before smoothing. This is the case if we deal with a periodic dimension like longitude (but only if -180 and 180 - dx are present) or any of the climatology time dimensions ("dayofyear", "hourofyear", "month" or "week")

    :param da: dataarray on which smoothing will be applied
    :type da: xr.DataArray
    :param dim: dimension on which the smoothing is applied
    :type dim: str
    :return: whether or not to wrap_pad
    :rtype: bool
    """
    resolution = da[dim][1] - da[dim][0]
    if dim in ["lon", "longitude"]:
        return 360 >= da[dim][-1] >= 360 - resolution and da[dim][0] == 0.0
    return dim in ["dayofyear", "hourofyear", "month", "week"]


def _window_smoothing(
    da: xr.DataArray, dim: str, winsize: int, center: bool = True
) -> xr.DataArray:
    """
    inner worker function for window smoothing. Either simply calls the xarray method or implements the complicated logic for "hourofyear" smoothing. 

    :param da: DataArray on which the smoothing is applied
    :type da: xr.DataArray
    :param dim: dimension along which to window-smooth
    :type dim: str
    :param winsize: length of the running window
    :type winsize: int
    :param center: whether the result of the window smoothing is at the center or at the start of the window, defaults to True
    :type center: bool
    :return: Input array smoothed along `dim`. 
    :rtype: xr.DataArray
    """
    if dim != "hourofyear":
        return da.rolling({dim: winsize}, center=True, min_periods=1).mean()
    groups = da.groupby(da.hourofyear % 24)
    to_concat = []
    winsize = winsize // len(groups)
    if "time" in da.dims:
        dim = "time"
    for group in groups.groups.values():
        to_concat.append(
            da.isel(**{dim: group})
            .rolling({dim: winsize // 4}, center=center, min_periods=1)
            .mean()
        )
    return xr.concat(to_concat, dim=dim).sortby(dim)


def window_smoothing(
    da: xr.DataArray, dim: str, winsize: int, center: bool = True
) -> xr.DataArray:
    """
    Outer function for window smoothing. Determines is wrap padding needs to be done, and if yes does and undoes it. In the middle, calls `_window_smoothing()`

    :param da: DataArray on which the smoothing is applied
    :type da: xr.DataArray
    :param dim: dimension along which to window-smooth
    :type dim: str
    :param winsize: length of the running window
    :type winsize: int
    :param center: whether the result of the window smoothing is at the center or at the start of the window, defaults to True
    :type center: bool
    :return: Input array smoothed along `dim`. 
    :rtype: xr.DataArray
    """
    dims = dim.split("+")
    for dim in dims:
        if pad_wrap(da, dim):
            halfwinsize = int(np.ceil(winsize / 2))
            da = da.pad({dim: halfwinsize}, mode="wrap")
            newda = _window_smoothing(da, dim, halfwinsize, center)
            newda = newda.isel({dim: slice(halfwinsize, -halfwinsize)})
        else:
            newda = _window_smoothing(da, dim, winsize, center)
    newda.attrs = da.attrs
    return newda


def fft_smoothing(da: xr.DataArray, dim: str, winsize: int) -> xr.DataArray:
    """
    Probably broken for now fft_smoothing. FFT means Fast Fourier Transform, which is the central function we use to perforn this smoothing, whose more correct name would be a low-pass filter. Transforms the data along `dim` in the frequency domain, zeroes out the elements corresponding to the `winsize` highest frequencies, and transforms this back into real space.

    :param da: DataArray to smooth along one or several dimensions
    :type da: xr.DataArray
    :param dim: Dim name or a string constructed like "{dim1}+{dim2}" for 2D fft smoothing.
    :type dim: str
    :param winsize: Number of the highest frequencies to zero out
    :type winsize: int
    :return: Input DataArray smoothed along `dim`.
    :rtype: xr.DataArray
    """
    import xrft

    name = da.name
    dim = dim.split("+")
    extra_dims = [coord for coord in da.coords if coord not in da.dims]
    extra_coords = []
    for extra_dim in extra_dims:
        extra_coords.append(da[extra_dim])
        da = da.reset_coords(extra_dim, drop=True)
    da = da.where(~da.isnull(), 0)
    ft = xrft.fft(da, dim=dim)
    mask = 0
    for dim_ in dim:
        mask = mask + np.abs(ft[f"freq_{dim_}"])
    mask = mask < winsize
    ft = ft.where(mask, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        newda = (
            xrft.ifft(ft, dim=[f"freq_{dim_}" for dim_ in dim])
            .real.assign_coords(da.coords)
            .rename(name)
        )
    newda.attrs = da.attrs
    for i, extra_dim in enumerate(extra_dims):
        newda.assign_coords({extra_dim: extra_coords[i]})
    return newda


def spharm_smoothing(da: xr.DataArray, trunc: int): 
    """
    One day i'll implement this. This is meant to be another type of spectral smoothing like fft, but with spherical harmonics instead.

    :param da: _description_
    :type da: xr.DataArray
    :param trunc: _description_
    :type trunc: int
    :raises NotImplementedError: _description_
    """
    raise NotImplementedError("Spharmt is broken for now")
    # spharmt = Spharmt(len(da.lon), len(da.lat))
    # fieldspec = spharmt.grdtospec(da.values, ntrunc=trunc)
    # fieldtrunc = spharmt.spectogrd(fieldspec)
    # return fieldtrunc


def smooth(
    da: xr.DataArray | xr.Dataset,
    smooth_map: Mapping | None,
) -> xr.DataArray:
    """
    Unpacks the smooth_map and calls the appropriate functions along each dimension specified.
    A smooth map is a dictionnary whose keys are dimensions and whose values are 2-tuples. The first element of the tuple if the type of smoothing ("win" or "fft") and the second is the strength of the smoothing, the window size for window smoothing and the number of frequencies to zero out for fft.

    :param da: _description_
    :type da: xr.DataArray | xr.Dataset
    :param smooth_map: _description_
    :type smooth_map: Mapping | None
    :return: _description_
    :rtype: xr.DataArray
    """
    import xrft

    if smooth_map is None:
        return da
    for dim, value in smooth_map.items():
        if dim == "detrended":
            if value:
                da = da.map_blocks(xrft.detrend, template=da, args=["time", "linear"])
            continue
        smooth_type, winsize = value
        if smooth_type.lower() in ["lowpass", "fft", "fft_smoothing"]:
            da = fft_smoothing(da, dim, winsize)
        elif smooth_type.lower() in ["win", "window", "window_smoothing"]:
            da = window_smoothing(da, dim, winsize)
        elif smooth_type.lower() in ["trunc", "spherical", "truncation", "windspharm"]:
            da = spharm_smoothing(da, winsize)
    return da


def compute_hourofyear(da: xr.DataArray) -> xr.DataArray:
    return da.time.dt.hour + 24 * (da.time.dt.dayofyear - 1)


def assign_clim_coord(da: xr.DataArray, clim_type: str):
    if clim_type.lower() == "hourofyear":
        da = da.assign_coords(hourofyear=compute_hourofyear(da))
        coord = da.hourofyear
    elif clim_type.lower() in [
        att for att in dir(da.time.dt) if not att.startswith("_")
    ]:
        coord = getattr(da.time.dt, clim_type)
    else:
        raise NotImplementedError
    return da, coord


def compute_clim(da: xr.DataArray, clim_type: str) -> xr.DataArray:
    da, coord = assign_clim_coord(da, clim_type)
    clim = da.groupby(coord).mean()
    return compute(clim, progress_flag=True)


def compute_anom(
    anom: xr.DataArray, clim: xr.DataArray, clim_type: str, normalized: bool = False
):
    anom, coord = assign_clim_coord(anom, clim_type)
    this_gb = anom.groupby(coord)
    if not normalized:
        anom = this_gb - clim
    else:
        variab = anom.groupby(coord).std()
        anom = ((this_gb - clim).groupby(coord) / variab).reset_coords(
            "hourofyear", drop=True
        )
        anom = anom.where((anom != np.nan) & (anom != np.inf) & (anom != -np.inf), 0)
    return anom.reset_coords(clim_type, drop=True)


def compute_all_smoothed_anomalies(
    dataset: str,
    level_type: Literal["plev"]
    | Literal["thetalev"]
    | Literal["surf"]
    | Literal["2PVU"]
    | None = None,
    varname: str | None = None,
    resolution: str | None = None,
    clim_type: str | None = None,
    clim_smoothing: Mapping = None,
    smoothing: Mapping = None,
) -> None:
    path, clim_path, anom_path = data_path(
        dataset,
        level_type,
        varname,
        resolution,
        clim_type,
        clim_smoothing,
        smoothing,
        True,
    )
    anom_path.mkdir(parents=True, exist_ok=True)

    dest_clim = clim_path.joinpath("clim.nc")
    dests_anom = [
        anom_path.joinpath(fn.name) for fn in path.iterdir() if fn.suffix == ".nc"
    ]
    if dest_clim.is_file() and all([dest_anom.is_file() for dest_anom in dests_anom]):
        return

    sources = [
        source
        for source in path.iterdir()
        if source.is_file() and source.suffix == ".nc"
    ]

    if clim_type is None:
        for source, dest in tqdm(zip(sources, dests_anom), total=len(dests_anom)):
            if dest.is_file():
                continue
            anom = standardize(_open_many_da_wrapper(source, varname))
            anom = compute(smooth(anom, smoothing).astype(np.float32))
            to_netcdf(anom, dest)
        return
    if dest_clim.is_file():
        clim = open_dataarray(dest_clim)
    else:
        da = open_da(
            dataset, level_type, varname, resolution, period="all", levels="all"
        )
        if "lev" in da.dims:
            clims = []
            for lev in tqdm(da["lev"].values):
                da_ = extract_levels(da, lev)
                clim = compute_clim(da_, clim_type)
                clim = smooth(clim, clim_smoothing)
                clims.append(clim)
            clim = xr.concat(clims, dim="lev").assign_coords(lev=da["lev"])
        else:
            clim = compute_clim(da, clim_type)
            clim = smooth(clim, clim_smoothing)
        to_netcdf(clim.astype(np.float32), dest_clim)
    if len(sources) > 1:
        iterator_ = tqdm(zip(sources, dests_anom), total=len(dests_anom))
    else:
        iterator_ = zip(sources, dests_anom)
    for source, dest in iterator_:
        anom = _open_many_da_wrapper(source)
        anom = compute_anom(anom, clim, clim_type, False)
        if smoothing is not None:
            anom = smooth(anom, smoothing)
            anom = compute(anom.astype(np.float32))
        to_netcdf(anom, dest)


def time_mask(time_da: xr.DataArray, filename: str) -> np.ndarray:
    if filename == "full.nc":
        return np.ones(len(time_da)).astype(bool)

    filename = int(filename.rstrip(".nc"))
    try:
        t1, t2 = (
            pd.to_datetime(filename, format="%Y%M"),
            pd.to_datetime(filename + 1, format="%Y%M"),
        )
    except ValueError:
        t1, t2 = (
            pd.to_datetime(filename, format="%Y"),
            pd.to_datetime(filename + 1, format="%Y"),
        )
    return ((time_da >= t1) & (time_da < t2)).values


def get_nao(df: pl.DataFrame) -> pl.DataFrame:
    nao = pl.read_csv(f"{DATADIR}/ERA5/daily_nao.csv")
    nao = (
        nao.with_columns(
            time=pl.datetime(pl.col("year"), pl.col("month"), pl.col("day"))
        )
        .drop(["year", "month", "day"])
        .cast({"time": df["time"].dtype})
    )
    return df.join_asof(nao, on="time")


def compute_extreme_climatology(da: xr.DataArray, opath: Path):
    q = da.quantile(np.arange(60, 100) / 100, dim=["lon", "lat"])
    q_clim = compute_clim(q, "dayofyear")
    q_clim = smooth(q_clim, {"dayofyear": ("win", 60)})
    to_netcdf(q_clim, opath)


def compute_anomalies_ds(
    ds: xr.Dataset, clim_type: str, normalized: bool = False, return_clim: bool = False
) -> xr.Dataset:
    ds, coord = assign_clim_coord(ds, clim_type)
    clim = ds.groupby(coord).mean()
    clim = smooth(clim, {clim_type: ("win", 61)})
    this_gb = ds.groupby(coord)
    if not normalized:
        if return_clim:
            return (this_gb - clim).reset_coords(clim_type, drop=True), clim
        return (this_gb - clim).reset_coords(clim_type, drop=True)
    variab = ds.groupby(coord).std()
    variab = smooth(variab, {clim_type: ("win", 61)})
    if return_clim:
        return ((this_gb - clim).groupby(coord) / variab).reset_coords(
            clim_type, drop=True
        ), clim
    return ((this_gb - clim).groupby(coord) / variab).reset_coords(clim_type, drop=True)


def periodic_rolling_pl(
    df: pl.DataFrame, winsize: int, data_vars: list, dim: str = "dayofyear"
):
    df = df.cast({dim: pl.Int32})
    halfwinsize = winsize // 2
    other_columns = get_index_columns(df, ("member", "jet"))
    descending = [False, *[col == "jet" for col in other_columns]]
    len_ = [df[col].unique().len() for col in other_columns]
    len_ = np.prod(len_)
    min_doy = df[dim].min()
    max_doy = df[dim].max()
    df = df.sort([dim, *other_columns], descending=descending)
    df = pl.concat(
        [
            df.tail(halfwinsize * len_).with_columns(
                pl.col(dim) - max_doy + min_doy - 1
            ),
            df,
            df.head(halfwinsize * len_).with_columns(
                pl.col(dim) + max_doy - min_doy + 1
            ),
        ]
    )
    df = df.rolling(
        pl.col(dim),
        period=f"{winsize}i",
        offset=f"-{halfwinsize + 1}i",
        group_by=other_columns,
    ).agg(*[pl.col(col).mean() for col in data_vars])
    df = df.sort([dim, *other_columns], descending=descending)
    df = df.slice(halfwinsize * len_, (max_doy - min_doy + 1) * len_)
    return df


def compute_anomalies_pl(
    df: pl.DataFrame,
    other_index_columns: Tuple = ("jet",),
    smooth_clim: int = 0,
    normalize: bool = False,
):
    data_columns = [
        col for col in df.columns if col not in ["time", *other_index_columns]
    ]
    df = df.with_columns(dayofyear=pl.col("time").dt.ordinal_day().cast(pl.Int32))
    clim = df.group_by(pl.col("dayofyear"), *other_index_columns).mean().drop("time")
    if smooth_clim > 1:
        clim = periodic_rolling_pl(clim, smooth_clim, data_columns)
    df = df.join(clim, on=["dayofyear", *other_index_columns], suffix="_clim")
    if not normalize:
        df = df.select(
            *["time", *other_index_columns],
            **{col: pl.col(col) - pl.col(f"{col}_clim") for col in data_columns},
        )
        return df
    std = df.group_by(pl.col("dayofyear"), *other_index_columns).agg(
        *[pl.col(col).std() for col in data_columns]
    )
    if smooth_clim > 1:
        std = periodic_rolling_pl(std, smooth_clim, data_columns)
    df = df.join(std, on=["dayofyear", *other_index_columns], suffix="_std")
    df = df.select(
        *["time", *other_index_columns],
        **{
            col: (pl.col(col) - pl.col(f"{col}_clim")) / pl.col(f"{col}_std")
            for col in data_columns
        },
    )
    return df


def _fix_dict_lists(dic: dict) -> dict:
    for key, val in dic.items():
        if isinstance(val, np.ndarray):
            dic[key] = val.tolist()
    return dic


def find_spot(basepath: Path, metadata: Mapping) -> Path:
    found = False
    metadata = _fix_dict_lists(metadata)
    for dir in basepath.iterdir():
        if not dir.is_dir():
            continue
        try:
            other_mda = _fix_dict_lists(load_pickle(dir.joinpath("metadata.pkl")))
            if "varnames" in other_mda:
                other_mda["varnames"].sort()
        except FileNotFoundError:
            continue
        if metadata == other_mda:
            newpath = basepath.joinpath(dir.name)
            found = True
            break

    if not found:
        seq = [
            int(dir.name)
            for dir in basepath.iterdir()
            if dir.is_dir() and dir.name.isnumeric()
        ]
        id = max(seq) + 1 if len(seq) != 0 else 1
        newpath = basepath.joinpath(str(id))
        newpath.mkdir()
        save_pickle(metadata, newpath.joinpath("metadata.pkl"))
    return newpath


def flatten_by(ds: xr.Dataset, by: str = "s") -> xr.Dataset:
    if "lev" not in ds.dims:
        return ds
    unique_levs = np.unique(ds.lev.values)
    ope = np.nanargmin if by[0] == "-" else np.nanargmax
    by = by.lstrip("-")
    if ds[by].chunks is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds = compute(ds, progress_flag=True)
    ds[by] = ds[by].interpolate_na("time", method="linear", fill_value="extrapolate")
    levmax = ds[by].reduce(ope, dim="lev")
    ds = ds.isel(lev=levmax).reset_coords("lev")  # but not drop
    ds["lev"] = ds["lev"].astype(np.float32)
    ds.attrs["orig_lev"] = unique_levs
    ds.attrs["flattened"] = 1
    return ds


def metadata_from_da(
    da: xr.DataArray | xr.Dataset, varname: str | list | None = None
) -> dict:
    if isinstance(da, xr.DataArray) and varname is None:
        varname = da.name
    elif isinstance(da, xr.Dataset) and varname is None:
        varname = list(da.data_vars)
        varname.sort()
    period = np.unique(da.time.dt.year).tolist()
    season = np.unique(da.time.dt.month).tolist()
    nullseason = {None: list(range(1, 13))}
    for seasonname, monthlist in (SEASONS | nullseason).items():
        if monthlist == season:
            season = seasonname
            break
    region = get_region(da)
    if "lev" in da.dims:
        levels = da.lev.values.tolist()
    elif "orig_lev" in da.attrs:
        levels = da.attrs["orig_lev"]
    else:
        levels = None
    metadata = {
        "varname": varname,
        "period": period,
        "season": season,
        "region": region,
        "levels": levels,
    }
    if "member" in da.dims:
        metadata["members"] = np.unique(da.member).tolist()
    if "flattened" in da.attrs:
        metadata["flattened"] = int(da.attrs["flattened"])
    return metadata


class DataHandler(object):
    def __init__(
        self,
        path: Path | str,
        da: xr.DataArray | xr.Dataset | None = None,
    ) -> None:
        self.path = Path(path)
        if da is None:
            try:
                self.da = open_dataarray(self.path.joinpath("da.nc"))
            except ValueError:
                self.da = open_dataset(self.path.joinpath("da.nc"))
        else:
            self.da = da
        self._setup_dims()
        self.metadata = load_pickle(self.path.joinpath("metadata.pkl"))

    @classmethod
    def from_basepath_and_da(
        cls, basepath: Path | str, da: xr.DataArray | xr.Dataset, save_da: bool = False
    ) -> "DataHandler":
        basepath = Path(basepath)
        metadata = metadata_from_da(da)
        path = find_spot(basepath, metadata)
        if save_da and not path.joinpath("da.nc").is_file():
            da = compute(da, progress_flag=True)
            to_netcdf(da, path.joinpath("da.nc"))
        return cls(path, da)

    def _setup_dims(self):
        self.sample_dims = determine_sample_dims(self.da)
        self.feature_dims = determine_feature_dims(self.da)
        self.extra_dims = {}
        for coordname, coord in self.da.coords.items():
            if coordname not in self.da.dims:
                self.extra_dims[coordname] = coord.values
        if "lev" in self.da.dims and "lev":
            self.extra_dims = {"lev": self.da.lev.values}
        self.flat_shape = (
            np.prod([len(dim) for dim in self.sample_dims.values()]),
            np.prod([len(dim) for dim in self.feature_dims.values()]),
        )

    def get_path(self) -> Path:
        return self.path

    def load_da(self, progress: bool = False, **kwargs):
        self.da = compute(self.da, progress_flag=progress, **kwargs)

    def get_metadata(self) -> Mapping:
        return self.metadata

    def get_sample_dims(self) -> Mapping:
        return self.sample_dims

    def get_feature_dims(self) -> Mapping:
        return self.feature_dims

    def get_extra_dims(self) -> Mapping:
        return self.extra_dims

    def get_flat_shape(self) -> Mapping:
        return self.flat_shape

    @classmethod
    def from_specs(
        cls,
        dataset: str,
        level_type: Literal["plev"]
        | Literal["thetalev"]
        | Literal["surf"]
        | Literal["2PVU"]
        | None = None,
        varname: str | None = None,
        resolution: str | None = None,
        period: list | tuple | Literal["all"] | int | str = "all",
        season: list | str = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        levels: int | str | tuple | list | Literal["all"] = "all",
        clim_type: str = None,
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
        reduce_da: bool = True,
    ) -> "DataHandler":
        if isinstance(varname, Tuple):
            varname_in_path, varname_in_file = varname
        else:
            varname_in_path, varname_in_file = varname, varname
        path = data_path(
            dataset,
            level_type,
            varname_in_path,
            resolution,
            clim_type,
            clim_smoothing,
            smoothing,
            False,
        ).joinpath("results")
        path.mkdir(exist_ok=True)
        if level_type == "surf":
            levels = None
        open_da_args = (
            dataset,
            level_type,
            varname,
            resolution,
            period,
            season,
            minlon,
            maxlon,
            minlat,
            maxlat,
            levels,
            clim_type,
            clim_smoothing,
            smoothing,
        )
        if levels != "all" and levels is not None:
            levels, level_names = unpack_levels(levels)

        region = (minlon, maxlon, minlat, maxlat)

        if isinstance(period, str) and period == "all":
            period = determine_period(path.parent)
        elif isinstance(period, np.ndarray):
            period = period.tolist()
        elif isinstance(period, tuple):
            period = np.arange(period[0], period[-1]).tolist()

        metadata = {
            "varname": varname,
            "period": period,
            "season": season,
            "region": region,
            "levels": levels,
        }
        if varname_in_file in ["u", "v", "s", "high_wind", "low_wind"]:
            metadata["flattened"] = int(reduce_da)
        path = find_spot(path, metadata)
        da_path = path.joinpath("da.nc")
        if da_path.is_file():
            da = open_dataset(
                da_path, chunks={"time": 100, "lat": -1, "lon": -1, "lev": -1}
            )
            if len(da.data_vars) == 1:
                da = da[list(da.data_vars)[0]]
        else:
            da = open_da(*open_da_args)
            da = smooth(da, smoothing)
            da = compute(da, progress_flag=True)
            if reduce_da:
                if isinstance(da, xr.DataArray):
                    da = flatten_by(xr.Dataset({varname: da}), varname)[varname]
                else:
                    da = flatten_by(da, "s")
            to_netcdf(da, da_path, format="NETCDF4")
        return cls(path, da)

    @classmethod
    def from_several_dhs(
        cls,
        data_handlers: Mapping[str, "DataHandler"],
        flatten_ds: bool = True,
    ):
        varnames = list(data_handlers)
        if flatten_ds:
            varnames.append("lev")
        varnames.sort()
        data_handlers_list = list(data_handlers.values())
        paths = [dh.get_path() for dh in data_handlers_list]
        basepath = commonpath(paths)
        path = Path(basepath, "results")
        path.mkdir(exist_ok=True)

        all_mdas = [dh.get_metadata() for dh in data_handlers_list]
        for mda in all_mdas:
            mda.pop("varname")
        assert all([mda == all_mdas[0] for mda in all_mdas])

        metadata = {
            "varname": varnames,
        } | all_mdas[0]

        if any([varname in ["u", "v", "s"] for varname in varnames]):
            metadata["flattened"] = int(flatten_ds)

        path = find_spot(path, metadata)

        dspath = path.joinpath("ds.nc")
        if dspath.is_file():
            ds = open_dataset(
                dspath, chunks={"time": 100, "lat": -1, "lon": -1, "lev": -1}
            )
            return cls(ds, path.parent)

        ds = {}
        for varname, dh in data_handlers.items():
            ds[varname] = dh.da
        ds = xr.Dataset(ds)
        ds = compute(ds, progress_flag=True)
        if flatten_ds:
            ds = flatten_by(ds, "s")
        to_netcdf(ds, dspath)
        return cls(path, ds)

    @classmethod
    def from_intake(
        cls,
        url: str,
        varname: str,
        basepath: Path,
        component="atm",
        experiment: str | list[str] = "historical",
        frequency: Literal["daily", "monthly"] = "daily",
        forcing_variant: Literal["cmip6", "smbb"] = "cmip6",
        period: list | tuple | Literal["all"] | int | str = "all",
        season: list | str = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        levels: int | str | tuple | list | Literal["all"] = "all",
        members: str | list | Literal["all"] = "all",
        reduce_da: bool = True,
    ) -> "DataHandler":
        import intake

        basepath = Path(basepath)
        if varname == "s":
            varname_to_search = ["U", "V"]
        else:
            varname_to_search = varname
        varname = varname.lower()
        indexers = [component, experiment, frequency, forcing_variant]
        indexers = [np.atleast_1d(indexer) for indexer in indexers]
        indexers = [[str(idx_) for idx_ in indexer] for indexer in indexers]
        indexers = list(product(*indexers))
        ensemble_keys = [".".join(indexer) for indexer in indexers]

        metadata = {
            "varname": varname.lower(),
            "ensemble_keys": ensemble_keys,
            "members": members,
            "period": period,
            "season": season,
            "region": (minlon, maxlon, minlat, maxlat),
            "levels": levels,
        }

        path = find_spot(basepath, metadata)
        da_path = path.joinpath("da.zarr")
        if da_path.exists():
            da = open_dataarray(da_path, engin="zarr")
            return cls(da, path.parent)

        catalog = intake.open_esm_datastore(url)
        catalog_subset = catalog.search(
            variable=varname_to_search,
            component=component,
            experiment=experiment,
            frequency=frequency,
            forcing_variant=forcing_variant,
        )
        dsets = catalog_subset.to_dataset_dict(storage_options={"anon": True})
        if len(ensemble_keys) == 1:
            da = dsets[ensemble_keys[0]]
        else:
            da = []
            for ek in ensemble_keys:
                da.append(dsets[ek])
            da = xr.concat(da, dim="time")

        if varname == "s":
            da = np.sqrt(da["U"] ** 2 + da["V"] ** 2)
        else:
            da = da[varname]

        da = extract(
            da,
            period=period,
            season=season,
            minlon=minlon,
            maxlon=maxlon,
            minlat=minlat,
            maxlat=maxlat,
            levels=levels,
            members=members,
        )

        da = da.chunk({"member": 1, "time": 10000, "lev": -1, "lon": -1, "lat": -1})

        if reduce_da:
            da = da.max("lev")

        return cls(path, da)
