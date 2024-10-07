import warnings
from os.path import commonpath

from typing import Union, Optional, Mapping, Sequence, Tuple, Literal
from itertools import product
from nptyping import NDArray
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import flox.xarray
import xrft
import intake
from tqdm import tqdm
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from jetstream_hugo.definitions import (
    TIMERANGE,
    DEFAULT_VARNAME,
    DATADIR,
    YEARS,
    COMPUTE_KWARGS,
    get_region,
    save_pickle,
    load_pickle,
    _compute,
)

SEASONS = {
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def get_land_mask() -> xr.DataArray:
    mask = xr.open_dataarray(f"{DATADIR}/ERA5/grid_info/land_sea.nc")
    mask = (
        mask.squeeze()
        .rename(longitude="lon", latitude="lat")
        .reset_coords("time", drop=True)
    )
    return mask.astype(bool)


def determine_file_structure(path: Path) -> str:
    if path.joinpath("full.nc").is_file():
        return "one_file"
    if any([path.joinpath(f"{year}.nc").is_file() for year in YEARS]):
        return "yearly"
    if any([path.joinpath(f"{year}01.nc").is_file() for year in YEARS]):
        return "monthly"
    print("Could not determine file structure")
    raise RuntimeError


def data_path(
    dataset: str,
    level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
    varname: str,
    resolution: str,
    clim_type: str | None = None,
    clim_smoothing: Mapping | None = None,
    smoothing: Mapping | None = None,
    for_compute_anomaly: bool = False,
) -> Path | Tuple[Path, Path, Path]:
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

    path = Path(DATADIR, dataset, level_type, varname, resolution)

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
    standard_dict = {
        "longitude": "lon",
        "latitude": "lat",
        "level": "lev",
        "member_id": "member",
        "U": "u",
        "V": "v",
        "T": "t",
    }
    for key, value in standard_dict.items():
        try:
            da = da.rename({key: value})
        except ValueError:
            pass
    da["time"] = da.indexes["time"].to_datetimeindex()
    da = da.astype(np.float32)
    if (da.lon.max() > 180) and (da.lon.min() >= 0):
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = da.sortby("lon")
    if np.diff(da.lat.values)[0] < 0:
        da = da.reindex(lat=da.lat[::-1])
    return da.unify_chunks()


def extract_period(da, period: list | tuple | Literal["all"] | int | str = "all"):
    if period == "all":
        return da
    if isinstance(period, tuple):
        if len(period) == 2:
            period = np.arange(period[0], period[1] + 1)
        elif len(period) == 3:
            period = np.arange(period[0], period[1] + 1, period[2])
    elif isinstance(period, int):
        period = [period]
    return da.isel(time=np.isin(da.time.dt.year, period))


def unpack_levels(levels: int | str | tuple | list) -> Tuple[list, list]:
    if isinstance(levels, int | str | tuple):
        levels = [levels]
    to_sort = []
    for level in levels:
        to_sort.append(float(level) if isinstance(level, int | str) else level[0])
    levels = [levels[i] for i in np.argsort(to_sort)]
    level_names = []
    for level in levels:
        if isinstance(level, tuple | list):
            level_names.append(f"{level[0]}-{len(level)}-{level[-1]}")
        else:
            level_names.append(str(level))
    return levels, level_names


def extract_levels(da: xr.DataArray, levels: int | str | list | tuple | Literal["all"]):
    if levels == "all" or (isinstance(levels, Sequence) and "all" in levels):
        return da.squeeze()

    levels, level_names = unpack_levels(levels)

    if not any([isinstance(level, tuple) for level in levels]):
        try:
            da = da.isel(lev=levels).squeeze()
        except ValueError:
            da = da.sel(lev=levels).squeeze()
        if len(levels) == 1:
            return da.reset_coords("lev", drop=True)
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


def _open_dataarray(filename: Path | list[Path], varname: str) -> xr.DataArray:
    if isinstance(filename, list) and len(filename) == 1:
        filename = filename[0]
    if isinstance(filename, list):
        da = xr.open_mfdataset(filename, chunks=None)
        da = da.unify_chunks()
    else:
        da = xr.open_dataset(filename, chunks="auto")
        da = da.unify_chunks()
    if "expver" in da.dims:
        da = da.sel(expver=1).reset_coords("expver", drop=True)
    try:
        da = da[varname]
    except KeyError:
        try:
            da = da[DEFAULT_VARNAME]
        except KeyError:
            da = da[list(da.data_vars)[-1]]
    return da.rename(varname)


def extract(
    da: xr.DataArray,
    period: list | tuple | Literal["all"] | int | str = "all",
    season: list | str | tuple | None = None,
    minlon: Optional[int | float] = None,
    maxlon: Optional[int | float] = None,
    minlat: Optional[int | float] = None,
    maxlat: Optional[int | float] = None,
    levels: int | str | list | tuple | Literal["all"] = "all",
    members: str | list | Literal["all"] = "all",
):
    da = standardize(da)
    
    da = extract_period(da, period)
    
    if season is not None:
        da = extract_season(da, season)

    if "member" in da.dims and members != "all":
        try:
            da = da.isel(member=members)
        except ValueError:
            da = da.sel(member=members)
        
    if all([bound is not None for bound in [minlon, maxlon, minlat, maxlat]]):
        da = da.sel(lon=slice(minlon, maxlon), lat=slice(minlat, maxlat))

    if "lev" in da.dims:
        if len(da.lev) == 1:
            da = da.reset_coords("lev", drop=True)
            
    if "lev" in da.dims and levels != "all":
        da = extract_levels(da, levels)

    return da


def open_da(
    dataset: str,
    level_type: (
        Literal["plev"] | Literal["thetalev"] | Literal["2PVU"] | Literal["surf"]
    ),
    varname: str,
    resolution: str,
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
    path = data_path(
        dataset,
        level_type,
        varname,
        resolution,
        clim_type,
        clim_smoothing,
        smoothing,
        False,
    )
    file_structure = determine_file_structure(path)

    if isinstance(period, tuple):
        period = np.arange(int(period[0]), int(period[1] + 1))
    elif isinstance(period, list):
        period = np.asarray(period).astype(int)
    elif period == "all":
        period = YEARS
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

    da = _open_dataarray(files_to_load, varname)
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
    resolution = da[dim][1] - da[dim][0]
    if dim in ["lon", "longitude"]:
        return 360 >= da[dim][-1] >= 360 - resolution and da[dim][0] == 0.0
    return dim in ["dayofyear", "hourofyear"]


def _window_smoothing(
    da: xr.DataArray | xr.Dataset, dim: str, winsize: int, center: bool = True
) -> xr.DataArray:
    if dim != "hourofyear":
        return da.rolling({dim: winsize}, center=True, min_periods=1).mean()
    groups = da.groupby(da.hourofyear % 24)
    to_concat = []
    winsize = winsize // len(groups)
    if "time" in da.dims:
        dim = "time"
    for group in groups.groups.values():
        to_concat.append(
            da.loc[{dim: da.hourofyear[group]}]
            .rolling({dim: winsize // 4}, center=center, min_periods=1)
            .mean()
        )
    return xr.concat(to_concat, dim=dim).sortby(dim)


def window_smoothing(
    da: xr.DataArray, dim: str, winsize: int, center: bool = True
) -> xr.DataArray:
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
    raise NotImplementedError("Broken for now")
    # spharmt = Spharmt(len(da.lon), len(da.lat))
    # fieldspec = spharmt.grdtospec(da.values, ntrunc=trunc)
    # fieldtrunc = spharmt.spectogrd(fieldspec)
    # return fieldtrunc


def smooth(
    da: xr.DataArray | xr.Dataset,
    smooth_map: Mapping | None,
) -> xr.DataArray:
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
    with ProgressBar():
        clim = flox.xarray.xarray_reduce(
            da,
            coord,
            func="mean",
            method="cohorts",
            expected_groups=np.unique(coord.values),
        ).compute(**COMPUTE_KWARGS)
    return clim


def compute_anom(
    anom: xr.DataArray, clim: xr.DataArray, clim_type: str, normalized: bool = False
):
    anom, coord = assign_clim_coord(anom, clim_type)
    this_gb = anom.groupby(coord)
    if not normalized:
        anom = this_gb - clim
    else:
        variab = flox.xarray.xarray_reduce(
            anom,
            coord,
            func="std",
            method="cohorts",
            expected_groups=np.unique(coord.values),
        )
        anom = ((this_gb - clim).groupby(coord) / variab).reset_coords(
            "hourofyear", drop=True
        )
        anom = anom.where((anom != np.nan) & (anom != np.inf) & (anom != -np.inf), 0)
    return anom.reset_coords(clim_type, drop=True)


def compute_all_smoothed_anomalies(
    dataset: str,
    level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
    varname: str,
    resolution: str,
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
            anom = standardize(_open_dataarray(source, varname))
            anom = smooth(anom, smoothing).astype(np.float32).compute(**COMPUTE_KWARGS)
            anom.to_netcdf(dest)
        return
    if dest_clim.is_file():
        clim = xr.open_dataarray(dest_clim)
    else:
        da = open_da(
            dataset, level_type, varname, resolution, period="all", levels="all"
        )
        clim = compute_clim(da, clim_type)
        clim = smooth(clim, clim_smoothing)
        clim.astype(np.float32).to_netcdf(dest_clim)
    if len(sources) > 1:
        iterator_ = tqdm(zip(sources, dests_anom), total=len(dests_anom))
    else:
        iterator_ = zip(sources, dests_anom)
    for source, dest in iterator_:
        anom = standardize(xr.open_dataarray(source))
        anom = compute_anom(anom, clim, clim_type, False)
        if smoothing is not None:
            anom = smooth(anom, smoothing)
            anom = anom.astype(np.float32).compute(**COMPUTE_KWARGS)
        anom.to_netcdf(dest)


def time_mask(time_da: xr.DataArray, filename: str) -> NDArray:
    if filename == "full.nc":
        return np.ones(len(time_da)).astype(bool)

    filename = int(filename.rstrip(".nc"))
    try:
        t1, t2 = pd.to_datetime(filename, format="%Y%M"), pd.to_datetime(
            filename + 1, format="%Y%M"
        )
    except ValueError:
        t1, t2 = pd.to_datetime(filename, format="%Y"), pd.to_datetime(
            filename + 1, format="%Y"
        )
    return ((time_da >= t1) & (time_da < t2)).values


def get_nao(df: pl.DataFrame) -> pl.DataFrame:
    nao = pl.read_csv(f"{DATADIR}/ERA5/daily_nao.csv")
    nao = (
        nao
        .with_columns(time=pl.datetime(pl.col("year"), pl.col("month"), pl.col("day")))
        .drop(["year", "month", "day"])
        .cast({"time": df["time"].dtype})
    )
    return df.join_asof(nao, on="time")


def compute_extreme_climatology(da: xr.DataArray, opath: Path):
    q = da.quantile(np.arange(60, 100) / 100, dim=["lon", "lat"])
    q_clim = compute_clim(q, "dayofyear")
    q_clim = smooth(q_clim, {"dayofyear": ("win", 60)})
    q_clim.to_netcdf(opath)


def compute_anomalies_ds(
    ds: xr.Dataset, clim_type: str, normalized: bool = False, return_clim: bool = False
) -> xr.Dataset:
    ds, coord = assign_clim_coord(ds, clim_type)
    clim = flox.xarray.xarray_reduce(
        ds,
        coord,
        func="nanmean",
        method="cohorts",
        expected_groups=np.unique(coord.values),
    )
    clim = smooth(clim, {clim_type: ("win", 61)})
    this_gb = ds.groupby(coord)
    if not normalized:
        if return_clim:
            return (this_gb - clim).reset_coords(clim_type, drop=True), clim
        return (this_gb - clim).reset_coords(clim_type, drop=True)
    variab = flox.xarray.xarray_reduce(
        ds,
        coord,
        func="nanstd",
        method="cohorts",
        expected_groups=np.unique(coord.values),
    )
    variab = smooth(variab, {clim_type: ("win", 61)})
    if return_clim:
        return ((this_gb - clim).groupby(coord) / variab).reset_coords(
            clim_type, drop=True
        ), clim
    return ((this_gb - clim).groupby(coord) / variab).reset_coords(clim_type, drop=True)


def _fix_dict_lists(dic: dict) -> dict:
    for key, val in dic.items():
        if isinstance(val, NDArray):
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
        seq = [int(dir.name) for dir in basepath.iterdir() if dir.is_dir()]
        id = max(seq) + 1 if len(seq) != 0 else 1
        newpath = basepath.joinpath(str(id))
        newpath.mkdir()
        save_pickle(metadata, newpath.joinpath("metadata.pkl"))
    return newpath


def flatten_by(ds: xr.Dataset, by: str = "-criterion") -> xr.Dataset:
    if "lev" not in ds.dims:
        return ds
    unique_levs = np.unique(ds.lev.values)
    ope = np.nanargmin if by[0] == "-" else np.nanargmax
    by = by.lstrip("-")
    if ds["s"].chunks is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with ProgressBar():
                ds = ds.compute(**COMPUTE_KWARGS)
    ds[by] = ds[by].interpolate_na("time", method="linear", fill_value="extrapolate")
    levmax = ds[by].reduce(ope, dim="lev")
    ds = ds.isel(lev=levmax).reset_coords("lev")  # but not drop
    ds["lev"] = ds["lev"].astype(np.float32)
    ds.attrs["orig_lev"] = unique_levs
    ds.attrs["flattened"] = 1
    return ds


def metadata_from_da(da: xr.DataArray | xr.Dataset, varname: str | list | None = None) -> dict:
    if isinstance(da, xr.DataArray) and varname is None:
        varname = da.name
    elif isinstance(da, xr.Dataset) and varname is None:
        varname = list(da.data_vars)
        varname.sort()
    period = np.unique(da.time.dt.year)
    season = np.unique(da.time.dt.month)
    nullseason = {None: list(range(1, 13))}
    for seasonname, monthlist in (SEASONS | nullseason).items():
        if monthlist == season.tolist():
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
        "period": period.tolist(),
        "season": season,
        "region": region,
        "levels": levels,
    }
    if "member" in da.dims:
        metadata["members"] = np.unique(da.member).tolist()
    if "flattened" in da.attrs:
        metadata["flattened"] = da.attrs["flattened"]
    return metadata


class DataHandler(object):
    def __init__(
        self,
        da: xr.DataArray | xr.Dataset,
        basepath: Path | str,
    ) -> None:
        basepath = Path(basepath)
        self.da = da
        self._setup_dims()
        self.metadata = metadata_from_da(self.da)
        self.path = find_spot(basepath, self.metadata)

    def _setup_dims(self):
        self.sample_dims = {"time": self.da.time.values}
        try:
            self.sample_dims["member"] = self.da.member.values
        except AttributeError:
            pass
        self.lon, self.lat = self.da.lon.values, self.da.lat.values
        if "lev" in self.da.dims:
            self.feature_dims = {"lev": self.da.lev.values}
            len(self.da.lev.values)
        else:
            self.feature_dims = {}
        self.feature_dims["lat"] = self.lat
        self.feature_dims["lon"] = self.lon
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
        self.da = _compute(self.da, progress=progress, **kwargs)

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
        level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
        varname: str,
        resolution: str,
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
        path = data_path(
            dataset,
            level_type,
            varname,
            resolution,
            clim_type,
            clim_smoothing,
            smoothing,
            False,
        ).joinpath("results")
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
        
        if period == "all":
            period = YEARS
        
        metadata = {
            "varname": varname,
            "period": period.tolist(),
            "season": season,
            "region": region,
            "levels": levels,
        }
        if varname in ["u", "v", "s"]:
            metadata["flattened"] = reduce_da
        path = find_spot(path, metadata)
        da_path = path.joinpath("da.nc")
        if da_path.is_file():
            da = xr.open_dataarray(
                da_path, chunks={"time": 100, "lat": -1, "lon": -1, "lev": -1}
            )
        else:
            da = open_da(*open_da_args)
            da = smooth(da, smoothing)
            with ProgressBar():
                da = da.load(**COMPUTE_KWARGS)
            if reduce_da:
                da = flatten_by(xr.Dataset({varname: da}), varname)[varname]
            da.to_netcdf(da_path, format="NETCDF4")
        return cls(da, path.parent)
    
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
            metadata["flattened"] = flatten_ds

        path = find_spot(path, metadata)
        
        dspath = path.joinpath("ds.nc")
        if dspath.is_file():
            ds = xr.open_dataset(dspath, chunks={"time": 100, "lat": -1, "lon": -1, "lev": -1})
            return cls(ds, path.parent)
        
        ds = {}
        for varname, dh in data_handlers.items():
            ds[varname] = dh.da
        ds = xr.Dataset(ds)
        with ProgressBar():
            ds = ds.load(**COMPUTE_KWARGS)
        if flatten_ds:
            ds = flatten_by(ds, "s")
        ds.to_netcdf(dspath)
        return cls(ds, path.parent)
    
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
            da = xr.open_dataarray(da_path, engin="zarr")
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
        
        da = da.chunk({"member": 1, "time": 10000, "lev":-1, "lon": -1, "lat": -1})
        
        if reduce_da:
            da = da.max("lev")

        return cls(da, path.parent)
        