from typing import Union, Optional, Mapping, Sequence, Tuple, Literal
from nptyping import NDArray
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xrft
from tqdm import tqdm
from dask.diagnostics import ProgressBar, ResourceProfiler
from jetstream_hugo.definitions import (
    DEFAULT_VARNAME,
    DATADIR,
    CLIMSTOR,
    YEARSPL_EXT,
    COMPUTE_KWARGS
)

def filenamesml(
    y: int, m: int, d: int
) -> str:  # Naming conventions of the files on climstor (why are they so different?)
    return [
        f"{CLIMSTOR}/ML/data/{str(y)}/P{str(y)}{str(m).zfill(2)}{str(d).zfill(2)}_{str(h).zfill(2)}"
        for h in range(0, 24, 6)
    ]


def filenamessfc(
    y: int, m: int, d: int
) -> str:  # Naming conventions of the files on climstor (why are they so different?)
    return [
        f"{CLIMSTOR}/SFC/data/{str(y)}/an_sfc_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"
    ]


def filenamespl(y: int, m: int, d: int) -> str:
    return [
        f"{CLIMSTOR}/PL/data/an_pl_ERA5_{str(y)}-{str(m).zfill(2)}-{str(d).zfill(2)}.nc"
    ]  # returns iterable to have same call signature as filenamescl(y, m, d)


def filenamegeneric(y: int, m: int, folder: int) -> str:
    return [f"{DATADIR}/{folder}/{y}{str(m).zfill(2)}.nc"]


def _fn(date: pd.Timestamp, which: str) -> str:
    if which == "ML":
        return filenamesml(date.year, date.month, date.day)
    elif which == "PL":
        return filenamespl(date.year, date.month, date.day)
    elif which == "SFC":
        return filenamessfc(date.year, date.month, date.day)
    else:
        return filenamegeneric(date.year, date.month, which)


# instead takes pandas.timestamp (or iterable of _) as input
def fn(date: Union[list, NDArray, pd.DatetimeIndex, pd.Timestamp], which):
    if isinstance(date, (list, NDArray, pd.DatetimeIndex)):
        filenames = []
        for d in date:
            filenames.extend(_fn(d, which))
        return filenames
    elif isinstance(date, pd.Timestamp):
        return _fn(date, which)
    else:
        raise TypeError(f"Invalid type : {type(date)}")
    
    
def determine_file_structure(path: Path) -> str:
    if path.joinpath("full.nc").is_file():
        return "one_file"
    if any([path.joinpath(f"{year}.nc").is_file() for year in YEARSPL_EXT]):
        return "yearly"
    if any([path.joinpath(f"{year}01.nc").is_file() for year in YEARSPL_EXT]):
        return "monthly"
    print("Could not determine file structure")
    raise RuntimeError


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


def data_path(
    dataset: str,
    varname: str,
    resolution: str,
    clim_type: str = None,
    clim_smoothing: Mapping = None,
    smoothing: Mapping = None,
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

    path = Path(DATADIR, dataset, varname, resolution)

    unpacked = unpack_smooth_map(clim_smoothing)
    underscore = "_" if unpacked != "" else ""
    
    clim_path = path.joinpath(clim_type + underscore + unpacked)
    anom_path = clim_path.joinpath(unpack_smooth_map(smoothing))
    if not anom_path.is_dir() and not for_compute_anomaly:
        if not clim_type == '':
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


def determine_chunks(da: xr.DataArray, chunk_type=None) -> Mapping:
    dims = list(da.coords.keys())
    lon_name = "lon" if "lon" in dims else "longitude"
    lat_name = "lat" if "lat" in dims else "latitude"
    lev_name = "lev" if "lev" in dims else "level"
    if lev_name in dims:
        chunks = {lev_name: -1}
    else:
        chunks = {}
        
    if chunk_type in ["horiz", "horizointal", "lonlat"]:
        chunks = {"time": 31, lon_name: -1, lat_name: -1} | chunks
    elif chunk_type in ["time"]:
        if da.nbytes > 5e9:
            chunks = {"time": -1, lon_name: 100, lat_name: 100} | chunks
        else:
            chunks = {"time": -1, lon_name: -1, lat_name: -1} | chunks
    else:
        chunks = {"time": 31, lon_name: 40, lat_name: -1} | chunks
    return chunks


def rename_coords(da: xr.DataArray) -> xr.DataArray:
    try:
        da = da.rename({"longitude": "lon", "latitude": "lat"})
    except ValueError:
        pass
    try:
        da = da.rename({"level": "lev"})
    except ValueError:
        pass
    return da


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


def extract_levels(da, levels):
    if levels == "all" or (isinstance(levels, Sequence) and 'all' in levels):
        return da.squeeze()

    levels, level_names = unpack_levels(levels)

    if ~any([isinstance(level, tuple) for level in levels]):
        da = da.sel(lev=levels)
        return da.squeeze()

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


def pad_wrap(da: xr.DataArray, dim: str) -> bool:
    resolution = da[dim][1] - da[dim][0]
    if dim in ["lon", "longitude"]:
        return (
            da[dim][-1] <= 360 and da[dim][-1] >= 360 - resolution and da[dim][0] == 0.0
        )
    return dim == "dayofyear"


def window_smoothing(da: xr.DataArray, dim: str, winsize: int) -> xr.DataArray:
    halfwinsize = int(np.ceil(winsize / 2))
    if pad_wrap(da, dim):
        da = da.pad({dim: halfwinsize}, mode="wrap")
        newda = da.rolling({dim: winsize}, center=True).mean()
        newda = newda.isel({dim: slice(halfwinsize, -halfwinsize)})
    else:
        newda = da.rolling({dim: winsize}, center=True, min_periods=1).mean()
    newda.attrs = da.attrs
    return newda


def fft_smoothing(da: xr.DataArray, dim: str, winsize: int) -> xr.DataArray:
    if dim == "time":
        winsize *= 24 * 3600
    name = da.name
    ft = xrft.fft(da, dim=dim, true_phase=True, true_amplitude=True)
    loc_kwargs = {f"freq_{dim}": np.abs(ft[f"freq_{dim}"]) > 1 / winsize}
    ft.loc[loc_kwargs] = 0
    newda = (
        xrft.ifft(ft, dim=f"freq_{dim}", true_phase=True, true_amplitude=True, lag=ft[f"freq_{dim}"].direct_lag)
        .real.assign_coords(da.coords)
        .rename(name)
    )
    if (da < 0).sum() == 0:
        newda = np.abs(newda)
    newda.attrs = da.attrs
    return newda


def smooth(
    da: xr.DataArray,
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
    return da


def _open_dataarray(filename: Path | list[Path], varname: str) -> xr.DataArray:
    if isinstance(filename, list) and len(filename) == 1:
        filename = filename[0]
    if isinstance(filename, list):
        da = xr.open_mfdataset(filename, chunks=None)
    else:
        da = xr.open_dataset(filename, chunks='auto')
    try:
        da = da[varname]
    except KeyError:
        try:
            da = da[DEFAULT_VARNAME]
        except KeyError:
            da = da[list(da.data_vars)[-1]]
    return da


def open_da(
    dataset: str,
    varname: str,
    resolution: str,
    period: list | tuple | Literal["all"] | int | str = "all",
    season: list | str = None,
    minlon: Optional[int | float] = None,
    maxlon: Optional[int | float] = None,
    minlat: Optional[int | float] = None,
    maxlat: Optional[int | float] = None,
    levels: int | str | list | tuple | Literal["all"] = "all",
    clim_type: str = None,
    clim_smoothing: Mapping = None,
    smoothing: Mapping = None,
) -> xr.DataArray:
    """_summary_

    Args:
        dataset (str): _description_
        varname (str): _description_
        resolution (str): Time resolution like '6H' -> Change to data_type (str), '6H_p' ? To accomodate for isentrope data ?
        period (list | tuple | Literal[&quot;all&quot;] | int | str, optional): _description_. Defaults to "all".
        season (list | str, optional): _description_. Defaults to None.
        minlon (Optional[int  |  float], optional): _description_. Defaults to None.
        maxlon (Optional[int  |  float], optional): _description_. Defaults to None.
        minlat (Optional[int  |  float], optional): _description_. Defaults to None.
        maxlat (Optional[int  |  float], optional): _description_. Defaults to None.
        levels (int | str | list | tuple | Literal[&quot;all&quot;], optional): _description_. Defaults to "all".
        clim_type (str, optional): _description_. Defaults to None.
        clim_smoothing (Mapping, optional): _description_. Defaults to None.
        smoothing (Mapping, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        xr.DataArray: _description_
    """
    path = data_path(
        dataset, varname, resolution, clim_type, clim_smoothing, smoothing, False
    )
    file_structure = determine_file_structure(path)

    if isinstance(period, tuple):
        period = np.arange(int(period[0]), int(period[1] + 1))
    elif isinstance(period, list):
        period = np.asarray(period).astype(int)
    elif period == "all":
        period = YEARSPL_EXT
    elif isinstance(period, int | str):
        period = [int(period)]

    if file_structure == "one_file":
        files_to_load = [path.joinpath("full.nc")]
    elif file_structure == "yearly":
        files_to_load = [path.joinpath(f"{year}.nc") for year in period]
    elif file_structure == "monthly":
        files_to_load = [
            path.joinpath(f"{year}{month}.nc")
            for month in range(1, 13)
            for year in period
        ]

    files_to_load = [fn for fn in files_to_load if fn.is_file()]

    da = _open_dataarray(files_to_load, varname)
    da = da.rename(varname)
    da = rename_coords(da)
    if np.diff(da.lat.values)[0] < 0:
        da = da.reindex(lat=da.lat[::-1])
    if all([bound is not None for bound in [minlon, maxlon, minlat, maxlat]]):
        da = da.sel(lon=slice(minlon, maxlon + 0.1), lat=slice(minlat, maxlat + 0.1))

    if (file_structure == "one_file") and (period != "all"):
        da = da.isel(time=np.isin(da.time.dt.year, period))

    if isinstance(season, list):
        da.isel(time=np.isin(da.time.dt.month, season))
    elif isinstance(season, str):
        if season in ["DJF", "MAM", "JJA", "SON"]:
            da = da.isel(time=da.time.dt.season == season)
        else:
            print(f"Wrong season specifier : {season} is not a valid xarray season")
            raise ValueError
        
    if 'lev' in da.dims:
        da = extract_levels(da, levels)
        
    if clim_type is not None or smoothing is None:
        return da
    
    return smooth(da, smoothing)


def compute_all_smoothed_anomalies(
    dataset: str,
    varname: str,
    resolution: str,
    clim_type: str = None,
    clim_smoothing: Mapping = None,
    smoothing: Mapping = None,
) -> None:
    path, clim_path, anom_path = data_path(
        dataset, varname, resolution, clim_type, clim_smoothing, smoothing, True
    )
    anom_path.mkdir(parents=True, exist_ok=True)

    dest_clim = clim_path.joinpath("clim.nc")
    dests_anom = [
        anom_path.joinpath(fn.name) for fn in path.iterdir() if fn.suffix == ".nc"
    ]
    if dest_clim.is_file() and all([dest_anom.is_file() for dest_anom in dests_anom]):
        return

    sources = [source for source in path.iterdir() if source.is_file()]
    
    if clim_type is None:
        for source, dest in tqdm(zip(sources, dests_anom), total=len(dests_anom)):
            if dest.is_file():
                continue
            anom = rename_coords(_open_dataarray(source, varname))
            anom = smooth(anom, smoothing).compute(**COMPUTE_KWARGS)
            anom.to_netcdf(dest)
        return
    if clim_type.lower() in ["doy", "dayofyear"]:
        coordname = "dayofyear"
    elif clim_type.lower() in ["month", "monthly"]:
        coordname = "month"
    else:
        raise NotImplementedError()
    if dest_clim.is_file():
        clim = xr.open_dataarray(dest_clim)
    else:
        da = open_da(
            dataset, varname, resolution, period="all", levels="all"
        )
        gb = da.groupby(f"time.{coordname}")
        with ProgressBar():
            clim = gb.mean(dim="time").compute(**COMPUTE_KWARGS)
        clim = smooth(clim, clim_smoothing)
        clim.to_netcdf(dest_clim)
    if len(sources) > 1:
        iterator_ = tqdm(zip(sources, dests_anom), total=len(dests_anom))
    else:
        iterator_ = zip(sources, dests_anom)
    for source, dest in iterator_:
        anom = rename_coords(_open_dataarray(source, varname))
        this_gb = anom.groupby(f"time.{coordname}")
        anom = (this_gb - clim).reset_coords(coordname, drop=True)
        if smoothing is not None :
            anom = smooth(anom, smoothing)
        if len(sources) > 1:
            anom = anom.compute(**COMPUTE_KWARGS)
        else:
            with ResourceProfiler() as a, ProgressBar() as b:
                anom = anom.compute(**COMPUTE_KWARGS)
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