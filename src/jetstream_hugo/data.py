from typing import Union, Optional, Mapping, Sequence, Tuple, Literal
from dask.distributed import Client
from nptyping import NDArray
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import flox.xarray
import xrft
from tqdm import tqdm
from dask.diagnostics import ProgressBar, ResourceProfiler
from jetstream_hugo.definitions import (
    N_WORKERS,
    DEFAULT_VARNAME,
    DATADIR,
    CLIMSTOR,
    YEARSPL_EXT,
    COMPUTE_KWARGS
)
    
    
def get_land_mask() -> xr.DataArray:
    mask = xr.open_dataarray(f"{DATADIR}/ERA5/land_sea.nc")
    mask = mask.squeeze().rename(longitude="lon", latitude="lat").reset_coords("time", drop=True)
    return mask.astype(bool)
    
    
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
    level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
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


def extract_levels(da: xr.DataArray, levels: int | str | list | tuple | Literal["all"]):
    if levels == "all" or (isinstance(levels, Sequence) and "all" in levels):
        return da.squeeze()

    levels, level_names = unpack_levels(levels)

    if ~any([isinstance(level, tuple) for level in levels]):
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


def extract_season(da: xr.DataArray | xr.Dataset, season: list | str) -> xr.DataArray | xr.Dataset:
    if isinstance(season, list):
        da = da.isel(time=np.isin(da.time.dt.month, season))
    elif isinstance(season, str):
        if season in ["DJF", "MAM", "JJA", "SON"]:
            da = da.isel(time=da.time.dt.season == season)
        else:
            print(f"Wrong season specifier : {season} is not a valid xarray season")
            raise ValueError
    return da


def pad_wrap(da: xr.DataArray, dim: str) -> bool:
    resolution = da[dim][1] - da[dim][0]
    if dim in ["lon", "longitude"]:
        return (
                360 >= da[dim][-1] >= 360 - resolution and da[dim][0] == 0.0
        )
    return dim == "dayofyear"


def _window_smoothing(da: xr.DataArray, dim: str, winsize: int, center: bool=True) -> xr.DataArray:
    if dim != "hourofyear":
        return da.rolling({dim: winsize}, center=True, min_periods=1).mean()
    groups = da.groupby(da.hourofyear % 24)
    to_concat = []
    winsize = winsize // len(groups)
    if "time" in da.dims:
        dim = "time"
    for group in groups.groups.values():
        to_concat.append(da[group].rolling({dim: winsize // 4}, center=center, min_periods=1).mean())
    return xr.concat(to_concat, dim=dim).sortby(dim)


def window_smoothing(da: xr.DataArray, dim: str, winsize: int, center: bool=True) -> xr.DataArray:
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
    ft = xrft.fft(da, dim=dim)
    mask = 0
    for dim_ in dim:
        mask = mask + np.abs(ft[f"freq_{dim_}"])
    mask = mask < winsize
    ft = ft.where(mask, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        newda = xrft.ifft(ft, dim=[f"freq_{dim_}" for dim_ in dim]).real.assign_coords(da.coords).rename(name)
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
    return da


def open_da(
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
    levels: int | str | list | tuple | Literal["all"] = "all",
    clim_type: str = None,
    clim_smoothing: Mapping = None,
    smoothing: Mapping = None,
) -> xr.DataArray:
    """_summary_

    Args:
        dataset (str): _description_
        varname (str): _description_
        resolution (str): Time resolution like '6H' -> Change to data_type (str), '6H_p' ?
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
        dataset, level_type, varname, resolution, clim_type, clim_smoothing, smoothing, False
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

    files_to_load = []

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
    if season is not None:
        da = extract_season(da, season)
        
    if "lev" in da.dims and levels != "all":
        da = extract_levels(da, levels)
    elif "lev" in da.dims:
        da = da.chunk({"lev": 1})
        
    if clim_type is not None or smoothing is None:
        return da
    
    return smooth(da, smoothing)


def compute_hourofyear(da: xr.DataArray) -> xr.DataArray:
    return da.time.dt.hour + 24 * (da.time.dt.dayofyear - 1)


def assign_clim_coord(da: xr.DataArray, clim_type: str):
    if clim_type.lower() == "hourofyear":
        da = da.assign_coords(hourofyear=compute_hourofyear(da))
        coord = da.hourofyear
    elif clim_type.lower() in [att for att in dir(da.time.dt) if not att.startswith("_")]:
        coord = getattr(da.time.dt, clim_type)
    else:
        raise NotImplementedError
    return da, coord


def compute_clim(da: xr.DataArray, clim_type: str) -> xr.DataArray:
    da, coord = assign_clim_coord(da, clim_type)
    try:
        da = da.chunk({"lev": 1})
    except ValueError:
        pass
    da = da.chunk({"time": 100, "lon": -1, "lat": -1})
    with Client(**COMPUTE_KWARGS):
        clim = da.groupby(coord).mean(method="cohorts", engine="flox").compute()
    return clim
    

def compute_anom(anom: xr.DataArray, clim: xr.DataArray, clim_type: str, normalized: bool = False):
    anom, coord = assign_clim_coord(anom, clim_type)
    this_gb = anom.groupby(coord)
    if not normalized:
        with ProgressBar():
            anom = (this_gb - clim)
    else:
        anom  = ((this_gb - clim) / anom)
        anom = anom.where((anom != np.nan) & (anom != np.inf) & (anom != -np.inf), 0)
    with ProgressBar():
        anom = anom.compute(**COMPUTE_KWARGS)
    return anom.reset_coords(clim_type, drop=True)
    

def compute_all_smoothed_anomalies(
    dataset: str,
    level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
    varname: str,
    resolution: str,
    clim_type: str = None,
    clim_smoothing: Mapping = None,
    smoothing: Mapping = None,
) -> None:
    path, clim_path, anom_path = data_path(
        dataset, level_type, varname, resolution, clim_type, clim_smoothing, smoothing, True
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
    da = open_da(
        dataset, level_type, varname, resolution, period="all", levels="all"
    )
    if dest_clim.is_file():
        clim = xr.open_dataarray(dest_clim)
    else:
        clim = compute_clim(da, clim_type)
        clim = smooth(clim, clim_smoothing)
        clim.to_netcdf(dest_clim)
    if len(sources) > 1:
        iterator_ = tqdm(zip(sources, dests_anom), total=len(dests_anom))
    else:
        iterator_ = zip(sources, dests_anom)
    for source, dest in iterator_:
        anom = rename_coords(_open_dataarray(source, varname))
        anom = compute_anom(anom, clim, clim_type, False)
        if smoothing is not None:
            anom = smooth(anom, smoothing)
        if len(sources) > 1:
            anom = anom.compute(**COMPUTE_KWARGS)
        else:
            with ResourceProfiler(), ProgressBar():
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


def open_pvs(da_template: xr.DataArray, q: float = 0.9) -> Tuple[xr.DataArray]:
    ofile = Path(f"{DATADIR}/ERA5/plev/pvs/6H")
    ofile1 = ofile.joinpath("full.nc")
    ofile2 = ofile.joinpath("anom.nc")
    ofile3 = ofile.joinpath("anom_normd.nc")
    try:
        da_pvs = xr.open_dataarray(ofile1).load()
        da_pvs_anom = xr.open_dataarray(ofile2).load()
        da_pvs_anom_normd = xr.open_dataarray(ofile3).load()
    except FileNotFoundError:
        print("Events to xarray")
        events = gpd.read_parquet(f"{DATADIR}/ERA5/RWB_index/era5_pv_streamers_350K_1959-2022.parquet")
        events = events[events.event_area >= events.event_area.quantile(q)]
        events = events[np.isin(events.date.dt.month, [6, 7, 8])]
        mask_anti = events.intensity >= 0
        mask_cycl = events.intensity < 0
        mask_tropo = events.mean_var < events.level
        events["flag"] = events.index
        events_anti = events[mask_anti & mask_tropo]
        events_cycl = events[mask_cycl & mask_tropo]
        from wavebreaking import to_xarray
        da_template = da_template.sel(time=da_template.time.dt.year>=1959)
        da_pvs_anti = to_xarray(da_template, events_anti, flag="flag")
        da_pvs_cycl = to_xarray(da_template, events_cycl, flag="flag")
        da_pvs = {"anti": da_pvs_anti, "cycl": da_pvs_cycl}
        da_pvs = xr.Dataset(da_pvs)
        da_pvs = da_pvs.to_array(dim="type").transpose("time", "type", "lat", "lon").chunk({"lon": 10, "lat": 10})
        clim = compute_clim(da_pvs, "hourofyear")
        da_pvs_anom = compute_anom(da_pvs, clim, "hourofyear")
        da_pvs_anom_normd = compute_anom(da_pvs, clim, "hourofyear", True) 
        return da_pvs, da_pvs_anom, da_pvs_anom_normd
    #     da_pvs.astype(int).to_netcdf(ofile1)
    #     da_pvs_anom.astype(np.float32).to_netcdf(ofile2)
    #     da_pvs_anom_normd.astype(np.float32).to_netcdf(ofile3)
    # return da_pvs, da_pvs_anom, da_pvs_anom_normd


def get_nao(interp_like: xr.DataArray | xr.Dataset | None = None) -> xr.DataArray:
    df = pd.read_csv(f"{DATADIR}/ERA5/daily_nao.csv", delimiter=",")
    index = pd.to_datetime(df.iloc[:, :3])
    series = xr.DataArray(df.iloc[:, 3].values, coords={"time": index})
    if interp_like is None:
        return series
    return series.interp_like(interp_like)