import warnings
from typing import Union, Optional, Mapping, Sequence, Tuple, Literal
from nptyping import NDArray
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import flox.xarray
import xrft
from tqdm import tqdm
from deepdiff import DeepHash
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from jetstream_hugo.definitions import (
    DEFAULT_VARNAME,
    DATADIR,
    YEARS,
    COMPUTE_KWARGS,
    get_region,
    save_pickle,
    load_pickle,
)
SEASONS = {
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
    
    
def get_land_mask() -> xr.DataArray:
    mask = xr.open_dataarray(f"{DATADIR}/ERA5/grid_info/land_sea.nc")
    mask = mask.squeeze().rename(longitude="lon", latitude="lat").reset_coords("time", drop=True)
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


def _window_smoothing(da: xr.DataArray | xr.Dataset, dim: str, winsize: int, center: bool=True) -> xr.DataArray:
    if dim != "hourofyear":
        return da.rolling({dim: winsize}, center=True, min_periods=1).mean()
    groups = da.groupby(da.hourofyear % 24)
    to_concat = []
    winsize = winsize // len(groups)
    if "time" in da.dims:
        dim = "time"
    for group in groups.groups.values():
        to_concat.append(da.loc[{dim: da.hourofyear[group]}].rolling({dim: winsize // 4}, center=center, min_periods=1).mean())
    return xr.concat(to_concat, dim=dim).sortby(dim)


def window_smoothing(da: xr.DataArray, dim: str, winsize: int, center: bool=True) -> xr.DataArray:
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
        newda = xrft.ifft(ft, dim=[f"freq_{dim_}" for dim_ in dim]).real.assign_coords(da.coords).rename(name)
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
    level_type: Literal["plev"] | Literal["thetalev"] | Literal["2PVU"] | Literal["surf"],
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
            else:
                monthlist = np.atleast_1d(season)
            files_to_load = [
                path.joinpath(f"{year}{str(month).zfill(2)}.nc")
                for month in monthlist
                for year in period
            ]
            

    files_to_load = [fn for fn in files_to_load if fn.is_file()]

    da = _open_dataarray(files_to_load, varname)
    da = da.rename(varname)
    da = rename_coords(da)
    if "lev" in da.dims and level_type not in ["plev", "thetalev"]:
        da = da.reset_coords("lev", drop=True)
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
    with ProgressBar():
        clim = flox.xarray.xarray_reduce(
            da,
            coord,
            func="mean",
            method="cohorts",
            expected_groups=np.unique(coord.values),
        ).compute(**COMPUTE_KWARGS)
    return clim
    

def compute_anom(anom: xr.DataArray, clim: xr.DataArray, clim_type: str, normalized: bool = False):
    anom, coord = assign_clim_coord(anom, clim_type)
    this_gb = anom.groupby(coord)
    if not normalized:
        anom = (this_gb - clim)
    else:
        variab = flox.xarray.xarray_reduce(
            anom,
            coord,
            func="std",
            method="cohorts",
            expected_groups=np.unique(coord.values),
        )
        anom = ((this_gb - clim).groupby(coord) / variab).reset_coords("hourofyear", drop=True)
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
        dataset, level_type, varname, resolution, clim_type, clim_smoothing, smoothing, True
    )
    anom_path.mkdir(parents=True, exist_ok=True)

    dest_clim = clim_path.joinpath("clim.nc")
    dests_anom = [
        anom_path.joinpath(fn.name) for fn in path.iterdir() if fn.suffix == ".nc"
    ]
    if dest_clim.is_file() and all([dest_anom.is_file() for dest_anom in dests_anom]):
        return

    sources = [source for source in path.iterdir() if source.is_file() and source.suffix == ".nc"]
    
    if clim_type is None:
        for source, dest in tqdm(zip(sources, dests_anom), total=len(dests_anom)):
            if dest.is_file():
                continue
            anom = rename_coords(_open_dataarray(source, varname))
            anom = smooth(anom, smoothing).compute(**COMPUTE_KWARGS)
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
        clim.to_netcdf(dest_clim)
    if len(sources) > 1:
        iterator_ = tqdm(zip(sources, dests_anom), total=len(dests_anom))
    else:
        iterator_ = zip(sources, dests_anom)
    for source, dest in iterator_:
        anom = rename_coords(xr.open_dataarray(source))
        anom = compute_anom(anom, clim, clim_type, False)
        if smoothing is not None:
            anom = smooth(anom, smoothing)
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


def get_nao(interp_like: xr.DataArray | xr.Dataset | None = None) -> xr.DataArray:
    df = pd.read_csv(f"{DATADIR}/ERA5/daily_nao.csv", delimiter=",")
    index = pd.to_datetime(df.iloc[:, :3])
    series = xr.DataArray(df.iloc[:, 3].values, coords={"time": index})
    if interp_like is None:
        return series
    return series.interp_like(interp_like)


def compute_extreme_climatology(da: xr.DataArray, opath: Path):
    q = da.quantile(np.arange(60, 100) / 100, dim=["lon", "lat"])
    q_clim = compute_clim(q, "dayofyear")
    q_clim = smooth(q_clim, {"dayofyear": ("win", 60)})
    q_clim.to_netcdf(opath)
    
    
def compute_anomalies_ds(ds: xr.Dataset, clim_type: str, normalized: bool = False, return_clim: bool = False) -> xr.Dataset:
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
        return ((this_gb - clim).groupby(coord) / variab).reset_coords(clim_type, drop=True), clim
    return ((this_gb - clim).groupby(coord) / variab).reset_coords(clim_type, drop=True)


def find_spot(basepath: Path, metadata: Mapping) -> Path:
    found = False
    for dir in basepath.iterdir():
        if not dir.is_dir():
            continue
        try:
            other_mda = load_pickle(dir.joinpath("metadata.pkl"))
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
    

class DataHandlerBase(object):
    def __init__(
        self,
        da: xr.DataArray,
        path: Path,
    ) -> None:
        self.da = da
        self.path = path
        self._setup_dims()
        self._extract_metadata()
        
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
            np.prod([len(dim) for dim in self.feature_dims.values()])
        )
        
    def _extract_metadata(self):
        period = np.unique(self.da.time.dt.year)
        season = np.unique(self.da.time.dt.month)
        nullseason = {None: list(range(1, 13))}
        for seasonname, monthlist in (SEASONS | nullseason).items():
            if monthlist == season:
                season = seasonname
                break
        region = get_region(self.da)
        if "lev" in self.da.dims:
            levels = self.da.lev.values
        else:
            levels = "all"
        self.metadata = {
            "period": period,
            "season": season,
            "region": region,
            "levels": levels,
        }
        
    def get_path(self) -> Path:
        return self.path
        
    def get_da(self, load: bool = False) -> xr.DataArray:
        if load:
            try:
                self.da = self.da.load()
            except AttributeError:
                pass
        return self.da
    
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
    

class DataHandler(DataHandlerBase):
    def __init__(
        self,
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
    ) -> None:
        self.path = data_path(
            dataset, level_type, varname, resolution, clim_type, clim_smoothing, smoothing, False
        ).joinpath("results")
        self.path.mkdir(exist_ok=True)
        if level_type == "surf":
            levels = "all"
        self.open_da_args = (
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

        self.varname = varname
        self.region = (minlon, maxlon, minlat, maxlat)
        self.clim_type = clim_type
        if levels != 'all':
            self.levels, self.level_names = unpack_levels(levels)
        else: 
            self.levels = 'all'

        self.metadata = {
            "period": period,
            "season": season,
            "region": (minlon, maxlon, minlat, maxlat),
            "levels": self.levels,
        }
        
        self.path = find_spot(self.path, self.metadata)

        da_path = self.path.joinpath("da.nc")
        if da_path.is_file():
            self.da = xr.open_dataarray(da_path, chunks="auto")
        else:
            self.da = open_da(*self.open_da_args)
            with ProgressBar():
                self.da = self.da.load()
            self.da.to_netcdf(da_path, format="NETCDF4")

        if reduce_da:
            try:
                self.da = self.da.isel(lev=self.da.argmax("lev"))
            except ValueError:
                pass
            
        self._setup_dims()