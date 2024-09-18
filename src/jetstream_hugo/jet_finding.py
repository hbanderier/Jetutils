import warnings
from pathlib import Path
from os.path import commonpath
from functools import partial, wraps
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Literal
from nptyping import NDArray
from multiprocessing import Pool
from itertools import combinations, product

import numpy as np
import pandas as pd
from dask.distributed import progress
import xarray as xr
from scipy.stats import linregress
from contourpy import contour_generator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from tqdm.autonotebook import tqdm, trange
from numba import njit, prange

from jetstream_hugo.definitions import (
    COMPUTE_KWARGS,
    N_WORKERS,
    OMEGA,
    RADIUS,
    coarsen_da,
    degsin,
    degcos,
    labels_to_mask,
    to_zero_one,
    slice_1d,
    get_runs_fill_holes,
    save_pickle,
    load_pickle,
    _compute,
)
from jetstream_hugo.data import (
    flatten_by,
    compute_extreme_climatology,
    smooth,
    DataHandler,
)
from jetstream_hugo.clustering import Experiment


DIRECTION_THRESHOLD = 0.33
SMOOTHING = 0.15

@njit
def distance(x1: float, x2: float, y1: float, y2: float) -> float:
    dx = x2 - x1
    if np.abs(dx) > 180:
        dx = 360 - np.abs(dx)  # sign is irrelevant
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2)


@njit(parallel=False)
def my_pairwise(X1: NDArray, X2: Optional[NDArray] = None) -> NDArray:
    x1 = X1[:, 0]
    y1 = X1[:, 1]  
    half = False
    if X2 is None:
        X2 = X1
        half = True
    x2 = X2[:, 0]
    y2 = X2[:, 1]
    output = np.zeros((len(X1), len(X2)))
    for i in prange(X1.shape[0] - int(half)):
        if half:
            for j in range(i + 1, X1.shape[0]):
                output[i, j] = distance(x1[j], x2[i], y1[j], y2[i])
                output[j, i] = output[i, j]
        else:
            for j in range(X2.shape[0]):
                output[i, j] = distance(x1[j], x2[i], y1[j], y2[i])
    return output


def preprocess(ds: xr.Dataset, smooth_s: float = None) -> xr.Dataset:
    ds = flatten_by(ds, "s")
    if (ds.lon[1] - ds.lon[0]) <= 0.75:
        ds = coarsen_da(ds, 1.5)
    if smooth_s is not None:
        for var in ["u", "v", "s"]:
            ds[var] = smooth(ds[var], smooth_map={"lon+lat": ("fft", smooth_s)})
    ds = ds.assign_coords(
        {
            "x": np.radians(ds["lon"]) * RADIUS,
            "y": RADIUS
            * np.log(
                (1 + np.sin(np.radians(ds["lat"])) / np.cos(np.radians(ds["lat"])))
            ),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds["sigma"] = (
            ds["u"] * ds["s"].differentiate("y") - ds["v"] * ds["s"].differentiate("x")
        ) / ds["s"]
    fft_smoothing = 1.0 if ds["sigma"].min() < -0.0001 else 0.8
    ds["sigma"] = smooth(ds["sigma"], smooth_map={"lon+lat": ("fft", fft_smoothing)})
    return ds.reset_coords(["x", "y"], drop=True)


def interp_xy_ds(ds: xr.Dataset, xy: NDArray) -> xr.Dataset:
    take_from = ["lon", "lat"]
    for optional_ in ["lev", "theta", "P"]:
        if optional_ in ds:
            take_from.append(optional_)
    take_from.extend(["u", "v", "s"])
    lon, lat = ds.lon.values, ds.lat.values
    indexers = {
        "lon": np.clip(xy[:, 0], lon.min(), lon.max()),
        "lat": np.clip(xy[:, 1], lat.min(), lat.max()),
    }
    group = slice_1d(ds, indexers)
    group = group.reset_coords(["lon", "lat"])
    if group["lat"].isnull().any().item():
        print("NaNs")
    return group


def compute_alignment(group: xr.Dataset) -> xr.Dataset:
    dgdp = group.differentiate("points")
    dgdp["ds"] = np.sqrt(dgdp["lon"] ** 2 + dgdp["lat"] ** 2)
    dgdp["align_x"] = group["u"] / group["s"] * dgdp["lon"] / dgdp["ds"]
    dgdp["align_y"] = group["v"] / group["s"] * dgdp["lat"] / dgdp["ds"]
    group["alignment"] = dgdp["align_x"] + dgdp["align_y"]
    return group


@njit
def haversine(lon1: NDArray, lat1: NDArray, lon2: NDArray, lat2: NDArray) -> NDArray:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return RADIUS * c


@njit
def haversine_from_dl(lat: NDArray, dlon: NDArray, dlat: NDArray) -> NDArray:
    lat, dlon, dlat = map(np.radians, [lat, dlon, dlat])
    a = (
        np.sin(dlat / 2.0) ** 2 * np.cos(dlon / 2) ** 2
        + np.cos(lat) ** 2 * np.sin(dlon / 2) ** 2
    )
    return 2 * RADIUS * np.arcsin(np.sqrt(a))


@njit
def jet_integral_haversine(jet: NDArray, x_is_one: bool = False):
    X = jet[:, :2]
    ds = haversine(X[:-1, 0], X[:-1, 1], X[1:, 0], X[1:, 1])
    ds = np.append([0], ds)
    if x_is_one:
        return np.trapz(np.ones(len(ds)), x=np.cumsum(ds))
    return np.trapz(jet[:, 2], x=np.cumsum(ds))


def jet_integral_lon(jet: NDArray) -> float:
    return np.trapz(jet[:, 2], dx=np.mean(np.abs(np.diff(jet[:, 0]))))


def jet_integral_flat(jet: NDArray) -> float:
    path = np.append(0, np.sqrt(np.diff(jet[:, 0]) ** 2 + np.diff(jet[:, 1]) ** 2))
    return np.trapz(jet[:, 2], x=np.cumsum(path))


def find_jets(
    ds: xr.Dataset,
    wind_threshold: float = 23,
    jet_threshold: float = 1.0e8,
    alignment_threshold: float = 0.3,
    mean_alignment_threshold: float = 0.7,
    smooth_s: float = 0.3,
    hole_size: int = 1,
):
    ds = preprocess(ds, smooth_s=smooth_s)
    lon, lat = ds.lon.values, ds.lat.values
    dx = lon[1] - lon[0]
    contours, types = contour_generator(
        x=lon, y=lat, z=ds["sigma"].values, line_type="SeparateCode", quad_as_tri=False
    ).lines(0.0)
    groups = []
    for contour, types_ in zip(contours, types):
        if len(contour) < 15:
            continue
        cyclic: bool = 79 in types_
        group = interp_xy_ds(ds, contour[::-1])
        group = compute_alignment(group)
        mask = (group["alignment"] > alignment_threshold) & (
            group["s"].values > wind_threshold
        )
        mask = mask.values
        indicess = get_runs_fill_holes(mask, hole_size=hole_size, cyclic=cyclic)
        for indices in indicess:
            indices = np.unique(indices)
            if len(indices) < 15:
                continue
            group_df = group.to_dataframe()
            for potential_to_drop in ["time", "member", ["ratio", "label"]]:
                try:
                    group_df = group_df.drop(columns=potential_to_drop)
                except KeyError:
                    pass
            group_df = group_df.iloc[indices]
            group_ = group_df[["lon", "lat"]].values.astype(np.float32)
            labels = (
                AgglomerativeClustering(
                    n_clusters=None, distance_threshold=dx * 1.9, linkage="single"
                )
                .fit(group_)
                .labels_
            )
            masks = labels_to_mask(labels)
            for mask in masks.T:
                groups.append(group_df.iloc[mask])
    jets = []
    for group_df in groups:
        bigjump = np.diff(group_df["lon"]) < -3 * dx
        if any(bigjump):
            here = np.where(bigjump)[0][0] + 1
            group_df = group_df.apply(np.roll, args=(-here,), raw=True)
        group_ = group_df[["lon", "lat", "s"]].values.astype(np.float32)
        jet_int = jet_integral_haversine(group_)
        mean_alignment = np.mean(group_df["alignment"].values)
        if jet_int > jet_threshold and mean_alignment > mean_alignment_threshold:
            jets.append(group_df)
    return jets


def all_jets_to_one_df(all_jets: list, **extra_dims):
    outer_names = list(extra_dims)
    multiindex = list(product(*list(extra_dims.values())))
    daily = []
    multiindex_to_keep = []
    names = ["jet ID", "orig_points"]

    for jets, mind in zip(all_jets, multiindex):
        jets = [jet.reset_index() for jet in jets]
        keys = [f"{i}" for i in range(len(jets))]
        if len(jets) > 0:
            daily.append(pd.concat(jets, keys=keys, names=names))
            multiindex_to_keep.append(mind)
    return pd.concat(daily, keys=multiindex_to_keep, names=outer_names)


def inner_find_all_jets(ds_block, basepath: Path, **kwargs):
    extra_dims = {}
    for potential in ["member", "time", "cluster"]:
        if potential in ds_block.dims:
            extra_dims[potential] = ds_block[potential].values
    # len_ = np.prod([len(co) for co in extra_dims.values()])
    iter_ = list(product(*list(extra_dims.values())))
    unique_hash = hash(tuple(iter_))
    opath = basepath.joinpath(f"jets/j{unique_hash}.pkl")
    if opath.is_file():
        return ds_block.isel(lon=0, lat=0).reset_coords(drop=True)
    all_jets = []
    for vals in iter_:
        indexer = {dim: val for dim, val in zip(extra_dims, vals)}
        this_ds = ds_block.loc[indexer]
        if "threshold" in this_ds:
            kwargs["wind_threshold"] = this_ds["threshold"].item()
            kwargs["jet_threshold"] = kwargs["wind_threshold"] / 23 * 1e8
        these_jets = find_jets(this_ds, **kwargs)
        all_jets.append(these_jets)
    all_jets_one_df = all_jets_to_one_df(all_jets, **extra_dims)
    
    save_pickle(all_jets_one_df, opath)
    return ds_block.isel(lon=0, lat=0).reset_coords(drop=True)


def find_all_jets(ds, basepath: Path, threshold: xr.DataArray | None = None, **kwargs):
    jets_path = basepath.joinpath("jets")
    jets_path.mkdir(exist_ok=True, mode=0o777)
    template = ds.isel(lon=0, lat=0).reset_coords(drop=True)
    if threshold is not None:
        ds["threshold"] = ("time", threshold.data)
    try:
        to_comp = ds.map_blocks(inner_find_all_jets, template=template, kwargs=dict(basepath=basepath, **kwargs)).persist() 
        progress(to_comp, notebook=False)
        to_comp.compute()
    except ValueError:
        print("No Dask client found, reverting to sequential")
        _ = inner_find_all_jets(ds, basepath=basepath, **kwargs)
    dfs = []
    for f in basepath.joinpath("jets").glob("*.pkl"):
        dfs.append(load_pickle(f))
    df = pd.concat(dfs).sort_index(level=list(dfs[0].index.names))
    return df


def _compute_jet_width_one_side(
    da: xr.DataArray, normallons: NDArray, normallats: NDArray, slice_: slice
) -> float:
    normal_s = slice_1d(
        da, {"lon": normallons[slice_], "lat": normallats[slice_]}
    ).values
    normal_s = np.concatenate([normal_s, [0]])
    s = normal_s[0]
    stop = np.argmax(normal_s <= max(s / 2, 25))
    try:
        endlo = normallons[slice_][stop]
        endla = normallats[slice_][stop]
    except IndexError:
        endlo = normallons[slice_][-1]
        endla = normallats[slice_][-1]
    return haversine(normallons[slice_][0], normallats[slice_][0], endlo, endla)


def compute_jet_width(jet: pd.DataFrame, da: xr.DataArray) -> xr.DataArray:
    lon, lat = da.lon.values, da.lat.values
    lo, la, s = jet[["lon", "lat", "s"]].to_numpy().T
    dxds = np.gradient(lo)
    dyds = np.gradient(la)
    theta = np.arctan2(dyds, dxds)
    dn = 0.5
    t = np.arange(-12, 12 + dn, dn)
    half_length = len(t) // 2
    widths = np.zeros(len(jet))
    for k in range(len(jet)):
        normallons = np.cos(theta[k] + np.pi / 2) * t + lo[k]
        normallats = np.sin(theta[k] + np.pi / 2) * t + la[k]
        mask_valid = (
            (normallons >= lon.min())
            & (normallons <= lon.max())
            & (normallats >= lat.min())
            & (normallats <= lat.max())
        )
        if all(mask_valid):
            slice_ = slice(half_length, 0, -1)
            width1 = _compute_jet_width_one_side(da, normallons, normallats, slice_)
            slice_ = slice(half_length + 1, len(t))
            width2 = _compute_jet_width_one_side(da, normallons, normallats, slice_)
            widths[k] = 2 * min(width1, width2)
        elif np.mean(mask_valid[:half_length]) > np.mean(mask_valid[half_length + 1 :]):
            slice_ = slice(half_length, 0, -1)
            widths[k] = 2 * _compute_jet_width_one_side(
                da, normallons, normallats, slice_
            )
        else:
            slice_ = slice(half_length + 1, -1)
            widths[k] = 2 * _compute_jet_width_one_side(
                da, normallons, normallats, slice_
            )
    return np.average(widths, weights=s)


def compute_jet_props(jet: pd.DataFrame, da: xr.DataArray) -> dict:
    jet_numpy = jet[["lon", "lat", "s"]].to_numpy()
    x, y, s = jet_numpy.T
    dic = {}
    for optional_ in ["lon", "lat", "lev", "P", "theta"]:
        if optional_ in jet:
            dic[f"mean_{optional_}"] = np.average(jet[optional_].to_numpy(), weights=s)
    dic["mean_spe"] = np.mean(s)
    dic["is_polar"] = dic["mean_lat"] - 0.4 * dic["mean_lon"] > 40
    maxind = np.argmax(s)
    dic["lon_star"] = x[maxind]
    dic["lat_star"] = y[maxind]
    dic["spe_star"] = s[maxind]
    dic["lon_ext"] = np.amax(x) - np.amin(x)
    dic["lat_ext"] = np.amax(y) - np.amin(y)
    slope, _, r_value, _, _ = linregress(x, y)
    dic["tilt"] = slope
    dic["waviness1"] = 1 - r_value**2
    dic["waviness2"] = np.sum((y - dic["mean_lat"]) ** 2)
    sorted_order = np.argsort(x)
    dic["wavinessR16"] = np.sum(np.abs(np.diff(y[sorted_order]))) / dic["lon_ext"]
    dic["wavinessDC16"] = (
        jet_integral_haversine(jet_numpy, x_is_one=True)
        / RADIUS
        * degcos(dic["mean_lat"])
    )
    dic["wavinessFV15"] = np.average(
        (jet["v"] - jet["v"].mean()) * np.abs(jet["v"]) / jet["s"] / jet["s"], weights=s
    )
    dic["width"] = compute_jet_width(jet[::5], da)
    if dic["width"] > 1e7:
        dic["width"] = 0.0
    try:
        dic["int_over_europe"] = jet_integral_haversine(jet_numpy[x > -10])
    except ValueError:
        dic["int_over_europe"] = 0
    dic["int"] = jet_integral_haversine(jet_numpy)
    return dic


def compute_jet_props_wrapper(args: Tuple) -> list:
    jets, da = args
    jets = jets.droplevel(level=0)
    props = []
    for _, jet in jets.groupby(level=0):
        jet = jet.droplevel(level=0)
        props.append(compute_jet_props(jet, da))
    return pd.DataFrame.from_dict(props)


def map_maybe_parallel(iterator: Iterable, func: Callable, len_: int, processes: int = N_WORKERS, chunksize: int = 100) -> list:
    if processes == 1:
        return list(
            tqdm(map(func, iterator), total=len_)
        )
    with Pool(processes=processes) as pool:
        return list(
            tqdm(
                pool.imap(func, iterator, chunksize=chunksize),
                total=len_,
            )
        )
        
def create_mappable_iterator(df: pd.DataFrame, das: Sequence | None = None, others: Sequence | None = None, potentials: Tuple = ("member", "time", "cluster")) -> Tuple:
    if das is None:
        das = []
    if others is None:
        others = []
    iter_dims = []
    for potential in potentials:
        if all([potential in da.dims for da in das]) and potential in df.index.names:
            iter_dims.append(potential)
    iterator_ = df.groupby(iter_dims)
    indices = list(iterator_.indices)
    len_ = len(indices)
    iterator = ((jets, *[_compute(da.sel({dim: values for dim, values in zip(iter_dims, index)})) for da in das], *others) for index, jets in iterator_)
    return iter_dims, indices, len_, iterator


def compute_all_jet_props(
    all_jets_one_df: pd.DataFrame,
    da: xr.DataArray,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.Dataset:
    iter_dims, indices, len_, iterator = create_mappable_iterator(all_jets_one_df, [da])
    all_props_dfs = map_maybe_parallel(iterator, compute_jet_props_wrapper, len_=len_, processes=processes, chunksize=chunksize)
    print("Computing jet properties")
    all_props_df = pd.concat(all_props_dfs, keys=indices, names=[*iter_dims, "jet"])
    return xr.Dataset.from_dataframe(all_props_df)


def compute_one_wb_props(jet: pd.DataFrame, da_pvs: xr.DataArray, every: int = 5) -> dict:
    jet = jet.iloc[::every]
    lo, la = jet[["lon", "lat"]].to_numpy().T
    lon, lat = da_pvs.lon.values, da_pvs.lat.values
    dxds = np.gradient(lo)
    dyds = np.gradient(la)
    theta = np.arctan2(dyds, dxds)

    distances = np.full((len(jet), 2), fill_value=-1)
    intensities = np.zeros((len(jet), 2))
    jet_pvs = slice_1d(
        da_pvs, {"lon": lo, "lat": la}
    ).values.T
    up_or_down = np.argmax(np.abs(jet_pvs), axis=1)
    jet_pvs = np.take_along_axis(jet_pvs, up_or_down[:, None], axis=1).flatten()
    distances[np.where(jet_pvs)[0], :] = 0.
    intensities[np.where(jet_pvs)[0], :] = jet_pvs[np.where(jet_pvs)[0], None]

    dn = 2
    t_ = np.arange(dn, 20 + dn, dn)

    for k in range(len(jet)):
        if any(distances[k] != -1):
            continue 
        for l, side in enumerate([-1, 1]):
            t = side * t_
            normallons = np.cos(theta[k] + np.pi / 2) * t + lo[k]
            normallats = np.sin(theta[k] + np.pi / 2) * t + la[k]
            mask_valid = (
                (normallons >= lon.min())
                & (normallons <= lon.max())
                & (normallats >= lat.min())
                & (normallats <= lat.max())
            )
            if np.sum(mask_valid) == 0:
                continue
            normallons = normallons[mask_valid]
            normallats = normallats[mask_valid]
            normal_pvs = slice_1d(
                da_pvs, {"lon": normallons, "lat": normallats}
            ).values
            for type_ in [0, 1]:
                if not any(normal_pvs[type_]):
                    continue
                imin = np.argmax(normal_pvs[type_] != 0)
                distance = haversine(lo[k], la[k], normallons[imin], normallats[imin])
                reject_new = (distance >= distances[k, l]) and (distances[k, l] != -1)
                if reject_new:
                    continue
                distances[k, l] = distance
                intensities[k, l] = normal_pvs[type_][imin]
            
    props = {}
    props["above"] = (np.sum(np.abs(intensities[k, 0])) < np.sum(np.abs(intensities[k, 1]))).astype(np.float32)
    dists_good_direction = distances[:, int(props["above"])]
    props["affected_from"] = np.argmax(dists_good_direction != -1)
    props["affected_until"] = len(jet) - np.argmax(dists_good_direction[::-1] != -1) - 1
    slice_ = slice(props["affected_from"], props["affected_until"] + 1)
    props["mean_int"] = np.mean(intensities[slice_, int(props["above"])]).astype(np.float32)
    props["mean_dist"] = np.mean(dists_good_direction[slice_]).astype(np.float32)
    props["affected_from"] = props["affected_from"].astype(np.float32) * every
    props["affected_until"] = props["affected_until"].astype(np.float32) * every
    return props


def compute_wb_props_wrapper(args: Tuple) -> list:
    jets, da_pvs, mask = args
    jets = jets.droplevel(0)
    if not mask:
        keys = ["above", "affected_from", "affected_until", "mean_int", "mean_dist"]
        props = [{key: 0. for key in keys} for _ in jets.groupby(level=0)]
    else:
        props = []
        for _, (_, jet) in enumerate(jets.groupby(level=0)):
            props.append(compute_one_wb_props(jet, da_pvs, every=4))
    return pd.DataFrame.from_dict(props, dtype=np.float32)


def compute_all_wb_props(
    all_jets_one_df: pd.DataFrame,
    da_pvs: xr.DataArray,
    event_mask: NDArray,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.Dataset:
    iter_dims, indices, len_, iterator = create_mappable_iterator(all_jets_one_df, [da_pvs], [event_mask])
    print("Computing RWB properties")
    all_props_dfs = map_maybe_parallel(iterator, compute_wb_props_wrapper, len_=len_, processes=processes, chunksize=chunksize)
    all_props_df = pd.concat(all_props_dfs, keys=indices, names=[*iter_dims, "jet"])
    return xr.Dataset.from_dataframe(all_props_df)


def round_half(x):
    return np.round(x * 2) / 2


def do_one_int_low(args):
    jets, da_low_ = args
    jets = jets.droplevel(0)
    ints = []
    for j, (jet_name, jet) in enumerate(jets.groupby(level=0)):
        x, y = round_half(jet[["lon", "lat"]].to_numpy().T)
        x_ = xr.DataArray(x, dims="points")
        y_ = xr.DataArray(y, dims="points")
        s_low = da_low_.sel(lon=x_, lat=y_).values
        jet_low = np.asarray([x, y, s_low]).T
        ints.append(jet_integral_haversine(jet_low))
    return xr.DataArray(ints, coords={"jet": np.arange(len(ints))})


def compute_int_low( # broken with members
    all_jets_one_df: pd.DataFrame, props_as_ds: xr.Dataset, exp_low_path: Path, processes: int = N_WORKERS, chunksize: int = 100,
) -> xr.Dataset:
    this_path = exp_low_path.joinpath("int_low.nc")
    if this_path.is_file():
        props_as_ds["int_low"] = xr.open_dataarray(this_path)
        props_as_ds["int_ratio"] = props_as_ds["int_low"] / props_as_ds["int"]
        return props_as_ds
    print("computing int low")
    props_as_ds["int_low"] = props_as_ds["mean_lon"].copy()
    
    da_low = xr.open_dataarray(exp_low_path.joinpath("da.nc"))
    iter_dims, indices, len_, iterator = create_mappable_iterator(all_jets_one_df, [da_low])
    all_jet_ints = map_maybe_parallel(iterator, do_one_int_low, len_=len_, processes=processes, chunksize=chunksize)
    
    props_as_ds["int_low"] = ((*iter_dims, "jet"), np.stack([jet_ints.values for jet_ints in all_jet_ints]))
    props_as_ds["int_ratio"] = props_as_ds["int_low"] / props_as_ds["int"]
    props_as_ds["int_low"].to_netcdf(exp_low_path.joinpath("int_low.nc"))
    return props_as_ds


def is_polar_v2(props_as_ds: xr.Dataset) -> xr.Dataset:
    props_as_ds["is_polar"] = (
        props_as_ds["mean_lat"] * 200
        - props_as_ds["mean_lon"] * 30
        + props_as_ds["int_low"] / RADIUS
    ) > 9000
    return props_as_ds


def extract_features(
    props_as_ds: xr.Dataset,
    feature_names: Sequence = None,
    season: Optional[str] | Optional[list] | Optional[int] = None,
) -> Tuple[NDArray, float]:
    if season is not None:
        if isinstance(season, str):
            props_as_ds = props_as_ds.sel(time=props_as_ds.time.dt.season == season)
        elif isinstance(season, list):
            props_as_ds = props_as_ds.sel(
                time=np.isin(props_as_ds.time.dt.month, season)
            )
        elif isinstance(season, int):
            props_as_ds = props_as_ds.sel(time=props_as_ds.time.dt.month == season)
    if feature_names is None:
        feature_names = ["mean_lon", "mean_lat", "spe_star"]

    lat = props_as_ds["mean_lat"].values
    mask = ~np.isnan(lat)
    X = []
    for feature_name in feature_names:
        X.append(props_as_ds[feature_name].values[mask])
    X = np.stack(X).T
    return X, np.where(mask)


def one_gmix(X):
    X, _, _ = to_zero_one(X)
    model = GaussianMixture(
        n_components=3
    )  # to help with class imbalance, 1 for sub 2 for polar
    labels = model.fit_predict(X)
    masks = labels_to_mask(labels)
    mls = []
    for mask in masks.T:
        mls.append(X[mask, 0].mean())
    return labels != np.argmin(mls)


def is_polar_gmix(
    props_as_ds: xr.Dataset,
    feature_names: list,
    mode: Literal["year"] | Literal["season"] | Literal["month"] = "year",
) -> xr.Dataset:
    better_is_polar = props_as_ds["mean_lat"].copy()
    if mode == "year":
        X, where = extract_features(props_as_ds, feature_names, None)
        labels = one_gmix(X)
        better_is_polar[
            xr.DataArray(where[0], dims="points"), xr.DataArray(where[1], dims="points")
        ] = labels
    elif mode == "season":
        for season in ["DJF", "MAM", "JJA", "SON"]:
            X, where = extract_features(props_as_ds, feature_names, season)
            season_indices = np.where(better_is_polar.time.dt.season == season)[0]
            labels = one_gmix(X)
            better_is_polar[
                xr.DataArray(season_indices[where[0]], dims="points"),
                xr.DataArray(where[1], dims="points"),
            ] = labels
    elif mode == "month":
        for month in range(1, 13):
            X, where = extract_features(props_as_ds, feature_names, month)
            month_indices = np.where(better_is_polar.time.dt.month == month)[0]
            labels = one_gmix(X)
            better_is_polar[
                xr.DataArray(month_indices[where[0]], dims="points"),
                xr.DataArray(where[1], dims="points"),
            ] = labels
    props_as_ds["is_polar"] = better_is_polar
    return props_as_ds


def categorize_ds_jets(props_as_ds: xr.Dataset):
    time_name, time_val = [
        (cooname, coo)
        for (cooname, coo) in props_as_ds.coords.items()
        if cooname != "jet"
    ][
        0
    ]  # ugh
    ds = xr.Dataset(coords={time_name: time_val, "jet": ["subtropical", "polar"]})
    cond = props_as_ds["is_polar"]
    subtropical_weights = props_as_ds["int"].where(1 - cond).fillna(0)
    polar_weights = props_as_ds["int"].where(cond).fillna(0)
    for varname in props_as_ds.data_vars:
        if varname == "is_polar":
            continue
        values = np.zeros((len(time_val), 2))
        subtropical_values = props_as_ds[varname].where(1 - cond)
        polar_values = props_as_ds[varname].where(cond)
        if varname in ["lon_ext", "lat_ext"]:
            values[:, 0] = subtropical_values.sum(dim="jet").values
            values[:, 1] = polar_values.sum(dim="jet").values
        elif varname == "spe_star":
            values[:, 0] = subtropical_values.max(dim="jet").values
            values[:, 1] = polar_values.max(dim="jet").values
        else:
            values[:, 0] = (
                subtropical_values.weighted(subtropical_weights).mean(dim="jet").values
            )
            values[:, 1] = polar_values.weighted(polar_weights).mean(dim="jet").values
        ds[varname] = ((time_name, "jet"), values)
    return ds


def jet_overlap_values(jet1: NDArray, jet2: NDArray) -> Tuple[float, float]:
    _, idx1 = np.unique(jet1[:, 0], return_index=True)
    _, idx2 = np.unique(jet2[:, 0], return_index=True)
    x1, y1 = jet1[idx1, :2].T
    x2, y2 = jet2[idx2, :2].T
    mask12 = np.isin(x1, x2)
    mask21 = np.isin(x2, x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        overlap = (np.mean(mask12) + np.mean(mask21)) / 2
        vert_dist = np.mean(np.abs(y1[mask12] - y2[mask21]))
    return overlap, vert_dist


def compute_all_overlaps(
    all_jets: list, props_as_ds: xr.Dataset
) -> Tuple[NDArray, NDArray]:
    overlaps = np.full(len(all_jets), np.nan)
    vert_dists = np.full(len(all_jets), np.nan)
    time = props_as_ds.time.values
    for i, (jets, are_polar) in enumerate(
        zip(all_jets, props_as_ds["is_polar"].values)
    ):
        nj = min(len(are_polar), len(jets))
        if nj < 2 or sum(are_polar[:nj]) == nj or sum(are_polar[:nj]) == 0:
            continue
        polars = []
        subtropicals = []
        for jet, is_polar in zip(jets, are_polar):
            if is_polar:
                polars.append(jet)
            else:
                subtropicals.append(jet)
        polars = np.concatenate(polars, axis=0)
        subtropicals = np.concatenate(subtropicals, axis=0)
        overlaps[i], vert_dists[i] = jet_overlap_values(polars, subtropicals)
    overlaps = xr.DataArray(overlaps, coords={"time": time})
    vert_dists = xr.DataArray(vert_dists, coords={"time": time})
    return overlaps, vert_dists


def overlaps_vert_dists_as_da(
    da: xr.DataArray, all_jets: list, props_as_ds_uncat: xr.Dataset, basepath: Path
) -> Tuple[xr.DataArray, xr.DataArray]:
    try:
        da_overlaps = xr.open_dataarray(basepath.joinpath("overlaps.nc"))
        da_vert_dists = xr.open_dataarray(basepath.joinpath("vert_dists.nc"))
    except FileNotFoundError:
        time, lon = da.time.values, da.lon.values
        coords = {"time": time, "lon": lon}
        da_overlaps = xr.DataArray(
            np.zeros([len(val) for val in coords.values()], dtype=np.float32),
            coords=coords,
        )
        da_overlaps[:] = np.nan
        da_vert_dists = da_overlaps.copy()

        for it, (jets, are_polar) in tqdm(
            enumerate(zip(all_jets, props_as_ds_uncat["is_polar"])), total=len(all_jets)
        ):
            nj = len(jets)
            if nj < 2:
                continue
            for jet1, jet2 in combinations(jets, 2):
                _, idx1 = np.unique(jet1[:, 0], return_index=True)
                _, idx2 = np.unique(jet2[:, 0], return_index=True)
                x1, y1, s1 = jet1[idx1, :3].T
                x2, y2, s2 = jet2[idx2, :3].T
                mask12 = np.isin(x1, x2)
                mask21 = np.isin(x2, x1)
                s_ = (s1[mask12] + s2[mask21]) / 2
                vd_ = np.abs(y1[mask12] - y2[mask21])
                x_ = xr.DataArray(x1[mask12], dims="points")
                da_overlaps.loc[time[it], x_] = s_
                da_vert_dists.loc[time[it], x_] = np.fmax(
                    da_vert_dists.loc[time[it], x_], vd_
                )
        da_overlaps.to_netcdf(basepath.joinpath("overlaps.nc"))
        da_vert_dists.to_netcdf(basepath.joinpath("vert_dists.nc"))
    return da_overlaps, da_vert_dists


@njit
def isin(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False, dtype=np.bool_)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)


def _track_jets(df, progress: bool = True):
    df = df[["lon", "lat"]].copy()
    time = np.unique(df.index.get_level_values(0))
    df.insert(2, "raw_index", np.arange(len(df)))
    time_starts = df.loc[(slice(None), "0", 0), "raw_index"].values # iloc vs loc on large dfs: 1000x speedup
    df.drop(columns="raw_index", inplace=True)
    dt = time[1] - time[0]
    df = round_half(df)
    guess_nflags = len(time)
    guess_len = 100
    all_jets_over_time = np.zeros((guess_nflags, guess_len), dtype=[("time", "datetime64[s]"), ("jet ID", "i4")])
    all_jets_over_time[:] = (np.datetime64("NaT"), -1)
    last_valid_index = np.full(guess_nflags, fill_value=guess_len, dtype="int32")
    for jet_id, jet in df.iloc[:time_starts[1]].groupby("jet ID"):
        jet_id = int(jet_id)
        all_jets_over_time[int(jet_id), 0] = (time[0], int(jet_id))
        last_valid_index[int(jet_id)] = 0
    last_flag = int(jet_id)
    flags = df.loc[:, :, 0].copy()
    flags.insert(2, "raw_index", np.arange(len(flags)))
    time_starts_flags = flags.loc[(slice(None), "0"), "raw_index"].values # iloc vs loc on large dfs: 1000x speedup
    flags.drop(columns="raw_index", inplace=True)
    flags = flags["lon"].astype(np.int32).copy().rename("flags")
    flags.iloc[:] = 0
    flags.loc[time[0], :str(jet_id)] = np.arange(last_flag + 1, dtype=np.int32)
    if progress:
        iter_ = enumerate(pbar := tqdm(df.iloc[time_starts[1]:].groupby("time"), total=len(time) - 1, leave=False))
    else:
        iter_ =  enumerate(df.iloc[time_starts[1]:].groupby("time"))
    for it, (t, jets) in iter_: 
        min_time_index = max(it - 3, 0)
        subdf = df.iloc[time_starts[min_time_index]:time_starts[it + 1]].copy()
        jets = jets.droplevel(0).copy()
        from_ = max(0, last_flag - 20)
        potentials = all_jets_over_time[from_:last_flag + 1]
        these_lvis = last_valid_index[from_:last_flag + 1]
        potentials = [potential[lvi] for potential, lvi in zip(potentials, these_lvis)]
        times_to_test = [ind[0] for ind in potentials]
        timesteps_to_check = [t - n * dt for n in range(1, 5)]
        condition = np.isin(times_to_test, timesteps_to_check)
        potentials = [ind for con, ind in zip(condition, potentials) if con]
        jets_grouped = jets.groupby(level=0)
        num_valid_jets = len(jets_grouped)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        overlaps = dist_mat.copy()
        for i, jtt_idx in enumerate(potentials):
            i = int(i)
            jet_to_try = subdf.loc[jtt_idx[0], str(jtt_idx[1]), :].values
            for j, jet in jets_grouped:
                j = int(j)
                jet = jet.values
                overlaps[i, j], dist_mat[i, j] = jet_overlap_values(jet, jet_to_try)
        try:
            dist_mat[np.isnan(dist_mat)] = np.nanmax(dist_mat) + 1
        except ValueError:
            pass
        connected_mask = (overlaps > 0.4) & (dist_mat < 15)
        flagged = np.zeros(num_valid_jets, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if flagged[j]:
                    continue
                this_flag = from_ + i
                last_valid_index[this_flag] = last_valid_index[this_flag] + 1
                all_jets_over_time[this_flag, last_valid_index[this_flag]] = (t, j)
                flagged[j] = True
                flags.iloc[time_starts_flags[it + 1] + j] = this_flag
                break
        pass
        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                last_valid_index[last_flag] = 0
                all_jets_over_time[last_flag, 0] = (t, j)
                flags.iloc[time_starts_flags[it + 1] + j] = last_flag
                flagged[j] = True
        if progress:
            pbar.set_description(f"last_flag: {last_flag}")
    ajot_df = []
    for ajot in all_jets_over_time[:last_flag + 1]:
        times = ajot["time"]   
        ajot = ajot[:np.argmax(np.isnat(times))]
        ajot_df.append(pd.DataFrame(ajot))
    ajot_df = pd.concat(ajot_df, keys=np.arange(last_flag + 1), names=["flag", "timesteps"])
    return ajot_df, flags
    
    
def track_jets(all_jets_one_df, processes: int = N_WORKERS):
    inner = ["time", "jet ID", "orig_points"]
    levels = all_jets_one_df.index.names
    outer = [level for level in levels if level not in inner]
    if len(outer) == 0:
        return _track_jets(all_jets_one_df, progress=True)
    iter_dims, indices, len_, iterator = create_mappable_iterator(all_jets_one_df, potentials=tuple(outer))
    iterator = (a[0].droplevel(iter_dims) for a in iterator)
    res = map_maybe_parallel(iterator, _track_jets, len_=len_, processes=processes, chunksize=1)
    
    ajots, all_flags = tuple(zip(*res))
    ajots = pd.concat(ajots, keys=indices, names=[*iter_dims, *ajots[0].index.names])
    all_flags = pd.concat(all_flags, keys=indices, names=[*iter_dims, *all_flags[0].index.names])
    return ajots, all_flags


def extract_props_over_time(jet_over_time: pd.DataFrame, props_as_ds):
    times = xr.DataArray(jet_over_time.loc[:, "time"].values, dims="point")
    jets = xr.DataArray(jet_over_time.loc[:, "jet ID"].values, dims="point")
    props_over_time = props_as_ds.loc[{"time": times, "jet": jets}].rename(
        jet="jet_index"
    )
    props_over_time = props_over_time.reset_coords(["time", "jet_index"], drop=False)
    props_over_time = props_over_time.assign_coords(
        point=np.arange(props_over_time.point.shape[0])
    )
    return props_over_time


def add_persistence_to_props(ds_props: xr.Dataset, flags: NDArray):
    names = ["time", "jet"]
    dt1 = (ds_props.time.values[1] - ds_props.time.values[0]).astype("timedelta64[h]")
    dt2 = np.timedelta64(1, "D")
    factor = float(dt1 / dt2)
    num_jets = ds_props["mean_lon"].shape[1]
    jet_persistence_prop = flags.loc[:, :str(num_jets - 1)].copy().astype(float).values
    unique_flags, jet_persistence = np.unique(flags, return_counts=True)
    for i, flag in enumerate(unique_flags[:-1]):
        jet_persistence_prop = jet_persistence[i] * factor
    ds_props["persistence"] = (names, jet_persistence_prop)
    return ds_props


def compute_prop_anomalies(ds_props: xr.Dataset) -> xr.Dataset:
    prop_anomalies = ds_props.copy()

    for varname in ds_props.data_vars:
        gb = ds_props[varname].groupby("time.year")
        prop_anomalies[varname] = gb - gb.mean(dim="time")
        prop_anomalies[varname] = prop_anomalies[varname] / ds_props[varname].std(
            dim="time"
        )
    return prop_anomalies


def jet_position_as_da_wrapper(args: Tuple):
    (time, jets), are_polar, da_template = args
    jets = jets.droplevel(0)
    jet_names = da_template.jet.values
    for j, (_, jet) in enumerate(jets.groupby(level=0)):
        jet = jet.droplevel(0)
        is_polar = are_polar.sel(jet=j)
        x, y, s = jet[["lon", "lat", "s"]].to_numpy().T
        x_ = xr.DataArray(round_half(x), dims="points")
        y_ = xr.DataArray(round_half(y), dims="points")
        try:
            is_polar = int(is_polar)
        except ValueError:
            continue
        da_template.loc[jet_names[int(is_polar)], y_, x_] += s
    return da_template


def jet_position_as_da(
    da_s: xr.DataArray,
    props_as_ds_uncat: xr.Dataset,
    all_jets_one_df: pd.DataFrame,
    basepath: Path,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.DataArray:
    ofile = basepath.joinpath("jet_pos.nc")
    if ofile.is_file():
        return xr.open_dataarray(ofile).load()

    jet_names = np.asarray(["subtropical", "polar"])
    time, lat, lon = (
        da_s.time.values,
        da_s.lat.values,
        da_s.lon.values,
    )

    times = all_jets_one_df.index.levels[0]
    coords = {"time": time, "jet": jet_names, "lat": lat, "lon": lon}
    da_jet_pos = xr.DataArray(
        np.zeros([len(val) for val in coords.values()], dtype=np.float32),
        coords=coords,
    )
    iterable = zip(
        all_jets_one_df.groupby(level=0),
        props_as_ds_uncat["is_polar"].sel(time=times),
        da_jet_pos.sel(time=times),
    )
    print("Jet position as da")
    if processes == 1:
        da_jet_pos = list(
            tqdm(map(jet_position_as_da_wrapper, iterable), total=len(times))
        )
    else:
        with Pool(processes=processes) as pool:
            da_jet_pos = list(
                tqdm(
                    pool.imap(
                        jet_position_as_da_wrapper, iterable, chunksize=chunksize
                    ),
                    total=len(times),
                )
            )
    da_jet_pos = xr.concat(da_jet_pos, dim="time")
    da_jet_pos = coarsen_da(da_jet_pos, 1.5)
    da_jet_pos.to_netcdf(ofile)
    return da_jet_pos


def wave_activity_flux(u, v, z, u_c=None, v_c=None, z_c=None):
    lon, lat = z.lon.values, z.lat.values
    cos_lat = degcos(lat[None, :])
    f = 2 * OMEGA * degsin(lat[:, None])
    dlon = np.gradient(lon) * np.pi / 180.0
    dlat = np.gradient(lat) * np.pi / 180.0
    psi_p = (z - z_c) / f  # Pertubation stream-function

    # 5 partial differential terms
    dpsi_dlon = np.gradient(psi_p, dlon[1])[1]
    dpsi_dlat = np.gradient(psi_p, dlat[1])[0]
    d2psi_dlon2 = np.gradient(dpsi_dlon, dlon[1])[1]
    d2psi_dlat2 = np.gradient(dpsi_dlat, dlat[1])[0]
    d2psi_dlondlat = np.gradient(dpsi_dlat, dlon[1])[1]

    termxu = dpsi_dlon * dpsi_dlon - psi_p * d2psi_dlon2
    termxv = dpsi_dlon * dpsi_dlat - psi_p * d2psi_dlondlat
    termyv = dpsi_dlat * dpsi_dlat - psi_p * d2psi_dlat2

    # coefficient
    p_lev = 300.0  # unit in hPa
    p = p_lev / 1000.0
    s_c = np.sqrt(u_c**2 + v_c**2)
    coeff = (p * degcos(lat[None, :])) / (2 * s_c)
    # x-component of TN-WAF
    px = (coeff / (RADIUS * RADIUS * cos_lat)) * (
        ((u_c) / cos_lat) * termxu + v_c * termxv
    )
    # y-component of TN-WAF
    py = (coeff / (RADIUS * RADIUS)) * (((u_c) / cos_lat) * termxv + v_c * termyv)


class JetFindingExperiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
    ) -> None:
        self.ds = data_handler.da
        self.path = data_handler.path
        self.data_handler = data_handler
        self.time = data_handler.get_sample_dims()["time"]

    def find_jets(self, **kwargs) -> pd.DataFrame:
        ofile_ajdf = self.path.joinpath("all_jets_one_df.pkl")

        if ofile_ajdf.is_file():
            all_jets_one_df = pd.read_pickle(ofile_ajdf)
            return all_jets_one_df
        try:
            qs_path = self.path.joinpath("s_q.nc")
            qs = xr.open_dataarray(qs_path)[2]
            kwargs["threshold"] = qs
        except FileNotFoundError:
            pass
        
        all_jets_one_df = find_all_jets(self.ds, self.path, **kwargs)
        all_jets_one_df.to_pickle(ofile_ajdf)
        return all_jets_one_df

    def compute_jet_props(
        self, processes: int = N_WORKERS, chunksize=100
    ) -> xr.Dataset:
        jet_props_incomplete_path = self.path.joinpath("props_as_ds_uncat_raw.nc")
        if jet_props_incomplete_path.is_file():
            return xr.open_dataset(jet_props_incomplete_path)
        all_jets_one_df = self.find_jets(processes=processes, chunksize=chunksize)
        print("Loading s")
        da_ = _compute(self.ds["s"], progress=True)
        props_as_ds_uncat = compute_all_jet_props(
            all_jets_one_df, da_, processes=processes, chunksize=chunksize
        )
        props_as_ds_uncat.to_netcdf(jet_props_incomplete_path)
        return props_as_ds_uncat

    def track_jets(self) -> Tuple:
        all_jets_one_df = self.find_jets()
        ofile_ajot = self.path.joinpath("all_jets_over_time.pkl")
        ofile_flags = self.path.joinpath("flags.pkl")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]):
            all_jets_over_time = load_pickle(ofile_ajot)
            flags = load_pickle(ofile_flags)

            return (
                all_jets_one_df,
                all_jets_over_time,
                flags,
            )
        all_jets_over_time, flags = track_jets(
            all_jets_one_df
        )

        save_pickle(all_jets_over_time, ofile_ajot)
        save_pickle(flags, ofile_flags)

        return (
            all_jets_one_df,
            all_jets_over_time,
            flags,
        )

    def props_as_ds(
        self, categorize: bool = True
    ) -> xr.Dataset:
        ofile_padu = self.path.joinpath("props_as_ds_uncat.nc")
        ofile_pad = self.path.joinpath("props_as_ds.nc")
        if ofile_padu.is_file() and not categorize:
            return xr.open_dataset(ofile_padu)
        if ofile_pad.is_file() and categorize:
            return xr.open_dataset(ofile_pad)
        if ofile_padu.is_file() and categorize:
            props_as_ds = categorize_ds_jets(xr.open_dataset(ofile_padu))
            props_as_ds.to_netcdf(ofile_pad)
            return props_as_ds
        all_jets_one_df, all_jets_over_time, flags = self.track_jets()
        props_as_ds_uncat = self.compute_jet_props()

        if self.level_type == "plev":
            props_as_ds_uncat = props_as_ds_uncat.interp(time=self.time)
            props_as_ds_uncat = is_polar_gmix(
                props_as_ds_uncat, ["mean_lat", "mean_lon", "mean_lev"], mode="month"
            )
        elif self.level_type in ["2PVU_red", "2PVU"]:
            props_as_ds_uncat = props_as_ds_uncat.interp(time=self.time)
            props_as_ds_uncat = is_polar_gmix(
                props_as_ds_uncat, ["mean_lat", "mean_lon", "mean_P"], mode="month"
            )
        else:
            props_as_ds_uncat = props_as_ds_uncat.interp(time=self.time)
            props_as_ds_uncat["is_polar"] = props_as_ds_uncat["mean_lat"] > 40.0
        props_as_ds_uncat = add_persistence_to_props(props_as_ds_uncat, flags)
        props_as_ds_uncat = self.add_com_speed(all_jets_over_time, props_as_ds_uncat)
        props_as_ds_uncat.to_netcdf(ofile_padu)
        props_as_ds = categorize_ds_jets(props_as_ds_uncat)
        props_as_ds.to_netcdf(ofile_pad)
        if categorize:
            props_as_ds
        return props_as_ds_uncat

    def props_over_time(
        self,
        all_jets_over_time: pd.DataFrame | None = None,
        props_as_ds_uncat: xr.Dataset | None = None,
        processes: int = N_WORKERS,
        chunksize: int = 100,
        save: bool = True,
    ) -> xr.Dataset:
        if all_jets_over_time is None:
            _, all_jets_over_time, _ = self.track_jets()
        if props_as_ds_uncat is None:
            props_as_ds_uncat = self.props_as_ds(categorize=False)
        incorrect = len(self.ds.time)
        out_path = self.path.joinpath("all_props_over_time.nc")
        if out_path.is_file():
            return xr.open_dataset(out_path)
        print("Props over time")
        all_props_over_time = []
        this_ajot = all_jets_over_time[all_jets_over_time[:, 0, 0] != incorrect]
        extract_props_over_time_wrapper = partial(
            extract_props_over_time, props_as_ds=props_as_ds_uncat, incorrect=incorrect
        )
        if processes == 1:
            all_props_over_time = list(
                tqdm(
                    map(extract_props_over_time_wrapper, this_ajot),
                    total=len(this_ajot),
                )
            )
        else:
            with Pool(processes=processes) as pool:
                all_props_over_time = list(
                    tqdm(
                        pool.imap(
                            extract_props_over_time_wrapper,
                            this_ajot,
                            chunksize=chunksize,
                        ),
                        total=len(this_ajot),
                    )
                )

        all_props_over_time = xr.concat(all_props_over_time, dim="jet")
        if save:
            all_props_over_time.to_netcdf(out_path)
        return all_props_over_time

    def add_com_speed(self, all_jets_over_time, props_as_ds_uncat):
        all_props_over_time = self.props_over_time(
            all_jets_over_time, props_as_ds_uncat, save=False,
        )
        dla = all_props_over_time["mean_lat"].differentiate("point").values
        dlo = all_props_over_time["mean_lon"].differentiate("point").values
        la = all_props_over_time["mean_lat"].values
        dt = all_props_over_time["time"].differentiate("point").values / 1e9
        all_props_over_time["com_speed"] = (
            ("jet", "point"),
            haversine_from_dl(la, dlo, dla) / dt,
        )
        all_props_over_time.to_netcdf(self.path.joinpath("all_props_over_time.nc"))
        to_update = ["com_speed"]
        for varname in to_update:
            print(f"computing {varname}")
            props_as_ds_uncat[varname] = props_as_ds_uncat["mean_lat"].copy()
            for jet_over_time in trange(all_props_over_time.jet.shape[0]):
                this_pot = all_props_over_time.loc[{"jet": jet_over_time}]
                valid = np.sum(~this_pot["mean_lon"].isnull()).item()
                times = this_pot["time"][:valid]
                jet_indices = this_pot["jet_index"][:valid]
                values = this_pot[varname].values[:valid]
                props_as_ds_uncat[varname].loc[times, jet_indices.astype(int)] = values
        return props_as_ds_uncat

    def compute_extreme_clim(self, varname: str, subsample: int = 5):
        da = self.ds[varname]
        time = pd.Series(self.time)
        years = time.dt.year.values
        mask = np.isin(years, np.unique(years)[::subsample])
        opath = self.path.joinpath(f"q{varname}_clim_{subsample}.nc")
        compute_extreme_climatology(da.isel(time=mask), opath)
        quantiles_clim = xr.open_dataarray(opath)
        quantiles = xr.DataArray(np.zeros((len(self.ds.time), quantiles_clim.shape[0])), coords={"time": self.ds.time.values, "quantile": quantiles_clim.coords["quantile"].values})
        for qcl in quantiles_clim.transpose():
            dayofyear = qcl.dayofyear
            quantiles[quantiles.time.dt.dayofyear == dayofyear, :] = qcl.values
        quantiles.to_netcdf(self.path.joinpath(f"q{varname}_{subsample}.nc"))


def compute_dx_dy(ds):
    lon = ds.lon.values
    lat = ds.lat.values
    lon_ = lon[None, :] * np.ones(len(lat))[:, None]
    lat_ = lat[:, None] * np.ones(len(lon))[None, :]

    dx_forward = haversine(lon_[:, 1:], lat_[:, 1:], lon_[:, :-1], lat_[:, :-1])
    dx_backwards = haversine(lon_[:, :-1], lat_[:, :-1], lon_[:, 1:], lat_[:, 1:])

    dy_forward = haversine(lon_[1:, :], lat_[1:, :], lon_[:-1, :], lat_[:-1, :])
    dy_backwards = haversine(lon_[:-1, :], lat_[:-1, :], lon_[1:, :], lat_[1:, :])

    dx = np.zeros(ds["s"].shape[1:])
    dx[:, 0] = dx_backwards[:, 0]
    dx[:, -1] = dx_forward[:, -1]
    dx[:, 1:-1] = (dx_forward[:, :-1] + dx_backwards[:, 1:]) / 2

    dy = np.zeros(ds["s"].shape[1:])
    dy[0, :] = dy_backwards[0, :]
    dy[-1, :] = dy_forward[-1, :]
    dy[1:-1, :] = (dy_forward[:-1, :] + dy_backwards[1:, :]) / 2
    return dx, dy
