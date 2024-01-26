import warnings
from pathlib import Path
from functools import partial
from typing import Callable, Mapping, Optional, Sequence, Tuple
from nptyping import NDArray
from multiprocessing import Pool
from itertools import pairwise

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from scipy.sparse.csgraph import csgraph_from_masked, shortest_path, csgraph_from_dense
from skimage.filters import frangi
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from libpysal.weights import DistanceBand
from tqdm import tqdm
from numba import njit, jit

from jetstream_hugo.definitions import (
    DATERANGEPL_SUMMER,
    DATERANGEPL_EXT_SUMMER,
    N_WORKERS,
    OMEGA,
    RADIUS,
    degsin,
    degcos,
    labels_to_mask,
    slice_1d,
)
from jetstream_hugo.data import smooth


def default_preprocess(da: xr.DataArray) -> xr.DataArray:
    return da


def smooth_wrapper(smooth_map: Mapping = None):
    return partial(smooth, smooth_map=smooth_map)


def preprocess_frangi(da: xr.DataArray, sigmas: list):
    X = da.values
    Xmax = X.max()
    X_norm = X / Xmax
    X_prime = frangi(X_norm, black_ridges=False, sigmas=sigmas, cval=1) * Xmax
    return X_prime


def frangi_wrapper(sigmas: list):
    return partial(preprocess_frangi, sigmas=sigmas)


def compute_criterion(ds: xr.Dataset) -> xr.Dataset:
    varnames = {
        varname: (f"{varname}_smo" if f"{varname}_smo" in ds else varname)
        for varname in ["s", "u", "v"]
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sigma = (
            ds[varnames["v"]] * ds[varnames["s"]].differentiate("lon")
            - ds[varnames["u"]] * ds[varnames["s"]].differentiate("lat")
        ) / ds[varnames["s"]]
        Udsigmadn = ds[varnames["v"]] * sigma.differentiate("lon") - ds[
            varnames["s"]
        ] * sigma.differentiate("lat")
    ds["criterion"] = Udsigmadn + 2 * sigma**2
    return ds


def compute_criterion_mona(ds: xr.Dataset) -> xr.Dataset:
    varname = "pv_smo" if "pv_smo" in ds else "pv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        criterion = np.log(ds[varname])
        criterion = criterion.differentiate("lon") ** 2 + criterion.differentiate("lat") ** 2
        ds["criterion"] = np.sqrt(criterion)
    return ds


def default_define_blobs(
    ds: xr.Dataset,
    criterion_threshold: float = 25.0,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 0.75,
    min_size: int = 750,
) -> Tuple[list, list, xr.Dataset]:
    print("Use a proper define_blobs step")
    raise ValueError


@njit
def my_pairwise(X):
    x = X[:, 0]
    y = X[:, 1]
    output = np.zeros((len(X), len(X)))
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            d1 = x[i] - x[j]
            d1 = np.minimum(d1, 360 - d1)
            d1 = d1 / 3
            d2 = y[i] - y[j]
            output[i, j] = np.sqrt(d1**2 + d2**2)
            output[j, i] = output[i, j]
    return output


def define_blobs_generic(
    criterion: xr.DataArray,
    *append_to_groups: xr.DataArray,
    criterion_threshold: float = 0,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 0.75,
    min_size: int = 50,
) -> Tuple[list, list]:
    lon, lat = criterion.lon.values, criterion.lat.values
    if "lev" in criterion.values:
        maxlev = criterion.argmax(dim="lev")
        append_to_groups = [atg.isel(lev=maxlev) for atg in append_to_groups]
        append_to_groups.append(criterion.lev[maxlev])
        criterion = criterion.isel(lev=maxlev)
    X = criterion.values
    idxs = np.where(X > criterion_threshold)
    append_names = [atg.name for atg in append_to_groups]
    append_to_groups = [atg.values[idxs[0], idxs[1]] for atg in append_to_groups]
    points = np.asarray([lon[idxs[1]], lat[idxs[0]], *append_to_groups]).T
    points = pd.DataFrame(points, columns=["lon", "lat", *append_names])
    dist_matrix = distance_function(points[["lon", "lat"]])
    labels = (
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="single",
        )
        .fit(dist_matrix)
        .labels_
    )
    masks = labels_to_mask(labels)
    valid_masks = [mask for mask in masks.T if np.sum(mask) > min_size]
    groups = [points.iloc[mask] for mask in valid_masks]
    dist_mats = [dist_matrix[mask, :][:, mask] for mask in valid_masks]
    return groups, dist_mats


def define_blobs_wind_speed(
    ds: xr.Dataset,
    criterion_threshold: float = 25.0,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 0.75,
    min_size: int = 750,
) -> Tuple[list, list]:
    return define_blobs_generic(
        ds["s_smo"],
        ds["s"],
        criterion_threshold=criterion_threshold,
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        min_size=min_size,
    )


def define_blobs_spensberger(
    ds: xr.Dataset,
    criterion_threshold: float = -60,
    distance_function: Callable = my_pairwise,
    distance_threshold: float = 0.75,
    min_size: int = 40,
) -> Tuple[list, list]:
    return define_blobs_generic(
        -ds["criterion"],
        ds["s"],
        criterion_threshold=-criterion_threshold,
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        min_size=min_size,
    )


def default_compute_weights(ds: xr.Dataset, group: NDArray, is_nei: NDArray) -> NDArray:
    print("Use a proper compute_weights step")
    raise ValueError


def default_refine_jets(
    ds: xr.Dataset,
    groups: list[NDArray],
    dist_mats: list[NDArray],
    compute_weights: Callable = default_compute_weights,
    jet_cutoff: float = 7.5e3,
) -> list[NDArray]:
    print("Use a proper refine_jets step")
    raise ValueError


def merge_jets(jets: list[NDArray], threshold: float = 1.5) -> list[NDArray]:
    to_merge = []
    for (i1, j1), (i2, j2) in pairwise(enumerate(jets)):
        if np.amin(pairwise_distances(j1[["lon", "lat"]], j2[["lon", "lat"]])) < threshold:
            for merger in to_merge:
                if i1 in merger:
                    merger.append(i2)
                    break
                if i2 in merger:
                    merger.append(i1)
                    break
            to_merge.append([i1, i2])
    newjets = []
    for i, jet in enumerate(jets):
        if not any([i in merger for merger in to_merge]):
            newjets.append(jet)
    for merger in to_merge:
        newjets.append(np.concatenate([jets[k] for k in merger]))
    return newjets


def jet_overlap(jet1: NDArray, jet2: NDArray) -> bool:
    _, idx1 = np.unique(jet1["lon"], return_index=True)
    _, idx2 = np.unique(jet2["lon"], return_index=True)
    x1, y1 = jet1.iloc[idx1, :2].T.to_numpy()
    x2, y2 = jet2.iloc[idx2, :2].T.to_numpy()
    mask12 = np.isin(x1, x2)
    mask21 = np.isin(x2, x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vert_dist = np.mean(np.abs(y1[mask12] - y2[mask21]))
    return (np.mean(mask21) > 0.75) and (vert_dist < 5)


def get_path_from_predecessors(Pr: NDArray, j: int) -> list:
    path = [j]
    k = j
    while Pr[k] != -9999:
        path.append(Pr[k])
        k = Pr[k]
    return path[::-1]


def get_splits(
    group: pd.DataFrame, Pr: NDArray | list, js: NDArray | list, cutoff: float
) -> list:
    # js in descending order
    splits = []
    for j in js:
        path = [j]
        k = j
        while Pr[k] != -9999:
            path.append(Pr[k])
            newk = Pr[k]
            Pr[k] = -9999
            k = newk
        if np.sum(group["s"].iloc[path]) > cutoff:
            splits.append(j)
    return splits


def last_elements(arr: NDArray, n_elements: int, sort: bool = False) -> NDArray:
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


@njit
def compute_weights_quadratic(x: NDArray, is_nei: NDArray):
    output = np.zeros(is_nei.shape)
    maxx = np.amax(x)
    z: float = 0.0
    for i in range(output.shape[0] - 1):
        for j in range(i + 1, output.shape[0]):
            if not is_nei[i, j]:
                continue
            z = 1 - ((x[i] + x[j]) / maxx / 2.1) ** 2
            output[i, j] = z
            output[j, i] = output[i, j]
    return output


@njit
def compute_weights_gaussian(x: NDArray, is_nei: NDArray):
    output = np.zeros(is_nei.shape)
    maxx = np.amax(x)
    z: float = 0.0
    for i in range(output.shape[0] - 1):
        for j in range(i + 1, output.shape[0]):
            if not is_nei[i, j]:
                continue
            z = 1 - (x[i] + x[j]) / maxx / 2.1
            output[i, j] = 1 - np.exp(-(z**2) / 2)
            output[j, i] = output[i, j]
    return output


def slice_from_df(da: xr.DataArray | xr.Dataset, indexer: pd.DataFrame, dim: str = "point") -> xr.DataArray | xr.Dataset:
    cols = [col for col in ["lev", "lon", "lat"] if col in indexer]
    indexer = {col: xr.DataArray(indexer[col].to_numpy(), dims=dim) for col in cols}
    return da.loc[indexer]


def compute_weights_wind_speed(
    ds: xr.Dataset, group: pd.DataFrame, is_nei: NDArray
) -> NDArray:
    x = slice_from_df(ds["s"], group).values
    return compute_weights_gaussian(x, is_nei)


def compute_weights_criterion(
    ds: xr.Dataset, group: pd.DataFrame, is_nei: NDArray
) -> NDArray:
    x = slice_from_df(ds["s"], group).values
    return compute_weights_gaussian(x, is_nei)


def jets_from_predecessor(
    group: pd.DataFrame, splits: Sequence, Pr: NDArray, jet_cutoff: float = 7.5e3
) -> list:
    thesejets = []
    jets = []
    for k in splits:
        sp = get_path_from_predecessors(Pr, k)
        jet = group[sp]
        if np.sum(jet["s"]) > jet_cutoff:
            thesejets.append(jet)
    if len(thesejets) == 0:
        return []
    thesejets = [
        thesejets[i] for i in np.argsort([np.sum(jet["s"]) for jet in thesejets])[::-1]
    ]
    jets.append(thesejets[0])
    for otherjet in thesejets[1:]:
        if not jet_overlap(thesejets[0], otherjet):
            jets.append(otherjet)
    return jets


def refine_jets_shortest_path(
    ds: xr.Dataset,
    groups: list[pd.DataFrame],
    dist_mats: list[NDArray],
    compute_weights: Callable = default_compute_weights,
    jet_cutoff: float = 7.5e3,
) -> list[NDArray]:
    jets = []
    for group, dist_mat in zip(groups, dist_mats):
        is_nei = (dist_mat > 0) & (dist_mat < 1)
        graph = compute_weights(ds, group, is_nei=is_nei)
        graph = np.ma.array(graph, mask=~is_nei)
        graph = csgraph_from_masked(graph)
        candidates = list(np.unravel_index(np.argmax(dist_mat), dist_mat.shape))
        im = candidates[np.argmax(group["s"].iloc[candidates])]
        dmat, Pr = shortest_path(
            graph, directed=False, return_predecessors=True, indices=im
        )
        furthest = np.argsort(dmat)[::-1]
        splits = get_splits(group, Pr.copy(), furthest, jet_cutoff)
        jets.extend(jets_from_predecessor(group, splits, Pr, jet_cutoff))
    return merge_jets(jets, 1.5)


def refine_jets_shortest_path_larger(
    ds: xr.Dataset,
    groups: list[pd.DataFrame],
    dist_mats: list[NDArray],
    compute_weights: Callable = default_compute_weights,
    jet_cutoff: float = 7.5e3,
) -> list[NDArray]:
    jets = []
    for group in groups:
        x = group["lon"].to_numpy()
        y = group["lat"].to_numpy()
        a, b = np.polyfit(x, y, deg=1)
        distance_to_line = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
        maxdist = distance_to_line.max()
        ux = np.unique(x)
        first = ux[np.argmax(np.diff(np.append([ux[-1]], ux)))]
        if (-180 in ux) and (179.5 in ux):
            x[x < first] += 360
        xmin, xmax = np.amin(x), np.amax(x)
        ymin, ymax = np.amin(y), np.amax(y)
        dy = 0.5
        dx = 0.5
        xmesh, ymesh = np.meshgrid(
            np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy)
        )
        distance_to_line_2 = np.abs(a * xmesh - ymesh + b) / np.sqrt(a**2 + 1)
        mask = distance_to_line_2 <= (maxdist)
        ix, iy = np.where(mask)
        xmesh[xmesh >= 180] -= 360
        append_names = [col for col in group.columns if col not in ["lon", "lat"]]
        append_to_group = [ds[col] for col in append_names]
        if "lev" in ds:
            maxlev = ds["s_smo"].argmax(dim="lev")
            append_to_mesh = [atm.isel(lev=maxlev) for atm in append_to_mesh]
        append_to_mesh = [slice_1d(atm, [ymesh[mask], xmesh[mask]]) for atm in append_to_mesh]
        mesh = np.asarray([xmesh[mask], ymesh[mask], *append_to_mesh]).T
        mesh = pd.DataFrame(mesh, columns=group.columns)
        grid_to_mesh = np.argmax(
            (xmesh[:, None] == x[None, :])
            & (ymesh[:, None] == y[None, :]),
            axis=0,
        )
        is_nei = pairwise_distances(np.asarray([ix, iy]).T, metric="manhattan") == 1
        graph = compute_weights(ds, mesh, is_nei)
        graph = np.ma.array(graph, mask=~is_nei)
        graph = csgraph_from_masked(graph)
        candidates = np.where(mesh[:, 0] == first)[0]
        im = candidates[np.argmax(mesh[candidates, 2])]
        dmat, Pr = shortest_path(
            graph, directed=False, return_predecessors=True, indices=im
        )
        furthest = grid_to_mesh[np.argsort(dmat[grid_to_mesh])][::-1]
        splits = get_splits(mesh, Pr.copy(), furthest, jet_cutoff)
        jets.extend(jets_from_predecessor(mesh, splits, Pr, jet_cutoff))
    return merge_jets(jets, 1.5)


class JetFinder(object):
    def __init__(
        self,
        preprocess: Callable = default_preprocess,
        compute_criterion: Callable = default_preprocess,
        define_blobs: Callable = default_define_blobs,
        refine_jets: Callable = default_refine_jets,
    ):
        self.preprocess = preprocess
        self.compute_criterion = compute_criterion
        self.define_blobs = define_blobs
        self.refine_jets = refine_jets

    def pre_loop_call(self, ds: xr.Dataset) -> xr.Dataset:
        ds["s_smo"] = self.preprocess(ds["s"])
        ds["s_smo"] = ds["s_smo"].where(ds["s_smo"] > 0, 0)
        for varname in ("u", "v"):
            if varname in ds:
                ds[f"{varname}_smo"] = self.preprocess(ds[varname])
        ds = self.compute_criterion(ds)
        return ds

    def loop_call(self, ds):
        groups, dist_mats, ds = self.define_blobs(ds)
        jets = self.refine_jets(ds, groups, dist_mats)
        return jets

    def call(
        self, ds: xr.Dataset, processes: int = N_WORKERS, chunksize: int = 2
    ) -> list:
        ds = self.pre_loop_call(ds)
        try:
            iterable = (ds.sel(time=time_) for time_ in ds.time.values)
        except AttributeError:
            iterable = (ds.sel(cluster=cluster_) for cluster_ in ds.cluster.values)
        if processes == 1:
            return list(tqdm(map(self.loop_call, iterable), total=ds["s"].shape[0]))
        with Pool(processes=processes) as pool:
            return list(
                tqdm(
                    pool.imap(self.loop_call, iterable, chunksize=chunksize),
                    total=ds["s"].shape[0],
                )
            )


def jet_trapz(jet: NDArray) -> float:
    return np.trapz(
        jet[:, 2], dx=np.mean(np.abs(np.diff(jet[:, 0])))
    )  # will fix the other one soon


def jet_integral(jet: NDArray) -> float:
    path = np.append(0, np.sqrt(np.diff(jet[:, 0]) ** 2 + np.diff(jet[:, 1]) ** 2))
    return np.trapz(jet[:, 2], x=np.cumsum(path))


def get_jet_width(x, y, s, da) -> Tuple[NDArray, NDArray]:
    lat = da.lat.values
    half_peak_mask = (da.loc[:, x] < s[None, :] / 2).values
    half_peak_mask[
        [0, -1], :
    ] = True  # worst case i clip the width at the top of the image
    below_mask = lat[:, None] <= y[None, :]
    above_mask = lat[:, None] >= y[None, :]
    below = (
        y - lat[len(lat) - np.argmax((half_peak_mask & below_mask)[::-1], axis=0) - 1]
    )
    above = lat[np.argmax(half_peak_mask & above_mask, axis=0)] - y
    return below, above


def compute_jet_props(jets: list, da: xr.DataArray = None) -> list:
    props = []
    for jet in jets:
        x, y, s = jet.T
        dic = {}
        dic["mean_lon"] = np.average(x, weights=s)
        dic["mean_lat"] = np.average(y, weights=s)
        dic["is_polar"] = dic["mean_lat"] - 0.4 * dic["mean_lon"] > 40
        maxind = np.argmax(s)
        dic["Lon"] = x[maxind]
        dic["Lat"] = y[maxind]
        dic["Spe"] = s[maxind]
        dic["lon_ext"] = np.amax(x) - np.amin(x)
        dic["lat_ext"] = np.amax(y) - np.amin(y)
        slope, _, r_value, _, _ = linregress(x, y)
        dic["tilt"] = slope
        dic["sinuosity"] = 1 - r_value**2
        try:
            above, below = get_jet_width(x, y, s, da)
            dic["width"] = np.mean(above + below + 1)
        except AttributeError:
            pass
        try:
            dic["int_over_europe"] = jet_integral(jet[x > -10])
        except ValueError:
            dic["int_over_europe"] = 0
        dic["int"] = jet_integral(jet)
        props.append(dic)
    return props


def unpack_compute_jet_props(args):  # No such thing as starimap
    return compute_jet_props(*args)


def compute_all_jet_props(
    all_jets: list,
    da: xr.DataArray = None,
    processes: int = N_WORKERS,
    chunksize: int = 10,
) -> list:
    if da is None:
        da = [None] * len(all_jets)
    with Pool(processes=processes) as pool:
        all_props = list(
            tqdm(
                pool.imap(
                    unpack_compute_jet_props, zip(all_jets, da), chunksize=chunksize
                ),
                total=len(all_jets),
            )
        )
    return all_props


def props_to_ds(
    all_props: list, time: NDArray | xr.DataArray = None, maxnjet: int = 4
) -> xr.Dataset:
    if time is None:
        time = DATERANGEPL_EXT_SUMMER
    time_name = "time"
    assert len(time) == len(all_props)
    varnames = list(all_props[0][0].keys())
    ds = {}
    for varname in varnames:
        ds[varname] = ((time_name, "jet"), np.zeros((len(time), maxnjet)))
        ds[varname][1][:] = np.nan
        for t in range(len(all_props)):
            for i in range(maxnjet):
                try:
                    props = all_props[t][i]
                except IndexError:
                    break
                ds[varname][1][t, i] = props[varname]
    ds = xr.Dataset(ds, coords={time_name: time, "jet": np.arange(maxnjet)})
    return ds


def props_to_np(props_as_ds: xr.Dataset) -> NDArray:
    props_as_ds_interpolated = props_as_ds.interpolate_na(
        dim="time", method="linear", fill_value="extrapolate"
    )
    props_as_np = np.zeros(
        (len(props_as_ds.time), len(props_as_ds.jet) * len(props_as_ds.data_vars))
    )
    i = 0
    for varname in props_as_ds.data_vars:
        for jet in props_as_ds.jet:
            props_as_np[:, i] = props_as_ds_interpolated[varname].sel(jet=jet).values
            i = i + 1
    return props_as_np


def better_is_polar(
    all_jets: list, props_as_ds: xr.Dataset, exp_low_path: Path
) -> xr.Dataset:
    this_path = exp_low_path.joinpath("better_is_polar.nc")
    if this_path.is_file():
        props_as_ds["int_low"] = xr.open_dataarray(exp_low_path.joinpath("int_low.nc"))
        props_as_ds["is_polar"] = xr.open_dataarray(this_path)
        return props_as_ds
    print("computing int low")
    da_low = xr.open_dataarray(exp_low_path.joinpath("da.nc"))
    props_as_ds["int_low"] = props_as_ds["mean_lon"].copy()
    for it, (jets, mean_lats) in tqdm(
        enumerate(zip(all_jets, props_as_ds["mean_lat"])), total=len(all_jets)
    ):
        for j, (jet, mean_lat) in enumerate(zip(jets, mean_lats.values)):
            x, y, s = jet.T
            x_ = xr.DataArray(x, dims="points")
            y_ = xr.DataArray(y, dims="points")
            s_low = da_low[it].sel(lon=x_, lat=y_).values
            jet_low = np.asarray([x, y, s_low]).T
            props_as_ds["int_low"][it, j] = jet_integral(jet_low).item()

    props_as_ds["is_polar"] = (
        props_as_ds["mean_lat"] * 200
        - props_as_ds["mean_lon"] * 30
        + props_as_ds["int_low"]
    ) > 9000
    props_as_ds["int_low"].to_netcdf(exp_low_path.joinpath("int_low.nc"))
    props_as_ds["is_polar"].to_netcdf(this_path)
    del props_as_ds["int_low"]
    return props_as_ds


def categorize_ds_jets(props_as_ds: xr.Dataset):
    time_name, time_val = list(props_as_ds.coords.items())[0]
    ds = xr.Dataset(coords={time_name: time_val, "jet": ["subtropical", "polar"]})
    cond = props_as_ds["is_polar"]
    for varname in props_as_ds.data_vars:
        if varname == "is_polar":
            continue
        values = np.zeros((len(time_val), 2))
        values[:, 0] = props_as_ds[varname].where(1 - cond).mean(dim="jet").values
        values[:, 1] = props_as_ds[varname].where(cond).mean(dim="jet").values
        ds[varname] = ((time_name, "jet"), values)
    return ds


def jet_overlap_values(jet1: NDArray, jet2: NDArray) -> Tuple[float, float]:
    _, idx1 = np.unique(jet1[:, 0], return_index=True)
    _, idx2 = np.unique(jet2[:, 0], return_index=True)
    x1, y1, s1 = jet1[idx1, :3].T
    x2, y2, s2 = jet2[idx2, :3].T
    mask12 = np.isin(x1, x2)
    mask21 = np.isin(x2, x1)
    overlap = np.sum((s1[mask12] + s2[mask21]) / 2)
    vert_dist = np.sum(np.abs(y1[mask12] - y2[mask21]))
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
            for jet1, jet2 in pairwise(jets):
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


def all_jets_to_one_array(all_jets: list):
    num_jets = [len(j) for j in all_jets]
    maxnjets = max(num_jets)
    num_indiv_jets = sum(num_jets)
    where_are_jets = np.full((len(all_jets), maxnjets, 2), fill_value=-1)
    all_jets_one_array = []
    k = 0
    l = 0
    for t, jets in enumerate(all_jets):
        for j, jet in enumerate(jets):
            l = k + len(jet)
            all_jets_one_array.append(jet)
            where_are_jets[t, j, :] = (k, l)
            k = l
    all_jets_one_array = np.concatenate(all_jets_one_array)
    return where_are_jets, all_jets_one_array


def one_array_to_all_jets(all_jets_one_array, where_are_jets):
    all_jets = []
    for where_are_jet in where_are_jets:
        all_jets.append([])
        for k, l in where_are_jet:
            all_jets[-1].append(all_jets_one_array[k:l])
    return all_jets


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


@njit
def amin_ax0(a):
    result = np.zeros(a.shape[1])
    for i, a_ in enumerate(a.T):
        result[i] = np.amin(a_) if len(a_) > 0 else 1e5
    return result


@njit
def amin_ax1(a):
    result = np.zeros(a.shape[0])
    for i, a_ in enumerate(a):
        result[i] = np.amin(a_) if len(a_) > 0 else 1e5
    return result


def track_jets(
    all_jets_one_array, where_are_jets, yearbreaks: int = None, progress_proxy=None
):
    factor: float = 0.2
    if yearbreaks is None:
        yearbreaks = (
            92 if where_are_jets.shape[0] < 10000 else 92 * 4
        )  # find better later
    guess_nflags: int = 13000
    all_jets_over_time = np.full(
        (guess_nflags, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32
    )
    last_valid_idx = np.full(guess_nflags, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full(where_are_jets.shape[:2], fill_value=guess_nflags, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    flags[0, : last_flag + 1] = np.arange(last_flag + 1)
    for t, jet_idxs in enumerate(where_are_jets[1:]):  # can't really parallelize
        if np.all(where_are_jets[t + 1] == -1):
            continue
        potentials = np.zeros(50, dtype=np.int32)
        from_ = max(0, last_flag - 30)
        times_to_test = np.take_along_axis(
            all_jets_over_time[from_ : last_flag + 1, :, 0],
            last_valid_idx[from_ : last_flag + 1, None],
            axis=1,
        ).flatten()
        timesteps_to_check = np.asarray([t, t - 1, t - 2, t - 3])
        potentials = (
            from_
            + np.where(
                isin(times_to_test, timesteps_to_check)
                & ((times_to_test // yearbreaks) == ((t + 1) // yearbreaks)).astype(
                    np.bool_
                )
            )[0]
        )
        num_valid_jets = np.sum(jet_idxs[:, 0] >= 0)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        for i, jtt_idx in enumerate(potentials):
            t_jtt, j_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            k_jtt, l_jtt = where_are_jets[t_jtt, j_jtt]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            for j in range(num_valid_jets):
                k, l = jet_idxs[j]
                jet = all_jets_one_array[k:l, :2]
                distances = np.sqrt(
                    np.sum(
                        (
                            np.radians(jet_to_try)[None, :, :]
                            - np.radians(jet)[:, None, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
                dist_mat[i, j] = np.mean(
                    np.array(
                        [
                            np.sum(amin_ax1(distances / len(jet_to_try))),
                            np.sum(amin_ax0(distances / len(jet))),
                        ]
                    )
                )
        connected_mask = dist_mat < factor
        flagged = np.zeros(num_valid_jets, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
            k_jtt, l_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if flagged[j]:
                    continue
                last_valid_idx[jtt_idx] = last_valid_idx[jtt_idx] + 1
                all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx], :] = (t + 1, j)
                flagged[j] = True
                flags[t + 1, j] = jtt_idx
                break

        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                all_jets_over_time[last_flag, 0, :] = (t + 1, j)
                last_valid_idx[last_flag] = 0
                flags[t + 1, j] = last_flag
                flagged[j] = True

        if progress_proxy is not None:
            progress_proxy.update(1)

    return all_jets_over_time, flags


def extract_props_over_time(jet, all_props):
    varnames = list(all_props[0][0].keys())
    props_over_time = {varname: np.zeros(len(jet)) for varname in varnames}
    for varname in varnames:
        for ti, (t, j) in enumerate(jet):
            props_over_time[varname][ti] = all_props[t][j][varname]
    return props_over_time


def add_persistence_to_props(ds_props: xr.Dataset, flags: NDArray):
    names = ["time", "jet"]
    dt1 = (ds_props.time.values[1] - ds_props.time.values[0]).astype("timedelta64[h]")
    dt2 = np.timedelta64(1, "D")
    factor = float(dt1 / dt2)
    num_jets = ds_props["mean_lon"].shape[1]
    jet_persistence_prop = flags[:, :num_jets].copy().astype(float)
    nan_flag = np.amax(flags)
    unique_flags, jet_persistence = np.unique(flags, return_counts=True)
    for i, flag in enumerate(unique_flags[:-1]):
        jet_persistence_prop[flags[:, :num_jets] == flag] = jet_persistence[i] * factor
    jet_persistence_prop[flags[:, :num_jets] == nan_flag] = np.nan
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


def jet_position_as_da(
    da_s: xr.DataArray,
    props_as_ds: xr.Dataset,
    props_as_ds_uncat: xr.Dataset,
    all_jets: list,
    basepath: Path,
) -> xr.DataArray:
    ofile = basepath.joinpath("jet_pos.nc")
    if ofile.is_file():
        da_jet_pos = xr.open_dataarray(ofile).load()
    else:
        time, jet_names, lat, lon = (
            da_s.time.values,
            props_as_ds.jet.values,
            da_s.lat.values,
            da_s.lon.values,
        )
        coords = {"time": time, "jet": jet_names, "lat": lat, "lon": lon}
        da_jet_pos = xr.DataArray(
            np.zeros([len(val) for val in coords.values()], dtype=np.float32),
            coords=coords,
        )

        for it, (jets, are_polar) in tqdm(
            enumerate(zip(all_jets, props_as_ds_uncat["is_polar"])), total=len(all_jets)
        ):
            for jet, is_polar in zip(jets, are_polar):
                x, y, s = jet.T
                x_ = xr.DataArray(x, dims="points")
                y_ = xr.DataArray(y, dims="points")
                da_jet_pos.loc[time[it], jet_names[int(is_polar)], y_, x_] += s
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
