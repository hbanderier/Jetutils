from re import S
from tkinter.ttk import Progressbar
import warnings
from pathlib import Path
from functools import partial, wraps
from typing import Callable, Mapping, Optional, Sequence, Tuple, Literal
from astropy.utils import masked
from nptyping import NDArray
from multiprocessing import Pool
from itertools import pairwise

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
import xarray as xr
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (
    csgraph_from_masked,
    shortest_path,
    connected_components,
)
from skimage.filters import frangi, meijering, sato
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from libpysal.weights import DistanceBand
from tqdm import tqdm
from numba import njit, jit
from numba_progress import ProgressBar as NumbaProgress

from jetstream_hugo.definitions import (
    DATADIR,
    COMPUTE_KWARGS,
    DATERANGEPL_SUMMER,
    DATERANGEPL_EXT_SUMMER,
    N_WORKERS,
    OMEGA,
    RADIUS,
    degsin,
    degcos,
    labels_to_mask,
    slice_1d,
    save_pickle,
    load_pickle,
)
from jetstream_hugo.data import smooth, unpack_levels, open_da


def default_preprocess(da: xr.DataArray) -> xr.DataArray:
    return da


def smooth_wrapper(smooth_map: Mapping = None):
    return partial(smooth, smooth_map=smooth_map)


def preprocess_frangi(da: xr.DataArray, sigmas: Optional[Sequence] = None):
    X = da.values
    Xmax = X.max()
    X_norm = X / Xmax
    X_prime = frangi(X_norm, black_ridges=False, sigmas=sigmas, cval=1) * Xmax
    return X_prime


def preprocess_meijering(da: xr.DataArray, sigmas: Optional[Sequence] = None):
    if sigmas is None:
        sigmas = range(2, 10, 2)
    da = meijering(da, black_ridges=False, sigmas=sigmas) * da
    return da


def frangi_wrapper(sigmas: list):
    return partial(preprocess_frangi, sigmas=sigmas)


def flatten_by(ds: xr.Dataset, by: str = "-criterion") -> xr.Dataset:
    ope = np.argmin if by[0] == "-" else np.argmax
    by = by.lstrip("-")
    if ds["s"].chunks is not None:
        with ProgressBar(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds = ds.compute(**COMPUTE_KWARGS)
    levmax = ds[by].reduce(ope, dim="lev")
    return ds.isel(lev=levmax).reset_coords("lev")  # but not drop


def compute_criterion(ds: xr.Dataset, flatten: bool = True) -> xr.Dataset:
    varnames = {
        varname: (f"{varname}_smo" if f"{varname}_smo" in ds else varname)
        for varname in ["s", "u", "v"]
    }
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
        sigma = (
            ds[varnames["v"]] * ds[varnames["s"]].differentiate("x")
            - ds[varnames["u"]] * ds[varnames["s"]].differentiate("y")
        ) / ds[varnames["s"]]
        Udsigmadn = ds[varnames["v"]] * sigma.differentiate("x") - ds[
            varnames["s"]
        ] * sigma.differentiate("y")
        ds["criterion"] = Udsigmadn + sigma**2
        ds["criterion"] = ds["criterion"].where(ds["criterion"] < 0, 0)
        ds["criterion"] = ds["criterion"].where(np.isfinite(ds["criterion"]), 0)
    if flatten and "lev" in ds.dims:
        ds = flatten_by(ds, "-criterion")
    return ds.reset_coords(["x", "y"], drop=True)


def compute_criterion_mona(ds: xr.Dataset, flatten: bool = True) -> xr.Dataset:
    varname = "pv_smo" if "pv_smo" in ds else "pv"
    criterion = np.log(ds[varname])
    criterion = (
        criterion.differentiate("lon") ** 2 + criterion.differentiate("lat") ** 2
    )
    ds["criterion"] = np.sqrt(criterion)
    if flatten and "lev" in ds.dims:
        return flatten_by(ds, "criterion")
    return ds


def preprocess(ds: xr.Dataset, sigmas=range(3, 6)):
    ds = flatten_by(ds, "s")
    ds["s_smooth"] = smooth(ds["s"], smooth_map={"lon+lat": ("fft", 0.25)})
    filtered = sato(ds["s_smooth"] / ds["s_smooth"].max(), black_ridges=False, sigmas=sigmas)
    filtered = filtered / filtered.max()
    ds["s_smo"] = ds["s_smooth"] * filtered
    # ds["s_smo"] = (("lat", "lon"), sato(ds["s_smooth"] / ds["s_smooth"].max(), black_ridges=black_ridges, sigmas=sigmas) * (4 if multiply else 1))
    return ds


def default_cluster(
    ds: xr.Dataset,
    criterion_threshold: float = 25.0,
    distance_function: Callable = pairwise_distances,
) -> Tuple[list, list, xr.Dataset]:
    print("Use a proper define_blobs step")
    raise ValueError


@njit
def my_pairwise_3(X: NDArray) -> NDArray:
    x = X[:, 0]
    y = X[:, 1]
    output = np.zeros((len(X), len(X)))
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            d1 = x[i] - x[j]
            d1 = np.minimum(d1, 360 - d1)
            d1_ = d1 / 3
            d2 = y[i] - y[j]
            output[i, j] = np.sqrt(d1_**2 + d2**2)
            output[j, i] = output[i, j]
    return output


def cluster_generic(
    criterion: xr.DataArray,
    *append_to_groups: xr.DataArray,
    criterion_threshold: float = 0,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 1.0,
    min_size: int = 500,
) -> Tuple[list, list]:
    append_to_groups = [atg for atg in append_to_groups if atg is not None]
    lon, lat = criterion.lon.values, criterion.lat.values
    if "lev" in criterion.dims:
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
    if "lev" in points:
        dlev = np.diff(np.unique(points["lev"]))
        dlev = np.amin(dlev[dlev > 0])
        to_distance = points[["lon", "lat", "lev"]]
        factors = np.ones(to_distance.shape)
        factors[:, 2] = 2 * dlev
        to_distance = to_distance / pd.DataFrame(factors, columns=["lon", "lat", "lev"])
        distance_matrix = distance_function(to_distance.to_numpy())
    else:
        distance_matrix = distance_function(points[["lon", "lat"]].to_numpy())
    labels = (
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="single",
        )
        .fit(distance_matrix)
        .labels_
    )
    masks = labels_to_mask(labels)
    valid_masks = [mask for mask in masks.T if np.sum(mask) > min_size]
    groups = [points.iloc[mask] for mask in valid_masks]
    dist_mats = [distance_matrix[mask, :][:, mask] for mask in valid_masks]
    return groups, dist_mats


def cluster_wind_speed(
    ds: xr.Dataset,
    criterion_threshold: float = 7.5,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 1.5,
    min_size: int = 500,
) -> Tuple[list, list]:
    return cluster_generic(
        ds["s_smo"],
        ds["s"],
        ds["s_smo"],
        ds["criterion"] if "criterion" in ds else None,
        ds["lev"] if "lev" in ds else None,
        ds["u"] if "u" in ds else None,
        ds["v"] if "v" in ds else None,
        criterion_threshold=(
            ds["threshold"].item() if "threshold" in ds else criterion_threshold
        ),
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        min_size=min_size,
    )


def cluster_criterion(
    ds: xr.Dataset,
    criterion_threshold: float = -60.0,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 1.0,
    min_size: int = 500,
) -> Tuple[list, list]:
    if 0.0 < criterion_threshold < 1.0:
        criterion_threshold = ds["criterion"].quantile(criterion_threshold).item()
    return cluster_generic(
        -ds["criterion"],
        ds["s"],
        ds["criterion"],
        ds["lev"] if "lev" in ds else None,
        ds["u"],
        ds["v"],
        criterion_threshold=(
            ds["threshold"].item() if "threshold" in ds else -criterion_threshold
        ),
        distance_function=distance_function,
    )


def jet_overlap_flat(jet1: NDArray, jet2: NDArray) -> bool:
    _, idx1 = np.unique(jet1["lon"], return_index=True)
    _, idx2 = np.unique(jet2["lon"], return_index=True)
    x1, y1 = jet1.iloc[idx1, :2].T.to_numpy()
    x2, y2 = jet2.iloc[idx2, :2].T.to_numpy()
    mask12 = np.isin(x1, x2)
    mask21 = np.isin(x2, x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vert_dist = np.mean(np.abs(y1[mask12] - y2[mask21]))
    overlap = max(np.mean(mask21), np.mean(mask12))
    return (overlap > 0.85) and (vert_dist < 5)


@njit
def my_pairwise(X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
    x1 = X[:, 0]
    y1 = X[:, 1]
    if Y is None:
        Y = X
        half = True
    else:
        x2 = Y[:, 0]
        y2 = Y[:, 1]
        half = False
    output = np.zeros((len(X), len(Y)))
    for i in range(X.shape[0] - int(half)):
        if half:
            for j in range(i + 1, X.shape[0]):
                d1 = x2[i] - x1[j]
                d2 = y2[i] - y1[j]
                output[i, j] = np.sqrt(d1**2 + d2**2)
                output[j, i] = output[i, j]
        else:
            for j in range(Y.shape[0]):
                d1 = x2[i] - x1[j]
                d2 = y2[i] - y1[j]
                output[i, j] = np.sqrt(d1**2 + d2**2)
    return output


@njit
def amin_ax0(a):
    result = np.zeros(a.shape[1])
    for i, a_ in enumerate(a.T):
        result[i] = np.amin(a_)
    return result


@njit
def amin_ax1(a):
    result = np.zeros(a.shape[0])
    for i, a_ in enumerate(a):
        result[i] = np.amin(a_)
    return result


@njit
def pairwise_jet_distance(jet1: NDArray, jet2: NDArray) -> float:
    distances = my_pairwise(jet1, jet2)
    distance = (
        np.sum(amin_ax1(distances / len(jet2)))
        + np.sum(amin_ax0(distances / len(jet1)))
    ) * 0.5
    return distance


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


def get_extremities(mask: NDArray, distance_matrix: NDArray) -> NDArray:
    idx = np.where(mask)[0]
    this_dist_mat = distance_matrix[idx, :][:, idx]
    extremities = np.concatenate(
        last_elements(this_dist_mat, min(this_dist_mat.shape[0], 100))
    )
    return idx[np.unique(extremities)]


def normalize_points_for_weights(points: pd.DataFrame, by: str = "-criterion"):
    sign = -1.0 if by[0] == "-" else 1.0
    by = by.lstrip("-")
    lon = points["lon"].to_numpy()
    lon_ = np.unique(lon)
    lat = points["lat"].to_numpy()
    lat_ = np.unique(lat)
    indexers = (xr.DataArray(lon, dims="points"), xr.DataArray(lat, dims="points"))

    da = xr.DataArray(
        np.zeros((len(lon_), len(lat_))), coords={"lon": lon_, "lat": lat_}
    )
    da[:] = np.nan
    da.loc[*indexers] = sign * points[by].to_numpy()
    maxx = da.max("lat").rolling(lon=1, min_periods=1).max()
    return (da / maxx).loc[*indexers].values


@njit
def compute_weights_quadratic(X: NDArray, is_nei: Optional[NDArray] = None):
    nx = X.shape[0]
    output = np.zeros((nx, nx), dtype=np.float32)
    z: float = 0.0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            if is_nei is not None and not is_nei[i, j]:
                continue
            z = 1 - (X[i] + X[j]) / 2
            output[i, j] = z
            output[j, i] = output[i, j]
    return output


@njit
def compute_weights_gaussian(X: NDArray, is_nei: Optional[NDArray] = None):
    nx = X.shape[0]
    output = np.zeros((nx, nx), dtype=np.float32)
    z: float = 0.0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            if is_nei is not None and not is_nei[i, j]:
                continue
            z = 1 - (X[i] + X[j]) / 2
            output[i, j] = 1 - np.exp(-(z**2) / 2)
            output[j, i] = output[i, j]
    return output


@njit
def compute_weights_mean(X: NDArray, is_nei: Optional[NDArray] = None):
    nx = X.shape[0]
    output = np.zeros((nx, nx), dtype=np.float32)
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            if is_nei is not None and not is_nei[i, j]:
                continue
            output[i, j] = 1 - (X[i] + X[j]) / 2
            output[j, i] = output[i, j]
    return output


def slice_from_df(
    da: xr.DataArray | xr.Dataset, indexer: pd.DataFrame, dim: str = "point"
) -> xr.DataArray | xr.Dataset:
    cols = [col for col in ["lev", "lon", "lat"] if col in indexer and col in da.dims]
    indexer = {col: xr.DataArray(indexer[col].to_numpy(), dims=dim) for col in cols}
    return da.loc[indexer]


def compute_weights_wind_speed_slice(
    ds: xr.Dataset, points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = slice_from_df(ds["s"], points).values
    x = x / x.max()
    return compute_weights_mean(x, is_nei)


def compute_weights_wind_speed(
    points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = normalize_points_for_weights(points, "s")
    return compute_weights_quadratic(x, is_nei)


def compute_weights_wind_speed_smoothed(
    points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = normalize_points_for_weights(points, "s_smo")
    return compute_weights_quadratic(x, is_nei)


def compute_weights_criterion_slice(
    ds: xr.Dataset, points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = -slice_from_df(ds["criterion"], points).values
    x = x / x.max()
    return compute_weights_mean(x, is_nei)


def compute_weights_criterion(
    points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = normalize_points_for_weights(points, "-criterion")
    return compute_weights_mean(x, is_nei)


@njit
def pairwise_difference(X: NDArray, is_nei: Optional[NDArray] = None):
    nx = X.shape[0]
    output = np.zeros((nx, nx), dtype=np.float32)
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            if is_nei is not None and not is_nei[i, j]:
                continue
            output[i, j] = X[j] - X[i]
            output[j, i] = -output[i, j]
    return output


@njit
def _compute_weights_direction(
    x: NDArray,
    y: NDArray,
    u: NDArray,
    v: NDArray,
    s: NDArray,
    distance_matrix: NDArray,
    is_nei: Optional[NDArray] = None,
) -> NDArray:
    dx = pairwise_difference(x, is_nei) / distance_matrix
    dy = pairwise_difference(y, is_nei) / distance_matrix
    u = u / s
    v = v / s
    return (1 - dx * u[:, None] + dy * v[:, None]) / 2


def compute_weights_direction(
    points: pd.DataFrame, distance_matrix: NDArray, is_nei: Optional[NDArray] = None
):
    x, y, u, v, s = points[["lon", "lat", "u", "v", "s"]].to_numpy().T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = _compute_weights_direction(x, y, u, v, s, distance_matrix, is_nei)
    return out


def compute_weights(points: pd.DataFrame, distance_matrix: NDArray) -> np.ma.array:
    is_nei = (distance_matrix > 0) & (distance_matrix < 1)
    weights_ws = compute_weights_wind_speed_smoothed(points, is_nei)
    weights_dir = compute_weights_direction(points, distance_matrix, is_nei)
    masked_weights = 0.5 * np.ma.array(weights_ws, mask=~is_nei) + 0.5 * np.ma.array(
        weights_dir, mask=~is_nei
    )
    return masked_weights


@njit
def haversine(lon1: NDArray, lat1: NDArray, lon2: NDArray, lat2: NDArray) -> NDArray:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return RADIUS * c


@njit
def jet_integral_haversine(jet: NDArray):
    X = jet[:, :2]
    ds = haversine(X[:-1, 0], X[:-1, 1], X[1:, 0], X[1:, 1])
    ds = np.append([0], ds)
    return np.trapz(jet[:, 2], x=np.cumsum(ds))


def jet_integral_lon(jet: NDArray) -> float:
    return np.trapz(
        jet[:, 2], dx=np.mean(np.abs(np.diff(jet[:, 0])))
    )  # will fix the other one soon


def jet_integral_flat(jet: NDArray) -> float:
    path = np.append(0, np.sqrt(np.diff(jet[:, 0]) ** 2 + np.diff(jet[:, 1]) ** 2))
    return np.trapz(jet[:, 2], x=np.cumsum(path))


def create_graph(masked_weights: np.ma.array, distance_matrix: NDArray) -> csr_matrix:
    graph = csgraph_from_masked(masked_weights)
    nco, labels = connected_components(graph)
    if nco == 1:
        return graph
    for label1, label2 in pairwise(range(nco)):
        idxs1 = np.where(labels == label1)[0]
        idxs2 = np.where(labels == label2)[0]
        thisdmat = distance_matrix[idxs1, :][:, idxs2]
        i, j = np.unravel_index(np.argmin(thisdmat), thisdmat.shape)
        i, j = idxs1[i], idxs2[j]
        masked_weights[i, j] = 0.5
        masked_weights.mask[i, j] = False
    return csgraph_from_masked(masked_weights)


@njit
def path_from_predecessors(
    predecessors: NDArray, end: np.int32
) -> NDArray:  # Numba this like jet tracking stuff
    path = np.full(predecessors.shape, fill_value=end, dtype=np.int32)
    for i, k in enumerate(path):
        newk = predecessors[k]
        if newk == -9999:
            break
        path[i + 1] = newk
        predecessors[k] = -9999
    return path[: (i + 1)]


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
def jets_from_many_predecessors(
    group: NDArray,
    predecessors: NDArray,  # 2d
    ends: NDArray,
    dmat_weighted: NDArray,
    dmat_unweighted: NDArray,
    cutoff: float,
) -> Sequence:
    # at most one jet
    path = None
    jet = None
    for istart, predecessors_ in enumerate(predecessors):
        if len(ends) > 1:
            dmat_ratio = (
                dmat_unweighted[istart, ends] ** 4 / dmat_weighted[istart, ends] ** 0.5
            )
            dmat_ratio[np.isnan(dmat_ratio)] = -1
            end = ends[np.argmax(dmat_ratio)]
        else:
            end = ends[0]
        candidate_path = path_from_predecessors(predecessors_, end)
        candidate_jet = group[candidate_path]
        if jet_integral_haversine(candidate_jet) < cutoff:
            continue
        if path is None or jet is None:
            path = candidate_path
            jet = candidate_jet
            continue
        integral1 = jet_integral_haversine(candidate_jet)
        integral2 = jet_integral_haversine(jet)
        if integral1 > integral2:
            path = candidate_path
            jet = candidate_jet
    return path


def find_jets_in_group(
    graph: csr_matrix, group: pd.DataFrame, dist_mat: NDArray, jet_cutoff: float = 5e7
):
    ncand = dist_mat.shape[0] // 5
    candidates = np.unique(np.concatenate(last_elements(dist_mat, ncand)))
    earlies = 1 + np.argmax(np.diff(candidates))
    starts = candidates[:earlies]
    ends = candidates[earlies:]
    dmat_w, predecessors = shortest_path(
        graph, directed=True, return_predecessors=True, indices=starts
    )
    dmat_uw, _ = shortest_path(
        graph, unweighted=True, directed=True, return_predecessors=True, indices=starts
    )
    thesejets = jets_from_many_predecessors(
        group, predecessors, ends, dmat_w, dmat_uw, jet_cutoff
    )
    return thesejets


def find_jets_in_group_v2(
    graph: csr_matrix, group: pd.DataFrame, dist_mat: NDArray, jet_cutoff: float = 5e7
):
    lon, lat, s = group[["lon", "lat", "s"]].to_numpy().T
    c1, c0 = np.polyfit(lon, lat, w=s, deg=1)
    x0 = np.amin(lon)
    y0 = x0 * c1 + c0
    v0 = np.asarray([[x0, y0]])
    c = np.asarray([[1, c1]]) / np.sqrt(1 + c1**2)
    points = group[["lon", "lat"]].to_numpy() - v0
    projections = np.sum(c * points, axis=1)
    ncand = projections.shape[0] // 15
    starts = first_elements(projections, ncand).astype(np.int16)
    ncand = projections.shape[0] // 2
    ends = last_elements(projections, ncand).astype(np.int16)
    dmat_weighted, predecessors = shortest_path(
        graph, directed=True, return_predecessors=True, indices=starts
    )
    dmat_unweighted, _ = shortest_path(
        graph, unweighted=True, directed=True, return_predecessors=True, indices=starts
    )
    path = jets_from_many_predecessors(
        group[["lon", "lat", "s"]].to_numpy(),
        predecessors,
        ends,
        dmat_weighted,
        dmat_unweighted,
        jet_cutoff,
    )
    jet = group.iloc[path] if path is not None else None
    return jet


def jets_from_mask(
    groups: Sequence[pd.DataFrame],
    dist_mats: Sequence[NDArray],
    jet_cutoff: float = 1e8,
) -> Sequence[pd.DataFrame]:
    jets = []
    for group, dist_mat in zip(groups, dist_mats):
        masked_weights = compute_weights(group, dist_mat)
        graph = create_graph(masked_weights, dist_mat)
        jet = find_jets_in_group_v2(graph, group, dist_mat, jet_cutoff)
        if jet is not None:
            jets.append(jet)
    return jets


class JetFinder(object):
    def __init__(
        self,
        preprocess: Callable = default_preprocess,
        cluster: Callable = default_cluster,
        refine_jets: Callable = jets_from_mask,
    ):
        self.preprocess = preprocess
        self.cluster = cluster
        self.refine_jets = refine_jets

    def loop_call(self, ds):
        ds = self.preprocess(ds)
        groups, dist_masts = self.cluster(ds)
        jets = self.refine_jets(groups, dist_masts)
        return jets

    def call(
        self,
        ds: xr.Dataset,
        thresholds: Optional[xr.DataArray] = None,
        processes: int = N_WORKERS,
        chunksize: int = 2,
    ) -> list:
        if thresholds is not None:
            thresholds = thresholds.loc[getattr(ds.time.dt, thresholds.dims[0])].values
            ds["threshold"] = ("time", thresholds)
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


def get_jet_width(x: NDArray, y: NDArray, s: NDArray, da: xr.DataArray) -> Tuple[NDArray, NDArray]:
    lat = da.lat.values
    half_peak_mask = (da.loc[:, x] < s[None, :] / 2).values
    half_peak_mask[[0, -1], :] = (
        True  # worst case i clip the width at the top of the image
    )
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
        jet_numpy = jet[["lon", "lat", "s"]].to_numpy()
        x, y, s = jet_numpy.T
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
            dic["int_over_europe"] = jet_integral_haversine(jet_numpy[x > -10])
        except ValueError:
            dic["int_over_europe"] = 0
        dic["int"] = jet_integral_haversine(jet_numpy)
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
            props_as_ds["int_low"][it, j] = jet_integral_haversine(jet_low).item()

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
    where_are_jets = np.full((len(all_jets), maxnjets, 2), fill_value=-1)
    all_jets_one_array = []
    k = 0
    l = 0
    for t, jets in enumerate(all_jets):
        for j, jet in enumerate(jets):
            jet = jet[["lon", "lat"]].to_numpy()
            l = k + len(jet)
            all_jets_one_array.append(jet)
            where_are_jets[t, j, :] = (k, l)
            k = l
    all_jets_one_array = np.concatenate(all_jets_one_array)
    return where_are_jets, all_jets_one_array


def track_jets(
    all_jets_one_array, where_are_jets, yearbreaks: int = None, progress_proxy=None
):
    factor: float = 0.2
    if yearbreaks is None:
        yearbreaks = (
            92 if where_are_jets.shape[0] < 10000 else 92 * 4
        )  # find better later, anyways this is usually wrapped and given externally
    guess_nflags: int = 20000
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
                dist_mat[i, j] = pairwise_jet_distance(jet, jet_to_try)

        connected_mask = dist_mat < factor
        flagged = np.zeros(num_valid_jets, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
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
                x, y, s = jet[["lon", "lat", "s"]].to_numpy().T
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
    
    
def MultiVarExperiment(object):
    """
    For now cannot handle anomalies, only raw fields.
    """
    def __init__(
        self,
        dataset: str,
        level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
        varnames: Sequence[str],
        resolution: str,
        period: list | tuple | Literal["all"] | int | str = "all",
        season: list | str = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        levels: int | str | tuple | list | Literal["all"] = "all",
        inner_norm: int = None,
    ) -> None:
        self.path = Path(DATADIR, dataset, level_type, "results")
        self.path.mkdir(exist_ok=True)
        self.open_da_kwargs = {
            "dataset": dataset,
            "level_type": level_type,
            "resolution": resolution,
            "period": period,
            "season": season,
            "minlon": minlon,
            "maxlon": maxlon,
            "minlat": minlat,
            "maxlat": maxlat,
            "levels": levels,
        }

        self.varnames = varnames
        self.region = (minlon, maxlon, minlat, maxlat)
        if levels != 'all':
            self.levels, self.level_names = unpack_levels(levels)
        else: 
            self.levels = 'all'
        self.inner_norm = inner_norm

        self.metadata = {
            "varnames": varnames,
            "period": period,
            "season": season,
            "region": (minlon, maxlon, minlat, maxlat),
            "levels": self.levels,
            "inner_norm": inner_norm,
        }

        found = False
        for dir in self.path.iterdir():
            if not dir.is_dir():
                continue
            try:
                other_mda = load_pickle(dir.joinpath("metadata.pkl"))
            except FileNotFoundError:
                continue
            if self.metadata == other_mda:
                self.path = self.path.joinpath(dir.name)
                found = True
                break

        if not found:
            seq = [int(dir.name) for dir in self.path.iterdir() if dir.is_dir()]
            id = max(seq) + 1 if len(seq) != 0 else 1
            self.path = self.path.joinpath(str(id))
            self.path.mkdir()
            save_pickle(self.metadata, self.path.joinpath("metadata.pkl"))

        ds_path = self.path.joinpath("ds.nc")
        if ds_path.is_file():
            self.ds = xr.open_dataarray(ds_path).load()
        else:
            self.ds = xr.Dataset()
            for varname in varnames:
                open_kwargs = self.open_da_args | {"varname": varname}
                self.ds[varname] = open_da(**open_kwargs)
            with ProgressBar():
                self.ds = self.ds.load()
            self.ds.to_netcdf(ds_path, format="NETCDF4")

        self.samples_dims = {"time": self.ds.time.values}
        try:
            self.samples_dims["member"] = self.ds.member.values
        except AttributeError:
            pass
        self.lon, self.lat = self.ds.lon.values, self.ds.lat.values
        try:
            self.feature_dims = {"lev": self.ds.lev.values}
        except AttributeError:
            self.feature_dims = {}
        self.feature_dims["lon"] = self.lon
        self.feature_dims["lat"] = self.lat
        self.flat_shape = (
            np.prod([len(dim) for dim in self.samples_dims.values()]),
            np.prod([len(dim) for dim in self.feature_dims.values()])
        )

    
    def _only_windspeed(func):
        @wraps(func)
        def wrapper_decorator(self, *args, **kwargs):
            if "s" not in self.varnames:
                print("Only valid for absolute wind speed, single pressure level")
                print(self.varname, self.levels)
                raise RuntimeError
            value = func(self, *args, **kwargs)

            return value

        return wrapper_decorator

    @_only_windspeed
    def find_jets(self, **kwargs) -> Tuple:
        ofile_aj = self.path.joinpath("all_jets.pkl")
        ofile_waj = self.path.joinpath("where_are_jets.npy")
        ofile_ajoa = self.path.joinpath("all_jets_one_array.npy")

        if all([ofile.is_file() for ofile in (ofile_aj, ofile_waj, ofile_ajoa)]):
            all_jets = load_pickle(ofile_aj)
            where_are_jets = np.load(ofile_waj)
            all_jets_one_array = np.load(ofile_ajoa)
            return all_jets, where_are_jets, all_jets_one_array
        jet_finder = JetFinder(
            preprocess=preprocess,
            cluster = cluster_wind_speed,
            refine_jets=jets_from_mask,
        )
        all_jets = jet_finder.call(self.ds, **kwargs)
        where_are_jets, all_jets_one_array = all_jets_to_one_array(all_jets)
        save_pickle(all_jets, ofile_aj)
        np.save(ofile_waj, where_are_jets)
        np.save(ofile_ajoa, all_jets_one_array)
        return all_jets, where_are_jets, all_jets_one_array

    @_only_windspeed
    def compute_jet_props(self, processes: int = N_WORKERS, chunksize=2) -> Tuple:
        all_jets, _, _ = self.find_jets(processes, chunksize)
        return compute_all_jet_props(all_jets, self.da, processes, chunksize)

    @_only_windspeed
    def track_jets(self, processes: int = N_WORKERS, chunksize=2) -> Tuple:
        all_jets, where_are_jets, all_jets_one_array = self.find_jets(
            processes, chunksize
        )
        ofile_ajot = self.path.joinpath("all_jets_over_time.pkl")
        ofile_flags = self.path.joinpath("flags.npy")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]):
            all_jets_over_time = load_pickle(ofile_ajot)
            flags = np.load(ofile_flags)

            return (
                all_jets,
                where_are_jets,
                all_jets_one_array,
                all_jets_over_time,
                flags,
            )
        yearbreaks = np.sum(
            self.ds.time.dt.year.values == self.ds.time.dt.year.values[0]
        )
        with NumbaProgress(total=self.ds.time.shape[0]) as progress:
            all_jets_over_time, flags = track_jets(
                all_jets_one_array,
                where_are_jets,
                yearbreaks=yearbreaks,
                progress_proxy=progress,
            )

        save_pickle(all_jets_over_time, ofile_ajot)
        np.save(ofile_flags, flags)

        return all_jets, where_are_jets, all_jets_one_array, all_jets_over_time, flags

    @_only_windspeed
    def props_as_ds(
        self, categorize: bool = True, processes: int = N_WORKERS, chunksize=2
    ) -> xr.Dataset:
        _, where_are_jets, _, _, flags = self.track_jets()
        all_props = self.compute_jet_props(processes, chunksize)
        props_as_ds = props_to_ds(all_props, self.time, where_are_jets.shape[1])
        props_as_ds = add_persistence_to_props(props_as_ds, flags)
        if categorize:
            return categorize_ds_jets(props_as_ds)
        return props_as_ds
