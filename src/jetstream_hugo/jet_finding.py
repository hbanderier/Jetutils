import warnings
from pathlib import Path
from functools import partial
from typing import Tuple
from nptyping import NDArray
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy.stats import linregress
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import csgraph_from_masked, shortest_path
from libpysal.weights import DistanceBand
from tqdm import tqdm
from numba import njit

from jetstream_hugo.definitions import DATERANGEPL_SUMMER, DATERANGEPL_EXT_SUMMER, N_WORKERS, labels_to_mask


def jet_overlap(jet1: NDArray, jet2: NDArray) -> bool:
    _, idx1 = np.unique(jet1[:, 0], return_index=True)
    _, idx2 = np.unique(jet2[:, 0], return_index=True)
    x1, y1 = jet1[idx1, :2].T
    x2, y2 = jet2[idx2, :2].T
    mask12 = np.isin(x1, x2)
    mask21 = np.isin(x2, x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vert_dist = np.mean(np.abs(y1[mask12] - y2[mask21]))
    return (np.mean(mask12) > 0.75) and (vert_dist < 5)


def get_path_from_predecessors(Pr: NDArray, j: int) -> list:
    path = [j]
    k = j
    while Pr[k] != -9999:
        path.append(Pr[k])
        k = Pr[k]
    return path[::-1]


def get_splits(group: NDArray, Pr: NDArray | list, js: NDArray | list, cutoff: float) -> list:
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
        if np.sum(group[path, 2]) > 1900:
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


def define_blobs(da: xr.DataArray, height: float = 25., cutoff: int = 750) -> Tuple[NDArray, list, list]:
    lon, lat = da.lon.values, da.lat.values
    X = da.values
    mask = (X > height)
    idxs = np.where(mask)
    points = np.asarray([lon[idxs[1]], lat[idxs[0]], X[idxs[0], idxs[1]]]).T
    dist_matrix = pairwise_distances(points[:, :2])
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.9, metric="precomputed", linkage="single",).fit(dist_matrix).labels_
    masks = labels_to_mask(labels)
    groups = [points[mask] for mask in masks.T if np.sum(mask) > cutoff]
    dist_mats = [dist_matrix[mask, :][:, mask] for mask in masks.T if np.sum(mask) > cutoff]
    return lon, groups, dist_mats


def find_jets_v32(da: xr.DataArray, height: float = 30, cutoff_blobs: int = 750, cutoff_jets: float = 2400) -> list:
    lon, groups, dist_mats = define_blobs(da, height, cutoff_blobs)
    maxX = da.max().item()
    jets = []
    for group, dist_mat in zip(groups, dist_mats):
        npoint = group.shape[0] // 100 * 6
        xi, xj = last_elements(dist_mat, npoint)
        extremities = np.unique(np.append(xi, xj))
        eccentricity = np.abs(group[extremities, 0] - np.mean(lon))
        idx_max_speed = np.argmax(eccentricity * group[extremities, 2])
        im = extremities[idx_max_speed]
        
        is_nei = DistanceBand(group[:, [0, 1]], 0.9).full()[0].astype(bool)
        s1, s2 = group[:, None, 2], group[None, :, 2]
        x = np.clip(1 - (s1 + s2) / maxX / 2.1, a_min=0, a_max=1)
        weights = 1 - np.exp(-x ** 2 / 2)
        weights_masked = np.ma.array(weights, mask=~is_nei)
        graph = csgraph_from_masked(weights_masked)
        
        dmat, Pr = shortest_path(graph, directed=False, return_predecessors=True, indices=im)
        furthest = last_elements(dmat, 700, True)[::-1]
        furthest = furthest[np.argsort(np.abs(group[furthest, 0] - group[im, 0]) * group[furthest, 2])][::-1]      
        splits = get_splits(group, Pr.copy(), furthest, cutoff_jets)
        thesejets = []
        for k in splits:
            sp = get_path_from_predecessors(Pr, k)
            jet = group[sp]
            if np.sum(jet[:, 2]) > cutoff_jets:
                thesejets.append(jet)
        if not thesejets:
            continue
        thesejets = [thesejets[i] for i in np.argsort([np.sum(jet[:, 2]) for jet in thesejets])[::-1]]
        jets.append(thesejets[0])
        for otherjet in thesejets[1:]:
            if not jet_overlap(thesejets[0], otherjet):
                jets.append(otherjet)
    return jets


def find_all_jets(
    da: xr.DataArray,
    processes: int = N_WORKERS,
    chunksize: int = 20,
    **kwargs,
) -> Tuple[list, xr.DataArray]:
    func = partial(find_jets_v32, **kwargs)
    with Pool(processes=processes) as pool:
        alljets = list(tqdm(pool.imap(func, da, chunksize=chunksize), total=da.shape[0]))
    return alljets


def jet_trapz(jet: NDArray) -> float:
    return np.trapz(jet[:, 2], dx=np.mean(np.abs(np.diff(jet[:, 0]))))  # will fix the other one soon


def jet_integral(jet: NDArray) -> float:
    path = np.append(0, np.sqrt(np.diff(jet[:, 0]) ** 2 + np.diff(jet[:, 1]) ** 2))
    return np.trapz(jet[:, 2], x=np.cumsum(path))


def get_jet_width(x, y, s, da) -> Tuple[NDArray, NDArray]:
    lat = da.lat.values
    half_peak_mask = (da.loc[:, x] < s[None, :] / 2).values
    half_peak_mask[[0, -1], :] = True # worst case i clip the width at the top of the image
    below_mask = lat[:, None] <= y[None, :]
    above_mask = lat[:, None] >= y[None, :]
    below = y - lat[len(lat) - np.argmax((half_peak_mask & below_mask)[::-1], axis=0) - 1]
    above = lat[np.argmax(half_peak_mask & above_mask, axis=0)] - y
    return below, above


def compute_jet_props(jets: list, da: xr.DataArray) -> list:
    props = []
    
    for jet in jets:
        x, y, s = jet.T
        dic = {}
        dic["mean_lon"] = np.average(x, weights=s)
        dic["mean_lat"] = np.average(y, weights=s)
        dic["is_polar"] = dic["mean_lat"] > 45
        maxind = np.argmax(s)
        dic["Lon"] = x[maxind]
        dic["Lat"] = y[maxind]
        dic["Spe"] = s[maxind]
        dic["lon_ext"] = np.amax(x) - np.amin(x)
        dic["lat_ext"] = np.amax(y) - np.amin(y)
        slope, _, r_value, _, _ = linregress(x, y)
        dic["tilt"] = slope
        dic["sinuosity"] = 1 - r_value ** 2
        above, below = get_jet_width(x, y, s, da)
        dic["width"] = np.mean(above + below + 1)
        try:
            dic["int_over_europe"] = jet_integral(jet[x > -10])
        except ValueError:
            dic["int_over_europe"] = 0
        dic["int"] = jet_integral(jet)
        props.append(dic)
    return props


def unpack_compute_jet_props(args): # No such thing as starimap
    return compute_jet_props(*args)


def compute_all_jet_props(
    all_jets: list,
    da: xr.DataArray,
    processes: int = N_WORKERS,
    chunk_size: int = 50,
) -> list:
    with Pool(processes=processes) as pool:
        all_props = list(tqdm(pool.imap(unpack_compute_jet_props, zip(all_jets, da), chunksize=chunk_size), total=len(all_jets)))
    return all_props
    

def props_to_ds(all_props: list, time: NDArray | xr.DataArray = None, maxnjet: int = 4) -> xr.Dataset:
    if time is None:
        time = DATERANGEPL_SUMMER
    try:
        time_name = time.name
        time = time.values
    except AttributeError:
        time_name = 'time'
    assert len(time) == len(all_props)
    varnames = list(all_props[0][0].keys())
    ds = {}
    for varname in varnames:
        ds[varname] = ((time_name, 'jet'), np.zeros((len(time), maxnjet)))
        ds[varname][1][:] = np.nan
        for t in range(len(all_props)):
            for i in range(maxnjet):
                try:
                    props = all_props[t][i]
                except IndexError:
                    break
                ds[varname][1][t, i] = props[varname]
    ds = xr.Dataset(
        ds, 
        coords={time_name: time, 'jet': np.arange(maxnjet)}
    )
    return ds


def props_to_np(props_as_ds: xr.Dataset) -> NDArray:
    props_as_ds_interpolated = props_as_ds.interpolate_na(dim='time', method='linear', fill_value="extrapolate")
    props_as_np = np.zeros((len(props_as_ds.time), len(props_as_ds.jet) * len(props_as_ds.data_vars)))
    i = 0
    for varname in props_as_ds.data_vars:
        for jet in props_as_ds.jet:
            props_as_np[:, i] = props_as_ds_interpolated[varname].sel(jet=jet).values
            i = i + 1
    return props_as_np


def better_is_polar(all_jets:list, props_as_ds: xr.Dataset, exp_low_path: Path) -> xr.Dataset:
    this_path = exp_low_path.joinpath('better_is_polar.nc')
    if this_path.is_file():
        props_as_ds['int_low'] = xr.open_dataarray(exp_low_path.joinpath('int_low.nc'))
        props_as_ds['is_polar'] = xr.open_dataarray(this_path)
        return props_as_ds
    print('computing int low')
    da_low = xr.open_dataarray(exp_low_path.joinpath('da.nc'))
    props_as_ds['int_low'] = props_as_ds['mean_lon'].copy()
    for it, (jets, mean_lats) in tqdm(enumerate(zip(all_jets, props_as_ds['mean_lat'])), total=len(all_jets)):
        for j, (jet, mean_lat) in enumerate(zip(jets, mean_lats.values)):
            x, y, s = jet.T
            x_ = xr.DataArray(x, dims='points')
            y_ = xr.DataArray(y, dims='points')
            s_low = da_low[it].sel(lon=x_, lat=y_).values
            jet_low = np.asarray([x, y, s_low]).T
            props_as_ds['int_low'][it, j] = jet_integral(jet_low).item()
            
    props_as_ds['is_polar'] = (props_as_ds['mean_lat'] * 200 - props_as_ds['mean_lon'] * 30 + props_as_ds['int_low']) > 9000
    props_as_ds['int_low'].to_netcdf(exp_low_path.joinpath('int_low.nc'))
    props_as_ds['is_polar'].to_netcdf(this_path)
    del props_as_ds['int_low']
    return props_as_ds


def categorize_ds_jets(props_as_ds: xr.Dataset):
    time_name, time_val = list(props_as_ds.coords.items())[0]
    ds = xr.Dataset(coords={time_name: time_val, 'jet': ['subtropical', 'polar']})
    for varname in props_as_ds.data_vars:
        if varname == 'is_polar':
            continue
        cond = props_as_ds['is_polar']
        values = np.zeros((len(time_val), 2))
        values[:, 0] = props_as_ds[varname].where(1 - cond).mean(dim='jet').values
        values[:, 1] = props_as_ds[varname].where(cond).mean(dim='jet').values
        ds[varname] = ((time_name, 'jet'), values)
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


def compute_all_overlaps(all_jets: list, props_as_ds: xr.Dataset) -> Tuple[NDArray, NDArray]:
    overlaps = np.full(len(all_jets), np.nan)
    vert_dists = np.full(len(all_jets), np.nan)
    time = props_as_ds.time.values
    for i, (jets, are_polar) in enumerate(zip(all_jets, props_as_ds['is_polar'].values)):
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
    overlaps = xr.DataArray(overlaps, coords={'time': time})
    vert_dists = xr.DataArray(vert_dists, coords={'time': time})
    return overlaps, vert_dists
    
    
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


@njit
def track_jets(all_jets_one_array, where_are_jets, progress_proxy=None):
    factor: float = 0.2
    yearbreaks = 92 if where_are_jets.shape[0] < 10000 else 92 * 4 # find better later
    guess_nflags: int = 6000
    all_jets_over_time = np.full(
        (guess_nflags, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32
    )
    last_valid_idx = np.full(guess_nflags, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full(where_are_jets.shape[:2], fill_value=guess_nflags, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    flags[0, :last_flag + 1] = np.arange(last_flag + 1)
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
        potentials = (
            from_
            + np.where(
                isin(times_to_test, [t, t - 1, t - 2, t - 3])
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
    names = tuple(ds_props.coords.keys())
    dt1 = (ds_props.time.values[1] - ds_props.time.values[0]).astype('timedelta64[h]')
    dt2 = np.timedelta64(1, 'D') 
    factor = float(dt1 / dt2)
    num_jets = ds_props['mean_lon'].shape[1]
    jet_persistence_prop = flags[:, :num_jets].copy().astype(float)
    nan_flag = np.amax(flags)
    unique_flags, jet_persistence = np.unique(flags, return_counts=True)
    for i, flag in enumerate(unique_flags[:-1]):
        jet_persistence_prop[flags[:, :num_jets] == flag] = jet_persistence[i] * factor
    jet_persistence_prop[flags[:, :num_jets] == nan_flag] = np.nan
    ds_props['persistence'] = (names, jet_persistence_prop)
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