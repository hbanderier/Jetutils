import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from numba import njit, jit


from tqdm.auto import tqdm, trange
basepath = Path(DATADIR).joinpath("ERA5/plev")
for year in tqdm(YEARS):
    year_str = str(year).zfill(4)
    for month in trange(1, 13):
        outpath = basepath.joinpath(f"s/6H/{year_str}{month}.nc")
        if outpath.is_file():
            continue
        month = str(month).zfill(2)
        u = xr.open_dataarray(basepath.joinpath(f"u/6H/{year_str}{month}.nc"))
        v = xr.open_dataarray(basepath.joinpath(f"v/6H/{year_str}{month}.nc"))
        s = np.sqrt(u ** 2 + v ** 2)
        s.to_netcdf(basepath.joinpath(f"s/6H/{year_str}{month}.nc"))

@jit
def track_jets(all_jets, all_props):
    factor = 0.2
    yearbreaks: int = 92
    i0 = 0
    il = len(all_jets)
    all_jets_over_time = [None] * len(all_jets[0])
    for j, (jet, prop) in enumerate(zip(all_jets[i0], all_props[i0])):
        all_jets_over_time[j] = [j, {i0: (jet, prop)}]
    flags = [list(range(len(all_jets_over_time)))]
    last_flag = flags[0][-1]
    for tp, jets in enumerate(all_jets[i0 + 1:il]): # can't really parallelize
        t = tp + i0
        flags.append([0] * len(jets))
        potentials = np.empty(len(all_jets_over_time), dtype=bool)
        for j, jtt in enumerate(all_jets_over_time):
            cond1 = t in jtt[1]
            cond2 = (t - 1 in jtt[1]) and ((t + 1) // yearbreaks == (t - 1) // yearbreaks)
            cond3 = (t - 1 in jtt[1]) and ((t + 1) // yearbreaks == (t - 1) // yearbreaks)
            potentials[j] = cond1 or cond2 or cond3
        jets_to_try = np.nonzero(potentials)[0]
        dist_mat = np.zeros((len(jets_to_try), len(jets)))
        if (t + 1) % yearbreaks == 0:
            dist_mat = np.full((len(jets_to_try), len(jets)), fill_value=10 * factor)
        else:
            for i, jtt_idx in enumerate(jets_to_try):
                jet_to_try = all_jets_over_time[jtt_idx]
                
                jet_to_try_ = list(jet_to_try[1].values())[-1][0][:, :2]
                for j, jet in enumerate(jets):
                    distances = haversine_distances(np.radians(jet_to_try_), np.radians(jet[:, :2])) 
                    dist_mat[i, j] = np.mean([
                        np.sum(np.amin(distances / len(jet_to_try_), axis=1)), 
                        np.sum(np.amin(distances / len(jet), axis=0))
                    ])
        connected_mask = dist_mat < factor
        flagged = [1000] # can't have empty lists
        for i, jtt_idx in enumerate(jets_to_try):
            jet_to_try = all_jets_over_time[jtt_idx]
            js = np.argsort(dist_mat[i])
            for j in js:
                if not connected_mask[i, j]:
                    break
                if j in flagged:
                    continue
                jet_to_try[1][t + 1] = (jets[j], all_props[t + 1][j])
                flagged.append(j)
                flags[-1][j] = jet_to_try[0]
                break
                
            
        for j, jet in enumerate(jets):
            if j not in flagged:
                last_flag += 1
                all_jets_over_time.append([last_flag, {t + 1: (jet, all_props[t + 1][j])}]) 
                flags[-1][j] = last_flag
    return all_jets_over_time, flags


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
        result[i] = np.amin(a_)
    return result

@njit
def amin_ax1(a):
    result = np.zeros(a.shape[0])
    for i, a_ in enumerate(a):
        result[i] = np.amin(a_)
    return result

@njit
def track_jets_numba(all_jets_one_array, where_are_jets, maxnjets, num_indiv_jets):
    factor: float = 0.2
    yearbreaks: int = 92
    all_jets_over_time = np.full((12000, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32)
    last_valid_idx = np.full(12000, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full((len(where_are_jets), maxnjets), fill_value=num_indiv_jets, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    for t, jet_idxs in enumerate(where_are_jets[1:]): # can't really parallelize
        potentials = np.zeros(50, dtype=np.int32)
        from_ = max(0, last_flag - 30)
        times_to_test = np.take_along_axis(all_jets_over_time[from_:last_flag + 1, :, 0], last_valid_idx[from_:last_flag + 1, None], axis=1).flatten()
        potentials = from_ + np.where(
            isin(times_to_test, [t, t - 1, t - 2]) &
            ((times_to_test // yearbreaks) == ((t + 1) // yearbreaks)).astype(np.bool_)
        )[0]
        num_valid_jets = np.sum(jet_idxs[:, 0] >= 0)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        for i, jtt_idx in enumerate(potentials):
            t_jtt, j_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            k_jtt, l_jtt = where_are_jets[t_jtt, j_jtt]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            for j in range(num_valid_jets):
                k, l = jet_idxs[j]
                jet = all_jets_one_array[k:l, :2]
                distances = np.sqrt(np.sum((np.radians(jet_to_try)[None, :, :] - np.radians(jet)[:, None, :]) ** 2, axis=-1))
                # distances = haversine_distances(np.radians(jet_to_try), np.radians(jet)) 
                dist_mat[i, j] = np.mean(np.array([
                    np.sum(amin_ax1(distances / len(jet_to_try))), 
                    np.sum(amin_ax0(distances / len(jet)))
                ]))
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
                flags[t + j, j] = jtt_idx
                break
                
        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                all_jets_over_time[last_flag, 0, :] = (t + 1, j)
                last_valid_idx[last_flag] = 0
                flags[t + j, j] = last_flag
                flagged[j] = True

    return all_jets_over_time, flags


def track_jets_numpy(all_jets_one_array, where_are_jets, maxnjets, num_indiv_jets): # no numba, numpy version
    factor: float = 0.2
    yearbreaks: int = 92
    n_jet_guess = 2000
    all_jets_over_time = np.full((n_jet_guess, yearbreaks, 2), fill_value=len(where_are_jets), dtype=np.int32)
    last_valid_idx = np.full(n_jet_guess, fill_value=yearbreaks, dtype=np.int32)
    for j in range(np.sum(where_are_jets[0, 0] >= 0)):
        all_jets_over_time[j, 0, :] = (0, j)
        last_valid_idx[j] = 0
    flags = np.full((len(where_are_jets), maxnjets), fill_value=num_indiv_jets, dtype=np.int32)
    last_flag = np.sum(where_are_jets[0, 0] >= 0) - 1
    for t, jet_idxs in enumerate(where_are_jets[1:]): # can't really parallelize
        potentials = np.zeros(50, dtype=np.int32)
        from_ = max(0, last_flag - 30)
        times_to_test = np.take_along_axis(all_jets_over_time[from_:last_flag + 1, :, 0], last_valid_idx[from_:last_flag + 1, None], axis=1).flatten()
        potentials = from_ + np.where(
            np.isin(times_to_test, [t, t - 1, t - 2]) &
            ((times_to_test // yearbreaks) == ((t + 1) // yearbreaks)).astype(np.bool_)
        )[0]
        num_valid_jets = np.sum(jet_idxs[:, 0] >= 0)
        dist_mat = np.zeros((len(potentials), num_valid_jets), dtype=np.float32)
        for i, jtt_idx in enumerate(potentials):
            t_jtt, j_jtt = all_jets_over_time[jtt_idx, last_valid_idx[jtt_idx]]
            k_jtt, l_jtt = where_are_jets[t_jtt, j_jtt]
            jet_to_try = all_jets_one_array[k_jtt:l_jtt, :2]
            for j in range(num_valid_jets):
                k, l = jet_idxs[j]
                jet = all_jets_one_array[k:l, :2]
                distances = haversine_distances(np.radians(jet_to_try), np.radians(jet)) 
                dist_mat[i, j] = np.mean(np.array([
                    np.sum(np.amin(distances / len(jet_to_try), axis=1)), 
                    np.sum(np.amin(distances / len(jet)), axis=0)
                ]))
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
                flags[t + j, j] = jtt_idx
                break
                
        for j in range(num_valid_jets):
            if not flagged[j]:
                last_flag += 1
                all_jets_over_time[last_flag, 0, :] = (t + 1, j)
                last_valid_idx[last_flag] = 0
                flags[t + j, j] = last_flag
                flagged[j] = True

    return all_jets_over_time, flags


def extract_props_over_time_old(jet):
    timesteps = np.asarray(list(jet.keys()))
    varnames = list(jet[timesteps[0]][1].keys())
    props_over_time = {varname: np.zeros(len(timesteps)) for varname in varnames}
    for varname in varnames:
        for t, timestep in enumerate(timesteps):
            props_over_time[varname][t] = jet[timestep][1][varname]
    return props_over_time


def jet_overlap(jet1: NDArray, jet2: NDArray) -> bool:
    x1, y1 = jet1[:, :2].T
    x2, y2 = jet2[:, :2].T
    mask12 = np.isin(x1, x2)
    mask21 = np.isin(x2, x1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vert_dist = np.mean(np.abs(y1[mask12] - y2[mask21]))
    return (np.mean(mask12) > 0.75) and (vert_dist < 5)

def concat_levels(jets1: list, jets2: list) -> Tuple[list, list]:
    jets = jets1.copy()
    added = []
    for jet in jets1:
        for j, otherjet in enumerate(jets2):
            if not jet_overlap(jet, otherjet):
                jets.append(otherjet)
                added.append(j)
    return jets, added
    
def check_polar(high_jets: list, low_jets: list) -> NDArray:  
    # is_polar = [np.average(jet[:, 1], weights=jet[:, -1]) > 45 for jet in high_jets]
    is_polar = np.zeros(len(high_jets), dtype=bool)
    for i, jet in enumerate(high_jets):
        for other_jet in low_jets:
            if jet_overlap(jet, other_jet) or (np.average(jet[:, 1], weights=jet[:, -1]) > 50):
                is_polar[i] = True
    return is_polar




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


# def compute_all_overlaps(
#     all_jets: list, props_as_ds: xr.Dataset
# ) -> Tuple[NDArray, NDArray]:
#     overlaps = np.full(len(all_jets), np.nan)
#     vert_dists = np.full(len(all_jets), np.nan)
#     time = props_as_ds.time.values
#     for i, (jets, are_polar) in enumerate(
#         zip(all_jets, props_as_ds["is_polar"].values)
#     ):
#         nj = min(len(are_polar), len(jets))
#         if nj < 2 or sum(are_polar[:nj]) == nj or sum(are_polar[:nj]) == 0:
#             continue
#         polars = []
#         subtropicals = []
#         for jet, is_polar in zip(jets, are_polar):
#             if is_polar:
#                 polars.append(jet)
#             else:
#                 subtropicals.append(jet)
#         polars = np.concatenate(polars, axis=0)
#         subtropicals = np.concatenate(subtropicals, axis=0)
#         overlaps[i], vert_dists[i] = jet_overlap_values(polars, subtropicals)
#     overlaps = xr.DataArray(overlaps, coords={"time": time})
#     vert_dists = xr.DataArray(vert_dists, coords={"time": time})
#     return overlaps, vert_dists


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