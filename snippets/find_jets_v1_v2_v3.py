import math
import networkx as nx
from itertools import pairwise, permutations, combinations


def compute_distance_matrix_(points):
    points_weighed = points[:, [1, 0]]
    dist_matrix = haversine_distances(np.radians(points_weighed))
    if points.shape[1] <= 3:
        return dist_matrix
    return np.sqrt(dist_matrix ** 2 + (points[:, None, 2] - points[None, :, 2]) ** 2)


def find_jets_2D_(points, eps, kind="AgglomerativeClustering"):
    dist_matrix = compute_distance_matrix_(points)
    if kind == "DBSCAN":
        model = DBSCAN(
            eps=eps,
            metric="precomputed",
        )
    elif kind == "AgglomerativeClustering":  # strategy pattern ?
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=eps,
            metric="precomputed",
            linkage="single",
        )
    elif kind == "HDBSCAN":  # strategy pattern ?
        model = HDBSCAN(
            min_cluster_size=100,
            cluster_selection_epsilon=eps,
            metric='precomputed',
        )
    elif kind == "OPTICS":  # strategy pattern ?
        model = OPTICS(
            metric='precomputed',
        )
    labels = model.fit(dist_matrix).labels_
    masks = labels_to_mask(labels)
    return labels, masks


def find_jets(
    X,
    lon: NDArray,
    lat: NDArray,
    height=20,
    distance=20,
    width=6,
    grad=4,
    juncdistlo=5,
    juncdistla=6,
    cutoff=60,
) -> list:
    jets = []
    for i, x in enumerate(X.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=distance, width=width)[0]
        for la, sp in zip(lat[peaks], x[peaks]):
            found = False
            for jet in jets:
                if (np.abs(jet[-1][0] - lo) + np.abs(jet[-1][1] - la)) <= grad:
                    jet.append((lo, la, sp))
                    found = True
                    break
            if not found:
                jets.append([(lo, la, sp)])

    jets = [np.asarray(jet) for jet in jets]
    jets = [jet[np.argsort(jet[:, 0])] for jet in jets]
    done = False
    while not done:
        done = True
        njets = len(jets)
        for i1, i2 in permutations(range(njets), 2):
            jet1, jet2 = jets[i1], jets[i2]
            dlo1 = np.abs(jet1[-1][0] - jet2[0][0])
            dla1 = np.abs(jet1[-1][1] - jet2[0][1])
            dlo2 = np.abs(jet1[-1][0] - jet2[-1][0])
            dla2 = np.abs(jet1[-1][1] - jet2[-1][1])
            if dlo1 <= juncdistlo and dla1 <= juncdistla:
                jets[i1] = np.concatenate([jet1, jet2], axis=0)
                jets[i1] = jets[i1][np.argsort(jets[i1][:, 0])]
                del jets[i2]
                done = False
                break
            if dlo2 <= juncdistlo and dla2 <= juncdistla:
                jets[i1] = np.concatenate([jet1, jet2[::-1]], axis=0)
                jets[i1] = jets[i1][np.argsort(jets[i1][:, 0])]
                del jets[i2]
                done = False
                break

    for i in range(len(jets) - 1, -1, -1):
        if len(jets[i]) < cutoff:
            del jets[i]

    return jets


def compute_distance_matrix_(points, weights=None):
    if weights is None:
        weights = [1, 1]
    points_weighed = points[:, [1, 0]] * np.asarray(weights)[None, :]
    return haversine_distances(np.radians(points_weighed))


def find_jets_v2_(
    points,
    eps,
    weights=None,
    kind = 'AgglomerativeClustering'
):
    dist_matrix = compute_distance_matrix_(points, weights=weights)
    if kind == 'DBSCAN':
        model = DBSCAN(
            eps=eps,
            metric="precomputed",
        )
    elif kind == 'AgglomerativeClustering': # strategy pattern ?
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=eps,
            metric="precomputed",
            linkage='single'
        )
    labels = model.fit(dist_matrix).labels_
    masks = labels_to_mask(labels)
    return [points[mask] for mask in masks.T]


def find_jets_v2(
    X,
    lon: NDArray,
    lat: NDArray,
    height=20,
    eps=3,
    cutoff=1900,
    weights=None,
    kind: str = 'DBSCAN',
    remove_outliers: int = 0,
) -> list:
    points = []
    res = lon[1] - lon[0]
    cutoff = cutoff / res
    for i, x in enumerate(X.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=10, width=1)[0]
        for peak in peaks:
            for plus in [-2, -1, 0, 1, 2]:
                try:
                    if x[peak + plus] > height:
                        points.append([lo, lat[peak + plus], x[peak + plus]])
                except IndexError:
                    pass
    # for j, x in enumerate(X):
    #     if j % 2 == 1:
    #         continue
    #     la = lat[j]
    #     peaks = find_peaks(x, height=height, distance=20000, width=5)[0]
    #     for peak in peaks:
    #         for plus in [-2, -1, 0, 1, 2]:
    #             try:
    #                 if x[peak + plus] > height:
    #                     points.append([lon[peak + plus], la, x[peak + plus]])
    #             except IndexError:
    #                 pass
    if len(points) == 0:
        return []
    points = np.atleast_2d(points)
    argsort = np.argsort(points[:, 0])
    points = points[argsort, :]
    dist_matrix = compute_distance_matrix_(points, weights=[1, 1])
    eps  = eps * np.radians(res)
    if remove_outliers:
        
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=np.radians(res) * 2.1, metric='precomputed', linkage='single').fit(dist_matrix).labels_
        masks = labels_to_mask(labels)
        groups = [points[mask] for mask in masks.T]
        
        points = np.concatenate([group for group in groups if len(group) > remove_outliers / res])
        argsort = np.argsort(points[:, 0])
        points = points[argsort, :]
    
    potential_jets = find_jets_v2_(points, eps, weights=weights, kind=kind)    
    k = 0
    this_eps = eps
    for i, jet in enumerate(potential_jets):
        jet = potential_jets[i]
        strength = np.sum(jet[:, 2])
        strong_enough = strength > cutoff
        
        diffx = np.abs(np.diff(jet[:, 0]))
        diffy = np.abs(np.diff(jet[:, 1]))
        jumpsx = np.sum(diffx[diffx > 2 * res])
        jumpsy = np.sum(diffy[diffy > 6 * res])
        too_split = jumpsx > 4 * res or jumpsy > 6 * 10 * res
        if not strong_enough:
            potential_jets[i] = None
        if all((j is None for j in potential_jets)):
            break
        if strong_enough and too_split and this_eps >= min_eps:
            k += 1
            this_eps = eps / (k / 2 + 1)
            potential_jets[i] = None
            potential_jets.extend(find_jets_v2_(jet, this_eps, kind))
        print(k, f'{this_eps:.2f}', f'{jumpsx:.2f}', f'{jumpsy:.2f}', f'{strength:.2f}')    
    jets = [j for j in potential_jets if j is not None]
    return jets


def weight(X, i, j, k, l, maxX):
    s1 = X[i, j]
    s2 = X[k, l]
    x = 1 - (s1 + s2) / maxX / 2
    return 1 - np.exp(-x ** 2 / 0.5)


def is_nei(i, j, k, l):
    return abs(k - i) <= 1 and abs(l - j) <= 1


def find_jets_v3(X, lon, lat, len_cutoff=20):
    sps = []
    maxX = np.amax(X)
    mask = (X > 20).ravel(order='F')
    grid = np.stack(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[1])), axis=-1).reshape(-1, 2)
    masked_grid = grid[mask]
    labels = DBSCAN(eps=math.sqrt(2) * 1.1, metric='euclidean').fit(masked_grid).labels_
    masks = labels_to_mask(labels)
    groups = [masked_grid[mask] for mask in masks.T if np.sum(mask) > len_cutoff]
    for group in groups:
        G = nx.empty_graph(0)

        G.add_nodes_from((i, j) for (i, j) in group)
        G.add_weighted_edges_from(((i, j), (k, l), weight(X, i, j, k, l, maxX)) for ((i, j), (k, l)) in combinations(group, 2) if is_nei(i, j, k, l))

        minlon, maxlon = np.amin(group[:, 1]), np.amax(group[:, 1])
        lats1 = group[group[:, 1] == minlon, 0]
        edges1 = [0, *np.where(np.diff(lats1) > 6)[0] + 1, len(lats1) - 1]
        origins = []
        for i1, i2 in pairwise(edges1):
            origins.append((lats1[i1 + np.argmax(X[lats1[i1:i2 + 1], minlon])], minlon))
            
        lats2 = group[group[:, 1] == maxlon, 0]
        edges2 = [0, *np.where(np.diff(lats2) > 6)[0] + 1, len(lats2) - 1]
        dests = []
        for i1, i2 in pairwise(edges2):
            dests.append((lats2[i1 + np.argmax(X[lats2[i1:i2 + 1], maxlon])], maxlon))
            
        for orig, dest in product(origins, dests):
            sp = np.asarray(nx.shortest_path(G, orig, dest, weight='weight'))
            sp = np.asarray([lon[sp[:, 1]], lat[sp[:, 0]], X[sp[:, 0], sp[:, 1]]]).T
            sps.append(sp)
    return sps


def update_jet(jet: NDArray):
    x, y, s = jet.T
    x_, c_ = np.unique(x, return_index=True)
    s_split = np.split(s, c_[1:])
    y_ = np.empty(len(s_split))
    s_ = np.empty(len(s_split))
    for i, sp in enumerate(s_split):
        j = np.argmax(sp) + c_[i]
        y_[i] = y[j]
        s_[i] = s[j]
    return np.stack([x_, y_, s_], axis=-1)


def find_jets_2D(
    da: xr.DataArray, height=15, eps: float = 10, sigmas: Iterable = None, cutoff: float = 1500, kind: str = 'DBSCAN'
) -> list:
    if sigmas is None:
        sigmas = range(8, 24, 5)
    lon, lat = da.lon.values, da.lat.values
    res = lon[1] - lon[0]
    cutoff = cutoff / res
    X = da.values
    Xmax = X.max()
    X_norm = X / Xmax
    X_prime = frangi(X_norm, black_ridges=False, sigmas=sigmas, cval=1) * Xmax
    points = []
    for i, x in enumerate(X_prime.T):
        lo = lon[i]
        peaks = find_peaks(x, height=height, distance=10, width=2)[0]
        for peak in peaks:
            points.append([lo, lat[peak], X[peak, i]])
    for j, x in enumerate(X_prime):
        la = lat[j]
        peaks = find_peaks(x, height=height, distance=10, width=2)[0]
        for peak in peaks:
            points.append([lon[peak], la, X[j, peak]])
    if len(points) == 0:
        return []
    points = np.atleast_2d(points)
    argsort = np.argsort(points[:, 0])
    points = points[argsort, :]
    eps = eps * np.radians(res)
    labels, masks = find_jets_2D_(points, eps=eps, kind=kind) 
    potential_jets = [points[mask] for mask in masks.T]

    jets = [jet for jet in potential_jets if np.sum(jet[:, 2]) > cutoff]
    sorted_order = np.argsort(
        [np.average(jet[:, 1], weights=jet[:, 2]) for jet in jets]
    )
    return [update_jet(jets[i]) for i in sorted_order]


def find_jets_3D(da: xr.DataArray, height=20, eps: float = 10, sigmas: Iterable = None, vert_factor=2, kind: str = 'HDBSCAN'):
    if sigmas is None:
        sigmas = range(10, 21, 5)
    alts = vert_factor * metpy.calc.pressure_to_height_std(da.lev).values
    lon, lat = da.lon.values, da.lat.values
    all_points = []
    for il, X in enumerate(da.values):
        points = []
        Xmax = X.max()
        X_norm = X / Xmax
        X_prime = frangi(X_norm, black_ridges=False, sigmas=range(10, 21, 5), cval=1) * Xmax
        for i, x in enumerate(X_prime.T):
            lo = lon[i]
            peaks = find_peaks(x, height=height, distance=10, width=1)[0]
            for peak in peaks:
                points.append([lo, lat[peak], alts[il], X[peak, i]])
        for j, x in enumerate(X_prime):
            la = lat[j]
            peaks = find_peaks(x, height=height, distance=20000, width=5)[0]
            for peak in peaks:
                points.append([lon[peak], la, alts[il], X[j, peak]])
        points = np.atleast_2d(points)
        argsort = np.argsort(points[:, 0])
        points = points[argsort, :]
        all_points.append(points)
    all_points = np.concatenate(all_points)
    
    labels, masks = find_jets_2D_(all_points, eps, kind=kind)
        
    split_idx = 1 + np.where(np.diff(all_points[:, 2]))[0]
    split_labels = np.split(labels, split_idx)
    max_label = np.amax(labels)
    for il, labels in enumerate(split_labels):
        unique, idxs, counts = np.unique(
            labels, return_counts=True, return_index=True
        )
        for un, co, idx in zip(unique, counts, idxs):
            if (co > 30) or (un == -1):
                continue
            split_labels[il][idx] = -1
            max_label = max_label + 1
            if il == 0:
                continue
            for iln in range(il + 1, len(split_labels)):
                split_labels[iln][split_labels[iln] == un] = max_label
                
    split_points = np.split(all_points, split_idx)
    split_masks = [labels_to_mask(labels) for labels in split_labels]  
    jets = [[points[mask] for mask in masks.T] for masks, points in zip(split_masks, split_points)]
    return jets, split_labels 


def define_blobs_generic(
    criterion: xr.DataArray,
    *append_to_groups: xr.DataArray,
    criterion_threshold: float = 0,
    distance_function: Callable = pairwise_distances,
    distance_threshold: float = 0.75,
    min_size: int = 50,
) -> Tuple[list, list]:
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
    dist_matrix = distance_function(points[["lon", "lat"]].to_numpy())
    if isinstance(dist_matrix, tuple):
        real_dist_mat, dist_matrix = dist_matrix
    else:
        real_dist_mat = dist_matrix.copy()
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
    dist_mats = [real_dist_mat[mask, :][:, mask] for mask in valid_masks]
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
        ds["lev"] if "lev" in ds else None,
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
        ds["criterion"],
        ds["lev"] if "lev" in ds else None,
        criterion_threshold=-criterion_threshold,
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        min_size=min_size,
    )
    
    
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
        if (-180 in ux) and (179.5 in ux):
            first = ux[np.argmax(np.diff(np.append([ux[-1]], ux)))]
            x[x < first] += 360
        else:
            first = np.amin(ux)
        xmin, xmax = np.amin(x), np.amax(x)
        ymin, ymax = np.amin(y), np.amax(y)
        dy = 0.5
        dx = 0.5
        xmesh, ymesh = np.meshgrid(
            np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy)
        )
        distance_to_line_2 = np.abs(a * xmesh - ymesh + b) / np.sqrt(a**2 + 1)
        mask = distance_to_line_2 <= (maxdist)
        xmesh[xmesh >= 180] -= 360
        append_names = [col for col in group.columns if col not in ["lon", "lat"]]
        append_to_mesh = [ds[col] for col in append_names]
        if "lev" in ds.dims:
            maxlev = ds["s_smo"].argmax(dim="lev")
            append_to_mesh = [atm.isel(lev=maxlev) for atm in append_to_mesh]
        append_to_mesh = [slice_1d(atm, [ymesh[mask], xmesh[mask]]) for atm in append_to_mesh]
        mesh = np.asarray([xmesh[mask], ymesh[mask], *append_to_mesh]).T
        mesh = pd.DataFrame(mesh, columns=group.columns)
        grid_to_mesh = np.argmax(
            (xmesh[mask][:, None] == x[None, :])
            & (ymesh[mask][:, None] == y[None, :]),
            axis=0,
        )
        dist_mat = pairwise_distances(np.asarray([xmesh[mask], ymesh[mask]]).T)
        is_nei = (dist_mat > 0) & (dist_mat < 1)
        weights = compute_weights(ds, mesh, is_nei)
        masked_weights = np.ma.array(weights, mask=~is_nei)
        graph = csgraph_from_masked(masked_weights)
        candidates = np.where(mesh["lon"] == first)[0]
        im = candidates[np.argmax(mesh["s"].iloc[candidates])]
        dmat, Pr = shortest_path(
            graph, directed=False, return_predecessors=True, indices=im
        )
        furthest = grid_to_mesh[np.argsort(dist_mat[im, grid_to_mesh])][::-1]
        splits = get_splits(mesh["s"].to_numpy(), Pr.copy(), furthest, jet_cutoff)
        jets.extend(jets_from_predecessor(mesh, splits, Pr, jet_cutoff))
    return merge_jets(jets, 1.5)


def merge_jets(jets: list[pd.DataFrame], threshold: float = 1.5) -> list[NDArray]:
    to_merge = []
    for (i1, j1), (i2, j2) in pairwise(enumerate(jets)):
        if np.amin(pairwise_distances(j1[["lon", "lat"]].to_numpy(), j2[["lon", "lat"]].to_numpy())) < threshold:
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
        newjets.append(pd.concat([jets[k] for k in merger]))
    return newjets


def create_and_separate_graphs(
    masked_weights: np.ma.array,
    points: pd.DataFrame,
    distance_matrix: NDArray,
    min_size: int = 400,
) -> Tuple[Sequence[csr_matrix], Sequence[pd.DataFrame], list[NDArray]]:
    graph = csgraph_from_masked(masked_weights)
    nco, labels = connected_components(graph)
    masks = labels_to_mask(labels).T
    extremities_ = [get_extremities(mask, distance_matrix) for mask in masks]
    to_merge = []
    for (im1, mask1), (im2, mask2) in pairwise(enumerate(masks)):
        extremities1 = extremities_[im1]
        extremities2 = extremities_[im2]
        this_dist_mat = distance_matrix[extremities1, :][:, extremities2]
        i, j = np.unravel_index(np.argmin(this_dist_mat), this_dist_mat.shape)
        i, j = extremities1[i], extremities2[j]
        if distance_matrix[i, j] < 2:
            to_merge.append([im1, im2])
            masked_weights[i, j] = (
                0.1  # doesn't matter i guess, as long as it's between 0 and 1 excluded
            )
            masked_weights.mask[i, j] = False

    groups = []
    graphs = []
    dist_mats = []
    for im, mask in enumerate(masks):
        if not any([im in merger for merger in to_merge]):
            if np.sum(mask) < min_size:
                continue
            groups.append(points.iloc[mask])
            graphs.append(graph[mask, :][:, mask])
            dist_mats.append(distance_matrix[mask, :][:, mask])
    for merger in to_merge:
        mask = np.sum([masks[k] for k in merger], axis=0).astype(bool)
        if np.sum(mask) < min_size:
            continue
        groups.append(points.iloc[mask])
        graphs.append(csgraph_from_masked(masked_weights[mask, :][:, mask]))
        dist_mats.append(distance_matrix[mask, :][:, mask])
    return graphs, groups, dist_mats


def default_cluster(
    ds: xr.Dataset,
    criterion_threshold: float = 25.0,
    distance_function: Callable = pairwise_distances,
) -> Tuple[list, list, xr.Dataset]:
    print("Use a proper define_blobs step")
    raise ValueError


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


def compute_criterion_spensberger(ds: xr.Dataset, flatten: bool = True) -> xr.Dataset:
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
            ds[varnames["u"]] * ds[varnames["s"]].differentiate("y")
            - ds[varnames["v"]] * ds[varnames["s"]].differentiate("x")
        ) / ds[varnames["s"]]
        Udsigmadn = ds[varnames["u"]] * sigma.differentiate("y") - ds[varnames["v"]] * sigma.differentiate("x")
        ds["sigma"] = sigma
        ds["Udsigmadn"] = Udsigmadn
        ds["criterion_spensberger"] = Udsigmadn + sigma ** 2
        ds["criterion_spensberger"] = ds["criterion_spensberger"].where(ds["criterion_spensberger"] < 0, 0)
        ds["criterion_spensberger"] = ds["criterion_spensberger"].where(np.isfinite(ds["criterion_spensberger"]), 0)
    if flatten and "lev" in ds.dims:
        ds = flatten_by(ds, "-criterion_spensberger")
    return ds.reset_coords(["x", "y"], drop=True)


def compute_criterion_sato(ds: xr.Dataset, flatten: bool = True, **kwargs): 
    if "time" in ds.dims:
        filtered = np.zeros_like(ds["s_smo"])
        for t in trange(len(filtered)):
            da = ds["s_smo"][t]
            filtered[t, :, :] = sato(da / da.max(), black_ridges=False, **kwargs)
            filtered[t] = filtered[t] / filtered[t].max()
    else:
        filtered = sato(
            ds["s_smo"] / ds["s_smo"].max(), black_ridges=False, **kwargs
        )
        filtered = filtered / filtered.max()
    ds["criterion_sato"] = ds["s_smo"].copy(data=filtered)
    if flatten and "lev" in ds.dims:
        ds = flatten_by(ds, "criterion_sato")
    return ds


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


def preprocess(ds: xr.Dataset, flatten_by_var: str | None = "s", smooth_all: bool = False):
    if "u" in ds:
        ds["u"] = ds["u"].where(ds["u"] > 0).interpolate_na("lat", fill_value="extrapolate") # I don't want stratosphere
    if "s" not in ds:
        ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    if flatten_by_var is not None:
        ds = flatten_by(ds, flatten_by_var)
    if -180 not in ds.lon.values:
        ds["s_smo"] = smooth(ds["s"], smooth_map={"lon+lat": ("fft", SMOOTHING)})
        ds["s_smo"] = ds["s_smo"].where(ds["s_smo"] > 0, 0)
        if smooth_all:
            ds["u_smo"] = smooth(ds["u"], smooth_map={"lon+lat": ("fft", SMOOTHING)})
            ds["v_smo"] = smooth(ds["v"], smooth_map={"lon+lat": ("fft", SMOOTHING)})
    else:
        raise NotImplementedError("FIIXX")
        w = VectorWind(ds["u"].fillna(0), ds["v"].fillna(0))
        ds["u_smo"] = w.truncate(w.u(), truncation=84)
        ds["v_smo"] = w.truncate(w.v(), truncation=84)
        ds["s_smo"] = np.sqrt(ds["u_smo"] ** 2 + ds["v_smo"] ** 2)
    return ds


def cluster_generic(
    criterion: xr.DataArray,
    *append_to_groups: xr.DataArray,
    criterion_threshold: float = 0,
    distance_function: Callable = my_pairwise,
    distance_threshold: float = 1.0,
    min_size: int = 400,
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
    # if "lev" in points:
    #     dlev = np.diff(np.unique(points["lev"]))
    #     dlev = np.amin(dlev[dlev > 0])
    #     to_distance = points[["lon", "lat", "lev"]]
    #     factors = np.ones(to_distance.shape)
    #     factors[:, 2] = 2 * dlev
    #     to_distance = to_distance / pd.DataFrame(factors, columns=["lon", "lat", "lev"])
    #     distance_matrix = distance_function(to_distance.to_numpy())
    # else:
    #     distance_matrix = distance_function(points[["lon", "lat"]].to_numpy())
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
    distance_function: Callable = my_pairwise,
    distance_threshold: float = 1.5,
    min_size: int = 400,
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


def cluster_criterion_neg(
    ds: xr.Dataset,
    criterion_threshold: float = 1e-9,
    distance_function: Callable = my_pairwise,
    distance_threshold: float = 1.0,
    min_size: int = 400,
) -> Tuple[list, list]:
    return cluster_generic(
        -ds["criterion"],
        ds["s"],
        ds["s_smo"],
        ds["criterion"],
        ds["lev"] if "lev" in ds else None,
        ds["u"],
        ds["v"],
        criterion_threshold=(
            ds["threshold"].item() if "threshold" in ds else -criterion_threshold
        ),
        distance_function=distance_function,
        distance_threshold=distance_threshold,
        min_size=min_size,
    )


def cluster_criterion(
    ds: xr.Dataset,
    criterion_threshold: float = 9,
    distance_function: Callable = my_pairwise,
    distance_threshold: float = 1.5,
    min_size: int = 400,
) -> Tuple[list, list]:
    return cluster_generic(
        ds["criterion"],
        ds["s"],
        ds["s_smo"],
        ds["criterion"],
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
    maxx = da.max("lat")
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
            output[i, j] = 1 - np.exp(-(z**2) / (2 * 0.5**2))
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
    return compute_weights_gaussian(x, is_nei)


def compute_weights_wind_speed_smoothed(
    points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = normalize_points_for_weights(points, "s_smo")
    return compute_weights_gaussian(x, is_nei)


def compute_weights_criterion_slice_neg(
    ds: xr.Dataset, points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = -slice_from_df(ds["criterion"], points).values
    x = x / x.max()
    return compute_weights_mean(x, is_nei)


def compute_weights_criterion_slice(
    ds: xr.Dataset, points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = slice_from_df(ds["criterion"], points).values
    x = x / x.max()
    return compute_weights_gaussian(x, is_nei)


def compute_weights_criterion_neg(
    points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = normalize_points_for_weights(points, "-criterion")
    return compute_weights_mean(x, is_nei)


def compute_weights_criterion(
    points: pd.DataFrame, is_nei: Optional[NDArray] = None
) -> NDArray:
    x = normalize_points_for_weights(points, "criterion")
    return compute_weights_gaussian(x, is_nei)




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
    dx = pairwise_difference(x, is_nei)
    wrap_mask = np.abs(dx) > 180
    dx = np.where(wrap_mask, -np.sign(dx) * (360 - np.abs(dx)), dx)
    dx = dx / distance_matrix
    dy = pairwise_difference(y, is_nei) / distance_matrix
    u = u / s
    v = v / s
    return (1 - dx * u[:, None] - dy * v[:, None]) / 2


def compute_weights_direction(
    points: pd.DataFrame, distance_matrix: NDArray, is_nei: Optional[NDArray] = None
):
    x, y, u, v, s = points[["lon", "lat", "u", "v", "s"]].to_numpy().T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = _compute_weights_direction(x, y, u, v, s, distance_matrix, is_nei)
    return out


def compute_weights(points: pd.DataFrame, distance_matrix: NDArray) -> np.ma.array:
    sample = np.random.choice(np.arange(distance_matrix.shape[0]), size=100)
    sample = distance_matrix[sample]
    dx = np.amin(sample[sample > 0])
    is_nei = (distance_matrix > 0) & (distance_matrix < (2 * dx))
    weights_dir = compute_weights_direction(points, distance_matrix, is_nei)
    is_nei = is_nei & (weights_dir < (2 * DIRECTION_THRESHOLD))
    weights = np.where(weights_dir > DIRECTION_THRESHOLD, 3 * weights_dir, 0)
    masked_weights = np.ma.array(weights, mask=~is_nei)
    return masked_weights


def compute_weights_2(points: pd.DataFrame, distance_matrix: NDArray) -> np.ma.array:
    sample = np.random.choice(np.arange(distance_matrix.shape[0]), size=100)
    sample = distance_matrix[sample]
    dx = np.amin(sample[sample > 0])
    is_nei = (distance_matrix > 0) & (distance_matrix < (2 * dx))
    weights_ws = compute_weights_criterion(points, is_nei)
    if "u" in points and "v" in points:
        weights_dir = compute_weights_direction(points, distance_matrix, is_nei)
        is_nei = is_nei & (weights_dir < DIRECTION_THRESHOLD)
    masked_weights = np.ma.array(weights_ws, mask=~is_nei)
    return masked_weights

    
def create_graph(masked_weights: np.ma.array, distance_matrix: NDArray) -> csr_matrix:
    graph = csgraph_from_masked(masked_weights)
    nco, labels = connected_components(graph)
    if nco == 1:
        return graph
    for label1, label2 in combinations(range(nco), 2):
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


def jets_from_predecessor(
    group: NDArray,
    predecessors: NDArray,  # 2d
    ends: NDArray,
    dmat_weighted: NDArray,
    dmat_unweighted: NDArray,
    cutoff: float,
) -> Sequence:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dmat_ratio = dmat_unweighted[ends] ** 4 / dmat_weighted[ends] ** 0.5
    dmat_ratio = np.where(np.isnan(dmat_ratio) | np.isinf(dmat_ratio), -1, dmat_ratio)
    ends = ends[np.argsort(dmat_ratio)]
    for end in ends:
        path = path_from_predecessors(predecessors, end)
        jet = group[path]
        if jet_integral_haversine(jet) > cutoff:
            return path
    print("no jet found")
    return None


def jets_from_many_predecessors(
    group: NDArray,
    predecessors: NDArray,  # 2d
    ends: NDArray,
    dmat_weighted: NDArray,
    dmat_unweighted: NDArray,
    cutoff: float,
) -> Sequence:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dmat_ratio = (
            dmat_unweighted[:, ends] ** 4
            / dmat_weighted[:, ends] ** 0.25
        )
    dmat_ratio = np.where(np.isnan(dmat_ratio) | np.isinf(dmat_ratio), -1, dmat_ratio)
    starts, ends_ = last_elements(dmat_ratio, len(dmat_ratio) // 5)
    ends_ = ends[ends_]
    for start, end in zip(starts, ends_):
        path = path_from_predecessors(predecessors[start], end)
        jet = group[path]
        if jet_integral_haversine(jet) > cutoff:
            return path
    # starts, ends_ = last_elements(dmat_ratio, 1000)
    # ends_ = ends[ends_]
    # for start, end in zip(starts, ends_):
    #     path = path_from_predecessors(predecessors[start], end)
    #     jet = group[path]
    #     if jet_integral_haversine(jet) > cutoff:
    #         return path
    print("no jet found")
    return None


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


def determine_start_global(
    ux: NDArray, lon: NDArray, lat: NDArray, masked_weights: np.ma.masked
) -> Tuple[NDArray, NDArray]:
    diffx = np.diff(ux)
    dx = np.amin(diffx)
    before = np.argwhere(lon == ux[-1]).flatten()
    after = np.argwhere(lon == ux[0]).flatten()
    newmask = masked_weights.mask.copy()
    newmask[np.ix_(before, after)] = np.ones((len(before), len(after)), dtype=bool)
    newmask[np.ix_(after, before)] = np.ones((len(after), len(before)), dtype=bool)
    masked_weights_2 = np.ma.array(masked_weights.data, mask=newmask)
    graph2 = csgraph_from_masked(masked_weights_2)
    nco, labels = connected_components(graph2)
    if nco == 1 and len(ux) == (360 / dx):
        start = ux[0]
        end = ux[-1]
    elif len(ux) == (360 / dx):
        ulab, counts = np.unique(labels, return_counts=True)
        importants = last_elements(counts, 2)
        lon1 = np.unique(lon[labels == ulab[importants[0]]])
        min1, max1 = min(lon1), max(lon1)
        lon2 = np.unique(lon[labels == ulab[importants[1]]])
        min2, max2 = min(lon2), max(lon2)
        if min2 == -180:
            end = max2
            start = min1
        elif min1 == -180:
            end = max1
            start = min2
    else:
        maxd = np.argmax(diffx)
        fakex = lon.copy()
        fakex[fakex <= ux[maxd]] += 360
        neworder = np.argsort(fakex)
        reverse_neworder = np.argsort(neworder)
        starts, ends = determine_start_poly(fakex[neworder], lat[neworder])
        starts = reverse_neworder[starts]
        ends = reverse_neworder[ends]
        return starts, ends   
    starts = np.where(lon == start)[0].astype(np.int16)
    ends = np.where(lon == end)[0].astype(np.int16)
    return starts, ends


def determine_start_poly(lon: NDArray, lat: NDArray) -> Tuple[NDArray, NDArray]:
    c1, c0 = np.polyfit(lon, lat, deg=1, rcond=1e-10)
    x0 = np.amin(lon)
    y0 = x0 * c1 + c0
    v0 = np.asarray([[x0, y0]])
    c = np.asarray([[1, c1]]) / np.sqrt(1 + c1**2)
    points = np.vstack([lon, lat]).T - v0
    projections = np.sum(c * points, axis=1)
    ncand = projections.shape[0] // 15
    starts = first_elements(projections, ncand).astype(np.int16)
    ncand = projections.shape[0] // 10
    ends = last_elements(projections, ncand).astype(np.int16)
    return starts, ends


def adjust_edges(
    starts: NDArray, ends: NDArray, lon: NDArray, ux: NDArray, edges: Optional[Tuple[float]] = None
) -> Tuple[NDArray, NDArray]:
    if edges is not None and -180 not in ux:
        west_border = np.isin(lon, [edges[0], edges[0] + 0.5, edges[0] + 1.0])
        east_border = np.isin(lon, [edges[1], edges[1] - 0.5, edges[1] - 1.0])
        if any(west_border):
            west_border = np.nonzero(west_border)[0]
            starts = west_border[last_elements(s[west_border], 3)].astype(np.int16)
        if any(east_border):
            east_border = np.nonzero(east_border)[0]
            ends = east_border[last_elements(s[east_border], 3)].astype(np.int16)
    return starts, ends


def find_jets_in_group_v2(
    graph: csr_matrix,
    group: pd.DataFrame,
    masked_weights: NDArray,
    jet_cutoff: float = 8e7,
    edges: Optional[Tuple[float]] = None,
):
    lon, lat, s = group[["lon", "lat", "s"]].to_numpy().T
    ux = np.unique(lon)
    if -180 in ux:
        starts, ends = determine_start_global(ux, lon, lat, masked_weights)
    else:
        # starts, ends = determine_start_poly(lon, lat)
        ends = depth_first_order(graph, 0)[0][-2:]
        starts = depth_first_order(graph, ends[-1], directed=False)[0][-2:]
    starts, ends = adjust_edges(starts, ends, lon, ux, edges)
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


def find_jets_in_group_v3(
    graph: csr_matrix,
    group: pd.DataFrame,
    masked_weights: NDArray,
    jet_cutoff: float = 8e7,
    edges: Optional[Tuple[float]] = None,
):
    lon, lat, s = group[["lon", "lat", "s"]].to_numpy().T
    ux = np.unique(lon)
    if -180 in ux:
        starts, ends = determine_start_global(ux, lon, lat, masked_weights)
    else:
        starts, ends = determine_start_poly(lon, lat)
    starts, ends = adjust_edges(starts, ends, lon, ux, edges)
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
    jet_cutoff: float = 8e7,
    edges: Optional[Tuple[float]] = None,
) -> Sequence[pd.DataFrame]:
    jets = []
    for group, dist_mat in zip(groups, dist_mats):
        masked_weights = compute_weights(group, dist_mat)
        graph = create_graph(masked_weights, dist_mat)
        jet = find_jets_in_group_v2(graph, group, masked_weights, jet_cutoff, edges)
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
            len_ = len(ds.time.values)
        except AttributeError:
            iterable = (ds.sel(cluster=cluster_) for cluster_ in ds.cluster.values)
            len_ = len(ds.cluster.values)
        if processes == 1:
            return list(tqdm(map(self.loop_call, iterable), total=len_))
        with Pool(processes=processes) as pool:
            return list(
                tqdm(
                    pool.imap(self.loop_call, iterable, chunksize=chunksize),
                    total=len_,
                )
            )


def find_jets(ds: xr.DataArray, **kwargs):
    jet_finder = JetFinder(
        preprocess=preprocess,
        cluster=cluster_criterion,
        refine_jets=jets_from_mask,
    )
    return jet_finder.call(ds, **kwargs)


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


def get_extremities(mask: NDArray, distance_matrix: NDArray) -> NDArray:
    idx = np.where(mask)[0]
    this_dist_mat = distance_matrix[idx, :][:, idx]
    extremities = np.concatenate(
        last_elements(this_dist_mat, min(this_dist_mat.shape[0], 100))
    )
    return idx[np.unique(extremities)]


def compute_criterion_tophat(ds: xr.Dataset):
    structure = generate_binary_structure(2, 2)
    crit, smin, smax = to_zero_one(ds["s_smo"].values)
    thresh = (38 - smin) / (smax - smin)
    fgnd_crit = crit - thresh
    mask_bgnd = fgnd_crit < 0
    fgnd_crit[mask_bgnd] = 0
    crit = white_tophat(fgnd_crit, structure=structure, mode="constant", cval=1e8)
    ds["criterion_tophat"] = ds["s_smo"].copy(data=crit)
    return ds


def skeletonize_wrapper(ds: xr.Dataset, threshold: float = 0.):
    criterion_mask = (ds["criterion_tophat"] > threshold) | (ds_["criterion_spensberger"] < -1e-8)
    criterion_mask = criterion_mask.values
    # skel = skeletonize(criterion_mask)
    # skel = binary_dilation(skel, iterations=2)
    # skel = skeletonize(skel)
    skel = criterion_mask
    return skel


def compute_weights(points: pd.DataFrame, distance_matrix: NDArray, is_nei: NDArray) -> np.ma.array:
    weights_dir = compute_weights_direction(points, distance_matrix, is_nei)
    weights_dir = np.ma.array(weights_dir, mask=~is_nei)
    triu = np.triu_indices_from(weights_dir, k=1)
    tril = np.tril_indices_from(weights_dir, k=-1)
    cond = weights_dir[triu] < weights_dir[tril]
    x1, y1, x2, y2 = triu[0][~cond], triu[1][~cond], tril[0][cond], tril[1][cond]
    weights_dir[x1, y1] = np.nan
    weights_dir.mask[x1, y1] = True
    weights_dir[x2, y2] = np.nan
    weights_dir.mask[x2, y2] = True
    x, y = np.arange(weights_dir.shape[0]), np.arange(weights_dir.shape[0])
    weights_dir[x, y] = np.nan
    weights_dir.mask[x, y] = True
    weights_speed = compute_weights_wind_speed(points, ~weights_dir.mask)
    weights_speed = np.ma.array(weights_speed, mask=weights_dir.mask)
    weights_dir = np.clip(np.where(weights_dir > DIRECTION_THRESHOLD, 3 * weights_dir, 0.0001), 0, 1)
    weights = 0.1 * weights_dir + 0.9 * weights_speed
    # is_nei_2 = np.zeros_like(is_nei)
    # ouais = np.argmin(weights, axis=1)
    # is_nei_2[np.arange(is_nei_2.shape[0]), ouais] = True
    # is_nei_2 = is_nei & is_nei_2
    # weights.mask = ~is_nei_2
    return weights


def graph_from_group(group, and_unweighted: bool = True, full: bool = False):
    distance_matrix = my_pairwise(group[["lon", "lat"]].to_numpy())
    sample = np.random.choice(np.arange(distance_matrix.shape[0]), size=100)
    sample = distance_matrix[sample]
    dx = np.amin(sample[sample > 0])
    is_nei = (distance_matrix > 0) & (distance_matrix < (2 * dx))
    weights = compute_weights(group, distance_matrix, is_nei)
    graph = csgraph_from_masked(weights)
    if not and_unweighted:
        if full:
            return graph, weights
        return graph
    weights_distances = np.ma.array(distance_matrix, mask=~is_nei)
    graph_uw = csgraph_from_masked(weights_distances)
    if full:
        return graph, graph_uw, weights, weights_distances,
    return graph, graph_uw


def subset_graph(group: pd.DataFrame, weights: np.ma.array, mask: NDArray, weights_distances: np.ma.array = None, full: bool = False):
    group = group.iloc[mask]
    weights_ = weights[mask, :][:, mask]
    graph = csgraph_from_masked(weights_)
    if weights_distances is None and full:
        return group, graph, weights_
    if weights_distances is None:
        return group, graph
    weights_distances_ = weights_distances[mask, :][:, mask]
    graph_uw = csgraph_from_masked(weights_distances_)
    if full:
        return group, graph, graph_uw, weights_, weights_distances_
    return group, graph, graph_uw


def split_graph(big_group, big_graph, big_weights, big_weights_distances=None, full: bool = False):
    groups = []
    graphs = []
    if full:
        weights = []
    if full and big_weights_distances is not None:
        weights_distances = []
    if big_weights_distances is not None:
        graphs_uw = []
    _, labels = connected_components(big_graph)
    masks = labels_to_mask(labels)
    for mask in masks.T:
        if np.sum(mask) < 35:
            continue
        
        out = subset_graph(big_group, big_weights, mask, big_weights_distances, full=full)
        if big_weights_distances is not None and not full:
            group, graph, graph_uw = out
            groups.append(group)
            graphs.append(graph)
            graphs_uw.append(graph_uw)
        elif big_weights_distances is not None:
            group, graph, graph_uw, weights_, weights_distances_ = out
            groups.append(group)
            graphs.append(graph)
            graphs_uw.append(graph_uw)
            weights.append(weights_)
            weights_distances.append(weights_distances_)
        elif big_weights_distances is None and full:
            group, graph, weights_ = out
            groups.append(group)
            graphs.append(graph)
            weights.append(weights_)
        else:
            group, graph = out
            groups.append(group)
            graphs.append(graph)    
    if big_weights_distances is not None and not full:
        return groups, graphs, graphs_uw
    elif big_weights_distances is not None and full:
        return groups, graphs, graphs_uw, weights, weights_distances
    elif big_weights_distances is None and full:
        return groups, graphs, weights
    return groups, graphs


def best_paths_subset(weights, graph_uw, full: bool = False):
    nbgraph = csr.csr_to_nbgraph(graph_uw)
    paths = csr._build_skeleton_path_graph(nbgraph)
    degrees = np.diff(graph_uw.indptr)
    
    subset = np.where(degrees > 2)[0].tolist()
    if full:
        paths_list = []
        losses = []
    for i in range(paths.shape[0]):
        start, stop = paths.indptr[i:i + 2]
        indices = paths.indices[start:stop]
        length = 0
        for j in range(len(indices) - 1):
            j1, j2 = indices[j: j + 2]
            if weights.mask[j1, j2]:
                length += weights[j2, j1]
            else:
                length += weights[j1, j2]
        loss = length / len(indices)
        # if (loss > 0.02) and (len(indices) > 4):
        #     continue
        subset.extend(indices)
        if full:
            paths_list.append(indices)
            losses.append(loss)
    if full:
        return subset, paths_list, losses
    return subset


def find_main_path(graph):
    distances, predecessors = shortest_path(graph, return_predecessors=True, directed=True)
    distances_uw = shortest_path(graph, return_predecessors=False, unweighted=True, directed=True)
    starts, ends = last_elements(distances_uw, distances_uw.shape[0] // 10)
    acceptables = first_elements(distances[starts, ends], len(starts) // 2)
    start = starts[acceptables[0]]
    end = ends[acceptables[0]]
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     start, end = np.unravel_index(np.argmax(np.nan_to_num(distances_uw ** 4 / distances ** 0.25, nan=10, posinf=0.1)), distances.shape)
    main_path = path_from_predecessors(predecessors[start].copy(), end)
    return main_path


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


def compute_jet_width(jet: pl.DataFrame, da: xr.DataArray) -> xr.DataArray:
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


def compute_jet_props(jet: pl.DataFrame, da: xr.DataArray) -> dict:
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
    props = []
    for _, jet in jets.group_by("jet ID", maintain_order=True):
        props.append(compute_jet_width(jet, da))
    return props


def compute_jet_width_wrapper(args: Tuple) -> list:
    jets, da = args
    props = []
    for _, jet in jets.group_by("jet ID", maintain_order=True):
        props.append(compute_jet_props(jet, da))
    df = pl.from_dicts(props)
    return df


def compute_all_jet_props(
    all_jets_one_df: pl.DataFrame,
    da: xr.DataArray,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.Dataset:
    len_, iterator = create_mappable_iterator(all_jets_one_df, [da])
    print("Computing jet properties")
    all_props_dfs = map_maybe_parallel(
        iterator,
        compute_jet_props_wrapper,
        len_=len_,
        processes=processes,
        chunksize=chunksize,
    )
    index_columns = get_index_columns(all_props_dfs)
    all_props_df = pl.concat(all_props_dfs)
    cast_arg = {
        key: (pl.Boolean if key == "is_polar" else pl.Float32)
        for key in all_props_df.columns
    }
    all_props_df = all_props_df.cast(cast_arg)
    all_props_df = all_props_df.to_pandas().set_index(index_columns)
    return xr.Dataset.from_dataframe(all_props_df)


def do_one_int_low(args):
    jets, da_low_ = args
    ints = []
    for _, jet in jets.group_by("jet ID", maintain_order=True):
        x, y = round_half(jet.select(["lon", "lat"]).to_numpy().T)
        x_ = xr.DataArray(x, dims="points")
        y_ = xr.DataArray(y, dims="points")
        s_low = da_low_.sel(lon=x_, lat=y_).values
        jet_low = np.asarray([x, y, s_low]).T
        ints.append(jet_integral_haversine(jet_low))
    return xr.DataArray(ints, coords={"jet": np.arange(len(ints))})


def compute_int_low(  # broken with members
    all_jets_one_df: pl.DataFrame,
    props_as_ds: xr.Dataset,
    exp_low_path: Path,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.Dataset:
    this_path = exp_low_path.joinpath("int_low.nc")
    if this_path.is_file():
        props_as_ds["int_low"] = xr.open_dataarray(this_path)
        props_as_ds["int_ratio"] = props_as_ds["int_low"] / props_as_ds["int"]
        return props_as_ds
    print("computing int low")
    props_as_ds["int_low"] = props_as_ds["mean_lon"].copy()

    da_low = xr.open_dataarray(exp_low_path.joinpath("da.nc"))
    len_, iterator = create_mappable_iterator(all_jets_one_df, [da_low])
    all_jet_ints = map_maybe_parallel(
        iterator, do_one_int_low, len_=len_, processes=processes, chunksize=chunksize
    )

    props_as_ds["int_low"] = (
        tuple(props_as_ds.dims),
        np.stack([jet_ints.values for jet_ints in all_jet_ints]),
    )
    props_as_ds["int_ratio"] = props_as_ds["int_low"] / props_as_ds["int"]
    props_as_ds["int_low"].to_netcdf(exp_low_path.joinpath("int_low.nc"))
    return props_as_ds


def is_polar_v2(props_as_ds: xr.Dataset) -> xr.Dataset:
    props_as_ds[:, "is_polar"] = (
        props_as_ds.select("mean_lat") * 200
        - props_as_ds.select("mean_lon") * 30
        + props_as_ds.select("int_low") / RADIUS
    ) > 9000
    return props_as_ds