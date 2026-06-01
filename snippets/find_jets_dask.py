@njit
def distance(x1: float, x2: float, y1: float, y2: float) -> float:
    dx = x2 - x1
    if np.abs(dx) > 180:
        dx = 360 - np.abs(dx)  # sign is irrelevant
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2)


@njit(parallel=False)
def my_pairwise(X1: NDArray, X2: NDArray | None = None) -> NDArray:
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
            group_df = pl.from_pandas(group.to_dataframe())
            float_columns = ["lev", "lon", "lat", "u", "v", "s", "alignment", "sigma"]
            cast_arg = {
                fc: pl.Float32 for fc in float_columns if fc in group_df.columns
            }
            group_df = group_df.cast(cast_arg)
            for potential_to_drop in ["ratio", "label"]:
                try:
                    group_df = group_df.drop(potential_to_drop)
                except pl.exceptions.ColumnNotFoundError:
                    pass
            group_df = group_df[indices]
            group_ = group_df[["lon", "lat"]].to_numpy()
            labels = (
                AgglomerativeClustering(
                    n_clusters=None, distance_threshold=dx * 1.9, linkage="single"
                )
                .fit(group_)
                .labels_
            )
            masks = labels_to_mask(labels)
            for mask in masks.T:
                groups.append(group_df.filter(mask))
    jets = []
    for group_df in groups:
        bigjump = np.diff(group_df["lon"]) < -3 * dx
        if any(bigjump):
            here = np.where(bigjump)[0][0] + 1
            group_df = group_df[np.arange(len(group_df)) - here]
        group_ = group_df[["lon", "lat", "s"]].to_numpy().astype(np.float32)
        jet_int = jet_integral_haversine(group_)
        mean_alignment = np.mean(group_df["alignment"].to_numpy())
        if jet_int > jet_threshold and mean_alignment > mean_alignment_threshold:
            jets.append(group_df)
    for j, jet in enumerate(jets):
        ser = pl.Series("jet ID", np.full(len(jet), j, dtype=np.int16))
        jet.insert_column(0, ser)
    return jets


def inner_find_all_jets(ds_block, basepath: Path, **kwargs):
    extra_dims = {}
    for potential in ["member", "time", "cluster"]:
        if potential in ds_block.dims:
            extra_dims[potential] = ds_block[potential].values
    # len_ = np.prod([len(co) for co in extra_dims.values()])
    iter_ = list(product(*list(extra_dims.values())))
    unique_hash = hash(tuple(iter_))
    opath = basepath.joinpath(f"jets/j{unique_hash}.parquet")
    if opath.is_file():
        return ds_block.isel(lon=0, lat=0).reset_coords(drop=True)
    all_jets = []
    for vals in tqdm(iter_):
        indexer = {dim: val for dim, val in zip(extra_dims, vals)}
        this_ds = ds_block.loc[indexer]
        if "threshold" in this_ds:
            kwargs["wind_threshold"] = this_ds["threshold"].item()
            kwargs["jet_threshold"] = kwargs["wind_threshold"] / 23 * 1e8
        these_jets = find_jets(this_ds, **kwargs)
        all_jets.append(these_jets)
    df = pl.concat([pl.concat(jets) for jets in all_jets])

    index_columns = get_index_columns(df)
    other_columns = ["lon", "lat", "lev", "u", "v", "s", "sigma", "alignment"]
    df = df.select([*index_columns, *other_columns])
    df = df.sort(index_columns)

    df.write_parquet(opath)
    return ds_block.isel(lon=0, lat=0).reset_coords(drop=True)


def find_all_jets(ds, basepath: Path, threshold: xr.DataArray | None = None, **kwargs):
    jets_path = basepath.joinpath("jets")
    jets_path.mkdir(exist_ok=True, mode=0o777)
    template = ds.isel(lon=0, lat=0).reset_coords(drop=True)
    if threshold is not None:
        ds["threshold"] = ("time", threshold.data)
    if _get_global_client() is None or len(ds.chunks) == 0:
        print("No Dask client found or ds is not lazy, reverting to sequential")
        _ = inner_find_all_jets(ds, basepath=basepath, **kwargs)
    else:
        to_comp = ds.map_blocks(
            inner_find_all_jets,
            template=template,
            kwargs=dict(basepath=basepath, **kwargs),
        ).persist()
        progress(to_comp, notebook=False)
        to_comp.compute()
    dfs = []
    for f in basepath.joinpath("jets").glob("*.parquet"):
        dfs.append(pl.read_parquet(f))
    df = pl.concat(dfs)
    index_columns = ["member", "time", "cluster", "jet ID"]
    index_columns = [ic for ic in index_columns if ic in df.columns]
    df = df.sort(index_columns)
    return df