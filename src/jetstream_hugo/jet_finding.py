import warnings
from pathlib import Path
from sys import stderr
from os.path import commonpath
from functools import partial, wraps
from datetime import timedelta
from typing import Callable, Iterable, Mapping, Sequence, Tuple, Literal
from nptyping import NDArray
from multiprocessing import Pool, current_process, get_context
from itertools import combinations, product

import numpy as np
import polars as pl
import polars_ols as pls
from dask.distributed import progress
from dask.distributed.client import _get_global_client
import xarray as xr
from scipy.stats import linregress
from contourpy import contour_generator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from tqdm import tqdm, trange
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
    SEASONS,
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


def get_index_columns(df, potentials: tuple = ("member", "time", "cluster", "jet ID")):
    index_columns = [ic for ic in potentials if ic in df.columns]
    return index_columns


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


def haversine_polars(
    lon1: pl.Expr, lat1: pl.Expr, lon2: pl.Expr, lat2: pl.Expr
) -> pl.Expr:
    lon1 = lon1.radians()
    lat1 = lat1.radians()
    lon2 = lon2.radians()
    lat2 = lat2.radians()

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


# lat: NDArray, dlon: NDArray, dlat: NDArray
# a = (
#     np.sin(dlat / 2.0) ** 2 * np.cos(dlon / 2) ** 2
#     + np.cos(lat) ** 2 * np.sin(dlon / 2) ** 2
# )
# return 2 * RADIUS * np.arcsin(np.sqrt(a))
def haversine_from_dl_polars(lat: pl.Expr, dlon: pl.Expr, dlat: pl.Expr) -> pl.Expr:
    lat = lat.radians()
    dlon = dlon.radians()
    dlat = dlat.radians()

    a = (dlat / 2.0).sin().pow(2) * (dlon / 2.0).cos().pow(2) + lat.cos().pow(2) * (
        dlon / 2.0
    ).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def trapz_polars(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    return 0.5 * ((x - x.shift()) * (y + y.shift())).sum()


def jet_integral_haversine_polars(
    lon: pl.Expr, lat: pl.Expr, s: pl.Expr | None = None, x_is_one: bool = False
) -> pl.Expr:
    ds = haversine_polars(
        lon,
        lat,
        lon.shift(),
        lat.shift(),
    )
    if x_is_one:
        return ds.sum()
    return 0.5 * (ds * (s + s.shift())).sum()


def compute_jet_props(df: pl.DataFrame) -> pl.DataFrame:
    aggregations = [
        *[
            ((pl.col(col) * pl.col("s")).sum() / pl.col("s").sum()).alias(f"mean_{col}")
            for col in ["lon", "lat", "lev"]
        ],
        pl.col("s").mean().alias("mean_s"),
        *[
            pl.col(col).get(pl.col("s").arg_max()).alias(f"{col}_star")
            for col in ["lon", "lat", "s"]
        ],
        *[
            (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_ext")
            for col in ["lon", "lat"]
        ],
        pl.col("lat")
        .least_squares.ols(pl.col("lon"), mode="statistics", add_intercept=True)
        .struct.field(["r2", "coefficients"]),
        (pl.col("lat") - pl.col("lat").mean()).pow(2).sum().alias("waviness2"),
        (
            pl.col("lat").gather(pl.col("lon").arg_sort()).diff().abs().sum()
            / (pl.col("lon").max() - pl.col("lon").min())
        ).alias("wavinessR16"),
        (
            jet_integral_haversine_polars(pl.col("lon"), pl.col("lat"), x_is_one=True)
            / pl.lit(RADIUS)
            * pl.col("lat").mean().radians().cos()
        ).alias("wavinessDC16"),
        jet_integral_haversine_polars(pl.col("lon"), pl.col("lat"), pl.col("s")).alias(
            "int"
        ),
        jet_integral_haversine_polars(
            pl.col("lon").filter(pl.col("lon") > -10),
            pl.col("lat").filter(pl.col("lon") > -10),
            pl.col("s").filter(pl.col("lon") > -10),
        ).alias("int_over_europe"),
    ]

    df_lazy = df.lazy()
    index_columns = get_index_columns(df)
    if "member" not in get_index_columns(df):
        gb = df_lazy.group_by(index_columns, maintain_order=True)

        props_as_df = gb.agg(*aggregations)
        props_as_df = props_as_df.with_columns(
            tilt=pl.col("coefficients").list.get(0),
            waviness1=1 - pl.col("r2"),
        ).drop(["r2", "coefficients"])
        return props_as_df

    # streaming mode doesn't work well
    collected = []
    for member in tqdm(df["member"].unique(maintain_order=True).to_numpy()):
        gb = df_lazy.filter(pl.col("member") == member).group_by(
            get_index_columns(df), maintain_order=True
        )
        props_as_df = gb.agg(*aggregations)
        props_as_df = props_as_df.with_columns(
            tilt=pl.col("coefficients").list.get(0),
            waviness1=1 - pl.col("r2"),
        ).drop(["r2", "coefficients"])
        collected.append(props_as_df.collect())
    return pl.concat(collected).sort("member")


def distances_to_coord(
    da_df: pl.DataFrame, jet: pl.DataFrame, coord: str, prefix: str = ""
):
    unique = da_df[coord].unique()
    ind = unique.search_sorted(jet[f"{prefix}{coord}"])
    dx = unique.diff()[1]
    dist_to_next = unique.gather(ind) - jet[f"{prefix}{coord}"]
    dist_to_previous = dx - dist_to_next
    n = unique.len()
    return n, ind, dx, dist_to_previous, dist_to_next


def interp_from_other(jets: pl.DataFrame, da_df: pl.DataFrame):
    n_lon, ind_lon, dlon, dist_to_previous_lon, dist_to_next_lon = distances_to_coord(
        da_df, jets, "lon", "normal"
    )
    n_lat, ind_lat, dlat, dist_to_previous_lat, dist_to_next_lat = distances_to_coord(
        da_df, jets, "lat", "normal"
    )
    s_above_right = da_df[n_lon * ind_lat + ind_lon, "s"]
    s_below_right = da_df[n_lon * (ind_lat - 1) + ind_lon, "s"]
    s_above_left = da_df[n_lon * ind_lat + ind_lon - 1, "s"]
    s_below_left = da_df[n_lon * (ind_lat - 1) + ind_lon - 1, "s"]

    s_below = (
        s_below_right * dist_to_next_lon / dlon
        + s_below_left * dist_to_previous_lon / dlon
    )
    s_above = (
        s_above_right * dist_to_next_lon / dlon
        + s_above_left * dist_to_previous_lon / dlon
    )

    s_interp = s_below * dist_to_next_lat / dlat + s_above * dist_to_previous_lat / dlat
    return s_interp


def inner_compute_widths(args):
    jets, da = args

    dn = 0.5
    ns_df = pl.Series("n", np.delete(np.arange(-12, 12 + dn, dn), 24)).to_frame()

    # Expr theta
    dxds = pl.col("lon").diff()
    dyds = pl.col("lat").diff()
    theta = pl.arctan2(dyds, dxds).interpolate("linear")

    # Expr normals
    normallon = pl.col("lon") + (pl.col("theta") + np.pi / 2 * pl.col("n")).cos()
    normallat = pl.col("lat") + (pl.col("theta") + np.pi / 2 * pl.col("n")).sin()

    # Expr half_width
    below = pl.col("s_interp") <= pl.max_horizontal(pl.col("s") / 2, pl.lit(25))
    filter_ = pl.repeat(1, below.len()).append(pl.repeat(-1, below.len()))
    below = below.append(below.reverse())
    below = below.filter(filter_ == pl.col("side").get(0))
    stop = below.arg_max()
    nlo = pl.col("normallon").gather(stop)
    nla = pl.col("normallon").gather(stop)
    half_width = haversine_polars(
        nlo, nla, pl.col("lon").get(0), pl.col("lat").get(0)
    ).cast(pl.Float32)

    index_columns = get_index_columns(jets, ("member", "time", "cluster"))
    agg_out = {ic: pl.col(ic).first() for ic in [*index_columns, "s"]}

    first_agg_out = agg_out | {"half_width": half_width}
    second_agg_out = agg_out | {"half_width": pl.col("half_width").mean()}
    third_agg_out = agg_out | {
        "width": (pl.col("half_width") * pl.col("s")).sum() / pl.col("s").sum()
    }

    da_df = pl.from_pandas(da.to_dataframe().reset_index()).drop(index_columns)
    jets = jets[[*index_columns, "jet ID", "lon", "lat", "s"]]

    jets = jets.with_columns(
        jets.group_by("jet ID", maintain_order=True)
        .agg(theta=theta, index=pl.int_range(pl.len()))
        .explode(["index", "theta"])
    )
    jets = jets.join(ns_df, how="cross")

    jets = jets.with_columns(normallon=normallon, normallat=normallat)
    jets = jets[
        [
            *index_columns,
            "jet ID",
            "index",
            "lon",
            "lat",
            "s",
            "n",
            "normallon",
            "normallat",
        ]
    ]
    jets = jets.filter(
        pl.col("normallon") >= da_df["lon"].min(),
        pl.col("normallon") <= da_df["lon"].max(),
        pl.col("normallat") >= da_df["lat"].min(),
        pl.col("normallat") <= da_df["lat"].max(),
    )

    jets = jets.with_columns(s_interp=interp_from_other(jets, da_df))
    jets = jets.with_columns(side=pl.col("n").sign().cast(pl.Int8))

    jets = jets.group_by(["jet ID", "index", "side"], maintain_order=True).agg(
        **first_agg_out
    )
    jets = jets.with_columns(
        half_width=pl.col("half_width").list.first(),
    )
    jets = jets.group_by(["jet ID", "index"], maintain_order=True).agg(**second_agg_out)
    jets = jets.group_by("jet ID", maintain_order=True).agg(**third_agg_out)
    print("tictac", file=stderr, end="\r")  # to prevent server from disconnecting
    return jets.drop("s").cast({"width": pl.Float32})


def map_maybe_parallel(
    iterator: Iterable,
    func: Callable,
    len_: int,
    processes: int = N_WORKERS,
    chunksize: int = 100,
    progress: bool = True,
    pool_kwargs: dict | None = None,
    ctx=None,
) -> list:
    processes = min(processes, len_)
    if processes == 1 and progress:
        return list(tqdm(map(func, iterator), total=len_))
    if processes == 1:
        return list(map(func, iterator))
    if pool_kwargs is None:
        pool_kwargs = {}
    pool_func = Pool if ctx is None else ctx.Pool
    if not progress:
        with pool_func(processes=processes, **pool_kwargs) as pool:
            to_ret = pool.imap(func, iterator, chunksize=chunksize)
            return list(to_ret)
    with pool_func(processes=processes, **pool_kwargs) as pool:
        to_ret = tqdm(
            pool.imap(func, iterator, chunksize=chunksize),
            total=len_,
        )
        return list(to_ret)


def create_mappable_iterator(
    df: pl.DataFrame,
    das: Sequence | None = None,
    others: Sequence | None = None,
    potentials: Tuple = ("member", "time", "cluster"),
) -> Tuple:
    if das is None:
        das = []
    if others is None:
        others = []
    iter_dims = []
    for potential in potentials:
        if potential in df.columns:
            iter_dims.append(potential)
    gb = df.group_by(iter_dims, maintain_order=True)
    len_ = len(gb.first())
    iterator = (
        (
            jets,
            *[
                _compute(
                    da.sel(
                        {
                            dim: values
                            for dim, values in zip(iter_dims, index)
                            if dim in da.dims
                        }
                    )
                )
                for da in das
            ],
            *others,
        )
        for index, jets in gb
    )
    return len_, iterator


def compute_widths_parallel(
    all_jets_one_df: pl.DataFrame,
    da: xr.DataArray,
    processes: int = N_WORKERS,
    chunksize: int = 100,
):
    len_, iterator = create_mappable_iterator(all_jets_one_df, [da])
    print("Computing widths")
    all_widths = map_maybe_parallel(
        iterator,
        inner_compute_widths,
        len_=len_,
        processes=processes,
        chunksize=chunksize,
        ctx=get_context("spawn"),
    )
    return pl.concat(all_widths)


def round_half(x):
    return np.round(x * 2) / 2


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


def extract_season_from_df(
    df: pl.DataFrame,
    season: list | str | tuple | int | None = None,
) -> pl.DataFrame:
    if season is None:
        return df
    if isinstance(season, str):
        season = SEASONS[season]
    if isinstance(season, int):
        season = [season]
    return df.filter(pl.col("time").dt.month().is_in(season))


def extract_features(
    props_as_df: pl.DataFrame,
    feature_names: Sequence = None,
    season: list | str | tuple | int | None = None,
) -> Tuple[NDArray, float]:
    props_as_df = extract_season_from_df(props_as_df, season)
    if feature_names is None:
        feature_names = ["mean_lon", "mean_lat", "s_star"]

    X = props_as_df[feature_names].to_numpy()
    return X


def one_gmix(X):
    X, _, _ = to_zero_one(X)
    model = GaussianMixture(
        n_components=2
    )  # to help with class imbalance, 1 for sub 2 for polar
    labels = model.fit_predict(X)
    masks = labels_to_mask(labels)
    mls = []
    for mask in masks.T:
        mls.append(X[mask, 0].mean())
    return labels != np.argmin(mls)


def is_polar_gmix(
    props_as_df: pl.DataFrame,
    feature_names: list,
    mode: Literal["year"] | Literal["season"] | Literal["month"] = "year",
) -> xr.Dataset:
    if mode == "year":
        X = extract_features(props_as_df, feature_names, None)
        labels = one_gmix(X)
        return props_as_df.with_columns(is_polar=labels)
    index_columns = get_index_columns(props_as_df)
    to_concat = []
    if mode == "season":
        for season in tqdm(["DJF", "MAM", "JJA", "SON"]):
            X = extract_features(props_as_df, feature_names, season)
            labels = one_gmix(X)
            to_concat.append(extract_season_from_df(props_as_df, season).with_columns(is_polar=labels))
    elif mode == "month":
        for month in trange(1, 13):
            X = extract_features(props_as_df, feature_names, month)
            labels = one_gmix(X)
            to_concat.append(extract_season_from_df(props_as_df, month).with_columns(is_polar=labels))
    return pl.concat(to_concat).sort(index_columns)


def categorize_df_jets(props_as_df: pl.DataFrame):
    index_columns = get_index_columns(props_as_df, ("member", "time", "cluster", "jet ID"))
    other_columns = [col for col in props_as_df.columns if col not in [*index_columns, "is_polar"]]
    agg = {col: (pl.col(col) * pl.col("int")).sum() / pl.col("int").sum() for col in other_columns}
    agg["int"] = pl.col("int").mean()
    agg["s_star"] = pl.col("s_star").max()
    agg["lon_ext"] = pl.col("lon_ext").max()
    agg["lat_ext"] = pl.col("lat_ext").max()
    gb_columns = get_index_columns(props_as_df, ("member", "time", "cluster", "is_polar"))
    props_as_df_cat = props_as_df.group_by(gb_columns, maintain_order=True).agg(**agg).sort(gb_columns)
    props_as_df_cat = props_as_df_cat.with_columns(pl.when(pl.col("is_polar")).then(pl.lit("EDJ")).otherwise(pl.lit("STJ")).alias("jet")).drop("is_polar")
    
    if "member" in index_columns:
        dummy_indexer = props_as_df_cat["member"].unique(maintain_order=True).to_frame().join(props_as_df_cat["time"].unique(maintain_order=True).to_frame(), how="cross").join(props_as_df_cat["jet"].unique(maintain_order=True).to_frame(), how="cross")
    else:
        dummy_indexer = props_as_df_cat["time"].unique(maintain_order=True).to_frame().join(props_as_df_cat["jet"].unique(maintain_order=True).to_frame(), how="cross")
    new_index_columns = get_index_columns(props_as_df, ("member", "time", "cluster", "jet"))
    props_as_df_cat = dummy_indexer.join(props_as_df_cat, on=[pl.col(col) for col in new_index_columns], how="left").sort(new_index_columns)
    return props_as_df_cat


def overlap_vert_dist_polars() -> Tuple[pl.Expr]:
    x1 = pl.col("lon").flatten()
    y1 = pl.col("lat").flatten()
    x2 = pl.col("lon_right").flatten()
    y2 = pl.col("lat_right").flatten()

    row = pl.first().cum_count()

    a1 = x1.arg_unique()
    a2 = x2.arg_unique()

    x1 = x1.gather(a1)
    y1 = y1.gather(a1)
    x2 = x2.gather(a2)
    y2 = y2.gather(a2)

    inter12 = x1.is_in(x2)
    inter21 = x2.is_in(x1)
    vert_dist = (y1.filter(inter12) - y2.filter(inter21)).abs().mean()
    overlap = 0.5 * (inter12.mean() + inter21.mean())
    return vert_dist.over(row), overlap.over(row)


def _track_jets(df: pl.DataFrame):
    index_columns = get_index_columns(df)
    df = df.select([*index_columns, "lon", "lat"]).clone()
    unique_times = (
        df.select("time")
        .with_row_index()
        .unique("time", keep="first", maintain_order=True)
    )
    time_index_df = unique_times["index"]
    unique_times = unique_times["time"]
    df = df.with_columns(df.select(pl.col(["lon", "lat"]).map_batches(round_half)))
    guess_nflags = max(50, len(unique_times))
    guess_len = 50
    all_jets_over_time = np.zeros(
        (guess_nflags, guess_len), dtype=[("time", "datetime64[ms]"), ("jet ID", "i2")]
    )
    all_jets_over_time[:] = (np.datetime64("NaT"), -1)
    last_valid_index = np.full(guess_nflags, fill_value=guess_len, dtype="int16")
    flags = df.group_by(["time", "jet ID"], maintain_order=True).first()
    flags = flags.select([*index_columns]).clone()
    flags = flags.insert_column(
        -1, pl.Series("flag", np.zeros(len(flags), dtype=np.uint32))
    )
    time_index_flags = (
        flags.select("time")
        .with_row_index()
        .unique("time", keep="first", maintain_order=True)["index"]
    )
    for last_flag, _ in df[: time_index_df[1]].group_by("jet ID", maintain_order=True):
        last_flag = last_flag[0]
        all_jets_over_time[last_flag, 0] = (unique_times[0], last_flag)
        last_valid_index[last_flag] = 0
        flags[last_flag, "flag"] = last_flag
    current = current_process()
    if current.name == "MainProcess":
        iterator = (pbar := trange(1, len(unique_times), position=0, leave=True))
    else:
        iterator = range(1, len(unique_times))
    for it in iterator:
        # create working dataframes: current timestep, previous 4 timesteps
        last_time = (
            time_index_df[it + 1] if (it < (len(time_index_df) - 1)) else df.shape[0]
        )
        current_df = df[time_index_df[it] : last_time]
        t = unique_times[it]
        min_it = max(0, it - 4)
        previous_df = df[time_index_df[min_it] : time_index_df[it]]
        from_ = max(0, last_flag - 20)

        # Filter actual candidates in previous df: no long time jump, no duplicate flag. If flag happens several times, pick the most recent
        potentials = all_jets_over_time[from_ : last_flag + 1]
        these_lvis = last_valid_index[from_ : last_flag + 1]
        potentials = [potential[lvi] for potential, lvi in zip(potentials, these_lvis)]
        times_of_jets = [ind[0] for ind in potentials]
        timesteps_to_check = unique_times[min_it:it]
        condition = np.isin(times_of_jets, timesteps_to_check)
        potentials = [ind for con, ind in zip(condition, potentials) if con]

        # Cumbersome construction for pairwise operations in polars
        # 1. Put potential previous jets in one df
        potentials_df = pl.concat(
            [
                previous_df.filter(
                    pl.col("time") == jtt_idx[0], pl.col("jet ID") == jtt_idx[1]
                )
                for jtt_idx in potentials
            ]
        )

        # 2. Turn into lists
        potentials_df = potentials_df.group_by(
            ["jet ID", "time"], maintain_order=True
        ).agg(pl.col("lon"), pl.col("lat"))
        current_df = current_df.group_by(["jet ID", "time"], maintain_order=True).agg(
            pl.col("lon"), pl.col("lat")
        )

        # 3. create expressions (see function)
        vert_dist, overlap = overlap_vert_dist_polars()

        # perform pairwise using cross-join
        result = potentials_df.join(current_df, how="cross").select(
            old_jet="jet ID",
            new_jet="jet ID_right",
            vert_dist=vert_dist,
            overlap=overlap,
        )

        n_old = potentials_df.shape[0]
        n_new = current_df.shape[0]
        dist_mat = result["vert_dist"].to_numpy().reshape(n_old, n_new)
        overlaps = result["overlap"].to_numpy().reshape(n_old, n_new)

        try:
            dist_mat[np.isnan(dist_mat)] = np.nanmax(dist_mat) + 1
        except ValueError:
            pass

        index_start_flags = time_index_flags[it]
        connected_mask = (overlaps > 0.4) & (dist_mat < 15)
        flagged = np.zeros(n_new, dtype=np.bool_)
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
                flags[int(index_start_flags + j), "flag"] = this_flag
                break
        pass
        for j in range(n_new):
            if not flagged[j]:
                last_flag += 1
                last_valid_index[last_flag] = 0
                all_jets_over_time[last_flag, 0] = (t, j)
                flags[int(index_start_flags + j), "flag"] = last_flag
                flagged[j] = True
        if current.name == "MainProcess":
            pbar.set_description(f"last_flag: {last_flag}")
    ajot_df = []
    for j, ajot in enumerate(all_jets_over_time[: last_flag + 1]):
        times = ajot["time"]
        ajot = ajot[: np.argmax(np.isnat(times))]
        ajot = pl.DataFrame(ajot)
        ajot = ajot.insert_column(
            0, pl.Series("flag", np.full(len(ajot), j, dtype=np.uint32))
        )
        if "member" in index_columns:
            ajot = ajot.insert_column(
                0,
                pl.Series("member", np.full(len(ajot), df["member"][0], dtype=object)),
            )
        ajot_df.append(ajot)
    ajot_df = pl.concat(ajot_df)
    return ajot_df, flags


def track_jets(all_jets_one_df: pl.DataFrame, processes: int = N_WORKERS):
    inner = ["time", "jet ID", "orig_points"]
    levels = all_jets_one_df.columns[: all_jets_one_df.columns.index("lon")]
    outer = [level for level in levels if level not in inner]
    if len(outer) == 0:
        return _track_jets(all_jets_one_df)
    len_, iterator = create_mappable_iterator(all_jets_one_df, potentials=tuple(outer))
    iterator = (a[0] for a in iterator)

    ctx = get_context("spawn")
    lock = ctx.RLock()  # I had to create a fresh lock
    tqdm.set_lock(lock)
    pool_kwargs = dict(initializer=tqdm.set_lock, initargs=(lock,))
    res = map_maybe_parallel(
        iterator,
        _track_jets,
        len_=len_,
        processes=processes,
        chunksize=1,
        pool_kwargs=pool_kwargs,
        ctx=ctx,
    )

    ajots, all_flags = tuple(zip(*res))
    ajots = pl.concat(ajots)
    all_flags = pl.concat(all_flags)
    return ajots, all_flags


def add_persistence_to_props(props_as_df: pl.DataFrame, flags: pl.DataFrame): 
    unique_to_count = flags.group_by("member", maintain_order=True).agg(
        flag=pl.col("flag").unique(),
        flag_count=pl.col("flag").unique_counts(),
    ).explode(["flag", "flag_count"])
    factor = flags["time"].unique().diff()[1] / timedelta(days=1)
    persistence = flags.join(unique_to_count, on=[pl.col("member"), pl.col("flag")])
    persistence = persistence["flag_count"] * factor
    props_as_df = props_as_df.with_columns(persistence=persistence)
    return props_as_df


def compute_prop_anomalies(ds_props: xr.Dataset) -> xr.Dataset:
    prop_anomalies = ds_props.copy()

    for varname in ds_props.data_vars:
        gb = ds_props[varname].groupby("time.year")
        prop_anomalies[varname] = gb - gb.mean(dim="time")
        prop_anomalies[varname] = prop_anomalies[varname] / ds_props[varname].std(
            dim="time"
        )
    return prop_anomalies


def inner_jet_pos_as_da(args: Tuple):
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
    all_jets_one_df: pl.DataFrame,
    basepath: Path,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.DataArray:
    ofile = basepath.joinpath("jet_pos.nc")
    if ofile.is_file():
        return xr.open_dataarray(ofile).load()

    coords = ["member", "time", "lat", "lon"]
    coords = {coord: da_s[coord].values for coord in coords if coord in da_s.dims}
    jet_names = np.asarray(["subtropical", "polar"])
    coords["jet"] = jet_names

    da_jet_pos = xr.DataArray(
        np.zeros([len(val) for val in coords.values()], dtype=np.float32),
        coords=coords,
    )
    len_, iterable = create_mappable_iterator(
        all_jets_one_df, [props_as_ds_uncat["is_polar"], da_jet_pos]
    )

    print("Jet position as da")
    da_jet_pos = map_maybe_parallel(
        iterable,
        inner_jet_pos_as_da,
        len_=len_,
        processes=processes,
        chunksize=chunksize,
    )

    da_jet_pos = xr.concat(da_jet_pos, dim="time")
    da_jet_pos = coarsen_da(da_jet_pos, 1.5)
    da_jet_pos.to_netcdf(ofile)
    return da_jet_pos


# def wave_activity_flux(u, v, z, u_c=None, v_c=None, z_c=None):
#     lon, lat = z.lon.values, z.lat.values
#     cos_lat = degcos(lat[None, :])
#     f = 2 * OMEGA * degsin(lat[:, None])
#     dlon = np.gradient(lon) * np.pi / 180.0
#     dlat = np.gradient(lat) * np.pi / 180.0
#     psi_p = (z - z_c) / f  # Pertubation stream-function

#     # 5 partial differential terms
#     dpsi_dlon = np.gradient(psi_p, dlon[1])[1]
#     dpsi_dlat = np.gradient(psi_p, dlat[1])[0]
#     d2psi_dlon2 = np.gradient(dpsi_dlon, dlon[1])[1]
#     d2psi_dlat2 = np.gradient(dpsi_dlat, dlat[1])[0]
#     d2psi_dlondlat = np.gradient(dpsi_dlat, dlon[1])[1]

#     termxu = dpsi_dlon * dpsi_dlon - psi_p * d2psi_dlon2
#     termxv = dpsi_dlon * dpsi_dlat - psi_p * d2psi_dlondlat
#     termyv = dpsi_dlat * dpsi_dlat - psi_p * d2psi_dlat2

#     # coefficient
#     p_lev = 300.0  # unit in hPa
#     p = p_lev / 1000.0
#     s_c = np.sqrt(u_c**2 + v_c**2)
#     coeff = (p * degcos(lat[None, :])) / (2 * s_c)
#     # x-component of TN-WAF
#     px = (coeff / (RADIUS * RADIUS * cos_lat)) * (
#         ((u_c) / cos_lat) * termxu + v_c * termxv
#     )
#     # y-component of TN-WAF
#     py = (coeff / (RADIUS * RADIUS)) * (((u_c) / cos_lat) * termxv + v_c * termyv)


class JetFindingExperiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
    ) -> None:
        self.ds = data_handler.da
        self.path = data_handler.path
        self.data_handler = data_handler
        self.time = data_handler.get_sample_dims()["time"]

    def find_jets(self, **kwargs) -> pl.DataFrame:
        ofile_ajdf = self.path.joinpath("all_jets_one_df.parquet")

        if ofile_ajdf.is_file():
            all_jets_one_df = pl.read_parquet(ofile_ajdf)
            return all_jets_one_df
        try:
            qs_path = self.path.joinpath("s_q.nc")
            qs = xr.open_dataarray(qs_path)[2]
            kwargs["threshold"] = qs
        except FileNotFoundError:
            pass

        all_jets_one_df = find_all_jets(self.ds, self.path, **kwargs)
        all_jets_one_df.write_parquet(ofile_ajdf)
        return all_jets_one_df

    def compute_jet_props(
        self, processes: int = N_WORKERS, chunksize=100
    ) -> xr.Dataset:
        jet_props_incomplete_path = self.path.joinpath("props_as_ds_uncat_raw.parquet")
        if jet_props_incomplete_path.is_file():
            return pl.read_parquet(jet_props_incomplete_path)
        all_jets_one_df = self.find_jets(processes=processes, chunksize=chunksize)
        props_as_df = compute_jet_props(all_jets_one_df)
        print("Loading s")
        da_ = _compute(self.ds["s"], progress=True)
        width = compute_widths_parallel(
            all_jets_one_df, da_, processes=processes, chunksize=chunksize
        )
        props_as_df = props_as_df.with_columns(width=width["width"])
        props_as_df.write_parquet(jet_props_incomplete_path)
        props_as_ds_uncat = xr.Dataset.from_dataframe(
            props_as_df.to_pandas().set_index(
                get_index_columns(props_as_df, potentials=("member", "time", "cluster"))
            )
        )
        props_as_ds_uncat.to_netcdf(jet_props_incomplete_path)
        return props_as_df

    def track_jets(self) -> Tuple:
        all_jets_one_df = self.find_jets()
        ofile_ajot = self.path.joinpath("all_jets_over_time.parquet")
        ofile_flags = self.path.joinpath("flags.parquet")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]):
            all_jets_over_time = pl.read_parquet(ofile_ajot)
            flags = pl.read_parquet(ofile_flags)

            return (
                all_jets_one_df,
                all_jets_over_time,
                flags,
            )
        all_jets_over_time, flags = track_jets(all_jets_one_df)

        all_jets_over_time.write_parquet(ofile_ajot)
        flags.write_parquet(ofile_flags)

        return (
            all_jets_one_df,
            all_jets_over_time,
            flags,
        )

    def props_as_df(self, categorize: bool = True) -> xr.Dataset:
        ofile_padu = self.path.joinpath("props_as_ds_uncat.parquet")
        ofile_pad = self.path.joinpath("props_as_ds.parquet")
        if ofile_padu.is_file() and not categorize:
            return pl.read_parquet(ofile_padu)
        if ofile_pad.is_file() and categorize:
            return pl.read_parquet(ofile_pad)
        if ofile_padu.is_file() and categorize:
            props_as_df = categorize_df_jets(pl.read_parquet(ofile_padu))
            props_as_df.write_parquet(ofile_pad)
            return props_as_df
        _, all_jets_over_time, flags = self.track_jets()
        props_as_df = self.compute_jet_props()

        props_as_df = is_polar_gmix(
            props_as_df, ["mean_lat", "mean_lon", "mean_lev"], mode="month"
        )
        props_as_df = add_persistence_to_props(props_as_df, flags)
        props_as_df = self.add_com_speed(all_jets_over_time, props_as_df)
        props_as_df.write_parquet(ofile_padu)
        props_as_df_cat = categorize_df_jets(props_as_df)
        props_as_df_cat.write_parquet(ofile_pad)
        if categorize:
            props_as_df_cat
        return props_as_df

    def props_over_time(
        self,
        all_jets_over_time: pl.DataFrame,
        props_as_df_uncat: pl.DataFrame,
        save: bool = True,
    ) -> pl.DataFrame:
        out_path = self.path.joinpath("all_props_over_time.parquet")
        if out_path.is_file():
            return pl.read_parquet(out_path)
        index_columns = get_index_columns(props_as_df_uncat)
        index_exprs = [pl.col(col) for col in index_columns]
        props_as_df_uncat = props_as_df_uncat.cast(
            {"time": all_jets_over_time["time"].dtype}
        )
        all_props_over_time = all_jets_over_time.join(props_as_df_uncat, on=index_exprs)
        sort_on = ["member"] if "member" in index_columns else []
        sort_on.extend(("flag", "time"))
        all_props_over_time = all_props_over_time.sort(sort_on)
        if save:
            all_props_over_time.write_parquet(out_path)
        return all_props_over_time

    def add_com_speed(
        self, all_jets_over_time: pl.DataFrame, props_as_df: pl.DataFrame
    ) -> pl.DataFrame:
        all_props_over_time = self.props_over_time(
            all_jets_over_time,
            props_as_df,
            save=False,
        )
        com_speed = haversine_from_dl_polars(
            pl.col("mean_lat"),
            pl.col("mean_lat").diff(),
            pl.col("mean_lon").diff(),
        ) / (pl.col("time").cast(pl.Float32).diff() / 1e3)
        agg = {
            "time": pl.col("time"),
            "jet ID": pl.col("jet ID"),
            "com_speed": com_speed,
        }
        com_speed = (
            all_props_over_time.group_by(
                get_index_columns(all_props_over_time, ("member", "flag")),
                maintain_order=True,
            )
            .agg(**agg)
            .explode(["time", "jet ID", "com_speed"])
        )
        index_columns = get_index_columns(
            all_props_over_time, ("member", "time", "jet ID")
        )
        index_exprs = [pl.col(col) for col in index_columns]
        props_as_df = props_as_df.cast({"time": com_speed["time"].dtype}).join(
            com_speed, on=index_exprs
        )
        return props_as_df

    def compute_extreme_clim(self, varname: str, subsample: int = 5):
        da = self.ds[varname]
        time = pl.Series("time", self.time)
        years = time.dt.year.values
        mask = np.isin(years, np.unique(years)[::subsample])
        opath = self.path.joinpath(f"q{varname}_clim_{subsample}.nc")
        compute_extreme_climatology(da.isel(time=mask), opath)
        quantiles_clim = xr.open_dataarray(opath)
        quantiles = xr.DataArray(
            np.zeros((len(self.ds.time), quantiles_clim.shape[0])),
            coords={
                "time": self.ds.time.values,
                "quantile": quantiles_clim.coords["quantile"].values,
            },
        )
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
