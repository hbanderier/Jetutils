# coding: utf-8
"""
This probably too big module contains all the utilities relative to jet extraction from 2D fields, jet tracking, jet categorization and jet properties. All of the functions are wrapped by the convenience class `JetFindingExperiment`.
"""
import datetime
import warnings
from itertools import product
from typing import Sequence, Literal
from pathlib import Path

import numpy as np
import polars as pl
from polars import DataFrame, Series, Expr
from polars.exceptions import ColumnNotFoundError
import polars_ds as pds
from scipy.linalg import sqrtm
import xarray as xr
from contourpy import contour_generator
from tqdm import tqdm, trange
from numba import njit
from sklearn.mixture import GaussianMixture
import rustworkx as rx
import dask

from .definitions import (
    RADIUS,
    compute,
    do_rle,
    xarray_to_polars,
    get_index_columns,
    extract_season_from_df,
    explode_rle,
    squarify,
    map_maybe_parallel,
    to_expr,
    iterate_over_year_maybe_member,
    degcos,
)
from .data import (
    compute_extreme_climatology,
    DataHandler,
    open_da,
    open_dataarray,
    smooth,
    to_netcdf,
    coarsen_da,
)
from .anyspell import get_spells
from .geospatial import (
    haversine,
    haversine_from_dl,
    jet_integral_haversine, 
    central_diff,
    diff_maybe_periodic,
    gather_normal_da_jets,
)
from .frechet import fdfd_matrix, earth_haversine_numba


def has_periodic_x(df: DataFrame | xr.Dataset | xr.DataArray) -> bool:
    """
    Checks if the `lon` column contains both sides of the +-180 line. Only makes sense if data went through `.data.standardize()`.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the `lon` column.

    Returns
    -------
    bool
    """
    if isinstance(df, DataFrame):
        lon = df["lon"].unique().sort().to_numpy()
    else:
        lon = np.sort(df["lon"].values)
    dx = lon[1] - lon[0]
    return (-180 in lon) and ((180 - dx) in lon)


def coarsen_pl(df: DataFrame, coarsen_map: dict[str, float]) -> DataFrame:
    """
    Coarsening for polars DataFrame
    """
    index_columns: list[str] = get_index_columns(df)
    other_columns = [
        col for col in df.columns if col not in [*index_columns, *list(coarsen_map)]
    ]
    by = [
        *index_columns,
        *[pl.col(col).floordiv(val) for col, val in coarsen_map.items()],
    ]
    agg = [pl.col(col).max() for col in other_columns]
    # agg = [*[pl.col(col).alias(f"{col}_").mean() for col, val in coarsen_map.items()], *agg]
    df = df.group_by(by, maintain_order=True).agg(*agg)
    # df = df.drop(list(coarsen_map)).rename({f"{col}_": col for col in coarsen_map})
    return df


def round_polars(col: str, factor: int = 2) -> Expr:
    """
    Generates an Expression that rounds the given column to a given base, one over the factor.
    """
    return (pl.col(col) * factor).round() / factor


def compute_norm_derivative(ds: xr.Dataset, of: str = "s"):
    lon, lat = ds["lon"].values, ds["lat"].values
    xlon, ylat = np.meshgrid(lon, lat)

    dlaty, dlatx = np.gradient(ylat)
    dlony, dlonx = np.gradient(xlon)

    dy = RADIUS * np.radians(dlaty)
    dx = RADIUS * np.radians(dlaty) * degcos(ylat)
    
    axis_y = np.where(np.array(ds["s"].dims) == "lat")[0].item()
    axis_x = np.where(np.array(ds["s"].dims) == "lon")[0].item()
    
    u = ds["u"]
    v = ds["v"]
    s = np.hypot(u, v)
    da = ds[of]
    
    ddady = da.copy(data=np.gradient(da, axis=axis_y)) / dy[None, :, :]
    ddadx = da.copy(data=np.gradient(da, axis=axis_x)) / dx[None, :, :]
    
    return (- u * ddady + v * ddadx) / s


def preprocess_ds(
    ds: xr.Dataset, n_coarsen: int = 3, smooth_s: int | None = 5
) -> xr.Dataset:
    ds = coarsen_da(ds, n_coarsen=n_coarsen)
    if "s" not in ds:
        ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)

    to_smooth = ["u", "v", "s", "theta"]
    if smooth_s is not None:
        smooth_map = ("win", smooth_s)
        smooth_map = {"lon": smooth_map, "lat": smooth_map}
        ds = ds.rename({var: f"{var}_orig" for var in to_smooth})
        for var in to_smooth:
            to_smooth = ds[f"{var}_orig"]
            ds[var] = to_smooth.copy(data=smooth(to_smooth, smooth_map=smooth_map))
    ds["sigma"] = compute_norm_derivative(ds, "s")
    ds["sigma_theta"] = compute_norm_derivative(ds, "theta")
    ds["sigma"] = smooth(ds["sigma"], smooth_map=smooth_map)
    ds["sigma_theta"] = smooth(ds["sigma_theta"], smooth_map=smooth_map)
    return ds


def nearest_mapping(df1: DataFrame, df2: DataFrame, col: str):
    """
    Uses the amazing polars' `join_asof` to get a mapping from the unique values in `df1[col]` to the nearest element in the unique values in `df2[col]`.
    """
    df1 = df1.select(col).unique().sort(col)
    df2 = df2.select(col).unique().sort(col).rename({col: f"{col}_"})
    return df1.join_asof(
        df2, left_on=pl.col(col), right_on=pl.col(f"{col}_"), strategy="nearest"
    )


def round_contour(contour: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Coerces a (n, 2) array to the grid defined by `x` and `y`.
    """
    x_ = contour[:, 0]
    y_ = contour[:, 1]
    x_ = x[np.argmin(np.abs(x[:, None] - x_[None, :]), axis=0)]
    y_ = y[np.argmin(np.abs(y[:, None] - y_[None, :]), axis=0)]
    return np.stack([x_, y_], axis=1)


@njit
def distance(x1: float, x2: np.ndarray, y1: float, y2: np.ndarray) -> float:
    dx = x2 - x1
    if np.abs(dx) >= 180:
        dx = 360 - np.abs(dx)  # sign is irrelevant
    dy = y2 - y1
    return np.sqrt(dx**2 + np.sum(dy**2))


@njit()
def my_pairwise(X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
    x1 = X1[:, 0]
    y1 = X1[:, 1:]
    half = False
    if X2 is None:
        X2 = X1
        half = True
    x2 = X2[:, 0]
    y2 = X2[:, 1:]
    output = np.zeros((len(X1), len(X2)))
    for i in range(X1.shape[0] - int(half)):
        if half:
            for j in range(i + 1, X1.shape[0]):
                output[i, j] = distance(x1[j], x2[i], y1[j], y2[i])
                output[j, i] = output[i, j]
        else:
            for j in range(X2.shape[0]):
                output[i, j] = distance(x1[j], x2[i], y1[j], y2[i])
    return output


def find_contours_agg(x, y, z):
    contours, types = contour_generator(
        x, y, (z > 0).astype(np.uint8), line_type="SeparateCode", quad_as_tri=False
    ).lines(0)
    cyclic = np.asarray([79 in types_ for types_ in types])
    contours = [round_contour(contour, x, y) for contour in contours]
    return contours, cyclic


def inner_compute_contours(args):
    """
    Worker function to compute the zero-sigma-contours in a parallel context
    """
    indexer, ds = args
    lon = ds["lon"].values
    lat = ds["lat"].values
    sigma = ds["sigma"].values
    contours, cyclic = find_contours_agg(lon, lat, sigma)
    valid_index = [i for i, contour in enumerate(contours) if len(contour) > 5]
    contours = [
        DataFrame(contours[i], schema={"lon": pl.Float32, "lat": pl.Float32})
        .with_columns(**indexer)
        .with_columns(cyclic=pl.lit(cyclic[i]))
        .with_columns(contour=pl.lit(i))
        for i in valid_index
    ]
    if len(contours) > 0:
        return pl.concat(contours)
    return None


def compute_contours(ds: xr.Dataset) -> DataFrame:
    """
    Potentially parallel wrapper around `inner_compute_contours`. Finds all zero-sigma-contours in all timesteps of `df`.
    """
    extra_dims = {dim: ds[dim] for dim in ds.dims if dim not in ["lon", "lat"]}
    iter1 = list(product(*list(extra_dims.values())))
    iter2 = list((dict(zip(extra_dims, stuff)) for stuff in iter1))
    iter3 = ((indexer, ds.loc[indexer]) for indexer in iter2)
    all_contours = map_maybe_parallel(
        iter3,
        inner_compute_contours,
        len(iter2),
        processes=1,
    )
    all_contours = pl.concat(
        [contour for contour in all_contours if contour is not None]
    )
    return all_contours


def compute_alignment(
    all_contours: DataFrame, periodic: bool = False
) -> DataFrame:
    """
    This function computes the alignment criterion for zero-sigma-contours. It is the scalar product betweeen the vector from a contour point to the next and the horizontal wind speed vector.
    """
    index_columns = get_index_columns(
        all_contours, ("member", "time", "cluster", "spell", "relative_index")
    )
    dlon = diff_maybe_periodic("lon", periodic)
    dlat = central_diff("lat")
    ds = haversine_from_dl(pl.col("lat"), dlon, dlat)
    align_x = pl.col("u") / pl.col("s") * RADIUS * pl.col("lat").radians().cos() * dlon.radians() / ds
    align_y = pl.col("v") / pl.col("s") * RADIUS * dlat.radians() / ds
    alignment = align_x + align_y
    return all_contours.with_columns(
        alignment=alignment.over([*index_columns, "contour"])
    )


def find_all_jets(
    ds: xr.Dataset,
    thresholds: xr.DataArray | None = None,
    n_coarsen: int = 3, 
    smooth_s: int = 5,
    base_s_thresh: float = 0.5,
    alignment_thresh: float = 0.6,
    int_thresh_factor: float = 0.6,
    hole_size: int = 10,
):
    """
    Main function to find all jets in a polars DataFrame containing at least the "lon", "lat", "u", "v" and "s" columns. Will group by any potential index columns to compute jets independently for every, timestep, member and / or cluster. Any other non-index column present in `df` (like "theta" or "lev") will be interpolated to the jet cores in the output.

    Thresholds passed as a DataArray are wind speed thresholds. This Dataarray needs to have one value per timestep present in `df`. If not passed, `base_s_thresh` is used for all times.

    The jet integral threshold is computed from the wind speed threshold.
    """
    # process input
    ds = preprocess_ds(ds, n_coarsen=n_coarsen, smooth_s=smooth_s)
    smoothed_to_remove = ("u", "v", "s")
    df = xarray_to_polars(
        ds.drop_vars(smoothed_to_remove).rename(
            {f"{var}_orig": var for var in smoothed_to_remove}
        )
    )
    x_periodic = has_periodic_x(ds)
    index_columns = get_index_columns(
        df,
        (
            "member",
            "time",
            "cluster",
            "jet ID",
            "spell",
            "relative_index",
            "sample_index",
            "inside_index",
        ),
    )

    # thresholds
    dl = np.radians(df["lon"].max() - df["lon"].min())
    base_int_thresh = (
        RADIUS * dl * base_s_thresh * np.cos(np.pi / 4) * int_thresh_factor
    )
    if base_s_thresh <= 1.0:
        thresholds = df.group_by(index_columns).agg(
            pl.col("s").quantile(base_s_thresh).alias("s_thresh")
        )
        base_s_thresh = thresholds["s_thresh"].mean()  # disgusting
        base_int_thresh = (
            RADIUS * dl * base_s_thresh * np.cos(np.pi / 4) * int_thresh_factor
        )
    elif thresholds is not None:
        thresholds = (
            pl.from_pandas(thresholds.to_dataframe().reset_index())
            .drop("quantile")
            .cast({"s": pl.Float32})
            .rename({"s": "s_thresh"})
        )
    if thresholds is not None:
        df = df.join(
            thresholds, on=[ic for ic in index_columns if ic != "member"]
        )  # ugly
        df = df.with_columns(
            int_thresh=base_int_thresh * pl.col("s_thresh") / base_s_thresh
        )
        condition_expr = (pl.col("s") > pl.col("s_thresh")) & (
            pl.col("alignment") > alignment_thresh
        )
        condition_expr2 = pl.col("int") > pl.col("int_thresh")
    else:
        condition_expr = (pl.col("s") > base_s_thresh) & (
            pl.col("alignment") > alignment_thresh
        )
        condition_expr2 = pl.col("int") > base_int_thresh

    # contours
    all_contours = compute_contours(ds)

    diff_exp = pl.col("lon").diff().abs()
    diff_exp = pl.when(diff_exp > 180).then(360 - diff_exp).otherwise(diff_exp)
    diff_exp = (diff_exp.abs() + pl.col("lat").diff().abs()).fill_null(10.0)
    newindex = (pl.col("index").cast(pl.Int32()) - diff_exp.arg_max()) % (
        pl.col("index").max() + 1
    )
    int_expr = jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s"))
    int_expr = int_expr.over([*index_columns, "jet ID"])
    distance_ends_expr = haversine(pl.col("lon").first(), pl.col("lat").first(), pl.col("lon").last(), pl.col("lat").last())
    distance_ends_expr = distance_ends_expr.over([*index_columns, "jet ID"])

    all_contours = (
        all_contours.with_columns(len=pl.len().over([*index_columns, "contour"]))
        .filter(pl.col("len") > 6)
        .with_columns(index=pl.int_range(0, pl.len()).over([*index_columns, "contour"]))
        .unique([*index_columns, "contour", "index"])
        .sort([*index_columns, "contour", "index"])
        .with_columns(newindex.over([*index_columns, "contour"]))
        .unique([*index_columns, "contour", "index"])
        .sort([*index_columns, "contour", "index"])
    )

    for index_column in index_columns:
        try:
            all_contours = all_contours.cast({index_column: df[index_column].dtype})
        except ColumnNotFoundError:
            pass
    all_contours = all_contours.join(df, on=[*index_columns, "lon", "lat"], how="left")
    all_contours = compute_alignment(all_contours, x_periodic)

    # jets from contours
    ## consecutive runs of contour points respecting both point wise conditions, allowing holes of size up to three
    valids = (
        explode_rle(
            do_rle(
                all_contours.with_columns(condition=condition_expr),
                [*index_columns, "contour"],
            )
            .with_columns(value=pl.col("value").fill_null(False))
            .filter(
                (pl.col("value").not_() & (pl.col("len") < hole_size)) | pl.col("value")
            )
        )
        .drop("value", "len", "start")
        .group_by([*index_columns, "contour"], maintain_order=True)
        .agg(pl.col("index"), len=pl.col("index").len())
        .filter(pl.col("len") > 5)
        .explode("index")
    )
    jets = valids.join(all_contours, on=[*index_columns, "contour", "index"])
    jets = (
        jets.with_columns(len=pl.len().over([*index_columns, "contour"]))
        .filter(pl.col("len") > 6)
        .with_columns(index=pl.int_range(0, pl.len()).over([*index_columns, "contour"]))
        .unique([*index_columns, "contour", "index"])
        .sort([*index_columns, "contour", "index"])
        .with_columns(newindex.over([*index_columns, "contour"]))
        .unique([*index_columns, "contour", "index"])
        .sort([*index_columns, "contour", "index"])
        .with_columns(diff=diff_exp.over([*index_columns, "contour"]))
        .with_columns(
            contour=(pl.col("contour") + 0.01 * (pl.col("diff") > 10).cum_sum())
            .rle_id()
            .over(index_columns)
        )
        .rename({"contour": "jet ID"})
        .drop("cyclic", "len", "len_right")
    )

    jets = (
        jets.with_columns(len=pl.len().over([*index_columns, "jet ID"]))
        .filter(pl.col("len") > 6)
        .with_columns(index=pl.int_range(0, pl.len()).over([*index_columns, "jet ID"]))
        .unique([*index_columns, "jet ID", "index"])
        .sort([*index_columns, "jet ID", "index"])
        .with_columns(newindex.over([*index_columns, "jet ID"]))
        .unique([*index_columns, "jet ID", "index"])
        .sort([*index_columns, "jet ID", "index"])
        .with_columns(
            int=int_expr,
            distance_ends=distance_ends_expr,
        )
        .filter(condition_expr2, pl.col("distance_ends") > 1e6)
        .drop("distance_ends")
        .with_columns(pl.col("jet ID").rle_id().over([*index_columns]))
        .with_columns(newindex.over([*index_columns, "jet ID"]))
        .unique([*index_columns, "jet ID", "index"])
        .sort([*index_columns, "jet ID", "index"])
    )

    return jets


def weighted_mean_pl(col: Expr | str, by: Expr | str | None = None):
    col = to_expr(col)
    if by is None:
        return col.mean()
    by = to_expr(by)
    return ((col * by).sum() / by.sum()).alias(col.meta.output_name())


def circular_mean(col: Expr | str, weights: Expr | str | None = None):
    col = to_expr(col)
    col = col.radians()
    mean_sin = weighted_mean_pl(col.sin(), weights)
    mean_cos = weighted_mean_pl(col.cos(), weights)
    return pl.arctan2(mean_sin, mean_cos).degrees().alias(col.meta.output_name())


def compute_jet_props(df: DataFrame) -> DataFrame:
    """
    Computes all basic jet properties from a DataFrame containing many jets.
    """
    position_columns = [col for col in ["lat", "lev", "theta"] if col in df.columns]

    def dl(col):
        return pl.col(col).max() - pl.col(col).min()

    mean_lon = circular_mean("lon", "s").alias("mean_lon")

    diff_lon = pl.col("lon").diff()
    diff_lon = (
        pl.when(diff_lon > 180)
        .then(diff_lon - 360)
        .when(diff_lon <= -180)
        .then(diff_lon + 360)
        .otherwise(diff_lon)
    )

    unraveled_lon = pl.col("lon").first() + diff_lon.cum_sum().fill_null(0.0)

    aggregations = [
        mean_lon,
        *[weighted_mean_pl(col, "s").alias(f"mean_{col}") for col in position_columns],
        pl.col("s").mean().alias("mean_s"),
        *[
            pl.col(col).get(pl.col("s").arg_max()).alias(f"{col}_star")
            for col in ["lon", "lat", "s"]
        ],
        *[dl(col).alias(f"{col}_ext") for col in ["lon", "lat"]],
        (
            pds.lin_reg_report(unraveled_lon, target=pl.col("lat"), add_bias=True)
            .struct.field("beta")
            .first()
            .alias("tilt")
        ),
        (
            1
            - pds.lin_reg_report(unraveled_lon, target=pl.col("lat"), add_bias=True)
            .struct.field("r2")
            .first()
        ).alias("waviness1"),
        (pl.col("lat") - pl.col("lat").mean()).pow(2).sum().alias("waviness2"),
        (
            pl.col("lat").gather(pl.col("lon").arg_sort()).diff().abs().sum()
            / dl("lon")
        ).alias("wavinessR16"),
        (
            jet_integral_haversine(pl.col("lon"), pl.col("lat"), x_is_one=True)
            / pl.lit(RADIUS)
            / pl.col("lat").mean().radians().cos()
            / dl("lon")
        ).alias("wavinessDC16"),
        (
            ((pl.col("v") - pl.col("v").mean()) * pl.col("v").abs() / pl.col("s")).sum()
            / pl.col("s").sum()
        ).alias("wavinessFV15"),
        jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s")).alias("int"),
        jet_integral_haversine(
            pl.col("lon").filter(pl.col("lon") > -10),
            pl.col("lat").filter(pl.col("lon") > -10),
            pl.col("s").filter(pl.col("lon") > -10),
        ).alias("int_over_europe"),
        pl.col("is_polar").mean(),
    ]

    df_lazy = df.lazy()
    index_columns = get_index_columns(df)
    if "member" not in get_index_columns(df):
        gb = df_lazy.group_by(index_columns, maintain_order=True)
        props_as_df = gb.agg(*aggregations)
        props_as_df = props_as_df.with_columns(
            [
                pl.col(col)
                .replace([float("inf"), float("-inf")], None)
                .clip(pl.col(col).quantile(0.00001), pl.col(col).quantile(0.99999))
                for col in ["tilt", "waviness1", "wavinessDC16"]
            ]
        )
        return props_as_df.collect()

    # streaming mode doesn't work well
    collected = []
    for member in tqdm(df["member"].unique(maintain_order=True).to_numpy()):
        gb = df_lazy.filter(pl.col("member") == member).group_by(
            get_index_columns(df), maintain_order=True
        )
        props_as_df = gb.agg(*aggregations)
        collected.append(props_as_df.collect())
    props_as_df = pl.concat(collected).sort("member")
    return props_as_df


def compute_widths(jets: DataFrame, da: xr.DataArray):
    """
    Computes the width of each jet using normally interpolated wind speed on either side of the jet.
    """
    jets = gather_normal_da_jets(jets, da, 12.0, 1.0, delete_middle=True)

    index_columns = get_index_columns(
        jets, ("member", "time", "cluster", "spell", "relative_index", "jet ID")
    )

    # Expr half_width
    below = pl.col("s_interp") <= pl.max_horizontal(pl.col("s") / 4 * 3, pl.lit(10))
    stop_up = below.arg_max()
    nlo_up = pl.col("normallon").gather(stop_up)
    nla_up = pl.col("normallat").gather(stop_up)
    half_width_up = haversine(
        nlo_up, nla_up, pl.col("lon").first(), pl.col("lat").first()
    ).cast(pl.Float32)
    half_width_up = pl.when(below.any()).then(half_width_up).otherwise(float("nan"))

    stop_down = below.len() - below.reverse().arg_max() - 1
    nlo_down = pl.col("normallon").gather(stop_down)
    nla_down = pl.col("normallat").gather(stop_down)

    agg_out = {ic: pl.col(ic).first() for ic in ["lon", "lat", "s"]}

    half_width_down = haversine(
        nlo_down, nla_down, pl.col("lon").first(), pl.col("lat").first()
    ).cast(pl.Float32)
    half_width_down = pl.when(below.any()).then(half_width_down).otherwise(float("nan"))

    half_width = (
        pl.when(pl.col("side") == -1)
        .then(pl.col("half_width_down"))
        .otherwise(pl.col("half_width_up"))
        .list.first()
    )

    first_agg_out = agg_out | {
        "half_width_up": half_width_up,
        "half_width_down": half_width_down,
    }
    second_agg_out = agg_out | {"half_width": pl.col("half_width").sum()}
    third_agg_out = agg_out | {
        "width": (pl.col("half_width").fill_nan(None) * pl.col("s")).sum()
        / pl.col("s").sum()
    }

    jets = (
        jets.lazy()
        .group_by([*index_columns, "index", "side"], maintain_order=True)
        .agg(**first_agg_out)
        .with_columns(half_width=half_width)
        .drop(["half_width_up", "half_width_down", "side"])
        .group_by([*index_columns, "index"])
        .agg(**second_agg_out)
        .group_by(index_columns)
        .agg(**third_agg_out)
        .drop("lon", "lat", "s")
        .cast({"width": pl.Float32})
        .sort(index_columns)
    )
    return jets.collect()


def join_wrapper(
    df: DataFrame, da: xr.DataArray | xr.Dataset, suffix: str = "_right", **kwargs
):
    """
    Joins a DataFrame with a DataArray on the latter's dimensions. Explicitly iterates over years and members to limit memory usage

    Parameters
    ----------
    df : DataFrame
        A DataFrame with columns also found in da
    da : xr.DataArray | xr.Dataset
        Xarray object whose values to join to the DataFrame
    suffix : str, optional
        join suffix, by default "_right"
    kwargs :
        keyword arguments passed to ``iterate_over_year_maybe_member``

    Returns
    -------
    DataFrame
        Original DataFrame with one or several extra columns from da.
    """
    indexer = iterate_over_year_maybe_member(df, da, **kwargs)
    df_upd = []
    dims = da.dims
    for idx1, idx2 in tqdm(list(indexer)):
        these_jets = df.filter(*idx1)
        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            da_ = compute(da.sel(**idx2), progress_flag=False)
        da_ = xarray_to_polars(da_)
        these_jets = these_jets.join(da_, on=dims, how="left", suffix=suffix)
        df_upd.append(these_jets)
    df = pl.concat(df_upd)
    return df


def round_half(x):
    """
    Round to the nearest integer or half integer

    Parameters
    ----------
    x : Number or array
        Array-like to round

    Returns
    -------
    Same as input
        Rounded input
    """
    return np.round(x * 2) / 2


def extract_features(
    df: DataFrame,
    feature_names: Sequence = None,
    season: list | str | tuple | int | None = None,
) -> DataFrame:
    """
    Tiny wrapper to extract columns and subset time from a polars DataFrame.
    """
    df = extract_season_from_df(df, season)
    if feature_names is None:
        feature_names = ["mean_lon", "mean_lat", "s_star"]

    return df[feature_names]


def one_gmix(
    X,
    n_components=2,
    n_init=20,
    init_params="random_from_data",
):
    """
    Trains one Gaussian Mixture model, and outputs the predicted probability of all data points on the component identified as the eddy-driven jet.

    Parameters
    ----------
    X : DataFrame
        Input data
    n_components : int
        Number of Gaussian components, by default 2
    n_init : int
        Number of repeated independent training with. Can massively increase run time. By default 20
    init_params : str, optional
        Type of init, by default "random_from_data"

    Returns
    -------
    ndarray
        Probabilities of every point on the Gaussian component identified as the eddy-driven jet.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """
    model = GaussianMixture(
        n_components=n_components, init_params=init_params, n_init=n_init
    )
    if "ratio" in X.columns:
        X = X.with_columns(ratio=pl.col("ratio").fill_null(1.0).fill_nan(1.0))
    model = model.fit(X)
    if X.columns[1] == "theta":
        return 1 - model.predict_proba(X)[:, np.argmax(model.means_[:, 1])]
    elif X.columns[1] == "lat":
        return 1 - model.predict_proba(X)[:, np.argmin(model.means_[:, 1])]
    return 1 - model.predict_proba(X)[:, np.argmax(model.means_[:, 0])]


def one_gmix_v2(
    X,
    n_components=2,
    n_init=20,
    init_params="random_from_data",
):
    """
    Trains one Gaussian Mixture model, and outputs the predicted probability of all data points on the component identified as the eddy-driven jet.

    Parameters
    ----------
    X : DataFrame
        Input data
    n_components : int
        Number of Gaussian components, by default 2
    n_init : int
        Number of repeated independent training with. Can massively increase run time. By default 20
    init_params : str, optional
        Type of init, by default "random_from_data"

    Returns
    -------
    ndarray
        Probabilities of every point on the Gaussian component identified as the eddy-driven jet.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """
    model = GaussianMixture(
        n_components=n_components, init_params=init_params, n_init=n_init
    )
    if "ratio" in X.columns:
        X = X.with_columns(ratio=pl.col("ratio").fill_null(1.0))
    model = model.fit(X)
    scores = []
    for mean, covar in zip(model.means_, model.covariances_):
        covar = sqrtm(np.linalg.inv(covar))
        x = X.to_numpy() - mean[None, :]
        score = np.linalg.norm(np.einsum("jk,ik->ij", covar, x), axis=1)
        scores.append(score)
    if X.columns[1][:5] == "theta":
        order = np.argsort(model.means_[:, 1])
    else:
        order = np.argsort(model.means_[:, 1])[::-1]
    otherscores = np.sum([scores[k] for k in order[:-1]], axis=0)
    return 1 / (1 + otherscores / scores[order[-1]])


def is_polar_gmix(
    df: DataFrame,
    feature_names: list,
    mode: (
        Literal["all"] | Literal["season"] | Literal["month"] | Literal["week"]
    ) = "week",
    n_components: int | Sequence = 2,
    n_init: int = 20,
    init_params: str = "random_from_data",
    v2: bool = True,
) -> DataFrame:
    """
    Trains one or several Gaussian Mixture model independently, depending on the `mode` argument.

    Parameters
    ----------
    df : DataFrame
        DataFrame to cluster
    feature_names : list
        Which columns to cluster on
    mode : "all", "season", "month" or "week
        Trains one model if "all", otherwise train one independent model for each season, month or week of the year. By default "week".
    n_components : int
        Number of Gaussian components, by default 2
    n_init : int
        Number of repeated independent training with. Can massively increase run time. By default 20
    init_params : str, optional
        Type of init, by default "random_from_data"

    Returns
    -------
    DataFrame
        DataFrame of same length as `df` with the same index columns, the `feature_names` columns and a new `is_polar` column, corresponding to the proability of each row to belong to the eddy-driven jet component.
    """
    # TODO: assumes at least one year of data, check for season / month actually existing in the data, figure out output
    kwargs = dict(n_init=n_init, init_params=init_params)
    gmix_fn = one_gmix_v2 if v2 else one_gmix
    if "time" not in df.columns:
        mode = "all"
    if mode == "all":
        X = extract_features(df, feature_names, None)
        kwargs["n_components"] = n_components
        probas = gmix_fn(X, **kwargs)
        return df.with_columns(is_polar=probas)
    index_columns = get_index_columns(df)
    to_concat = []
    if mode == "season":
        if isinstance(n_components, int):
            n_components = [n_components] * 4
        else:
            assert len(n_components) == 4
        for season, n_components_ in zip(
            tqdm(["DJF", "MAM", "JJA", "SON"]), n_components
        ):
            X = extract_features(df, feature_names, season)
            kwargs["n_components"] = n_components_
            probas = gmix_fn(X, **kwargs)
            to_concat.append(
                extract_season_from_df(df, season).with_columns(is_polar=probas)
            )
    elif mode == "month":
        if isinstance(n_components, int):
            n_components = [n_components] * 12
        else:
            assert len(n_components) == 12
        for month, n_components_ in zip(trange(1, 13), n_components):
            X = extract_features(df, feature_names, month)
            kwargs["n_components"] = n_components_
            probas = gmix_fn(X, **kwargs)
            to_concat.append(
                extract_season_from_df(df, month).with_columns(is_polar=probas)
            )
    elif mode == "week":
        weeks = df["time"].dt.week().unique().sort().to_numpy()
        if isinstance(n_components, int):
            n_components = [n_components] * len(weeks)
        else:
            assert len(n_components) == len(weeks)
        for week, n_components_ in zip(tqdm(weeks, total=len(weeks)), n_components):
            X = df.filter(pl.col("time").dt.week() == week)
            X_ = extract_features(X, feature_names)
            kwargs["n_components"] = n_components_
            probas = gmix_fn(X_, **kwargs)
            to_concat.append(X.with_columns(is_polar=probas))

    return pl.concat(to_concat).sort([*index_columns, "index"])


def add_feature_for_cat(
    jets: DataFrame,
    feature: str,
    ds: xr.Dataset | None = None,
    ds_low: xr.Dataset | xr.DataArray | None = None,
    ofile_ajdf: Path | None = None,
    force: bool = False,
):
    if (feature in jets.columns) and not force:
        return jets
    if feature in jets.columns:
        jets = jets.drop(feature)
    if feature == "ratio":
        if isinstance(ds_low, xr.Dataset):
            da = ds_low["s"]
        else:
            da = ds_low
        if "s_low" in jets.columns:
            jets = jets.drop("s_low")
        jets = join_wrapper(jets, da, suffix="_low")
        jets = jets.with_columns(ratio=pl.col("s_low") / pl.col("s"))
    else:
        da = ds[feature]
        jets = join_wrapper(jets, da)
    if ofile_ajdf is not None:
        jets.write_parquet(ofile_ajdf)
    return jets


def categorize_jets(
    jets: DataFrame,
    ds: xr.Dataset | None = None,
    low_wind: xr.Dataset | xr.DataArray | None = None,
    feature_names: tuple | None = None,
    force: int = 0,
    mode: (
        Literal["year"] | Literal["season"] | Literal["month"] | Literal["week"]
    ) = "week",
    n_components: int | Sequence = 2,
    n_init: int = 20,
    init_params: str = "random_from_data",
    v2: bool = True,
):
    if "is_polar" in jets.columns and not force:
        return jets
    if feature_names is None:
        feature_names = ("ratio", "theta")
    if "ratio" in feature_names and low_wind is None:
        print("you need to provide low wind")
        raise ValueError

    if isinstance(low_wind, xr.Dataset):
        low_wind = low_wind[["s"]]

    lon = jets["lon"].unique().sort().to_numpy()
    lat = jets["lat"].unique().sort().to_numpy()
    if low_wind is not None:
        low_wind = low_wind.interp(lon=lon, lat=lat)
    for feat in feature_names:
        jets = add_feature_for_cat(
            jets,
            feat,
            ds,
            low_wind,
            force=force > 1,
        )

    jets = is_polar_gmix(
        jets,
        feature_names=feature_names,
        mode=mode,
        n_components=n_components,
        n_init=n_init,
        init_params=init_params,
        v2=v2,
    )
    return jets


def average_jet_categories_v2(props_as_df: DataFrame):
    """
    For every timestep, member and / or cluster (whichever applicable), aggregates each jet property (with a different rule for each property but usually a mean) into a single number for each category: subtropical or eddy driven, summarizing this property for all the jets in this snapshot that fit this category. This version does it as a weighted mean over all jets, with weights either ?`is_polar` or `1 - is_polar`. Honestly nonsensical, don't use except to prove that v1 is better.

    Parameters
    ----------
    props_as_df : DataFrame
        Uncategorized jet properties, that contain at least the `jet ID` column.

    Returns
    -------
    props_as_df
        Categorized jet properties. The columns `jet ID` does not exist anymore, and a new column `jet` with two or three possible values has been added. Two possible values if `allow_hybrid=False`: "STJ" or "EDJ". If `allow_hybrid=True`, the third `hybrid` category can also be found in the output `props_as_df`.
    """

    def polar_weights(is_polar: bool = False):
        return pl.col("is_polar") if is_polar else 1 - pl.col("is_polar")

    index_columns = get_index_columns(
        props_as_df, ("member", "time", "cluster", "spell", "relative_index")
    )
    other_columns = [
        col for col in props_as_df.columns if col not in [*index_columns, "jet"]
    ]
    agg = [
        {
            col: weighted_mean_pl(
                pl.col(col).fill_nan(0.0), pl.col("int") * polar_weights(bool(i))
            )
            for col in other_columns
        }
        for i in range(2)
    ]
    for i in range(2):
        agg[i]["int"] = weighted_mean_pl("int", polar_weights(bool(i)))
        agg[i]["is_polar"] = weighted_mean_pl("is_polar", polar_weights(bool(i)))
        agg[i]["njets"] = (pl.col("is_polar") < 0.5).sum().cast(pl.UInt8())

    jet_names = ["STJ", "EDJ"]
    props_as_df_cat = [
        props_as_df.group_by(index_columns)
        .agg(**agg[i])
        .with_columns(jet=pl.lit(jet_names[i]))
        for i in range(2)
    ]
    props_as_df_cat = pl.concat(props_as_df_cat)

    if "member" in index_columns:
        dummy_indexer = (
            props_as_df_cat["member"]
            .unique()
            .sort()
            .to_frame()
            .join(
                props_as_df_cat["time"].unique().sort().to_frame(),
                how="cross",
            )
            .join(
                props_as_df_cat["jet"].unique().sort(descending=True).to_frame(),
                how="cross",
            )
        )
    elif "cluster" in index_columns:
        dummy_indexer = (
            props_as_df_cat["cluster"]
            .unique()
            .sort()
            .to_frame()
            .join(
                props_as_df_cat["jet"].unique().sort(descending=True).to_frame(),
                how="cross",
            )
        )
    else:
        dummy_indexer = (
            props_as_df_cat["time"]
            .unique()
            .sort()
            .to_frame()
            .join(
                props_as_df_cat["jet"].unique().sort(descending=True).to_frame(),
                how="cross",
            )
        )
    new_index_columns = get_index_columns(
        props_as_df_cat, ("member", "time", "cluster", "jet", "spell", "relative_index")
    )

    sort_descending = [False] * len(new_index_columns)
    sort_descending[-1] = True
    props_as_df_cat = dummy_indexer.join(
        props_as_df_cat, on=[pl.col(col) for col in new_index_columns], how="left"
    ).sort(new_index_columns, descending=sort_descending)
    props_as_df_cat = props_as_df_cat.with_columns(pl.col("njets").fill_null(0))
    return props_as_df_cat


def average_jet_categories(
    props_as_df: DataFrame,
    polar_cutoff: float | None = None,
    allow_hybrid: bool = False,
):
    """
    For every timestep, member and / or cluster (whichever applicable), aggregates each jet property (with a different rule for each property but usually a mean) into a single number for each category: subtropical, eddy driven jet and potentially hybrid, summarizing this property fo all the jets in this snapshot that fit this category, based on their mean `is_polar` value and a threshold given by `polar_cutoff`.

    E.g. on the 1st of January 1999, there are two jets with `is_polar < polar_cutoff` and one with `is_polar > polar_cutoff`. We pass `allow_hybrid=False` to the function. In the output, for the row corresponding to this date and `jet=STJ`, the value for the `"mean_lat"` column will be the mean of the `"mean_lat"` values of two jets that had `is_polar < polar_cutoff`.

    Parameters
    ----------
    props_as_df : DataFrame
        Uncategorized jet properties, that contain at least the `jet ID` column.
    polar_cutoff : float | None, optional
        Cutoff, by default None
    allow_hybrid : bool, optional
        Whether to output two or three jet categories (hybrid jet between EDJ and STJ), by default False

    Returns
    -------
    props_as_df
        Categorizes jet properties. The columns `jet ID` does not exist anymore, and a new column `jet` with two or three possible values has been added. Two possible values if `allow_hybrid=False`: "STJ" or "EDJ". If `allow_hybrid=True`, the third `hybrid` category can also be found in the output `props_as_df`.
    """
    if allow_hybrid and polar_cutoff is None:
        polar_cutoff = 0.15
    elif polar_cutoff is None:
        polar_cutoff = 0.5
    if allow_hybrid:
        props_as_df = props_as_df.with_columns(
            pl.when(pl.col("is_polar") > 1 - polar_cutoff)
            .then(pl.lit("EDJ"))
            .when(pl.col("is_polar") < polar_cutoff)
            .then(pl.lit("STJ"))
            .otherwise(pl.lit("Hybrid"))
            .alias("jet")
        )
    else:
        props_as_df = props_as_df.with_columns(
            pl.when(pl.col("is_polar") >= polar_cutoff)
            .then(pl.lit("EDJ"))
            .otherwise(pl.lit("STJ"))
            .alias("jet")
        )
    index_columns = get_index_columns(
        props_as_df, ("member", "time", "cluster", "jet ID", "spell", "relative_index")
    )
    other_columns = [
        col for col in props_as_df.columns if col not in [*index_columns, "jet"]
    ]
    agg = {
        col: (pl.col(col) * pl.col("int")).sum() / pl.col("int").sum()
        for col in other_columns
    }
    agg["int"] = pl.col("int").mean()
    agg["is_polar"] = pl.col("is_polar").mean()
    agg["s_star"] = pl.col("s_star").max()
    agg["lon_ext"] = pl.col("lon_ext").max()
    agg["lat_ext"] = pl.col("lat_ext").max()
    agg["njets"] = pl.col("int").len().cast(pl.UInt8())

    gb_columns = get_index_columns(
        props_as_df, ("member", "time", "cluster", "jet", "spell", "relative_index")
    )
    props_as_df_cat = (
        props_as_df.group_by(gb_columns, maintain_order=True)
        .agg(**agg)
        .sort(gb_columns)
    )

    if "member" in index_columns:
        dummy_indexer = (
            props_as_df_cat["member"]
            .unique(maintain_order=True)
            .to_frame()
            .join(
                props_as_df_cat["time"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
            .join(
                props_as_df_cat["jet"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
        )
    elif "cluster" in index_columns:
        dummy_indexer = (
            props_as_df_cat["cluster"]
            .unique(maintain_order=True)
            .to_frame()
            .join(
                props_as_df_cat["jet"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
        )
    else:
        dummy_indexer = (
            props_as_df_cat["time"]
            .unique(maintain_order=True)
            .to_frame()
            .join(
                props_as_df_cat["jet"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
        )
    new_index_columns = get_index_columns(
        props_as_df_cat, ("member", "time", "cluster", "jet", "spell", "relative_index")
    )

    sort_descending = [False] * len(new_index_columns)
    sort_descending[-1] = True
    props_as_df_cat = dummy_indexer.join(
        props_as_df_cat, on=[pl.col(col) for col in new_index_columns], how="left"
    ).sort(new_index_columns, descending=sort_descending)
    props_as_df_cat = props_as_df_cat.with_columns(pl.col("njets").fill_null(0))
    return props_as_df_cat


def _frechet_of_row(row):
    if any(r is None for r in row):
        return 0
    p = np.asarray(row[0])
    q = np.asarray(row[1])
    return fdfd_matrix(p, q, earth_haversine_numba) 


def _lons_from_points(points: str | pl.Expr = "points") -> pl.Expr:
    points = to_expr(points)
    return points.list.eval(pl.element().arr.first())


def _lon_overlap(points: str | pl.Expr = "points", points_right: str | pl.Expr = "points_right") -> pl.Expr:
    overlap = _lons_from_points(points).list.set_intersection(_lons_from_points(points_right))
    overlap = overlap.list.len() * (1 / _lons_from_points(points_right).list.len() + 1 / _lons_from_points(points_right).list.len()) / 2
    return overlap


def track_jets_one_year(
    jets: DataFrame, year: int, member: str | None = None
) -> DataFrame:
    """
    Performs one year of explicit jet tracking

    Parameters
    ----------
    jets : DataFrame
        _description_
    year : int
        _description_
    member : str | None, optional
        _description_, by default None

    Returns
    -------
    cross: DataFrame
        cross-jet distance metrics from a jet and its best successor
    """
    jets = jets.cast({"time": pl.Datetime("ms")})
    if "len_right" in jets.columns:
        jets = jets.drop("len_right")
    deltas = ["s", "theta"]
    if "is_polar" in jets.columns:
        deltas.append("is_polar")
    deltas2 = [f"d{col}" for col in deltas]
    typical_group_by = ["time", "time_right", "jet ID", "jet ID_right"]
    filter_ = pl.col("time").dt.year() == year
    if (year + 1) in jets["time"].dt.year().unique():
        filter_ = filter_ | pl.col("time").is_in(
            pl.col("time")
            .filter(pl.col("time").dt.year() == year + 1)
            .unique()
            .bottom_k(1)
            .implode()
        )
    if "member" in jets.columns and member is not None:
        filter_ = filter_ & pl.col("member") == member
        typical_group_by_ = ["member"]
        typical_group_by_.extend(typical_group_by)
        typical_group_by = typical_group_by_

    jets_current = jets.filter(filter_).with_columns(
        len=pl.col("lon").len().over(["time", "jet ID"]),
        index=pl.int_range(0, pl.col("lon").len()).over(["time", "jet ID"]),
    )
    dt = jets_current["time"].unique().bottom_k(2).sort()
    dt = dt[1] - dt[0]
    jets_next = jets_current.with_columns(time_shifted=pl.col("time") - dt)
    jets_current = jets_current.filter(pl.col("time").dt.year() == year)
    
    aggs_first = {"points": pl.concat_arr("lon", "lat"), "s": weighted_mean_pl("s")} | {d: weighted_mean_pl(d, "s") for d in deltas[1:]}
    
    jets_current= jets_current.group_by("time", "jet ID").agg(**aggs_first)
    jets_next = jets_next.group_by("time", "time_shifted", "jet ID").agg(**aggs_first)
    cross = jets_current.join(jets_next, left_on="time", right_on="time_shifted", how="left")
    overlap = _lon_overlap()
    more_aggs = {
        "dist": cross["points", "points_right"].map_rows(_frechet_of_row)["map"],
        "lon_overlap": overlap,
    } | {d2: (pl.col(d1) - pl.col(f"{d1}_right")).abs() for d1, d2 in zip(deltas, deltas2)}
    cross = (
        cross
        .with_columns(**more_aggs)
        .drop(*[f"{name}{suffix}" for name in ["points"] for suffix in ["", "_right"]])
        .drop_nulls("lon_overlap")
        .sort(*typical_group_by)
    )
    return cross


def track_jets(all_jets_one_df: DataFrame) -> DataFrame:
    """
    Iterates over years and maybe members and performs explicit jet tracking.

    Parameters
    ----------
    all_jets_one_df : DataFrame
        Data source

    Returns
    -------
    DataFrame
        _description_
    """
    cross = []
    gb = ["time", "jet ID"]
    iterator = all_jets_one_df["time"].dt.year().unique()
    total = len(iterator)
    if "member" in all_jets_one_df.columns:
        members = all_jets_one_df["member"].unique()
        iterator = product(members, iterator)
        gb = ["member"].extend(gb)
        total = total * len(members)
    else:
        iterator = zip([None] * len(iterator), iterator)

    for member, year in tqdm(iterator, total=total):
        cross.append(
            track_jets_one_year(
                all_jets_one_df,
                year,
                member,
            )
        )
    cross = pl.concat(cross)
    return cross


def connected_from_cross(
    all_jets_one_df: DataFrame,
    cross: DataFrame | None = None,
    dist_thresh: float = 2e5,
    overlap_thresh: float = 0.5,
    dis_polar_thresh: float | None = 1.0,
) -> DataFrame:
    all_jets_one_df = all_jets_one_df.cast({"time": pl.Datetime("ms")})
    if cross is None:
        cross = track_jets(all_jets_one_df)
    cross = cross.filter(
        pl.col("dist") < dist_thresh,
        pl.col("lon_overlap") > overlap_thresh,
        pl.col("dis_polar") < dis_polar_thresh,
    )
    gb = ["time", "jet ID"]
    mem = []
    mem_k = []
    if "member" in all_jets_one_df.columns:
        gb_ = ["member"]
        gb_.extend(gb)
        gb = gb_
        mem = ["member"]
        mem_k = ["member_k"]
    summary = (
        all_jets_one_df.group_by("time", "jet ID", maintain_order=True)
        .agg()
        .with_row_index()
    )
    cross = (
        cross
        .with_columns(
            dt=((pl.col("time_right") - pl.col("time")) / (pl.col("time_right") - pl.col("time")).min()).cast(pl.Int32())
        )
        .with_columns(pers=persistence_expr() / pl.col("dt"))
        .group_by("time", "jet ID", maintain_order=True)
        .agg(
            pl.col("time_right").get(pl.col("pers").arg_max()),
            pl.col("jet ID_right").get(pl.col("pers").arg_max()),
            pl.col("dt").get(pl.col("pers").arg_max())
        )
        .join(cross, on=["time", "jet ID", "time_right", "jet ID_right"])
    )
    cross = (
        cross.join(
            summary[[*gb, "index"]],
            on=gb,
        )
        .rename({"index": "a"})
        .join(
            summary[[*gb, "index"]],
            left_on=[
                *mem,
                pl.col("time_right"),
                pl.col("jet ID_right"),
            ],
            right_on=[*mem, "time", "jet ID"],
            suffix="_k",
        )
        .drop(*mem_k)
        .rename({"index": "b"})
    )
    deltas = ["s", "theta"]
    if "is_polar" in all_jets_one_df.columns:
        deltas.append("is_polar")

    edges = cross[
        [
            "a",
            "b",
            "dist",
            "lon_overlap",
            *[f"d{col}" for col in deltas],
        ]
    ].to_dicts()
    edges = [
        (edge["a"], edge["b"], {k: v for k, v in edge.items() if k not in ["a", "b"]})
        for edge in edges
    ]
    G = rx.PyGraph(multigraph=False)
    G.add_nodes_from(summary.rows())
    G.add_edges_from(edges)
    conn_comp = rx.connected_components(G)
    summary_comp = []
    index_df = (
        pl.from_dicts(
            [
                {"spell": i, "index": list(comp), "len": len(comp)}
                for i, comp in enumerate(conn_comp)
            ]
        )
        .explode("index")
        .cast({"index": pl.UInt32()})
    )
    summary_comp = (
        index_df.join(summary, on="index")
        .join(
            cross.drop("time_right", "jet ID_right"),
            how="left",
            on=["time", "jet ID"],
        )
        .sort("spell", "time")
        .drop("index", "b")
    )
    return cross, summary_comp


def persistence_expr() -> Expr:
    return pl.col("lon_overlap") / pl.col("dist").replace(0, RADIUS * 0.1) * pl.col("time").diff().mode().first().cast(pl.Duration("ms")).cast(pl.Float64()) / 1000


def spells_from_cross(
    all_jets_one_df: DataFrame,
    cross: DataFrame,
    dist_thresh: float = 2e5,
    overlap_thresh: float = 0.5,
    dis_polar_thresh: float | None = 1.0,
    q_STJ: float = 0.99,
    q_EDJ: float = 0.95,
    season: Series | None = None,
    subtropical_cutoff: float = 0.4,
    polar_cutoff: float = 0.6,
):
    _, summary_comp = connected_from_cross(
        all_jets_one_df,
        cross,
        dist_thresh=dist_thresh,
        overlap_thresh=overlap_thresh,
        dis_polar_thresh=dis_polar_thresh,
    )
    if season is not None:
        summary_comp = summary_comp.filter(
            pl.col("time").is_in(pl.lit(season.implode().first(), pl.List(pl.Datetime('ms')))).over("spell")
        )
    spells = (
        summary_comp.filter(pl.col("len") > 2)
        .with_columns(pers=persistence_expr())
        .group_by("spell", maintain_order=True)
        .agg(
            pl.col("time"),
            pl.col("jet ID"),
            pl.col("len").first(),
            pl.col("lon_overlap"),
            pl.col("pers"),
            pl.col("dis_polar"),
            pl.col("is_polar"),
            mean_is_polar=pl.col("is_polar").mean(),
            pers_sum=pl.col("pers").sum(),
        )
    )

    spells_list = {}
    spells_list["STJ"] = (
        spells.filter(pl.col("mean_is_polar") < subtropical_cutoff)
        .filter(pl.col("pers_sum") > pl.col("pers_sum").quantile(q_STJ))
        .explode("time", "jet ID", "pers", "is_polar", "lon_overlap", "dis_polar")
        .with_columns(
            spell_of=pl.lit("STJ"),
            spell2=pl.col("spell").rle_id(),
            relative_index=pl.col("time").rle_id().over("spell"),
        )
        .drop("is_polar")
    )
    spells_list["EDJ"] = (
        spells.filter(pl.col("mean_is_polar") > polar_cutoff)
        .filter(pl.col("pers_sum") > pl.col("pers_sum").quantile(q_EDJ))
        .explode("time", "jet ID", "pers", "is_polar", "lon_overlap", "dis_polar")
        .with_columns(
            spell_of=pl.lit("EDJ"),
            spell2=pl.col("spell").rle_id(),
            relative_index=pl.col("time").rle_id().over("spell"),
        )
        .drop("is_polar")
    )
    return spells_list


def pers_from_cross_catd(cross: DataFrame) -> DataFrame:
    cross = (
        cross
        .filter(pl.col("jet ID") == pl.col("jet ID_right"))
        .with_columns(
            spell_of=pl.when(pl.col("jet ID") == 0)
            .then(pl.lit("STJ"))
            .otherwise(pl.lit("EDJ")),
            pers=persistence_expr()
        )
    )
    # cross = (
    #     cross
    #     .with_columns(
    #         dt=((pl.col("time_right") - pl.col("time")) / (pl.col("time_right") - pl.col("time")).min()).cast(pl.Int32())
    #     )
    #     .with_columns(pers2=pl.col("pers") / pl.col("dt"))
    #     .group_by("time", "jet ID", maintain_order=True)
    #     .agg(
    #         pl.col("time_right").get(pl.col("pers").arg_max()),
    #         pl.col("jet ID_right").get(pl.col("pers").arg_max()),
    #         pl.col("dt").get(pl.col("pers").arg_max())
    #     )
    #     .join(cross, on=["time", "jet ID", "time_right", "jet ID_right"])
    # )
    return cross


def spells_from_cross_catd_simple(
    cross: DataFrame,
    q_STJ: float = 0.99,
    q_EDJ: float = 0.95,
    season: Series | None = None,
    minlen: datetime.timedelta = datetime.timedelta(days=5)
) -> dict[str, DataFrame]:
    cross = pers_from_cross_catd(cross)
    
    if season is not None:
        cross = season.rename("time").to_frame().join(cross, on="time", how="left")
    
    cross = squarify(cross, ["time", "spell_of"])

    spells_list: dict[str, DataFrame] = {
        spell_of: get_spells(cross.filter(pl.col("spell_of") == spell_of), pl.col("pers") > pl.col("pers").quantile(q), minlen=minlen).with_columns(spell_of=pl.lit(spell_of))
        for spell_of, q in zip(["STJ", "EDJ"], [q_STJ, q_EDJ])
    }
    return spells_list


def spells_from_cross_catd(
    cross: DataFrame,
    base_q: float = 0.5,
    n_STJ: int = 30,
    n_EDJ: int = 30,
    season: Series | None = None,
    minlen: datetime.timedelta = datetime.timedelta(days=5),
    smooth: datetime.timedelta | None = None,
    fill_holes: datetime.timedelta | int = 0,
) -> dict[str, DataFrame]:
    cross = pers_from_cross_catd(cross)
    cross = squarify(cross, ["time", "spell_of"])
    cross = cross.with_columns(**{"jet ID": (pl.col("spell_of") == "EDJ").cast(pl.UInt32())})
    
    if smooth is not None:
        cross = cross.rolling(
            pl.col("time"),
            period=smooth,
            group_by=["jet ID", "spell_of"],
        ).agg(*[pl.col(col).mean() for col in ["lon_overlap", "ds", "dtheta", "dis_polar", "dist", "pers"]])
    
    if season is not None:
        cross = season.rename("time").to_frame().join(cross, on="time", how="left")
        
    spells_base = get_spells(cross, pl.col("pers") > pl.col("pers").quantile(base_q), group_by=["spell_of"], minlen=minlen, fill_holes=fill_holes)
    stats: DataFrame = spells_base.group_by(["spell_of", "spell"], maintain_order=True).agg(pl.col("len").first(), pl.col("pers").sum())

    spells_list: dict[str, DataFrame] = {
        spell_of: (
            stats
            .filter(pl.col("spell_of") == spell_of)
            .top_k(n, by="pers")
            .rename({"pers": "pers_sum"})
            .join(
                spells_base.filter(pl.col("spell_of") == spell_of).drop("len", "value"),
                on=["spell_of", "spell"]
            )
            .with_columns(spell=pl.col("spell").rle_id())
        )
        for spell_of, n in zip(["STJ", "EDJ"], [n_STJ, n_EDJ])
    }
    return spells_list


def jet_position_as_da(
    all_jets_one_df: DataFrame,
) -> xr.DataArray:
    """
    Constructs a `DataArray` of dimensions (*index_columns, lat, lon) from jets. The DataArray starts with NaNs everywhere. Then, for every jet point, the DataArray is filled with the jet point's `"is_polar"` value.
    """
    index_columns = get_index_columns(
        all_jets_one_df, ("member", "time", "cluster", "spell", "relative_index")
    )
    all_jets_pandas = (
        all_jets_one_df.group_by([*index_columns, "lon", "lat"], maintain_order=True)
        .agg(pl.col("is_polar").mean())
        .to_pandas()
    )
    da_jet_pos = xr.Dataset.from_dataframe(
        all_jets_pandas.set_index([*index_columns, "lat", "lon"])
    )["is_polar"]
    return da_jet_pos


def get_double_jet_index(props_as_df: DataFrame, jet_pos_da: xr.DataArray):
    """
    Adds a new columns to props_as_df; `"double_jet_index"`, by checking, for all longitudes, if there are at least two jet core points along the latitude, then averaging this over longitudes above 20° West.
    """
    overlap = (~np.isnan(jet_pos_da)).sum("lat") >= 2
    index_columns = get_index_columns(props_as_df, ["member", "time", "cluster"])
    dji = pl.concat(
        [
            props_as_df.select(index_columns).unique(maintain_order=True),
            Series(
                "double_jet_index",
                overlap.sel(lon=slice(-20, None, None)).mean("lon").values,
            ).to_frame(),
        ],
        how="horizontal",
    )
    props_as_df = props_as_df.join(dji, on=index_columns, how="left")
    return props_as_df


class JetFindingExperiment(object):
    """
    Convenience class that wraps basically all the functions in this module, applying it to the data held by its `DataHandler` and storing the results to avoid recomputing in the subfolder of its `DataHandler`.

    Attributes
    ----------
    data_handler : DataHandler
        Stores data and provides a subfolder in which to store results that is uniquely defined by the data (see `.data.DataHandler`)
    ds : xr.Dataset
        shortcut to `self.data_handler.da`
    path : Path
        shortcut to `self.data_handler.path`
    metadata : dict
        shortcut to `self.data_handler.metadata`
    time : xr.Dataset
        shortcut to `self.data_handler.get_sample_dims()["time"]`
    """

    def __init__(
        self,
        data_handler: DataHandler,
    ) -> None:
        self.ds = data_handler.da
        self.path = data_handler.path
        self.data_handler = data_handler
        self.metadata = self.data_handler.metadata
        self.time = data_handler.get_sample_dims()["time"]

    def find_low_wind(self):
        """
        Assuming the same file structure as me, finds the data corresponding to this Experiment's specs except for another `varname`: `"mid_wind"`.

        Returns
        -------
        ds: xr.Dataset
            A Dataset with the same specs as this object's `DataHandler`, but for another varname.
        """
        metadata = self.data_handler.metadata
        dataset = "CESM2" if "member" in metadata else "ERA5"
        dt = self.time[1] - self.time[0]
        resolution = "6H" if dt == np.timedelta64(6, "H") else "dailymean"
        ds_ = open_da(
            dataset,
            "plev",
            "mid_wind",
            resolution,
            metadata["period"],
            metadata["season"],
            *metadata["region"],
            metadata["levels"],
            None,
            None,
            None,
        )
        return ds_

    def find_jets(self, force: bool = False, **kwargs) -> DataFrame:
        """
        Wraps `find_all_jets(**kwargs)` and stores the output
        """
        ofile_ajdf = self.path.joinpath("all_jets_one_df.parquet")

        if ofile_ajdf.is_file() and not force:
            all_jets_one_df = pl.read_parquet(ofile_ajdf)
            return all_jets_one_df
        try:
            qs_path = self.path.joinpath("s_q.nc")
            qs = open_dataarray(qs_path).sel(quantile=0.7)
            kwargs["thresholds"] = qs.rename("s")
        except FileNotFoundError:
            pass
        else:
            print(f"Using thresholds at {qs_path}")

        all_jets_one_df = []
        if "member" in self.metadata or self.ds.nbytes > 8e10:
            several_years = 1
        else:
            several_years = 3
        iterator = iterate_over_year_maybe_member(
            da=self.ds, several_years=several_years
        )
        for indexer in iterator:
            ds_ = compute(self.ds.isel(**indexer), progress_flag=True)
            all_jets_one_df.append(find_all_jets(ds_, **kwargs))
        all_jets_one_df = pl.concat(all_jets_one_df)
        all_jets_one_df.write_parquet(ofile_ajdf)
        return all_jets_one_df

    def categorize_jets(
        self,
        low_wind: xr.Dataset | xr.DataArray | None = None,
        feature_names: tuple | None = None,
        force: int = 0,
        mode: (
            Literal["year"] | Literal["season"] | Literal["month"] | Literal["week"]
        ) = "week",
        n_components: int | Sequence = 2,
        n_init: int = 20,
        init_params: str = "random_from_data",
        v2: bool = True,
    ):
        """
        Makes sure the necessary columns are present in `jets`, then wraps `is_polar_gmix()` and and stores the output.

        Parameters
        ----------
        low_wind : xr.Dataset | xr.DataArray
            Wind at lower levels to compute the vertical wind shear
        force : int
            to recompute the categorization even if the jets already have a `is_polar` column. With `force > 1`, also re-interpolates `theta` and `ratio` on the jet points. By default 0
        mode : "all", "season", "month" or "week
            Trains one model if "all", otherwise train one independent model for each season, month or week of the year. By default "week".
        n_components : int
            Number of Gaussian components, by default 2
        n_init : int
            Number of repeated independent training with. Can massively increase run time. By default 20
        init_params : str, optional
            Type of init, by default "random_from_data"

        Returns
        -------
        DataFrame
            DataFrame of same length as the jets with the same index columns, the `feature_names` columns: "ratio" and "theta", and a new `is_polar` column, corresponding to the proability of each row to belong to the eddy-driven jet component.
        """
        jets = self.find_jets()
        ofile_ajdf = self.path.joinpath("all_jets_one_df.parquet")
        jets = categorize_jets(
            jets=jets,
            ds=self.ds,
            low_wind=low_wind,
            feature_names=feature_names,
            force=force,
            mode=mode,
            n_components=n_components,
            n_init=n_init,
            init_params=init_params,
            v2=v2,
        )
        jets.write_parquet(ofile_ajdf)
        return jets

    def compute_jet_props(
        self,
        force: bool = False,
    ) -> DataFrame:
        """
        Compute "raw" jet properties from the jets

        Parameters
        ----------
        force : bool
            To recompute the properties even if the file "props_as_df_raw.parquet" exists in the subfolder, by default False

        Returns
        -------
        DataFrame
            Jet properties for all jets
        """
        jet_props_incomplete_path = self.path.joinpath("props_as_df_raw.parquet")
        if jet_props_incomplete_path.is_file() and not force:
            return pl.read_parquet(jet_props_incomplete_path)
        all_jets_one_df = self.find_jets()
        props_as_df = compute_jet_props(all_jets_one_df)
        width = []
        if "s" not in self.ds:
            self.ds["s"] = np.sqrt(self.ds["u"] ** 2 + self.ds["v"] ** 2)
        da = self.ds["s"]
        indexer = iterate_over_year_maybe_member(all_jets_one_df, da)
        for idx1, idx2 in tqdm(list(indexer)):
            these_jets = all_jets_one_df.filter(*idx1)
            da_ = compute(da.sel(**idx2), progress_flag=False)
            width_ = compute_widths(these_jets, da_)
            width.append(width_)
        width = pl.concat(width)
        index_columns = get_index_columns(width)
        props_as_df = props_as_df.join(width, on=index_columns, how="inner").sort(
            index_columns
        )
        props_as_df.write_parquet(jet_props_incomplete_path)
        return props_as_df

    def props_as_df(self, categorize: bool = True, force: int = 0) -> DataFrame:
        """
        Compute full jet properties from the jets, with or without categorization

        Parameters
        ----------
        categorize: bool
            whether to return the uncatd or categorized jet properties. Will contain either the `jet ID` column and have a value for each jet at each timestep, or the `jet` category aggregating over the two or three categories of jets: EDJ, STJ and maybe hybrid.
        force : int
            To recompute the properties even if the file "props_as_df.parquet" exists in the subfolder, by default False. If set to *2* or more, also recomputes the `raw` props.s

        Returns
        -------
        DataFrame
            Jet properties for all jets
        """
        ofile_padu = self.path.joinpath("props_as_df_uncat.parquet")
        ofile_pad = self.path.joinpath("props_as_df.parquet")
        if ofile_padu.is_file() and not categorize and not force:
            return pl.read_parquet(ofile_padu)
        if ofile_pad.is_file() and categorize and not force:
            return pl.read_parquet(ofile_pad)
        if ofile_padu.is_file() and categorize and not force:
            props_as_df = average_jet_categories(pl.read_parquet(ofile_padu))
            props_as_df.write_parquet(ofile_pad)
            return props_as_df
        props_as_df = self.compute_jet_props(force=force > 1)
        if force == 1:
            jets = self.categorize_jets()
            index_columns = get_index_columns(jets)
            is_polar = (
                jets
                .group_by(index_columns, maintain_order=True)
                .agg(pl.col("is_polar").mean())
            )
            props_as_df = (
                props_as_df
                .drop("is_polar")
                .join(is_polar, on=index_columns)
            )
        props_as_df.write_parquet(ofile_padu)
        props_as_df_cat = average_jet_categories(props_as_df) 
        props_as_df_cat.write_parquet(ofile_pad)
        if categorize:
            return props_as_df_cat
        return props_as_df

    def track_jets(
        self,
        dist_thresh: float = 2,
        overlap_min_thresh: float = 0.5,
        overlap_max_thresh: float = 0.6,
        dis_polar_thresh: float | None = 1.0,
        n_next: int = 1,
        force: int = 0,
    ) -> DataFrame:
        """
        Wraps cross and summary

        Parameters
        ----------
        dist_thresh : float, optional
            _description_, by default 2
        overlap_min_thresh : float, optional
            _description_, by default 0.5
        overlap_max_thresh : float, optional
            _description_, by default 0.6
        dis_polar_thresh : float | None, optional
            _description_, by default 1.0
        force : int, optional
            _description_, by default 0

        Returns
        -------
        DataFrame
            _description_
        """
        cross_opath = self.path.joinpath("cross.parquet")
        summary_opath = self.path.joinpath("summary.parquet")
        all_jets_one_df = self.find_jets().cast({"time": pl.Datetime("ms")})
        if not cross_opath.is_file() or force > 1:
            cross = track_jets(all_jets_one_df, n_next=n_next)
            cross.write_parquet(cross_opath)
        else:
            cross = pl.read_parquet(cross_opath)
        if not summary_opath.is_file() or force:
            cross, summary = connected_from_cross(
                all_jets_one_df,
                cross,
                dist_thresh,
                overlap_min_thresh,
                overlap_max_thresh,
                dis_polar_thresh,
            )
            summary.write_parquet(summary_opath)
        else:
            summary = pl.read_parquet(summary_opath)
        return cross, summary

    def jet_position_as_da(self, force: bool = False):
        """
        Creates and saves a DataArray with the same dimensions as `self.ds`. Its values are NaN where no jet is present, and if a jet is present the value is this jet's local `is_polar` value.
        """
        ofile = self.path.joinpath("jet_pos.nc")
        if ofile.is_file() and not force:
            return open_dataarray(ofile)

        all_jets_one_df = self.find_jets()
        da_jet_pos = jet_position_as_da(all_jets_one_df)
        to_netcdf(da_jet_pos, ofile)
        return da_jet_pos

    def compute_extreme_clim(self, varname: str, subsample: int = 5):
        """
        Computes this object's data's extreme climatology.

        Parameters
        ----------
        varname : str
            If `self.ds` is a dataset, the variable to look at.
        subsample : int, optional
            Don't take every year in the data but only one every `subsample`, by default 5
        """
        da = self.ds[varname]
        time = Series("time", self.time)
        years = time.dt.year().to_numpy()
        mask = np.isin(years, np.unique(years)[::subsample])
        opath = self.path.joinpath(f"{varname}_q_clim.nc")
        compute_extreme_climatology(da.isel(time=mask), opath)
        quantiles_clim = open_dataarray(opath)
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
        to_netcdf(quantiles, self.path.joinpath(f"{varname}_q.nc"))
