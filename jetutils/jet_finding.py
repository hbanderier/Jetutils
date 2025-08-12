# coding: utf-8
"""
This probably too big module contains all the utilities relative to jet extraction from 2D fields, jet tracking, jet categorization and jet properties. All of the functions are wrapped by the convenience class `JetFindingExperiment`.
"""
import datetime
import warnings
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence, Tuple, Literal
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
from polars.exceptions import ColumnNotFoundError
import polars.selectors as cs
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
    N_WORKERS,
    RADIUS,
    compute,
    do_rle,
    xarray_to_polars,
    get_index_columns,
    extract_season_from_df,
    explode_rle,
    Timer,
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


def haversine(lon1: pl.Expr, lat1: pl.Expr, lon2: pl.Expr, lat2: pl.Expr) -> pl.Expr:
    """
    Generates a polars Expression to compute the haversine distance, in meters, between points defined with the columns (lon1, lat1) and the points defined with the columns (lon2, lat2).
    TODO: support other planets by passing the radius as an argument.
    """
    lon1 = lon1.radians()
    lat1 = lat1.radians()
    lon2 = lon2.radians()
    lat2 = lat2.radians()

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def haversine_from_dl(lat: pl.Expr, dlon: pl.Expr, dlat: pl.Expr) -> pl.Expr:
    """
    Alternative definition of the haversine distance, in meters, this time using the latitude of the first point, and the *differences* in longitues and latitudes between points.
    """
    lat = lat.radians()
    dlon = dlon.radians()
    dlat = dlat.radians()

    a = (dlat / 2.0).sin().pow(2) * (dlon / 2.0).cos().pow(2) + lat.cos().pow(2) * (
        dlon / 2.0
    ).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def jet_integral_haversine(
    lon: pl.Expr = pl.col("lon"),
    lat: pl.Expr = pl.col("lon"),
    s: pl.Expr | None = pl.col("s"),
    x_is_one: bool = False,
) -> pl.Expr:
    """
    Generates an expression to integrate the column `s` along a path on the sphere defined by `lon`and `lat`. Assumes we are on Earth since `haversine` uses the Earth's radius.
    """
    ds = haversine(
        lon,
        lat,
        lon.shift(),
        lat.shift(),
    )
    if x_is_one:
        return ds.sum()
    return 0.5 * (ds * (s + s.shift())).sum()


def has_periodic_x(df: pl.DataFrame | xr.Dataset | xr.DataArray) -> bool:
    """
    Checks if the `lon` column contains both sides of the +-180 line. Only makes sense if data went through `.data.standardize()`.

    Parameters
    ----------
    df : pl.DataFrame
        A DataFrame containing the `lon` column.

    Returns
    -------
    bool
    """
    if isinstance(df, pl.DataFrame):
        lon = df["lon"].unique().sort().to_numpy()
    else:
        lon = np.sort(df["lon"].values)
    dx = lon[1] - lon[0]
    return (-180 in lon) and ((180 - dx) in lon)


def coarsen_pl(df: pl.DataFrame, coarsen_map: Mapping[str, float]) -> pl.DataFrame:
    """
    Coarsening for polars DataFrame
    """
    index_columns = get_index_columns(df)
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


def round_polars(col: str, factor: int = 2) -> pl.Expr:
    """
    Generates an Expression that rounds the given column to a given base, one over the factor.
    """
    return (pl.col(col) * factor).round() / factor


def central_diff(by: str) -> pl.Expr:
    """
    Generates Expression to implement central differences for the given columns; and adds sensical numbers to the first and last element of the differentiation.
    """
    diff_2 = pl.col(by).diff(2, null_behavior="ignore").slice(2)
    diff_1 = pl.col(by).diff(1, null_behavior="ignore")
    return diff_1.gather(0).append(diff_2).append(diff_1.gather(-1))


def diff_maybe_periodic(by: str, periodic: bool = False) -> pl.Expr:
    """
    Wraps around `central_diff` to generate an Expression that implements central differences over a potentially periodic column like longitude.
    """
    if not periodic:
        return central_diff(by)
    max_by = pl.col(by).max() - pl.col(by).min()
    diff_by = central_diff(by).abs()
    return pl.when(diff_by > max_by / 2).then(max_by - diff_by).otherwise(diff_by)


def directional_diff(
    df: pl.DataFrame, col: str, by: str, periodic: bool = False
) -> pl.DataFrame:
    """
    Wraps around `central_diff` and `diff_maybe_periodic` to generate an Expression that differentiates a column `col` by another `by`. The output Expression will create a column with name `f"d{col}d{by}"`.
    """
    others = {
        "lon": "lat",
        "lat": "lon",
        "x": "y",
        "y": "x",
    }
    other = others[by]
    index_columns = get_index_columns(df)
    name = f"d{col}d{by}"
    diff_by = diff_maybe_periodic(by, periodic)
    agg = {name: central_diff(col) / diff_by, by: pl.col(by)}
    return (
        df.group_by([*index_columns, other], maintain_order=True)
        .agg(**agg)
        .explode(name, by)
    )


def preprocess_ds(
    ds: xr.Dataset, n_coarsen: int = 3, smooth_s: int | None = 13
) -> xr.Dataset:
    ds = coarsen_da(ds, n_coarsen=n_coarsen)

    if smooth_s is not None:
        smooth_map = ("win", smooth_s)
        smooth_map = {"lon": smooth_map, "lat": smooth_map}
        ds = ds.rename({var: f"{var}_orig" for var in ["u", "v", "s"]})
        for var in ["u", "v", "s"]:
            to_smooth = ds[f"{var}_orig"]
            ds[var] = to_smooth.copy(data=smooth(to_smooth, smooth_map=smooth_map))
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
            -ds["u"] * ds["s"].differentiate("y") + ds["v"] * ds["s"].differentiate("x")
        ) / ds["s"]
    # fft_smoothing = 1.0 if ds["sigma"].min() < -0.0001 else 0.8
    ds["sigma"] = smooth(ds["sigma"], smooth_map=smooth_map)
    return ds.reset_coords(["x", "y"], drop=True)


def nearest_mapping(df1: pl.DataFrame, df2: pl.DataFrame, col: str):
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
        pl.DataFrame(contours[i], schema={"lon": pl.Float32, "lat": pl.Float32})
        .with_columns(**indexer)
        .with_columns(cyclic=pl.lit(cyclic[i]))
        .with_columns(contour=pl.lit(i))
        for i in valid_index
    ]
    if len(contours) > 0:
        return pl.concat(contours)
    return None


def compute_contours(ds: xr.Dataset) -> pl.DataFrame:
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
    all_contours: pl.DataFrame, periodic: bool = False
) -> pl.DataFrame:
    """
    This function computes the alignment criterion for zero-sigma-contours. It is the scalar product betweeen the vector from a contour point to the next and the horizontal wind speed vector.
    """
    index_columns = get_index_columns(
        all_contours, ("member", "time", "cluster", "spell", "relative_index")
    )
    dlon = diff_maybe_periodic("lon", periodic)
    dlat = central_diff("lat")
    ds = (dlon.pow(2) + dlat.pow(2)).sqrt()
    align_x = pl.col("u") / pl.col("s") * dlon / ds
    align_y = pl.col("v") / pl.col("s") * dlat / ds
    alignment = align_x + align_y
    # alignment = (
    #     all_contours.group_by([*index_columns, "contour"], maintain_order=True)
    #     .agg(alignment=alignment)
    #     .explode("alignment")
    # )
    return all_contours.with_columns(
        alignment=alignment.over([*index_columns, "contour"])
    )


def find_all_jets(
    ds: xr.Dataset,
    thresholds: xr.DataArray | None = None,
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
    ds = preprocess_ds(ds, n_coarsen=3, smooth_s=5)
    dx = (ds["lon"][1] - ds["lon"][0]).item()
    dy = (ds["lat"][1] - ds["lat"][0]).item()
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
        drop = ["contour", "index", "cyclic", "condition", "int"]

    # contours
    all_contours = compute_contours(ds)

    diff_exp = pl.col("lon").diff().abs()
    diff_exp = pl.when(diff_exp > 180).then(360 - diff_exp).otherwise(diff_exp)
    diff_exp = (diff_exp.abs() + pl.col("lat").diff().abs()).fill_null(10.0)
    newindex = (pl.col("index").cast(pl.Int32()) - diff_exp.arg_max()) % (
        pl.col("index").max() + 1
    )

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
            int=jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s")).over(
                [*index_columns, "jet ID"]
            )
        )
        .filter(condition_expr2)
        .with_columns(pl.col("jet ID").rle_id().over([*index_columns]))
        .with_columns(newindex.over([*index_columns, "jet ID"]))
        .unique([*index_columns, "jet ID", "index"])
        .sort([*index_columns, "jet ID", "index"])
    )

    return jets


def to_expr(expr: pl.Expr | str):
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr


def weighted_mean_pl(col: pl.Expr | str, by: pl.Expr | str | None = None):
    col = to_expr(col)
    if by is None:
        return col.mean()
    by = to_expr(by)
    return ((col * by).sum() / by.sum()).alias(col.meta.output_name())


def circular_mean(col: pl.Expr | str, weights: pl.Expr | str | None = None):
    col = to_expr(col)
    col = col.radians()
    mean_sin = weighted_mean_pl(col.sin(), weights)
    mean_cos = weighted_mean_pl(col.cos(), weights)
    return pl.arctan2(mean_sin, mean_cos).degrees().alias(col.meta.output_name())


def compute_jet_props(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes all basic jet properties from a DataFrame containing many jets.
    """
    position_columns = [col for col in ["lat", "lev", "theta"] if col in df.columns]

    def dl(col):
        return pl.col(col).max() - pl.col(col).min()

    mean_lon = circular_mean("lon", "s")

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


def interp_from_other(jets: pl.DataFrame, da_df: pl.DataFrame, varname: str):
    """
    Bilinear interpolation. Values in `da_df[varname]` will be bilinearly interpolated to the jet points' `lon`-`lat` coordinates, resulting in a new column in `jets` with a name constructed as `f"{varname}_interp"`.
    """
    # assumes regular grid
    index_columns = get_index_columns(da_df)
    lon = da_df["lon"].unique().sort()
    lat = da_df["lat"].unique().sort()
    dlon = lon.diff().filter(lon.diff() > 0).min()
    dlat = lat.diff().filter(lat.diff() > 0).min()
    da_df = da_df.rename({"lon": "lon_", "lat": "lat_"})
    if varname in jets.columns:
        jets = jets.rename({varname: f"{varname}_core"})
        revert_rename = True
    else:
        revert_rename = False
    indices_right = lon.search_sorted(jets["normallon"], side="right").clip(
        1, len(lon) - 1
    )
    indices_above = lat.search_sorted(jets["normallat"], side="right").clip(
        1, len(lat) - 1
    )
    jets = jets.with_columns(
        left=lon[indices_right - 1],
        right=lon[indices_right],
        below=lat[indices_above - 1],
        above=lat[indices_above],
    )
    da_df = da_df[[*index_columns, "lon_", "lat_", varname]]
    for pair in [
        ["left", "below"],
        ["left", "above"],
        ["right", "below"],
        ["right", "above"],
    ]:
        jets = jets.join(
            da_df,
            left_on=[*index_columns, *pair],
            right_on=[*index_columns, "lon_", "lat_"],
        ).rename({varname: "".join(pair)})
    below = (pl.col("right") - pl.col("normallon")) * pl.col("leftbelow") / dlon + (
        pl.col("normallon") - pl.col("left")
    ) * pl.col("rightbelow") / dlon
    above = (pl.col("right") - pl.col("normallon")) * pl.col("leftabove") / dlon + (
        pl.col("normallon") - pl.col("left")
    ) * pl.col("rightabove") / dlon
    jets = jets.with_columns(r1=below, r2=above).drop(
        "leftbelow", "leftabove", "rightbelow", "rightabove", "left", "right"
    )
    center = (pl.col("above") - pl.col("normallat")) * pl.col("r1") / dlat + (
        pl.col("normallat") - pl.col("below")
    ) * pl.col("r2") / dlat
    jets = jets.with_columns(**{f"{varname}_interp": center}).drop(
        "below", "above", "r1", "r2"
    )
    if revert_rename:
        jets = jets.rename({f"{varname}_core": varname})
    return jets


def add_normals(
    jets: pl.DataFrame,
    half_length: float = 12.0,
    dn: float = 1.0,
    delete_middle: bool = False,
) -> pl.DataFrame:
    is_polar = ["is_polar"] if "is_polar" in jets.columns else []
    ns_df = np.arange(-half_length, half_length + dn, dn)
    if delete_middle:
        ns_df = np.delete(ns_df, int(half_length // dn))
    ns_df = pl.Series("n", ns_df).to_frame()

    # Expr angle
    if "u" in jets.columns and "v" in jets.columns:
        angle = pl.arctan2(pl.col("v"), pl.col("u")).interpolate("linear") + np.pi / 2
        wind_speed = ["u", "v", "s"]
    else:
        angle = (
            pl.arctan2(
                pl.col("lat").shift() - pl.col("lat"),
                pl.col("lon").shift() - pl.col("lon"),
            ).interpolate("linear")
            + np.pi / 2
        )
        angle = angle.fill_null(0)
        angle = (angle.shift(2, fill_value=0) + angle) / 2
        wind_speed = []

    # Expr normals
    normallon = pl.col("lon") + pl.col("angle").cos() * pl.col("n")
    normallon = (normallon + 180) % 360 - 180
    normallat = pl.col("lat") + pl.col("angle").sin() * pl.col("n")

    index_columns = get_index_columns(
        jets,
        (
            "member",
            "time",
            "cluster",
            "spell",
            "relative_index",
            "relative_time",
            "jet ID",
            "sample_index",
            "inside_index",
        ),
    )

    jets = jets[[*index_columns, "lon", "lat", *wind_speed, *is_polar]]

    jets = jets.with_columns(
        jets.group_by(index_columns, maintain_order=True)
        .agg(angle=angle, index=pl.int_range(pl.len()))
        .explode(["index", "angle"])
    )
    jets = jets.join(ns_df, how="cross")

    jets = jets.with_columns(normallon=normallon, normallat=normallat)
    jets = jets[
        [
            *index_columns,
            "index",
            "lon",
            "lat",
            *wind_speed,
            "n",
            "normallon",
            "normallat",
            *is_polar,
        ]
    ]
    return jets


def gather_normal_da_jets(
    jets: pl.DataFrame,
    da: xr.DataArray,
    half_length: float = 12.0,
    dn: float = 1.0,
    delete_middle: bool = False,
) -> pl.DataFrame:
    """
    Creates normal half-segments on either side of all jet core points, each of length `half_length` and with flat spacing `dn`. Then, interpolates the values of `da` onto each point of each normal segment.
    """
    index_columns = get_index_columns(
        jets,
        (
            "member",
            "time",
            "cluster",
            "spell",
            "relative_index",
            "relative_time",
            "jet ID",
            "sample_index",
            "inside_index",
        ),
    )
    jets = add_normals(jets, half_length, dn, delete_middle)
    dlon = (da.lon[1] - da.lon[0]).item()
    dlat = (da.lat[1] - da.lat[0]).item()
    lon = pl.Series("normallon_rounded", da.lon.values).to_frame()
    lat = pl.Series("normallat_rounded", da.lat.values).to_frame()
    jets = (
        jets.with_row_index("big_index")
        .sort("normallon")
        .join_asof(
            lon,
            left_on="normallon",
            right_on="normallon_rounded",
            strategy="nearest",
            tolerance=dlon,
        )
        .sort("normallat")
        .join_asof(
            lat,
            left_on="normallat",
            right_on="normallat_rounded",
            strategy="nearest",
            tolerance=dlat,
        )
        .sort("big_index")
        .drop("big_index")
        .drop_nulls(["normallon_rounded", "normallat_rounded"])
    )

    lonslice = jets["normallon_rounded"].unique()
    latslice = jets["normallat_rounded"].unique()
    da = da.sel(
        lon=lonslice,
        lat=latslice,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if "time" in da.dims:
            if da["time"].dtype == np.dtype("object"):
                da["time"] = da.indexes["time"].to_datetimeindex(time_unit="ms")
            da = da.sel(time=jets["time"].unique().sort().to_numpy())
    da_df = xarray_to_polars(da)
    if "time" in da_df.columns:
        da_df = da_df.cast({"time": jets.schema["time"]})

    varname = da.name
    jets = interp_from_other(jets, da_df, varname).sort([*index_columns, "index", "n"])
    jets = jets.with_columns(side=pl.col("n").sign().cast(pl.Int8))
    return jets


def compute_widths(jets: pl.DataFrame, da: xr.DataArray):
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


def expand_jets(jets: pl.DataFrame, max_t: float, dt: float) -> pl.DataFrame:
    """
    Expands the jets by appending segments before the start and after the end, following the tangent angle at the start and the end of the original jet, respectively.

    Parameters
    ----------
    jets : pl.DataFrame
        Jets to extend
    max_t : float
        Length of the added segments
    dt : float
        Spacing of the added segment

    Returns
    -------
    pl.DataFrame
        Jets DataFrame with all the index dimensions kept original, only lon and lat as additional columns (the rest is dropped), and longer jets with added segments.
    """
    index_columns = get_index_columns(jets, ["member", "time", "jet ID"])
    jets = jets.sort(*index_columns, "lon", "lat")
    angle = pl.arctan2(pl.col("v"), pl.col("u")).interpolate("linear")
    tangent_n = pl.linear_space(0, max_t, int(max_t / dt) + 1)

    tangent_lon_before_start = (
        pl.col("lon").first() - angle.head(5).mean().cos() * tangent_n.reverse()
    )
    tangent_lat_before_start = (
        pl.col("lat").first() - angle.head(5).mean().sin() * tangent_n.reverse()
    )

    tangent_lon_after_end = (
        pl.col("lon").last() + angle.tail(5).mean().cos() * tangent_n
    )
    tangent_lat_after_end = (
        pl.col("lat").last() + angle.tail(5).mean().sin() * tangent_n
    )

    bigger_lon = tangent_lon_before_start.append(pl.col("lon")).append(
        tangent_lon_after_end
    )
    bigger_lat = tangent_lat_before_start.append(pl.col("lat")).append(
        tangent_lat_after_end
    )

    jets = (
        jets.group_by(index_columns, maintain_order=True).agg(bigger_lon, bigger_lat)
    ).explode("lon", "lat")
    return jets


def join_wrapper(
    df: pl.DataFrame, da: xr.DataArray | xr.Dataset, suffix: str = "_right", **kwargs
):
    """
    Joins a DataFrame with a DataArray on the latter's dimensions. Explicitly iterates over years and members to limit memory usage

    Parameters
    ----------
    df : pl.DataFrame
        A DataFrame with columns also found in da
    da : xr.DataArray | xr.Dataset
        Xarray object whose values to join to the DataFrame
    suffix : str, optional
        join suffix, by default "_right"
    kwargs :
        keyword arguments passed to ``iterate_over_year_maybe_member``

    Returns
    -------
    pl.DataFrame
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


def map_maybe_parallel(
    iterator: Iterable,
    func: Callable,
    len_: int,
    processes: int = N_WORKERS,
    chunksize: int | None = None,
    progress: bool = True,
    pool_kwargs: dict | None = None,
    ctx=None,
) -> list:
    """
    Maps a function on the components of an Iterable. Can be parallel if processes is greater than one. In this case the other arguments are used to create a `multiprocessing.Pool`. In most cases, I recommend using `ctx = get_context("spawn")` instead of the default (on linux) `fork`.

    Parameters
    ----------
    iterator : Iterable
        Data
    func : Callable
        Function to apply to each element of `iterator`
    len_ : int
        len of the `iterator`, so we can display a progress bar.
    processes : int, optional
        Number of parallel processes, will not create a `Pool` if 1, by default N_WORKERS
    chunksize : int, optional
        How many elements to send to a worker at once, by default 100
    progress : bool, optional
        Show a progress bar using `tqdm`, by default True
    pool_kwargs : dict | None, optional
        Keyword arguments passed to `multiprocessing.Pool`, by default None
    ctx : optional
        Multiporcessing context, created using `multiprocessing.get_context()`, by default None, will be `spawn` on windowd and mac, and `fork` on linux at time of writing, but it should change in python 3.15.

    Returns
    -------
    list
        result of the map coerced into a list.
    """
    processes = min(processes, len_)
    if processes == 1 and progress:
        return list(tqdm(map(func, iterator), total=len_))
    if processes == 1:
        return list(map(func, iterator))
    if pool_kwargs is None:
        pool_kwargs = {}
    if chunksize is None:
        chunksize = min(int(len_ // processes), 200)
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
    """
    I' not sure about this one anymore. I don't think I use it ?

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    das : Sequence | None, optional
        _description_, by default None
    others : Sequence | None, optional
        _description_, by default None
    potentials : Tuple, optional
        _description_, by default ("member", "time", "cluster")

    Returns
    -------
    Tuple
        _description_
    """
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

    def _extract_da(da, index):
        return compute(
            da.sel(
                {dim: values for dim, values in zip(iter_dims, index) if dim in da.dims}
            )
        )

    iterator = (
        (jets, *[_extract_da(da, index) for da in das], *others) for index, jets in gb
    )
    return len_, iterator


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
    df: pl.DataFrame,
    feature_names: Sequence = None,
    season: list | str | tuple | int | None = None,
) -> pl.DataFrame:
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
    X : pl.DataFrame
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
    X : pl.DataFrame
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
    if X.columns[1] == "theta":
        order = np.argsort(model.means_[:, 1])
    else:
        order = np.argsort(model.means_[:, 1])[::-1]
    otherscores = np.sum([scores[k] for k in order[:-1]], axis=0)
    return 1 / (1 + otherscores / scores[order[-1]])


def is_polar_gmix(
    df: pl.DataFrame,
    feature_names: list,
    mode: (
        Literal["all"] | Literal["season"] | Literal["month"] | Literal["week"]
    ) = "week",
    n_components: int | Sequence = 2,
    n_init: int = 20,
    init_params: str = "random_from_data",
    v2: bool = True,
) -> pl.DataFrame:
    """
    Trains one or several Gaussian Mixture model independently, depending on the `mode` argument.

    Parameters
    ----------
    df : pl.DataFrame
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
    pl.DataFrame
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
    jets: pl.DataFrame,
    feature: str,
    ds: xr.Dataset | None = None,
    ds_low: xr.Dataset | None = None,
    ofile_ajdf: Path | None = None,
    force: bool = False,
):
    if (feature in jets.columns) and not force:
        return jets
    if feature in jets.columns:
        jets = jets.drop(feature)
    if feature == "ratio":
        da = ds_low["s"]
        if "s_low" in jets.columns:
            jets.drop("s_low")
        jets = join_wrapper(jets, da, suffix="_low")
        jets = jets.with_columns(ratio=pl.col("s_low") / pl.col("s"))
    else:
        da = ds[feature]
        jets = join_wrapper(jets, da)
    if ofile_ajdf is not None:
        jets.write_parquet(ofile_ajdf)
    return jets


def categorize_jets(
    jets: pl.DataFrame,
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


def average_jet_categories_v2(props_as_df: pl.DataFrame):
    """
    For every timestep, member and / or cluster (whichever applicable), aggregates each jet property (with a different rule for each property but usually a mean) into a single number for each category: subtropical, eddy driven jet and potentially hybrid, summarizing this property fo all the jets in this snapshot that fit this category, based on their mean `is_polar` value and a threshold given by `polar_cutoff`.

    E.g. on the 1st of January 1999, there are two jets with `is_polar < polar_cutoff` and one with `is_polar > polar_cutoff`. We pass `allow_hybrid=False` to the function. In the output, for the row corresponding to this date and `jet=STJ`, the value for the `"mean_lat"` column will be the mean of the `"mean_lat"` values of two jets that had `is_polar < polar_cutoff`.

    Parameters
    ----------
    props_as_df : pl.DataFrame
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
    props_as_df: pl.DataFrame,
    polar_cutoff: float | None = None,
    allow_hybrid: bool = False,
):
    """
    For every timestep, member and / or cluster (whichever applicable), aggregates each jet property (with a different rule for each property but usually a mean) into a single number for each category: subtropical, eddy driven jet and potentially hybrid, summarizing this property fo all the jets in this snapshot that fit this category, based on their mean `is_polar` value and a threshold given by `polar_cutoff`.

    E.g. on the 1st of January 1999, there are two jets with `is_polar < polar_cutoff` and one with `is_polar > polar_cutoff`. We pass `allow_hybrid=False` to the function. In the output, for the row corresponding to this date and `jet=STJ`, the value for the `"mean_lat"` column will be the mean of the `"mean_lat"` values of two jets that had `is_polar < polar_cutoff`.

    Parameters
    ----------
    props_as_df : pl.DataFrame
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


def track_jets_one_year(
    jets: pl.DataFrame, year: int, member: str | None = None, n_next: int = 1
):
    if "len_right" in jets.columns:
        jets = jets.drop("len_right")
    deltas = ["u", "v", "s", "theta"]
    distargmin = pl.col("dist").arg_min()
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
            .bottom_k(n_next)
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
    jets_next = []
    dt = jets_current["time"].unique().bottom_k(2).sort()
    dt = dt[1] - dt[0]
    for n_nex in range(1, n_next + 1):
        jets_next.append(
            jets_current.with_columns(time_shifted=pl.col("time") - n_nex * dt)
        )
    jets_next = pl.concat(jets_next)
    jets_current = jets_current.filter(pl.col("time").dt.year() == year)
    cross = (
        jets_current.join(
            jets_next, left_on="time", right_on="time_shifted", how="left"
        )
        .with_columns(
            **{
                f"d{col}": (pl.col(col) - pl.col(f"{col}_right")).abs()
                for col in deltas
            },
            dist=haversine(
                pl.col("lon"), pl.col("lat"), pl.col("lon_right"), pl.col("lat_right")
            ),
        )
        .with_columns(overlap=pl.col("dist") < 5e5)
    )
    cross = (
        cross.group_by(typical_group_by)
        .agg(
            pl.col("len").first(),
            pl.col("len_right").first(),
            pl.col("index").filter("overlap"),
            pl.col("index_right").filter("overlap"),
            pl.col("dist").filter("overlap"),
            overlap_forward=pl.col("index").filter("overlap").n_unique()
            / pl.col("len").first(),
            overlap_backward=pl.col("index_right").filter("overlap").n_unique()
            / pl.col("len_right").first(),
            **{
                f"d{col}": (pl.col(col) - pl.col(f"{col}_right"))
                .abs()
                .filter("overlap")
                for col in deltas
            },
        )
        .with_columns(
            overlap_min=pl.min_horizontal("overlap_forward", "overlap_backward"),
            overlap_max=pl.max_horizontal("overlap_forward", "overlap_backward"),
        )
    )

    forward = (
        cross.explode("index", "index_right", "dist", *deltas2)
        .group_by(*typical_group_by, "index")
        .agg(
            *[
                pl.col(col).get(distargmin)
                for col in [*[f"d{d}" for d in deltas], "dist"]
            ]
        )
        .group_by(typical_group_by)
        .agg(pl.col(col).mean().alias(f"{col}_forward") for col in [*deltas2, "dist"])
    )
    backward = (
        cross.explode("index", "index_right", "dist", *deltas2)
        .group_by(*typical_group_by, "index_right")
        .agg(*[pl.col(col).get(distargmin) for col in [*deltas2, "dist"]])
        .group_by(typical_group_by)
        .agg(pl.col(col).mean().alias(f"{col}_backward") for col in [*deltas2, "dist"])
    )
    cross = (
        cross.drop("index", "index_right", "dist", *deltas2)
        .join(forward, on=typical_group_by)
        .join(backward, on=typical_group_by)
        .drop_nulls(["du_forward"])
        .sort(*typical_group_by)
        .with_columns(
            pl.mean_horizontal(f"{col}_backward", f"{col}_forward").alias(col)
            for col in [*deltas2, "dist"]
        )
        .drop(
            *[f"{col}_backward" for col in [*deltas2, "dist"]],
            *[f"{col}_forward" for col in [*deltas2, "dist"]],
        )
    )
    return cross


def track_jets(all_jets_one_df: pl.DataFrame, n_next: int = 1):
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
                n_next=n_next,
            )
        )
    cross = pl.concat(cross)
    return cross


def connected_from_cross(
    all_jets_one_df: pl.DataFrame,
    cross: pl.DataFrame | None = None,
    dist_thresh: float = 2e5,
    overlap_min_thresh: float = 0.5,
    overlap_max_thresh: float = 0.6,
    dis_polar_thresh: float | None = 1.0,
) -> pl.DataFrame:
    if cross is None:
        cross = track_jets(all_jets_one_df)
    cross = cross.filter(
        pl.col("dist") < dist_thresh,
        pl.col("overlap_min") > overlap_min_thresh,
        pl.col("overlap_max") > overlap_max_thresh,
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
        .agg(pl.col("is_polar").mean())
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
    deltas = ["u", "v", "s", "theta"]
    if "is_polar" in all_jets_one_df.columns:
        deltas.append("is_polar")

    edges = cross[
        [
            "a",
            "b",
            "dist",
            "overlap_min",
            "overlap_max",
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
            cross.drop("time_right", "jet ID_right", "len", "len_right"),
            how="left",
            on=["time", "jet ID"],
        )
        .sort("spell", "time")
        .drop("index", "b")
    )
    return cross, summary_comp


def persistence_expr():
    return pl.col("overlap_min") * RADIUS / pl.col("dist").replace(0, RADIUS * 0.1)


def spells_from_cross(
    all_jets_one_df: pl.DataFrame,
    cross: pl.DataFrame,
    dist_thresh: float = 2e5,
    overlap_min_thresh: float = 0.5,
    overlap_max_thresh: float = 0.6,
    dis_polar_thresh: float | None = 1.0,
    q_STJ: float = 0.99,
    q_EDJ: float = 0.95,
    season: pl.Series | None = None,
    subtropical_cutoff: float = 0.4,
    polar_cutoff: float = 0.6,
):
    _, summary_comp = connected_from_cross(
        all_jets_one_df,
        cross,
        dist_thresh=dist_thresh,
        overlap_min_thresh=overlap_min_thresh,
        overlap_max_thresh=overlap_max_thresh,
        dis_polar_thresh=dis_polar_thresh,
    )
    if season is not None:
        summary_comp = summary_comp.filter(
            (pl.col("time").is_in(season.implode()).mean() > 0.8).over("spell")
        )
    summer_spells = (
        summary_comp.filter(pl.col("len") > 2)
        .with_columns(pers=persistence_expr())
        .group_by("spell", maintain_order=True)
        .agg(
            pl.col("time"),
            pl.col("jet ID"),
            pl.col("len").first(),
            pl.col("overlap_min"),
            pl.col("overlap_max"),
            pl.col("pers"),
            pl.col("dis_polar"),
            pl.col("is_polar"),
            mean_is_polar=pl.col("is_polar").mean(),
            pers_sum=pl.col("pers").sum(),
        )
    )

    spells_list = {}
    spells_list["STJ"] = (
        summer_spells.filter(pl.col("mean_is_polar") < subtropical_cutoff)
        .filter(pl.col("pers_sum") > pl.col("pers_sum").quantile(q_STJ))
        .explode("time", "jet ID", "pers", "is_polar", "overlap_min", "overlap_max", "dis_polar")
        .with_columns(
            spell_of=pl.lit("STJ"),
            spell2=pl.col("spell").rle_id(),
            relative_index=pl.col("time").rle_id().over("spell"),
        )
        .drop("is_polar")
    )
    spells_list["EDJ"] = (
        summer_spells.filter(pl.col("mean_is_polar") > polar_cutoff)
        .filter(pl.col("pers_sum") > pl.col("pers_sum").quantile(q_EDJ))
        .explode("time", "jet ID", "pers", "is_polar", "overlap_min", "overlap_max", "dis_polar")
        .with_columns(
            spell_of=pl.lit("EDJ"),
            spell2=pl.col("spell").rle_id(),
            relative_index=pl.col("time").rle_id().over("spell"),
        )
        .drop("is_polar")
    )
    return spells_list


def pers_from_cross_catd(cross: pl.DataFrame, season: pl.Series | None = None) -> pl.DataFrame:
    if season is not None:
        cross = season.rename("time").to_frame().join(cross, on="time")
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
    cross = (
        cross
        .with_columns(
            dt=((pl.col("time_right") - pl.col("time")) / (pl.col("time_right") - pl.col("time")).min()).cast(pl.Int32())
        )
        .with_columns(pers2=pl.col("pers") / pl.col("dt"))
        .group_by("time", "jet ID", maintain_order=True)
        .agg(
            pl.col("time_right").get(pl.col("pers").arg_max()),
            pl.col("jet ID_right").get(pl.col("pers").arg_max()),
            pl.col("dt").get(pl.col("pers").arg_max())
        )
        .join(cross, on=["time", "jet ID", "time_right", "jet ID_right"])
    )
    pers = cross.rolling("time", period="3d", group_by="spell_of").agg(
        pl.col("pers").mean()
    )
    return cross, pers


def spells_from_cross_catd(
    # all_jets_one_df: pl.DataFrame,
    cross: pl.DataFrame,
    q_STJ: float = 0.99,
    q_EDJ: float = 0.95,
    season: pl.Series | None = None,
    minlen: datetime.timedelta = datetime.timedelta(days=5)
):
    cross, pers = pers_from_cross_catd(cross, season)
    exprs = {
        spell_of: pl.col("pers") > pl.col("pers").quantile(q) for spell_of, q in zip(["STJ", "EDJ"], [q_STJ, q_EDJ])
    }
    spells_list = {
        spell_of: (
            get_spells(pers.filter(pl.col("spell_of") == spell_of), expr, minlen=minlen)
            .with_columns(spell_of=pl.lit(spell_of))
            .sort("spell")
            .join(cross.drop("len", "len_right"), on=["time", "spell_of"])
        )
        for spell_of, expr in exprs.items()
    }
    return spells_list


def jet_position_as_da(
    all_jets_one_df: pl.DataFrame,
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


def get_double_jet_index(props_as_df: pl.DataFrame, jet_pos_da: xr.DataArray):
    """
    Adds a new columns to props_as_df; `"double_jet_index"`, by checking, for all longitudes, if there are at least two jet core points along the latitude, then averaging this over longitudes above 20° West.
    """
    overlap = (~np.isnan(jet_pos_da)).sum("lat") >= 2
    index_columns = get_index_columns(props_as_df, ["member", "time", "cluster"])
    dji = pl.concat(
        [
            props_as_df.select(index_columns).unique(maintain_order=True),
            pl.Series(
                "double_jet_index",
                overlap.sel(lon=slice(-20, None, None)).mean("lon").values,
            ).to_frame(),
        ],
        how="horizontal",
    )
    props_as_df = props_as_df.join(dji, on=index_columns, how="left")
    return props_as_df


def iterate_over_year_maybe_member(
    df: pl.DataFrame | None = None,
    da: xr.DataArray | xr.Dataset | None = None,
    several_years: int = 1,
    several_members: int = 1,
):
    """
    Constructs iterators over time and member, for up to a polars DataFrame and a xarray DataArray that have the same indices.
    """
    if df is None and da is None:
        return 0
    if da is None and df is not None:
        years = df["time"].dt.year().unique(maintain_order=True).to_numpy()
        try:
            year_lists = np.array_split(years, len(years) // several_years)
        except ValueError:
            year_lists = [years]
        indexer_polars = (
            pl.col("time").dt.year().is_in(year_list) for year_list in year_lists
        )
        if "member" not in df.columns:
            return zip(indexer_polars)
        members = df["member"].unique(maintain_order=True).to_numpy()
        member_lists = np.array_split(members, len(members) // several_members)
        indexer_polars_2 = (
            pl.col("member").is_in(member_list) for member_list in member_lists
        )
        indexer_polars = product(indexer_polars, indexer_polars_2)
        return indexer_polars
    elif da is not None and df is None:
        years = np.unique(da["time"].dt.year.values)
        try:
            year_lists = np.array_split(years, len(years) // several_years)
        except ValueError:
            # ValueError when too few years. Then a one list should suffice
            year_lists = [years]
        indexer_xarray = (
            {"time": np.isin(da["time"].dt.year.values, year_list)}
            for year_list in year_lists
        )
        if "member" not in da.dims:
            return indexer_xarray
        members = np.unique(da["member"].values)
        member_lists = np.array_split(members, len(members) // several_members)
        indexer_xarray_2 = (
            {"member": np.isin(da["member"].values, member_list)}
            for member_list in member_lists
        )
        indexer_xarray = product(indexer_xarray, indexer_xarray_2)
        indexer_xarray = (indexer[0] | indexer[1] for indexer in indexer_xarray)
        return indexer_xarray
    years = df["time"].dt.year().unique(maintain_order=True).to_numpy()
    year_lists = np.array_split(years, len(years) // several_years)
    indexer_polars = (
        pl.col("time").dt.year().is_in(year_list) for year_list in year_lists
    )
    indexer_xarray = (
        {"time": np.isin(da["time"].dt.year.values, year_list)}
        for year_list in year_lists
    )
    if "member" not in df.columns:
        return zip(zip(indexer_polars), indexer_xarray)
    """
        weird inner zip: don't worry lol. I want to always be able call::
        
            for idx in indexer: df.filter(*idx)
            
        so I need to put it in zip by itself if it's not out of product, so it's always a tuple.
    """
    members = df["member"].unique(maintain_order=True).to_numpy()
    member_lists = np.array_split(members, len(members) // several_members)
    indexer_polars_2 = (
        pl.col("member").is_in(member_list) for member_list in member_lists
    )
    indexer_polars = product(indexer_polars, indexer_polars_2)
    indexer_xarray_2 = (
        {"member": np.isin(da["member"].values, member_list)}
        for member_list in member_lists
    )
    indexer_xarray = product(indexer_xarray, indexer_xarray_2)
    indexer_xarray = (indexer[0] | indexer[1] for indexer in indexer_xarray)
    return zip(indexer_polars, indexer_xarray)


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

    def find_jets(self, force: bool = False, **kwargs) -> pl.DataFrame:
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
        pl.DataFrame
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
    ) -> pl.DataFrame:
        """
        Compute "raw" jet properties from the jets

        Parameters
        ----------
        force : bool
            To recompute the properties even if the file "props_as_df_raw.parquet" exists in the subfolder, by default False

        Returns
        -------
        pl.DataFrame
            Jet properties for all jets
        """
        jet_props_incomplete_path = self.path.joinpath("props_as_df_raw.parquet")
        if jet_props_incomplete_path.is_file() and not force:
            return pl.read_parquet(jet_props_incomplete_path)
        all_jets_one_df = self.find_jets()
        props_as_df = compute_jet_props(all_jets_one_df)
        width = []
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

    def props_as_df(self, categorize: bool = True, force: int = 0) -> pl.DataFrame:
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
        pl.DataFrame
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
    ) -> pl.DataFrame:
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
        pl.DataFrame
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
        time = pl.Series("time", self.time)
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
