from jetutils.anyspell import extend_spells, extend_spells_jet_ID
import datetime
from scipy.stats import ttest_ind_from_stats
from jetutils.stats import create_bootstrapped_times, bs_times_with_jet_ID
from pathlib import Path
import warnings
from itertools import product
from typing import Literal, Callable
from multiprocessing import get_context
from functools import reduce

import numpy as np
import polars as pl
import polars.selectors as cs
import polars_st as st
import xarray as xr
from contourpy import contour_generator
from polars import DataFrame, Expr, Series
from tqdm import tqdm

from .definitions import (
    JJADOYS,
    RADIUS,
    get_index_columns,
    iterate_over_year_maybe_member,
    map_maybe_parallel,
    to_expr,
    xarray_to_polars,
    compute,
    circular_mean,
    weighted_mean_pl,
    polars_to_xarray,
    FACTORS_UNITS,
)
from .data import (
    standardize_polars_dtypes,
    compute_anomalies_pl,
    average_jet_categories,
)


def euclidean_geographic(
    lon1: Expr | str, lat1: Expr | str, lon2: Expr | str, lat2: Expr | str
) -> Expr:
    """
    Slightly modified eucliean distance as a polars expression in longitude in latitude, with periodic longitudes in degrees.

    Parameters
    ----------
    lon1 : Expr | str
        _description_
    lat1 : Expr | str
        _description_
    lon2 : Expr | str
        _description_
    lat2 : Expr | str
        _description_

    Returns
    -------
    Expr
        _description_
    """
    lon1 = to_expr(lon1)
    lat1 = to_expr(lat1)
    lon2 = to_expr(lon2)
    lat2 = to_expr(lat2)

    dlon = (lon2 - lon1).abs()
    dlon = pl.when(dlon > 180).then(360 - dlon).otherwise(dlon)
    dlat = lat2 - lat1

    return (dlon.pow(2) + dlat.pow(2)).sqrt()


def haversine(
    lon1: Expr | str, lat1: Expr | str, lon2: Expr | str, lat2: Expr | str
) -> Expr:
    """
    Generates a polars Expression to compute the haversine distance, in meters, between points defined with the columns (lon1, lat1) and the points defined with the columns (lon2, lat2).
    TODO: support other planets by passing the radius as an argument.

    Parameters
    ----------
    lon1 : Expr | str
        first longitude column
    lat1 : Expr | str
        first latitude column
    lon2 : Expr | str
        second longitude column
    lat2 : Expr | str
        second latitude column

    Returns
    -------
    Expr
        Distance expression
    """
    lon1 = to_expr(lon1).radians()
    lat1 = to_expr(lat1).radians()
    lon2 = to_expr(lon2).radians()
    lat2 = to_expr(lat2).radians()

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def haversine_from_dl(lat: Expr | str, dlon: Expr | str, dlat: Expr | str) -> Expr:
    """
    Alternative definition of the haversine distance, in meters, this time using the latitude of the first point, and the *differences* in longitues and latitudes between points.

    Parameters
    ----------
    lat : Expr | str
        First or second latitude column
    dlon : Expr | str
        Column representing differences of longitudes
    dlat : Expr | str
        Column representing differences of latitudes

    Returns
    -------
    Expr
        Distance
    """
    lat = to_expr(lat).radians()
    dlon = to_expr(dlon).radians()
    dlat = to_expr(dlat).radians()

    a = (dlat / 2.0).sin().pow(2) * (dlon / 2.0).cos().pow(2) + lat.cos().pow(2) * (
        dlon / 2.0
    ).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def jet_integral_haversine(
    lon: Expr | str = pl.col("lon"),
    lat: Expr | str = pl.col("lon"),
    s: Expr | str | None = pl.col("s"),
    x_is_one: bool = False,
) -> Expr:
    """
    Generates an `Expr` to integrate the column `s` along a path on the sphere defined by `lon`and `lat`. Assumes we are on Earth since `haversine` uses the Earth's radius.

    Parameters
    ----------
    lon : Expr, optional
        Longitude column, by default pl.col("lon")
    lat : Expr, optional
        Latitude column, by default pl.col("lon")
    s : Expr | None, optional
        Wind speed magnitude column, by default pl.col("s")
    x_is_one : bool, optional
        Ignores `s` and integrates ones instead, to compute a length, by default False

    Returns
    -------
    Expr
        Integral, will reduce to a number.
    """
    ds: Expr = haversine(
        lon,
        lat,
        to_expr(lon).shift(),
        to_expr(lat).shift(),
    )
    if x_is_one or s is None:
        return ds.sum()
    s = to_expr(s)
    return 0.5 * (ds * (s + s.shift())).sum()


def central_diff(by: str | pl.Expr) -> pl.Expr:
    """
    Generates Expression to implement central differences for the given columns; and adds sensical numbers to the first and last element of the differentiation.

    Parameters
    ----------
    by : str | pl.Expr
        Column to differentiate

    Returns
    -------
    pl.Expr
        Result
    """
    by = to_expr(by)
    diff_2 = by.diff(2, null_behavior="ignore").slice(2) / 2
    diff_1 = by.diff(1, null_behavior="ignore")
    return diff_1.gather(1).append(diff_2).append(diff_1.gather(-1))


def diff_maybe_periodic(by: str, periodic: bool = False) -> pl.Expr:
    """
    Wraps around `central_diff` to generate an Expression that implements central differences over a potentially periodic column like longitude.


    Parameters
    ----------
    by : str
        Column to differentiate
    periodic : bool, optional
        Is this column periodic, by default False

    Returns
    -------
    pl.Expr
        Result
    """
    if not periodic:
        return central_diff(by)
    max_by = pl.col(by).max() - pl.col(by).min()
    diff_by = central_diff(by).abs()
    return pl.when(diff_by > max_by / 2).then(max_by - diff_by).otherwise(diff_by)


def directional_diff(
    df: DataFrame, col: str, by: str, periodic: bool = False
) -> DataFrame:
    """
    Wraps around `central_diff` and `diff_maybe_periodic` to generate an Expression that differentiates a column `col` by another `by` and executes it. The output Expression will create a column with name `f"d{col}d{by}"`.

    Parameters
    ----------
    df : DataFrame
        Data source
    col : str
        what to derive
    by : str
        by what to derive
    periodic : bool, optional
        is the `"by"` column periodic, by default False

    Returns
    -------
    DataFrame
        Data augmented with one extra column.
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


def difflon() -> Expr:
    """
    Periodic difference in longitude in degrees

    Returns
    -------
    Expr
    """
    expr = pl.col("lon").diff().abs()
    expr = pl.when(expr > 180).then(360 - expr).otherwise(expr)
    return expr


def signed_difflon() -> Expr:
    """
    Signed periodic difference

    Returns
    -------
    Expr
    """
    expr = pl.col("lon").diff()
    expr = (
        pl.when(expr.abs() > 180).then((360 - expr.abs()) * expr.sign()).otherwise(expr)
    )
    return expr


def diff_exp() -> Expr:
    """
    Periodic L^1 distance for lon and lat

    Returns
    -------
    Expr
    """
    expr = difflon()
    return (expr.abs() + pl.col("lat").diff().abs()).fill_null(10.0)


def newindex() -> Expr:
    """
    Indexes a string of lon-lat points, starting from the first point after the largest jump in `diff_exp`.

    Returns
    -------
    Expr
    """
    return (pl.col("index").cast(pl.Int32()) - diff_exp().arg_max()) % pl.col(
        "index"
    ).max()


def sort_by_index(
    df: pl.DataFrame, index_columns: list[str], other: str
) -> pl.DataFrame:
    """
    Sorts by index_columns and other, plus some random order for what's left, indexed by "index"

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    index_columns : list[str]
        _description_
    other : str
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    return (
        df.with_columns(index=pl.int_range(0, pl.len()).over([*index_columns, other]))
        .unique([*index_columns, other, "index"])
        .sort([*index_columns, other, "index"])
    )


def sort_by_difflon(
    df: pl.DataFrame, index_columns: list[str], other: str
) -> pl.DataFrame:
    """
    Sorts purely by increasing longitude after the jump

    Parameters
    ----------
    df : pl.DataFrame
        _description_
    index_columns : _type_
        _description_
    other : _type_
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    return (
        df.with_columns(diff_exp().over([*index_columns, other]))
        .unique([*index_columns, other, "index"])
        .sort([*index_columns, other, "index"])
    )


def sort_by_newindex(df: pl.DataFrame, index_columns: list[str], other: str):
    return (
        df.with_columns(index=pl.int_range(0, pl.len()).over([*index_columns, other]))
        .with_columns(newindex().over([*index_columns, other]))
        .unique([*index_columns, other, "index"])
        .sort([*index_columns, other, "index"])
    )


def sort_by_index_then_difflon(df, index_columns, other):
    interm = sort_by_index(df, index_columns, other)
    return sort_by_difflon(interm, index_columns, other)


def sort_by_index_then_newindex(df, index_columns, other):
    interm = sort_by_index(df, index_columns, other)
    return sort_by_newindex(interm, index_columns, other)


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


def inner_detect_contours(args):
    """
    Worker function to compute the zero-sigma-contours in a parallel context
    """
    da, levels, spatial_dims, do_round = args
    x = da[spatial_dims[0]].values
    y = da[spatial_dims[1]].values
    z = da.values
    l1 = contour_generator(
        x, y, z, line_type="SeparateCode", quad_as_tri=False
    ).multi_lines(levels)
    if len(l1[0][0]) == 0:
        return [], [], [], []
    f = round_contour if do_round else lambda x, y, z: x
    to_ret = [
        (i, level, f(contour, x, y), 79 in types_)
        for level, (contours, types) in zip(levels, l1)
        for contour, types_, i in zip(contours, types, range(10000000))
    ]
    return tuple(zip(*to_ret))


def detect_contours(
    da: xr.DataArray,
    levels: list[float],
    spatial_dims: tuple = ("lon", "lat"),
    processes: int = 1,
    ctx: str | None = None,
    do_round: bool = True,
) -> DataFrame:
    """
    Potentially parallel wrapper around `inner_detect_contours`. Finds contours in a DataArray at levels specified by the user, returns the results as a polars DataFrame with index columns gathered from `da`.

    Parameters
    ----------
    da : xr.DataArray
        _description_
    levels : list[float]
        _description_
    spatial_dims : tuple, optional
        _description_, by default ("lon", "lat")
    processes : int, optional
        _description_, by default 1
    ctx : str | None, optional
        _description_, by default None
    do_round : bool, optional
        _description_, by default True

    Returns
    -------
    DataFrame
        _description_
    """
    extra_dims = {dim: da[dim] for dim in da.dims if dim not in spatial_dims}
    extra_dims_values = {key: val.values for key, val in extra_dims.items()}
    key = list(extra_dims_values)[0]
    extra_dims_df = pl.DataFrame({key: extra_dims_values[key]})
    for key in list(extra_dims_values)[1:]:
        extra_dims_df = extra_dims_df.join(
            pl.DataFrame({key: extra_dims_values[key]}), how="cross"
        )
    iter1 = list(product(*list(extra_dims.values())))
    iter2 = list((dict(zip(extra_dims, stuff)) for stuff in iter1))
    iter3 = ((da.loc[indexer], levels, spatial_dims, do_round) for indexer in iter2)
    if processes > 1 and ctx is None:
        ctx = get_context("fork")
    elif processes > 1:
        ctx = get_context(ctx)
    res = map_maybe_parallel(
        iter3,
        inner_detect_contours,
        len(iter2),
        processes=processes,
        ctx=ctx,
        progress=False,
    )
    all_is, all_levels, all_contours, all_cyclics = list(zip(*res))
    all_contours = pl.DataFrame(
        {
            "contour": all_is,
            "level": all_levels,
            "contours": all_contours,
            "cyclic": all_cyclics,
        }
    )
    aggs = {col: pl.col("contours").arr.get(i) for i, col in enumerate(spatial_dims)}
    all_contours = (
        pl.concat([extra_dims_df, all_contours], how="horizontal")
        .explode("contour", "level", "contours", "cyclic")
        .explode("contours")
        .with_columns(**aggs)
        .drop("contours")
    )
    return standardize_polars_dtypes(all_contours)


def detect_contours_lonlat(
    da: xr.DataArray,
    levels: list[float],
    repeat_lons: int = 120,
    processes: int = 1,
    ctx: str | None = None,
    do_round: bool = False,
) -> DataFrame:
    """
    Wrapper around `detect_contours` with extra heuristics for spherical geometry. Can extend in longitude to capture the contours over the -180 line
    """
    if repeat_lons > 0:
        da = xr.concat(
            [
                da,
                da.isel(lon=slice(repeat_lons)).assign_coords(
                    lon=da.lon[:repeat_lons].values + 360
                ),
            ],
            dim="lon",
        )
    else:
        da = da.copy()

    all_contours = detect_contours(
        da,
        levels,
        spatial_dims=("lon", "lat"),
        processes=processes,
        ctx=ctx,
        do_round=do_round,
    )

    extra_dims = {dim: da[dim] for dim in da.dims if dim not in ["lon", "lat"]}
    extra_cols = list(extra_dims)
    index_columns = [*extra_cols, "level", "contour"]

    all_contours = (
        all_contours.unique(
            [*index_columns, pl.col("lon").round(2), pl.col("lat").round(2)],
            maintain_order=True,
        )
        .with_columns(side=(pl.col("lon") >= 180.0).cast(pl.UInt8()))
        .filter(~(pl.col("side") == 1).all().over(index_columns))
        .with_columns(len=pl.len().over(index_columns))
        .filter(pl.col("len") > 10)
    )

    # filter intersections of fully zeros within nonzero
    lon_wrapped = (pl.col("lon") + 180) % 360 - 180
    points = pl.concat_str(lon_wrapped.round(2), pl.col("lat").round(2), separator=" ")
    df = all_contours.group_by(index_columns).agg(
        points=points, side=pl.col("side").mean(), len=points.len()
    )

    filters = [
        pl.col("contour") != pl.col("contour_right"),
        (pl.col("side") > 0.0) & (pl.col("side_right") == 0.0),
    ]
    intersection = (
        pl.col("points").list.set_intersection(pl.col("points_right")).list.len()
    )
    huh_right = [f"{col}_right" for col in extra_cols]
    to_drop = (
        df.join(df, on=[*extra_cols, "level"], how="full")
        .drop(*huh_right, "level_right")
        .filter(filters)
        .with_columns(intersection=intersection)
        .filter(pl.col("intersection") > 0.97 * pl.col("len_right"))
        .select(*[*index_columns[:-1], "contour_right"], drop=pl.lit(True))
        .rename({"contour_right": "contour"})
        .sort(index_columns)
    )

    all_contours = (
        all_contours.join(to_drop, on=index_columns, how="left")
        .filter(~pl.col("drop").fill_null(False))
        .drop("drop")
    )

    # # filter points that appear twice in a mean-side > 0 contours
    # all_contours = all_contours.with_columns(
    #     inside_index=pl.int_range(pl.len()).over(index_columns)
    # )
    # p1 = pl.concat_arr(pl.col("lon"), pl.col("lat").round(1))
    # p2 = pl.concat_arr(pl.col("lon") - 360, pl.col("lat").round(1))
    # to_drop = (
    #     all_contours
    #     .filter(pl.col("side").mean().over(index_columns) > 0.0)
    #     .with_columns(p1=p1, p2=p2)
    #     .filter((pl.col("lon") - 360).is_in(pl.col("lon").implode()).over(index_columns))
    #     .filter(pl.col("p1").arr.get(1) == pl.col("p2").arr.get(1))
    #     .select(*index_columns, "inside_index", drop=True)
    # )
    # bring back within -180 -- +180
    all_contours = (
        all_contours
        # .join(to_drop, on=[*index_columns, "inside_index"], how="left")
        # .filter(~pl.col("drop").fill_null(False))
        # .drop("drop")
        .with_columns(lon=lon_wrapped)
        .unique([*index_columns, "lat", "lon"], maintain_order=True)
        .with_columns(len=pl.len().over(index_columns))
        .filter(pl.col("len") > 10)
    )
    all_contours = sort_by_newindex(all_contours, index_columns[:-1], "contour")
    backward = signed_difflon().sum() < 0
    index = pl.when(backward).then(pl.col("index").reverse()).otherwise("index")
    all_contours.with_columns(index=index.over(index_columns))
    return all_contours.sort([*index_columns, "index"])


def compute_alignment(all_contours: DataFrame, periodic: bool = False) -> DataFrame:
    """
    This function computes the alignment criterion for zero-sigma-contours. It is the scalar product betweeen the vector from a contour point to the next and the horizontal wind speed vector.
    """
    index_columns = get_index_columns(
        all_contours, ("member", "time", "cluster", "spell", "relative_index")
    )
    dlon = diff_maybe_periodic("lon", periodic)
    dlat = central_diff("lat")
    ds = haversine_from_dl(pl.col("lat"), dlon, dlat)
    align_x = (
        pl.col("u")
        / pl.col("s")
        * RADIUS
        * pl.col("lat").radians().cos()
        * dlon.radians()
        / ds
    )
    align_y = pl.col("v") / pl.col("s") * RADIUS * dlat.radians() / ds
    alignment = align_x + align_y
    return all_contours.with_columns(
        alignment=alignment.over([*index_columns, "contour"])
    )


def event_geometry(
    events: pl.DataFrame,
    mode: Literal["envelope", "convex_hull", "polygon"] = "envelope",
    index_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Turns overturning and streamer events into dataframes with a geometry column.

    Parameters
    ----------
    events : pl.DataFrame
        _description_
    mode : Literal[&quot;envelope&quot;, &quot;convex_hull&quot;, &quot;polygon&quot;], optional
        _description_, by default "envelope"
    index_columns : list[str] | None, optional
        _description_, by default None

    Returns
    -------
    DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if index_columns is None:
        # lev is vertical level (e.g, 250 hPa, 330K, 2PVU)
        # level is contour level (e.g. 2PVU, 9.4AVU)
        index_columns = get_index_columns(
            events, ["member", "time", "lev", "level", "index"]
        )
    if mode == "envelope":
        geometry = st.linestring("points").st.envelope()
    elif mode == "convex_hull":
        geometry = st.linestring("points").st.envelope()
    elif mode == "polygon":
        geometry = st.polygon(
            pl.col("points")
            .list.concat(pl.col("points").list.gather([0]))
            .implode()
            .over([*index_columns, "side"])
        ).st.make_valid()  # god
    else:
        raise ValueError
    other_columns = [
        pl.col(col).first()
        for col in events.columns
        if col not in ["lon", "lat", *index_columns, "side"]
    ]
    join_geoms = pl.col("geometry").first().st.union(pl.col("geometry").last())
    join_geoms = (
        pl.when((pl.len() > 1) & (pl.col("points").list.first().len() > 1))
        .then(join_geoms)
        .otherwise(pl.col("geometry").first())
    )
    events = (
        events.group_by([*index_columns, "side"])
        .agg(points=pl.concat_arr("lon", "lat"), *other_columns)
        .filter(pl.col("points").list.eval(pl.element().len() > 1).list.all())
        .with_columns(geometry=geometry)
        .group_by(index_columns)
        .agg(join_geoms, "points", "side", *other_columns)
    )
    return events


def detect_overturnings(
    contours: pl.DataFrame,
    max_difflon: float = 5,
    min_lon_ext: float = 5,
    min_lat_ext: float = 5,
    min_len: int = 5,
) -> pl.DataFrame:
    """
    Detects overturnings from absolute or potential vorticity contours, using Barnes and Hartmann 2013.

    Parameters
    ----------
    contours : pl.DataFrame
        _description_
    max_difflon : float, optional
        _description_, by default 5
    min_lon_ext : float, optional
        _description_, by default 5
    min_lat_ext : float, optional
        _description_, by default 5
    min_len : int, optional
        _description_, by default 5

    Returns
    -------
    pl.DataFrame
        _description_
    """
    index_columns = get_index_columns(
        contours, ["member", "time", "lev", "level", "contour"]
    )
    unique_counts = (
        contours.group_by(index_columns)
        .agg(lon=pl.col("lon").unique(), counts=pl.col("lon").unique_counts())
        .explode("lon", "counts")
    )
    overturnings = contours.join(
        unique_counts, on=[*index_columns, "lon"], how="left"
    ).filter(pl.col("counts") >= 3)

    subindex = (difflon() > max_difflon).fill_null(False).cum_sum()
    newindex = (pl.int_range(pl.len()) - difflon().arg_max()) % pl.len()
    newindex = (
        pl.when(difflon().max() > max_difflon)
        .then(newindex)
        .otherwise(pl.int_range(pl.len()))
    )
    newside = 1 - (pl.col("lon") >= pl.col("lon").get(difflon().arg_max()))
    newside = (
        pl.when(difflon().max().over([*index_columns, "subindex"]) > 10)
        .then(newside)
        .otherwise(pl.lit(0))
    )

    indexer = (
        overturnings.unique([*index_columns, "lon"])
        .sort(*index_columns, "lon")
        .with_columns(newindex=newindex.over(index_columns))
        .sort(*index_columns, "newindex")
        .with_columns(subindex=subindex.over(index_columns))
        .drop("len", "index", "counts")
    )

    overturnings = (
        overturnings.join(
            indexer[*index_columns, "lon", "subindex", "newindex"],
            on=[*index_columns, "lon"],
            how="left",
        )
        .group_by(*index_columns, "subindex", maintain_order=True)
        .agg(
            pl.col("lon"),
            pl.col("lat"),
            lat_ext=pl.col("lat").max() - pl.col("lat").min(),
            lon_ext=(pl.col("lon").last() - pl.col("lon").first()).abs(),
            len=pl.len(),
            side=pl.col("side"),
            inside_index=newindex,
        )
        .filter(
            pl.col("lon_ext") > min_lon_ext,
            pl.col("lat_ext") > min_lat_ext,
            pl.col("len") > min_len,
        )
        .explode("lon", "lat", "side", "inside_index")
        .with_columns(
            side=(pl.col("side") > 0)
            .any()
            .over(*index_columns, "subindex", pl.col("lon"))
            .cast(pl.UInt8())
        )
        .with_columns(side=newside.over([*index_columns, "subindex"]))
        .sort([*index_columns, "subindex", "inside_index"])
    )

    index = pl.concat_arr(
        pl.col("contour").cast(pl.UInt32()), pl.col("subindex").cast(pl.UInt32())
    ).rle_id()
    index_columns.remove("contour")
    index = index.over(index_columns)
    overturnings = overturnings.with_columns(index=index).drop("contour", "subindex")
    index_columns.append("index")

    backward = signed_difflon().sum() < 0
    inside_index = (
        pl.when(backward)
        .then(pl.col("inside_index").reverse())
        .otherwise("inside_index")
    )
    overturnings = overturnings.with_columns(
        inside_index=inside_index.over(index_columns)
    )
    overturnings = overturnings.sort([*index_columns, "inside_index"])

    lat_west = pl.col("lat").first()
    lat_east = pl.col("lat").last()
    orientation = (
        pl.when(lat_west.abs() <= lat_east.abs())
        .then(pl.lit("cyclonic"))
        .otherwise(pl.lit("anticyclonic"))
    )

    overturnings = overturnings.with_columns(
        orientation=orientation.over(index_columns)
    )

    return event_geometry(overturnings, "envelope", index_columns)


def detect_streamers(
    contours: pl.DataFrame,
    max_realdist: float = 8e5,
    min_contourdist: float = 1e6,
    max_contourdist: float = 1e7,
    min_ratio: float = 10,
) -> pl.DataFrame:
    """
    Detects streamers from potential or absolute vorticity contours using wernli sprenger 2015.

    Parameters
    ----------
    contours : pl.DataFrame
        _description_
    max_realdist : float, optional
        _description_, by default 8e5
    min_contourdist : float, optional
        _description_, by default 1e6
    max_contourdist : float, optional
        _description_, by default 1e7
    min_ratio : float, optional
        _description_, by default 10

    Returns
    -------
    pl.DataFrame
        _description_
    """
    index_columns = get_index_columns(
        contours, ["member", "time", "lev", "level", "contour"]
    )

    ds = haversine(
        "lon",
        "lat",
        pl.col("lon").shift(),
        pl.col("lat").shift(),
    )
    ds = ds.fill_null(0.0)
    s = ds.cum_sum()
    contours = contours.with_columns(
        s=s.over(index_columns),
        max_s=s.max().over(index_columns),
        max_n=pl.col("index").max().over(index_columns),
    ).with_columns(cs.signed_integer().cast(pl.Int32()))

    contourslazy = contours.lazy()

    dist_forward = pl.col("s_right") - pl.col("s")
    dist_backward = pl.col("max_s") - (pl.col("s_right") - pl.col("s"))
    dist2 = pl.min_horizontal(dist_forward, dist_backward)
    forward = dist_forward <= dist_backward

    streamers = (
        contourslazy.join(
            contourslazy.select(*index_columns, "index", "lon", "lat", "s"),
            on=index_columns,
        )
        .filter(pl.col("index_right") > pl.col("index"))
        .with_columns(
            dist1=haversine("lon", "lat", "lon_right", "lat_right"),
            dist2=dist2,
            forward=forward,
            ratio=dist2 / haversine("lon", "lat", "lon_right", "lat_right"),
        )
        .filter(
            pl.col("dist1") < max_realdist,
            pl.col("dist2") > min_contourdist,
            pl.col("dist2") < max_contourdist,
            pl.col("ratio") > min_ratio,
            pl.col("forward") | (~pl.col("cyclic")),
        )
        .collect(streaming=True)
    )

    max_ratio = pl.col("dist1") == pl.col("dist1").min()
    max_ratio_left = max_ratio.over([*index_columns, "index_right"])
    max_ratio_right = max_ratio.over([*index_columns, "index"])

    range_ = pl.int_ranges(pl.col("index"), pl.col("index_right") + 1)
    other_range = pl.int_ranges(
        pl.col("index_right"), pl.col("index") + 1 + pl.col("max_n")
    ) % pl.col("max_n")
    range_ = pl.when("forward").then(range_).otherwise(other_range)

    streamers = (
        streamers[*index_columns, "index", "index_right", "forward", "max_n", "dist1"]
        .filter(max_ratio_right)
        .filter(max_ratio_left)
        .with_columns(range=range_)
        .sort(*index_columns, "index")
    )

    to_drop = (
        streamers.join(streamers, on=index_columns, suffix="_other", how="left")
        .filter(pl.col("range").list.len() < pl.col("range_other").list.len())
        .with_columns(
            drop=pl.col("range_other").list.contains(pl.col("index"))
            & pl.col("range_other").list.contains(pl.col("index_right"))
        )
        .group_by([*index_columns, "index", "index_right"])
        .agg(pl.col("drop").any())
    )

    streamers = (
        streamers.join(
            to_drop,
            on=[*index_columns, "index", "index_right"],
            how="left",
        )
        .filter(~pl.col("drop").fill_null(False))
        .drop("drop")
        .with_columns(subindex=pl.int_range(pl.len()).over(index_columns))
    )
    l1 = pl.col("range").list.len()
    l2 = pl.col("range_other").list.len()
    linter = pl.col("range").list.set_intersection(
        pl.col("range_other")
    ).list.len() / pl.min_horizontal(l1, l2)

    streamers = (
        streamers[
            *index_columns,
            "range",
            "subindex",
            "index",
            "index_right",
            "forward",
            "max_n",
        ]
        .join(
            streamers[*index_columns, "range", "subindex", "index", "index_right"],
            on=index_columns,
            how="left",
            suffix="_other",
        )
        .with_columns(l1=l1, l2=l2)
        .with_columns(linter=linter)
        .filter(pl.col("linter") > 0.8)
        .group_by(*index_columns, "index", "index_right", "forward", "subindex")
        .agg(
            minio=pl.col("index_other").min(),
            maxio=pl.col("index_other").max(),
            miniro=pl.col("index_right_other").min(),
            maxiro=pl.col("index_right_other").max(),
            max_n=pl.col("max_n").first(),
        )
        .with_columns(
            index=pl.when("forward").then("minio").otherwise("maxio"),
            index_right=pl.when("forward").then("maxiro").otherwise("miniro"),
        )
        .drop(["minio", "maxio", "miniro", "maxiro"])
        .unique([*index_columns, "index", "index_right", "forward"])
        .sort(*index_columns, "index", "index_right")
        .with_columns(subindex=pl.int_range(pl.len()).over(index_columns))
        .with_columns(range=range_)
    )

    index = (
        pl.when("forward").then(pl.col("index").min()).otherwise(pl.col("index").max())
    )
    index_right = (
        pl.when("forward")
        .then(pl.col("index_right").max())
        .otherwise(pl.col("index_right").min())
    )

    streamers = (
        streamers.explode("range")
        .unique([*index_columns, "range"])
        .group_by([*index_columns, "forward", "subindex"])
        .agg(
            pl.col("max_n").first(),
            index=index.first(),
            index_right=index_right.first(),
        )
        .with_columns(range=range_)
        .drop("index", "index_right", "max_n")
        .explode("range")
        .with_columns(cs.signed_integer().cast(pl.Int32()))
        .rename({"range": "index"})
        .join(
            contours.drop("s", "len", "max_s", "max_n", "cyclic"),
            on=[*index_columns, "index"],
        )
        .sort(*index_columns, "forward", "subindex", "index")
        .drop("index")
    )

    index = pl.concat_arr(
        pl.col("contour").cast(pl.UInt32()), pl.col("subindex").cast(pl.UInt32())
    ).rle_id()
    index_columns.remove("contour")
    index = index.over(index_columns)
    streamers = streamers.with_columns(index=index)
    index_columns.append("index")
    return event_geometry(streamers, "polygon", index_columns=index_columns)


def sjoin_to_grid(
    events: pl.DataFrame,
    da: xr.DataArray,
    varname: str = "ones",
    buffer: float | None = None,
) -> pl.DataFrame:
    """
    Taking an events DataFrame containing a geometry, augments it with lon and lat columns which, for each geometry, will contain all the lon, lat points within the geometry. The potential lon, lat points to look for are the ones defined by da, who only needs to be 2D since the other dimensions will be discarded.

    Parameters
    ----------
    events : pl.DataFrame
        _description_
    da : xr.DataArray
        _description_
    varname : str, optional
        _description_, by default "ones"
    buffer : float | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if da.name is None:
        da = da.rename("dummy")
    da_name = da.name
    dx = (da.lon[1] - da.lon[0]).item()
    dy = (da.lat[1] - da.lat[0]).item()
    if buffer is None:
        buffer = min(dx, dy) / 2
    indexer_grid = [slice(None) if dim in ["lon", "lat"] else 0 for dim in da.dims]
    nogrid = [dim for dim in da.dims if dim not in ["lon", "lat"]]
    da_df = da[*indexer_grid]
    dtype = {
        "ones": pl.UInt32(),
        "intensity": pl.Float32(),
        "mean_var": pl.Float32(),
        "event_area": pl.Float32(),
    }[varname]

    da_df = (
        pl.from_pandas(da_df.to_dataframe().reset_index())
        .drop(da_name, *nogrid)
        .cast({"lon": pl.Float32, "lat": pl.Float32})
        .unique(["lat", "lon"])
        .sort(["lat", "lon"])
        .with_columns(geometry=st.point(pl.concat_arr("lon", "lat")))
    )

    index_columns = get_index_columns(
        events, ["member", "time", "lev", "level", "index"]
    )
    events = events.drop(
        "points", "side"
    )  # .with_columns(pl.col("geometry").st.buffer(buffer))
    if varname == "ones":
        events = events.with_columns(ones=pl.lit(1))
    events = events.cast({varname: dtype})
    events = events.st.sjoin(
        da_df, on="geometry", how="inner", predicate="within"
    ).drop("geometry_right")
    # events = events.sort(*index_columns, "lat", "lon")
    return events


def to_xarray_sjoin(
    da: xr.DataArray,
    events: pl.DataFrame | None = None,
    events_on_grid: pl.DataFrame | None = None,
    varname: str = "ones",
    buffer: float | None = None,
) -> xr.DataArray:
    """
    Turns a event dataframe into a gridded DataArray, with zeros everywhere except where and when there is an overlap with a geometry, and there the value of the variable "varname", typically just ones to create a mask.

    This is an expensive operation, optimise to do this only once.

    Parameters
    ----------
    da : xr.DataArray
        _description_
    events : pl.DataFrame | None, optional
        _description_, by default None
    events_on_grid : pl.DataFrame | None, optional
        _description_, by default None
    varname : str, optional
        _description_, by default "ones"
    buffer : float | None, optional
        _description_, by default None

    Returns
    -------
    xr.DataArray
        _description_
    """
    if events_on_grid is None:
        events_on_grid = sjoin_to_grid(events, da, varname, buffer)
    index_columns = get_index_columns(
        events_on_grid, ["member", "time", "lev", "level"]
    )
    index_columns.extend(["lon", "lat"])
    for i, index_column in enumerate(index_columns):
        if index_column in da.dims:
            continue
        unique_vals = events_on_grid[index_column].unique().to_list()
        da = da.expand_dims({index_column: unique_vals}, axis=i).copy(deep=True)
    indexer = {name: xr.DataArray(events_on_grid[name]) for name in index_columns}
    value = events_on_grid[varname].to_numpy()
    dtype = np.uint8 if varname == "ones" else np.float32
    da[:] = 0
    da.loc[indexer] = value
    da = da.fillna(0)
    da = da.astype(dtype)
    return da


def join_wrapper(
    df: DataFrame,
    da: xr.DataArray | xr.Dataset,
    join_dims: list | None = None,
    suffix: str = "_right",
    **kwargs,
):
    """
    Joins a DataFrame with a DataArray on the latter's dimensions. Explicitly iterates over years and members to limit memory usage.

    Should be merged cleanly with `join_on_ds` since they do similar things, but also not really.

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
    if join_dims is None:
        join_dims = da.dims
    for idx1, idx2 in indexer:
        these_jets = df.filter(*idx1)
        da_ = compute(da.sel(**idx2), progress_flag=False)
        da_ = xarray_to_polars(da_)
        these_jets = these_jets.join(da_, on=join_dims, how="left", suffix=suffix)
        df_upd.append(these_jets)
    df = pl.concat(df_upd)
    return df


def event_props(
    events: pl.DataFrame,
    das: list[xr.DataArray],
    events_on_grid: pl.DataFrame | None = None,
):
    """
    Computes various properties of event geometries such as area-mean quantities like zeta or momentum flux.

    Parameters
    ----------
    events : pl.DataFrame
        _description_
    das : list[xr.DataArray]
        _description_
    events_on_grid : pl.DataFrame | None, optional
        _description_, by default None

    Returns
    -------
    pl.DataFrame
        Same as input but with a few more columns
    """
    index_columns = get_index_columns(
        events, ["member", "time", "lev", "level", "index"]
    )
    if events_on_grid is None:
        events_on_grid = sjoin_to_grid(events, das[0])

    dx = (das[0].lon[1] - das[0].lon[0]).item()
    dy = (das[0].lat[1] - das[0].lat[0]).item()
    cell_area = (
        (
            (pl.col("lat") + pl.lit(dy / 2)).radians().sin()
            - (pl.col("lat") - pl.lit(dy / 2)).radians().sin()
        )
        * pl.lit(dx).radians()
        * RADIUS**2
    )
    cell_area = cell_area.abs().cast(pl.Float32())
    com_x = circular_mean(pl.col("lon"), "cell_area").cast(pl.Float32())
    com_y = weighted_mean_pl(pl.col("lat"), "cell_area").cast(pl.Float32())
    events_on_grid = events_on_grid.with_columns(cell_area=cell_area)

    aggs = {"area": pl.col("cell_area").sum(), "com_x": com_x, "com_y": com_y}

    for i, da in enumerate(das):
        if da.name is None:
            da.rename(f"da_{i}")
        events_on_grid = join_wrapper(events_on_grid, da)
        aggs[da.name] = (pl.col(da.name) * pl.col("cell_area")).sum() / pl.col(
            "cell_area"
        ).sum()

    events_on_grid_ = events_on_grid.group_by(index_columns).agg(**aggs)

    events = events.join(events_on_grid_, on=index_columns, how="left")

    events = events.with_columns(
        cs.float().cast(pl.Float32()),
        cs.signed_integer().cast(pl.Int32()),
        cs.unsigned_integer().cast(pl.UInt32()),
    )
    return events, events_on_grid


def calculate_streamer_angle(
    streamers: pl.DataFrame, vort_name: str = "PV"
) -> pl.DataFrame:
    points = pl.col("points").explode()
    lon0 = points.list.get(0).arr.get(0)
    lat0 = points.list.get(0).arr.get(1)
    lon1 = points.list.get(-1).arr.get(0)
    lat1 = points.list.get(-1).arr.get(1)
    lon_midpoint = (lon0 + lon1) / 2
    lat_midpoint = (lat0 + lat1) / 2

    lons = points.list.eval(pl.element().arr.first())
    lats = points.list.eval(pl.element().arr.last())
    southernmost = lats.list.eval(pl.element().abs()).list.arg_min()
    lon_southernmost = lons.list.get(southernmost)
    lat_southernmost = lats.list.get(southernmost)
    northernmost = lats.list.eval(pl.element().abs()).list.arg_max()
    lon_northernmost = lons.list.get(northernmost)
    lat_northernmost = lats.list.get(northernmost)

    angle_up = pl.arctan2(
        lat_midpoint - lat_southernmost, lon_midpoint - lon_southernmost
    ).degrees()
    angle_down = pl.arctan2(
        lat_northernmost - lat_midpoint, lon_northernmost - lon_midpoint
    ).degrees()
    angle = (
        pl.when(pl.col(vort_name).abs() > pl.col("level").abs())
        .then(angle_up)
        .otherwise(angle_down)
    )

    return streamers.with_columns(axis_angle=angle)


def interp_from_other(jets: DataFrame, da_df: DataFrame, varname: str) -> DataFrame:
    """
    Bilinear interpolation. Values in `da_df[varname]` will be bilinearly interpolated to the jet points' `lon`-`lat` coordinates, resulting in a new column in `jets` with a name constructed as `f"{varname}_interp"`.

    Parameters
    ----------
    jets : DataFrame
        Interpolation target
    da_df : DataFrame
        Interpolation source, already translated to a DataFrame
    varname : str
        columns of `da_df` to take values from. The rest is either index or ignored

    Returns
    -------
    DataFrame
        `jets` with one extra column
    """
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


def add_normals_meters(
    jets: DataFrame,
    half_length: float = 1.2e6,
    dn: float = 5e4,
    delete_middle: bool = False,
) -> DataFrame:
    """
    Augments a `DataFrame` containing `"lon"`, `"lat"`, `"u"` and `"v"` columns with new columns `"normallon"` and `"normallat"`.

    For each unique jet point, adds segments normal to the wind direction at this point in the same plane and in both directions. Each half-segment has a length of `half_length`, in m, and is discretized by a point every `dn` m.

    Parameters
    ----------
    jets : DataFrame
        Must contain `"lon"`, `"lat"`, `"u"` and `"v"` columns.
    half_length : float, optional
        Length of each half segment, above and under the jet at each point, by default 1e6
    dn : float, optional
        Half-segments are discretized every `dn`, by default 5e4
    delete_middle : bool, optional
        Whether the half-segments also contain the jet point itself or not, by default False

    Returns
    -------
    DataFrame
        Original DataFrame augmented by new columns and longer by a factor `2 * half_length / dn - delete_middle`
    """
    is_polar = ["is_polar"] if "is_polar" in jets.columns else []
    ns_df = np.arange(-half_length, half_length + dn, dn)
    if delete_middle:
        ns_df = np.delete(ns_df, int(half_length // dn))
    ns_df = Series("n", ns_df).to_frame()

    # Expr angle
    if "u" in jets.columns and "v" in jets.columns:
        angle = (
            pl.arctan2(pl.col("v"), pl.col("u")).interpolate("linear") + np.pi / 2
        ) % np.pi
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

    # Expr normals from https://www.movable-type.co.uk/scripts/latlong.html
    lon = pl.col("lon").radians()
    lat = pl.col("lat").radians()
    arc_distances = pl.col("n") / RADIUS
    # bearing = np.pi / 2 - angle
    normallat = (
        lat.sin() * arc_distances.cos() + lat.cos() * arc_distances.sin() * angle.sin()
    ).arcsin()
    normallon = lon + pl.arctan2(
        angle.cos() * arc_distances.sin() * lat.cos(),
        arc_distances.cos() - lat.sin() * normallat.sin(),
    )
    normallat = normallat.degrees().cast(pl.Float32())
    normallon = ((normallon.degrees() + 540) % 360 - 180).cast(pl.Float32())

    index_columns = get_index_columns(
        jets,
        [
            "member",
            "time",
            "cluster",
            "spell",
            "relative_index",
            "relative_time",
            "jet ID",
            "jet",
            "sample_index",
            "inside_index",
        ],
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
            "angle",
            *is_polar,
        ]
    ]
    return jets


def add_normals(
    jets: DataFrame,
    half_length: float = 12.0,
    dn: float = 1.0,
    delete_middle: bool = False,
) -> DataFrame:
    """
    Augments a `DataFrame` containing `"lon"`, `"lat"`, `"u"` and `"v"` columns with new columns `"normallon"` and `"normallat"`.

    For each unique jet point, adds segments normal to the wind direction at this point in the same plane and in both directions. Each half-segment has a length of `half_length`, in degrees, and is discretized by a point every `dn` degrees.

    Parameters
    ----------
    jets : DataFrame
        Must contain `"lon"`, `"lat"`, `"u"` and `"v"` columns.
    half_length : float, optional
        Length of each half segment, above and under the jet at each point, by default 12.0
    dn : float, optional
        Half-segments are discretized every `dn`, by default 1.0
    delete_middle : bool, optional
        Whether the half-segments also contain the jet point itself or not, by default False

    Returns
    -------
    DataFrame
        Original DataFrame augmented by new columns and longer by a factor `2 * half_length / dn - delete_middle`
    """
    is_polar = ["is_polar"] if "is_polar" in jets.columns else []
    ns_df = np.arange(-half_length, half_length + dn, dn)
    if delete_middle:
        ns_df = np.delete(ns_df, int(half_length // dn))
    ns_df = Series("n", ns_df).to_frame()

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
            "jet",
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
    jets: DataFrame,
    # da: xr.DataArray,
    *das: tuple[xr.DataArray],
    half_length: float = 2e6,
    dn: float = 1e5,
    delete_middle: bool = False,
    in_meters: bool = True,
) -> DataFrame:
    """
    Creates normal half-segments on either side of all jet core points, each of length `half_length` and with flat spacing `dn`. Then, interpolates the values of `da` onto each point of each normal segment.

    Parameters
    ----------
    jets : DataFrame
        Target
    da : xr.DataArray
        Source of data
    half_length : float, optional
        Length of each half segment, above and under the jet at each point, by default 12.0
    dn : float, optional
        Half-segments are discretized every `dn`, by default 1.0
    delete_middle : bool, optional
        Whether the half-segments also contain the jet point itself or not, by default False
    in_meters : bool, optional
        Whether the half-segments are discretize in meters (True) or in degrees (False), by default False

    Returns
    -------
    DataFrame
        `jets`, augmented with normal segments at each point, on whose points the data of `da` are interpolated.
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
            "jet",
            "sample_index",
            "inside_index",
        ),
    )
    schema = jets.collect_schema()
    for col in index_columns:
        dtype = schema[col]
        if all([col not in da.coords for da in das]):
            continue
        coord_vals = []
        for da in das:
            if col in da.coords:
                coord_vals.append(da[col].values)
        coord_vals = reduce(np.intersect1d, coord_vals)
        coord_vals = pl.Series(col, coord_vals).cast(dtype).implode()
        jets = jets.filter(pl.col(col).is_in(coord_vals))
    if in_meters:
        jets = add_normals_meters(jets, half_length, dn, delete_middle)
    else:
        jets = add_normals(jets, half_length, dn, delete_middle)
    da = das[0]
    dlon = (da.lon[1] - da.lon[0]).item()
    dlat = (da.lat[1] - da.lat[0]).item()
    lon = Series("normallon_rounded", da.lon.values).to_frame()
    lat = Series("normallat_rounded", da.lat.values).to_frame()
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

    for da in das:
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
        jets = interp_from_other(jets, da_df, varname).sort(
            [*index_columns, "index", "n"]
        )
    jets = jets.with_columns(side=pl.col("n").sign().cast(pl.Int8))
    return standardize_polars_dtypes(jets)


def interp_jets_to_zero_one(
    jets: pl.DataFrame, varnames: list[str] | str, n_interp: int = 30
) -> DataFrame:
    """
    Interpolates data along the `"index"` column from 0 to 1, independently for each unique jet.

    Parameters
    ----------
    jets : pl.DataFrame
        Data source
    varnames : list[str] | str
        Columns to interpolate
    n_interp : int, optional
        How many points to interpolate between 0 and 1, by default 30

    Returns
    -------
    DataFrame
        Data source with the `"index"` integer column replaced with the `"norm_index"` float column, and the variables `"varnames"` interpolated accordingly.
    """
    if isinstance(varnames, str):
        varnames = [varnames]
    index_columns = get_index_columns(jets)
    if "relative_index" in index_columns and "time" in index_columns:
        index_columns.remove("time")
        varnames.append("time")
    jets = jets.with_columns(
        norm_index=jets.group_by(index_columns, maintain_order=True)
        .agg(pl.col("index") / pl.col("index").max())["index"]
        .explode()
    )
    jets = jets.group_by(
        [*index_columns, ((pl.col("norm_index") * n_interp) // 1) / n_interp, "n"],
        maintain_order=True,
    ).agg([pl.col(varname).mean() for varname in varnames])
    return standardize_polars_dtypes(jets)


def expand_jets(jets: DataFrame, max_t: float, dt: float) -> DataFrame:
    """
    Expands the jets by appending segments before the start and after the end, following the tangent angle at the start and the end of the original jet, respectively. Broken?

    Parameters
    ----------
    jets : DataFrame
        Jets to extend
    max_t : float
        Length of the added segments
    dt : float
        Spacing of the added segment

    Returns
    -------
    DataFrame
        Jets DataFrame with all the index dimensions kept original, only lon and lat as additional columns (the rest is dropped), and longer jets with added segments.

    TODO: redo using https://www.movable-type.co.uk/scripts/latlong.html
    """
    index_columns = get_index_columns(jets, ["member", "time", "jet ID", "jet"])
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


def bias_correct(
    jets: pl.DataFrame,
    ds: xr.Dataset,
    smooth_index: int = 11,
    smooth_n: int = 2,
    period: int = 15,
    same_len: bool = False,
) -> pl.DataFrame:
    """
    Inputs and interpolation need to fit in memory. This will crash your code if you send too much at it

    Interpolates wind speed around the jet, normal-derive it (like sigma), and finds smooth 0 contours of this alternative sigma in index-n space.

    Parameters
    ----------
    jets : pl.DataFrame
        _description_
    ds : xr.Dataset
        _description_
    smooth_index : int, optional
        _description_, by default 11
    smooth_n : int, optional
        _description_, by default 2
    period : int, optional
        _description_, by default 15
    same_len : bool, optional
        _description_, by default False

    Returns
    -------
    pl.DataFrame
        _description_
    """
    offset = int(np.ceil(period / 2))
    index_columns = get_index_columns(jets)
    idxmax = jets.select(
        *index_columns, idxmax=pl.col("index").max().over(index_columns)
    )
    jets = gather_normal_da_jets(jets, ds["s"].compute(), half_length=5e5, dn=2.5e4)
    useful = jets.filter(pl.col("n") == 0)[
        *index_columns, "index", "lon", "lat", "angle"
    ]
    jets: xr.DataArray = polars_to_xarray(jets, [*index_columns, "index", "n"])[
        "s_interp"
    ]
    jets = (
        jets.rolling(index=smooth_index, min_periods=1)
        .mean()
        .rolling(n=smooth_n, min_periods=1)
        .mean()
        .differentiate("n")
    )
    kinda_len = pl.col("index_right").n_unique().over(*index_columns, "contour")
    jets = (
        detect_contours(jets, levels=[0.0], spatial_dims=("n", "index"), do_round=False)
        .drop_nulls("contour")
        .filter(~pl.col("cyclic"))
        .drop("cyclic")
        .join(idxmax, on=index_columns)
        .unique([*index_columns, "contour", "index"])
        .sort(*index_columns, "contour", "index")
        .join(useful, on=[*index_columns, pl.col("index").round().cast(pl.Int32())])
        .drop(*[f"{col}_right" for col in index_columns])
        .filter(kinda_len >= pl.col("idxmax") * 0.9)
        .with_columns(len=kinda_len)
        .drop("index_right", "idxmax", "contour", "level", "len")
    )
    if same_len:
        jets = (
            jets.with_columns(index=pl.col("index").round().cast(pl.Int32()))
            .unique([*index_columns, "index"])
            .sort(*index_columns, "index")
            .rolling(
                "index",
                period=f"{period}i",
                offset=f"-{offset}i",
                group_by=(index_columns),
            )
            .agg(pl.col("n").mean(), cs.exclude("n").first())
        )
        idx = [*index_columns, "index"]
        idx_old = useful.select(idx).unique(idx)
        idx_new = jets.select(idx).unique(idx)
        left_behind = idx_old.join(idx_new, on=idx, how="anti")
        jets = pl.concat(
            [
                jets,
                left_behind.join(useful, on=idx)
                .with_columns(n=pl.lit(0.0, pl.Float32()))
                .select(jets.columns),
            ]
        ).sort([*index_columns, "index"])
    else:
        jets = (
            jets.with_columns(
                index=pl.col("index").rle_id().over(*index_columns).cast(pl.Int32())
            )
            .sort(*index_columns, "index")
            .rolling(
                "index",
                period=f"{period}i",
                offset=f"-{offset}i",
                group_by=(index_columns),
            )
            .agg(pl.col("n").mean(), cs.exclude("n").first())
        )

    return jets


def create_bias_correction(
    jets: pl.DataFrame,
    ds: xr.Dataset,
    smooth_index: int = 20,
    smooth_n: int = 2,
    period: int = 15,
):
    """
    Iterates over time and members and call `bias_correct`.

    Parameters
    ----------
    jets : pl.DataFrame
        _description_
    ds : xr.Dataset
        _description_
    smooth_index : int, optional
        _description_, by default 20
    smooth_n : int, optional
        _description_, by default 2
    period : int, optional
        _description_, by default 15

    Returns
    -------
    _type_
        _description_
    """
    indexer = list(iterate_over_year_maybe_member(jets, ds))
    to_average = []
    for idx1, idx2 in tqdm(indexer, total=len(indexer)):
        jets_ = jets.filter(*idx1)
        ds_ = ds.sel(**idx2)
        jets_ = bias_correct(
            jets_,
            ds_,
            smooth_index=smooth_index,
            smooth_n=smooth_n,
            period=period,
            same_len=True,
        )
        to_average.append(jets_)
    return pl.concat(to_average)


def create_jet_relative_dataset(
    jets,
    *das: tuple[xr.DataArray],
    bias_correction: pl.DataFrame | None = None,
    half_length: float = 2e6,
    dn: float = 1e5,
    n_interp: int = 30,
    in_meters: bool = True,
    align_2d: str | None = None,
) -> pl.DataFrame:
    """
    Wrapper wrappy wraps. Iterates over time, member etc and calls `gather_normal_da_jets`, potentially bias-correts the results if bias_correction is not None, then interpolates the `index` dimension to 0-1 using `interp_jets_to_zero_one`.

    Parameters
    ----------
    jets : _type_
        _description_
    da : _type_
        _description_
    bias_correction : pl.DataFrame | None, optional
        _description_, by default None
    half_length : float, optional
        _description_, by default 2e6
    dn : float, optional
        _description_, by default 1e5
    n_interp : int, optional
        _description_, by default 30
    in_meters : bool, optional
        _description_, by default True

    Returns
    -------
    pl.DataFrame
        _description_
    """
    indexer = list(iterate_over_year_maybe_member(jets, das[0]))
    to_average = []
    index_columns = get_index_columns(
        jets,
        (
            "member",
            "time",
            "cluster",
            "spell",
            "relative_index",
            "relative_time",
            "sample_index",
            "inside_index",
        ),
    )
    which_jet = "jet" if "jet" in jets.columns else "jet ID"
    varnames = [da.name + "_interp" for da in das]
    if len(das) == 2 and align_2d is not None:
        varnames.append(f"{align_2d}_interp")
    for idx1, idx2 in tqdm(indexer, total=len(indexer)):
        jets_ = jets.filter(*idx1)
        das_ = [compute(da.sel(**idx2)) for da in das]
        if bias_correction is not None:
            bias_correction_ = bias_correction.filter(*idx1)
            extra_n = bias_correction["n"].abs().max()
            extra_n = (extra_n // dn) * dn
        else:
            bias_correction_ = None
            extra_n = 0
        try:
            jets_with_interp = gather_normal_da_jets(
                jets_,
                *das_,
                half_length=half_length + extra_n,
                dn=dn,
                in_meters=in_meters,
            )
        except (KeyError, ValueError) as e:
            print(e)
            break
        if bias_correction_ is not None:
            mapping = nearest_mapping(bias_correction_, jets_with_interp, "n")
            bias_correction_ = (
                bias_correction_.join(mapping, on="n").drop("n").rename({"n_": "n"})
            )
            jets_with_interp = (
                jets_with_interp.join(
                    bias_correction_, on=[*index_columns, which_jet, "index"]
                )
                .with_columns(n=pl.col("n") - pl.col("n_right"))
                .filter(pl.col("n").abs() <= half_length)
                .with_columns(side=pl.col("n").sign().cast(pl.Int32()))
                .drop("n_right")
            )
            jets_with_interp = jets_with_interp.filter(pl.col("n").abs() <= half_length)
        if len(das) == 2 and align_2d is not None:
            angle = pl.col("angle") - pl.lit(np.pi / 2)
            agg = angle.cos() * pl.col(varnames[0]) + angle.sin() * pl.col(varnames[1])
            agg = agg.cast(pl.Float32())
            agg = {f"{align_2d}_interp": agg}
            jets_with_interp = jets_with_interp.with_columns(**agg)
        jets_with_interp = interp_jets_to_zero_one(
            jets_with_interp, [*varnames, "is_polar"], n_interp=n_interp
        )
        to_average.append(jets_with_interp)
    return pl.concat(to_average)


def compute_relative_clim(
    jets: pl.DataFrame | pl.LazyFrame, varname: str
) -> pl.DataFrame:
    return (
        jets.group_by(
            pl.col("time").dt.ordinal_day().alias("dayofyear"),
            "norm_index",
            "n",
            "jet",
        )
        .agg(pl.col(f"{varname}_interp").mean())
        .sort("jet", "dayofyear", "norm_index", "n")
    )


def compute_relative_std(jets: pl.DataFrame | pl.LazyFrame, varname: str):
    return (
        jets.group_by(
            pl.col("time").dt.ordinal_day().alias("dayofyear"),
            "norm_index",
            "n",
            "jet",
        )
        .agg(pl.col(f"{varname}_interp").std())
        .sort("jet", "dayofyear", "norm_index", "n")
    )


def compute_relative_sm(
    clim: pl.DataFrame | pl.LazyFrame, varname: str, season_doy: pl.Series | None = None
):
    if season_doy is None:
        season_doy = pl.Series("", JJADOYS.values)
    return clim.with_columns(
        **{
            f"{varname}_interp": pl.col(f"{varname}_interp")
            .filter(pl.col("dayofyear").is_in(season_doy.implode()))
            .mean()
            .over("jet", "n", "norm_index")
        }
    )


def compute_relative_anom(
    jets: pl.DataFrame | pl.LazyFrame,
    varname: str,
    clim: pl.DataFrame | pl.LazyFrame,
    clim_std: pl.DataFrame | None = None,
):
    varname_ = f"{varname}_interp"
    if clim_std is None:
        return (
            jets.with_columns(dayofyear=pl.col("time").dt.ordinal_day())
            .join(clim, on=["jet", "dayofyear", "norm_index", "n"])
            .with_columns(pl.col(varname_) - pl.col(f"{varname_}_right"))
            .drop(f"{varname_}_right", "dayofyear")
        )
    return (
        jets.with_columns(dayofyear=pl.col("time").dt.ordinal_day())
        .join(clim, on=["jet", "dayofyear", "norm_index", "n"])
        .with_columns(pl.col(varname_) - pl.col(f"{varname_}_right"))
        .drop(f"{varname_}_right")
        .join(clim_std, on=["jet", "dayofyear", "norm_index", "n"])
        .with_columns(pl.col(varname_) / pl.col(f"{varname_}_right"))
        .drop(f"{varname_}_right", "dayofyear")
    )
    
    
def common_relative_plot(
    varname: str,
    basepath: Path,
    jet: str,
    season: pl.Series,
    phat: bool = False,
    and_std: bool = False,
):
    winsize = 31
    halfwinsize = int(winsize / 2)
    season_doy = season.dt.ordinal_day().unique()

    clims_path = basepath.joinpath("relative_clims")
    clims_path.mkdir(exist_ok=True)
    clim_path = clims_path.joinpath(
        f"{varname}_{season_doy[0]}-{season_doy[-1]}.parquet"
    )

    first_ = pl.col("time").min() - datetime.timedelta(days=halfwinsize)
    last_ = pl.col("time").max() + datetime.timedelta(days=halfwinsize)
    interval = season[1] - season[0]
    season_ = (
        season.to_frame()
        .group_by(pl.col("time").dt.year().alias("year"))
        .agg(time=pl.datetime_range(first_, last_, interval=interval, closed="both"))
        .explode("time")["time"]
    )

    if ":" in varname:
        varname, mode = varname.split(":")
    else:
        mode = ""
    grad = mode == "grad"
    varname_no_number = varname.rstrip("0123456789")
    varname_ = f"{varname}_interp"
    prefix = "phat_" if phat else ""
    which_jet = "jet" if phat else "jet ID"

    df = pl.scan_parquet(basepath.joinpath(f"{prefix}{varname}_relative.parquet"))
    df = season_.to_frame().lazy().join(df, on="time")
    if varname_ not in df.collect_schema().names():
        if f"{varname_no_number}_interp" in df.collect_schema().names():
            df = df.rename({f"{varname_no_number}_interp": varname_})
            print(varname_no_number, "->", varname_)
    expr = pl.col(varname_).replace([float("-inf"), float("inf"), float("nan")], None)
    df = df.with_columns(expr)
    if grad:
        grad_expr = (
            (
                central_diff(pl.col(varname_).sort_by("n"))
                / central_diff(pl.col("n").sort())
            )
            * 1e6
        ).abs()
        df = df.with_columns(
            **{varname_: grad_expr.over("norm_index", "time", which_jet)}
        )
    if not phat:
        props = pl.scan_parquet(basepath.joinpath("props.parquet"))
        join_ = props.select(
            "time",
            "jet ID",
            "int",
            jet=pl.when(pl.col("is_polar") < 0.5)
            .then(pl.lit("STJ"))
            .otherwise(pl.lit("EDJ")),
        )
        df = df.join(join_, on=["time", "jet ID"])

    if clim_path.is_file():
        clim = pl.read_parquet(clim_path)
    elif not phat:
        df_cat = (
            df.group_by("time", "jet", "n", "norm_index")
            .agg((pl.col(varname_) * pl.col("int")).sum() / pl.col("int").sum())
            .sort("time", "jet", "n", "norm_index")
        )
        clim: pl.DataFrame = compute_relative_clim(df_cat, varname).collect()
        clim.write_parquet(clim_path)
    else:
        clim = compute_relative_clim(df, varname).collect()
        clim.write_parquet(clim_path)

    clim_sm = compute_relative_sm(clim, varname, season_doy)
    clim_sm = clim_sm.filter(pl.col("dayofyear") == season_doy[0], pl.col("jet") == jet)
    clim_sm = clim_sm.drop("jet", "dayofyear")

    clim = clim.rolling(
        pl.col("dayofyear").cast(pl.UInt32()),
        period=f"{winsize}i",
        offset=f"-{halfwinsize + 1}i",
        group_by=["jet", "norm_index", "n"],
    ).agg(pl.col(varname_).mean())
    if not and_std:
        return varname_, df, clim, clim_sm, props 
    
    clim_std_path = clims_path.joinpath(
        f"{varname}_std_{season_doy[0]}-{season_doy[-1]}.parquet"
    )
    
    if clim_std_path.is_file():
        clim_std = pl.read_parquet(clim_std_path)
    elif not phat:
        df_cat = (
            df.group_by("time", "jet", "n", "norm_index")
            .agg((pl.col(varname_) * pl.col("int")).sum() / pl.col("int").sum())
            .sort("time", "jet", "n", "norm_index")
        )
        clim_std: pl.DataFrame = compute_relative_std(df_cat, varname).collect()
        clim_std.write_parquet(clim_std_path)
    else:
        clim_std = compute_relative_std(df, varname).collect()
        clim_std.write_parquet(clim_std_path)

    clim_std_sm = compute_relative_sm(clim_std, varname, season_doy)
    clim_std_sm = clim_std_sm.filter(pl.col("dayofyear") == season_doy[0], pl.col("jet") == jet)
    clim_std_sm = clim_std_sm.drop("jet", "dayofyear")
    return varname_, df, clim, clim_sm, clim_std_sm, props 


def create_relative_plot(
    varname: str,
    basepath: Path,
    jet: str,
    spells: pl.DataFrame,
    season: pl.Series,
    n_bootstraps: int = 40,
    factor: float = 1.0,
    phat: bool = False,
):
    varname_, df, clim, clim_sm, props = common_relative_plot(varname, basepath, jet, season, phat)
    to_plot = compute_relative_anom(df, varname, clim.lazy())
    if phat:
        ts_bootstrapped = create_bootstrapped_times(spells, season, n_bootstraps).lazy()
        to_plot = (
            ts_bootstrapped.join(to_plot, on="time")
            .filter(pl.col("jet") == jet)
            .sort("sample_index", "spell", "inside_index", "norm_index", "n")
        )
    else:
        ts_bootstrapped = bs_times_with_jet_ID(
            spells, season, n_bootstraps, props.collect()
        ).lazy()
        to_plot = ts_bootstrapped.join(to_plot, on=["time", "jet ID"]).sort(
            "sample_index", "spell", "inside_index", "norm_index", "n"
        )
    to_plot = to_plot.group_by(
        "sample_index", "norm_index", "n", maintain_order=True
    ).agg(pl.col(varname_).mean())
    pvals = (
        to_plot.group_by("norm_index", "n", maintain_order=True)
        .agg((pl.col(varname_).rank().last() - 1) / n_bootstraps)
        .sort("norm_index", "n")
    )
    pvals = pvals.with_columns(
        **{varname_: 2 * pl.min_horizontal(pl.col(varname_), 1 - pl.col(varname_))}
    )
    to_plot = (
        to_plot.filter(pl.col("sample_index") == n_bootstraps)
        .drop("sample_index")
        .sort("norm_index", "n")
    )
    to_plot = to_plot.with_columns(pl.col(varname_) * factor)
    to_plot = to_plot.collect()
    to_plot = polars_to_xarray(to_plot, ["n", "norm_index"])
    pvals = pvals.collect()
    pvals = polars_to_xarray(pvals, ["n", "norm_index"])
    clim_sm = polars_to_xarray(clim_sm, ["n", "norm_index"]) * factor
    return to_plot, pvals, clim_sm


def create_all_relative_plots(
    spells_list: dict[str, pl.DataFrame],
    data_vars: list[str],
    basepath: Path,
    odir: Path,
    season: pl.DataFrame | None = None,
    n_bootstraps: int = 100,
) -> None:
    for name, spells in spells_list.items():
        if "_" in name:
            rest, jet = name.split("_")
            ipath = basepath.joinpath(rest)
        else:
            jet = name
            ipath = basepath
        for varname in tqdm(data_vars):
            if ":" in varname:
                varname_, mode = varname.split(":")
            else:
                varname_, mode = varname, ""
            grad = mode == "grad"
            suffix1 = ":grad" if grad else ""
            suffix2 = "_grad" if grad else ""
            if varname_ == "PV":
                varname = f"PV330{suffix1}" if jet == "EDJ" else f"PV350{suffix1}"
            factor = FACTORS_UNITS.get(varname_.replace("any", "").rstrip("0123456789"), 1)
            ofile = Path(odir, f"{name}_{varname_}:anom{suffix2}.nc")
            ofile_pvals = Path(odir, f"{name}_{varname_}:anom{suffix2}_pvals.nc")
            ofile_clim_sm = Path(odir, f"{name}_{varname_}:clim{suffix2}.nc")
            if ofile.is_file():
                continue
            to_plot, pvals, clim_sm = create_relative_plot(
                varname, ipath, jet, spells, season, n_bootstraps, factor
            )
            to_plot.to_netcdf(ofile)
            pvals.to_netcdf(ofile_pvals)
            clim_sm.to_netcdf(ofile_clim_sm)


def create_relative_diff_plot(
    varname: str,
    basepath: Path,
    jet: str,
    spells_list: dict[str, pl.DataFrame],
    season: pl.Series,
    factor: float,
    phat: bool = False,
):
    season_doy = season.dt.ordinal_day().unique()
    clims_sm = []
    clims_std_sm = []
    anom_diff = []
    for run, spells in spells_list.items():
        varname_, df, clim, clim_sm, clim_std_sm, props = common_relative_plot(varname, basepath.joinpath(run), jet, season, phat, and_std=True)
        clims_sm.append(clim_sm)
        clims_std_sm.append(clim_std_sm)
        if phat:
            during_ = (
                spells
                .lazy()
                .join(df, on="time")
                .filter(pl.col("jet") == jet)
            )
        else:
            during_ = (
                spells
                .lazy()
                .join(df, on=["time", "jet ID"])
            )
        aggs = {
            varname_: pl.col(varname_).mean(),
            f"{varname_}_std": pl.col(varname_).std(),
            "ddof": pl.col(varname_).len(),
        }
        during_ = during_.group_by("norm_index", "n").agg(**aggs).sort("norm_index", "n")
        anom_diff.append(during_.collect())
            
    ddof_clim = len(season_doy) * df.select(pl.col("time").dt.year().n_unique()).collect()["time"].item()

    pvals_clim = ttest_ind_from_stats(
        clims_sm[0][varname_],
        clims_std_sm[0][varname_],
        ddof_clim,
        clims_sm[1][varname_],
        clims_std_sm[1][varname_],
        ddof_clim,
        equal_var=False,
    ).pvalue
    clim_diff = clims_sm[0].join(clims_sm[1], on=["norm_index", "n"])
    clim_diff = clim_diff.with_columns(
        **{varname_: pl.col(varname_) - pl.col(f"{varname_}_right")}
    ).drop(f"{varname_}_right")
    clim_diff = clim_diff.with_columns(pl.col(varname_) * factor)
    clim_diff = polars_to_xarray(clim_diff, ["norm_index", "n"]).T
    pvals_clim = clim_diff.copy(data=pvals_clim.reshape(clim_diff.T.shape).T)
    
    pvals = ttest_ind_from_stats(
        anom_diff[0][varname_].to_numpy(),
        anom_diff[0][f"{varname_}_std"].to_numpy(),
        anom_diff[0]["ddof"].to_numpy(),
        anom_diff[1][varname_].to_numpy(),
        anom_diff[1][f"{varname_}_std"].to_numpy(),
        anom_diff[1]["ddof"].to_numpy(),
        equal_var=False,
    ).pvalue
    anom_diff[0] = anom_diff[0].drop(f"{varname_}_std", "ddof")
    anom_diff[1] = anom_diff[1].drop(f"{varname_}_std", "ddof")
    anom_diff = anom_diff[0].join(anom_diff[1], on=["norm_index", "n"])
    anom_diff = anom_diff.with_columns(
        **{varname_: pl.col(varname_) - pl.col(f"{varname_}_right")}
    ).drop(f"{varname_}_right")
    anom_diff = anom_diff.with_columns(pl.col(varname_) * factor)
    anom_diff = polars_to_xarray(anom_diff, ["norm_index", "n"]).T
    pvals = anom_diff.copy(data=pvals.reshape(anom_diff.T.shape).T)
    return clim_diff, pvals_clim, anom_diff, pvals


def create_all_relative_diff_plots(
    spells_list: dict[str, pl.DataFrame],
    data_vars: list[str],
    runs: list[str],
    basepath: Path,
    odir: Path,
    season: pl.DataFrame | None = None,
) -> None:
    for jet in ["STJ", "EDJ"]:
        sub_spells_list = {run: spells_list[f"{run}_{jet}"] for run in runs}
        for varname in tqdm(data_vars):
            if ":" in varname:
                varname_, mode = varname.split(":")
            else:
                varname_, mode = varname, ""
            grad = mode == "grad"
            suffix1 = ":grad" if grad else ""
            suffix2 = "_grad" if grad else ""
            if varname_ == "PV":
                varname = f"PV330{suffix1}" if jet == "EDJ" else f"PV350{suffix1}"
            factor = FACTORS_UNITS.get(varname_.replace("any", "").rstrip("0123456789"), 1)
            ofile_clim_diff = Path(odir, f"{jet}_{varname_}:clim{suffix2}.nc")
            ofile_clim_diff_pvals = Path(odir, f"{jet}_{varname_}:clim{suffix2}_pvals.nc")
            ofile = Path(odir, f"{jet}_{varname_}:anom{suffix2}.nc")
            ofile_pvals = Path(odir, f"{jet}_{varname_}:anom{suffix2}_pvals.nc")
            if ofile.is_file():
                continue
            clim_diff, pvals_clim, anom_diff, pvals = create_relative_diff_plot(
                varname, basepath, jet, sub_spells_list, season, factor
            )
            clim_diff.to_netcdf(ofile_clim_diff)
            pvals_clim.to_netcdf(ofile_clim_diff_pvals)
            anom_diff.to_netcdf(ofile)
            pvals.to_netcdf(ofile_pvals)


def prepare_last_step_1(
    basepath: Path,
    filters_for_variables: dict[str, list[str]],
    props: pl.DataFrame,
    reduce_dict: dict[str, Callable] | None = None,
) -> pl.DataFrame:
    if reduce_dict is None:
        reduce_dict = {key: pl.mean for key in filters_for_variables}
    index_columns = get_index_columns(props)
    which_jet = "jet" if "jet" in props.columns else "jet ID"

    cold = pl.col("n") >= 0
    warm = pl.col("n") <= 0
    reduced = pl.col("n").abs() <= 1e6
    entrance = pl.col("norm_index") <= 0.5
    exit_ = pl.col("norm_index") >= 0.5
    all_region_filters = {
        "cold": [cold, reduced],
        "warm": [warm, reduced],
        "cold_entrance": [cold, entrance, reduced],
        "warm_entrance": [warm, entrance, reduced],
        "cold_exit": [cold, exit_, reduced],
        "warm_exit": [warm, exit_, reduced],
        "core": [pl.col("n").abs() <= 5e5],
        "warm_far_entrance": [pl.col("n") <= -1e6, entrance],
    }

    for varname, filter_list in tqdm(filters_for_variables.items()):
        if ":" in varname:
            oname = varname
            varname, mode = varname.split(":")
        else:
            oname = varname
            mode = None
        varname_ = f"{varname}_interp"
        varname_no_number = varname.rstrip("0123456789")
        varname_no_number_ = f"{varname_no_number}_interp"
        factor = FACTORS_UNITS.get(varname_no_number, 1)
        df = pl.scan_parquet(basepath.joinpath(f"{varname}_relative.parquet"))
        if varname == "hor":
            df = df.drop("hor1_interp", "hor2_interp")
        if varname_ not in df.collect_schema().names():
            if varname_no_number_ in df.collect_schema().names():
                df = df.rename({varname_no_number_: varname_})
                print(varname_no_number, "->", varname_)
        df = df.with_columns(
            pl.col(varname_).replace([float("-inf"), float("inf"), float("nan")], None)
            * factor
        )
        if mode is not None and mode == "grad":
            df = df.with_columns()
            grad_expr = (
                (
                    central_diff(pl.col(varname_).sort_by("n"))
                    / central_diff(pl.col("n").sort())
                )
                * 1e6
            ).abs()
            df = df.with_columns(
                **{varname_: grad_expr.over("norm_index", "time", which_jet)}
            )
        reduce_func = reduce_dict.get(varname, pl.Expr.mean)
        aggs = {
            f"{oname}-{filter_name}": reduce_func(
                pl.col(varname_).filter(*all_region_filters[filter_name])
            )
            for filter_name in filter_list
        }
        this_df = df.group_by(index_columns, maintain_order=True).agg(**aggs).collect()
        props = props.join(this_df, on=index_columns)

    props = props.with_columns(
        cs.exclude(index_columns)
        .cast(pl.Float32())
        .replace([float("-inf"), float("inf"), float("nan")], None)
    )
    return props


def prepare_last_step_2(
    props_with_extras: pl.DataFrame,
    spells: pl.DataFrame,
    season: pl.Series,
    grams_wr: pl.DataFrame | None = None,
    n_bootstraps: int = 400,
):
    thejet = spells["spell_of"].mode().item()
    index_columns = get_index_columns(props_with_extras)
    which_jet = "jet ID" if "jet ID" in props_with_extras.columns else "jet"
    props_with_extras = season.to_frame().join(props_with_extras, on="time")
    # props_with_extras = compute_anomalies_pl(
    #     props_with_extras, other_index_columns=[which_jet], standardize=True
    # )

    if which_jet == "jet ID":
        spells = extend_spells_jet_ID(
            spells, props_with_extras, time_before=datetime.timedelta(days=4)
        )
        times = bs_times_with_jet_ID(spells, season, n_bootstraps, props_with_extras)
        props_with_extras = props_with_extras.with_columns(
            jet=pl.when(pl.col("is_polar") < 0.5)
            .then(pl.lit("STJ"))
            .otherwise(pl.lit("EDJ"))
        )
    else:
        spells = extend_spells(spells, time_before=datetime.timedelta(days=4))
        times = create_bootstrapped_times(spells, season, n_bootstraps)
        times = times.with_columns(jet=pl.lit(thejet))
    data_vars = [
        col
        for col in props_with_extras.columns
        if col not in index_columns and props_with_extras[col].dtype.is_numeric()
    ]
    masked = times.join(props_with_extras, on=["time", which_jet]).sort(
        "sample_index", "spell", "relative_index"
    )
    if which_jet == "jet ID":
        props_with_extras_ = average_jet_categories(props_with_extras)
    else:
        props_with_extras_ = props_with_extras
    other_jet = "STJ" if thejet == "EDJ" else "EDJ"
    masked_other = times.join(
        props_with_extras_.filter(pl.col("jet") == other_jet), on=["time"]
    ).sort("sample_index", "spell", "relative_index")
    masked = masked.join(
        masked_other, on=["sample_index", "spell", "relative_index"], suffix="-other"
    )

    time_filters = {
        "before": pl.col("relative_time") < pl.duration(days=0),
        "during": pl.col("relative_time") >= pl.duration(days=0),
    }
    aggs = {}
    data_vars_ = []
    for (name_tf, time_filter), varname, suffix in product(
        time_filters.items(), data_vars, ["", "-other"]
    ):
        name = f"{varname}.{name_tf}{suffix}"
        aggs[name] = pl.col(f"{varname}{suffix}").filter(
            time_filter
        ).mean() * FACTORS_UNITS.get(varname, 1)
        data_vars_.append(name)
    masked = masked.group_by("sample_index", "spell", maintain_order=True).agg(**aggs)
    means = masked.clone()
    aggs = {}
    for col in data_vars_:
        aggs[col] = pl.col(col).last()
        agg_pval = (pl.col(col).rank().last() - 1) / n_bootstraps
        agg_pval = 2 * pl.min_horizontal(agg_pval, 1 - agg_pval)
        aggs[f"{col}.pvals"] = agg_pval.cast(pl.Float32())
    masked = (
        masked.group_by("spell", maintain_order=True)
        .agg(**aggs)
        .cast({"spell": pl.Int32()})
    )
    masked = standardize_polars_dtypes(masked)
    mean_over_spells = masked.select(
        pl.lit(-1, pl.Int32()).alias("spell"),
        cs.exclude("spell").mean().cast(pl.Float32()),
    )
    aggs = {}
    for col in masked.columns:
        if col == "spell":
            aggs[col] = pl.lit(-2, pl.Int32)
        elif col[-5:] == "pvals":
            aggs[col] = pl.lit(0.0, pl.Float32())
        else:
            aggs[col] = pl.col(col).mean().cast(pl.Float32())
    mean_over_season = means.select(**aggs)
    aggs = {}
    for col in masked.columns:
        if col == "spell":
            aggs[col] = pl.lit(-3, pl.Int32)
        elif col[-5:] == "pvals":
            aggs[col] = pl.lit(0.0, pl.Float32())
        else:
            aggs[col] = pl.col(col).std().cast(pl.Float32())
    std_over_season = means.select(**aggs)
    masked = pl.concat([std_over_season, mean_over_season, mean_over_spells, masked])
    if grams_wr is None:
        return masked

    grams_wr = grams_wr.cast({"time": pl.Datetime("ms")})
    regime_stuff = (
        spells.join(grams_wr, on="time")
        .group_by("spell")
        .agg(
            **{
                f"regime.{when}": pl.col("winner")
                .filter(time_filter_, pl.col("winner") != 0)
                .mode()
                .first()
                .fill_null(0)
                for when, time_filter_ in time_filters.items()
            }
        )
    )
    masked = masked.join(regime_stuff, on="spell", how="left")

    early = season.dt.ordinal_day().unique().head(10)
    early = pl.col("time").dt.ordinal_day().is_in(early.implode())
    late = season.dt.ordinal_day().unique().tail(10)
    late = pl.col("time").dt.ordinal_day().is_in(late.implode())
    is_early_or_late = (
        spells.filter(pl.col("relative_index") >= 0)
        .group_by(pl.col("spell").cast(pl.Int32()))
        .agg(is_early_or_late=(early.mean() > 0.5) | (late.mean() > 0.5))
    )
    masked = masked.join(is_early_or_late, on="spell", how="left")
    return masked
