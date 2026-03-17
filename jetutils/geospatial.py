from numba.core.tracing import event
import warnings
from itertools import product
from typing import Literal
from multiprocessing import get_context

import numpy as np
import polars as pl
import polars.selectors as cs
import polars_st as st
import xarray as xr
from contourpy import contour_generator
from numba import njit
from polars import DataFrame, Expr, Series
from tqdm import tqdm

from .definitions import (
    JJADOYS,
    RADIUS,
    YEARS,
    get_index_columns,
    iterate_over_year_maybe_member,
    map_maybe_parallel,
    to_expr,
    xarray_to_polars,
    compute,
    circular_mean,
    weighted_mean_pl,
)


def euclidean_geographic(
    lon1: Expr | str, lat1: Expr | str, lon2: Expr | str, lat2: Expr | str
) -> Expr:
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


def difflon():
    expr = pl.col("lon").diff().abs()
    expr = pl.when(expr > 180).then(360 - expr).otherwise(expr)
    return expr


def signed_difflon():
    expr = pl.col("lon").diff()
    expr = pl.when(expr.abs() > 180).then((360 - expr.abs()) * expr.sign()).otherwise(expr)
    return expr


def diff_exp():
    expr = difflon()
    return (expr.abs() + pl.col("lat").diff().abs()).fill_null(10.0)


def newindex():
    return (pl.col("index").cast(pl.Int32()) - diff_exp().arg_max()) % pl.col("index").max()


def sort_by_index(df, index_columns, other):
    return (
        df.with_columns(index=pl.int_range(0, pl.len()).over([*index_columns, other]))
        .unique([*index_columns, other, "index"])
        .sort([*index_columns, other, "index"])
    )


def sort_by_difflon(df, index_columns, other):
    return (
        df.with_columns(diff_exp().over([*index_columns, other]))
        .unique([*index_columns, other, "index"])
        .sort([*index_columns, other, "index"])
    )


def sort_by_newindex(df, index_columns, other):
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


def inner_detect_contours(args):
    """
    Worker function to compute the zero-sigma-contours in a parallel context
    """
    da, levels = args
    lon = da.lon.values
    lat = da.lat.values
    z = da.values
    l1 = contour_generator(
        lon, lat, z, line_type="SeparateCode", quad_as_tri=False
    ).multi_lines(levels)
    to_ret = [
        (i, level, round_contour(contour, lon, lat), 79 in types_)
        for level, (contours, types) in zip(levels, l1)
        for contour, types_, i in zip(contours, types, range(10000000))
        if len(contours) > 5
    ]
    return tuple(zip(*to_ret))


def detect_contours(
    da: xr.DataArray,
    levels: list[float],
    repeat_lons: int = 120,
    processes: int = 1,
    ctx: str | None = None,
) -> DataFrame:
    """
    Potentially parallel wrapper around `inner_detect_contours`. Finds all zero-sigma-contours in all timesteps of `df`. Extend in longitude to capture the contours over the -180 line
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

    extra_dims = {dim: da[dim] for dim in da.dims if dim not in ["lon", "lat"]}
    index_columns = list(extra_dims)
    extra_dims_values = {key: val.values for key, val in extra_dims.items()}
    iter1 = list(product(*list(extra_dims.values())))
    iter2 = list((dict(zip(extra_dims, stuff)) for stuff in iter1))
    iter3 = ((da.loc[indexer], levels) for indexer in iter2)
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
    index_columns = [*index_columns, "level", "contour"]
    all_contours = (
        pl.DataFrame(
            {
                "contour": all_is,
                "level": all_levels,
                "contours": all_contours,
                "cyclic": all_cyclics,
            }
            | extra_dims_values
        )
        .explode("contour", "level", "contours", "cyclic")
        .explode("contours")
        .with_columns(
            lon=pl.col("contours").arr.get(0),
            lat=pl.col("contours").arr.get(1),
        )
        .drop("contours")
        .unique([*index_columns, "lon", pl.col("lat").round(2)], maintain_order=True)
        .with_columns(side=(pl.col("lon") >= 180.0).cast(pl.UInt8()))
        .filter(~(pl.col("side") == 1).all().over(index_columns))
        .with_columns(len=pl.len().over(index_columns))
        .filter(pl.col("len") > 10)
    )

    # filter intersections of fully zeros within nonzero
    lon_wrapped = (pl.col("lon") + 180) % 360 - 180
    points = pl.concat_str(
        lon_wrapped.cast(pl.Int32()), pl.col("lat").cast(pl.Int32()), separator=" "
    )
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
    to_drop = (
        df.join(df, on=["time", "level"], how="full")
        .drop("time_right", "level_right")
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


def gather_normal_da_jets_wrapper(
    jets: pl.DataFrame,
    times: pl.DataFrame,
    da: xr.DataArray,
    n_interp: int = 30,
    clim: xr.DataArray | None = None,
    clim_std: xr.DataArray | None = None,
    half_length: float = 1.2e6,
    dn: float = 5e4,
    in_meters: bool = False,
    grad: bool = False,
):
    jets = times.join(jets, on="time", how="left")
    jets = gather_normal_da_jets(
        jets, da, half_length=half_length, dn=dn, in_meters=in_meters
    )
    varname = da.name + "_interp"

    jets = interp_jets_to_zero_one(jets, [varname, "is_polar"], n_interp=n_interp)

    if grad:
        expr = central_diff(pl.col(varname).sort_by("n")) / central_diff(
            pl.col("n").sort()
        )
        possible_cols = get_index_columns(jets, ["time", "sample_index"])
        jets = jets.filter(
            pl.col("n").len().over([*possible_cols, "jet ID", "norm_index"]) > 2
        )
        jets = jets.with_columns(
            **{varname: expr.over([*possible_cols, "jet ID", "norm_index"])}
        )

    if clim is None:
        return jets
    if grad and clim_std is not None:
        print("Grad and clim_std, not possible")
        raise NotImplementedError
    clim = xarray_to_polars(clim)
    jets = jets.with_columns(
        dayofyear=pl.col("time").dt.ordinal_day(),
        is_polar=pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5,
    ).cast({"n": pl.Float64})
    if grad:
        clim = clim.with_columns(
            **{varname: expr.over(["dayofyear", "is_polar", "norm_index"])}
        )
    jets = (
        jets.join(clim, on=["dayofyear", "is_polar", "norm_index", "n"])
        .with_columns(pl.col(varname) - pl.col(f"{varname}_right"))
        .drop(f"{varname}_right")
    )
    if clim_std is None:
        return jets.drop("dayofyear")
    clim_std = xarray_to_polars(clim_std)
    jets = (
        jets.join(clim_std, on=["dayofyear", "is_polar", "norm_index", "n"])
        .with_columns(pl.col(varname) / pl.col(f"{varname}_right"))
        .drop(f"{varname}_right", "dayofyear")
    )
    return jets


def event_geometry(
    events: pl.DataFrame,
    mode: Literal["envelope", "convex_hull", "polygon"] = "envelope",
    index_columns: list[str] | None = None,
):
    if index_columns is None:
        index_columns = get_index_columns(events, ["member", "time", "level", "index"])
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
    contours,
    max_difflon: float = 5,
    min_lon_ext: float = 5,
    min_lat_ext: float = 5,
    min_len: int = 5,
):
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
    
    index_columns = get_index_columns(contours, ["member", "time", "level", "contour"])
    
    subindex = (difflon() > max_difflon).fill_null(False).cum_sum()
    newindex = (pl.int_range(pl.len()) - difflon().arg_max()) % pl.len()
    newindex = (
        pl.when(difflon().max() > max_difflon)
        .then(newindex)
        .otherwise(pl.int_range(pl.len()))
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
    )

    index = pl.concat_arr(
        pl.col("contour").cast(pl.UInt32()), pl.col("subindex").cast(pl.UInt32())
    ).rle_id()
    index_columns.remove("contour")
    index = index.over(index_columns)
    overturnings = overturnings.with_columns(index=index).drop("contour", "subindex")
    overturnings = overturnings.with_columns(inside_index=newindex.over(index_columns)).sort([*index_columns, "inside_index"])
    index_columns.append("index")
    overturnings = overturnings.with_columns(inside_index=newindex.over(index_columns)).sort([*index_columns, "inside_index"])
    
    backward = signed_difflon().sum() < 0
    inside_index = pl.when(backward).then(pl.col("inside_index").reverse()).otherwise("inside_index")
    overturnings = overturnings.with_columns(inside_index=inside_index.over(index_columns))
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
    Broken for now, can't repair it now

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
    index_columns = get_index_columns(contours, ["member", "time", "level", "contour"])

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
    forward = dist_forward < dist_backward

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
        )
        .filter(
            pl.col("dist1") < max_realdist,
            pl.col("dist2") > min_contourdist,
            pl.col("dist2") < max_contourdist,
            pl.col("dist2") / pl.col("dist1") > min_ratio,
            pl.col("forward") | (~pl.col("cyclic")),
        )
        .collect(streaming=True) 
    )

    max_right = pl.col("index_right") == pl.col("index_right").max().over(
        [*index_columns, "index"]
    )
    min_right = pl.col("index_right") == pl.col("index_right").min().over(
        [*index_columns, "index"]
    )
    max_right = pl.when("forward").then(max_right).otherwise(min_right)

    min_left = pl.col("index") == pl.col("index").min().over(
        [*index_columns, "index_right"]
    )
    max_left = pl.col("index") == pl.col("index").max().over(
        [*index_columns, "index_right"]
    )
    min_left = pl.when("forward").then(min_left).otherwise(max_left)

    range_ = pl.int_ranges(pl.col("index"), pl.col("index_right") + 1)
    other_range = pl.int_ranges(
        pl.col("index_right"), pl.col("index") + 1 + pl.col("max_n")
    ) % pl.col("max_n")
    range_ = pl.when("forward").then(range_).otherwise(other_range)

    streamers = (
        streamers[*index_columns, "index", "index_right", "forward", "max_n"]
        .filter(max_right)
        .filter(min_left)
        .with_columns(range=range_)
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

    subindex = (
        pl.concat_arr(
            pl.col("index").cast(pl.UInt32()),
            pl.col("index_right").cast(pl.UInt32()),
            pl.col("forward").cast(pl.UInt32()),
        )
        .rle_id()
        .over(index_columns)
    )

    streamers = (
        streamers.join(
            to_drop,
            on=[*index_columns, "index", "index_right"],
            how="left",
        )
        .filter(~pl.col("drop").fill_null(False))
        .drop("drop")
        .with_columns(subindex=subindex)
    )

    l1 = pl.col("range").list.len()
    l2 = pl.col("range_other").list.len()
    linter = pl.col("range").list.set_intersection(
        pl.col("range_other")
    ).list.len() / pl.min_horizontal(l1, l2)

    index = (
        pl.when("forward").then(pl.col("index").min()).otherwise(pl.col("index").max())
    )
    index_right = (
        pl.when("forward")
        .then(pl.col("index_right").max())
        .otherwise(pl.col("index_right").min())
    )

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
        .group_by(*index_columns, "index", "index_right", "subindex", "forward")
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
        .with_columns(subindex=subindex)
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
        streamers.with_columns(
            subindex=pl.any_horizontal(
                pl.col("index").diff().abs() > 10,
                pl.col("index_right").diff().abs() > 10,
                pl.col("forward").cast(pl.Int8()).diff().abs() > 0,
            )
            .fill_null(False)
            .over(index_columns)
            .cum_sum()
            .over(index_columns)
        )
        .explode("range")
        .unique([*index_columns, "range"])
        .group_by([*index_columns, "forward", "subindex"])
        .agg(
            pl.col("max_n").first(),
            index=index.first(),
            index_right=index_right.first(),
        )
        .with_columns(range=range_)
        .sort(*index_columns, "forward", "subindex")
        .with_columns(subindex=pl.col("subindex").rle_id().over(index_columns))
        .drop("index", "index_right", "max_n")
        .explode("range")
        .with_columns(cs.signed_integer().cast(pl.Int32()))
        .rename({"range": "index"})
        .join(
            contours.drop("s", "len", "max_s", "max_n", "cyclic"),
            on=[*index_columns, "index"],
        )
        .drop("index")
    )

    phys_len = jet_integral_haversine("lon", "lat", x_is_one=True)
    phys_len = phys_len.over([*index_columns, "subindex"])
    index = pl.concat_arr(
        pl.col("contour").cast(pl.UInt32()), pl.col("subindex").cast(pl.UInt32())
    ).rle_id()
    index_columns.remove("contour")
    index = index.over(index_columns)
    streamers = streamers.with_columns(phys_len=phys_len, index=index)
    index_columns.append("index")
    return event_geometry(streamers, "polygon", index_columns=index_columns)


def sjoin_to_grid(
    events: pl.DataFrame,
    da: xr.DataArray,
    varname: str = "ones",
    buffer: float | None = None,
):
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

    index_columns = get_index_columns(events, ["member", "time", "level", "index"])
    events = events.drop("points").with_columns(pl.col("geometry").st.buffer(buffer))
    if varname == "ones":
        events = events.with_columns(ones=pl.lit(1))
    events = events.cast({varname: dtype})
    events = events.st.sjoin(
        da_df, on="geometry", how="inner", predicate="within"
    ).drop("geometry_right")
    events = events.sort(*index_columns, "lat", "lon")
    return events


def to_xarray_sjoin(
    da: xr.DataArray,
    events: pl.DataFrame | None = None,
    events_on_grid: pl.DataFrame | None = None,
    varname: str = "ones",
    buffer: float | None = None,
):
    if events_on_grid is None:
        events_on_grid = sjoin_to_grid(events, da, varname, buffer)
    index_columns = get_index_columns(events_on_grid, ["member", "time", "level"])
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


def event_props(events: pl.DataFrame, das: list[xr.DataArray]):
    index_columns = get_index_columns(events, ["member", "time", "level", "index"])
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
    com_x = circular_mean(pl.col("lat"), "cell_area").cast(pl.Float32())
    com_y = weighted_mean_pl(pl.col("lon"), "cell_area").cast(pl.Float32())
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
    da: xr.DataArray,
    half_length: float = 1.2e6,
    dn: float = 5e4,
    delete_middle: bool = False,
    in_meters: bool = False,
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
            "sample_index",
            "inside_index",
        ),
    )
    if in_meters:
        jets = add_normals_meters(jets, half_length, dn, delete_middle)
    else:
        jets = add_normals(jets, half_length, dn, delete_middle)
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
    return jets


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


def create_jet_relative_dataset(
    jets,
    path,
    da,
    suffix="",
    half_length: float = 2e6,
    dn: float = 5e4,
    n_interp: int = 40,
    in_meters: bool = True,
):
    jets = jets.with_columns(pl.col("time").dt.round("1d"))
    jets = jets.with_columns(
        jets.group_by("time", maintain_order=True)
        .agg(pl.col("jet ID").rle_id())["jet ID"]
        .explode()
    )
    indexer = iterate_over_year_maybe_member(jets, da)
    to_average = []
    varname = da.name + "_interp"
    for idx1, idx2 in tqdm(indexer, total=len(YEARS)):
        jets_ = jets.filter(*idx1)
        da_ = da.sel(**idx2)
        try:
            jets_with_interp = gather_normal_da_jets(
                jets_, da_, half_length=half_length, dn=dn, in_meters=in_meters
            )
        except (KeyError, ValueError) as e:
            print(e)
            break
        jets_with_interp = interp_jets_to_zero_one(
            jets_with_interp, [varname, "is_polar"], n_interp=n_interp
        )
        jets_with_interp = jets_with_interp.group_by(
            "time",
            pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5,
            "norm_index",
            "n",
            maintain_order=True,
        ).agg(pl.col(varname).mean())
        to_average.append(jets_with_interp)
    (
        pl.concat(to_average)
        .cast({"norm_index": pl.Float32(), "n": pl.Float32(), varname: pl.Float32()})
        .write_parquet(path.joinpath(f"{da.name}{suffix}_relative.parquet"))
    )
    return


def compute_relative_clim(df: pl.DataFrame | pl.LazyFrame, varname: str):
    return (
        df.group_by(
            pl.col("time").dt.ordinal_day().alias("dayofyear"),
            "norm_index",
            "n",
            "jet ID",
        )
        .agg(pl.col(f"{varname}_interp").mean())
        .sort("jet ID", "dayofyear", "norm_index", "n")
    )


def compute_relative_std(df: pl.DataFrame | pl.LazyFrame, varname: str):
    return (
        df.group_by(
            pl.col("time").dt.ordinal_day().alias("dayofyear"),
            "norm_index",
            "n",
            "jet ID",
        )
        .agg(pl.col(f"{varname}_interp").std())
        .sort("jet ID", "dayofyear", "norm_index", "n")
    )


def compute_relative_sm(
    clim: pl.DataFrame | pl.LazyFrame, varname: str, season_doy: pl.Series | None = None
):
    if season_doy is None:
        season_doy = JJADOYS
    return clim.with_columns(
        **{
            f"{varname}_interp": pl.col(f"{varname}_interp")
            .filter(pl.col("dayofyear").is_in(season_doy.implode()))
            .mean()
            .over("jet ID", "n", "norm_index")
        }
    )


def compute_relative_anom(
    df: pl.DataFrame | pl.LazyFrame,
    varname: str,
    clim: pl.DataFrame | pl.LazyFrame,
    clim_std: pl.DataFrame | None = None,
):
    varname_ = f"{varname}_interp"
    if clim_std is None:
        return (
            df.with_columns(dayofyear=pl.col("time").dt.ordinal_day())
            .join(clim, on=["jet ID", "dayofyear", "norm_index", "n"])
            .with_columns(pl.col(varname_) - pl.col(f"{varname_}_right"))
            .drop(f"{varname_}_right", "dayofyear")
        )
    return (
        df.with_columns(dayofyear=pl.col("time").dt.ordinal_day())
        .join(clim, on=["jet ID", "dayofyear", "norm_index", "n"])
        .with_columns(pl.col(varname_) - pl.col(f"{varname_}_right"))
        .drop(f"{varname_}_right")
        .join(clim_std, on=["jet ID", "dayofyear", "norm_index", "n"])
        .with_columns(pl.col(varname_) / pl.col(f"{varname_}_right"))
        .drop(f"{varname_}_right", "dayofyear")
    )
