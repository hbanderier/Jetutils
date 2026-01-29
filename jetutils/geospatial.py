import warnings

import numpy as np
import xarray as xr
import polars as pl
from tqdm import tqdm
from polars import DataFrame, Series, Expr
from .definitions import YEARS, RADIUS, get_index_columns, xarray_to_polars, to_expr, iterate_over_year_maybe_member


def haversine(lon1: Expr | str, lat1: Expr | str, lon2: Expr | str, lat2: Expr | str) -> Expr:
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
        angle = (pl.arctan2(pl.col("v"), pl.col("u")).interpolate("linear") + np.pi / 2) % np.pi
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
        lat.sin() * arc_distances.cos() + 
        lat.cos() * arc_distances.sin() * angle.sin()
    ).arcsin()
    normallon = lon + pl.arctan2(
        angle.cos() * arc_distances.sin() * lat.cos(), 
        arc_distances.cos() - lat.sin() * normallat.sin()
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
        ]
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
    jets = gather_normal_da_jets(jets, da, half_length=half_length, dn=dn, in_meters=in_meters)
    varname = da.name + "_interp"

    jets = interp_jets_to_zero_one(jets, [varname, "is_polar"], n_interp=n_interp)
    
    if grad:
        expr = central_diff(pl.col(varname).sort_by("n")) / central_diff(pl.col("n").sort())
        possible_cols = get_index_columns(jets, ["time", "sample_index"])
        jets = jets.filter(pl.col("n").len().over([*possible_cols, "jet ID", "norm_index"]) > 2)
        jets = jets.with_columns(**{varname: expr.over([*possible_cols, "jet ID", "norm_index"])})
        
    if clim is None:
        return jets
    if grad and clim_std is not None:
        print("Grad and clim_std, not possible")
        raise NotImplementedError
    clim = xarray_to_polars(clim)
    jets = jets.with_columns(
        dayofyear=pl.col("time").dt.ordinal_day(), is_polar=pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5
    ).cast({"n": pl.Float64})
    if grad:
        clim = clim.with_columns(**{varname: expr.over(["dayofyear", "is_polar", "norm_index"])})
    jets = (
        jets
        .join(clim, on=["dayofyear", "is_polar", "norm_index", "n"])
        .with_columns(pl.col(varname) - pl.col(f"{varname}_right"))
        .drop(f"{varname}_right")
    )
    if clim_std is None:
        return jets.drop("dayofyear")
    clim_std = xarray_to_polars(clim_std)
    jets = (
        jets
        .join(clim_std, on=["dayofyear", "is_polar", "norm_index", "n"])
        .with_columns(pl.col(varname) / pl.col(f"{varname}_right"))
        .drop(f"{varname}_right", "dayofyear")
    )
    return jets


# def gather_normal_da_jets_wrapper_wrapper(
#     jets: pl.DataFrame,
#     times: pl.DataFrame,
#     da: xr.DataArray,
#     all_times: pl.Series | None = None,
#     half_length: float = 20,
#     dn: float = 1,
#     n_interp: int = 30,
#     n_bootstraps: int = 0,
#     clim: xr.DataArray | None = None,
#     clim_std: xr.DataArray | None = None,
#     grad: bool = False,
#     time_before: datetime.timedelta = datetime.timedelta(0),
#     time_after: datetime.timedelta = datetime.timedelta(0),
# ):
#     times = extend_spells(times, time_before=time_before, time_after=time_after)
#     varname = da.name + "_interp"
#     if not n_bootstraps:
#         jets = _gather_normal_da_jets_wrapper(jets, times, da, n_interp, clim=clim, clim_std=clim_std, half_length=half_length, dn=dn, grad=grad)
#         jets = jets.group_by(
#             [pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5, "norm_index", "n"], maintain_order=True
#         ).agg(pl.col(varname).mean())
#         return polars_to_xarray(jets, index_columns=["is_polar", "norm_index", "n"])
#     if all_times is None:
#         all_times = jets["time"].unique().clone()
#     ts_bootstrapped = create_bootstrapped_times(times, all_times, n_bootstraps=n_bootstraps)
#     jets = _gather_normal_da_jets_wrapper(
#         jets, ts_bootstrapped, da, n_interp=n_interp, clim=clim, clim_std=clim_std, grad=grad
#     )
        
#     jets = (
#         jets.group_by(
#             ["sample_index", pl.col("is_polar"), "norm_index", "n"],
#         )
#         .agg(pl.col(varname).mean())
#         .sort("sample_index", "is_polar", "norm_index", "n")
#     )
#     jets = (
#         jets[["sample_index"]]
#         .unique()
#         .join(jets[["is_polar"]].unique(), how="cross")
#         .join(jets[["norm_index"]].unique(), how="cross")
#         .join(jets[["n"]].unique(), how="cross")
#         .sort("sample_index", "is_polar", "norm_index", "n")
#         .join(jets, on=["sample_index", "is_polar", "norm_index", "n"], how="left")
#     )
#     pvals = jets.group_by(
#         ["is_polar", "norm_index", "n"], maintain_order=True
#     ).agg((pl.col(varname).rank().last() - 1) / n_bootstraps)
#     jets = jets.filter(pl.col("sample_index") == n_bootstraps).drop("sample_index")
#     jets = jets.with_columns(pvals=pvals[varname])
#     return polars_to_xarray(jets, index_columns=["is_polar", "norm_index", "n"])


def expand_jets(jets: DataFrame, max_t: float, dt: float) -> DataFrame:
    """
    Expands the jets by appending segments before the start and after the end, following the tangent angle at the start and the end of the original jet, respectively.

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


def create_jet_relative_dataset(jets, path, da, suffix="", half_length: float = 2e6, dn: float = 5e4, n_interp: int = 40, in_meters: bool = True):
    jets = jets.with_columns(pl.col("time").dt.round("1d"))
    jets = jets.with_columns(jets.group_by("time", maintain_order=True).agg(pl.col("jet ID").rle_id())["jet ID"].explode())
    indexer = iterate_over_year_maybe_member(jets, da)
    to_average = []
    varname = da.name + "_interp"
    for idx1, idx2 in tqdm(indexer, total=len(YEARS)):
        jets_ = jets.filter(*idx1)
        da_ = da.sel(**idx2)
        try:
            jets_with_interp = gather_normal_da_jets(jets_, da_, half_length=half_length, dn=dn, in_meters=in_meters)
        except (KeyError, ValueError) as e:
            print(e)
            break
        jets_with_interp = interp_jets_to_zero_one(jets_with_interp, [varname, "is_polar"], n_interp=n_interp)
        jets_with_interp = jets_with_interp.group_by("time", pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5, "norm_index", "n", maintain_order=True).agg(pl.col(varname).mean())
        to_average.append(jets_with_interp)
    (
        pl
        .concat(to_average)
        .cast({"norm_index": pl.Float32(), "n": pl.Float32(), varname: pl.Float32()})
        .write_parquet(path.joinpath(f"{da.name}{suffix}_relative.parquet"))
    )
    return


def compute_relative_clim(df: pl.DataFrame, varname: str): 
    return df.group_by(pl.col("time").dt.ordinal_day().alias("dayofyear"), "norm_index", "n", "jet ID").agg(pl.col(f"{varname}_interp").mean()).sort("jet ID", "dayofyear", "norm_index", "n")


def compute_relative_std(df: pl.DataFrame, varname: str): 
    return df.group_by(pl.col("time").dt.ordinal_day().alias("dayofyear"), "norm_index", "n", "jet ID").agg(pl.col(f"{varname}_interp").std()).sort("jet ID", "dayofyear", "norm_index", "n")


def compute_relative_sm(clim: pl.DataFrame, varname: str, season_doy: pl.Series | None = None): 
    return clim.with_columns(**{f"{varname}_interp": pl.col(f"{varname}_interp").filter(pl.col("dayofyear").is_in(season_doy.implode())).mean().over("jet ID", "n", "norm_index")})


def compute_relative_anom(df: pl.DataFrame, varname: str, clim: pl.DataFrame, clim_std: pl.DataFrame | None = None): 
    varname_ = f"{varname}_interp"
    if clim_std is None:
        return (
            df
            .with_columns(dayofyear=pl.col("time").dt.ordinal_day())
            .join(clim, on=["jet ID", "dayofyear", "norm_index", "n"])
            .with_columns(pl.col(varname_) - pl.col(f"{varname_}_right"))
            .drop(f"{varname_}_right", "dayofyear")
        )
    return (
        df
        .with_columns(dayofyear=pl.col("time").dt.ordinal_day())
        .join(clim, on=["jet ID", "dayofyear", "norm_index", "n"])
        .with_columns(pl.col(varname_) - pl.col(f"{varname_}_right"))
        .drop(f"{varname_}_right")
        .join(clim_std, on=["jet ID", "dayofyear", "norm_index", "n"])
        .with_columns(pl.col(varname_) / pl.col(f"{varname_}_right"))
        .drop(f"{varname_}_right", "dayofyear")
    )