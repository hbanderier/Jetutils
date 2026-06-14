# coding: utf-8
"""
This probably too big module contains all the utilities relative to jet extraction from 2D fields, jet tracking, jet categorization and jet properties. All of the functions are wrapped by the convenience class `JetFindingExperiment`.
"""
from numpy.typing import NDArray
import datetime
from itertools import product
from pathlib import Path
from typing import Literal, Sequence, Callable

import numpy as np
import polars as pl
import polars_ds as pds
import polars_st as st
import rustworkx as rx
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splev, splprep
from polars import DataFrame, Expr, Series
from polars.exceptions import ColumnNotFoundError
from scipy.linalg import sqrtm
from sklearn.mixture import GaussianMixture
from tqdm import tqdm, trange

from .anyspell import get_spells
from .data import (
    DataHandler,
    coarsen_da,
    compute_extreme_climatology,
    open_da,
    open_dataarray,
    smooth,
    to_netcdf,
    standardize_polars_dtypes, 
    standardize
)
from .definitions import (
    N_WORKERS,
    RADIUS,
    compute,
    do_rle,
    explode_rle,
    extract_season_from_df,
    get_index_columns,
    iterate_over_year_maybe_member,
    squarify,
    to_expr,
    xarray_to_polars,
    polars_to_xarray,
    circular_mean,
    weighted_mean_pl
)
from .derived_quantities import compute_norm_derivative
from .geospatial import (
    compute_alignment,
    detect_contours,
    detect_contours_lonlat,
    diff_exp,
    gather_normal_da_jets,
    haversine,
    jet_integral_haversine,
    join_wrapper,
    sort_by_index_then_difflon,
    sort_by_index_then_newindex, 
    nearest_mapping,
    create_jet_relative_dataset, 
    bias_correct,
    create_bias_correction,
)


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
    return (to_expr(col) * factor).round() / factor


def window_smooth_func(da: xr.DataArray, win_size: int = 5):
    smooth_map = ("win", win_size)
    smooth_map = {"lon": smooth_map, "lat": smooth_map}
    return smooth(da, smooth_map)


def gaussian_smooth_func(da: xr.DataArray, sigma_lon: float = 1., sigma_lat: float = 1.):
    """
    Paper-thin wrapper around `scipy.ndimage.gaussian_smooth`

    Parameters
    ----------
    da : xr.DataArray
        _description_
    sigma_lon : float, optional
        _description_, by default 1.
    sigma_lat : float, optional
        _description_, by default 1.

    Returns
    -------
    _type_
        _description_
    """
    sigmas = tuple(
        [0] * len([dim for dim in da.dims if dim not in ["lon", "lat"]])
    )
    sigmas = sigmas + (sigma_lat, sigma_lon)
    return gaussian_filter(da.values, sigmas)


def join_on_ds(jets, ds: xr.Dataset | xr.DataArray | pl.DataFrame, fix_id: bool = False):
    """
    Wrapper around `.join`, `neareast_mapping` and potentially `xarray_to_polars` if you provided a `xarray` object. For now we're doing nearest interpolation but we want to support bilinear interpolation in the future too, since we already have the code for it in `geospatial.interp_from_other`. 

    Parameters
    ----------
    jets : _type_
        _description_
    ds : xr.Dataset | xr.DataArray | pl.DataFrame
        _description_
    fix_id : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    index_columns = get_index_columns(jets, ["member", "number", "cluster", "time"])
    other_indexer = get_index_columns(jets, ["jet", "jet ID", "contour", "level"])
    if isinstance(ds, xr.DataArray | xr.Dataset):
        ds = xarray_to_polars(ds)
    lo = nearest_mapping(jets, ds, "lon")
    la = nearest_mapping(jets, ds, "lat")
    
    jets = (
        jets
        .join(lo, on="lon")
        .drop("lon")
        .rename({"lon_": "lon"})
        .join(la, on="lat")
        .drop("lat")
        .rename({"lat_": "lat"})
        .unique([*index_columns, *other_indexer, "lon", "lat"], maintain_order=True)
        .join(ds, on=[*index_columns, "lon", "lat"], how="left")
    )
    if fix_id:
        jets = jets.with_columns(pl.col("index").rle_id().over([*index_columns, *other_indexer]))
    
    return jets


def preprocess_ds(ds: xr.Dataset, n_coarsen: int = 1, smooth_func: Callable = window_smooth_func, **smooth_kwargs):
    """
    Lego-type preprocessing, coarsens then applies `smooth_func` and also compute sigma the horizontal normal wind shear. It returns the unprocessed data too because it's useful. Extra kwargs are passed to smooth_func but you should really be building it with `functools.partial` because it's more explicit.

    Parameters
    ----------
    ds : xr.Dataset
        _description_
    n_coarsen : int, optional
        _description_, by default 1
    smooth_func : Callable, optional
        _description_, by default window_smooth_func

    Returns
    -------
    _type_
        _description_
    """
    ds = standardize(ds)
    if "s" not in ds:
        ds["s"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    ds_orig = ds.copy()
    ds = coarsen_da(ds, n_coarsen=n_coarsen, reduce_func=np.mean)
    to_smooth = ["u", "v", "s", "theta"]
    for var in to_smooth:
        ds[var] = ds[var].copy(data=smooth_func(ds[var], **smooth_kwargs))
    ds["sigma"] = compute_norm_derivative(ds, "s")
    ds["sigma"] = ds["sigma"].copy(data=smooth_func(ds["sigma"], **smooth_kwargs))
    if "sigma" not in ds_orig:
        ds_orig["sigma"] = compute_norm_derivative(ds_orig, "s")
    return ds, ds_orig


def find_all_jets(
    ds: xr.Dataset,
    n_coarsen: int = 3,
    smooth_func: Callable = window_smooth_func,
    thresholds: xr.DataArray | None = None,
    base_s_thresh: float = 0.5,
    alignment_thresh: float = 0.6,
    int_thresh_factor: float = 0.6,
    hole_size: int = 10,
    **smooth_kwargs,
):
    """
    Main function to find all jets in a polars DataFrame containing at least the "lon", "lat", "u", "v" and "s" columns. Will group by any potential index columns to compute jets independently for every, timestep, member and / or cluster. Any other non-index column present in `df` (like "theta" or "lev") will be interpolated to the jet cores in the output.

    Thresholds passed as a DataArray are wind speed thresholds. This Dataarray needs to have one value per timestep present in `df`. If not passed, `base_s_thresh` is used for all times.

    The jet integral threshold is computed from the wind speed threshold.
    """
    # process input
    ds, ds_orig = preprocess_ds(ds, n_coarsen=n_coarsen, smooth_func=smooth_func, **smooth_kwargs)
    df = xarray_to_polars(ds_orig)
    x_periodic = has_periodic_x(ds)
    index_columns = get_index_columns(
        df,
        (
            "member",
            "number",
            "forecast_init",
            "time",
            "cluster",
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
    ic_nomemb = [ic for ic in index_columns if ic not in ["member", "number"]]
    if base_s_thresh <= 1.0:
        thresholds = df.group_by(ic_nomemb).agg(
            pl.col("s").quantile(base_s_thresh).alias("s_thresh")
        )
        base_s_thresh = thresholds["s_thresh"].mean()  # disgusting
        base_int_thresh = (
            RADIUS * dl * base_s_thresh * np.cos(np.pi / 4) * int_thresh_factor
        )
    elif thresholds is not None:
        thresholds = (
            xarray_to_polars(thresholds)
            .drop("quantile")
            .cast({"s": pl.Float32})
            .rename({"s": "s_thresh"})
        )
    if thresholds is not None:
        df = df.join(
            thresholds, on=ic_nomemb
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
    repeat_lons = 120 if x_periodic else 0.
    all_contours = detect_contours_lonlat(
        (ds["sigma"] > 0).astype(np.uint8), [0.0], processes=N_WORKERS, repeat_lons=repeat_lons,
        do_round=False,
    ).drop("level")

    int_expr = jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s"))
    int_expr = int_expr.over([*index_columns, "jet ID"])
    distance_ends_expr = haversine(
        pl.col("lon").first(),
        pl.col("lat").first(),
        pl.col("lon").last(),
        pl.col("lat").last(),
    )
    distance_ends_expr = distance_ends_expr.over([*index_columns, "jet ID"])

    for index_column in index_columns:
        try:
            all_contours = all_contours.cast({index_column: df[index_column].dtype})
        except ColumnNotFoundError:
            pass
    all_contours = all_contours.join(df, on=[*index_columns, "lon", "lat"], how="left")
    all_contours = compute_alignment(all_contours, x_periodic)
    
    # jets from contours
    ## consecutive runs of contour points respecting both point wise conditions, allowing holes of size up to three
    # return all_contours, condition_expr, index_columns, hole_size
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
            .filter(
                pl.col("value").cum_sum().over(*index_columns, "contour") > 0,
                ~(
                    ~pl.col("value")
                    & (
                        pl.col("start")
                        == pl.col("start").max().over(*index_columns, "contour")
                    )
                ),
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
        sort_by_index_then_difflon(
            jets.with_columns(len=pl.len().over([*index_columns, "contour"])).filter(
                pl.col("len") > 6
            ),
            index_columns,
            "contour",
        )
        .with_columns(diff=diff_exp().over([*index_columns, "contour"]))
        .with_columns(
            contour=(pl.col("contour") + 0.01 * (pl.col("diff") > 3).cum_sum())
            .rle_id()
            .over(index_columns)
        )
        .rename({"contour": "jet ID"})
        .drop("cyclic", "len", "len_right")
    )
    jets = (
        sort_by_index_then_newindex(
            jets.with_columns(len=pl.len().over([*index_columns, "jet ID"])).filter(
                pl.col("len") > 6
            ),
            index_columns,
            "jet ID",
        )
        .with_columns(
            int=int_expr,
            distance_ends=distance_ends_expr,
        )
        .filter(condition_expr2, pl.col("distance_ends") > 1e6)
        .drop("distance_ends")
        .with_columns(pl.col("jet ID").rle_id().over([*index_columns]))
    )
    jets = sort_by_index_then_newindex(jets, index_columns, "jet ID")
    jets = standardize_polars_dtypes(jets)
    return jets


def online_bias_correct(jets: pl.DataFrame, ds: xr.Dataset) -> pl.DataFrame:
    """
    This is a fun idea but it does not work great. 
    Use the bias correction function to find how far the actual sigma contour is around the detected (biased) jet is, then send a ray there using the formulae for normallon and normallat, and define the new jet at these new points. You shoud spline-interpolate after this for good measure.
    
    For now, use biased jets and only bias correct the jet-centred composites, it's fine. Issue is of course sigma contours are super noisy, and while the smoothing works much better in jet-centred composites it still biases the position so you can't do too much of it. You're left with the same problem as in real space almost.

    Parameters
    ----------
    jets : pl.DataFrame
        _description_
    ds : xr.Dataset
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    index_columns = get_index_columns(jets)
    jets_orig = jets.clone()
    jets = bias_correct(jets, ds, same_len=False)

    arc_distances = pl.col("n").first() / RADIUS
    lon = pl.col("lon").first().radians()
    lat = pl.col("lat").first().radians()
    angle = pl.col("angle").first()

    newlat = (
        lat.sin() * arc_distances.cos() + lat.cos() * arc_distances.sin() * angle.sin()
    ).arcsin()
    newlat = newlat.degrees()

    newlon = lon + pl.arctan2(
        angle.cos() * arc_distances.sin() * lat.cos(),
        arc_distances.cos() - lat.sin() * newlat.sin(),
    )
    newlon = newlon.degrees()

    jets = (
        jets
        .group_by(*index_columns, "index", maintain_order=True)
        .agg(
            lon=newlon,
            lat=newlat,
        )
    )
    idx_new = jets.select(index_columns).unique(index_columns)
    idx_old = jets_orig.select(index_columns).unique(index_columns)
    left_behind = idx_old.join(idx_new, on=index_columns, how="anti")
    jets = pl.concat([jets, left_behind.join(jets_orig, on=index_columns).select(*jets.columns)]).sort([*index_columns, "index"])
    return jets 

def spline_smooth(jets: pl.DataFrame, s: float = 0., factor: int = 3) -> pl.DataFrame:
    """
    This works but was mostly useful in conjunction with bias_correct. 
    
    One day, polars-splines will be updated and this function will use it to be even faster instead of looping. maybe.

    Parameters
    ----------
    jets : pl.DataFrame
        _description_
    s : float, optional
        _description_, by default 0.
    factor : int, optional
        _description_, by default 3

    Returns
    -------
    pl.DataFrame
        _description_
    """
    index_columns = get_index_columns(jets)
    newjets = []
    oldjets = []
    for indexer, newjet in tqdm(jets.group_by(index_columns), disable=True):
        lo, la = newjet["lon"].to_numpy(), newjet["lat"].to_numpy()
        try:
            tck, u = splprep([lo, la], s=s)
        except ValueError:
            oldjets.append(newjet[*index_columns, "index", "lon", "lat"])
            continue
        unew = np.linspace(u.min(), u.max(), len(u) * factor)
        los, las = splev(unew, tck)
        indexer = dict(zip(index_columns, indexer))
        df = indexer | {"index": np.arange(len(unew)), "lon": los, "lat": las}
        newjets.append(pl.DataFrame(df))
    newjets = standardize_polars_dtypes(pl.concat(newjets)).cast({"jet ID": pl.UInt32()})
    if len(oldjets) > 0:
        newjets = pl.concat([newjets, pl.concat(oldjets)])
    newjets =  newjets.unique([*index_columns, "index"]).sort(*index_columns, "index")
    return newjets


def compute_jet_props(jets: DataFrame) -> DataFrame:
    """
    Computes all basic jet properties from a DataFrame containing many jets.
    """
    position_columns = [col for col in ["lat", "lev", "theta"] if col in jets.columns]

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
            .struct
            .field("r2")
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
    
    aggregations = [agg.replace([float("-inf"), float("inf"), float("nan")], None).first() for agg in aggregations]

    jets_lazy = jets.lazy()
    index_columns = get_index_columns(jets)
    if "member" not in index_columns:
        gb = jets_lazy.group_by(index_columns, maintain_order=True)
        props_as_df = gb.agg(*aggregations)
        props_as_df = props_as_df.collect()
    else:
        # streaming mode doesn't work well
        collected = []
        for member in tqdm(jets["member"].unique(maintain_order=True).to_numpy()):
            gb = jets_lazy.filter(pl.col("member") == member).group_by(
                index_columns, maintain_order=True
            )
            props_as_df = gb.agg(*aggregations)
            collected.append(props_as_df.collect())
        props_as_df = pl.concat(collected).sort("member")
    which_jet = "jet" if "jet" in index_columns else "jet ID"
    index_columns.remove(which_jet)
    unique_lon_over_europe = jets["lon"].unique()
    unique_lon_over_europe = unique_lon_over_europe.filter(unique_lon_over_europe > -10)
    dji = jets.group_by([*index_columns, "lon"]).agg(
        pl.col(which_jet).n_unique()
    )
    dji = unique_lon_over_europe.to_frame().join(dji, on="lon", how="left").group_by(index_columns).agg(double_jet_index=(pl.col(which_jet).fill_null(0) >= 2).mean())
    props_as_df = props_as_df.join(dji, on=index_columns)
    props_as_df = standardize_polars_dtypes(props_as_df)
    return props_as_df


def compute_widths(jets: DataFrame, da: xr.DataArray):
    """
    Computes the width of each jet using normally interpolated wind speed on either side of the jet.
    """
    jets = gather_normal_da_jets(jets, da, half_length=1.3e6, dn=5e4, delete_middle=True, in_meters=True)

    index_columns = get_index_columns(
        jets, ("member", "time", "cluster", "spell", "relative_index", "jet ID", "jet")
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
    feature_names: Sequence | None = None,
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
    previous: NDArray | None = None,
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
    previous: np.ndarray | None = None
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
        n_components=n_components, init_params=init_params, n_init=n_init, means_init=previous
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
    return 1 / (1 + otherscores / scores[order[-1]]), previous


def is_polar_gmix(
    jets: DataFrame,
    feature_names: list,
    mode: (
        Literal["all"] | Literal["season"] | Literal["month"] | Literal["week"]
    ) = "week",
    n_components: int | Sequence = 2,
    n_init: int = 20,
    init_params: str = "random_from_data",
    v2: bool = True,
    use_prev: bool = True,
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
    if "time" not in jets.columns:
        mode = "all"
    if mode == "all":
        X = extract_features(jets, feature_names)
        kwargs["n_components"] = n_components
        probas, _ = gmix_fn(X, **kwargs)
        return jets.with_columns(is_polar=probas)
    index_columns = get_index_columns(jets, ["member", "number", "forecast_init", "time", "jet ID", "index"])
    to_concat = []
    previous = None
    if mode == "season":
        if isinstance(n_components, int):
            n_components = [n_components] * 4
        else:
            assert len(n_components) == 4
        for season, n_components_ in zip(
            tqdm(["DJF", "MAM", "JJA", "SON"]), n_components
        ):
            X = extract_season_from_df(jets, season)
            X = X[[*index_columns, *feature_names]]
            kwargs["n_components"] = n_components_
            probas, previous = gmix_fn(X[feature_names], **kwargs, previous=previous)
            if not use_prev:
                previous = None
            to_concat.append(X.with_columns(is_polar=probas).drop(feature_names))
    elif mode == "month":
        months = jets["time"].dt.month().unique().sort().to_numpy()
        if isinstance(n_components, int):
            n_components = [n_components] * len(months)
        else:
            assert len(n_components) == len(months)
        for month, n_components_ in zip(tqdm(months), n_components):
            X = extract_season_from_df(jets, month)
            X = X[[*index_columns, *feature_names]]
            kwargs["n_components"] = n_components_
            probas, previous = gmix_fn(X[feature_names], **kwargs, previous=previous)
            if not use_prev:
                previous = None
            to_concat.append(X.with_columns(is_polar=probas).drop(feature_names))
    elif mode == "week":
        weeks = jets["time"].dt.week().unique().sort().to_numpy()
        if isinstance(n_components, int):
            n_components = [n_components] * len(weeks)
        else:
            assert len(n_components) == len(weeks)
        for week, n_components_ in zip(tqdm(weeks, total=len(weeks)), n_components):
            X = jets.filter(pl.col("time").dt.week() == week)
            X = X[[*index_columns, *feature_names]]
            kwargs["n_components"] = n_components_
            probas, previous = gmix_fn(X[feature_names], **kwargs, previous=previous)
            if not use_prev:
                previous = None
            to_concat.append(X.with_columns(is_polar=probas).drop(feature_names))

    return jets.join(pl.concat(to_concat), on=index_columns).sort(index_columns)


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


def to_one_large(jets, int_EDJ_threshold: float = 1.3e8):
    index_columns = get_index_columns(jets)
    jets = jets.filter(
        (pl.col("is_polar").mean().over(index_columns) < 0.5)
        | (
            (pl.col("is_polar").mean().over(index_columns) > 0.5)
            & (pl.col("int").mode().first().over(index_columns) > int_EDJ_threshold)
        )
    )
    jet = pl.when(pl.col("is_polar").mean().over(index_columns) > 0.5).then(pl.lit("EDJ")).otherwise(pl.lit("STJ"))
    jets = jets.with_columns(jet=jet).drop("jet ID")
    return jets


def _lons_from_points(points: str | pl.Expr = "points") -> pl.Expr:
    points = to_expr(points)
    return points.list.eval(pl.element().arr.first())


def _lon_overlap(
    points: str | pl.Expr = "points", points_right: str | pl.Expr = "points_right"
) -> pl.Expr:
    overlap = _lons_from_points(points).list.set_intersection(
        _lons_from_points(points_right)
    )
    overlap = (
        overlap.list.len()
        * (
            1 / _lons_from_points(points_right).list.len()
            + 1 / _lons_from_points(points_right).list.len()
        )
        / 2
    )
    return overlap


def _frechet_st(
    points: str | pl.Expr = "points", points_right: str | pl.Expr = "points_right"
):
    points = st.linestring(points).st.set_srid(4326).st.to_srid(3857)
    points_right = st.linestring(points_right).st.set_srid(4326).st.to_srid(3857)
    return points.st.frechet_distance(points_right)
    

def track_jets_(
    jets: DataFrame, member: str | None = None, year: int | None = None, month: int | None = None,
) -> DataFrame:
    """
    Performs jet tracking for real

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
    which_jet = "jet" if "jet" in jets.columns else "jet ID"
    typical_group_by = ["time", "time_right", which_jet, f"{which_jet}_right"]
    filter_ = pl.lit(True) 
    if year is not None:
        filter_ = filter_ & (pl.col("time").dt.year() == year)
    if month is not None:
        filter_ = filter_ & (pl.col("time").dt.month() == month)
    if year is not None and (year + 1) in jets["time"].dt.year().unique() and (month is None or month == 12):
        filter_ = filter_ | pl.col("time").is_in(
            pl.col("time")
            .filter(pl.col("time").dt.year() == year + 1)
            .unique()
            .bottom_k(1)
            .implode()
        )
    if "member" in jets.columns and member is not None:
        filter_ = filter_ & (pl.col("member") == member)
        typical_group_by.insert(0, "member")

    cols_i_want = get_index_columns(jets, ["member", "time", "jet ID", "jet", "lon", "lat"])
    jets_current = (
        jets.filter(filter_)
        .select(*cols_i_want, *deltas)
        .with_columns(
            len=pl.col("lon").len().over(["time", which_jet]),
            index=pl.int_range(0, pl.col("lon").len()).over(["time", which_jet]),
        )
    )
    dt = jets_current["time"].unique().bottom_k(2).sort()
    dt = dt[1] - dt[0]
    jets_next = jets_current.with_columns(time_shifted=pl.col("time") - dt)
    jets_current = jets_current.filter(filter_)

    aggs_first = {"points": pl.concat_arr("lon", "lat"), "s": weighted_mean_pl("s")} | {
        d: weighted_mean_pl(d, "s") for d in deltas[1:]
    }
    
    memb = ["member"] if "member" in jets_current.columns else []

    jets_current = jets_current.group_by(*memb, "time", which_jet).agg(**aggs_first)
    jets_next = jets_next.group_by(*memb, "time", "time_shifted", which_jet).agg(**aggs_first)
    cross = jets_current.join(
        jets_next, left_on=[*memb, "time"], right_on=[*memb, "time_shifted"], how="left"
    )
    if which_jet == "jet":
        cross = cross.filter(pl.col(which_jet) == pl.col(f"{which_jet}_right"))
    cross = cross.filter(pl.col("points").is_not_null(), pl.col("points_right").is_not_null())
    overlap = _lon_overlap()
    more_aggs = {
        "dist": _frechet_st(),
        "lon_overlap": overlap,
    } | {
        d2: (pl.col(d1) - pl.col(f"{d1}_right")).abs()
        for d1, d2 in zip(deltas, deltas2)
    }
    cross = (
        cross.with_columns(**more_aggs)
        .drop(*[f"{name}{suffix}" for name in ["points"] for suffix in ["", "_right"]])
        .drop_nulls("lon_overlap")
        .sort(*typical_group_by)
    )
    return standardize_polars_dtypes(cross)


def track_jets(all_jets_one_df: DataFrame, yearly: bool = False, monthly: bool = False) -> DataFrame:
    """
    Iterates over maybe years and maybe members and performs explicit jet tracking. Only use yearly and monthly if severely memory bound, it makes everything much slower

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
    do_member = "member" in all_jets_one_df.columns
    if do_member:
        members = all_jets_one_df["member"].unique()
    else:
        members = [None]
        
    if yearly:
        years = all_jets_one_df["time"].dt.year().unique()
    else:
        years = [None]
        
    if monthly:
        months = all_jets_one_df["time"].dt.year().unique()
    else:
        months = [None]
    iterator = list(product(members, years, months))
    total = len(iterator)

    for member, year, month in tqdm(iterator, total=total):
        cross.append(
            track_jets_(
                all_jets_one_df,
                member,
                year,
                month,
            )
        )
    cross = pl.concat(cross)
    return cross


def connected_from_cross(
    jets: DataFrame,
    cross: DataFrame | None = None,
    dist_thresh: float = 2e5,
    overlap_thresh: float = 0.5,
    dis_polar_thresh: float | None = 1.0,
) -> DataFrame:
    jets = jets.cast({"time": pl.Datetime("ms")})
    if cross is None:
        cross = track_jets(jets)
    cross = cross.filter(
        pl.col("dist") < dist_thresh,
        pl.col("lon_overlap") > overlap_thresh,
        pl.col("dis_polar") < dis_polar_thresh,
    )
    gb = ["time", "jet ID"]
    mem = []
    mem_k = []
    if "member" in jets.columns:
        gb_ = ["member"]
        gb_.extend(gb)
        gb = gb_
        mem = ["member"]
        mem_k = ["member_k"]
    summary = (
        jets.group_by("time", "jet ID", maintain_order=True)
        .agg()
        .with_row_index()
    )
    cross = (
        cross.with_columns(
            dt=(
                (pl.col("time_right") - pl.col("time"))
                / (pl.col("time_right") - pl.col("time")).min()
            ).cast(pl.Int32())
        )
        .with_columns(slowness=slowness_expr() / pl.col("dt"))
        .group_by("time", "jet ID", maintain_order=True)
        .agg(
            pl.col("time_right").get(pl.col("slowness").arg_max()),
            pl.col("jet ID_right").get(pl.col("slowness").arg_max()),
            pl.col("dt").get(pl.col("slowness").arg_max()),
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
    if "is_polar" in jets.columns:
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


def slowness_expr() -> Expr:
    return (
        pl.col("lon_overlap")
        / pl.col("dist")
        * (
            pl.col("time").unique()
            .diff()
            .mode()
            .first()
            .cast(pl.Duration("ms"))
            .cast(pl.Float32())
            / 1000
        )
        .replace([float("inf"), float("nan")], 0.)
        .cast(pl.Float32())
    )
        

def spells_from_cross(
    jets: DataFrame,
    cross: DataFrame,
    dist_thresh: float = 2e5,
    overlap_thresh: float = 0.5,
    dis_polar_thresh: float | None = 1.0,
    q_STJ: float | None = None,
    q_EDJ: float | None = None,
    n_STJ: int | None = None,
    n_EDJ: int | None = None,
    season: Series | None = None,
    subtropical_cutoff: float = 0.4,
    polar_cutoff: float = 0.6,
):
    _, summary_comp = connected_from_cross(
        jets,
        cross,
        dist_thresh=dist_thresh,
        overlap_thresh=overlap_thresh,
        dis_polar_thresh=dis_polar_thresh,
    )
    if season is not None:
        summary_comp = summary_comp.filter(
            pl.col("time")
            .is_in(pl.lit(season.implode().first(), pl.List(pl.Datetime("ms"))))
            .over("spell")
        )
    spells = (
        summary_comp.filter(pl.col("len") > 2)
        .with_columns(slowness=slowness_expr())
        .group_by("spell", maintain_order=True)
        .agg(
            pl.col("time"),
            pl.col("jet ID"),
            pl.col("lon_overlap"),
            pl.col("slowness"),
            pl.col("dis_polar"),
            pl.col("is_polar"),
            len=pl.len(),
            mean_is_polar=pl.col("is_polar").mean(),
            slowness_sum=pl.col("slowness").sum(),
        )
    )
    
    spells_list = {}
    filters = {
        "STJ": pl.col("mean_is_polar") < subtropical_cutoff,
        "EDJ": pl.col("mean_is_polar") > polar_cutoff,
    }
    for jet, q, n in zip(
        ["STJ", "EDJ"],
        [q_STJ, q_EDJ],
        [n_STJ, n_EDJ],
    ):
        spell =  spells.filter(filters[jet])
        if n is not None:
            other_filter = spell.select(pl.col("spell").top_k_by("len", n))
            spell = other_filter.join(spell, on="spell")
        elif q is not None:
            filter_ = pl.col("slowness_sum") > pl.col("slowness_sum").quantile(q)
            spell = spell.filter(filter_)
        else:
            raise ValueError
        spell = (
            spell
            .explode("time", "jet ID", "slowness", "is_polar", "lon_overlap", "dis_polar")
            .with_columns(
                spell_of=pl.lit(jet),
                spell_orig=pl.col("spell"),
                spell=pl.col("spell").rle_id(),
                relative_index=pl.col("time").rle_id().over("spell").cast(pl.Int32()),
                relative_time=pl.col("time") - pl.col("time").first().over("spell")
            )
            .drop("is_polar")
            
        )
        spells_list[jet] = spell
    return spells_list


def slowness_from_cross(cross: DataFrame) -> DataFrame:
    index_columns = get_index_columns(cross)
    if "jet" in index_columns:
        return (
            cross
            .filter(pl.col("jet") == pl.col("jet_right"))
            .with_columns(slowness=slowness_expr())
            .group_by(index_columns, maintain_order=True)
            .agg(
                slowness=pl.col("slowness").max(),
                is_polar=pl.col("is_polar").mean()
            )
        )
    else:
        return (
            cross
            .with_columns(slowness=slowness_expr())
            .group_by(index_columns, maintain_order=True)
            .agg(
                slowness=pl.col("slowness").max(),
                is_polar=pl.col("is_polar").mean()
            )
        )


def spells_from_cross_catd_simple(
    cross: DataFrame,
    q_STJ: float = 0.99,
    q_EDJ: float = 0.95,
    season: pl.DataFrame | None = None,
    minlen: datetime.timedelta = datetime.timedelta(days=5),
    smooth: datetime.timedelta | None = None,
    fill_holes: datetime.timedelta | int = 0,
) -> dict[str, DataFrame]:
    cross = cross.with_columns(slowness=slowness_expr())
    cross = squarify(cross, ["time", "jet"])

    if smooth is not None:
        cross = cross.rolling(
            pl.col("time"),
            period=smooth,
            group_by="jet",
        ).agg(
            *[
                pl.col(col).mean()
                for col in ["lon_overlap", "ds", "dtheta", "dis_polar", "dist", "slowness"]
            ]
        )

    if season is not None:
        cross = season.join(cross, on="time", how="left")

    spells_list: dict[str, DataFrame] = {
        jet: get_spells(
            cross.filter(pl.col("jet") == jet),
            pl.col("slowness") > pl.col("slowness").quantile(q),
            minlen=minlen,
            fill_holes=fill_holes,
        ).with_columns(spell_of=pl.lit(jet))
        for jet, q in zip(["STJ", "EDJ"], [q_STJ, q_EDJ])
    }
    return spells_list


def spells_from_cross_catd(
    cross: DataFrame,
    base_q: float = 0.5,
    n_STJ: int = 30,
    n_EDJ: int = 30,
    season: pl.DataFrame | None = None,
    minlen: datetime.timedelta = datetime.timedelta(days=5),
    smooth: datetime.timedelta | None = None,
    fill_holes: datetime.timedelta | int = 0,
) -> dict[str, DataFrame]:
    cross = cross.with_columns(slowness=slowness_expr())
    cross = squarify(cross, ["time", "jet"])

    if smooth is not None:
        cross = cross.rolling(
            pl.col("time"),
            period=smooth,
            group_by="jet",
        ).agg(
            *[
                pl.col(col).mean()
                for col in ["lon_overlap", "ds", "dtheta", "dis_polar", "dist", "slowness"]
            ]
        )

    if season is not None:
        cross = season.join(cross, on="time", how="left")

    spells_base = get_spells(
        cross,
        pl.col("slowness") > pl.col("slowness").quantile(base_q),
        group_by=["jet"],
        minlen=minlen,
        fill_holes=fill_holes,
    )
    stats: DataFrame = spells_base.group_by(
        ["jet", "spell"], maintain_order=True
    ).agg(pl.col("len").first(), pl.col("slowness").sum())

    spells_list: dict[str, DataFrame] = {
        jet: (
            stats.filter(pl.col("jet") == jet)
            .top_k(n, by="slowness")
            .rename({"slowness": "slowness_sum"})
            .join(
                spells_base.filter(pl.col("jet") == jet).drop("len", "value"),
                on=["jet", "spell"],
            )
            .with_columns(spell=pl.col("spell").rle_id())
            .with_columns(spell_of=pl.lit(jet))
        )
        for jet, n in zip(["STJ", "EDJ"], [n_STJ, n_EDJ])
    }
    return spells_list


def jet_position_as_da(
    jets: DataFrame,
) -> xr.DataArray:
    """
    Constructs a `DataArray` of dimensions (*index_columns, lat, lon) from jets. The DataArray starts with NaNs everywhere. Then, for every jet point, the DataArray is filled with the jet point's `"is_polar"` value.
    """
    index_columns = get_index_columns(
        jets, ("member", "time", "cluster", "spell", "relative_index")
    )
    all_jets_pandas = (
        jets.group_by([*index_columns, "lon", "lat"], maintain_order=True)
        .agg(pl.col("is_polar").mean())
        .to_pandas()
    )
    da_jet_pos = xr.Dataset.from_dataframe(
        all_jets_pandas.set_index([*index_columns, "lat", "lon"])
    )["is_polar"]
    return da_jet_pos


def get_double_jet_index(jet_pos_da: xr.DataArray, diff_cat: bool = False):
    """
    Adds a new columns to props_as_df; `"double_jet_index"`, by checking, for all longitudes, if there are at least two jet core points along the latitude, then averaging this over longitudes above 20° West.
    """
    if diff_cat:
        overlap = (jet_pos_da >= 0.5).any("lat") & (jet_pos_da < 0.5).any("lat") 
    else:
        overlap = (~np.isnan(jet_pos_da)).sum("lat") >= 2
    overlap = overlap.rename("double_jet_index")
    return xarray_to_polars(overlap.sel(lon=slice(-20, None, None)).mean("lon"))
    

def do_everything(ds: xr.Dataset, save_path: Path, bias_correct_realspace: bool = False, do_smooth_spline: bool = False, track_large: bool = True, feature_names: tuple | None = None, n_init: int = 10, **find_jets_kwargs):
    """
    More easily maintainable than object-oriented approach for quick testing of the whole pipeline

    Parameters
    ----------
    ds : xr.Dataset
        _description_
    save_path : Path
        _description_
    bias_correct_realspace : bool, optional
        _description_, by default False
    do_smooth_spline : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    save_path.mkdir(exist_ok=True)
    ds = standardize(ds)
    jets_path = save_path.joinpath("jets.parquet")
    props_path = save_path.joinpath("props.parquet")
    props_final_path = save_path.joinpath("props_full.parquet")
    cross_path = save_path.joinpath("cross.parquet")
    bc_path = save_path.joinpath("bias_correct.parquet")
    if not jets_path.is_file():
        iterator = iterate_over_year_maybe_member(da=ds, several_years=1)
        jets = []
        for indexer in tqdm(list(iterator)):
            ds_ = compute(ds.isel(**indexer), progress_flag=False)
            these_jets = find_all_jets(ds_, **find_jets_kwargs)
            if bias_correct_realspace:
                these_jets = online_bias_correct(these_jets, ds_)
            if do_smooth_spline:
                these_jets = spline_smooth(these_jets, s=2, factor=1)
            if bias_correct_realspace or do_smooth_spline:
                these_jets = join_on_ds(these_jets, ds_, fix_id=True)
            jets.append(these_jets)
        jets = pl.concat(jets)
        if "int" not in jets.columns:
            index_columns = get_index_columns(jets)
            int_expr = jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s"))
            int_expr = int_expr.over(index_columns)
            jets = jets.with_columns(int=int_expr)
        jets = standardize_polars_dtypes(jets)
        jets.write_parquet(jets_path)
    else:
        jets = standardize_polars_dtypes(pl.read_parquet(jets_path))
    if "is_polar" not in jets.columns:
        if feature_names is None:
            feature_names = ("s", "theta")
        jets = is_polar_gmix(jets, feature_names=feature_names, n_init=n_init)
        jets.write_parquet(jets_path)
        
    phat_jets = to_one_large(jets, int_EDJ_threshold=1.3e8)
    phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.3e8))
    if not cross_path.is_file():
        if track_large:
            cross = track_jets(phat_jets)
        else:
            cross = track_jets(jets)
        cross.write_parquet(cross_path)
    else:
        cross = standardize_polars_dtypes(pl.read_parquet(cross_path))
        
    slowness = slowness_from_cross(cross)    
    
    if not props_path.is_file():
        props = compute_jet_props(jets)
        width = []
        da = ds["s"]
        indexer = iterate_over_year_maybe_member(jets, da)
        for idx1, idx2 in tqdm(list(indexer)):
            these_jets = jets.filter(*idx1)
            da_ = compute(da.sel(**idx2), progress_flag=False)
            width_ = compute_widths(these_jets, da_)
            width.append(width_)
        width = pl.concat(width)
        index_columns = get_index_columns(width)
        props = props.join(width, on=index_columns, how="left").sort(index_columns)
        props.write_parquet(props_path)
    else:
        props = standardize_polars_dtypes(pl.read_parquet(props_path))
        index_columns = get_index_columns(props)
        
    if not props_final_path.is_file():
        if not track_large:
            props = props.join(slowness, on=index_columns)
        phat_props = props.filter(phat_filter)

        phat_props = average_jet_categories(phat_props, polar_cutoff=0.5)
        index_columns = get_index_columns(phat_props)
        if track_large:
            phat_props = phat_props.join(slowness, on=index_columns)
        phat_props.write_parquet(props_final_path)
    else:
        phat_props = standardize_polars_dtypes(pl.read_parquet(props_final_path))
        
    if not bias_correct_realspace and not bc_path.is_file():
        bc = create_bias_correction(phat_jets, ds)
        bc.write_parquet(bc_path)

    return jets, phat_jets, props, phat_props


class JetFindingExperiment(object):
    """
    Convenience class that wraps basically all the functions in this module, applying it to the data held by its `DataHandler` and storing the results to avoid recomputing in the subfolder of its `DataHandler`.
    
    I think it's a bit broken right now, fix coming at some point. In the mean time use "do_everything" or unwrap it yourself.
    
    TODO: do_everything and this class's methods should be broken down into functions with a decorator for saving.
    Any function that has the argument "subpath", the decorator takes an argument "filename" a str, constructs the file path from DATADIR, subpath and filename, tries to load it and return it (parquet or nc) or does the computation defined by the function. The function does not know anything about saving.

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
            several_years = 2
        else:
            several_years = 5
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
            is_polar = jets.group_by(index_columns, maintain_order=True).agg(
                pl.col("is_polar").mean()
            )
            props_as_df = props_as_df.drop("is_polar").join(is_polar, on=index_columns)
        props_as_df.write_parquet(ofile_padu)
        props_as_df_cat = average_jet_categories(props_as_df)
        props_as_df_cat.write_parquet(ofile_pad)
        if categorize:
            return props_as_df_cat
        return props_as_df

    # def track_jets(
    #     self,
    #     dist_thresh: float = 2,
    #     overlap_min_thresh: float = 0.5,
    #     overlap_max_thresh: float = 0.6,
    #     dis_polar_thresh: float | None = 1.0,
    #     n_next: int = 1,
    #     force: int = 0,
    # ) -> DataFrame:
    #     """
    #     Wraps cross and summary

    #     Parameters
    #     ----------
    #     dist_thresh : float, optional
    #         _description_, by default 2
    #     overlap_min_thresh : float, optional
    #         _description_, by default 0.5
    #     overlap_max_thresh : float, optional
    #         _description_, by default 0.6
    #     dis_polar_thresh : float | None, optional
    #         _description_, by default 1.0
    #     force : int, optional
    #         _description_, by default 0

    #     Returns
    #     -------
    #     DataFrame
    #         _description_
    #     """
    #     cross_opath = self.path.joinpath("cross.parquet")
    #     summary_opath = self.path.joinpath("summary.parquet")
    #     all_jets_one_df = self.find_jets().cast({"time": pl.Datetime("ms")})
    #     if not cross_opath.is_file() or force > 1:
    #         cross = track_jets(all_jets_one_df)
    #         cross.write_parquet(cross_opath)
    #     else:
    #         cross = pl.read_parquet(cross_opath)
    #     if not summary_opath.is_file() or force:
    #         cross, summary = connected_from_cross(
    #             all_jets_one_df,
    #             cross,
    #             dist_thresh,
    #             overlap_min_thresh,
    #             overlap_max_thresh,
    #             dis_polar_thresh,
    #         )
    #         summary.write_parquet(summary_opath)
    #     else:
    #         summary = pl.read_parquet(summary_opath)
    #     return cross, summary

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
