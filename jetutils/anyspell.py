# coding: utf-8
from typing import Mapping, Literal, Callable, Sequence
from functools import partial
from multiprocessing import set_start_method as set_mp_start_method
import datetime

import numpy as np
import polars as pl
from polars import DataFrame, Series, Expr
import xarray as xr
import polars_ds as pds
import polars.selectors as cs
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import pairwise_distances
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor

    xgboost_avail = True
except ModuleNotFoundError:
    xgboost_avail = False

try:
    from fasttreeshap import TreeExplainer, Explainer
except ModuleNotFoundError:
    from shap import TreeExplainer, Explainer


from .definitions import (
    DEFAULT_VALUES,
    YEARS,
    N_WORKERS,
    RESULTS,
    compute,
    do_rle_fill_hole,
    load_pickle,
    save_pickle,
    xarray_to_polars,
    polars_to_xarray,
    get_index_columns,
    extract_season_from_df,
    gb_index,
    explode_rle,
)
from .data import (
    compute_anomalies_pl,
    find_spot,
    get_land_mask,
    extract_season,
    DataHandler,
    open_dataarray,
    to_netcdf,
)

set_mp_start_method("spawn", force=True)


def brier_score(y_true, y_proba=None, *, sample_weight=None, pos_label=None):
    """Compute the Brier score.

    The higher the Brier score, the better.
    The Brier score measures the mean squared difference between the predicted
    probability and the actual outcome. The Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1). It can be decomposed as the sum of refinement loss and
    calibration loss.

    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another). Which label is considered to be the positive
    label is controlled via the parameter `pos_label`, which defaults to
    the greater label unless `y_true` is all 0 or all -1, in which case
    `pos_label` defaults to 1.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_proba : array-like of shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    pos_label : int, float, bool or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:

        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

        .. deprecated:: 1.5
            `y_prob` is deprecated and will be removed in 1.7. Use
            `y_proba` instead.

    Returns
    -------
    score : float
        Brier score loss.

    something else : float
        another thing.

    References
    ----------
    .. [1] `Wikipedia entry for the Brier score
            <https://en.wikipedia.org/wiki/Brier_score>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
    >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    >>> brier_score_loss(y_true, y_prob)
    np.float64(0.037...)
    >>> brier_score_loss(y_true, 1-y_prob, pos_label=0)
    np.float64(0.037...)
    >>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
    np.float64(0.037...)
    >>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
    np.float64(0.0)
    """
    return 1 - brier_score_loss(
        y_true, y_proba, sample_weight=sample_weight, pos_label=pos_label
    )


ALL_SCORES = {
    func.__name__.split("loss")[0]: func
    for func in (
        roc_auc_score,
        f1_score,
        balanced_accuracy_score,
        brier_score,
    )
}

ALL_MODELS = {
    "lr": [LogisticRegression, LinearRegression],
    "rf": [RandomForestClassifier, RandomForestRegressor],
}

if xgboost_avail:
    ALL_MODELS["xgb"] = [XGBClassifier, XGBRegressor]
    type AnyModel = LogisticRegression | LinearRegression | RandomForestClassifier | RandomForestRegressor | XGBClassifier | XGBRegressor
else:
    type AnyModel = LogisticRegression | LinearRegression | RandomForestClassifier | RandomForestRegressor


def _get_spells_sigma(
    df: pl.DataFrame, dists: np.ndarray | None = None, sigma: int = 1
) -> pl.DataFrame:
    start = 0
    spells = []
    next_distance_cond = np.zeros(1, dtype=bool)
    while True:
        start_lab: int = df[int(start), "labels"]
        if dists is not None:
            next_distance_cond: np.ndarray = dists[start_lab, df[start:, "labels"].to_numpy()] > sigma
        else:
            next_distance_cond: np.ndarray = df[start:, "labels"].to_numpy() != start_lab
        if not np.any(next_distance_cond):
            spells.append(
                {
                    "rel_start": start,
                    "value": start_lab,
                    "len": len(df[start:, "labels"]),
                }
            )
            break
        to_next = np.argmax(next_distance_cond).item()
        val = df[int(start) : int(start + to_next), ["labels"]].with_columns(
            pl.col("labels").drop_nulls().mode().first().alias("mode")
        )[0, "mode"]
        spells.append({"rel_start": start, "value": val, "len": to_next})
        start = start + to_next
    return pl.DataFrame(spells).with_columns(year=df[0, "year"], my_len=df.shape[0])


def _get_persistent_spell_times_from_som(
    labels_df: pl.DataFrame,
    dists: np.ndarray | None = None,
    sigma: int = 0,
    minlen: int = 4,
    nt_before: int = 0,
    nt_after: int = 0,
    nojune: bool = True,
    daily: bool = False,
):
    if dists is None and sigma > 0:
        print("sigma > 0 needs a distance matrix")
        raise ValueError
    index_columns = get_index_columns(labels_df)
    index = labels_df[index_columns].unique(maintain_order=True)

    out = labels_df.group_by("year", maintain_order=True).map_groups(
        partial(_get_spells_sigma, dists=dists, sigma=sigma)
    )
    out = (
        out[["year", "my_len"]]
        .unique(maintain_order=True)
        .with_columns(
            my_len=pl.lit(0)
            .append(pl.col("my_len"))
            .cum_sum()
            .head(pl.col("year").len())
        )
        .join(out.drop("my_len"), on="year")
        .with_columns(start=pl.col("my_len") + pl.col("rel_start"))
        .with_columns(
            range=pl.int_ranges(
                pl.col("start") - nt_before, pl.col("start") + pl.col("len") + nt_after
            ),
            relative_index=pl.int_ranges(
                -nt_before, pl.col("len") + nt_after, dtype=pl.Int16
            ),
        )
        .with_row_index("spell")
        .explode(["range", "relative_index"])
        .filter(pl.col("range") < len(index), pl.col("range") >= 0)
    )
    out = (
        out.with_columns(index[out["range"]])
        .filter(pl.col("len") >= minlen)
        .with_columns(
            pl.col("spell").rle_id(),
            filter_=(pl.col("time").dt.year() == pl.col("time").dt.year().gather(
                pl.arg_where(pl.col("relative_index") == 0).first()
            )).over("spell")
        )
        .filter(pl.col("filter_"))
        .drop("filter_")
    )
    if nojune:
        june_filter = out.group_by("spell", maintain_order=True).agg(
            (pl.col("time").dt.ordinal_day() <= 160).sum() > 0.8
        )["time"]
        out = out.filter(pl.col("spell").is_in(june_filter.not_().arg_true()))
        out = out.with_columns(pl.col("spell").rle_id())
    out = out.with_columns(
        relative_time=(
            pl.col("time")
            - pl.col("time").gather(pl.arg_where(pl.col("relative_index") == 0).first())
        ).over("spell")
    )
    if "member" in labels_df.columns:
        out = out.with_columns(member=pl.lit(labels_df["member"].first()))
    if not daily:
        return out

    ratio = out.filter(pl.col("relative_index") == 1)[
        0, "relative_time"
    ] / datetime.timedelta(days=1)
    out = out.with_columns(pl.col("time").dt.round("1d")).unique(
        ["spell", "time"], maintain_order=True
    )
    out = out.with_columns(
        out.group_by("spell", maintain_order=True)
        .agg(
            pl.col("relative_index").rle_id()
            + (pl.col("relative_index").first() * ratio).round().cast(pl.Int16)
        )
        .explode("relative_index")
    )
    out = out.with_columns(relative_time=pl.col("relative_index") * pl.duration(days=1))
    return out


def get_persistent_spell_times_from_som(
    labels,
    dists: np.ndarray | None = None,
    sigma: float = 0.0,
    minlen: int = 4,
    nt_before: int = 0,
    nt_after: int = 0,
    nojune: bool = True,
    daily: bool = False,
):
    labels_df = xarray_to_polars(labels)
    labels_df = labels_df.with_columns(pl.col("time").dt.year().alias("year"))
    if "member" not in labels_df.columns:
        return _get_persistent_spell_times_from_som(
            labels_df,
            dists,
            sigma,
            minlen,
            nt_before,
            nt_after,
            nojune,
            daily,
        )
    func = partial(
        _get_persistent_spell_times_from_som,
        dists=dists,
        sigma=sigma,
        minlen=minlen,
        nt_before=nt_before,
        nt_after=nt_after,
        nojune=nojune,
        daily=daily,
    )
    # specify schema because all-Null subseries will have Null dtype and break the concat
    schema = {
        "spell": pl.UInt32,
        "time": labels_df["time"].dtype,
        "value": pl.Int64,
        "len": pl.Int64,
        "relative_time": pl.Duration("ns"),
        "memver": str,
    }
    schema = pl.Schema(schema)
    spells = (
        labels_df.lazy()
        .group_by("member", maintain_order=True)
        .map_groups(func, schema=schema)
        .collect()
    )
    return spells


def fill_null_mode(col: str, over: str | None = None):
    expr = pl.col(col).fill_null(pl.col(col).filter(pl.col(col).is_not_null()).mode().first())
    if over is None:
        return expr
    return expr.over(over)


def extend_spells(
    spells: pl.DataFrame,
    time_before: datetime.timedelta = datetime.timedelta(0),
    time_after: datetime.timedelta = datetime.timedelta(0),
    index_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    times = spells["time"].unique()
    dt = times[1] - times[0]
    if index_columns is None:
        index_columns = get_index_columns(spells, ["member", "region", "spell", "jet", "spell_of", "jet ID"])
    exprs = {
        "len": pl.col("len").first(),
        "time": pl.datetime_range(
            pl.col("time").first() - time_before,
            pl.col("time").last() + time_after,
            interval=dt,
            closed="both",
        ),
        "relative_time": pl.datetime_range(
            pl.col("time").first() - time_before,
            pl.col("time").last() + time_after,
            interval=dt,
            closed="both",
        )
        - pl.col("time").first(),
    }
    other_columns = [
        col for col in spells.columns
        if col not in [*index_columns, *list(exprs), "time", "relative_index"]
    ]
    spells = (
        spells.group_by(index_columns, maintain_order=True)
        .agg(**exprs)
        .explode(list(exprs)[1:])
        .join(
            spells[[*index_columns, "time", *other_columns]], 
            on=[*index_columns, "time"], 
            how="left"
        )
        .with_columns(
            *[fill_null_mode(col, "spell") for col in other_columns],
            relative_index=(pl.col("relative_time") / dt).cast(pl.Int32)
        )
    )
    return spells


def make_daily(
    df: pl.DataFrame,
    group_by: Sequence[str] | Sequence[pl.Expr] | str | pl.Expr | None = None,
    to_keep: Sequence[str] | Sequence[pl.Expr] | None = None
) -> pl.DataFrame:
    if to_keep is None:
        to_keep = []
    if "relative_time" in df.columns:
        min_rel_time = df["relative_time"].min()
    else:
        min_rel_time = pl.duration(days=0, time_unit=df["time"].dtype.time_unit)
    aggs = [
        *[pl.col(col).first() for col in to_keep],
        cs.numeric().exclude(to_keep).mean(),
    ]
    relative_time = (pl.col("time") - pl.col("time").first()).over(group_by)
    relative_time = relative_time + min_rel_time
    relative_index = (relative_time / pl.duration(days=1)).cast(pl.Int32())
    df = df.group_by_dynamic("time", every="1d", group_by=group_by).agg(*aggs)
    return df.with_columns(relative_time=relative_time, relative_index=relative_index).with_columns(len=pl.col("relative_index").filter(pl.col("relative_index") >= 0).len().over("spell"))


def get_spells(
    df: pl.DataFrame,
    expr: pl.Expr,
    group_by: Sequence[str] | Sequence[pl.Expr] | str | pl.Expr | None = None,
    fill_holes: datetime.timedelta = datetime.timedelta(hours=12),
    minlen: datetime.timedelta = datetime.timedelta(days=3),
    time_before: datetime.timedelta = datetime.timedelta(0),
    time_after: datetime.timedelta = datetime.timedelta(0),
    daily: bool = False,
) -> DataFrame:
    if isinstance(group_by, str | pl.Expr):
        group_by = [group_by]
    if group_by is None:
        group_by = []
    if daily:
        df = make_daily(df, group_by)
    times = df["time"].unique()
    dt = times[1] - times[0]
    minlen = int(minlen / dt)
    out = do_rle_fill_hole(df, expr, group_by, fill_holes)
    out = out.filter(pl.col("value"), pl.col("len") >= minlen)
    out = gb_index(out, group_by, "spell")
    out = explode_rle(out)
    df = gb_index(df[*group_by, "time", expr.meta.output_name()], group_by)
    out = out.join(df, on=[*group_by, "index"])
    out = extend_spells(out, index_columns=["spell", *group_by])
    out = out.sort([*group_by, "spell", "relative_index"])
    return out


def get_persistent_jet_spells(
    props_as_df,
    metric: str,
    q: float = 0.9,
    jet: str = "EDJ",
    season: list | str | None = None,
    **kwargs,
):
    props_as_df = extract_season_from_df(props_as_df, season)
    onejet = props_as_df.filter(pl.col("jet") == jet)
    if metric in ["com_speed", "speed"]:
        expr = pl.col(metric) < pl.col(metric).quantile(1 - q)
    else:
        expr = pl.col(metric) > pl.col(metric).quantile(q)
    group_by = "member" if ("member" in props_as_df.columns) else None
    return get_spells(onejet, expr, group_by=group_by, **kwargs)


def subset_around_onset(df, around_onset: int | datetime.timedelta | None = None):
    if isinstance(around_onset, int) and "relative_index" in df.columns:
        df = df.filter(pl.col("relative_index").abs() <= around_onset)
    elif isinstance(around_onset, datetime.timedelta) and "relative_time" in df.columns:
        df = df.filter(pl.col("relative_time").abs() <= around_onset)
    else:
        raise ValueError
    return df


def mask_from_spells_pl(
    spells: pl.DataFrame,
    to_mask: xr.DataArray | xr.Dataset | pl.DataFrame,
    force_pl: bool = False,
    time_before: datetime.timedelta = datetime.timedelta(0),
    time_after: datetime.timedelta = datetime.timedelta(0),
):
    spells = extend_spells(spells, time_before=time_before, time_after=time_after)
    index_columns = get_index_columns(spells, ("member", "time"))
    unique_index_spells = spells.select(index_columns).unique(index_columns)
    unique_times_to_mask = [
        pl.Series(index_column, to_mask[index_column].to_numpy())
        for index_column in index_columns
    ]
    unique_times_to_mask = pl.DataFrame(unique_times_to_mask).unique(index_columns)
    unique_times = np.intersect1d(unique_index_spells, unique_times_to_mask)
    if isinstance(to_mask, xr.DataArray | xr.Dataset):
        to_mask = compute(to_mask.sel(time=unique_times), progress=True)
        to_mask = xarray_to_polars(to_mask)
        index_columns_xarray = get_index_columns(
            to_mask, ["lat", "lon", "jet", "jet ID", "cluster"]
        )
    else:
        to_mask = to_mask.cast({"time": pl.Datetime("ns")})
        index_columns_xarray = None
    to_mask = to_mask.cast({"time": pl.Datetime("ns")})
    spells = spells.cast(
        {"time": pl.Datetime("ns"), "relative_time": pl.Duration("ns")}
    )
    index_columns = get_index_columns(to_mask, ["member", "time"])
    if "region" in spells.columns and "region" in to_mask.columns:
        index_columns.append("region")
    masked = spells.join(to_mask, on=index_columns)
    if "len_right" in masked:
        masked.drop_in_place("len_right")
    if not index_columns_xarray or (masked.shape[0] == 0) or force_pl:
        return masked
    index_to_mask = ["spell", "relative_index"]
    masked = polars_to_xarray(masked, [*index_to_mask, *index_columns_xarray])
    index_to_mask = ["spell", "relative_index"]
    i0 = np.argmax(np.asarray(list(masked.dims)) == "relative_index")
    j0 = np.argmax(masked["relative_index"].values == 0)
    ndim = masked["time"].ndim
    indexer = [0 if i >= len(index_to_mask) else slice(None) for i in range(ndim)]
    masked["time"] = masked["time"][*indexer]
    masked["relative_time"] = masked["relative_time"][*indexer]
    indexer[i0] = j0
    masked["len"] = masked["len"][*indexer]
    coords = ["time", "relative_time", "len"]
    if "value" in spells.columns:
        masked["value"] = masked["value"][*indexer]
        coords.append("value")
    masked["relative_time"] = masked["relative_time"].max(dim="spell", skipna=True)
    masked = masked.set_coords(coords)
    data_vars = list(masked.data_vars)
    if len(data_vars) == 1:
        masked = masked[data_vars[0]]
    return masked


def mask_name(mask: xr.DataArray | Literal["land"] | None = None) -> str:
    if isinstance(mask, xr.DataArray):
        return "custom"
    elif mask is None:
        return "None"
    return mask


def quantile_exceedence(
    da: xr.DataArray, q: float = 0.95, dim: str = "time"
) -> xr.DataArray:
    if q > 0.5:
        return da > da.quantile(q, dim=dim)
    return da < da.quantile(q, dim=dim)


def spatial_pairwise_jaccard(
    da: xr.DataArray,
    condition_function: Callable = lambda x: x,
    mask: xr.DataArray | Literal["land"] | None = None,
    season: str | list | None = None,
    metric: str = "jaccard",
) -> np.ndarray:
    lon, lat = da.lon.values, da.lat.values
    if mask is not None and not isinstance(mask, xr.DataArray) and mask == "land":
        mask = get_land_mask()
    if mask is not None:
        mask = mask.sel(lon=lon, lat=lat)
    da = extract_season(da, season)
    to_cluster = condition_function(da)
    stack_dims = {"lat_lon": ("lat", "lon")}
    to_cluster_flat = to_cluster.stack(stack_dims)
    if mask is not None:
        mask_flat = mask.stack(stack_dims)
        to_cluster_flat = to_cluster_flat.values[:, mask_flat.values]
    return pairwise_distances(to_cluster_flat.T, metric=metric, n_jobs=N_WORKERS)



def regionalize(da: xr.DataArray, clusters: xr.DataArray, sample_dims: list[str] | None = None):
    if sample_dims is None:
        sample_dims = [coord for coord in da.coords if coord not in ["lon", "lat"]]
    df = xarray_to_polars(da)
    name = clusters.name
    
    clusters = (
        xarray_to_polars(clusters)
        .drop_nulls(name)
        .rename({name: "region"})
        .cast({"region": pl.UInt16})
    )

    targets = (
        df.join(clusters, on=["lon", "lat"])
        .group_by([*sample_dims, "region"], maintain_order=True)
        .agg(pl.col(da.name).mean())
    )
    if "time" in targets.columns:        
        targets = targets.cast({"time": pl.Datetime("ms")})
    if "relative_time" in targets.columns:
        targets = targets.cast({"relative_time": pl.Duration("ms")})
    return targets


def handle_nan_predictors(
    predictors: pl.DataFrame,
    index_columns_no_time: list[str],
    nan_method: (
        Literal["fill"] | Literal["fill_mean"] | Literal["linear"] | Literal["nearest"]
    ) = "fill",
) -> pl.DataFrame:
    data_vars = [
        col for col in predictors.columns if col not in [*index_columns_no_time, "time"]
    ]
    if nan_method == "fill":
        fill_dict = {
            data_var: (
                pl.col(data_var)
                .fill_nan(DEFAULT_VALUES[data_var])
                .fill_null(DEFAULT_VALUES[data_var])
            )
            for data_var in data_vars
        }
        predictors = predictors.with_columns(**fill_dict)
        return predictors
    elif nan_method == "fill_mean":
        fill_dict = {
            data_var: (
                pl.col(data_var)
                .fill_nan(pl.mean(data_var))
                .fill_null(pl.mean(data_var))
            )
            for data_var in data_vars
        }
        predictors = predictors.with_columns(**fill_dict)
        return predictors
    elif nan_method not in ["linear", "nearest"]:
        print("Wrong nan method")
        raise ValueError("Wrong nan method")
    aggs = {
        data_var: (
            pl.col(data_var)
            .interpolate(nan_method)
            .fill_nan(DEFAULT_VALUES[data_var])
            .fill_null(DEFAULT_VALUES[data_var])
        )
        for data_var in data_vars
    }
    predictors = predictors.group_by(*index_columns_no_time, maintain_order=True).agg(
        **aggs
    )
    return predictors


def detrend_pl(
    df: pl.DataFrame,
    index_columns_no_time: list[str],
) -> pl.DataFrame:
    data_vars = [
        col for col in df.columns if col not in [*index_columns_no_time, "time"]
    ]
    aggs = {"time": pl.col("time")}
    aggs = aggs | {
        data_var: pds.detrend(pl.col(data_var), "linear") for data_var in data_vars
    }
    df = (
        df.group_by(index_columns_no_time, maintain_order=True)
        .agg(**aggs)
        .explode(list(aggs))
    )
    return df


def add_timescales_no_gb(
    df: pl.DataFrame,
    data_vars: list[str],
    timescales: list[datetime.timedelta] | None = None,
) -> pl.DataFrame:
    out = []
    for timescale in timescales:
        aggs = {
            col: (pl.mean(col).rolling("time", period=timescale)) for col in data_vars
        }
        out_ = df.with_columns(timescale=timescale, time=pl.col("time"), **aggs)
        out.append(out_)
    return pl.concat(out)


def add_timescales(
    df: pl.DataFrame,
    index_columns: list[str],
    timescales: list[datetime.timedelta] | None = None,
) -> pl.DataFrame:
    data_vars = [col for col in df.columns if col not in [*index_columns, "time"]]
    if not index_columns:
        return add_timescales_no_gb(df, data_vars, timescales)
    out = []
    for _, predictors_ in df.group_by(index_columns):
        predictors_ = add_timescales_no_gb(predictors_, data_vars, timescales)
        out.append(predictors_)
    return pl.concat(df)


def add_lags(
    df: pl.DataFrame,
    index_columns: list[str],
    lags: list[datetime.timedelta] | None = None,
) -> pl.DataFrame:
    unique_times = df["time"].unique()
    dt = unique_times[1] - unique_times[0]
    ilags = (np.asarray(lags) / dt).astype(int)
    jump_times = unique_times.filter((unique_times.diff() > dt).fill_null(True))
    data_vars = [col for col in df.columns if col not in [*index_columns, "time"]]

    def _aggs_add_lags(
        lag: datetime.timedelta,
        ilag: int,
    ) -> pl.DataFrame:
        flagged_times = (
            jump_times.to_frame()
            .group_by("time", maintain_order=True)
            .agg(
                pl.datetime_range(
                    pl.col("time"),
                    pl.col("time") + lag,
                    interval=dt,
                    closed="both" if ilag != 0 else "none",
                ).alias("times")
            )["times"]
            .explode()
        )
        aggs = {
            data_var: (
                pl.when(pl.col("time").is_in(flagged_times))
                .then(None)
                .otherwise(pl.col(data_var).shift(ilag))
            )
            for data_var in data_vars
        }
        return aggs

    out = []
    for lag, ilag in zip(lags, ilags):
        aggs = _aggs_add_lags(lag, ilag)
        predictors_ = df.with_columns(
            lag=lag,
            **aggs,
        )
        out.append(predictors_)
    return pl.concat(out)


def prepare_predictors(
    predictors: pl.DataFrame,
    subset: list | None = None,
    anomalize: bool = False,
    standardize: bool = False,
    detrend: bool = False,
    to_daily: bool = False,
    nan_method: (
        Literal["fill"] | Literal["fill_mean"] | Literal["linear"] | Literal["nearest"]
    ) = "fill",
    season: str | list | None = None,
    timescales: list[datetime.timedelta] | None = None,
    lags: list[datetime.timedelta] | None = None,
) -> pl.DataFrame:
    index_columns = get_index_columns(predictors, ["member", "time", "jet"])
    index_columns_no_time = get_index_columns(predictors, ["member", "jet"])
    index_columns_no_jet = get_index_columns(predictors, ["member", "time"])
    if subset is not None:
        predictors = predictors[[*index_columns, *subset]]
    if anomalize:
        predictors = compute_anomalies_pl(
            predictors,
            other_index_columns=index_columns_no_time,
            standardize=standardize,
        )
        if nan_method == "fill":
            nan_method = "fill_mean"
    predictors = handle_nan_predictors(
        predictors, index_columns_no_time, nan_method=nan_method
    )
    if detrend:
        predictors = detrend_pl(predictors, index_columns_no_time)
    if to_daily:
        predictors = make_daily(predictors, index_columns_no_time)
    if "jet" in predictors.columns:
        predictors = predictors.pivot("jet", index=index_columns_no_jet)
        index_columns = [col for col in index_columns if col != "jet"]
        index_columns_no_time = [col for col in index_columns_no_time if col != "jet"]
    if timescales:
        predictors = add_timescales(predictors, index_columns_no_time, timescales)
        predictors = predictors.with_columns(pl.col("timescale").dt.to_string())
        predictors = predictors.pivot("timescale", index=index_columns)
    if lags:
        predictors = add_lags(predictors, index_columns_no_time, lags)
    predictors = extract_season_from_df(predictors, season=season)
    return predictors


def regress_against_time(targets: pl.DataFrame) -> pl.DataFrame:
    index_columns = get_index_columns(targets, ["member", "region", "time"])
    index_columns_notime = get_index_columns(targets, ["member", "region"])
    targets = targets[[*index_columns, "len"]]
    targets = targets.with_columns(pl.col("len") > 0)
    base_pred = []
    for _, df in targets.group_by(index_columns_notime, maintain_order=True):
        time_ = np.arange(len(df)).reshape(-1, 1)
        pred = (
            LogisticRegression(solver="liblinear")
            .fit(time_, df["len"].cast(bool))
            .predict_proba(time_)
        )
        df = df.with_columns(pred=pred[:, -1])
        base_pred.append(df)
    return pl.concat(base_pred)


def compute_all_scores(y_test, y_pred, y_pred_prob) -> Mapping:
    scores = {}
    for scorename, scorefunc in ALL_SCORES.items():
        if scorename in ["roc_auc_score", "brier_score_loss"]:
            scores[scorename] = scorefunc(y_test, y_pred_prob)
        else:
            scores[scorename] = scorefunc(y_test, y_pred)
    return scores


def compute_all_importances(
    model: AnyModel,
    model_type: str,
    X: pl.DataFrame,
    y: pl.DataFrame,
    compute_shap: bool = False,
):
    predictor_names = X.columns
    names = []
    corr = pl.concat([X, y.to_frame()], how="horizontal").select(
        *[pl.corr(predictor, "target") for predictor in predictor_names]
    )
    corr = corr.to_dicts()[0]
    names.append("correlation")

    if model_type in ["rf", "xgb"]:
        model_imp = model.feature_importances_
        names.append("impurity")
    elif model_type == "lr":
        model_imp = model.coef_.ravel()
        names.append("coef_")
    model_imp = dict(zip(predictor_names, model_imp))

    perm_imp = permutation_importance(
        model, X.to_pandas(), y, n_repeats=30, random_state=0, n_jobs=1
    )["importances_mean"]
    perm_imp = dict(zip(predictor_names, perm_imp))
    names.append("permutation")

    if not compute_shap:
        to_ret = pl.from_dicts([corr, model_imp, perm_imp])
        to_ret = to_ret.with_columns(importance_type=names)
        return to_ret

    shap_explainer = TreeExplainer(model) if model_type == "rf" else Explainer(model)
    shap_values = shap_explainer(X.to_numpy(), y, check_additivity=False).values

    mean_shap = shap_values.mean(axis=0)
    mean_shap = dict(zip(predictor_names, mean_shap))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_shap = dict(zip(predictor_names, mean_abs_shap))
    names.extend(("mean_shap", "mean_abs_shap"))

    to_ret = pl.from_dicts([corr, model_imp, perm_imp, mean_shap, mean_abs_shap])
    to_ret = to_ret.with_columns(importance_type=pl.Series("importance_type", names))

    return to_ret


def predict_all(
    predictors: pl.DataFrame,
    orig_targets: pl.DataFrame,
    base_pred: pl.DataFrame | None = None,
    model_type: Literal["rf", "lr", "xgb"] = "rf",
    compute_shap: bool = False,
    n_folds: int = 1,
    **kwargs,
):
    """
    One model per region, lag and fold. One model for all times and members if applicable

    Parameters
    ----------
    predictors : pl.DataFrame
        _description_
    orig_targets : pl.DataFrame
        _description_
    base_pred : pl.DataFrame | None, optional
        _description_, by default None
    model_type : Literal[&quot;rf&quot;, &quot;lr&quot;, &quot;xgb&quot;], optional
        _description_, by default "rf"
    compute_shap : bool, optional
        _description_, by default False
    n_folds : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    index_columns_targets = get_index_columns(
        orig_targets, ["region", "member", "time"]
    )
    index_columns_predictors = get_index_columns(predictors, ["member", "time", "lag"])
    predictor_names = [
        col for col in predictors.columns if col not in index_columns_predictors
    ]
    # Handle base_pred
    if model_type == "lr" and base_pred is not None:
        print(f"Base pred is incompatible with {model_type}, ignoring")
        base_pred = None
    if base_pred is not None:
        targets = orig_targets[*index_columns_targets, "len"].join(
            base_pred.drop("len"), on=index_columns_targets
        )
        targets = targets.with_columns(
            target=pl.col("len").cast(pl.Float32) - pl.col("pred").cast(pl.Float32)
        )
    else:
        targets = orig_targets[[*index_columns_targets, "len"]].with_columns(
            target=pl.col("len").cast(pl.Boolean), pred=0.0
        )
    join_columns = np.intersect1d(index_columns_targets, index_columns_predictors)
    targets = targets.join(
        predictors.cast({"time": pl.Datetime("us")}), on=join_columns
    )
    group_by_columns = get_index_columns(targets, ["region", "lag"])
    Model_class = ALL_MODELS[model_type][int(base_pred is not None)]

    full_pred = []
    scores = []
    feature_importances = []
    for indexer, df in targets.group_by(group_by_columns, maintain_order=True):
        X = df[predictor_names]
        y = df["target"]
        y_base = df["pred"]
        y_orig = df["len"] > 0
        indexer_dict = dict(zip(group_by_columns, indexer))
        for fold in range(n_folds):
            indexer_dict_with_fold = indexer_dict | {"fold": fold}
            X_train, X_test, y_train, y_test, _, y_orig_test, _, y_base_test = (
                train_test_split(X, y, y_orig, y_base, test_size=0.2)
            )
            model = Model_class(n_jobs=1, **kwargs).fit(X_train, y_train)
            if base_pred is None:
                y_pred_test = model.predict(X_test)
                y_pred_prob_test = model.predict_proba(X_test)[:, 1]
                pred = model.predict_proba(X)[:, 1]
            else:
                y_pred_prob_test = model.predict(X_test) + y_base_test
                y_pred_prob_test = np.clip(y_pred_prob_test, 0, 1)
                y_pred_test = y_pred_prob_test > 0.5
                pred = model.predict(X) + y_base
            this_pred = df.drop(
                ["target", "len", "pred", *predictor_names]
            ).with_columns(pred=pred, fold=fold)
            full_pred.append(this_pred)

            these_scores = compute_all_scores(
                y_orig_test, y_pred_test, y_pred_prob_test
            )
            scores.append(indexer_dict_with_fold | these_scores)

            these_importances = compute_all_importances(
                model, model_type, X, y, compute_shap
            )
            these_importances = these_importances.with_columns(**indexer_dict_with_fold)
            feature_importances.append(these_importances)

    full_pred = pl.concat(full_pred)
    scores = pl.from_dicts(scores)
    feature_importances = pl.concat(feature_importances)
    return full_pred, scores, feature_importances


class ExtremeExperiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
        q: float = 0.95,
        mask: xr.DataArray | Literal["land"] | None = "land",
        season: str | list | None = "JJA",
        metric: str = "jaccard",
    ) -> None:
        self.data_handler = data_handler
        self.da = self.data_handler.da
        self.path = self.data_handler.path
        self.q = q
        self.mask_name = mask_name(mask)
        if mask and mask == "land":
            self.mask = get_land_mask()
        else:
            self.mask = mask
        if season is None:
            self.season = self.data_handler.get_metadata()["season"]
        else:
            self.season = season
        self.metric = metric
        self.path_suffix = f"{q}_{season}_{metric}_{self.mask_name}mask"
        self.region = self.data_handler.get_metadata()["region"]
        self.pred_path = self.path.joinpath("predictions")
        self.pred_path.mkdir(parents=True, exist_ok=True)

    def load_da(self, **kwargs):
        self.da = compute(self.da, **kwargs)

    def compute_linkage_quantile(
        self,
    ) -> np.ndarray:
        Z_path = f"Z_{self.path_suffix}.npy"
        Z_path = self.path.joinpath(Z_path)
        if Z_path.is_file():
            return np.load(Z_path)
        condition_function = partial(quantile_exceedence, q=self.q, dim="time")
        self.load_da()
        distances = spatial_pairwise_jaccard(
            self.da,
            condition_function,
            self.mask,
            season=self.season,
            metric=self.metric,
        )
        Z = linkage(squareform(distances), method="ward")
        np.save(Z_path, Z)
        return Z

    def spatial_clusters_as_da(
        self,
        n_clu: int,
    ) -> xr.DataArray:
        feature_dims = self.data_handler.get_feature_dims()
        clusters_da_file = f"clusters_{self.path_suffix}_{n_clu}.nc"
        clusters_da_file = self.path.joinpath(clusters_da_file)
        if clusters_da_file.is_file():
            return open_dataarray(clusters_da_file)

        Z = self.compute_linkage_quantile()
        clusters = cut_tree(Z, n_clusters=n_clu)[:, 0]
        lon, lat = feature_dims["lon"], feature_dims["lat"]
        stack_dims = {"lat_lon": ("lat", "lon")}
        if self.mask is not None:
            mask = self.mask.sel(lon=lon, lat=lat)
            mask_flat = mask.stack(stack_dims)
            clusters_da = np.zeros(mask_flat.shape, dtype=float)
            clusters_da[:] = np.nan
            clusters_da = mask_flat.copy(data=clusters_da)
            clusters_da[mask_flat] = clusters
        else:
            clusters_da = self.da.copy(data=np.zeros(self.da.shape))
            clusters_da = clusters_da.stack(stack_dims)
            clusters_da[:] = clusters
        clusters_da = clusters_da.unstack()
        to_netcdf(clusters_da, clusters_da_file)
        return clusters_da

    def create_targets(
        self,
        n_clu: int,
        q: float | None = None,
        simple: bool = False,
        return_folder: bool = False,
        **kwargs,
    ):
        if q is None:
            q = self.q
        metadata = dict(
            n_clu=n_clu,
            q=q,
            simple=simple,
            **kwargs,
        )
        sample_dims = list(self.data_handler.sample_dims)
        sample_dims_no_time = [dim for dim in sample_dims if dim != "time"]
        thispath = self.pred_path
        thispath = find_spot(thispath, metadata)
        ofiles = [
            "targets.parquet",
            "spells.parquet",
        ]
        ofiles = [thispath.joinpath(ofile) for ofile in ofiles]
        if all([ofile.is_file() for ofile in ofiles]):
            if return_folder:
                return thispath
            to_ret = []
            for ofile in ofiles:
                to_ret.append(pl.read_parquet(ofile))
            return tuple(to_ret)
        clusters = self.spatial_clusters_as_da(n_clu)
        targets = regionalize(self.da, clusters, sample_dims)

        targets = extract_season_from_df(targets, self.season)
        expr = pl.col(self.da.name)
        expr = expr > expr.quantile(q)
        spells = get_spells(targets, expr, group_by=[*sample_dims_no_time, "region"])
        targets = targets.join(
            spells[[*sample_dims, "region", "len"]],
            on=[*sample_dims, "region"],
            how="left",
        ).fill_null(0)
        targets = targets.rename({self.da.name: "value"})
        to_ret = targets, spells
        for to_save, ofile in zip(to_ret, ofiles):
            to_save.write_parquet(ofile)
        if return_folder:
            return thispath
        return to_ret

    def mask_timeseries(
        self,
        timeseries: xr.DataArray | xr.Dataset | pl.DataFrame,
        n_clu: int,
        i_clu: int | Sequence[int] | Literal["all"] = "all",
        q: float | None = None,
        simple: bool = False,
        **kwargs,
    ):
        _, spells = self.create_targets(n_clu, i_clu, q, simple, **kwargs)
        return mask_from_spells_pl(spells, timeseries)

    def full_prediction(
        self,
        predictors: xr.DataArray,
        create_target_kwargs: Mapping,
        type_: Literal["rf", "lr"] = "rf",
        do_base_pred: bool = True,
        n_folds: int = 1,
        prediction_kwargs: Mapping | None = None,
    ):
        targets_folder = self.create_targets(**create_target_kwargs, return_folder=True)
        targets = open_dataarray(targets_folder.joinpath("length_targets.nc")) > 0
        if do_base_pred:
            path_to_base_pred = targets_folder.joinpath("base_pred.nc")
            if path_to_base_pred.is_file():
                base_pred = open_dataarray(path_to_base_pred)
            else:
                base_pred = regress_against_time(targets)
                to_netcdf(base_pred, path_to_base_pred)
        else:
            base_pred = None
            path_to_base_pred = None
        predictor_names = predictors.predictor.values
        if "lag" in predictors:
            lags = predictors.lag.values.tolist()
        else:
            lags = [0]
        metadata = {
            "predictors": predictor_names.tolist(),
            "type": type_,
            "lags": lags,
            "base_pred": path_to_base_pred,
            "n_folds": n_folds,
            "prediction_kwargs": prediction_kwargs,
        }
        if prediction_kwargs is None:
            prediction_kwargs = {}
        path = targets_folder.joinpath("one_prediction")
        path.mkdir(mode=0o777, parents=True, exist_ok=True)
        path = find_spot(path, metadata)
        return predict_all(
            predictors,
            targets,
            base_pred,
            type_,
            True,
            n_folds=n_folds,
            save_path=path,
            **prediction_kwargs,
        )

    # def multi_combination_prediction(
    #     self,
    #     predictors: xr.DataArray,
    #     create_target_kwargs: Mapping,
    #     type_: Literal["rf", "lr"] = "rf",
    #     do_base_pred: bool = True,
    #     max_n_predictors: int = 10,
    #     prediction_kwargs: Mapping | None = None,
    #     winner_according_to: str = "roc_auc_score",
    # ):
    #     """
    #     Wrappy Wrappy wrap
    #     """
    #     targets_folder = self.create_targets(**create_target_kwargs, return_folder=True)
    #     targets = open_dataarray(targets_folder.joinpath("length_targets.nc")) > 0
    #     if do_base_pred:
    #         path_to_base_pred = targets_folder.joinpath("base_pred.nc")
    #         if path_to_base_pred.is_file():
    #             base_pred = open_dataarray(path_to_base_pred)
    #         else:
    #             base_pred = regress_against_time(targets)
    #             to_netcdf(base_pred, path_to_base_pred)
    #     else:
    #         base_pred = None
    #         path_to_base_pred = None
    #     predictor_names = predictors.predictor.values
    #     metadata = {
    #         "predictors": predictor_names.tolist(),
    #         "type": type_,
    #         "base_pred": path_to_base_pred,  # or None
    #         "prediction_kwargs": prediction_kwargs,
    #     }
    #     if prediction_kwargs is None:
    #         prediction_kwargs = {}
    #     path = targets_folder.joinpath("multi_combination")
    #     path.mkdir(mode=0o777, parents=True, exist_ok=True)
    #     path = find_spot(path, metadata)
    #     best_combinations = multi_combination_prediction(
    #         predictors,
    #         targets,
    #         base_pred,
    #         type_,
    #         max_n_predictors,
    #         save_path=path,
    #         **prediction_kwargs,
    #     )
    #     best_combination = get_best_combination(best_combinations, winner_according_to)
    #     best_predictors = {
    #         identifier: combination[-1]
    #         for identifier, combination in best_combination.items()
    #     }
    #     save_pickle(best_predictors, path.joinpath("best_predictors.pkl"))
    #     return best_combinations, best_combination, path

    # def best_combination_prediction(
    #     self,
    #     predictors: xr.DataArray,
    #     path: Path | str,
    #     prediction_kwargs: Mapping | None = None,
    # ):
    #     targets_folder = path.parent.parent
    #     targets = open_dataarray(targets_folder.joinpath("length_targets.nc")) > 0
    #     metadata = load_pickle(path.joinpath("metadata.pkl"))
    #     type_ = metadata["type"]
    #     path_to_base_pred = metadata["base_pred"]
    #     if prediction_kwargs is None:
    #         prediction_kwargs = metadata["prediction_kwargs"]
    #     if prediction_kwargs is None:
    #         prediction_kwargs = {}
    #     if path_to_base_pred is not None:
    #         base_pred = open_dataarray(path_to_base_pred)
    #     else:
    #         base_pred = None
    #     best_predictors = load_pickle(path.joinpath("best_predictors.pkl"))
    #     combination = {}
    #     for identifier, predictor_list in tqdm(best_predictors.items()):
    #         thispath = path.joinpath(identifier)
    #         full_pred_fn = thispath.joinpath("full_pred_best.nc")
    #         feature_importances_fn = thispath.joinpath("feature_importances_best.nc")
    #         raw_shap_fn = thispath.joinpath("raw_shap.pkl")
    #         if all(
    #             [
    #                 fn.is_file()
    #                 for fn in [full_pred_fn, feature_importances_fn, raw_shap_fn]
    #             ]
    #         ):
    #             full_pred = open_dataarray(full_pred_fn)
    #             feature_importances = open_dataarray(feature_importances_fn)
    #             raw_shap = load_pickle(raw_shap_fn)
    #             combination[identifier] = (full_pred, feature_importances, raw_shap)
    #             continue
    #         indexer_list = identifier.split("_")
    #         indexer = {}
    #         for indexer_ in indexer_list:
    #             dim, val = indexer_.split("=")
    #             try:
    #                 val = float(val)
    #             except ValueError:
    #                 pass
    #             indexer[dim] = val
    #         if base_pred is None:
    #             base_pred_ = None
    #         else:
    #             base_pred_ = base_pred.loc[indexer].squeeze()
    #         targets_ = targets.loc[indexer].squeeze()
    #         full_pred, feature_importances, raw_shap = predict_all(
    #             predictors.sel(predictor=predictor_list),
    #             targets_,
    #             base_pred_,
    #             type_,
    #             True,
    #             **prediction_kwargs,
    #         )
    #         to_netcdf(full_pred, full_pred_fn)
    #         to_netcdf(feature_importances, feature_importances_fn)
    #         save_pickle(raw_shap, raw_shap_fn)
    #         combination[identifier] = (full_pred, feature_importances, raw_shap)
    #     return combination
