# coding: utf-8
from typing import Tuple, Mapping, Literal, Callable, Sequence
from functools import partial
from itertools import product
from multiprocessing import get_context
from multiprocessing import set_start_method as set_mp_start_method
from pathlib import Path
import warnings
import datetime

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from numba import njit
from tqdm import tqdm, trange
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .definitions import (
    DEFAULT_VALUES,
    YEARS,
    N_WORKERS,
    RESULTS,
    compute,
    do_rle_fill_hole,
    load_pickle,
    save_pickle,
    get_runs_fill_holes,
    xarray_to_polars,
    polars_to_xarray,
    get_index_columns,
    extract_season_from_df,
)
from .data import (
    find_spot,
    get_land_mask,
    extract_season,
    DataHandler,
    compute_anomalies_ds,
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


def spells_from_da(  # TODO: remove, get_spells is the new cool kid
    da: xr.DataArray,
    q: float = 0.95,
    fill_holes: int = 2,
    minlen: np.timedelta64 = np.timedelta64(3, "D"),
    time_before: np.timedelta64 = np.timedelta64(0, "D"),
    time_after: np.timedelta64 = np.timedelta64(0, "D"),
    output_type: Literal["arr"] | Literal["list"] | Literal["both"] = "arr",
) -> xr.DataArray | Tuple[list[np.ndarray]]:
    dt = pd.Timedelta(da.time.values[1] - da.time.values[0])
    months = np.unique(da.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
    days = quantile_exceedence(da, q)
    runs = get_runs_fill_holes(days.values.copy(), hole_size=fill_holes)
    run_times = [da.time.values[run] for run in runs]
    spells = []
    spells_ts = []
    lengths = []
    for run in run_times:
        years = run.astype("datetime64[Y]").astype(int) + 1970
        cond1 = years[0] == years[-1]
        len_ = run[-1] - run[0]
        cond2 = len_ >= minlen
        if not (cond1 and cond2):
            continue
        spells.append([run[0], run[-1]])
        ts_extended = pd.date_range(
            run[0] - time_before, run[-1] + time_after, freq=dt
        ).values
        ts_extended = ts_extended[np.isin(ts_extended, da.time.values)]
        spells_ts.append(ts_extended)
        lengths.append(len_.astype("timedelta64[D]").astype(int))
    spells = np.asarray(spells)
    if output_type == "list":
        return spells_ts, spells
    da_spells = da.copy(data=np.zeros(da.shape, dtype=int))
    indexer = np.concatenate(spells_ts)
    lengths = np.concatenate(
        [np.full(len(spell_ts), len_) for spell_ts, len_ in zip(spells_ts, lengths)]
    )
    da_spells.loc[indexer] = lengths

    if output_type == "arr":
        return da_spells
    return spells_ts, spells, da_spells


def mask_from_spells(
    ds: xr.Dataset,
    spells_ts: list,
    spells: np.ndarray,
    da: xr.DataArray | None = None,
    time_before: np.timedelta64 = np.timedelta64(0, "D"),
) -> xr.Dataset:
    months = np.unique(ds.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
    try:
        lengths = spells[:, 1] - spells[:, 0]
        longest_spell = np.argmax(lengths)
        dt = np.amin(np.unique(np.diff(spells_ts[0])))
        time_around_beg = np.arange(
            -time_before,
            spells[longest_spell, 1] - spells[longest_spell, 0] + dt,
            dt,
            dtype="timedelta64[ns]",
        )
    except ValueError:
        time_around_beg = np.atleast_1d(np.timedelta64(0, "ns"))
    ds_masked = (
        ds.loc[dict(time=ds.time.values[0])]
        .reset_coords("time", drop=True)
        .copy(deep=True)
    )
    ds_masked.loc[dict()] = np.nan
    ds_masked = ds_masked.load()
    ds_masked = ds_masked.expand_dims(
        spell=np.arange(len(spells)),
        time_around_beg=time_around_beg,
    ).copy(deep=True)
    ds_masked = ds_masked.assign_coords(lengths=("spell", lengths))
    dims = list(ds_masked.sizes.values())[:2]
    dummy_da = np.zeros(dims) + np.nan
    ds_masked = ds_masked.assign_coords(
        avg_val=(["spell", "time_around_beg"], dummy_da)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ds_masked = ds_masked.assign_coords(
            absolute_time=(
                ["spell", "time_around_beg"],
                dummy_da.astype("datetime64[h]"),
            )
        )
    if len(ds_masked.time_around_beg) <= 1:
        return ds_masked
    for i, spell in enumerate(spells_ts):
        unexpected_offset = time_before - (spells[i][0] - spell[0])
        this_tab = time_around_beg[: len(spell)] + unexpected_offset
        this_tab = np.clip(
            this_tab, None, ds_masked.time_around_beg.max().values
        )  # Unexpected offset is weird
        spell = spell[: len(this_tab)]
        to_assign = (
            ds.loc[dict(time=spell)]
            .assign_coords(time=this_tab)
            .rename(time="time_around_beg")
        )
        indexer = dict(spell=i, time_around_beg=this_tab)
        ds_masked.loc[indexer] = to_assign
        ds_masked.absolute_time.loc[indexer] = spell
        if da is not None:
            ds_masked.avg_val.loc[indexer] = da.loc[dict(time=spell)].values
    return ds_masked


def get_spells_sigma(
    df: pl.DataFrame, dists: np.ndarray, sigma: int = 1
) -> pl.DataFrame:
    start = 0
    spells = []
    while True:
        start_lab = df[int(start), "labels"]
        next_distance_cond = dists[start_lab, df[start:, "labels"]] > sigma
        if not any(next_distance_cond):
            spells.append(
                {
                    "rel_start": start,
                    "value": start_lab,
                    "len": len(df[start:, "labels"]),
                }
            )
            break
        to_next = np.argmax(next_distance_cond)
        val = df[int(start) : int(start + to_next), ["labels"]].with_columns(
            pl.col("labels").drop_nulls().mode().first().alias("mode")
        )[0, "mode"]
        spells.append({"rel_start": start, "value": val, "len": to_next})
        start = start + to_next
    return pl.DataFrame(spells).with_columns(year=df[0, "year"], my_len=df.shape[0])


def get_persistent_spell_times_from_som(
    labels,
    dists: np.ndarray,
    sigma: float = 0.,
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


def _get_persistent_spell_times_from_som(
    labels_df: pl.DataFrame,
    dists: np.ndarray,
    sigma: int = 0,
    minlen: int = 4,
    nt_before: int = 0,
    nt_after: int = 0,
    nojune: bool = True,
    daily: bool = False,
):
    index_columns = get_index_columns(labels_df)
    index = labels_df[index_columns].unique(maintain_order=True)

    out = labels_df.group_by("year", maintain_order=True).map_groups(
        partial(get_spells_sigma, dists=dists, sigma=sigma)
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
    )
    out = out.with_columns(
        range=pl.int_ranges(
            pl.col("start") - nt_before, pl.col("start") + pl.col("len") + nt_after
        ),
        relative_index=pl.int_ranges(
            -nt_before, pl.col("len") + nt_after, dtype=pl.Int16
        ),
    )
    out = out.with_row_index("spell").explode(["range", "relative_index"])
    out = out.filter(pl.col("range") < len(index), pl.col("range") >= 0)
    out = out.with_columns(index[out["range"]])
    out = out.filter(pl.col("len") >= minlen)
    out = out.with_columns(pl.col("spell").rle_id())
    out = (
        out.group_by("spell", maintain_order=True)
        .agg(
            [
                pl.col(col).filter(
                    pl.col("time").dt.year()
                    == pl.col("time")
                    .dt.year()
                    .get(pl.arg_where(pl.col("relative_index") == 0).first())
                )
                for col in ["time", "relative_index", "value", "len"]
            ]
        )
        .explode(["time", "relative_index", "value", "len"])
    )
    if nojune:
        june_filter = out.group_by("spell", maintain_order=True).agg(
            (pl.col("time").dt.ordinal_day() <= 160).sum() > 0.8
        )["time"]
        out = out.filter(pl.col("spell").is_in(june_filter.not_().arg_true()))
        out = out.with_columns(pl.col("spell").rle_id())
    out = out.with_columns(
        out.group_by("spell", maintain_order=True)
        .agg(
            relative_time=pl.col("time")
            - pl.col("time").get(pl.arg_where(pl.col("relative_index") == 0).first())
        )
        .explode("relative_time")
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


def get_spells(
    df,
    varname: str,
    fill_holes: datetime.timedelta = datetime.timedelta(hours=12),
    minlen: datetime.timedelta = datetime.timedelta(days=3),
    time_before: datetime.timedelta = datetime.timedelta(0),
    time_after: datetime.timedelta = datetime.timedelta(0),
    q: float = 0.95,
    daily: bool = False,
):
    times = df["time"].unique()
    dt = times[1] - times[0]
    if varname in ["com_speed", "speed"]:
        expr = pl.col(varname) < df[varname].quantile(1 - q)
    else:
        expr = pl.col(varname) > df[varname].quantile(q)
    out = do_rle_fill_hole(df, expr, None, fill_holes)
    minlen = int(minlen / dt)
    nt_before, nt_after = int(time_before / dt), int(time_after / dt)
    out = out.filter(pl.col("value"), pl.col("len") >= minlen).with_columns(
        range=pl.int_ranges(
            pl.col("start") - nt_before, pl.col("start") + pl.col("len") + nt_after
        ),
        relative_index=pl.int_ranges(-nt_before, pl.col("len") + nt_after),
    )
    out = out.with_row_index("spell").explode(["range", "relative_index"])
    indices = out["range"].to_numpy()
    mask_valid = (indices < len(times)) & (indices >= 0)
    indices = indices[mask_valid]
    out = (
        out.filter(mask_valid)
        .with_columns(time=times[indices])
        .drop("value", "start", "range")
        .sort("spell", "relative_index")
        .cast({"time": pl.Datetime("ns")})
        .with_columns(pl.col("spell").rle_id())
    )
    out = out.with_columns(
        out.group_by("spell", maintain_order=True)
        .agg(
            relative_time=pl.col("time")
            - pl.col("time").gather(pl.arg_where(pl.col("relative_index") == 0)).first()
        )
        .explode("relative_time")
    )
    min_rel_index = out["relative_index"].min()
    max_rel_index = out["relative_index"].max()
    out = out.filter(
        pl.col("relative_time") >= pl.duration(days=min_rel_index),
        pl.col("relative_time") <= pl.duration(days=max_rel_index),
    )

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


def get_persistent_jet_spells(
    props_as_df,
    metric: str,
    jet: str | None = None,
    season: list | str | None = None,
    **kwargs,
):
    props_as_df = extract_season_from_df(props_as_df, season)
    onejet = props_as_df.filter(pl.col("jet") == jet)
    return get_spells(onejet, metric, **kwargs)


def subset_around_onset(df, around_onset: int | datetime.timedelta | None = None):
    if isinstance(around_onset, int) and "relative_index" in df.columns:
        df = df.filter(pl.col("relative_index").abs() <= around_onset)
    elif isinstance(around_onset, datetime.timedelta) and "relative_time" in df.columns:
        df = df.filter(pl.col("relative_time").abs() <= around_onset)
    return df


def mask_from_spells_pl(
    spells: pl.DataFrame,
    to_mask: xr.DataArray | xr.Dataset | pl.DataFrame,
    force_pl: bool = False,
):  # TODO: add time_before
    unique_times_spells = spells["time"].unique().to_numpy()
    unique_times_to_mask = np.unique(to_mask["time"].to_numpy())
    unique_times = np.intersect1d(unique_times_spells, unique_times_to_mask)
    spells = spells.filter(pl.col("time").is_in(unique_times))
    if isinstance(to_mask, xr.DataArray | xr.Dataset):
        to_mask = compute(to_mask.sel(time=unique_times), progress=True)
        to_mask = xarray_to_polars(to_mask)
        index_columns_xarray = get_index_columns(
            to_mask, ["lat", "lon", "jet", "jet ID", "cluster"]
        )
    else:
        to_mask = to_mask.cast({"time": pl.Datetime("ns")})
        to_mask = to_mask.filter(pl.col("time").is_in(unique_times))
        index_columns_xarray = None
    to_mask = to_mask.cast({"time": pl.Datetime("ns")})
    spells = spells.cast(
        {"time": pl.Datetime("ns"), "relative_time": pl.Duration("ns")}
    )
    index_columns = get_index_columns(to_mask, ["member", "time"])
    masked = spells.join(to_mask, on=index_columns)
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


def mask_from_da(
    da: xr.DataArray,
    ds: xr.Dataset,
    q: float = 0.95,
    fill_holes: bool = False,
    minlen: np.timedelta64 = np.timedelta64(3, "D"),
    time_before: pd.Timedelta = pd.Timedelta(0, "D"),
    time_after: pd.Timedelta = pd.Timedelta(0, "D"),
) -> xr.Dataset:
    spells_ts, spells = spells_from_da(
        da, q, fill_holes, minlen, time_before, time_after, output_type="list"
    )
    return mask_from_spells(ds, spells_ts, spells, da, time_before)


def mask_from_spells_multi_region(
    timeseries: xr.DataArray | xr.Dataset,
    targets: xr.DataArray,
    all_spells_ts: list,
    all_spells: list,
    time_before: pd.Timedelta = pd.Timedelta(0, "D"),
):
    all_masked_ts = []
    for spells_ts, spells, target in zip(
        all_spells_ts, all_spells, targets.transpose("region", ...)
    ):
        masked_ts = mask_from_spells(
            timeseries,
            spells_ts,
            spells,
            target,
            time_before=time_before,
        )
        all_masked_ts.append(masked_ts)
    regions = targets.region.values
    regions = xr.DataArray(regions, name="region", dims="region")
    all_masked_ts = xr.concat(all_masked_ts, dim=regions)
    return all_masked_ts


def quantile_exceedence(  # 2 directional based on quantile above or below 0.5
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


def _add_timescales(predictors, timescales: Sequence, indexer: Mapping):
    for timescale in timescales[1:]:
        indexer["timescale"] = timescale
        predictors.loc[indexer] = (
            predictors.loc[indexer].rolling(time=timescale, center=False).mean()
        )
    return predictors


def add_timescales_to_predictors(
    predictors, timescales: Sequence, yearbreak: bool = True
):
    # TODO: polarize with .join
    if 1 not in timescales:
        timescales.append(1)
    timescales.sort()
    predictors = predictors.expand_dims(axis=-1, **dict(timescale=timescales)).copy(
        deep=True
    )
    if not yearbreak:
        return _add_timescales(predictors, timescales, {})
    for year in np.unique(predictors.time.dt.year):
        year_mask = predictors.time.dt.year == year
        indexer = {"time": year_mask}
        predictors.loc[indexer] = _add_timescales(
            predictors, timescales, indexer.copy()
        ).loc[indexer]
    return predictors


def _add_lags(predictors, lags: Sequence, indexer: Mapping):
    for lag in lags[1:]:
        indexer["lag"] = lag
        predictors.loc[indexer] = predictors.loc[indexer].shift(time=lag)
    return predictors


def add_lags_to_predictors(predictors, lags: Sequence, yearbreak: bool = True):
    # TODO: polarize with group_by + shift
    if 0 not in lags:
        lags.append(0)
    lags.sort()
    predictors = predictors.expand_dims(axis=-1, **dict(lag=lags)).copy(deep=True)
    if not yearbreak:
        return _add_lags(predictors, lags, {})
    for year in np.unique(predictors.time.dt.year):
        year_mask = predictors.time.dt.year == year
        indexer = {"time": year_mask}
        predictors.loc[indexer] = _add_lags(predictors, lags, indexer.copy()).loc[
            indexer
        ]
    return predictors


def prepare_predictors(
    predictors,
    subset: list = None,
    anomalize: bool = False,
    normalize: bool = False,
    detrend: bool = False,
    nan_method: Literal["fill"] | Literal["linear"] | Literal["nearest"] = "fill",
    season: str | list | None = None,
    timescales: Sequence | None = None,
    lags: Sequence | None = None,
) -> xr.DataArray:
    if subset is not None:
        predictors = predictors[subset]
    if anomalize:
        predictors = compute_anomalies_ds(predictors, "dayofyear", normalize)
    if nan_method == "fill":
        for varname in predictors.data_vars:
            predictors[varname] = predictors[varname].fillna(DEFAULT_VALUES[varname])
    elif nan_method in ["linear", "nearest"]:
        predictors = predictors.interpolate_na(
            "time", method=nan_method, fill_value="extrapolate"
        )
    else:
        print("Wrong nan method")
        raise ValueError("Wrong nan method")

    if detrend:
        for varname in predictors.data_vars:
            p = predictors[varname].polyfit(dim="time", deg=1)
            fit = xr.polyval("time", p.polyfit_coefficients)
            predictors[varname] = predictors[varname] - fit

    predictors = predictors.to_array(dim="varname", name="predictors")
    yearbreak = np.all(
        np.isin(np.arange(1, 13), np.unique(predictors.time.dt.month.values))
    )
    if timescales is not None and len(timescales) > 1:
        predictors = add_timescales_to_predictors(predictors, timescales, yearbreak)
    dims_to_stack = ["varname"]
    for potential in ["jet", "timescale"]:
        if potential in predictors.dims:
            dims_to_stack.append(potential)
    predictors = predictors.stack(predictor=dims_to_stack)
    newindex = ["_".join([str(t) for t in ts]) for ts in predictors.predictor.values]
    dims_to_stack.append("predictor")
    predictors = predictors.assign_coords(predictor_name=("predictor", newindex))
    predictors = predictors.set_index(predictor="predictor_name")
    if lags is not None and len(lags) > 1:
        predictors = add_lags_to_predictors(predictors, lags, yearbreak)
    predictors = extract_season(predictors, season)
    return predictors


def augment_targets(targets, timescales):
    targets = targets.expand_dims(axis=-1, **dict(timescale=timescales)).copy(deep=True)
    for timescale in tqdm(timescales[1:]):
        for year in np.unique(targets.time.dt.year):
            year_mask = targets.time.dt.year == year
            indexer = {"time": year_mask, "timscale": timescale}
            targets.loc[indexer] = (
                targets.loc[indexer]
                .rolling(time=timescale, center=False)
                .mean()
                .shift(time=-timescale + 1)
            )
    return targets


def create_all_triplets(
    predictors: xr.DataArray, targets: xr.DataArray, lags: Sequence
):
    if 0 not in lags:
        lags.append(0)
    lags.sort()
    triplets = []

    for predictor in predictors.T:
        for lag in lags:
            triplets.append(((predictor.predictor.item(), lag), predictor, targets))
    return triplets


def stacked_lstsq(
    L, b, rcond=1e-10
):  # https://stackoverflow.com/questions/30442377/how-to-solve-many-overdetermined-systems-of-linear-equations-using-vectorized-co
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s >= s_min]
    x = np.einsum(
        "...ji,...j->...i", v, inv_s * np.einsum("...ji,...j->...i", u, b.conj())
    )
    return np.conj(x, x)


def compute_r(triplet, season: str | list | None = "JJA") -> np.ndarray:
    (predictor, lag), predictor_, target_ = triplet
    if "timescale" in target_.dims:
        timescale = target_.timescale.item()
    else:
        timescale = 1
    shape = target_.shape
    predictors, targets = [], []
    for year in YEARS:
        predictor = extract_season(
            predictor_.sel(time=predictor_.time.dt.year == year), season
        ).values
        predictor_resid = np.linalg.lstsq(
            predictor[timescale:, None], predictor[:-timescale], rcond=None
        )[0][0]
        predictor[:-timescale] = (
            predictor[:-timescale] - predictor[timescale:] * predictor_resid
        )

        target = extract_season(
            target_.sel(time=target_.time.dt.year == year), season
        ).values
        target = target.reshape(target.shape[0], -1)
        target_resid = stacked_lstsq(target[timescale:, None], target[:-timescale])
        target[:-timescale] = target[:-timescale] - target[timescale:] * target_resid
        target = target.reshape(target.shape[0], *shape[-2:])

        if lag > 0:
            predictor = predictor[lag:]
            target = target[:-lag]

        predictors.append(predictor)
        targets.append(target)
    predictor = np.concatenate(predictors, axis=0)
    target = np.concatenate(targets, axis=0)
    r = np.sum(predictor[:, None, None] * target, axis=0) / np.sqrt(
        np.sum(predictor[:, None, None] ** 2, axis=0) * np.sum(target**2, axis=0)
    )
    return r


def compute_all_responses(
    predictors: xr.DataArray, targets: xr.DataArray, lags: Sequence | None = None
) -> xr.DataArray:
    if "lag" in predictors.dims:
        if lags is None:
            lags = predictors.lags.values
        predictors = predictors[dict(lag=0)].reset_coords("lag", drop=True)
    if lags is None:
        lags = [0]
    coords = {
        "predictor": predictors.predictor.values,
        "lag": lags,
        "lat": targets.lat.values,
        "lon": targets.lon.values,
    }
    shape = [len(c) for c in coords.values()]
    corr_da = xr.DataArray(np.zeros(shape), coords=coords)
    triplets = create_all_triplets(predictors, targets, lags)
    ctx = get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        all_r = list(
            tqdm(
                pool.imap(compute_r, triplets, chunksize=1),
                total=len(triplets),
            )
        )
    # all_r = list(map(compute_r, triplets))
    all_r = np.asarray(all_r)
    corr_da[:] = all_r.reshape(shape)
    return corr_da


def compute_all_scores(y_test, y_pred, y_pred_prob) -> Mapping:
    scores = {}
    for scorename, scorefunc in ALL_SCORES.items():
        if scorename in ["roc_auc_score", "brier_score_loss"]:
            scores[scorename] = scorefunc(y_test, y_pred_prob)
        else:
            scores[scorename] = scorefunc(y_test, y_pred)
    return scores


def regress_against_time(targets: xr.DataArray) -> xr.DataArray:
    targets = targets.transpose("time", ...)
    X_base = (
        (targets.time.values - targets.time.values[0])
        .astype("timedelta64[h]")
        .astype(int)[:, None]
    )

    extra_dims = {dim: targets[dim].values for dim in targets.dims if dim != "time"}

    base_pred = targets.copy(data=np.zeros(targets.shape, dtype=np.float32))
    creation_template = (list(extra_dims), np.zeros(*targets.shape[1:]))
    to_assign = {score_name: creation_template for score_name in ALL_SCORES}
    base_pred = base_pred.assign_coords(to_assign)
    for vals in product(*list(extra_dims.values())):
        indexer = {dim: val for dim, val in zip(extra_dims, vals)}
        if not targets.loc[indexer].any("time"):
            continue
        y = targets.loc[indexer].values
        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2)
        lr = LogisticRegression(class_weight=None).fit(X_train, y_train)
        y_pred_prob = lr.predict_proba(X_test)[:, 1]
        y_pred = lr.predict(X_test)
        base_pred.loc[indexer] = lr.predict_proba(X_base)[:, 1]
        scores = compute_all_scores(y_test, y_pred, y_pred_prob)
        for scorename, score in scores.items():
            base_pred[scorename].loc[indexer] = score
    return base_pred


def predict_all(
    predictors: xr.DataArray,
    orig_targets: xr.DataArray,
    base_pred: xr.DataArray | None = None,
    type_: Literal["rf", "lr", "xgb"] = "rf",
    compute_shap: bool = False,
    n_folds: int = 1,
    save_path: Path | None = None,
    **kwargs,
):
    # Handle base_pred
    if type_ == "lr" and base_pred is not None:
        print(f"Base pred is incompatible with {type_}, ignoring")
        base_pred = None
    orig_targets = orig_targets.transpose("time", ...)
    if base_pred is not None:
        base_pred = base_pred.transpose("time", ...)
        targets = orig_targets - base_pred
    else:
        targets = orig_targets
        base_pred = orig_targets.copy(data=np.zeros(orig_targets.shape))
    # Prepare full_pred and iterator
    full_pred = targets.copy(data=np.zeros(targets.shape, dtype=np.float32))
    extra_dims = {dim: targets[dim].values for dim in targets.dims if dim != "time"}
    if "lag" in predictors.dims:
        lags = predictors.lag.values
        full_pred = full_pred.expand_dims(axis=-1, **dict(lag=lags))
        extra_dims["lag"] = lags
    if n_folds > 1:
        full_pred = full_pred.expand_dims(axis=-1, **dict(fold=np.arange(n_folds)))
        extra_dims["fold"] = np.arange(n_folds)
    # Create score coordinates
    if len(extra_dims) == 0:
        creation_template = 0.0
    else:
        creation_template = (list(extra_dims), np.zeros(full_pred.shape[1:]))
    to_assign = {score_name: creation_template for score_name in ALL_SCORES}
    full_pred = full_pred.assign_coords(to_assign).copy(deep=True)
    # Prepare feature importance da
    if type_ == "xgb" and "importance_type" not in kwargs:
        kwargs["importance_type"] = "cover"
    if "importance_type" in kwargs:
        importance_type = kwargs["importance_type"]
    elif type_ == "rf":
        importance_type = "impurity"
    else:
        importance_type = "coefs"
    importance_coords = {
        "type": [
            "correlation",
            importance_type,
            "permutation",
            "mean_shap",
            "mean_abs_shap",
        ],
        "predictor": predictors.predictor.values,
    } | extra_dims

    feature_importances = xr.DataArray(
        np.zeros([len(c) for c in importance_coords.values()]), coords=importance_coords
    )
    # Prepare loop
    raw_shap = {}
    len_ = np.prod([len(co) for co in extra_dims.values()])
    if len_ > 2:
        iter_ = tqdm(product(*list(extra_dims.values())), total=len_)
    else:
        iter_ = product(*list(extra_dims.values()))
    for vals in iter_:
        indexer = {dim: val for dim, val in zip(extra_dims, vals)}
        indexer_str = ["=".join((dim, str(val))) for dim, val in indexer.items()]
        indexer_str = "_".join(indexer_str)
        # Check if we actually need to proceed, load ofile otherwise
        if save_path is not None:
            full_pred_fn = save_path.joinpath(f"pred_{indexer_str}.nc")
            feature_importance_fn = save_path.joinpath(f"importance_{indexer_str}.nc")
            raw_shap_fn = save_path.joinpath(f"shap_{indexer_str}.pkl")
            if (
                full_pred_fn.is_file()
                and feature_importance_fn.is_file()
                and raw_shap_fn.is_file()
            ):
                this_full_pred = open_dataarray(full_pred_fn)
                full_pred.loc[indexer] = this_full_pred
                for score in ALL_SCORES:
                    full_pred[score].loc[indexer] = this_full_pred[score].item()
                feature_importances.loc[indexer] = open_dataarray(feature_importance_fn)
                raw_shap[indexer_str] = load_pickle(raw_shap_fn)
                continue
        # Sub indexers from main, main for output, one for targets, one for predictors
        indexer_targets = indexer.copy()
        try:
            lag = indexer_targets.pop("lag")
            indexer_pred = {"lag": lag}
            predictors_ = predictors.loc[indexer_pred]
        except KeyError:
            predictors_ = predictors 
        # Turn to pandas so shap keeps predictor names and makes easier to understand plots
        predictors_ = predictors_.transpose("time", ...)
        X = predictors_.drop_vars(
            ["varname", "jet", "timescale"], errors="ignore"
        ).to_pandas()
        if not targets.loc[indexer_targets].any("time"):
            continue
        y = targets.loc[indexer_targets].to_pandas()
        y_orig = orig_targets.loc[indexer_targets].to_pandas()
        y_base = base_pred.loc[indexer_targets].to_pandas()
        X_train, X_test, y_train, y_test, _, y_orig_test, _, y_base_test = (
            train_test_split(X, y, y_orig, y_base, test_size=0.2)
        )
        # Cannot handle the two cases the same way. Do prediction, prepare test and full outputs
        if base_pred.mean() == 0:
            if "class_weight" not in kwargs and type_ == "lr":
                kwargs["class_weight"] = "balanced"
            if type_ == "lr":
                model = LogisticRegression(n_jobs=1, **kwargs).fit(X_train, y_train)
            elif type_ == "rf":
                model = RandomForestClassifier(n_jobs=1, **kwargs).fit(X_train, y_train)
            elif type_ == "xgb":
                from xgboost import XGBClassifier

                model = XGBClassifier(n_jobs=1, **kwargs).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            full_pred.loc[indexer] = model.predict_proba(X)[:, 1]
        else:
            if type_ == "rf":
                model = RandomForestRegressor(n_jobs=1, **kwargs).fit(X_train, y_train)
            elif type_ == "xgb":
                from xgboost import XGBRegressor

                model = XGBRegressor(n_jobs=1, **kwargs).fit(X_train, y_train)
            y_pred_prob = model.predict(X_test) + y_base_test
            y_pred_prob = np.clip(y_pred_prob, 0, 1)
            y_pred = y_pred_prob > 0.5
            full_pred.loc[indexer] = model.predict(X) + y_base
        # Compute scores
        scores = compute_all_scores(y_orig_test, y_pred, y_pred_prob)
        for scorename, score in scores.items():
            full_pred[scorename].loc[indexer] = score
        # Predictor importances
        # 1. Correlation
        X = X.values
        y = y.values
        corr = (
            np.mean((X - X.mean(axis=0)[None, :]) * (y - y.mean())[:, None], axis=0)
            / np.std(X, axis=0)[None, :]
            / np.std(y)
        )
        feature_importances.loc[dict(type="correlation", **indexer)] = corr.squeeze()
        # 2. Impurity (rf) or regression coefficients (lr; should be exponentiated)
        if type_ in ["rf", "xgb"]:
            imp = model.feature_importances_
        elif type_ == "lr":
            imp = model.coef_.ravel()
        feature_importances.loc[dict(type=importance_type, **indexer)] = imp
        # 3. Permutation importance
        r = permutation_importance(
            model, X_test, y_test, n_repeats=30, random_state=0, n_jobs=1
        )
        feature_importances.loc[dict(type="permutation", **indexer)] = r[
            "importances_mean"
        ]
        # 3. Potentially shap (mean and mean_abs)
        if compute_shap:
            if type_ == "xgb":
                # to solve UnicodeDecodeError
                mybooster = model.get_booster()
                model_bytearray = mybooster.save_raw()[4:]

                def myfun(self=None):
                    return model_bytearray

                mybooster.save_raw = myfun
            try:
                from fasttreeshap import TreeExplainer, Explainer
            except ModuleNotFoundError:
                from shap import TreeExplainer, Explainer
            shap_explainer = TreeExplainer if type_ == "rf" else Explainer
            shap_ = shap_explainer(model)(X, y, check_additivity=False)
            raw_shap[indexer_str] = shap_
            shap_values = shap_.values
            feature_importances.loc[dict(type="mean_shap", **indexer)] = (
                shap_values.mean(axis=0)
            )
            feature_importances.loc[dict(type="mean_abs_shap", **indexer)] = np.abs(
                shap_values
            ).mean(axis=0)
        # Save because this is very slow, ~2 minutes per iteration without shap, 4+ with
        if save_path is not None:
            to_netcdf(full_pred.loc[indexer], full_pred_fn)
            to_netcdf(feature_importances.loc[indexer], feature_importance_fn)
            save_pickle(raw_shap[indexer_str], raw_shap_fn)
    return full_pred, feature_importances, raw_shap


def multi_combination_prediction(
    predictors: xr.DataArray,
    targets: xr.DataArray,
    base_pred: xr.DataArray | None = None,
    type_: Literal["rf", "lr"] = "rf",
    max_n_predictors: int = 10,
    save_path: Path | None = None,
    **kwargs,
):
    predictor_names = predictors.predictor.values
    extra_dims = {dim: targets[dim].values for dim in targets.dims if dim != "time"}
    if "lag" in predictors.dims:
        lags = predictors.lag.values
        extra_dims["lag"] = lags
    if save_path is None:
        identifier = [f"{len(co)}{dim}" for dim, co in extra_dims.items()]
        identifier = "_".join(identifier)
        save_path = Path(RESULTS, "multi_prediction", type_, identifier)
        save_path.mkdir(mode=0o777, parents=True, exist_ok=True)
    flat_len = np.prod([len(co) for co in extra_dims.values()])
    best_combinations = {}
    for vals in tqdm(product(*list(extra_dims.values())), total=flat_len):
        indexer = {dim: val for dim, val in zip(extra_dims, vals)}
        indexer_str = ["=".join((dim, str(val))) for dim, val in indexer.items()]
        indexer_str = "_".join(indexer_str)
        thispath = save_path.joinpath(indexer_str)
        thispath.mkdir(mode=0o777, parents=True, exist_ok=True)
        targets_ = targets.loc[indexer].squeeze()
        if base_pred is None:
            base_pred_ = None
        else:
            base_pred_ = base_pred.loc[indexer].squeeze()
        try:
            lag = indexer.pop("lag")
            indexer_pred = {"lag": lag}
            predictors_ = predictors.loc[indexer_pred]
        except KeyError:
            predictors_ = predictors
        combinations = [
            [
                predictor,
            ]
            for predictor in predictor_names
        ]
        best_combinations[indexer_str] = {}
        for n_predictors in trange(1, max_n_predictors + 1, leave=False):
            full_pred_fn = thispath.joinpath(f"full_pred_{n_predictors}")
            feature_importances_fn = thispath.joinpath(
                f"feature_importances_{n_predictors}"
            )
            if full_pred_fn.is_file() and feature_importances_fn.is_file():
                full_pred = open_dataarray(full_pred_fn)
                feature_importances = open_dataarray(feature_importances_fn)
                this_combination = feature_importances.predictor.values
                best_combinations[indexer_str][n_predictors] = (
                    full_pred,
                    feature_importances,
                    this_combination,
                )
                combinations = [
                    [*this_combination, predictor]
                    for predictor in predictor_names
                    if predictor not in this_combination
                ]
                continue
            full_pred_list = []
            feature_importance_list = []
            for predictor_list in combinations:
                full_pred, feature_importances, _ = predict_all(
                    predictors_.sel(predictor=predictor_list),
                    targets_,
                    base_pred_,
                    type_,
                    compute_shap=True,
                    **kwargs,
                )
                full_pred_list.append(full_pred)
                feature_importance_list.append(feature_importances)
            imax = np.argmax(
                [full_pred.roc_auc_score.item() for full_pred in full_pred_list]
            )

            best_combinations[indexer_str][n_predictors] = (
                full_pred_list[imax],
                feature_importance_list[imax],
                combinations[imax],
            )
            to_netcdf(full_pred_list[imax], full_pred_fn)
            to_netcdf(feature_importance_list[imax], feature_importances_fn)
            combinations = [
                [*combinations[imax], predictor]
                for predictor in predictor_names
                if predictor not in combinations[imax]
            ]
    return best_combinations


def get_best_combination(
    best_combinations: Mapping,
    according_to: str,
):
    best_combination = {}
    for identifier, best_combinations_ in best_combinations.items():
        df = []
        for n in best_combinations_:
            bcs = best_combinations_[n][0]
            df.append({scorekey: bcs[scorekey].item() for scorekey in ALL_SCORES})
        df = pd.DataFrame(df)
        imax = 1 + np.argmax(df[according_to])
        best_combination[identifier] = best_combinations[identifier][imax]
    return best_combination


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
        thispath = self.pred_path
        thispath = find_spot(thispath, metadata)
        ofiles = [
            "targets.nc",
            "length_targets.nc",
            "all_spells_ts.pkl",
            "all_spells.pkl",
        ]
        ofiles = [thispath.joinpath(ofile) for ofile in ofiles]
        if all([ofile.is_file() for ofile in ofiles]):
            if return_folder:
                return thispath
            to_ret = []
            for ofile in ofiles:
                if ofile.suffix == ".nc":
                    to_ret.append(open_dataarray(ofile))
                elif ofile.suffix == ".pkl":
                    to_ret.append(load_pickle(ofile))
            return tuple(to_ret)
        clusters_da = self.spatial_clusters_as_da(n_clu)

        targets = xr.DataArray(
            np.zeros((len(self.da.time), n_clu)),
            coords={"time": self.da.time.values, "region": np.arange(n_clu)},
        )
        for i_clu in trange(n_clu):
            targets.loc[:, i_clu] = compute(
                self.da.where(clusters_da == i_clu).mean(["lon", "lat"])
            )
        targets = extract_season(targets, self.season)
        length_targets = targets.copy(data=np.zeros(targets.shape, dtype=int))
        all_spells = []
        all_spells_ts = []
        for i_clu in trange(n_clu):
            targets_ = targets[:, i_clu]
            if simple:
                da_spells = quantile_exceedence(targets_, q)
            else:
                spells_ts, spells, da_spells = spells_from_da(
                    targets_,
                    q,
                    output_type="both",
                    **kwargs,
                )
                all_spells_ts.append(spells_ts)
                all_spells.append(spells)
            length_targets[:, i_clu] = da_spells
        to_ret = targets, length_targets, all_spells_ts, all_spells
        for to_save, ofile in zip(to_ret, ofiles):
            if ofile.suffix == ".nc":
                to_netcdf(to_save, ofile)
            elif ofile.suffix == ".pkl":
                save_pickle(to_save, ofile)
        if return_folder:
            return thispath
        return to_ret

    def mask_timeseries(
        self,
        timeseries: xr.DataArray | xr.Dataset,
        n_clu: int,
        i_clu: int | Sequence[int] | Literal["all"] = "all",
        q: float | None = None,
        simple: bool = False,
        **kwargs,
    ):
        targets, _, all_spells_ts, all_spells = self.create_targets(
            n_clu, i_clu, q, simple, **kwargs
        )
        if "time_before" in kwargs:
            new_kwargs = {"time_before": kwargs["time_before"]}
        else:
            new_kwargs = {}
        return mask_from_spells_multi_region(
            timeseries, targets, all_spells_ts, all_spells, **new_kwargs
        )

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

    def multi_combination_prediction(
        self,
        predictors: xr.DataArray,
        create_target_kwargs: Mapping,
        type_: Literal["rf", "lr"] = "rf",
        do_base_pred: bool = True,
        max_n_predictors: int = 10,
        prediction_kwargs: Mapping | None = None,
        winner_according_to: str = "roc_auc_score",
    ):
        """
        Wrappy Wrappy wrap
        """
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
        metadata = {
            "predictors": predictor_names.tolist(),
            "type": type_,
            "base_pred": path_to_base_pred,  # or None
            "prediction_kwargs": prediction_kwargs,
        }
        if prediction_kwargs is None:
            prediction_kwargs = {}
        path = targets_folder.joinpath("multi_combination")
        path.mkdir(mode=0o777, parents=True, exist_ok=True)
        path = find_spot(path, metadata)
        best_combinations = multi_combination_prediction(
            predictors,
            targets,
            base_pred,
            type_,
            max_n_predictors,
            save_path=path,
            **prediction_kwargs,
        )
        best_combination = get_best_combination(best_combinations, winner_according_to)
        best_predictors = {
            identifier: combination[-1]
            for identifier, combination in best_combination.items()
        }
        save_pickle(best_predictors, path.joinpath("best_predictors.pkl"))
        return best_combinations, best_combination, path

    def best_combination_prediction(
        self,
        predictors: xr.DataArray,
        path: Path | str,
        prediction_kwargs: Mapping | None = None,
    ):
        targets_folder = path.parent.parent
        targets = open_dataarray(targets_folder.joinpath("length_targets.nc")) > 0
        metadata = load_pickle(path.joinpath("metadata.pkl"))
        type_ = metadata["type"]
        path_to_base_pred = metadata["base_pred"]
        if prediction_kwargs is None:
            prediction_kwargs = metadata["prediction_kwargs"]
        if prediction_kwargs is None:
            prediction_kwargs = {}
        if path_to_base_pred is not None:
            base_pred = open_dataarray(path_to_base_pred)
        else:
            base_pred = None
        best_predictors = load_pickle(path.joinpath("best_predictors.pkl"))
        combination = {}
        for identifier, predictor_list in tqdm(best_predictors.items()):
            thispath = path.joinpath(identifier)
            full_pred_fn = thispath.joinpath("full_pred_best.nc")
            feature_importances_fn = thispath.joinpath("feature_importances_best.nc")
            raw_shap_fn = thispath.joinpath("raw_shap.pkl")
            if all(
                [
                    fn.is_file()
                    for fn in [full_pred_fn, feature_importances_fn, raw_shap_fn]
                ]
            ):
                full_pred = open_dataarray(full_pred_fn)
                feature_importances = open_dataarray(feature_importances_fn)
                raw_shap = load_pickle(raw_shap_fn)
                combination[identifier] = (full_pred, feature_importances, raw_shap)
                continue
            indexer_list = identifier.split("_")
            indexer = {}
            for indexer_ in indexer_list:
                dim, val = indexer_.split("=")
                try:
                    val = float(val)
                except ValueError:
                    pass
                indexer[dim] = val
            if base_pred is None:
                base_pred_ = None
            else:
                base_pred_ = base_pred.loc[indexer].squeeze()
            targets_ = targets.loc[indexer].squeeze()
            full_pred, feature_importances, raw_shap = predict_all(
                predictors.sel(predictor=predictor_list),
                targets_,
                base_pred_,
                type_,
                True,
                **prediction_kwargs,
            )
            to_netcdf(full_pred, full_pred_fn)
            to_netcdf(feature_importances, feature_importances_fn)
            save_pickle(raw_shap, raw_shap_fn)
            combination[identifier] = (full_pred, feature_importances, raw_shap)
        return combination
