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
