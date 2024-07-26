from typing import Union, Tuple, Iterable, Literal, Callable, Sequence
from functools import partial
from itertools import combinations, product
from multiprocessing import Pool
from pathlib import Path
from nptyping import NDArray
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import pairwise_distances
from xclim.indices.run_length import rle, run_bounds
from tqdm import tqdm
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
import shap

from jetstream_hugo.definitions import (
    DEFAULT_VALUES,
    YEARS,
    N_WORKERS,
)
from jetstream_hugo.data import (
    get_land_mask,
    extract_season,
    DataHandler,
    SEASONS,
)


def spells_from_da(
    da: xr.DataArray,
    q: float = 0.95,
    fill_holes: bool = False,
    minlen: np.timedelta64 = np.timedelta64(3, "D"),
    time_before: pd.Timedelta = pd.Timedelta(0, "D"),
    time_after: pd.Timedelta = pd.Timedelta(0, "D"),
    output_type: Literal["arr"] | Literal["list"] | Literal["both"] = "arr",
) -> xr.DataArray | Tuple[list[NDArray]]:
    dt = pd.Timedelta(da.time.values[1] - da.time.values[0])
    months = np.unique(da.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
    days = (
        (da > da.quantile(dim="time", q=q))
        if q > 0.5
        else (da < da.quantile(dim="time", q=q))
    )
    if fill_holes:
        holes = rle(~days)
        holes = np.where(holes.values == 1)[0]
        days[holes] = 1
    spells = run_bounds(days)
    mask = (spells[0].dt.year == spells[1].dt.year).values
    spells = spells[:, mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mask = spells.astype("datetime64[h]").values
    mask = (mask[1] - mask[0]) >= minlen
    spells = spells[:, mask].T
    spells_ts = []
    lengths = []
    for spell in spells:
        hw_len = (spell[1] - spell[0]).values.astype("timedelta64[D]")
        year = spell[0].dt.year.values
        min_time = np.datetime64(f"{year}-{months[0]}-01T00:00")
        max_time = np.datetime64(f"{year}-{months[1]}-01T00:00") - dt
        first_time = max(min_time, (spell[0] - time_before).values)
        last_time = min(max_time, (spell[1] + time_after).values)
        spell_ts = pd.date_range(first_time, last_time, freq="6h")
        spells_ts.append(spell_ts)
        lengths.append(np.full(len(spell_ts), hw_len.astype(int)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        spells = spells.astype("datetime64[h]").values
    if output_type == "list":
        return spells_ts, spells
    da_spells = da.copy(data=np.zeros(da.shape, dtype=int))
    da_spells.loc[np.concatenate(spells_ts)] = np.concatenate(lengths)
    if output_type == "arr":
        return da_spells
    return spells_ts, spells, da_spells


def mask_from_spells(
    da: xr.DataArray,
    ds: xr.Dataset,
    spells_ts,
    spells,
    time_before: pd.Timedelta = pd.Timedelta(0, "D"),
) -> xr.Dataset:
    months = np.unique(ds.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
    lengths = spells[:, 1] - spells[:, 0]
    longest_hotspell = np.argmax(lengths)
    time_around_beg = spells_ts[longest_hotspell] - spells[longest_hotspell, 0]
    time_around_beg = time_around_beg.values
    ds_masked = (
        ds.loc[dict(time=ds.time.values[0])]
        .reset_coords("time", drop=True)
        .copy(deep=True)
    )
    ds_masked.loc[dict()] = np.nan
    ds_masked = ds_masked.expand_dims(
        heat_wave=np.arange(len(spells)),
        time_around_beg=time_around_beg,
    ).copy(deep=True)
    ds_masked = ds_masked.assign_coords(lengths=("heat_wave", lengths))
    dims = list(ds_masked.sizes.values())[:2]
    dummy_da = np.zeros(dims) + np.nan
    ds_masked = ds_masked.assign_coords(
        temperature=(["heat_wave", "time_around_beg"], dummy_da)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ds_masked = ds_masked.assign_coords(
            absolute_time=(
                ["heat_wave", "time_around_beg"],
                dummy_da.astype("datetime64[h]"),
            )
        )
    for i, spell in enumerate(spells_ts):
        unexpected_offset = time_before - (spells[i][0] - spell[0])
        this_tab = time_around_beg[: len(spell)] + unexpected_offset
        to_assign = (
            ds.loc[dict(time=spell)]
            .assign_coords(time=this_tab)
            .rename(time="time_around_beg")
        )
        accessor_dict = dict(heat_wave=i, time_around_beg=this_tab)
        ds_masked.loc[accessor_dict] = to_assign
        ds_masked.temperature.loc[accessor_dict] = da.loc[dict(time=spell)].values
        ds_masked.absolute_time.loc[accessor_dict] = spell
    return ds_masked


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
    return mask_from_spells(da, ds, spells_ts, spells, time_before)


def quantile_exceedence(  # 2 directional based on quantile above or below 0.5
    da: xr.DataArray, q: float = 0.95, dim: str = "time"
) -> xr.DataArray:
    if q > 0.5:
        return da > da.quantile(q, dim=dim)
    return da < da.quantile(q, dim=dim)


def spatial_agglomerative_clustering(
    da: xr.DataArray,
    condition_function: Callable = lambda x: x,
    mask: xr.DataArray | Literal["land"] | None = None,
    season: str | list | None = "JJA",
    metric: str = "jaccard",
) -> NDArray:
    lon, lat = da.lon.values, da.lat.values
    if mask and mask == "land":
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
    return pairwise_distances(
        to_cluster_flat.T, metric=metric, n_jobs=N_WORKERS
    )


def _add_timescales_to_predictors(predictors, timescales: Sequence):
    predictors = predictors.expand_dims(axis=-1, **dict(timescale=timescales)).copy(deep=True)
    for year in np.unique(predictors.time.dt.year):
        year_mask = predictors.time.dt.year == year
        indexer = {"time": year_mask}
        for timescale in timescales[1:]:
            indexer["timescale"] = timescale
            predictors.loc[indexer] = (
                predictors.loc[indexer].rolling(time=timescale, center=False).mean()
            )     
    return predictors


def _add_lags_to_predictors(predictors, lags: Sequence):
    predictors = predictors.expand_dims(axis=-1, **dict(lag=lags)).copy(deep=True)
    for year in np.unique(predictors.time.dt.year):
        year_mask = predictors.time.dt.year == year
        indexer = {"time": year_mask}
        for lag in lags[1:]:
            indexer["lag"] = lag
            predictors.loc[indexer] = predictors.loc[indexer].shift(time=lag)
    return predictors


def augment_predictors(
    predictors: xr.Dataset,
    timescales: Sequence | None = None,
    lags: Sequence | None = None,
):
    predictors = predictors.to_array(dim="varname", name="predictors")
    if timescales is not None and len(timescales) > 1:
        predictors = _add_timescales_to_predictors(predictors, timescales)
    if lags is not None and len(lags) > 1:
        predictors = _add_lags_to_predictors(predictors, lags)
    
    dims_to_stack = ["varname"]
    for potential in ["jet", "timescale"]:
        if potential in predictors.dims:
            dims_to_stack.append(potential)
    predictors = predictors.stack(predictor=dims_to_stack)
    newindex = ['_'.join([str(t) for t in ts]) for ts in predictors.predictor.values]
    dims_to_stack.append('predictor')
    predictors = predictors.assign_coords(predictor_name=("predictor", newindex))
    return predictors    


def augment_targets(targets, timescales):
    targets = targets.expand_dims(axis=-1, **dict(timescale=timescales)).copy(
        deep=True
    )
    for timescale in tqdm(timescales[1:]):
        for year in np.unique(targets.time.dt.year):
            year_mask = targets.time.dt.year == year
            indexer = {"time": year_mask, "timscale": timescale}
            targets.loc[indexer] = (
                targets.loc[indexer].rolling(time=timescale, center=False).mean().shift(time=-timescale + 1)
            )
    return targets


def create_all_triplets(predictors: xr.DataArray, targets: xr.DataArray, lags: Sequence):
    triplets = []
    
    for predictor in predictors.T:
        for lag in lags:
            if "timescale" in targets.dims:
                timescale = predictor.timescale.item()
                target = targets.sel(dict(timescale=timescale))
            else:
                target = targets
            triplets.append(((predictor.predictor_name.item(), lag), predictor, target))
    return triplets


def stacked_lstsq(L, b, rcond=1e-10): # https://stackoverflow.com/questions/30442377/how-to-solve-many-overdetermined-systems-of-linear-equations-using-vectorized-co
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s>=s_min]
    x = np.einsum('...ji,...j->...i', v, inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)


def compute_r(triplet, season: str | list | None = "JJA") -> NDArray:
    (predictor_name, lag), predictor_, target_ = triplet
    timescale = np.argmax(~target_[:1000].isnull())
    shape = target_.shape
    predictors, targets = [], []
    for year in YEARS:
        predictor = extract_season(predictor_.sel(time=predictor_.time.dt.year==year), season)
        predictor_resid = np.linalg.lstsq(predictor[timescale:, None], predictor[:-timescale], rcond=None)[0][0]
        predictor[:-timescale] = predictor[:-timescale] - predictor[timescale:] * predictor_resid
        
        target = extract_season(target_.sel(time=target_.time.dt.year==year), season)
        target = target.reshape(target.shape[0], -1)
        target_resid = stacked_lstsq(target[timescale:, None], target[:-timescale])
        target[:-timescale] = target[:-timescale] - target[timescale:] * target_resid
        target = target.reshape(target.shape[0], *shape[-2:])
        
        if lag > 0:
            driver = driver[lag:]
            target = target[:-lag]
            
        predictors.append(predictor)
        targets.append(target)
    predictor = np.concatenate(predictors, axis=0)
    target = np.concatenate(targets, axis=0)
    r = np.sum(predictor[:, None, None] * target, axis=0) / np.sqrt(np.sum(predictor[:, None, None] ** 2, axis=0) * np.sum(target ** 2, axis=0))
    return r


def compute_all_responses(predictors: xr.DataArray, targets: xr.DataArray, lags: Sequence | None = None) -> xr.DataArray:
    if "lag" in predictors.dims:
        if lags is None:
            lags = predictors.lags.values
        predictors = predictors[dict(lag=0)].reset_coords("lag", drop=True)
    if lags is None:
        lags = [0]
    coords = {
        'predictor': predictors.predictor.values,
        'lag': lags,
        'lon': targets.lon.values,
        'lat': targets.lat.values,
    }
    corr_da = xr.DataArray(np.zeros([len(c) for c in coords.values()]), coords=coords)
    corr_da.assign_coords(pstar=('timescale', 5 * 10 ** (-4 - 0.3 * (corr_da.timescale.values - 1)))) # van straaten
    triplets = create_all_triplets(predictors, targets)
    with Pool(processes=N_WORKERS) as pool:
        all_r = list(tqdm(pool.imap(compute_r, triplets, chunksize=3), total=len(triplets),))
    all_r = np.asarray(all_r)
    corr_da[:] = all_r
    return corr_da
    

def one_time_logistic_regression(targets):
    X_base = (
        (targets.time.values - targets.time.values[0])
        .astype("timedelta64[h]")
        .astype(int)[:, None]
    )
    
def comb_random_forest(y: NDArray, ds: xr.Dataset, all_combinations: list):
    feature_importance = np.zeros((len(all_combinations), len(all_combinations[0]), 5))
    scores = np.zeros((len(all_combinations), 4))
    for j, comb in enumerate(all_combinations):
        X = np.nan_to_num(
            np.stack([ds[varname][:, jet].values for varname, jet in comb], axis=1),
            nan=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        forest_regressor = RandomForestClassifier(
            n_estimators=100, n_jobs=N_WORKERS, class_weight="balanced"
        ).fit(X_train, y_train)
        y_pred = forest_regressor.predict(X_test)
        y_pred_proba = forest_regressor.predict_proba(X_test)[:, 1]
        scores[j, 0] = roc_auc_score(y_test, y_pred_proba)
        scores[j, 1] = f1_score(y_test, y_pred)
        scores[j, 2] = balanced_accuracy_score(y_test, y_pred)
        scores[j, 3] = brier_score_loss(y_test, y_pred_proba)
        feature_importance[j, :, 0] = (
            np.mean(
                (X - X.mean(axis=1)[:, None]) * (y - y.mean()), axis=1
            )
            / np.std(X, axis=1)
            / np.std(y)
        )
        feature_importance[j, :, 1] = forest_regressor.feature_importances_
        feature_importance[j, :, 2] = permutation_importance(
            forest_regressor, X_test, y_test
        )
        shap_values = shap.TreeExplainer(forest_regressor)(X, check_additivity=False)
        feature_importance[j, :, 3] = shap_values.abs.mean(axis=0).values
        feature_importance[j, :, 4] = shap_values.mean(axis=0).values
    return feature_importance, scores


def comb_random_forest_base_rate(
    y: NDArray, y_base: NDArray, ds: xr.Dataset, all_combinations: list
):
    feature_importance = np.zeros((len(all_combinations), len(all_combinations[0]), 5))
    scores = np.zeros((len(all_combinations), 4))
    true_targets = y + y_base > 0.99
    for j, comb in enumerate(all_combinations):
        X = np.nan_to_num(
            np.stack([ds[varname][:, jet].values for varname, jet in comb], axis=1),
            nan=0,
        )
        X_train, X_test, y_train, y_test, _, y_base_test, _, true_targets_test = train_test_split(
            X, y, y_base, true_targets, test_size=0.2
        )
        forest_regressor = RandomForestRegressor(
            n_estimators=100, n_jobs=N_WORKERS
        ).fit(X_train, y_train)
        test_pred_proba = np.clip(forest_regressor.predict(X_test) + y_base_test, 0, 1)
        test_pred = test_pred_proba > 0.5
        scores[j, 0] = roc_auc_score(true_targets_test, test_pred_proba)
        scores[j, 1] = f1_score(true_targets_test, test_pred)
        scores[j, 2] = balanced_accuracy_score(true_targets_test, test_pred)
        scores[j, 3] = brier_score_loss(true_targets_test, test_pred_proba)
        feature_importance[j, :, 0] = (
            np.mean(
                (X - X.mean(axis=1)[:, None]) * (y - y.mean()), axis=1
            )
            / np.std(X, axis=1)
            / np.std(y)
        )
        feature_importance[j, :, 1] = forest_regressor.feature_importances_
        feature_importance[j, :, 2] = permutation_importance(
            forest_regressor, X_test, y_test
        )
        shap_values = shap.TreeExplainer(forest_regressor)(X, check_additivity=False)
        feature_importance[j, :, 3] = shap_values.abs.mean(axis=0).values
        feature_importance[j, :, 4] = shap_values.mean(axis=0).values
    return feature_importance, scores


def comb_logistic_regression(y: NDArray, ds: xr.Dataset, all_combinations: list):
    coefs = np.zeros((len(all_combinations), len(all_combinations[0])))
    scores = np.zeros((len(all_combinations), 4))
    for j, comb in enumerate(all_combinations):
        X = np.nan_to_num(
            np.stack([ds[varname][:, jet].values for varname, jet in comb], axis=1),
            nan=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        log = LogisticRegression().fit(X_train, y_train)
        coefs[j, :] = log.coef_[0]
        y_pred = log.predict(X_test)
        y_pred_proba = log.predict_proba(X_test)[:, 1]
        scores[j, 0] = roc_auc_score(y_test, y_pred_proba)
        scores[j, 1] = f1_score(y_test, y_pred)
        scores[j, 2] = balanced_accuracy_score(y_test, y_pred)
        scores[j, 3] = brier_score_loss(y_test, y_pred_proba)
    return coefs, scores


def all_logistic_regressions(
    ds: xr.Dataset, n_predictors: int, Y: xr.DataArray | NDArray
):
    predictors = list(product(ds.data_vars, [0, 1]))
    all_combinations = list(combinations(predictors, n_predictors))
    func = partial(comb_logistic_regression, ds=ds, all_combinations=all_combinations)
    try:
        Y = Y.values
    except AttributeError:
        pass
    with Pool(processes=Y.shape[1]) as pool:
        results = list(tqdm(pool.imap(func, Y.T, chunksize=1)))
    coefs, scores = zip(*results)
    return np.stack(coefs, axis=0), np.stack(scores, axis=0)


class ExtremeExperiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
        q: float = 0.95,
        mask: xr.DataArray | Literal["land"] | None = Literal["land"],
        season: str | list | None = "JJA",
        metric: str = "jaccard",
    ) -> None:
        self.data_handler = data_handler
        self.da = self.data_handler.get_da()
        self.path = self.data_handler.get_path()
        self.q = q
        self.mask = mask
        if season is None:
            self.subseason = self.data_handler.get_metadata()["season"]
        else:
            self.subseason = season
        self.metric = metric
        self.path_suffix = f"{q}_{season}_{metric}_{mask_name(mask)}mask"

    def compute_linkage_quantile(
        self,
    ) -> NDArray:
        Z_path = f"Z_{self.path_suffix}.npy"
        Z_path = self.path.joinpath(Z_path)
        if Z_path.is_file:
            return np.load(Z_path)
        condition_function = partial(quantile_exceedence, q=self.q, dim="time")
        distances = spatial_agglomerative_clustering(
            self.da,
            condition_function,
            self.mask,
            season=self.season,
            metric=self.metric,
        )
        Z = linkage(squareform(distances), method="average")
        np.save(Z_path, Z)
        return Z

    def spatial_clusters_as_da(
        self,
        n_clu: int,
    ) -> xr.DataArray:
        sample_dims = self.data_handler.get_sample_dims()
        clusters_da_file = f"clusters_{self.path_suffix}_{n_clu}.nc"
        clusters_da_file = self.path.joinpath(clusters_da_file)
        if clusters_da_file.is_file():
            return xr.open_dataarray(clusters_da_file)

        Z = self.compute_linkage_quantile()
        clusters = cut_tree(Z, n_clusters=n_clu)[:, 0]
        lon, lat = sample_dims["lon"], sample_dims["lat"]
        stack_dims = {"lat_lon": ("lat", "lon")}
        if mask and mask == "land":
            mask = get_land_mask()
        if mask is not None:
            mask = mask.sel(lon=lon, lat=lat)
            mask_flat = mask.stack(stack_dims)
        clusters_da = np.zeros(mask_flat.shape, dtype=float)
        clusters_da[:] = np.nan
        clusters_da = mask_flat.copy(data=clusters_da)
        clusters_da[mask_flat] = clusters
        clusters_da = clusters_da.unstack()
        clusters_da.to_netcdf(clusters_da_file)
        return clusters_da
    