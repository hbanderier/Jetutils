from pathlib import Path
from typing import Tuple
from nptyping import NDArray, Float, Shape
from functools import partial
from itertools import combinations, product
from multiprocessing import Pool
import pickle as pkl

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from definitions import N_WORKERS, infer_sym

def autocorrelation(path: Path, time_steps: int = 50) -> Path:
    ds = xr.open_dataset(path)
    name = path.parts[-1].split(".")[0]
    parent = path.parent
    autocorrs = {}
    for i, varname in enumerate(ds):
        if varname.split("_")[-1] == "climatology":
            continue
        autocorrs[varname] = ("lag", np.empty(time_steps))
        for j in range(time_steps):
            autocorrs[varname][1][j] = xr.corr(
                ds[varname], ds[varname].shift(time=j)
            ).values
    autocorrsda = xr.Dataset(autocorrs, coords={"lag": np.arange(time_steps)})
    opath = parent.joinpath(f"{name}_autocorrs.nc")
    autocorrsda.to_netcdf(opath)
    return opath  # a great swedish metal bEnd


def compute_autocorrs(
    X: NDArray[Shape["*, *"], Float], lag_max: int
) -> NDArray[Shape["*, *, *"], Float]:
    autocorrs = []
    i_max = X.shape[1]
    for i in range(lag_max):
        autocorrs.append(
            np.cov(X[i:], np.roll(X, i, axis=0)[i:], rowvar=False)[i_max:, :i_max]
        )
    return np.asarray(autocorrs)


def Hurst_exponent(path: Path, subdivs: int = 11) -> Path:
    ds = xr.open_dataset(path)
    subdivs = [2**n for n in range(11)]
    lengths = [len(ds.time) // n for n in subdivs]
    all_lengths = np.repeat(lengths, subdivs)
    N_chunks = np.sum(subdivs)
    Hurst = {}
    for i, varname in enumerate(ds.data_vars):
        adjusted_ranges = []
        for n_chunks, n in zip(subdivs, lengths):
            start = 0
            aranges = []
            for k in range(n_chunks):
                end = start + n
                series = ds[varname].isel(time=np.arange(start, end)).values
                mean = np.mean(series)
                std = np.std(series)
                series -= mean
                series = np.cumsum(series)
                raw_range = series.max() - series.min()
                aranges.append(raw_range / std)
            adjusted_ranges.append(np.mean(aranges))
        coeffs = np.polyfit(np.log(lengths), np.log(adjusted_ranges), deg=1)
        Hurst[varname] = [coeffs[0], np.exp(coeffs[1])]
    parent = path.parent
    name = path.parts[-1].split(".")[0]
    opath = parent.joinpath(f"{name}_Hurst.pkl")
    with open(opath, "wb") as handle:
        pkl.dump(Hurst, handle)
    return opath


def searchsortednd(
    a: NDArray, x: NDArray, **kwargs) -> NDArray:  # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy + reshapes
    orig_shapex, nx = x.shape[1:], x.shape[0]
    _, na = a.shape[1:], a.shape[0]
    m = np.prod(orig_shapex)
    a = a.reshape(na, m)
    x = x.reshape(nx, m)
    max_num = np.maximum(np.nanmax(a) - np.nanmin(a), np.nanmax(x) - np.nanmin(x)) + 1
    r = max_num * np.arange(m)[None, :]
    p = (
        np.searchsorted((a + r).ravel(order="F"), (x + r).ravel(order="F"), side="left")
        .reshape(m, nx)
        .T
    )
    return (p - na * (np.arange(m)[None, :])).reshape((nx, *orig_shapex))


def fdr_correction(p: NDArray, q: float = 0.02):
    pshape = p.shape
    p = p.ravel()
    num_p = len(p)
    fdrcorr = np.zeros(num_p, dtype=bool)
    argp = np.argsort(p)
    p = p[argp]
    line_below = q * np.arange(num_p) / (num_p - 1)
    line_above = line_below + (1 - q)
    fdrcorr[argp] = (p >= line_above) | (p <= line_below)
    return fdrcorr.reshape(pshape)


def field_significance(
    to_test: xr.DataArray,
    take_from: NDArray | xr.DataArray,
    n_sel: int = 100,
    q: float = 0.02,
) -> Tuple[xr.DataArray, xr.DataArray]:
    n_sam = to_test.shape[0]
    indices = np.random.rand(n_sel, take_from.shape[0]).argpartition(n_sam, axis=1)[
        :, :n_sam
    ]
    if isinstance(take_from, xr.DataArray):
        take_from = take_from.values
    empirical_distribution = []
    cs = 500
    for ns in range(0, n_sam, cs):
        end = min(ns + cs, n_sam)
        empirical_distribution.append(
            np.mean(np.take(take_from, indices[:, ns:end], axis=0), axis=1)
        )
    sym = infer_sym(empirical_distribution)
    empirical_distribution = np.mean(empirical_distribution, axis=0)
    q = q / 2 if sym else q
    p = norm.cdf(
        to_test.mean(dim="time").values,
        loc=np.mean(empirical_distribution, axis=0),
        scale=np.std(empirical_distribution, axis=0),
    )
    nocorr = (p > (1 - q)) | (p < q)
    return nocorr, fdr_correction(p, q)


def one_ks_cumsum(b: NDArray, a: NDArray, q: float = 0.02, n_sam: int = None):
    if n_sam is None:
        n_sam = len(a)
    x = np.concatenate([a, b], axis=0)
    idxs_ks = np.argsort(x, axis=0)
    y1 = np.cumsum(idxs_ks < n_sam, axis=0) / n_sam
    y2 = np.cumsum(idxs_ks >= n_sam, axis=0) / n_sam
    d = np.amax(np.abs(y1 - y2), axis=0)
    p = np.exp(-(d**2) * n_sam)
    nocorr = (p < q).astype(int)
    return nocorr, fdr_correction(p, q)


def one_ks_searchsorted(b: NDArray, a: NDArray, q: float = 0.02, n_sam: int = None):
    if n_sam is None:
        n_sam = len(a)
    x = np.concatenate([a, b], axis=0)
    idxs_ks = np.argsort(x, axis=0)
    y1 = np.cumsum(idxs_ks < n_sam, axis=0) / n_sam
    y2 = np.cumsum(idxs_ks >= n_sam, axis=0) / n_sam
    d = np.amax(np.abs(y1 - y2), axis=0)
    p = np.exp(-(d**2) * n_sam)
    nocorr = (p < q).astype(int)
    return nocorr, fdr_correction(p, q)


def field_significance_v2(
    to_test: xr.DataArray,
    take_from: NDArray,
    n_sel: int = 100,
    q: float = 0.02,
    method: str = "cumsum",
    processes: int = N_WORKERS,
    chunksize: int = 2,
) -> Tuple[xr.DataArray, xr.DataArray]:
    # Cumsum implementation is slightly less robust (tie problem) but so much faster
    nocorr = np.zeros((take_from.shape[1:]), dtype=int)
    fdrcorr = np.zeros((take_from.shape[1:]), dtype=int)
    a = to_test.values
    if method == "searchsorted":
        a = np.sort(a, axis=0)
        # b should be sorted as well but it's expensive to do it here, instead sort take_from before calling (since it's usually needed in many calls)
    n_sam = len(a)
    indices = np.random.rand(n_sel, take_from.shape[0]).argpartition(n_sam, axis=1)[
        :, :n_sam
    ]
    if method == "searchsorted":
        indices = np.sort(indices, axis=1)
        func = partial(one_ks_searchsorted, a=a, q=q, n_sam=n_sam)
    else:
        func = partial(one_ks_cumsum, a=a, q=q, n_sam=n_sam)

    with Pool(processes=processes) as pool:
        results = pool.map(
            func, (take_from[indices_] for indices_ in indices), chunksize=chunksize
        )
    nocorr, fdrcorr = zip(*results)
    nocorr = to_test[0].copy(data=np.sum(nocorr, axis=0) > (1 - q) * n_sel)
    fdrcorr = to_test[0].copy(data=np.sum(fdrcorr, axis=0) > (1 - q) * n_sel)
    return nocorr, fdrcorr


def comb_logistic_regression(y: NDArray, ds: xr.Dataset, all_combinations: list):
    coefs = np.zeros((len(all_combinations), len(all_combinations[0])))
    scores = np.zeros(len(all_combinations))
    for j, comb in enumerate(all_combinations):
        X = np.nan_to_num(np.stack([ds[varname][:, jet].values for varname, jet in comb], axis=1), nan=0)
        log = LogisticRegression().fit(X=X, y=y)
        coefs[j, :] = log.coef_[0]
        scores[j] = roc_auc_score(y, log.predict_proba(X)[:, 1])
    return coefs, scores


def all_logistic_regressions(ds: xr.Dataset, n_predictors: int, Y: xr.DataArray | NDArray):
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