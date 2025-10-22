# coding: utf-8
from pathlib import Path
from typing import Tuple
from functools import partial
from multiprocessing import Pool
from typing import Union
import pickle as pkl

import numpy as np
import xarray as xr
import polars as pl
from scipy.stats import norm
from .definitions import N_WORKERS, SEASONS, infer_direction
import polars_ds as pds


def create_bootstrapped_times(times: pl.DataFrame, all_times: pl.Series, n_bootstraps: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng()
    orig_times = times.clone()
    boostrap_len = orig_times["time"].n_unique()
    spell_cols = ["spell", "relative_index", "relative_time", "len"] if "spell" in times.columns else []
    times = (
        times[["time", *spell_cols]]
        .with_row_index("inside_index")
        .with_columns(sample_index=pl.lit(n_bootstraps, dtype=pl.UInt32))
    )
    if "spell" in orig_times.columns:  # then per-spell bootstrapping
        min_rel_index = times["relative_index"].min()
        unique_spells, lens = times.group_by("spell").agg(len=pl.col("time").len()).sort("spell")
        ts_bootstrapped = []
        for spell, len_ in zip(unique_spells, lens):
            bootstraps = rng.choice(all_times.shape[0] - len_, size=(n_bootstraps, 1)) # should be -len per year....
            bootstraps = bootstraps + np.arange(len_)[None, :]
            this_ts = all_times[bootstraps.flatten()].to_frame()
            this_ts = this_ts.with_columns(
                len=pl.lit(len_).cast(pl.UInt32()),
                sample_index=pl.row_index() // len_,
                inside_index=pl.row_index() % len_,
                relative_index=pl.row_index().cast(pl.Int32()) % len_ + min_rel_index,
                spell=pl.lit(spell).cast(pl.UInt32()),
            ).join(times[["spell", "relative_index", "relative_time"]], on=["spell", "relative_index"])
            ts_bootstrapped.append(this_ts)
        ts_bootstrapped = pl.concat(ts_bootstrapped)
    else:
        bootstraps = rng.choice(all_times.shape[0], size=(n_bootstraps, boostrap_len))
        ts_bootstrapped = all_times[bootstraps.flatten()].to_frame()
        ts_bootstrapped = ts_bootstrapped.with_columns(
            sample_index=pl.row_index() // boostrap_len,
            inside_index=pl.row_index() % boostrap_len,
        )
    columns = ["sample_index", "inside_index", "time", *spell_cols]
    ts_bootstrapped = pl.concat(
        [
            ts_bootstrapped[columns],
            times[columns],
        ]
    )
    return ts_bootstrapped


def cdf(timeseries: Union[xr.DataArray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the cumulative distribution function of a 1D DataArray

    Args:
        timeseries (xr.DataArray or npt.np.ndarray): will be cast to ndarray if DataArray.

    Returns:
        x (npt.np.ndarray): x values for plotting,
        y (npt.np.ndarray): cdf of the timeseries,
    """
    if isinstance(timeseries, xr.DataArray):
        timeseries = timeseries.values
    idxs = np.argsort(timeseries)
    y = np.cumsum(idxs) / np.sum(idxs)
    x = timeseries[idxs]
    return x, y


def trends_and_pvalues(
    props_as_df: pl.DataFrame,
    data_vars: list,
    season: str | None = None,
    std: bool = False,
    bootstrap_len: int = 4,
    n_boostraps: int = 10000,
):
    ncat = props_as_df["jet"].n_unique()

    if season is not None and season != "Year":
        month_list = SEASONS[season]
        props_as_df = props_as_df.filter(pl.col("time").dt.month().is_in(month_list))
    else:
        season = "all_year"

    def agg_func(col):
        return pl.col(col).std() if std else pl.col(col).mean()

    aggs = [agg_func(col) for col in data_vars]
    props_as_df = props_as_df.group_by(
        pl.col("time").dt.year().alias("year"), pl.col("jet"), maintain_order=True
    ).agg(*aggs)

    x = props_as_df["year"].unique()
    n = len(x)
    num_blocks = n // bootstrap_len

    rng = np.random.default_rng()

    sample_indices = rng.choice(
        n - bootstrap_len, size=(n_boostraps, n // bootstrap_len)
    )
    sample_indices = sample_indices[..., None] + np.arange(bootstrap_len)[None, None, :]
    sample_indices = sample_indices.reshape(n_boostraps, num_blocks * bootstrap_len)
    sample_indices = np.append(
        sample_indices, np.arange(sample_indices.shape[1])[None, :], axis=0
    )
    sample_indices = ncat * np.repeat(sample_indices.flatten(), ncat)
    for k in range(ncat):
        sample_indices[k::ncat] = sample_indices[k::ncat] + k

    ts_bootstrapped = props_as_df[sample_indices]
    ts_bootstrapped = ts_bootstrapped.with_columns(
        sample_index=np.arange(len(ts_bootstrapped))
        // (ncat * num_blocks * bootstrap_len),
        inside_index=np.arange(len(ts_bootstrapped))
        % (ncat * num_blocks * bootstrap_len),
    )

    slopes = ts_bootstrapped.group_by(["sample_index", "jet"], maintain_order=True).agg(
        **{
            data_var: pds.lin_reg_report(
                    pl.int_range(0, pl.col("year").len()).alias("year"), 
                    target=pl.col(data_var), 
                    add_bias=True
                ).struct.field("beta").first()
            for data_var in data_vars
        }
    )

    constants = props_as_df.group_by("jet", maintain_order=True).agg(
        **{
            data_var: pds.lin_reg_report(
                pl.col("year"), 
                target=pl.col(data_var), 
                add_bias=True
            ).struct.field("beta").last()
            for data_var in data_vars
        }
    )

    pvals = slopes.group_by("jet", maintain_order=True).agg(
        **{
            data_var: pl.col(data_var)
            .head(n_boostraps)
            .sort()
            .search_sorted(pl.col(data_var).get(-1)).first()
            / n_boostraps
            for data_var in data_vars
        }
    )
    return x, props_as_df, slopes, constants, pvals


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
    X: np.ndarray, lag_max: int
) -> np.ndarray:
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
    a: np.ndarray, x: np.ndarray, **kwargs
) -> (
    np.ndarray
):  # https://stackoverflow.com/questions/40588403/vectorized-searchsorted-numpy + reshapes
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


def fdr_correction(p: np.ndarray, q: float = 0.02):
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
    take_from: np.ndarray | xr.DataArray,
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
    direction = infer_direction(empirical_distribution)
    empirical_distribution = np.mean(empirical_distribution, axis=0)
    q = q / 2 if direction == 0 else q
    p = norm.cdf(
        to_test.mean(dim="time").values,
        loc=np.mean(empirical_distribution, axis=0),
        scale=np.std(empirical_distribution, axis=0),
    )
    nocorr = (p > (1 - q)) | (p < q)
    return nocorr, fdr_correction(p, q)


def one_ks_cumsum(b: np.ndarray, a: np.ndarray, q: float = 0.02, n_sam: int = None):
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


def one_ks_searchsorted(b: np.ndarray, a: np.ndarray, q: float = 0.02, n_sam: int = None):
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
    take_from: np.ndarray,
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

