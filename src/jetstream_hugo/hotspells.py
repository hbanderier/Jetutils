from typing import Union, Tuple, Iterable
from nptyping import NDArray

import numpy as np
import pandas as pd
import xarray as xr

from definitions import (
    REGIONS,
    DATERANGEPL,
)

def hotspells_mask(
    filename: str = "hotspells.csv",
    daysbefore: int = 21,
    daysafter: int = 5,
    timerange: NDArray | pd.DatetimeIndex | xr.DataArray = None,
    names: Iterable = None,
) -> xr.DataArray:
    """Returns timeseries mask of hotspells in several regions in `timerange` as a xr.DataArray with two dimensions and coordinates. It has shape (len(timerange), n_regions). n_regions is either inferred from the file or from the len of names if it is provided

    Args:
        filename (str, optional): path to hotspell center dates. Defaults to 'hotspells.csv'.
        daysbefore (int, optional): how many days before the center will the mask extend (inclusive). Defaults to 21.
        daysafter (int, optional): how many days after the center will the mask extend (inclusive). Defaults to 5.
        timerange (NDArray | pd.DatetimeIndex | xr.DataArray, optional): the time range to mask. Defaults to DATERANGEPL.
        names (Iterable, optional): names of the regions. See body for default values.

    Returns:
        xr.DataArray: at position (day, region) this is True if this day is part of a hotspell in this region
    """
    if names is None:
        names = REGIONS
    if timerange is None:
        timerange = DATERANGEPL
    else:
        try:
            timerange = timerange.values
        except AttributeError:
            pass
        timerange = pd.DatetimeIndex(timerange).floor(freq="1D")
    list_of_dates = np.loadtxt(filename, delimiter=",", dtype=np.datetime64)
    assert len(names) == list_of_dates.shape[1]
    data = np.zeros((len(timerange), len(names)), dtype=bool)
    coords = {"time": timerange, "region": names}
    data = xr.DataArray(data, coords=coords)
    for i, dates in enumerate(list_of_dates.T):
        dates = np.sort(dates)
        dates = dates[
            ~(np.isnat(dates) | (np.datetime_as_string(dates, unit="Y") == "2022"))
        ]
        for date in dates:
            tsta = date - np.timedelta64(daysbefore, "D")
            tend = date + np.timedelta64(daysafter, "D")
            data.loc[tsta:tend, names[i]] = True
    return data


def get_hostpells_v2(
    filename: str = "hotspells_v2.csv", lag_behind: int = 10, regions: list = None
) -> list:
    if regions is None:
        regions = REGIONS
    hotspells_raw = pd.read_csv(filename)
    hotspells = []
    maxlen = 0
    maxnhs = 0
    for i, key in enumerate(regions):
        hotspells.append([])
        for line in hotspells_raw[f"dates{i + 1}"]:
            if line == "-999":
                continue
            dateb, datee = [np.datetime64(d) for d in line.split("/")]
            dateb -= np.timedelta64(lag_behind, "D")
            hotspells[-1].append(pd.date_range(dateb, datee, freq="1D"))
            maxlen = max(maxlen, len(hotspells[-1][-1]))
        maxnhs = max(maxnhs, len(hotspells[-1]))
    return hotspells, maxnhs, maxlen


def apply_hotspells_mask_v2(
    hotspells: list,
    ds: xr.Dataset,
    maxlen: int = None,
    maxnhs: int = None,
    regions: list = None,
    lag_behind: int = 10,
) -> xr.Dataset:
    if regions is None:
        regions = REGIONS
    assert len(regions) == len(hotspells)
    if maxlen is None or maxnhs is None:
        maxlen = 0
        maxnhs = 0
        for region in hotspells:
            maxnhs = max(maxnhs, len(region))
            for hotspell in region:
                maxlen = max(maxlen, len(hotspell))
    hotspell_length = np.zeros((len(regions), maxnhs))
    hotspell_length[:] = np.nan
    for i, hss in enumerate(hotspells):
        hotspell_length[i, :len(hss)] = [len(hs) - lag_behind for hs in hss]
    data = {}
    other_coord = list(ds.coords.items())[1]
    for varname in ds.data_vars:
        data[varname] = (
            (other_coord[0], 'region', 'hotspell', 'day_after_beg'), 
            np.zeros((ds[varname].shape[1], len(hotspells), maxnhs, maxlen))
        )
        data[varname][1][:] = np.nan
    ds_masked = xr.Dataset(
        data,
        coords={
            other_coord[0]: other_coord[1].values,
            "region": regions,
            "hotspell": np.arange(maxnhs),
            "day_after_beg": np.arange(maxlen) - lag_behind,
        },
    )
    for varname in ds.data_vars:
        for i, regionhs in enumerate(hotspells):
            for j, hotspell in enumerate(regionhs):
                try:
                    ds_masked[varname][:, i, j, :len(hotspell)] = ds[varname].sel(
                        time=hotspells[i][j].values
                    ).values.T
                except KeyError:
                    ...
    ds_masked = ds_masked.assign_coords({'hotspell_length': (('region', 'hotspell'), hotspell_length)})
    return ds_masked


def get_hotspell_mask(da_time: xr.DataArray | NDArray, num_lags: int = 1) -> xr.DataArray:
    hotspells = get_hostpells_v2(lag_behind=num_lags)[0]
    if isinstance(da_time, xr.DataArray):
        da_time = da_time.values
    hs_mask = np.zeros((len(da_time), len(REGIONS), num_lags))
    hs_mask = xr.DataArray(hs_mask, coords={'time': da_time, 'region': REGIONS, 'lag': np.arange(num_lags)})
    for i, region in enumerate(REGIONS):
        for hotspell in hotspells[i]:
            try:
                hs_mask.loc[hotspell[:10], region, np.arange(num_lags)] += np.eye(num_lags) * len(hotspell)
            except KeyError:
                ...
    return hs_mask