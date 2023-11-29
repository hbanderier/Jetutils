from typing import Union, Tuple, Iterable
from nptyping import NDArray

import numpy as np
import pandas as pd
import xarray as xr

from jetstream_hugo.definitions import (
    DATADIR,
    REGIONS,
    DATERANGEPL,
)

def hotspells_mask(
    filename: str = f"{DATADIR}/hotspells.csv",
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


def get_hotspells_v2(
    filename: str = f"{DATADIR}/hotspells_v2.csv", lag_behind: int = 10, regions: list = None
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
    ds: xr.Dataset,
    timesteps_before: int = 12,
    timesteps_after: int = 0,
) -> xr.Dataset:
    hotspells = get_hotspells_v2(lag_behind=0)[0]
    maxnhs = 0
    maxlen = 0
    for region in hotspells:
        maxnhs = max(maxnhs, len(region))
        for hotspell in region:
            maxlen = max(maxlen, len(hotspell))
    hotspell_length = np.zeros((len(hotspells), maxnhs))
    hotspell_length[:] = np.nan
    dt = pd.Timedelta(ds.time.values[1] - ds.time.values[0])
    for i, hss in enumerate(hotspells):
        hotspell_length[i, :len(hss)] = [len(hs) for hs in hss]
    first_relative_time = - timesteps_before * dt
    longest_hotspell = np.unravel_index(np.nanargmax(hotspell_length), hotspell_length.shape)
    longest_hotspell = hotspells[longest_hotspell[0]][longest_hotspell[1]]
    last_relative_time = longest_hotspell[-1] - longest_hotspell[0] + pd.Timedelta(1, 'day') + (timesteps_after - 1) * dt
    time_around_beg = pd.timedelta_range(first_relative_time, last_relative_time, freq=dt)
    data = {}
    other_coord = list(ds.coords.items())[1]
    for varname in ds.data_vars:
        data[varname] = (
            (other_coord[0], 'region', 'hotspell', 'time_around_beg'), 
            np.zeros((ds[varname].shape[1], len(hotspells), maxnhs, len(time_around_beg)))
        )
        data[varname][1][:] = np.nan
    ds_masked = xr.Dataset(
        data,
        coords={
            other_coord[0]: other_coord[1].values,
            "region": REGIONS,
            "hotspell": np.arange(maxnhs),
            "time_around_beg": time_around_beg,
        },
    )
    for varname in ds.data_vars:
        for i, regionhs in enumerate(hotspells):
            for j, hotspell in enumerate(regionhs):
                year = hotspell[0].year
                min_time = np.datetime64(f'{year}-06-01T00:00')
                max_time = np.datetime64(f'{year}-09-01T00:00') - dt
                absolute_time = pd.date_range(hotspell[0] - dt * timesteps_before, hotspell[-1] + (timesteps_after + 3) * dt, freq=dt)
                mask_JJA = (absolute_time >= min_time) & (absolute_time <= max_time)
                first_valid_index = np.argmax(mask_JJA)
                last_valid_index = len(mask_JJA) - np.argmax(mask_JJA[::-1]) - 1
                region = REGIONS[i]
                this_tab = time_around_beg[first_valid_index:last_valid_index + 1]
                ds_masked[varname].loc[:, region, j, this_tab] = ds[varname].loc[absolute_time[mask_JJA].values, :].values.T
    ds_masked = ds_masked.assign_coords({'hotspell_length': (('region', 'hotspell'), hotspell_length)})
    return ds_masked


def hotspells_as_da(da_time: xr.DataArray | NDArray, timesteps_before: int = 0, timesteps_after: int = 0) -> xr.DataArray:
    hotspells = get_hotspells_v2(lag_behind=0)[0]
    if isinstance(da_time, xr.DataArray):
        da_time = da_time.values
    hs_da = np.zeros((len(da_time), len(REGIONS)))
    hs_da = xr.DataArray(hs_da, coords={'time': da_time, 'region': REGIONS})
    dt = pd.Timedelta(da_time[1] - da_time[0])
    for i, region in enumerate(REGIONS):
        for hotspell in hotspells[i]:
            year = hotspell[0].year
            min_time = np.datetime64(f'{year}-06-01T00:00')
            max_time = np.datetime64(f'{year}-09-01T00:00') - dt
            first_time = max(min_time, (hotspell[0] - dt * timesteps_before).to_datetime64())
            last_time = min(max_time, (hotspell[-1] + (timesteps_after + 3) * dt).to_datetime64())
            hs_da.loc[first_time:last_time, region] = len(hotspell)
    return hs_da

    
def get_hotspell_lag_mask(da_time: xr.DataArray | NDArray, num_lags: int = 1) -> xr.DataArray:
    hotspells = get_hotspells_v2(lag_behind=num_lags)[0]
    if isinstance(da_time, xr.DataArray):
        da_time = da_time.values
    hs_mask = np.zeros((len(da_time), len(REGIONS), num_lags))
    hs_mask = xr.DataArray(hs_mask, coords={'time': da_time, 'region': REGIONS, 'lag': np.arange(num_lags)})
    for i, region in enumerate(REGIONS):
        for hotspell in hotspells[i]:
            try:
                hs_mask.loc[hotspell[:num_lags], region, np.arange(num_lags)] += np.eye(num_lags) * len(hotspell)
            except KeyError:
                ...
    return hs_mask