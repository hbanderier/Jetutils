from typing import Union, Tuple, Iterable, Literal
from nptyping import NDArray

import numpy as np
import pandas as pd
import xarray as xr
from xclim.indices.run_length import rle, run_bounds
from jetstream_hugo.definitions import (
    DATADIR,
    REGIONS,
    DATERANGEPL,
)


def heat_waves_from_t(
    da: xr.DataArray,
    q: float = 0.95,
    fill_holes: bool = False,
    minlen: np.timedelta64 = np.timedelta64(3, 'D'),
    time_before: pd.Timedelta = pd.Timedelta(0, 'D'),
    time_after: pd.Timedelta = pd.Timedelta(0, 'D'),
    output_type: Literal['arr'] | Literal['list'] = 'arr'
) -> xr.DataArray | Tuple[list[NDArray]]:
    dt = pd.Timedelta(da.time.values[1] - da.time.values[0])
    months = np.unique(da.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(months[-1] + 1).zfill(2)]
    hot_days = (da > da.quantile(dim='time', q=q))
    if fill_holes:
        holes = rle(~hot_days)
        holes = np.where(holes.values == 1)[0]
        hot_days[holes] = 1
    heat_waves = run_bounds(hot_days)
    mask = (heat_waves[0].dt.year == heat_waves[1].dt.year).values
    heat_waves = heat_waves[:, mask]
    mask = heat_waves.astype('datetime64[h]').values
    mask = (mask[1] - mask[0]) >= minlen
    heat_waves = heat_waves[:, mask].T
    heat_waves_ts = []
    lengths = []
    for heat_wave in heat_waves:
        hw_len = (heat_wave[1] - heat_wave[0]).values.astype('timedelta64[D]')
        year = heat_wave[0].dt.year.values
        min_time = np.datetime64(f'{year}-{months[0]}-01T00:00')
        max_time = np.datetime64(f'{year}-{months[1]}-01T00:00') - dt
        first_time = max(min_time, (heat_wave[0] - time_before).values)
        last_time = min(max_time, (heat_wave[1] + time_after).values)
        this_hw = pd.date_range(first_time, last_time, freq='6h')
        heat_waves_ts.append(this_hw)
        lengths.append(np.full(len(this_hw), hw_len.astype(int)))
    if output_type == 'list':
        return heat_waves_ts, heat_waves.astype('datetime64[h]').values
    da_hs = da.copy(data=np.zeros(da.shape, dtype=int))
    da_hs.loc[np.concatenate(heat_waves_ts)] = np.concatenate(lengths)
    return da_hs


def mask_from_t(
    da: xr.DataArray,
    ds: xr.Dataset,
    q: float = 0.95,
    fill_holes: bool = False,
    minlen: np.timedelta64 = np.timedelta64(3, 'D'),
    time_before: pd.Timedelta = pd.Timedelta(0, 'D'),
    time_after: pd.Timedelta = pd.Timedelta(0, 'D'),
) -> xr.Dataset:
    heat_waves_ts, heat_waves = heat_waves_from_t(
        da, q, fill_holes, minlen, time_before, time_after, output_type='list'
    )
    dt = pd.Timedelta(da.time.values[1] - da.time.values[0])
    months = np.unique(ds.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(months[-1] + 1).zfill(2)]
    lengths = heat_waves[:, 1] - heat_waves[:, 0]
    longest_hotspell = np.argmax(lengths)
    time_around_beg = heat_waves_ts[longest_hotspell] - heat_waves[longest_hotspell, 0]
    time_around_beg = time_around_beg.values
    ds_masked = ds.loc[dict(time=ds.time.values[0])].reset_coords('time', drop=True).copy(deep=True)
    ds_masked.loc[dict()] = np.nan
    ds_masked = ds_masked.expand_dims(
        heat_wave=np.arange(len(heat_waves)),
        time_around_beg=time_around_beg,
    ).copy(deep=True)
    ds_masked = ds_masked.assign_coords(lengths=('heat_wave', lengths))
    dummy_da = np.zeros((list(ds_masked.dims.values())[:2])) + np.nan
    ds_masked = ds_masked.assign_coords(temperature=(['heat_wave', 'time_around_beg'], dummy_da))
    ds_masked = ds_masked.assign_coords(absolute_time=(['heat_wave', 'time_around_beg'], dummy_da.astype('datetime64[h]')))
    for i, heat_wave in enumerate(heat_waves_ts):
        unexpected_offset = time_before - (heat_waves[i][0] - heat_wave[0])
        this_tab = time_around_beg[:len(heat_wave)] + unexpected_offset
        to_assign = ds.loc[dict(time=heat_wave)].assign_coords(time=this_tab).rename(time='time_around_beg')
        accessor_dict = dict(heat_wave=i, time_around_beg=this_tab)
        ds_masked.loc[accessor_dict] = to_assign
        ds_masked.temperature.loc[accessor_dict] = da.loc[dict(time=heat_wave)].values
        ds_masked.absolute_time.loc[accessor_dict] = heat_wave
    return ds_masked


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
#%%
