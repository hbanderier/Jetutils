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
    output_type: Literal["arr"] | Literal["list"] | Literal["both"] = "arr"
) -> xr.DataArray | Tuple[list[NDArray]]:
    dt = pd.Timedelta(da.time.values[1] - da.time.values[0])
    months = np.unique(da.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
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
    heat_waves = heat_waves.astype('datetime64[h]').values
    if output_type == "list":
        return heat_waves_ts, heat_waves
    da_hs = da.copy(data=np.zeros(da.shape, dtype=int))
    da_hs.loc[np.concatenate(heat_waves_ts)] = np.concatenate(lengths)
    if output_type == "arr":
        return da_hs
    return heat_waves_ts, heat_waves, da_hs


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
    months = np.unique(ds.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
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
    dims = list(ds_masked.sizes.values())[:2]
    dummy_da = np.zeros(dims) + np.nan
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

