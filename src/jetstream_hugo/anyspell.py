from typing import Union, Tuple, Iterable, Literal
from nptyping import NDArray
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from xclim.indices.run_length import rle, run_bounds


def spells_from_da(
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
    days = (da > da.quantile(dim='time', q=q)) if q > 0.5 else (da < da.quantile(dim='time', q=q))
    if fill_holes:
        holes = rle(~days)
        holes = np.where(holes.values == 1)[0]
        days[holes] = 1
    spells = run_bounds(days)
    mask = (spells[0].dt.year == spells[1].dt.year).values
    spells = spells[:, mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mask = spells.astype('datetime64[h]').values
    mask = (mask[1] - mask[0]) >= minlen
    spells = spells[:, mask].T
    spells_ts = []
    lengths = []
    for spell in spells:
        hw_len = (spell[1] - spell[0]).values.astype('timedelta64[D]')
        year = spell[0].dt.year.values
        min_time = np.datetime64(f'{year}-{months[0]}-01T00:00')
        max_time = np.datetime64(f'{year}-{months[1]}-01T00:00') - dt
        first_time = max(min_time, (spell[0] - time_before).values)
        last_time = min(max_time, (spell[1] + time_after).values)
        spell_ts = pd.date_range(first_time, last_time, freq='6h')
        spells_ts.append(spell_ts)
        lengths.append(np.full(len(spell_ts), hw_len.astype(int)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        spells = spells.astype('datetime64[h]').values
    if output_type == "list":
        return spells_ts, spells
    da_spells = da.copy(data=np.zeros(da.shape, dtype=int))
    da_spells.loc[np.concatenate(spells_ts)] = np.concatenate(lengths)
    if output_type == "arr":
        return da_spells
    return spells_ts, spells, da_spells


def mask_from_spells(da: xr.DataArray, ds: xr.Dataset, spells_ts, spells, time_before: pd.Timedelta = pd.Timedelta(0, 'D')) -> xr.Dataset:
    months = np.unique(ds.time.dt.month.values)
    months = [str(months[0]).zfill(2), str(min(12, months[-1] + 1)).zfill(2)]
    lengths = spells[:, 1] - spells[:, 0]
    longest_hotspell = np.argmax(lengths)
    time_around_beg = spells_ts[longest_hotspell] - spells[longest_hotspell, 0]
    time_around_beg = time_around_beg.values
    ds_masked = ds.loc[dict(time=ds.time.values[0])].reset_coords('time', drop=True).copy(deep=True)
    ds_masked.loc[dict()] = np.nan
    ds_masked = ds_masked.expand_dims(
        heat_wave=np.arange(len(spells)),
        time_around_beg=time_around_beg,
    ).copy(deep=True)
    ds_masked = ds_masked.assign_coords(lengths=('heat_wave', lengths))
    dims = list(ds_masked.sizes.values())[:2]
    dummy_da = np.zeros(dims) + np.nan
    ds_masked = ds_masked.assign_coords(temperature=(['heat_wave', 'time_around_beg'], dummy_da))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ds_masked = ds_masked.assign_coords(absolute_time=(['heat_wave', 'time_around_beg'], dummy_da.astype('datetime64[h]')))
    for i, spell in enumerate(spells_ts):
        unexpected_offset = time_before - (spells[i][0] - spell[0])
        this_tab = time_around_beg[:len(spell)] + unexpected_offset
        to_assign = ds.loc[dict(time=spell)].assign_coords(time=this_tab).rename(time='time_around_beg')
        accessor_dict = dict(heat_wave=i, time_around_beg=this_tab)
        ds_masked.loc[accessor_dict] = to_assign
        ds_masked.temperature.loc[accessor_dict] = da.loc[dict(time=spell)].values
        ds_masked.absolute_time.loc[accessor_dict] = spell
    return ds_masked


def mask_from_da(
    da: xr.DataArray,
    ds: xr.Dataset,
    q: float = 0.95,
    fill_holes: bool = False,
    minlen: np.timedelta64 = np.timedelta64(3, 'D'),
    time_before: pd.Timedelta = pd.Timedelta(0, 'D'),
    time_after: pd.Timedelta = pd.Timedelta(0, 'D'),
) -> xr.Dataset:
    spells_ts, spells = spells_from_da(
        da, q, fill_holes, minlen, time_before, time_after, output_type='list'
    )
    return mask_from_spells(da, ds, spells_ts, spells, time_before)
    

