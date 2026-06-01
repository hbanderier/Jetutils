

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

from cdo import Cdo
cdo = Cdo()
basepath = Path(f"{DATADIR}/ERA5/thetalev")
fl1 = basepath.joinpath("u").joinpath("6H").glob("*.nc")
fl2 = basepath.joinpath("v").joinpath("6H").glob("*.nc")
for file1, file2 in tqdm(zip(fl1, fl2), total=64):
    dest_file = basepath.joinpath("s").joinpath("6H").joinpath(file1.name)
    if dest_file.is_file():
        continue
    print(file1.stem)
    cdo.chname(
        "u,s",
        input="-sqrt -add -sqr -selname,U "
        + file1.as_posix()
        + " -sqr -selname,V "
        + file2.as_posix(),
        output=dest_file.as_posix(),
    )