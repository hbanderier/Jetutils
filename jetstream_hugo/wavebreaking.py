# coding: utf-8
from typing import Tuple

import numpy as np
import xarray as xr
import polars as pl


from jetstream_hugo.definitions import N_WORKERS, slice_1d
from jetstream_hugo.jet_finding import haversine, create_mappable_iterator, map_maybe_parallel, get_index_columns


def compute_one_wb_props(
    jet: pl.DataFrame, da_pvs: xr.DataArray, every: int = 5
) -> dict:
    jet = jet[::every]
    lo, la = jet.select(["lon", "lat"]).to_numpy().T
    lon, lat = da_pvs.lon.values, da_pvs.lat.values
    dxds = np.gradient(lo)
    dyds = np.gradient(la)
    theta = np.arctan2(dyds, dxds)

    distances = np.full((len(jet), 2), fill_value=-1)
    intensities = np.zeros((len(jet), 2))
    jet_pvs = slice_1d(da_pvs, {"lon": lo, "lat": la}).values.T
    up_or_down = np.argmax(np.abs(jet_pvs), axis=1)
    jet_pvs = np.take_along_axis(jet_pvs, up_or_down[:, None], axis=1).flatten()
    distances[np.where(jet_pvs)[0], :] = 0.0
    intensities[np.where(jet_pvs)[0], :] = jet_pvs[np.where(jet_pvs)[0], None]

    dn = 2
    t_ = np.arange(dn, 20 + dn, dn)

    for k in range(len(jet)):
        if any(distances[k] != -1):
            continue
        for l, side in enumerate([-1, 1]):
            t = side * t_
            normallons = np.cos(theta[k] + np.pi / 2) * t + lo[k]
            normallats = np.sin(theta[k] + np.pi / 2) * t + la[k]
            mask_valid = (
                (normallons >= lon.min())
                & (normallons <= lon.max())
                & (normallats >= lat.min())
                & (normallats <= lat.max())
            )
            if np.sum(mask_valid) == 0:
                continue
            normallons = normallons[mask_valid]
            normallats = normallats[mask_valid]
            normal_pvs = slice_1d(da_pvs, {"lon": normallons, "lat": normallats}).values
            for type_ in [0, 1]:
                if not any(normal_pvs[type_]):
                    continue
                imin = np.argmax(normal_pvs[type_] != 0)
                distance = haversine(lo[k], la[k], normallons[imin], normallats[imin])
                reject_new = (distance >= distances[k, l]) and (distances[k, l] != -1)
                if reject_new:
                    continue
                distances[k, l] = distance
                intensities[k, l] = normal_pvs[type_][imin]

    props = {}
    props["above"] = (
        np.sum(np.abs(intensities[k, 0])) < np.sum(np.abs(intensities[k, 1]))
    ).astype(np.float32)
    dists_good_direction = distances[:, int(props["above"])]
    props["affected_from"] = np.argmax(dists_good_direction != -1)
    props["affected_until"] = len(jet) - np.argmax(dists_good_direction[::-1] != -1) - 1
    slice_ = slice(props["affected_from"], props["affected_until"] + 1)
    props["mean_int"] = np.mean(intensities[slice_, int(props["above"])]).astype(
        np.float32
    )
    props["mean_dist"] = np.mean(dists_good_direction[slice_]).astype(np.float32)
    props["affected_from"] = props["affected_from"].astype(np.float32) * every
    props["affected_until"] = props["affected_until"].astype(np.float32) * every
    return props


def compute_wb_props_wrapper(args: Tuple) -> list:
    jets, da_pvs, mask = args
    if not mask:
        keys = ["above", "affected_from", "affected_until", "mean_int", "mean_dist"]
        props = [
            {key: 0.0 for key in keys}
            for _ in jets.group_by("jet ID", maintain_order=True)
        ]
    else:
        props = []
        for _, (_, jet) in enumerate(jets.group_by("jet ID", maintain_order=True)):
            props.append(compute_one_wb_props(jet, da_pvs, every=4))
    return pl.DataFrame.from_dict(props, dtype=np.float32)


def compute_all_wb_props(
    all_jets_one_df: pl.DataFrame,
    da_pvs: xr.DataArray,
    event_mask: np.ndarray,
    processes: int = N_WORKERS,
    chunksize: int = 100,
) -> xr.Dataset:
    len_, iterator = create_mappable_iterator(all_jets_one_df, [da_pvs], [event_mask])
    print("Computing RWB properties")
    all_props_dfs = map_maybe_parallel(
        iterator,
        compute_wb_props_wrapper,
        len_=len_,
        processes=processes,
        chunksize=chunksize,
    )
    index_columns = get_index_columns(all_props_dfs)
    all_props_df = pl.concat(all_props_dfs).to_pandas().set_index(index_columns)
    return xr.Dataset.from_dataframe(all_props_df)