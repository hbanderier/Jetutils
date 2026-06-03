from functools import partial
import os
from pathlib import Path
from itertools import product

import dask.array as darr
import numpy as np
import polars as pl
import xarray as xr
from jetutils.data import open_da, smooth, extract, standardize
from jetutils.definitions import DATADIR, YEARS, N_WORKERS, DEFAULT_VARNAME, OMEGA, C_P_AIR, KAPPA, degsin, compute, get_index_columns, polars_to_xarray, RADIUS, degcos
from jetutils.derived_quantities import compute_absolute_vorticity, compute_2d_div, compute_norm_derivative, convolve_dask
from jetutils.geospatial import (
    gather_normal_da_jets,
    interp_jets_to_zero_one,
    detect_contours,
    detect_contours_lonlat,
    detect_overturnings,
    event_props,
    to_xarray_sjoin,
    detect_streamers, create_jet_relative_dataset,
)
from jetutils.jet_finding import (
    DataHandler,
    JetFindingExperiment,
    add_feature_for_cat,
    iterate_over_year_maybe_member, do_everything, gaussian_smooth_func,
)
from scipy.signal.windows import lanczos
from tqdm import tqdm, trange

os.environ["RUST_BACKTRACE"] = "full"

# block 1: compute zeta
for run in ["ctrl", "dobl", "ctrl_p4"]:
    basepath_i = Path(DATADIR, f"Henrik_data/{run}/high_wind/6H")
    basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
    basepath_zeta.mkdir(parents=True, exist_ok=True)

    glob = basepath_i.glob("????.nc")

    for f in glob:
        oname = f.name
        ofile = basepath_zeta.joinpath(oname)
        if not ofile.is_file():
            ds = xr.open_dataset(f)
            ds = ds[["u", "v"]]
            ds = ds.transpose("time", "lev", "lat", "lon")
            zeta = compute_absolute_vorticity(ds)
            zeta.to_netcdf(ofile)
            print("zeta", oname, run)

# block 2: compute eddy stuff
for run in ["ctrl", "dobl", "ctrl_p4"]:
    opath = Path(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results",
        "Eddy_NH_10days.zarr",
    )
    if opath.is_dir():
        continue
    half_len = 20
    ds = (
        xr.open_mfdataset(
            f"{DATADIR}/Henrik_data/{run}/high_wind/6H/*.nc",
            combine="nested",
            concat_dim="time",
        )[["u", "v", "theta"]]
        .sel(lat=slice(0, 90))
        .chunk("auto")
    )
    for var in ds.data_vars:
        ds[var] = ds[var].astype(np.float32)
    huh = xr.open_mfdataset(
        f"{DATADIR}/Henrik_data/{run}/vertical/6H/*.nc",
        combine="nested",
        concat_dim="time",
    )
    try:
        huh = huh.rename({"OMEGA": "omega"})
    except (ValueError, KeyError):
        pass
    ds["omega"] = (
        huh["omega"]
        .astype(np.float32)
        .sel(lat=slice(0, 90))
        .chunk("auto")
    )
    l_win = lanczos(2 * half_len + 1)[:, None, None, None]
    dims = ds.dims
    for var in ds.data_vars:
        ds[f"{var}bar"] = (
            dims,
            (convolve_dask(ds[var].data, l_win)[half_len:-half_len] / l_win.sum()).astype(
                np.float32
            ),
        )
        ds[f"{var}p"] = ds[var] - ds[f"{var}bar"]
        del ds[f"{var}bar"]
        del ds[var]
    ds = ds.chunk({"time": 1390, "lat": 72, "lon": 161})
    res = ds.to_zarr(opath, compute=False)
    compute(res, progress=True)
    
    
# block 3: EP Flux
# for run in ["ctrl", "dobl", "ctrl_p4"]:
#     ipath = Path(
#         f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results",
#         "Eddy_NH_10days.zarr",
#     )
#     odir = Path(f"{DATADIR}/Henrik_data/{run}/EPF/6H")
#     odir.mkdir(parents=True, exist_ok=True)
#     bigds = xr.open_dataset(ipath).sel(lev=[20000, 30000]).chunk("auto")
#     for year in trange(1969, 2021):
#         ofile = odir.joinpath(f"{year}.nc")
#         if ofile.is_file():
#             continue
#         ds = bigds.sel(time=bigds.time.dt.year == year)
#         other = xr.open_dataset(f"{DATADIR}/Henrik_data/{run}/vertical/6H/{year}.nc").sel(lat=slice(0, None))
#         ds = xr.merge([ds, other])
#         ds["u"] = xr.open_dataset(f"{DATADIR}/Henrik_data/{run}/high_wind/6H/{year}.nc")["u"].sel(lat=slice(0, None))
        
#         gamma = (-KAPPA / ds.lev * (100000 / ds.lev) ** KAPPA * ds["dthetadp"].mean(["time", "lon", "lat"])).astype(np.float32)
#         EAPE = (C_P_AIR * 0.5 * (ds.lev * 1e-5) ** (2 * KAPPA) * gamma * ds["thetap"] ** 2).astype(np.float32)
#         S = (0.5 * (ds["up"] ** 2 + ds["vp"] ** 2 - EAPE)).astype(np.float32)
#         f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)

#         ## Base 2 * 3
#         ds["F11"] = ds["up"] ** 2 - S
#         ds["F12"] = ds["up"] * ds["vp"]
#         ds["F13"] = - ds["vp"] * ds["thetap"] * f / ds["dthetadp"]
#         ds["F21"] = ds["up"] * ds["vp"]
#         ds["F22"] = ds["vp"] ** 2 - S
#         ds["F23"] = ds["up"] * ds["thetap"] * f / ds["dthetadp"]

#         ## Additional from original EP:
#         ds["F12_extra"] = - ds["dudp"] * ds["vp"] * ds["thetap"] / ds["dthetadp"]
#         ds["F13_extra"] = ds["up"] * ds["omegap"]
#         ds["F23_extra"] = ds["vp"] * ds["omegap"]
#         ds = ds.drop_vars([var for var in list(ds.data_vars) if var[0] != "F"])
#         ds = compute(ds, progress_flag=False)
#         ds.to_netcdf(ofile)
    

# block 4: WB
# levels = compute(xr.open_mfdataset(list(Path(DATADIR, "Henrik_data/ctrl/zeta/6H").glob("*0.nc")))[DEFAULT_VARNAME].quantile([0.5]), progress_flag=True).values
# levels = (levels * 1e5).round(1).tolist()
# for run in ["ctrl", "dobl", "ctrl_p4"]:
#     basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
#     da_mflux = xr.open_dataset(
#         f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
#     ).sel(lev=30000)
#     da_mflux = (da_mflux["up"] * da_mflux["vp"]).rename("EMF")
#     opath = Path(DATADIR, "Henrik_data", run) 
#     opath_rwb = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
#     opath_rwb.mkdir(exist_ok=True)
#     for year in trange(1969, 2021):
#         ofile = opath_rwb.joinpath(f"overturnings_{year}.parquet")
        
#         if opath.joinpath(f"CAVO/6H/{year}.nc").is_file() and ofile.is_file():
#             continue
            
#         zeta = xr.open_dataarray(basepath_zeta.joinpath(f"{year}.nc")).sel(lev=30000)
#         zeta = zeta.rename("zeta") * 1e5
#         for potential in ["lev", "loni", "lati"]:
#             try:
#                 zeta = zeta.reset_coords(potential, drop=True)
#             except ValueError:
#                 continue
#         mflux = da_mflux.sel(time=zeta.time)
#         mflux = mflux.rename("mflux").reset_coords("lev", drop=True)
#         zeta = smooth(zeta, {"lon": ("win", 5), "lat": ("win", 5)})
#         zeta = compute(zeta)
#         # mflux = smooth(mflux, {"lon": ("win", 5), "lat": ("win", 5)})
#         mflux = compute(mflux)
#         if ofile.is_file():
#             overturnings = pl.read_parquet(ofile)
#             overturnings_on_grid = None
#         else:
#             contours = detect_contours_lonlat(zeta, levels, processes=N_WORKERS, ctx="fork")
#             overturnings = detect_overturnings(contours, max_difflon=3)
#             overturnings, overturnings_on_grid = event_props(overturnings, [zeta, mflux])
#             overturnings.write_parquet(ofile)
            
#         for orientation in ["cyclonic", "anticyclonic"]:
#             name = f"{orientation[0].upper()}AVO"
#             odir = opath.joinpath(f"{name}/6H")
#             odir.mkdir(parents=True, exist_ok=True)
#             ofile = odir.joinpath(f"{year}.nc")
#             if ofile.is_file():
#                 continue
#             df = overturnings.filter(pl.col("orientation") == orientation)
#             da = to_xarray_sjoin(zeta, events=df)
#             da.to_netcdf(ofile)
            
#             odir = opath.joinpath(f"{name}/dailyany")
#             odir.mkdir(parents=True, exist_ok=True)
#             da = da.any("level").resample(time="1D").any().astype(np.uint8)
#             da.to_netcdf(odir.joinpath(f"{year}.nc"))
            
# block 4.5: Streamers
# levels = compute(xr.open_mfdataset(list(Path(DATADIR, "Henrik_data/ctrl/zeta/6H").glob("*0.nc")))["__xarray_dataarray_variable__"].quantile([0.5]), progress_flag=True).values
# levels = (levels * 1e5).round(1).tolist()

# filters_type = {
#     "stratospheric": pl.col("zeta") >= pl.col("level"),
#     "tropospheric": pl.col("zeta") < pl.col("level")
# }
# filters_orientation = {
#     "anticyclonic": pl.col("mflux") <= 0.,
#     "cyclonic": pl.col("mflux") > 0.
# }
# for run in ["ctrl", "dobl", "ctrl_p4"]:
#     basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
#     da_mflux = xr.open_dataset(
#         f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
#     ).sel(lev=30000)
#     da_mflux = (da_mflux["up"] * da_mflux["vp"]).rename("EMF")
#     opath = Path(DATADIR, "Henrik_data", run) 
#     opath_rwb = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
#     opath_rwb.mkdir(exist_ok=True)
#     for year in trange(1969, 2021):
#         ofile = opath_rwb.joinpath(f"streamers_{year}.parquet")
        
#         if opath.joinpath(f"TCAVS/6H/{year}.nc").is_file() and ofile.is_file():
#             continue
            
#         zeta = xr.open_dataarray(basepath_zeta.joinpath(f"{year}.nc")).sel(lev=30000)
#         zeta = zeta.rename("zeta") * 1e5
#         for potential in ["lev", "loni", "lati"]:
#             try:
#                 zeta = zeta.reset_coords(potential, drop=True)
#             except ValueError:
#                 continue
#         mflux = da_mflux.sel(time=zeta.time)
#         mflux = mflux.rename("mflux").reset_coords("lev", drop=True)
#         zeta = smooth(zeta, {"lon": ("win", 5), "lat": ("win", 5)})
#         zeta = compute(zeta)
#         # mflux = smooth(mflux, {"lon": ("win", 3), "lat": ("win", 3)})
#         mflux = compute(mflux)
#         if ofile.is_file():
#             streamers = pl.read_parquet(ofile)
#             streamers_on_grid = None
#         else:
#             contours = detect_contours_lonlat(zeta, levels, processes=N_WORKERS, ctx="fork")
#             streamers = detect_streamers(contours)
#             streamers, streamers_on_grid = event_props(streamers, [zeta, mflux])
#             streamers.write_parquet(ofile)
            
        
#         for type_, orientation in product(["stratospheric", "tropospheric"], ["cyclonic", "anticyclonic"]):
#             name = f"{type_[0].upper()}{orientation[0].upper()}AVS"
#             odir = opath.joinpath(f"{name}/6H")
#             odir.mkdir(parents=True, exist_ok=True)
#             ofile = odir.joinpath(f"{year}.nc")
#             if ofile.is_file():
#                 continue
#             f1 = filters_type[type_]
#             f2 = filters_orientation[orientation]
#             df = streamers.filter(f1, f2)
#             da = to_xarray_sjoin(zeta, events=df)
#             da.to_netcdf(ofile)
            
#             odir = opath.joinpath(f"{name}/dailyany")
#             odir.mkdir(parents=True, exist_ok=True)
#             da = da.any("level").resample(time="1D").any().astype(np.uint8)
#             da.to_netcdf(odir.joinpath(f"{year}.nc"))
            
# block 5: define jets (already computed probably)

kwargs = dict(
    n_coarsen=1,
    base_s_thresh=0.55,
    alignment_thresh=0.6,
    int_thresh_factor=0.6,
    hole_size=6,
    smooth_func=partial(gaussian_smooth_func, sigma_lon=2, sigma_lat=0.8),
)

both_jets = {}
both_paths = {}
for run in ["ctrl", "dobl"]:
    path = Path(DATADIR, "Henrik_data", run, "high_wind/6H/results/2")
    ds = xr.open_dataset(path.joinpath("../1/da.nc"))
    ds = standardize(ds)
    ds = extract(
        ds, minlon=-80, maxlon=40, minlat=15, maxlat=80
    )
    theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
    ds = xr.merge([ds, theta300])
    jets, ph_jets, props, props_full = do_everything(ds, path, feature_names=("s300", "theta300"), **kwargs)
    both_paths[run] = path
    both_jets[run] = ph_jets

# stage 6: Interpolate new fields
args = ["all", None, -100, 60, 0, 88]

to_do = (
    ("theta300", ("high_wind", "theta"), {"levels": 30000}),
    ("s300", ("high_wind", "s"), {"levels": 30000}),
    ("zeta300", "zeta", {"levels": 30000}),
    ("z500", "z", {}),
    ("t_low", "t_low", {"levels": 100000}),
    ("PTTEND500", ("heating", "PTTEND"), {}),
    ("DTCOND500", ("heating", "DTCOND"), {}),
    ("AAVO", "AAVO", {}),
    ("CAVO", "CAVO", {}),
    ("SAAVS", "SAAVS", {}),
    ("SCAVS", "SCAVS", {}),
    ("TAAVS", "TAAVS", {}),
    ("TCAVS", "TCAVS", {}),
    ("EKE300", "EKE", {}),
)

for run in ["ctrl", "dobl"]:
    jets = both_jets[run]
    bc = both_paths[run].joinpath("bias_correct.parquet")
    bc = pl.read_parquet(bc)
    for huh in to_do:
        rename, name, kwargs = huh
        ofile = both_paths[run].joinpath(f"{rename}_relative.parquet")
        if ofile.is_file():
            continue
        if name != "EKE":
            da_ = open_da("Henrik_data", run, name, "6H", *args, **kwargs).rename(rename)
        else:
            ds = xr.open_dataset(
                f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
            ).sel(lev=30000)
            ds = extract(ds, *args)
            da_ = 0.5 * np.sqrt(ds["up"] ** 2 + ds["vp"] ** 2)
            da_ = da_.rename(rename)
        if rename in ["AAVO", "CAVO"] and "lev" in da_.dims:
            da_ = da_.isel(lev=2)
        da_ = compute(da_)
        interpd = create_jet_relative_dataset(jets, da_, bias_correction=bc)
        del da_
        interpd.write_parquet(ofile)


for run in ["ctrl", "dobl"]:
    jets = both_jets[run]    
    opaths = {}
    half_length = 2e6
    dn = 1e5
    n_interp = 30
    mapping = {
        f"F{j}": [f"F1{j}", f"F2{j}"]
        for j in ["1", "2", "3", "3_extra"]
    } | {
        key: [f"{key}1", f"{key}2"]
        for key in ["hor", "vert", "vert_extra"]
    }
    for key in mapping:
        opaths[key] = both_paths[run].joinpath(f"{key}_relative.parquet")
    if all([opath.is_file() for opath in opaths.values()]):
        continue
    tmp_folder = both_paths[run].joinpath("tmp_rel")
    tmp_folder.mkdir(exist_ok=True)
    for year in trange(1969, 2021):
        df = jets.filter(pl.col("time").dt.year() == year)
        ds = xr.open_dataset(f"{DATADIR}/Henrik_data/ctrl/EPF/6H/{year}.nc")
        ds = extract(ds, *args)
        for f in ds.data_vars:
            if f[0] == "F":
                ds[f] = ds[f] * RADIUS * degcos(ds["lat"])
        if not opaths["vert"].is_file():
            ds["vert1"] = ds["F13"].differentiate("lev")
            ds["vert2"] = ds["F23"].differentiate("lev")
        if not opaths["vert_extra"].is_file():
            ds["vert_extra1"] = ds["F13_extra"].differentiate("lev")
            ds["vert_extra2"] = ds["F23_extra"].differentiate("lev")
        if not opaths["hor"].is_file():
            ds = ds.sel(lev=30000)
            ds["hor1"] = compute_2d_div(ds, "F11", "F12")
            ds["hor2"] = compute_2d_div(ds, "F21", "F22")
        for dest, sources in mapping.items():
            this_ofile = tmp_folder.joinpath(f"{dest}_{year}.parquet")
            if this_ofile.is_file() or opaths[dest].is_file():
                continue
            varname = f"{dest}_interp"
            df_interp = gather_normal_da_jets(
                df, ds[sources[0]], ds[sources[1]], half_length=half_length, dn=dn, in_meters=True
            )
            angle = pl.col("angle") - pl.lit(np.pi / 2)
            agg = angle.cos() * pl.col(f"{sources[0]}_interp") + angle.sin() * pl.col(f"{sources[1]}_interp")
            agg = agg.cast(pl.Float32())
            
            df_interp = df_interp.with_columns(**{varname: agg}).drop(f"{sources[0]}_interp", f"{sources[1]}_interp")
            
            df_interp = interp_jets_to_zero_one(
                df_interp, [varname, "is_polar"], n_interp=n_interp
            )
            df_interp.write_parquet(this_ofile)
    for key in mapping:
        opath = opaths[key]
        if opath.is_file():
            continue
        df = []
        for year in range(1969, 2021):
            df.append(pl.read_parquet(tmp_folder.joinpath(f"{key}_{year}.parquet")))
        df = pl.concat(df).cast({"norm_index": pl.Float32(), "n": pl.Float32(), f"{key}_interp": pl.Float32()})
        df.write_parquet(opath)
    for f in tmp_folder.iterdir():
        f.unlink()
    tmp_folder.rmdir()
            