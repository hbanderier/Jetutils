from functools import partial
import os
from pathlib import Path
from itertools import product

import dask.array as darr
import numpy as np
import polars as pl
import xarray as xr
from zarr.codecs.numcodecs import Blosc
from jetutils.data import open_da, smooth, extract, standardize, standardize_polars_dtypes, compute_all_smoothed_anomalies
from jetutils.definitions import (
    DATADIR,
    YEARS,
    N_WORKERS,
    DEFAULT_VARNAME,
    OMEGA,
    C_P_AIR,
    KAPPA,
    RADIUS,
    degcos,
    degsin,
    compute,
    get_index_columns,
    polars_to_xarray, R_SPECIFIC_AIR,
)
from jetutils.derived_quantities import (
    compute_absolute_vorticity,
    compute_2d_div,
    compute_norm_derivative,
    convolve_dask,
)
from jetutils.geospatial import (
    gather_normal_da_jets,
    interp_jets_to_zero_one,
    detect_contours,
    detect_contours_lonlat,
    detect_overturnings,
    event_props,
    to_xarray_sjoin,
    detect_streamers,
    create_jet_relative_dataset,
    create_bias_correction,
)
from jetutils.jet_finding import (
    DataHandler,
    JetFindingExperiment,
    add_feature_for_cat,
    iterate_over_year_maybe_member,
    do_everything,
    gaussian_smooth_func,
    to_one_large,
)
from scipy.signal.windows import lanczos
from scipy.constants import g
from tqdm import tqdm, trange

os.environ["RUST_BACKTRACE"] = "full"
Basepath = Path(DATADIR, "ERA5/plev/uv/6H")

compute_all_smoothed_anomalies("ERA5", "surf", "tp", "6H", "dayofyear", {"dayofyear": ("win", 15)})

# # block 1: compute eddy stuff
for n_days in [5, 10, 20, 30]:
    izarr = Basepath.joinpath("full.zarr")
    ozarr = Basepath.joinpath(f"results/eddy_stuff_{n_days}days.zarr")
    if not ozarr.is_dir():
        half_len = n_days * 2
        ds = xr.open_dataset(izarr, consolidated=False).chunk({"time": 1000, "lev": 1, "lat": -1, "lon": -1})
        ds = ds.sel(lev=[250, 300])
        ds["theta"] = ds["t"] * (1000 / ds["lev"]) ** KAPPA

        l_win = lanczos(2 * half_len + 1)[:, None, None, None]
        dims = ds.dims
        for var in ["u", "v", "theta", "omega"]:
            ds[f"{var}bar"] = (
                dims,
                (
                    convolve_dask(ds[var].data, l_win)[half_len:-half_len] / l_win.sum()
                ).astype(np.float32),
            )
            ds[f"{var}p"] = ds[var] - ds[f"{var}bar"]
            del ds[f"{var}bar"]
            del ds[var]
        encoding = {}
        compressor = Blosc(cname="zstd", clevel=3, shuffle=2)
        for data_var in ds.data_vars:
            encoding[data_var] = {"compressors": compressor}

        res = ds.to_zarr(ozarr, compute=False, consolidated=False, encoding=encoding)
        compute(res, progress_flag=True)
        
        
    izarr1 = Basepath.joinpath("full.zarr")
    izarr2 = Basepath.joinpath(f"results/eddy_stuff_{n_days}days.zarr")
    ozarr = Basepath.joinpath(f"results/eddy_forcing_{n_days}days.zarr")
    if not ozarr.is_dir():
        ds = xr.open_dataset(izarr2, consolidated=False).chunk({"time": 100, "lev": -1, "lat": -1, "lon": -1})
        theta = xr.open_dataset(izarr1, consolidated=False).chunk({"time": 100, "lev": -1, "lat": -1, "lon": -1})["t"]
        theta = theta * (1000 / theta["lev"]) ** KAPPA
        ds["dthetadp"] = theta.differentiate("lev")
        ds["EKE"] = (0.5 * (ds["up"] ** 2 + ds["vp"] ** 2)).astype(np.float32)
        gamma = (-KAPPA / ds.lev * (1000 / ds.lev) ** KAPPA * ds["dthetadp"].mean(["time", "lon", "lat"])).astype(np.float32)
        EAPE = (C_P_AIR * 0.5 * (ds.lev * 1e-3) ** (2 * KAPPA) * gamma * ds["thetap"] ** 2).astype(np.float32)
        S = (ds["EKE"] - 0.5 * EAPE).astype(np.float32)
        f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)

        ## Base 2 * 3
        ds["F11"] = S - ds["up"] ** 2
        ds["F12"] = - ds["up"] * ds["vp"]
        ds["F13"] = ds["vp"] * ds["thetap"] * f / ds["dthetadp"]
        ds["F22"] = S - ds["vp"] ** 2
        ds["F23"] = ds["up"] * ds["thetap"] * f / ds["dthetadp"]

        ## Additional from original EP:
        ds["F13_extra"] = - ds["up"] * ds["omegap"] / 100
        ds["F23_extra"] = - ds["vp"] * ds["omegap"] / 100

        for v in ds.data_vars:
            if v[0] == "F":
                ds[v] = ds[v] * RADIUS * degcos(ds.lat)

        ds = ds.unify_chunks()
            
        ds["hor1"] = ds.map_blocks(compute_2d_div, ["F11", "F12"], template=ds["F11"])
        ds["hor2"] = ds.map_blocks(compute_2d_div, ["F12", "F22"], template=ds["F11"])

        ds["vert1"] = (ds["F13"] + ds["F13_extra"]).differentiate("lev")
        ds["vert2"] = (ds["F23"] + ds["F23_extra"]).differentiate("lev")

        ds = ds.drop_vars(["up", "vp", "thetap", "omegap", "dthetadp", *[v for v in ds.data_vars if v[0] == "F"]])
        ds = ds.sel(lev=[300, 250])

        encoding = {}
        compressor = Blosc(cname="zstd", clevel=3, shuffle=2)
        for data_var in ds.data_vars:
            encoding[data_var] = {"compressors": compressor}

        res = ds.to_zarr(ozarr, compute=False, consolidated=False, encoding=encoding)
        compute(res, progress_flag=True)

n_days = 5
izarr = Basepath.joinpath("full.zarr")
ozarr = Basepath.joinpath("results/eady_growth.zarr")
if not ozarr.is_dir():
    ds = xr.open_dataset(izarr, consolidated=False).chunk({"time": 200, "lev": -1, "lat": -1, "lon": -1})
    theta = ds["t"] * (1000 / ds["lev"]) ** KAPPA
    rho = (ds["lev"] * 100 / ds["t"] / R_SPECIFIC_AIR)
    # hadas 2025
    dthetadp = (theta.sel(lev=300) - theta.sel(lev=850)) / (30000 - 85000)
    N350 = np.sqrt(- rho.sel(lev=350) * g * g / theta.sel(lev=350) * dthetadp)
    f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)
    s = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)

    dudp = (ds["u"].sel(lev=300) - ds["u"].sel(lev=850)) / (30000 - 85000)
    dvdp = (ds["v"].sel(lev=300) - ds["v"].sel(lev=850)) / (30000 - 85000)
    dsdp = (s.sel(lev=300) - s.sel(lev=850)) / (30000 - 85000)

    eady1 = 0.3098 * g * (np.abs(f) * np.abs(dudp) / N350).transpose("time", "lat", "lon")
    eady2 = 0.3098 * g * (np.abs(f) * np.abs(dvdp) / N350).transpose("time", "lat", "lon")
    eady3 = 0.3098 * g * (np.abs(f) * np.abs(dsdp) / N350).transpose("time", "lat", "lon")

    ds_eady = xr.Dataset({"eady_u": eady1, "eady_v": eady2, "eady_s": eady3})

    encoding = {}
    compressor = Blosc(cname="zstd", clevel=3, shuffle=2)
    for data_var in ds_eady.data_vars:
        encoding[data_var] = {"compressors": compressor}

    res = ds_eady.to_zarr(ozarr, compute=False, consolidated=False, encoding=encoding)
    compute(res, progress_flag=True)
else:
    print(ozarr)

# block 3: EP Flux
# ipath = Path(
#     f"{DATADIR}/ERA5/plev/uv/6H/results",
#     "Eddy_uv_natl_10days.zarr",
# )
# odir = Path(f"{DATADIR}/ERA5/plev/eddy_stuff/6H")
# odir.mkdir(parents=True, exist_ok=True)
# bigds = xr.open_dataset(ipath).chunk("auto")
# for year in tqdm(YEARS):
#     ofile = odir.joinpath(f"{year}.nc")
#     if ofile.is_file() or True:
#         continue
#     ds = bigds.sel(time=bigds.time.dt.year == year)
#     # ds["u"] = xr.open_dataset(f"{DATADIR}/Henrik_data/{run}/high_wind/6H/{year}.nc")["u"].sel(lat=slice(0, None))

#     # gamma = (-KAPPA / ds.lev * (100000 / ds.lev) ** KAPPA * ds["dthetadp"].mean(["time", "lon", "lat"])).astype(np.float32)
#     # EAPE = (C_P_AIR * 0.5 * (ds.lev * 1e-5) ** (2 * KAPPA) * gamma * ds["thetap"] ** 2).astype(np.float32)
#     # S = (0.5 * (ds["up"] ** 2 + ds["vp"] ** 2 - EAPE)).astype(np.float32)
#     EKE = 0.5 * (ds["up"] ** 2 + ds["vp"] ** 2)
#     EKE = EKE.astype(np.float32)
#     f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)

#     ## Base 2 * 3
#     ds["EKE"] = EKE
#     ds["F11"] = ds["up"] ** 2 - EKE
#     ds["F12"] = ds["up"] * ds["vp"]
#     # ds["F13"] = - ds["vp"] * ds["thetap"] * f / ds["dthetadp"]
#     # ds["F21"] = ds["up"] * ds["vp"]
#     ds["F22"] = ds["vp"] ** 2 - EKE
#     # ds["F23"] = ds["up"] * ds["thetap"] * f / ds["dthetadp"]

#     ## Additional from original EP:
#     # ds["F12_extra"] = - ds["dudp"] * ds["vp"] * ds["thetap"] / ds["dthetadp"]
#     # ds["F13_extra"] = ds["up"] * ds["omegap"]
#     # ds["F23_extra"] = ds["vp"] * ds["omegap"]
#     ds = ds.drop_vars([var for var in list(ds.data_vars) if var[0] not in ["E", "F"]])
#     ds = compute(ds, progress_flag=False)
#     ds.to_netcdf(ofile)


# odir = Path(f"{DATADIR}/ERA5/plev/eddy_stuff/6H")
# odir.mkdir(parents=True, exist_ok=True)
# ds_eddies = standardize(
#     xr.open_dataset(Path(DATADIR, "ERA5/plev/uv/6H/results/Eddy_uv_natl_10days.zarr"))
# ).sel(lev=250)
# if not odir.joinpath("full.zarr").is_dir():
#     for i, year in enumerate(tqdm(YEARS)):
#         bigds = ds_eddies.sel(time=ds_eddies.time.dt.year == year)
#         ds = {}
#         ds["EKE"] = 0.5 * (bigds["up"] ** 2 + bigds["vp"] ** 2)
#         ds["F11"] = bigds["up"] ** 2 - ds["EKE"]
#         ds["F12"] = bigds["up"] * bigds["vp"]
#         ds["F22"] = bigds["vp"] ** 2 - ds["EKE"]
#         ds = xr.Dataset(ds)
#         ds["hor1"] = compute_2d_div(ds, "F11", "F12")
#         ds["hor2"] = compute_2d_div(ds, "F12", "F22")
#         ds = compute(ds, progress_flag=False)
#         kwargs = {"mode": "w"} if i == 0 else {"mode": "a", "append_dim": "time"}
#         ds.to_zarr(odir.joinpath("full.zarr"), **kwargs)


# block 4: WB
# levels = compute(xr.open_mfdataset(list(Path(DATADIR, "Henrik_data/ctrl/zeta/6H").glob("*0.nc")))[DEFAULT_VARNAME].quantile([0.5]), progress_flag=True).values
# levels = (levels * 1e5).round(1).tolist()
# basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
# da_mflux = xr.open_dataset(
#     f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
# ).sel(lev=30000)
# da_mflux = (da_mflux["up"] * da_mflux["vp"]).rename("EMF")
# opath = Path(DATADIR, "Henrik_data", run)
# opath_rwb = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
# opath_rwb.mkdir(exist_ok=True)
# for year in trange(1959, 2025):
#     if True:
#         continue
    # ofile = opath_rwb.joinpath(f"overturnings_{year}.parquet")

    # if opath.joinpath(f"CAVO/6H/{year}.nc").is_file() and ofile.is_file():
    #     continue

    # zeta = xr.open_dataarray(basepath_zeta.joinpath(f"{year}.nc")).sel(lev=30000)
    # zeta = zeta.rename("zeta") * 1e5
    # for potential in ["lev", "loni", "lati"]:
    #     try:
    #         zeta = zeta.reset_coords(potential, drop=True)
    #     except ValueError:
    #         continue
    # mflux = da_mflux.sel(time=zeta.time)
    # mflux = mflux.rename("mflux").reset_coords("lev", drop=True)
    # zeta = smooth(zeta, {"lon": ("win", 5), "lat": ("win", 5)})
    # zeta = compute(zeta)
    # # mflux = smooth(mflux, {"lon": ("win", 5), "lat": ("win", 5)})
    # mflux = compute(mflux)
    # if ofile.is_file():
    #     overturnings = pl.read_parquet(ofile)
    #     overturnings_on_grid = None
    # else:
    #     contours = detect_contours_lonlat(zeta, levels, processes=N_WORKERS, ctx="fork")
    #     overturnings = detect_overturnings(contours, max_difflon=3)
    #     overturnings, overturnings_on_grid = event_props(overturnings, [zeta, mflux])
    #     overturnings.write_parquet(ofile)

    # for orientation in ["cyclonic", "anticyclonic"]:
    #     name = f"{orientation[0].upper()}AVO"
    #     odir = opath.joinpath(f"{name}/6H")
    #     odir.mkdir(parents=True, exist_ok=True)
    #     ofile = odir.joinpath(f"{year}.nc")
    #     if ofile.is_file():
    #         continue
    #     df = overturnings.filter(pl.col("orientation") == orientation)
    #     da = to_xarray_sjoin(zeta, events=df)
    #     da.to_netcdf(ofile)

    #     odir = opath.joinpath(f"{name}/dailyany")
    #     odir.mkdir(parents=True, exist_ok=True)
    #     da = da.any("level").resample(time="1D").any().astype(np.uint8)
    #     da.to_netcdf(odir.joinpath(f"{year}.nc"))

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

path = Path(DATADIR, "ERA5/plev/high_wind/6H/results/9")
ds = xr.open_dataset(path.joinpath("da.nc"))
ds = standardize(ds)
ds = extract(
    ds, minlon=-80, maxlon=40, minlat=15, maxlat=80
)
times = ds["time"].values

find_jets_kwargs = dict(
    n_coarsen=3,
    base_s_thresh=0.55,
    alignment_thresh=0.6,
    int_thresh_factor=0.6,
    hole_size=6,
    smooth_func=partial(gaussian_smooth_func, sigma_lon=2, sigma_lat=0.8),
)
# jets, phat_jets, props, props_full = do_everything(ds, path, **find_jets_kwargs, track_large=False)

# stage 6: Interpolate new fields
args = ["all", None, -80, 40, 0, 85]

to_do = (
    ("t2m", "surf", "t2m", {}),
    ("tp", "surf", "tp", {}),
    ("APVO", "thetalev", "APVO_new", {}),
    ("CPVO", "thetalev", "CPVO_new", {}),
    ("theta", "surf", ("alot2pvu", "theta"), {}),
    ("t850", "plev", ("uv", "t"), {"levels": 850}),
    ("t350", "plev", ("uv", "t"), {"levels": 350}),
    ("t300", "plev", ("uv", "t"), {"levels": 300}),
    ("t250", "plev", ("uv", "t"), {"levels": 250}),
    ("t225", "plev", ("uv", "t"), {"levels": 225}),
    ("t200", "plev", ("uv", "t"), {"levels": 200}),
)

jets = pl.read_parquet(path.joinpath("jets.parquet"))
bc_path = path.joinpath("bias_correct.parquet")
if not bc_path.is_file():
    bias_correction = create_bias_correction(jets, ds)
    bias_correction.write_parquet(path.joinpath("bias_correct.parquet"))
else:
    bias_correction = pl.read_parquet(path.joinpath("bias_correct.parquet"))
    
    
def _do_one(path, rename, levtype, name, args, kwargs):
    ofile = path.joinpath(f"{rename}_relative.parquet")
    if ofile.is_file():
        return
    da = open_da("ERA5", levtype, name, "6H", *args, **kwargs).rename(rename)
    if kwargs["levels"] == "all":
        da = da.any("lev")
    interpd = create_jet_relative_dataset(jets, da, bias_correction=bias_correction, dn=1e5, n_interp=30)
    del da
    interpd.write_parquet(ofile)

for huh in to_do:
    rename, levtype, name, kwargs = huh
    if rename in ["APVO", "CPVO"]:
        for lev in [320, 330, 340, 350, "all"]:
            kwargs_ = kwargs | {"levels": lev}
            if lev == "all":
                rename_ = f"{rename}any"                
            else:
                rename_ = f"{rename}{lev:.0f}"
            _do_one(path, rename_, levtype, name, args, kwargs_)
    else:
        _do_one(path, rename, levtype, name, args, kwargs)
    
    
# ds = xr.open_dataset(f"{DATADIR}/ERA5/thetalev/PV_and_wind/6H/full.zarr", consolidated=False)
# for lev, var in product([320, 330, 340, 350], ["PV"]):
#     key = f"{var}{lev}"
#     ofile = path.joinpath(f"{key}_relative.parquet")
#     if ofile.is_file():
#         continue
#     da = ds[var].sel(lev=lev).rename(key)
#     interpd = create_jet_relative_dataset(jets, da, bias_correction=bias_correction, dn=1e5, n_interp=30)
#     del da
#     interpd.write_parquet(ofile)
    
for n_days in [5, 10, 20, 30]:
    ds_eddies = xr.open_dataset(f"{DATADIR}/ERA5/plev/uv/6H/results/eddy_forcing_{n_days}days.zarr", consolidated=False)
    ds_eddies = ds_eddies.unify_chunks()
    ds_eddies = ds_eddies.sel(lat=slice(None, 85))

    for lev, var in product([300, 250], ["EKE"]):
        key = f"{var}{lev}_{n_days}days"
        ofile = path.joinpath(f"{key}_relative.parquet")
        if ofile.is_file():
            continue
        da = ds_eddies[var].sel(lev=lev).rename(key)
        interpd = create_jet_relative_dataset(jets, da, bias_correction=bias_correction, dn=1e5, n_interp=30)
        del da
        interpd.write_parquet(ofile)
        
        
    to_do = {
        "hor": ("hor1", "hor2"),
        "vert": ("vert1", "vert2"),
    }
    for lev in [300, 250]:
        for dest, sources in to_do.items():
            ofile = path.joinpath(f"{dest}{lev}_{n_days}days_relative.parquet")
            if ofile.is_file():
                continue
            das = [ds_eddies[source].sel(lev=lev) for source in sources]
            interpd = create_jet_relative_dataset(
                jets,
                *das,
                bias_correction=bias_correction,
                align_2d=dest,
                dn=1e5,
                n_interp=30,
            )
            interpd = interpd.drop(*[f"{source}_interp" for source in sources])
            interpd = interpd.rename({f"{dest}_interp": f"{dest}{lev}_{n_days}days_interp"})
            interpd.write_parquet(ofile)
            
    del ds_eddies
    ds_eddies = xr.open_dataset(Basepath.joinpath(f"results/eddy_stuff_{n_days}days.zarr"), consolidated=False)
    ds_eddies = ds_eddies.unify_chunks()
    ds_eddies = ds_eddies.sel(lat=slice(None, 85))

    for lev, var in product([300, 250], ["up", "vp"]):
        key = f"{var}{lev}_{n_days}days"
        ofile = path.joinpath(f"{key}_relative.parquet")
        if ofile.is_file():
            continue
        da = ds_eddies[var].sel(lev=lev).rename(key)
        interpd = create_jet_relative_dataset(jets, da, bias_correction=bias_correction, dn=1e5, n_interp=30)
        del da
        interpd.write_parquet(ofile)
    
    
ds_eady = xr.open_dataset(f"{DATADIR}/ERA5/plev/uv/6H/results/eady_growth.zarr", consolidated=False)
ds_eady = ds_eady.unify_chunks()
ds_eady = ds_eady.sel(lat=slice(None, 85))

var = "eady_s"
ofile = path.joinpath(f"{var}_relative.parquet")
if not ofile.is_file():
    da = ds_eady[var]
    interpd = create_jet_relative_dataset(jets, da, bias_correction=bias_correction, dn=1e5, n_interp=30)
    del da
    interpd.write_parquet(ofile)
    
    
to_do = {
    "eady": ("eady_u", "eady_v"),
}
for dest, sources in to_do.items():
    ofile = path.joinpath(f"{dest}_relative.parquet")
    if ofile.is_file():
        continue
    das = [ds_eady[source] for source in sources]
    interpd = create_jet_relative_dataset(
        jets,
        *das,
        bias_correction=bias_correction,
        align_2d=dest,
        dn=1e5,
        n_interp=30,
    )
    interpd = interpd.drop(*[f"{source}_interp" for source in sources])
    interpd.write_parquet(ofile)