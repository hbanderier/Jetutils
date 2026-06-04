from functools import partial
import os
from pathlib import Path
from itertools import product

import dask.array as darr
import numpy as np
import polars as pl
import xarray as xr
from jetutils.data import open_da, smooth, extract, standardize
from jetutils.definitions import RADIUS, DATADIR, YEARS, N_WORKERS, DEFAULT_VARNAME, OMEGA, C_P_AIR, KAPPA, degsin, compute, get_index_columns, polars_to_xarray, degcos
from jetutils.derived_quantities import compute_absolute_vorticity, compute_2d_conv, compute_norm_derivative, convolve_dask
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

# block 1: compute eddy stuff
opath = Path(
    f"{DATADIR}/ERA5/plev/uv/6H/results",
    "Eddy_uv_natl_10days.zarr",
)
if not opath.is_dir() and False:
    half_len = 20
    ds = standardize(
        xr.open_mfdataset(
            f"{DATADIR}/ERA5/plev/uv/6H/*.nc",
            combine="nested",
            concat_dim="time",
        )[["u", "v", "omega", "theta"]]
        .sel(lat=slice(0, 90), lev=250)
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
ipath = Path(
    f"{DATADIR}/ERA5/plev/uv/6H/results",
    "Eddy_uv_natl_10days.zarr",
)
odir = Path(f"{DATADIR}/ERA5/plev/eddy_stuff/6H")
odir.mkdir(parents=True, exist_ok=True)
bigds = xr.open_dataset(ipath).chunk("auto")
for year in tqdm(YEARS):
    ofile = odir.joinpath(f"{year}.nc")
    if ofile.is_file() or True:
        continue
    ds = bigds.sel(time=bigds.time.dt.year == year)
    # ds["u"] = xr.open_dataset(f"{DATADIR}/Henrik_data/{run}/high_wind/6H/{year}.nc")["u"].sel(lat=slice(0, None))
    
    # gamma = (-KAPPA / ds.lev * (100000 / ds.lev) ** KAPPA * ds["dthetadp"].mean(["time", "lon", "lat"])).astype(np.float32)
    # EAPE = (C_P_AIR * 0.5 * (ds.lev * 1e-5) ** (2 * KAPPA) * gamma * ds["thetap"] ** 2).astype(np.float32)
    # S = (0.5 * (ds["up"] ** 2 + ds["vp"] ** 2 - EAPE)).astype(np.float32)
    EKE = 0.5 * (ds["up"] ** 2 + ds["vp"] ** 2)
    EKE = EKE.astype(np.float32)
    f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)

    ## Base 2 * 3
    ds["EKE"] = EKE
    ds["F11"] = ds["up"] ** 2 - EKE
    ds["F12"] = ds["up"] * ds["vp"]
    # ds["F13"] = - ds["vp"] * ds["thetap"] * f / ds["dthetadp"]
    # ds["F21"] = ds["up"] * ds["vp"]
    ds["F22"] = ds["vp"] ** 2 - EKE
    # ds["F23"] = ds["up"] * ds["thetap"] * f / ds["dthetadp"]

    ## Additional from original EP:
    # ds["F12_extra"] = - ds["dudp"] * ds["vp"] * ds["thetap"] / ds["dthetadp"]
    # ds["F13_extra"] = ds["up"] * ds["omegap"]
    # ds["F23_extra"] = ds["vp"] * ds["omegap"]
    ds = ds.drop_vars([var for var in list(ds.data_vars) if var[0] not in ["E", "F"]])
    ds = compute(ds, progress_flag=False)
    ds.to_netcdf(ofile)
    
    
odir = Path(f"{DATADIR}/ERA5/plev/eddy_stuff/6H")
odir.mkdir(parents=True, exist_ok=True)
ds_eddies = standardize(xr.open_dataset(Path(DATADIR, "ERA5/plev/uv/6H/results/Eddy_uv_natl_10days.zarr"))).sel(lev=250)
if not odir.joinpath("full.zarr").is_dir():
    for i, year in enumerate(tqdm(YEARS)):
        bigds = ds_eddies.sel(time=ds_eddies.time.dt.year == year)
        ds = {}
        ds["EKE"] = 0.5 * (bigds["up"] ** 2 + bigds["vp"] ** 2)
        ds["F11"] = bigds["up"] ** 2 - ds["EKE"]
        ds["F12"] = bigds["up"] * bigds["vp"]
        ds["F22"] = bigds["vp"] ** 2 - ds["EKE"]
        ds = xr.Dataset(ds)
        for f in ["F11", "F12", "F22"]:
            ds[f] = ds[f] * RADIUS * degcos(ds.lat)
        ds["hor1"] = compute_2d_conv(ds, "F11", "F12")
        ds["hor2"] = compute_2d_conv(ds, "F12", "F22")
        ds = compute(ds, progress_flag=False)
        kwargs = {"mode": "w"} if i == 0 else {"mode": "a", "append_dim": "time"}
        ds.to_zarr(odir.joinpath("full.zarr"), **kwargs)
    

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
for year in tqdm(YEARS):
    if True:
        continue
    ofile = opath_rwb.joinpath(f"overturnings_{year}.parquet")
    
    if opath.joinpath(f"CAVO/6H/{year}.nc").is_file() and ofile.is_file():
        continue
        
    zeta = xr.open_dataarray(basepath_zeta.joinpath(f"{year}.nc")).sel(lev=30000)
    zeta = zeta.rename("zeta") * 1e5
    for potential in ["lev", "loni", "lati"]:
        try:
            zeta = zeta.reset_coords(potential, drop=True)
        except ValueError:
            continue
    mflux = da_mflux.sel(time=zeta.time)
    mflux = mflux.rename("mflux").reset_coords("lev", drop=True)
    zeta = smooth(zeta, {"lon": ("win", 5), "lat": ("win", 5)})
    zeta = compute(zeta)
    # mflux = smooth(mflux, {"lon": ("win", 5), "lat": ("win", 5)})
    mflux = compute(mflux)
    if ofile.is_file():
        overturnings = pl.read_parquet(ofile)
        overturnings_on_grid = None
    else:
        contours = detect_contours_lonlat(zeta, levels, processes=N_WORKERS, ctx="fork")
        overturnings = detect_overturnings(contours, max_difflon=3)
        overturnings, overturnings_on_grid = event_props(overturnings, [zeta, mflux])
        overturnings.write_parquet(ofile)
        
    for orientation in ["cyclonic", "anticyclonic"]:
        name = f"{orientation[0].upper()}AVO"
        odir = opath.joinpath(f"{name}/6H")
        odir.mkdir(parents=True, exist_ok=True)
        ofile = odir.joinpath(f"{year}.nc")
        if ofile.is_file():
            continue
        df = overturnings.filter(pl.col("orientation") == orientation)
        da = to_xarray_sjoin(zeta, events=df)
        da.to_netcdf(ofile)
        
        odir = opath.joinpath(f"{name}/dailyany")
        odir.mkdir(parents=True, exist_ok=True)
        da = da.any("level").resample(time="1D").any().astype(np.uint8)
        da.to_netcdf(odir.joinpath(f"{year}.nc"))
            
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

path = Path(DATADIR, "ERA5/plev/high_wind/6H/results/10")
ds = xr.open_dataset(path.joinpath("da.nc"))
ds = standardize(ds)
ds = extract(
    ds, minlon=-80, maxlon=40, minlat=15, maxlat=80
)
times = ds["time"].values

kwargs = dict(
    n_coarsen=3,
    base_s_thresh=0.55,
    alignment_thresh=0.6,
    int_thresh_factor=0.6,
    hole_size=6,
    smooth_func=partial(gaussian_smooth_func, sigma_lon=2, sigma_lat=0.8),
)
jets, ph_jets, props, props_full = do_everything(ds, path, **kwargs)

# stage 6: Interpolate new fields
args = ["all", None, -100, 60, 0, 88]

to_do = (
    ("t2m", "surf", "t2m", {}),
    ("tp", "surf", "tp", {}),
    ("PV330", "thetalev", "PV330", {}),
    ("PV350", "thetalev", "PV350", {}),
    ("APVO", "thetalev", "APVO", {}),
    ("CPVO", "thetalev", "CPVO", {}),
    ("SAPVS", "thetalev", "SAPVS", {}),
    ("TAPVS", "thetalev", "TAPVS", {}),
    ("SCPVS", "thetalev", "SCPVS", {}),
    ("TCPVS", "thetalev", "TCPVS", {}),
    ("theta", "surf", ("alot2pvu", "theta"), {}),
    ("EKE250", "plev", ("eddy_stuff", "EKE"), {}),
)

bias_correction = pl.read_parquet(path.joinpath("bias_correct.parquet"))

for huh in to_do:
    rename, levtype, name, kwargs = huh
    ofile = path.joinpath(f"{rename}_relative.parquet")
    if ofile.is_file():
        continue
    da = open_da("ERA5", levtype, name, "6H", *args, **kwargs).rename(rename)
    if rename in ["APVO", "CPVO", "SAPVS", "TAPVS", "SCPVS", "TCPVS"]:
        da = da.sel(lev=slice(340, 350)).any("lev")
    interpd = create_jet_relative_dataset(ph_jets, da, bias_correction=bias_correction, dn=1e5, n_interp=30)
    del da
    interpd.write_parquet(ofile)

ds_eddies = xr.open_dataset(f"{DATADIR}/ERA5/plev/eddy_stuff/6H/full.zarr")
ds_eddies = ds_eddies.sel(lat=slice(None, 85))
to_do = {
    # "F1": ("F11", "F12"),
    # "F2": ("F12", "F22"),
    "hor": ("hor1", "hor2"),
}
for dest, sources in to_do.items():
    ofile = path.joinpath(f"{dest}_relative.parquet")
    if ofile.is_file():
        continue
    das = [ds_eddies[source] for source in sources]
    interpd = create_jet_relative_dataset(ph_jets, *das, bias_correction=bias_correction, align_2d=dest, dn=1e5, n_interp=30)
    interpd.write_parquet(ofile)