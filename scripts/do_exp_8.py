import numpy as np
import xarray as xr
import polars as pl
from jetutils.definitions import compute
from jetutils.data import DataHandler, open_da
from jetutils.geospatial import create_jet_relative_dataset
from jetutils.jet_finding import JetFindingExperiment, average_jet_categories
from jetutils.derived_quantities import compute_emf_2d_conv, compute_relative_vorticity

all_times = (
    pl.datetime_range(
        start=pl.datetime(1959, 1, 1),
        end=pl.datetime(2023, 1, 1),
        closed="left",
        interval="6h",
        eager=True,
        time_unit="ms",
    )
    .rename("time")
    .to_frame()
)
summer_filter = (
    all_times
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
    .filter(pl.col("time").dt.ordinal_day() > 166)
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)
big_summer = all_times.filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
big_summer_daily = big_summer.filter(big_summer["time"].dt.hour() == 0)

dh = DataHandler("/storage/workspaces/giub_meteo_impacts/ci01/ERA5/plev/high_wind/6H/results/8")
exp = JetFindingExperiment(dh)
all_jets_one_df = exp.find_jets(force=False, base_s_thresh=0.55, hole_size=6)
all_jets_one_df = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=10, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})
props_uncat = exp.props_as_df(False).cast({"time": pl.Datetime("ms")})

props_as_df = average_jet_categories(props_uncat, polar_cutoff=0.5)

props_summer = summer_filter.join(props_as_df, on="time")
phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.3e8))

phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.3e8)))
phat_jets_catd = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)

phat_props_catd = phat_props_catd.join(phat_props_catd.rolling("time", period="2d", group_by="jet").agg(**{f"{col}_var": pl.col(col).var() for col in ["mean_lon", "mean_lat", "mean_s", "s_star"]}), on=["time", "jet"])

phat_props_catd_summer = summer_filter.join(phat_props_catd, on="time")

# args = ["all", None, -100, 60, 0, 90]
# da_T = open_da("ERA5", "surf", "t2m", "dailymean", *args)
# da_T = compute(da_T)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_T, suffix="_meters")
# del da_T

# da_tp = open_da("ERA5", "surf", "tp", "dailysum", *args)
# da_tp = compute(da_tp)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_tp, suffix="_meters")
# del da_tp

# da_pv = open_da("ERA5", "thetalev", "PV330", "dailymean", *args)
# da_pv = compute(da_pv).rename("PV330")
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_pv, suffix="_meters")
# del da_pv

# da_pv = open_da("ERA5", "thetalev", "PV350", "dailymean", *args)
# da_pv = compute(da_pv)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_pv, suffix="_meters")
# del da_pv

# varnames_rwb = ["APVO", "CPVO"]
# for var in varnames_rwb:
#     da_rwb = open_da("ERA5", "thetalev", var, "dailyany", *args)
#     da_rwb = compute(da_rwb)
#     create_jet_relative_dataset(phat_jets_catd, exp.path, da_rwb, suffix="_meters")
#     del da_rwb

# da_theta2pvu = open_da("ERA5", "surf", ("alot2pvu", "theta"), "dailymean", *args)
# da_theta2pvu = compute(da_theta2pvu)
# create_jet_relative_dataset(phat_jets_catd, exp.path, da_theta2pvu, suffix="_meters")
# del da_theta2pvu

# ds = xr.open_zarr("/storage/workspaces/giub_meteo_impacts/ci01/ERA5/plev/uv/6H/results/Eddy_uv_natl_10days.zarr")

# up = ds["up"].sel(lev=250).rename("up250")
# up = up.resample(time="1d").mean().astype(np.float32)
# up = compute(up, progress_flag=True)
# create_jet_relative_dataset(phat_jets_catd, exp.path, up, suffix="_meters")
# del up

# vp = ds["vp"].sel(lev=250).rename("vp250")
# vp = compute(vp, progress_flag=True)
# vp = vp.resample(time="1d").mean()
# create_jet_relative_dataset(phat_jets_catd, exp.path, vp, suffix="_meters")
# del vp

# EMFconv = compute_emf_2d_conv(ds.sel(lev=250)).rename("EMFconv250").astype(np.float32)
# EMFconv = compute(EMFconv, progress_flag=True)
# EMFconv = EMFconv.resample(time="1d").mean().astype(np.float32)
# create_jet_relative_dataset(phat_jets_catd, exp.path, EMFconv, suffix="_meters")
# del EMFconv

# EKE = (ds.sel(lev=250)["up"] ** 2 + ds.sel(lev=250)["vp"] ** 2) * 0.5
# EKE = compute(EKE, progress_flag=True).rename("EKE250")
# EKE = EKE.resample(time="1d").mean().astype(np.float32)
# create_jet_relative_dataset(phat_jets_catd, exp.path, EKE, suffix="_meters")
# del EKE
# del ds

ds = exp.ds

u = ds["u"]
u = compute(u, progress_flag=True)
u = u.resample(time="1d").mean().astype(np.float32)
create_jet_relative_dataset(phat_jets_catd, exp.path, u, suffix="_meters")
del u

v = ds["v"]
v = compute(v, progress_flag=True)
v = v.resample(time="1d").mean().astype(np.float32)
create_jet_relative_dataset(phat_jets_catd, exp.path, v, suffix="_meters")
del v

s = ds["s"]
s = compute(s, progress_flag=True)
s = s.resample(time="1d").mean().astype(np.float32)
create_jet_relative_dataset(phat_jets_catd, exp.path, s, suffix="_meters")
del s

vort = compute_relative_vorticity(ds).rename("vort").astype(np.float32)
vort = compute(vort, progress_flag=True)
vort = vort.resample(time="1d").mean().astype(np.float32)
create_jet_relative_dataset(phat_jets_catd, exp.path, vort, suffix="_meters")
del vort
del ds