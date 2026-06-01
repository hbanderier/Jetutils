import numpy as np
import xarray as xr
import polars as pl
from jetutils.data import standardize, DataHandler
from jetutils.jet_finding import JetFindingExperiment, track_jets, pers_from_cross_catd
from jetutils.definitions import KAPPA

both_jets = {}
both_paths = {}
for run in ["historical", "ssp370"]:
    ds = xr.open_dataset(f"/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/{run}/ds.zarr", engine="zarr")
    ds = standardize(ds)
    ds["theta"] = (ds["t"] * (1000 / ds["lev"]) ** KAPPA).astype(np.float32)
    ds = ds.drop_vars("t")
    dh = DataHandler.from_basepath_and_da(f"/storage/workspaces/giub_meteo_impacts/ci01/CESM2/high_wind/{run}/results", ds)
    exp = JetFindingExperiment(dh)
    all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
    if "is_polar" not in all_jets_one_df.columns:
        all_jets_one_df = exp.categorize_jets(None, ["s", "theta"], force=False, n_init=15, init_params="k-means++", mode="week").cast({"time": pl.Datetime("ms")})
    exp.props_as_df()
    
    phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["member", "time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["member", "time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["member", "time", "jet ID"]) > 1.3e8)))

    phat_jets_catd = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["member", "time", "jet ID"]) > 0.5).cast(pl.UInt32())})
    
    cross_catd_ofile = exp.path.joinpath("cross_catd.parquet")
    if cross_catd_ofile.is_file():
        cross_catd = pl.read_parquet(cross_catd_ofile)
    else:
        cross_catd = track_jets(phat_jets_catd)
        cross_catd = pers_from_cross_catd(cross_catd)
        cross_catd.write_parquet(cross_catd_ofile)