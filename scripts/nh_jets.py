from tqdm import tqdm
import polars as pl
from jetutils.definitions import YEARS, iterate_over_year_maybe_member, compute
from jetutils.data import DataHandler
from jetutils.jet_finding import JetFindingExperiment, gather_normal_da_jets, open_da, add_feature_for_cat, average_jet_categories
from jetutils.geospatial import interp_jets_to_zero_one
from jetutils.derived_quantities import compute_relative_vorticity


def create_jet_relative_dataset(jets, path, da, suffix="", half_length: float = 2e6, dn: float = 1e5, n_interp: int = 30, in_meters: bool = True):
    indexer = iterate_over_year_maybe_member(jets, da)
    to_average = []
    varname = da.name + "_interp"
    for idx1, idx2 in tqdm(indexer, total=len(YEARS)):
        jets_ = jets.filter(*idx1)
        da_ = da.sel(**idx2)
        try:
            jets_with_interp = gather_normal_da_jets(jets_, da_, half_length=half_length, dn=dn, in_meters=in_meters)
        except (KeyError, ValueError) as e:
            print(e)
            break
        jets_with_interp = interp_jets_to_zero_one(jets_with_interp, [varname, "is_polar"], n_interp=n_interp)
        # jets_with_interp = jets_with_interp.group_by("time", pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5, "norm_index", "n", maintain_order=True).agg(pl.col(varname).mean())
        to_average.append(jets_with_interp)
    pl.concat(to_average).write_parquet(path.joinpath(f"{da.name}{suffix}_relative.parquet"))

    
run = "ctrl"

dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=20000)
exp = JetFindingExperiment(dh)
basepath_ctrl = exp.path
all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "s300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "theta300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = exp.categorize_jets(None, ["s300", "theta300"], force=False, n_init=15, init_params="k-means++", mode="month").cast({"time": pl.Datetime("ms")})
props_uncat = exp.props_as_df(categorize=False).cast({"time": pl.Datetime("ms")})

all_times = props_uncat["time"].unique().sort().to_frame()
summer_filter = (
    all_times
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
    .filter(pl.col("time").dt.ordinal_day() > 166)
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)
big_summer = all_times.filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
big_summer_daily = big_summer.filter(big_summer["time"].dt.hour() == 0)

phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))
phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))

jets_ctrl = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)
phat_props_catd = phat_props_catd.join(phat_props_catd.rolling("time", period="2d", group_by="jet").agg(**{f"{col}_var": pl.col(col).var() for col in ["mean_lon", "mean_lat", "mean_s", "s_star"]}), on=["time", "jet"])
props_ctrl = phat_props_catd.clone()
props_summer_ctrl = summer_filter.join(props_ctrl, on="time")

args = ["all", None, -100, 60, 0, 90]
path = exp.path

da = open_da("Henrik_data", run, ("high_wind", "PTTEND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, ("high_wind", "DTCOND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, ("high_wind", "theta"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, "t_low", "6H", *args, levels=100000)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, "z", "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = compute_relative_vorticity(exp.ds).rename("vort")
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = exp.ds["s"].rename("s")
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

run = "dobl"

dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=20000)
exp = JetFindingExperiment(dh)
basepath_dobl = exp.path
all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "s300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = add_feature_for_cat(all_jets_one_df, "theta300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
all_jets_one_df = exp.categorize_jets(None, ["s300", "theta300"], force=False, n_init=15, init_params="k-means++", mode="month").cast({"time": pl.Datetime("ms")})
props_uncat = exp.props_as_df(categorize=False).cast({"time": pl.Datetime("ms")})

all_times = props_uncat["time"].unique().sort().to_frame()
summer_filter = (
    all_times
    .filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
    .filter(pl.col("time").dt.ordinal_day() > 166)
)
summer = summer_filter["time"]
summer_daily = summer.filter(summer.dt.hour() == 0)
big_summer = all_times.filter(pl.col("time").dt.month().is_in([6, 7, 8, 9]))
big_summer_daily = big_summer.filter(big_summer["time"].dt.hour() == 0)

phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))
phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))

jets_dobl = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})
phat_props = props_uncat.filter(phat_filter)
phat_props_catd = average_jet_categories(phat_props, polar_cutoff=0.5)
phat_props_catd = phat_props_catd.join(phat_props_catd.rolling("time", period="2d", group_by="jet").agg(**{f"{col}_var": pl.col(col).var() for col in ["mean_lon", "mean_lat", "mean_s", "s_star"]}), on=["time", "jet"])
props_dobl = phat_props_catd.clone()
props_summer_dobl = summer_filter.join(props_dobl, on="time")

args = ["all", None, -100, 60, 0, 90]
path = exp.path

da = open_da("Henrik_data", run, ("high_wind", "PTTEND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, ("high_wind", "DTCOND"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, ("high_wind", "theta"), "6H", *args, levels=30000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, "t_low", "6H", *args, levels=100000)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da, suffix="_meters")
del da

da = open_da("Henrik_data", run, "z", "6H", *args)
da = compute(da)
create_jet_relative_dataset(jets_dobl, path, da, suffix="_meters")
del da

da = compute_relative_vorticity(exp.ds).rename("vort")
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da

da = exp.ds["s"].rename("s")
da = compute(da)
create_jet_relative_dataset(jets_ctrl, path, da, suffix="_meters")
del da