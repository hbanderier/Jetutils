from jetutils.geospatial import detect_contours_lonlat, detect_overturnings, event_props, detect_streamers, to_xarray_sjoin
from jetutils.data import smooth
from tqdm import trange
from jetutils.jet_finding import is_polar_gmix,  to_one_large, track_jets, pers_from_cross, average_jet_categories
from jetutils.definitions import DATADIR, N_WORKERS
from pathlib import Path
import polars as pl
import xarray as xr


for run in ["historical", "ssp370"]:
    path = Path(DATADIR, "CESM2/high_wind", run, "results/1")
    jets = pl.read_parquet(path.joinpath("jets.parquet"))
    jets_newcat = is_polar_gmix(jets, ("s", "theta"), mode="week", n_init=20, init_params="random_from_data", v2=True, use_prev=True)
    jets_newcat = jets_newcat.rename({"is_polar": "is_polar_old", "is_polar_right": "is_polar"})
    
    newpath = Path(DATADIR, "CESM2/high_wind", run, "results/2")
    newpath.mkdir(exist_ok=True)
    jets = jets.drop("diff", "is_polar_old")
    jets.write_parquet(newpath.joinpath("jets.parquet"))
    props = pl.read_parquet(path.joinpath("props.parquet"))
    newcat = jets.group_by("member", "time", "jet ID").agg(pl.col("is_polar").mean())
    props = props.drop("is_polar").join(newcat, on=("member", "time", "jet ID"))
    props.write_parquet(newpath.joinpath("props.parquet"))
    phat_jets = to_one_large(jets, int_EDJ_threshold=1.3e8)
    cross = track_jets(phat_jets)
    cross.write_parquet(newpath.joinpath("cross.parquet"))
    pers = pers_from_cross(cross)    
    phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.3e8))
    phat_props = props.filter(phat_filter)
    phat_props = average_jet_categories(phat_props, polar_cutoff=0.5)
    
    
ds = xr.open_dataset(f"{DATADIR}/ERA5/thetalev/with_EMF.zarr")
ds = ds.chunk(ds["PV"].encoding["preferred_chunks"])
filters = {
    "APVO": pl.col("orientation") == "anticyclonic",
    "CPVO": pl.col("orientation") == "cyclonic",
    "SAPVS": (pl.col("EMF") <= 0.) & (pl.col("PV") >= pl.col("level")),
    "SCPVS": (pl.col("EMF") >= 0.) & (pl.col("PV") >= pl.col("level")),
    "TAPVS": (pl.col("EMF") <= 0.) & (pl.col("PV") <= pl.col("level")),
    "TCPVS": (pl.col("EMF") >= 0.) & (pl.col("PV") <= pl.col("level")),
}
odirs = {var: Path(f"{DATADIR}/ERA5/thetalev/{var}_new") for var in filters}
odir_pfiles = Path(f"{DATADIR}/ERA5/thetalev/rwb_index")
for odir in odirs.values():
    odir.mkdir(exist_ok=True)
for year in trange(1959, 2023):
    ds_ = ds.sel(time=ds.time.dt.year == year).load()
    opaths = {var: path.joinpath(f"{year}.nc") for var, path in odirs.items()}
    if all(opath.is_file() for opath in opaths.values()):
        continue
    
    ofile_contour = odir_pfiles.joinpath(f"pvcontour_{year}.parquet")
    if odir_pfiles.is_file():
        contours = pl.read_parquet(ofile_contour)
    else:        
        pv = smooth(ds_["PV"], {"lon": ("win", 5), "lat": ("win", 5)})
        contours = detect_contours_lonlat(pv, [2], processes=N_WORKERS, ctx="fork", repeat_lons=0, do_round=True)
        contours.write_parquet(ofile_contour)
        
    ofile_overturnings = odir_pfiles.joinpath(f"overturnings_{year}.parquet")
    if ofile_overturnings.is_file():
        overturnings = pl.read_parquet(ofile_overturnings)
        overturnings_on_grid = pl.read_parquet(odir_pfiles.joinpath(f"overturnings_on_grid_{year}.parquet"))
    else:        
        overturnings = detect_overturnings(contours)
        overturnings, overturnings_on_grid = event_props(overturnings, [ds_["PV"], ds_["EMF"]])
        overturnings.write_parquet(ofile_overturnings)
        overturnings_on_grid.write_parquet(odir_pfiles.joinpath(f"overturnings_on_grid_{year}.parquet"))
        
    ofile_streamers = odir_pfiles.joinpath(f"streamers{year}.parquet")
    if ofile_streamers.is_file():
        streamers = pl.read_parquet(ofile_streamers)
        streamers_on_grid = pl.read_parquet(odir_pfiles.joinpath(f"streamers_on_grid_{year}.parquet"))
    else:        
        streamers = detect_streamers(contours, min_ratio=5)
        streamers, streamers_on_grid = event_props(streamers, [ds_["PV"], ds_["EMF"]])
        streamers.write_parquet(ofile_streamers)
        streamers_on_grid.write_parquet(odir_pfiles.joinpath(f"streamers_on_grid_{year}.parquet"))
    
    to_do = {var: overturnings_on_grid for var in ["APVO", "CPVO"]} | {var: streamers_on_grid for var in ["SAPVS", "SCPVS", "TAPVS", "TCPVS"]}
    for var, thing_on_grid in to_do.items():
        opath = opaths[var]
        if opath.is_file():
            continue
        filter_ = filters[var]
        thing_on_grid_ = thing_on_grid.filter(filter_)
        da = to_xarray_sjoin(ds_["PV"], events_on_grid=thing_on_grid_)
        da.to_netcdf(opath)