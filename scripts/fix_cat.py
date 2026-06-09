from jetutils.definitions import DATADIR, N_WORKERS
from jetutils.data import smooth

from jetutils.geospatial import detect_contours_lonlat, detect_overturnings, event_props, detect_streamers, to_xarray_sjoin, calculate_streamer_angle
# from jetutils.jet_finding import is_polar_gmix,  to_one_large, track_jets, pers_from_cross, average_jet_categories
from pathlib import Path
import polars as pl
import xarray as xr
from tqdm import trange

ds = xr.open_dataset(f"{DATADIR}/ERA5/thetalev/with_EMF.zarr")
ds = ds.chunk(ds["PV"].encoding["preferred_chunks"])
strato = pl.col("PV") > pl.col("level")
tropo = pl.col("PV") < pl.col("level")
anti = (pl.col("axis_angle") >= 15) & (pl.col("axis_angle") <= 75)
cyclo = pl.col("axis_angle") >= 105
large = pl.col("area") > 1e11
filters = {
    "APVO": pl.col("orientation") == "anticyclonic",
    "CPVO": pl.col("orientation") == "cyclonic",
    "SAPVS": strato & anti & large,
    "SCPVS": strato & cyclo & large,
    "TAPVS": tropo & anti & large,
    "TCPVS": tropo & cyclo & large,
}
odirs = {var: Path(f"{DATADIR}/ERA5/thetalev/{var}_new") for var in filters}
odir_pfiles = Path(f"{DATADIR}/ERA5/thetalev/rwb_index")
odir_pfiles.mkdir(exist_ok=True)
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
        contours = contours.filter(~(pl.col("lat") > 85).any().over("time", "lev", "contour", "level"))
        contours.write_parquet(ofile_contour)
        
    ofile_overturnings = odir_pfiles.joinpath(f"overturnings_{year}.parquet")
    if ofile_overturnings.is_file():
        overturnings = pl.read_parquet(ofile_overturnings)
        overturnings_on_grid = pl.read_parquet(odir_pfiles.joinpath(f"overturnings_on_grid_{year}.parquet"))
    else:        
        overturnings = detect_overturnings(contours)
        overturnings, overturnings_on_grid = event_props(overturnings, [ds_["PV"], ds_["EMF"]])
        overturnings_props = overturnings["time", "lev", "level", "index", "area", "com_x", "com_y", "PV", "EMF"]
        overturnings_on_grid = overturnings_on_grid.join(overturnings_props, on=["time", "lev", "level", "index"])
        overturnings.write_parquet(ofile_overturnings)
        overturnings_on_grid.write_parquet(odir_pfiles.joinpath(f"overturnings_on_grid_{year}.parquet"))
        
    ofile_streamers = odir_pfiles.joinpath(f"streamers{year}.parquet")
    if ofile_streamers.is_file():
        streamers = pl.read_parquet(ofile_streamers)
        streamers_on_grid = pl.read_parquet(odir_pfiles.joinpath(f"streamers_on_grid_{year}.parquet"))
    else:        
        streamers = detect_streamers(contours, min_ratio=5)
        streamers, streamers_on_grid = event_props(streamers, [ds_["PV"], ds_["EMF"]])
        streamers = calculate_streamer_angle(streamers, "PV")
        streamers_props = streamers["time", "lev", "level", "index", "area", "com_x", "com_y", "PV", "EMF", "axis_angle"]
        streamers_on_grid = streamers_on_grid.join(streamers_props, on=["time", "lev", "level", "index"])
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