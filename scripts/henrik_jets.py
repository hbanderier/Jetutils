import os
from pathlib import Path
from itertools import product

import dask.array as darr
import numpy as np
import polars as pl
import xarray as xr
from jetutils.data import open_da, smooth, extract
from jetutils.definitions import DATADIR, YEARS, N_WORKERS, DEFAULT_VARNAME, OMEGA, C_P_AIR, KAPPA, degsin, compute, get_index_columns, polars_to_xarray
from jetutils.derived_quantities import compute_absolute_vorticity, compute_2d_conv, compute_norm_derivative
from jetutils.geospatial import (
    gather_normal_da_jets,
    interp_jets_to_zero_one,
    detect_contours,
    detect_contours_lonlat,
    detect_overturnings,
    event_props,
    to_xarray_sjoin,
    detect_streamers,
)
from jetutils.jet_finding import (
    DataHandler,
    JetFindingExperiment,
    add_feature_for_cat,
    iterate_over_year_maybe_member,
)
from scipy.signal.windows import lanczos
from tqdm import tqdm, trange

os.environ["RUST_BACKTRACE"] = "full"


def convolve(in1, in2, mode="full", method="fft", axes=None):
    from scipy.signal import fftconvolve, oaconvolve
    from scipy.signal._signaltools import _init_freq_conv_axes

    in1 = darr.asarray(in1)
    in2 = np.asarray(in2)

    # Checking for trivial cases and incorrect flags
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return np.array([])

    if mode != "full" and mode != "same" and mode != "valid" and mode != "periodic":
        raise ValueError(
            "acceptable mode flags are 'valid', 'same', 'full' or 'periodic'"
        )
    if method not in ["fft", "oa"]:
        raise ValueError("acceptable method flags are 'oa', or 'fft'")

    # Pre-formatting or the the inputs, mainly for the `axes` argument
    in1, in2, axes = _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False)

    # _init_freq_conv_axes calls a function that will swap out inputs if required
    # when mode == "valid".We want to avoid having in2 be a dask array thus check
    # to see if the inputs were swapped and raise an error.
    if isinstance(in1, np.ndarray):
        raise ValueError(
            "For 'valid' mode in1 has to be at least as large as in2 in every dimension"
        )

    s1 = in1.shape
    s2 = in2.shape

    # If all axe were removed by the preformatting we only have to rely
    # on multiplication broadcasting rules.
    if not len(axes):
        in_cv = in1 * in2
        # This is the "full" output that is also valid.
        # To get the "same" output we need to center in some dimensions.
        if mode == "same" or mode == "periodic":
            not_axes_but_s1_1 = [
                a
                for a in range(in1.ndim)
                if a not in axes and s1[a] == 1 and s2[a] != 1
            ]
            in_cv = in_cv[
                tuple(
                    (
                        slice((s2[a] - 1) // 2, (s2[a] - 1) // 2 + 1)
                        if a in not_axes_but_s1_1
                        else slice(None, None)
                    )
                    for a in range(in1.ndim)
                )
            ]
            return in_cv

    else:
        # This is kind of a hack but it works.
        not_axes_but_s1_1 = [
            a for a in range(in1.ndim) if a not in axes and s1[a] == 1 and s2[a] != 1
        ]
        if len(not_axes_but_s1_1) and (mode == "full" or mode == "valid"):
            new_shape = tuple(
                s1[i] for i in range(in1.ndim) if i not in not_axes_but_s1_1
            )
            in1 = in1.reshape(new_shape)
            for a in not_axes_but_s1_1:
                in1 = darr.stack([in1] * s2[a], axis=a)
            return convolve(in1, in2, mode=mode, method=method, axes=axes)

        # Deals with the case where there is at least one axis a in which we do not
        # do the convolution that has s2[a] == s1[a] != 1
        not_axes_but_same_shape = [
            a for a in range(in1.ndim) if a not in axes and s1[a] == s2[a] != 1
        ]
        if len(not_axes_but_same_shape):
            to_rechunk = [a for a in not_axes_but_same_shape if len(in1.chunks[a]) != 1]
            new_chunk_size = tuple(
                -1 if a in to_rechunk else "auto" for a in range(in1.ndim)
            )
            in1 = in1.rechunk(new_chunk_size)

        depth = {i: s2[i] // 2 for i in axes}

        # Flags even dimensions and removes them by adding zeros
        # This is done to avoid from having some results show up twice
        # at the edge of blocks
        even_flag = np.r_[[1 - s2[a] % 2 if a in axes else 0 for a in range(in1.ndim)]]
        target_shape = np.asarray(s2)
        target_shape += even_flag

        if any(target_shape != np.asarray(s2)):
            # padding axes where in2 is even
            pad_width = tuple(
                (even_flag[a], 0) if a in axes else (0, 0) for a in range(in1.ndim)
            )
            in2 = darr.pad(in2, pad_width)

        if mode != "valid":
            pad_width = tuple(
                (depth[i] - even_flag[i], depth[i]) if i in axes else (0, 0)
                for i in range(in1.ndim)
            )
            in1 = darr.pad(in1, pad_width)

        if mode == "periodic":
            boundary = "periodic"
        else:
            boundary = 0

        cv_dict = {"oa": oaconvolve, "fft": fftconvolve}

        def cv_func(x):
            return cv_dict[method](x, in2, mode="same", axes=axes)

        complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

        if complex_result:
            dtype = "complex"
        else:
            dtype = "float"

        # Actualy does the convolution with all the parameters preformatted
        in_cv = in1.map_overlap(
            cv_func, depth=depth, boundary=boundary, trim=True, dtype=dtype
        )

        # The output as to be reduced depending on the `mode` argument
        if mode == "valid":
            output_slicing = tuple(
                (
                    slice(depth[i], s1[i] - (depth[i] - even_flag[i]), 1)
                    if i in depth.keys()
                    else slice(0, None)
                )
                for i in range(in1.ndim)
            )
            in_cv = in_cv[output_slicing]

        elif mode != "full":
            # Only have to undo the padding
            output_slicing = tuple(
                slice(p[0], -p[1]) if p != (0, 0) else slice(0, None) for p in pad_width
            )
            in_cv = in_cv[output_slicing]

    return in_cv


def create_jet_relative_dataset(
    jets,
    da,
    bias_correction: pl.DataFrame | None = None,
    half_length: float = 2e6,
    dn: float = 1e5,
    n_interp: int = 30,
    in_meters: bool = True,
):
    indexer = list(iterate_over_year_maybe_member(jets, da))
    to_average = []
    index_columns = get_index_columns(
        jets, 
        (
            "member",
            "time",
            "cluster",
            "spell",
            "relative_index",
            "relative_time",
            "sample_index",
            "inside_index"
        )
    )
    varname = da.name + "_interp"
    for idx1, idx2 in tqdm(indexer, total=len(indexer)):
        jets_ = jets.filter(*idx1)
        da_ = da.sel(**idx2)
        if bias_correction is not None:
            bias_correction_ = bias_correction.filter(*idx1)
            extra_n = bias_correction["n_max"].abs().max()
        else:
            bias_correction_ = None
            extra_n = 0
        try:
            jets_with_interp = gather_normal_da_jets(
                jets_,
                da_,
                half_length=half_length + extra_n,
                dn=dn,
                in_meters=in_meters,
            )
        except (KeyError, ValueError) as e:
            print(e)
            break
        if bias_correction_ is not None:
            jets_with_interp = (
                jets_with_interp
                .join(bias_correction_, on=[*index_columns, "jet ID", "index"])
                .with_columns(n=pl.col("n") - pl.col("n_max"))
                .filter(pl.col("n").abs() <= 2e6)
                .with_columns(side=pl.col("n").sign().cast(pl.Int32()))
                .drop("n_max")
            )
            jets_with_interp = jets_with_interp.filter(pl.col("n").abs() <= 2e6)
        jets_with_interp = interp_jets_to_zero_one(
            jets_with_interp, [varname, "is_polar"], n_interp=n_interp
        )
        to_average.append(jets_with_interp)
    return pl.concat(to_average)


def create_bias_correction(
    jets,
    ds,
    period: int = 15
):
    offset = int(np.ceil(period / 2))
    indexer = list(iterate_over_year_maybe_member(jets, ds))
    to_average = []
    index_columns = get_index_columns(
        jets, 
        (
            "member",
            "time",
            "cluster",
            "spell",
            "relative_index",
            "relative_time",
            "sample_index",
            "inside_index"
        )
    )
    for idx1, idx2 in tqdm(indexer, total=len(indexer)):
        jets_ = jets.filter(*idx1)
        ds_ = ds.sel(**idx2)
        sigma = compute_norm_derivative(ds_, "s").rename("sigma")
        interpd = gather_normal_da_jets(
            jets_, sigma, dn=1e5, half_length=4e5, in_meters=True
        )
        interpd = interp_jets_to_zero_one(
            interpd, ["sigma_interp", "is_polar"], n_interp=30
        )
        mapper = interpd["norm_index"].unique().sort().to_frame().with_row_index("indexer")
        bias_correction = (
            interpd[*index_columns, "jet ID", "norm_index", "n", "sigma_interp"]
            .join(mapper, on="norm_index")
            .rolling("indexer", period=f"{period}i", offset=f"-{offset}i", group_by=(*index_columns, "jet ID", "n"))
            .agg(pl.col("sigma_interp").mean())
            .join(mapper, on="indexer")
            .drop("indexer")
        )
        bias_correction = polars_to_xarray(bias_correction, ["time", "jet ID", "n", "norm_index"])
        bias_correction = detect_contours(bias_correction, [0.], ("norm_index", "n"))
        bias_correction = bias_correction.filter((pl.lit(0.).is_in(pl.col("norm_index").implode()) & pl.lit(1.).is_in(pl.col("norm_index").implode())).over("time", "jet ID", "contour"))
        bias_correction.drop("level", "cyclic").group_by("time", "jet ID", "norm_index").agg(pl.col("n").get(pl.col("n").abs().arg_min()))
        to_average.append(bias_correction)
        return bias_correction
    return pl.concat(to_average)


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
            (convolve(ds[var].data, l_win)[half_len:-half_len] / l_win.sum()).astype(
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
for run in ["ctrl", "dobl", "ctrl_p4"]:
    ipath = Path(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results",
        "Eddy_NH_10days.zarr",
    )
    odir = Path(f"{DATADIR}/Henrik_data/{run}/EPF/6H")
    odir.mkdir(parents=True, exist_ok=True)
    bigds = xr.open_dataset(ipath).sel(lev=[20000, 30000]).chunk("auto")
    for year in trange(1969, 2021):
        ofile = odir.joinpath(f"{year}.nc")
        if ofile.is_file():
            continue
        ds = bigds.sel(time=bigds.time.dt.year == year)
        other = xr.open_dataset(f"{DATADIR}/Henrik_data/{run}/vertical/6H/{year}.nc").sel(lat=slice(0, None))
        ds = xr.merge([ds, other])
        ds["u"] = xr.open_dataset(f"{DATADIR}/Henrik_data/{run}/high_wind/6H/{year}.nc")["u"].sel(lat=slice(0, None))
        
        gamma = (-KAPPA / ds.lev * (100000 / ds.lev) ** KAPPA * ds["dthetadp"].mean(["time", "lon", "lat"])).astype(np.float32)
        EAPE = (C_P_AIR * 0.5 * (ds.lev * 1e-5) ** (2 * KAPPA) * gamma * ds["thetap"] ** 2).astype(np.float32)
        S = (0.5 * (ds["up"] ** 2 + ds["vp"] ** 2 - EAPE)).astype(np.float32)
        f = (2 * OMEGA * degsin(ds.lat)).astype(np.float32)

        ## Base 2 * 3
        ds["F11"] = ds["up"] ** 2 - S
        ds["F12"] = ds["up"] * ds["vp"]
        ds["F13"] = - ds["vp"] * ds["thetap"] * f / ds["dthetadp"]
        ds["F21"] = ds["up"] * ds["vp"]
        ds["F22"] = ds["vp"] ** 2 - S
        ds["F23"] = ds["up"] * ds["thetap"] * f / ds["dthetadp"]

        ## Additional from original EP:
        ds["F12_extra"] = - ds["dudp"] * ds["vp"] * ds["thetap"] / ds["dthetadp"]
        ds["F13_extra"] = ds["up"] * ds["omegap"]
        ds["F23_extra"] = ds["vp"] * ds["omegap"]
        ds = ds.drop_vars([var for var in list(ds.data_vars) if var[0] != "F"])
        ds = compute(ds, progress_flag=False)
        ds.to_netcdf(ofile)
    

# block 4: WB
levels = compute(xr.open_mfdataset(list(Path(DATADIR, "Henrik_data/ctrl/zeta/6H").glob("*0.nc")))[DEFAULT_VARNAME].quantile([0.5]), progress_flag=True).values
levels = (levels * 1e5).round(1).tolist()
for run in ["ctrl", "dobl", "ctrl_p4"]:
    basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
    da_mflux = xr.open_dataset(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
    ).sel(lev=30000)
    da_mflux = (da_mflux["up"] * da_mflux["vp"]).rename("EMF")
    opath = Path(DATADIR, "Henrik_data", run) 
    opath_rwb = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
    opath_rwb.mkdir(exist_ok=True)
    for year in trange(1969, 2021):
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

filters_type = {
    "stratospheric": pl.col("zeta") >= pl.col("level"),
    "tropospheric": pl.col("zeta") < pl.col("level")
}
filters_orientation = {
    "anticyclonic": pl.col("mflux") <= 0.,
    "cyclonic": pl.col("mflux") > 0.
}
for run in ["ctrl", "dobl", "ctrl_p4"]:
    basepath_zeta = Path(DATADIR, f"Henrik_data/{run}/zeta/6H")
    da_mflux = xr.open_dataset(
        f"{DATADIR}/Henrik_data/{run}/high_wind/6H/results/Eddy_NH_10days.zarr"
    ).sel(lev=30000)
    da_mflux = (da_mflux["up"] * da_mflux["vp"]).rename("EMF")
    opath = Path(DATADIR, "Henrik_data", run) 
    opath_rwb = Path(DATADIR, f"Henrik_data/{run}/rwb_index")
    opath_rwb.mkdir(exist_ok=True)
    for year in trange(1969, 2021):
        ofile = opath_rwb.joinpath(f"streamers_{year}.parquet")
        
        if opath.joinpath(f"TCAVS/6H/{year}.nc").is_file() and ofile.is_file():
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
        # mflux = smooth(mflux, {"lon": ("win", 3), "lat": ("win", 3)})
        mflux = compute(mflux)
        if ofile.is_file():
            streamers = pl.read_parquet(ofile)
            streamers_on_grid = None
        else:
            contours = detect_contours_lonlat(zeta, levels, processes=N_WORKERS, ctx="fork")
            streamers = detect_streamers(contours)
            streamers, streamers_on_grid = event_props(streamers, [zeta, mflux])
            streamers.write_parquet(ofile)
            
        
        for type_, orientation in product(["stratospheric", "tropospheric"], ["cyclonic", "anticyclonic"]):
            name = f"{type_[0].upper()}{orientation[0].upper()}AVS"
            odir = opath.joinpath(f"{name}/6H")
            odir.mkdir(parents=True, exist_ok=True)
            ofile = odir.joinpath(f"{year}.nc")
            if ofile.is_file():
                continue
            f1 = filters_type[type_]
            f2 = filters_orientation[orientation]
            df = streamers.filter(f1, f2)
            da = to_xarray_sjoin(zeta, events=df)
            da.to_netcdf(ofile)
            
            odir = opath.joinpath(f"{name}/dailyany")
            odir.mkdir(parents=True, exist_ok=True)
            da = da.any("level").resample(time="1D").any().astype(np.uint8)
            da.to_netcdf(odir.joinpath(f"{year}.nc"))
            
# block 5: define jets (already computed probably)

both_jets = {}
both_paths = {}
for run in ["ctrl", "dobl", "ctrl_p4"]:
    dh = DataHandler.from_specs("Henrik_data", run, ("high_wind", ["u", "v", "s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=20000)
    exp = JetFindingExperiment(dh)
    all_jets_one_df = exp.find_jets(force=False, n_coarsen=1, smooth_s=5, alignment_thresh=0.55, base_s_thresh=0.55, int_thresh_factor=0.2, hole_size=5)
    theta300 = open_da("Henrik_data", run, ("high_wind", ["s", "theta"]), "6H", minlon=-80, maxlon=40, minlat=15, maxlat=80, levels=30000).rename({"s": "s300", "theta": "theta300"})
    all_jets_one_df = add_feature_for_cat(all_jets_one_df, "s300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
    all_jets_one_df = add_feature_for_cat(all_jets_one_df, "theta300", theta300, ofile_ajdf=exp.path.joinpath("all_jets_one_df.parquet"), force=False)
    all_jets_one_df = exp.categorize_jets(None, ["s300", "theta300"], force=False, n_init=15, init_params="k-means++", mode="month").cast({"time": pl.Datetime("ms")})
    phat_filter = (pl.col("is_polar") < 0.5) | ((pl.col("is_polar") > 0.5) & (pl.col("int") > 1.e8))
    phat_jets = all_jets_one_df.filter((pl.col("is_polar").mean().over(["time", "jet ID"]) < 0.5) | ((pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5) & (pl.col("int").mode().first().over(["time", "jet ID"]) > 1.e8)))
    both_paths[run] = exp.path
    both_jets[run] = phat_jets.with_columns(**{"jet ID": (pl.col("is_polar").mean().over(["time", "jet ID"]) > 0.5).cast(pl.UInt32())})

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

for run in ["ctrl", "dobl", "ctrl_p4"]:
    jets = both_jets[run]
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
        interpd = create_jet_relative_dataset(jets, da_)
        del da_
        interpd.write_parquet(ofile)
    


for run in ["ctrl", "dobl", "ctrl_p4"]:
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
        if not opaths["vert"].is_file():
            ds["vert1"] = ds["F13"].differentiate("lev")
            ds["vert2"] = ds["F23"].differentiate("lev")
        if not opaths["vert_extra"].is_file():
            ds["vert_extra1"] = ds["F13_extra"].differentiate("lev")
            ds["vert_extra2"] = ds["F23_extra"].differentiate("lev")
        if not opaths["hor"].is_file():
            ds = ds.sel(lev=30000)
            ds["hor1"] = compute_2d_conv(ds, "F11", "F12")
            ds["hor2"] = compute_2d_conv(ds, "F21", "F22")
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
            