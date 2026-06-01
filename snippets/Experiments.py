DATERANGEPL = pd.date_range("19590101", "20211231")
YEARSPL = np.unique(DATERANGEPL.year)
DATERANGEPL_SUMMER = DATERANGEPL[np.isin(DATERANGEPL.month, [6, 7, 8])]

DATERANGEPL_EXT = pd.date_range("19400101", "20221231")
YEARSPL_EXT = np.unique(DATERANGEPL_EXT.year)
DATERANGEPL_EXT_SUMMER = DATERANGEPL_EXT[np.isin(DATERANGEPL_EXT.month, [6, 7, 8])]

DATERANGEPL_EXT_6H = pd.date_range("19400101", "20230101", freq="6h", inclusive="left")
DATERANGEPL_EXT_6H_SUMMER = DATERANGEPL_EXT_6H[np.isin(DATERANGEPL_EXT_6H.month, [6, 7, 8])]

DATERANGEML = pd.date_range("19770101", "20211231")

WINDBINS = np.arange(0, 25, 0.5)
LATBINS = np.arange(15, 75.1, 0.5)
LONBINS = np.arange(-90, 30, 1)
DEPBINS = np.arange(-25, 25.1, 0.5)

REGIONS = ["S-W", "West", "S-E", "North", "East", "N-E"]

class DataHandler(object):
    def __init__(
        self,
        dataset: str,
        varname: str,
        levels: int | str | list | tuple,
        resolution: str,
        region: Optional[str] = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        clim_type: str = None,
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
    ) -> None:
        self.dataset = dataset
        self.varname = varname
        if isinstance(levels, int | str | tuple):
            levels = [levels]
        to_sort = []
        for level in levels:
            to_sort.append(float(level) if isinstance(level, int | str) else level[0])
        self.levels = [levels[i] for i in np.argsort(to_sort)]
        self.level_names = []
        for level in self.levels:
            if isinstance(level, tuple | list):
                self.level_names.append(f"{tuple[0]}-{len(tuple)}-{tuple[-1]}")
            else:
                self.level_names.append(str(level))
        levels_str = "_".join(self.level_names)

        self.resolution = resolution
        if region is None:
            if np.any([bound is None for bound in [minlon, maxlon, minlat, maxlat]]):
                raise ValueError(
                    "Specify a region with either a string or 4 ints / floats"
                )
            self.region = f"box_{int(minlon)}_{int(maxlon)}_{int(minlat)}_{int(maxlat)}"
        else:
            self.region = region
        self.minlon, self.maxlon, self.minlat, self.maxlat = [
            int(bound) for bound in self.region.split("_")[-4:]
        ]
        if clim_smoothing is not None and clim_type is None:
            raise ValueError(
                "clim_smoothing is specified so you should specify and clim_type"
            )

        if clim_type is None:
            clim_type = "none"

        if clim_smoothing is None:
            clim_smoothing = {}

        if smoothing is None:
            smoothing = {}

        self.clim_type = clim_type
        self.clim_smoothing = clim_smoothing
        self.smoothing = smoothing

        self.base_path = Path(DATADIR, self.dataset, self.varname)
        self.raw_path = self.base_path.joinpath(f"{self.resolution}")
        self.path = self.base_path.joinpath("Experiments")
        self.id = f"{levels_str}_{self.resolution}_{self.region}"
        self.path = self.path.joinpath(self.id)
        unpacked = unpack_smooth_map(self.clim_smoothing)
        underscore = "_" if unpacked != "" else ""
        self.clim_path = self.path.joinpath(self.clim_type + underscore + unpacked)
        self.anom_path = self.clim_path.joinpath(
            unpack_smooth_map(self.smoothing)
        )  # may be same as clim_path if no smoothing or detrending
        self.anom_path.mkdir(parents=True, exist_ok=True)

        if self.raw_path.joinpath("full.nc").is_file():
            self.file_structure = "one_file"
        if any(
            [self.raw_path.joinpath(f"{year}.nc").is_file() for year in YEARSPL_EXT]
        ):
            self.file_structure = "yearly"
        if any(
            [self.raw_path.joinpath(f"{year}01.nc").is_file() for year in YEARSPL_EXT]
        ):
            self.file_structure = "monthly"

        self.copy_files()
        self.compute_anomaly(clim_type, clim_smoothing, smoothing)

    def _open_rename(self_, path: Path | list, chunk_type: str = None, **kwargs):
        da = xr.open_dataarray(path, **kwargs)

        try:
            da = da.rename({"longitude": "lon", "latitude": "lat"})
        except ValueError:
            pass
        try:
            da = da.rename({"level": "lev"})
        except ValueError:
            pass

        chunks = {"lev": 1} if "lev" in da.dims else {}
        if chunk_type in ["horiz", "horizointal", "lonlat"]:
            chunks = dict(time=20, lon=-1, lat=-1, **chunks)
        elif chunk_type in ["time"]:
            chunks = dict(time=-1, lon=30, lat=30, **chunks)
        else:
            chunks = dict(time=30, lon=30, lat=30, **chunks)
        return da.chunk(chunks)

    def open_da(
        self,
        which: str = "anom",
        period: list | tuple | Literal["all"] = "all",
        season: list | str = None,
        **kwargs,
    ):
        if which.lower() in ["raw", "orig", "original"]:
            path = self.raw_path
        if which.lower() in ["", "absolute", "abs"]:
            path = self.path
        if which.lower() in ["clim", "climatology"]:
            path = self.clim_path
        if which.lower() in ["anom", "anomaly", "detrended"]:
            path = self.anom_path

        if isinstance(period, tuple):
            period = np.arange(period[0], period[1] + 1)
        elif period == "all":
            period = np.unique([fn.name[:4] for fn in path.iterdir() if fn.is_file()])

        if self.file_structure == "one_file":
            files_to_load = ["full.nc"]
        elif self.file_structure == "yearly":
            files_to_load = [f"{year}.nc" for year in period]
        elif self.file_structure == "monthly":
            files_to_load = [
                f"{year}{month}.nc" for month in range(1, 13) for year in period
            ]

        da = []
        for file_to_load in files_to_load:
            da.append(self._open_rename(path.joinpath(file_to_load), **kwargs))
        da = xr.concat(da, dim="time")
        if "chunk_type" in kwargs and kwargs["chunk_type"] == "time":
            da.chunk({"time": -1})

        if isinstance(season, list):
            da.isel(time=np.isin(da.time.dt.month, season))
        elif isinstance(season, str):
            if season in ["DJF", "MAM", "JJA", "SON"]:
                da = da.isel(time=da.time.dt.season == season)
            else:
                raise ValueError(
                    f"Wrong season specifier : {season} is not a valid xarray season"
                )
        if (da.time[1] - da.time[0]).values > np.timedelta64(6, "h"):
            da = da.assign_coords({"time": da.time.values.astype("datetime64[D]")})
        try:
            units = da.attrs["units"]
        except KeyError:
            return da
        if units == "m**2 s**-2":
            da /= co.g
            da.attrs["units"] = "m"
        return da

    def copy_files(self) -> None:
        filenames = list(self.raw_path.iterdir())
        for filename in tqdm(filenames, total=len(filenames)):
            if not filename.is_file():
                continue
            dest = self.path.joinpath(filename.name)
            if dest.is_file():
                continue
            da = self._open_rename(filename, chunk_type="horiz")
            lon_resolution = da.lon[1] - da.lon[0]
            lat_resolution = da.lat[1] - da.lat[0]
            da = da.sel(
                lon=np.arange(
                    self.minlon, self.maxlon + lon_resolution, lon_resolution
                ),
                lat=np.arange(
                    self.minlat, self.maxlat + lat_resolution, lat_resolution
                ),
            )

            if ~any([isinstance(level, tuple) for level in self.levels]):
                da = da.sel(lev=self.levels)
                da.to_netcdf(dest)
                continue

            newcoords = {dim: da.coords[dim] for dim in ["time", "lat", "lon"]}
            if "lev" in da.coords:
                newcoords = newcoords | {"lev": self.level_names}
            shape = [len(coord) for coord in newcoords.values()]
            da2 = xr.DataArray(np.zeros(shape), coords=newcoords)
            for level, level_name in zip(self.levels, self.level_names):
                if isinstance(level, tuple):
                    val = da.sel(lev=level).mean(dim="lev").values
                else:
                    val = da.sel(lev=level).values
                da2.loc[:, :, :, level_name] = val
            da2.to_netcdf(dest)

    def compute_anomaly(
        self,
        clim_type: str = "none",
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
    ) -> None:
        dest_clim = self.clim_path.joinpath("clim.nc")
        dests_anom = [
            self.anom_path.joinpath(fn.name)
            for fn in self.path.iterdir()
            if fn.is_file()
        ]
        if dest_clim.is_file() and all(
            [dest_anom.is_file() for dest_anom in dests_anom]
        ):
            return

        sources = [source for source in self.path.iterdir() if source.is_file()]
        if clim_type == "none":
            for source, dest in tqdm(zip(sources, dests_anom), total=len(dests_anom)):
                if dest.is_file():
                    continue
                anom = self._open_rename(source, chunk_type="time")
                anom = smooth(anom, smoothing).compute(
                    n_workers=N_WORKERS, memory_limit=MEMORY_LIMIT
                )
                anom.to_netcdf(dest)
            return

        if clim_type.lower() in ["doy", "dayofyear"]:
            coordname = "dayofyear"
        else:
            raise NotImplementedError()
        da = self.open_da("abs", period="all", season=None, chunk_type="time")
        gb = da.groupby(f"time.{coordname}")
        with ProgressBar():
            clim = gb.mean(dim="time").compute(
                n_workers=N_WORKERS, memory_limit=MEMORY_LIMIT
            )
        clim = smooth(clim, clim_smoothing)
        clim.to_netcdf(dest_clim)
        for source, dest in tqdm(zip(sources, dests_anom), total=len(dests_anom)):
            time_mask = time_mask(da.time, source.name)
            this_gb = da.sel(time=time_mask).groupby(f"time.{coordname}")
            anom = (this_gb - clim).reset_coords(coordname, drop=True)
            anom = smooth(anom, smoothing).compute(
                n_workers=N_WORKERS, memory_limit=MEMORY_LIMIT
            )
            anom.to_netcdf(dest)

    def get_region(self) -> tuple:
        return (self.minlon, self.maxlon, self.minlat, self.maxlat)

    def _only_windspeed(func):
        @functools.wraps(func)
        def wrapper_decorator(self, *args, **kwargs):
            if self.variable != "s" or self.clim_type != "none":
                print("Only valid for absolute wind speed")
                raise RuntimeError
            value = func(self, *args, **kwargs)

            return value

        return wrapper_decorator

    @_only_windspeed
    def find_jets(
        self, level_name: str | int, season: str = None, force: bool = False
    ) -> Tuple:
        ofile_aj = self.anom_path.joinpath(f"all_jets_{level_name}.pkl")
        ofile_waj = self.anom_path.joinpath(f"where_are_jets_{level_name}.npy")
        ofile_ajoa = self.anom_path.joinpath(f"all_jets_one_array_{level_name}.npy")

        if (
            all([ofile.is_file() for ofile in (ofile_aj, ofile_waj, ofile_ajoa)])
            and not force
        ):
            all_jets = load_pickle(ofile_aj)
            where_are_jets = np.load(ofile_waj)
            all_jets_one_array = np.load(ofile_ajoa)
            return all_jets, where_are_jets, all_jets_one_array

        all_jets = find_all_jets(
            self.open_da("anom", season=season), height=0.12, cutoff=100, chunksize=100
        )
        where_are_jets, all_jets_one_array = all_jets_to_one_array(all_jets)
        save_pickle(all_jets, ofile_aj)
        np.save(ofile_waj, where_are_jets)
        np.save(ofile_ajoa, all_jets_one_array)
        return all_jets, where_are_jets, all_jets_one_array

    @_only_windspeed
    def compute_jet_props(self, season: str = None, force: bool = False) -> Tuple:
        all_jets, _, _ = self.find_jets(season=season, force=force)
        all_props, is_double, is_single, polys = compute_all_jet_props(all_jets)
        return all_props, is_double, is_single, polys

    @_only_windspeed
    def track_jets(self, season: str = None, force: bool = False) -> Tuple:
        all_jets, where_are_jets, all_jets_one_array = self.find_jets(
            season=season, force=force
        )
        ofile_ajot = self.path.joinpath("all_jets_over_time.pkl")
        ofile_flags = self.path.joinpath("flags.npy")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]) and not force:
            all_jets_over_time = load_pickle(ofile_ajot)
            flags = np.load(ofile_flags)

            return (
                all_jets,
                where_are_jets,
                all_jets_one_array,
                all_jets_over_time,
                flags,
            )

        all_jets_over_time, flags = track_jets(all_jets_one_array, where_are_jets)

        save_pickle(all_jets_over_time, ofile_ajot)
        np.save(ofile_flags, flags)
        return all_jets, where_are_jets, all_jets_one_array, all_jets_over_time, flags


class Experiment(object):
    def __init__(
        self,
        dataset: str,
        variable: str,
        level: int | str,
        region: Optional[str] = None,
        smallname: Optional[str] = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        clim_type: str = None,
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
    ) -> None:
        self.dataset = dataset
        self.variable = variable
        self.level = str(level)
        if smallname is None:
            self.smallname = SMALLNAME[variable]
        else:
            self.smallname = smallname
        if region is None:
            if np.any([bound is None for bound in [minlon, maxlon, minlat, maxlat]]):
                raise ValueError(
                    "Specify a region with either a string or 4 ints / floats"
                )
            self.region = f"box_{int(minlon)}_{int(maxlon)}_{int(minlat)}_{int(maxlat)}"
        else:
            self.region = region

        try:
            self.minlon, self.maxlon, self.minlat, self.maxlat = [
                int(bound) for bound in self.region.split("_")[-4:]
            ]
        except ValueError:
            if self.region == "dailymean":
                self.minlon, self.maxlon, self.minlat, self.maxlat = (
                    -180,
                    179.5,
                    -90,
                    90,
                )
            else:
                raise ValueError(f"{region=}, wrong specifier")

        if clim_smoothing is not None and clim_type is None:
            raise ValueError(
                "clim_smoothing is specified so you should specify and clim_type"
            )

        if clim_type is None:
            clim_type = "none"

        if clim_smoothing is None:
            clim_smoothing = {}

        if smoothing is None:
            smoothing = {}

        self.clim_type = clim_type
        self.clim_smoothing = clim_smoothing
        self.smoothing = smoothing

        self.base_path = Path(
            DATADIR, self.dataset, self.variable, self.level, self.region
        )
        unpacked = unpack_smooth_map(self.clim_smoothing)
        underscore = "_" if unpacked != "" else ""
        self.clim_path = self.base_path.joinpath(self.clim_type + underscore + unpacked)
        self.path = self.clim_path.joinpath(
            unpack_smooth_map(self.smoothing)
        )  # may be same as clim_path if no smoothing or detrending
        self.path.mkdir(parents=True, exist_ok=True)

        if not self.file().is_file() and not self.region == "dailymean":
            setup_cdo()
            ifile = self.orig_file()
            ofile = self.file()
            cdo.sellonlatbox(
                self.minlon,
                self.maxlon,
                self.minlat,
                self.maxlat,
                input=ifile.as_posix(),
                output=ofile.as_posix(),
            )
        if not self.file("anom").is_file():
            self.compute_anomaly(self.clim_type, self.clim_smoothing, self.smoothing)

    def orig_file(self) -> Path:
        joinpath = f"{self.smallname}.nc"
        return self.base_path.parent.joinpath("dailymean").joinpath(joinpath)

    def file(self, which: str = "") -> Path:
        if which.lower() in ["", "absolute", "abs"]:
            return self.base_path.joinpath(f"{self.smallname}.nc")
        if which.lower() in ["clim", "climatology"]:
            return self.clim_path.joinpath(f"{self.smallname}_clim.nc")
        if which.lower() in ["anom", "anomaly", "detrended"]:
            return self.path.joinpath(f"{self.smallname}_anom.nc")

    def compute_anomaly(
        self,
        clim_type: str = "none",
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
    ) -> None:
        da = self.open_da(chunks={"time": -1, "lon": 30, "lat": 30})
        if clim_type == "none":
            anom = da
        else:
            anom, clim = compute_anomaly(da, clim_type, clim_smoothing)
            clim.compute(n_workers=N_WORKERS, memory_limit=MEMORY_LIMIT)
            clim.to_netcdf(self.file("clim").as_posix())
            del clim
        anom = smooth(anom, smoothing).compute(
            n_workers=N_WORKERS, memory_limit=MEMORY_LIMIT
        )
        ofile = self.file("anom")
        anom.to_netcdf(ofile.as_posix())

    def open_da(
        self, which: str = "", season: list | str | None = None, **kwargs
    ) -> xr.DataArray:
        da = xr.open_dataset(self.file(which), **kwargs)[self.smallname]
        try:
            da = da.rename({"longitude": "lon", "latitude": "lat"})
        except ValueError:
            pass
        if "time" in da.dims:
            for daterange in (DATERANGEPL, DATERANGEPL_EXT):
                if len(da.time) == len(daterange):
                    da = da.assign_coords({"time": daterange.values})
                    break
                elif len(da.drop_duplicates(dim="time").time) == len(daterange):
                    da = da.drop_duplicates(dim="time").assign_coords(
                        {"time": daterange.values}
                    )
                    break
            if isinstance(season, list):
                da.isel(time=np.isin(da.time.dt.month, season))
            elif isinstance(season, str):
                if season in ["DJF", "MAM", "JJA", "SON"]:
                    da = da.isel(time=da.time.dt.season == season)
                else:
                    raise ValueError(
                        f"Wrong season specifier : {season} is not a valid xarray season"
                    )
            if (da.time[1] - da.time[0]).values > np.timedelta64(6, "h"):
                da = da.assign_coords({"time": da.time.values.astype("datetime64[D]")})
        try:
            units = da.attrs["units"]
        except KeyError:
            return da
        if units == "m**2 s**-2":
            da /= co.g
            da.attrs["units"] = "m"
        return da

    def get_region(self) -> tuple:
        return (self.minlon, self.maxlon, self.minlat, self.maxlat)

    def find_jets(self, season: str = None, force: bool = False) -> Tuple:
        if self.variable != "Wind":
            print("Only valid for wind experiment")
            raise RuntimeError

        ofile_aj = self.path.joinpath("all_jets.pkl")
        ofile_waj = self.path.joinpath("where_are_jets.npy")
        ofile_ajoa = self.path.joinpath("all_jets_one_array.npy")

        if (
            all([ofile.is_file() for ofile in (ofile_aj, ofile_waj, ofile_ajoa)])
            and not force
        ):
            all_jets = load_pickle(ofile_aj)
            where_are_jets = np.load(ofile_waj)
            all_jets_one_array = np.load(ofile_ajoa)
            return all_jets, where_are_jets, all_jets_one_array

        all_jets = find_all_jets(
            self.open_da("anom", season=season), height=0.12, cutoff=100, chunksize=100
        )
        where_are_jets, all_jets_one_array = all_jets_to_one_array(all_jets)
        save_pickle(all_jets, ofile_aj)
        np.save(ofile_waj, where_are_jets)
        np.save(ofile_ajoa, all_jets_one_array)
        return all_jets, where_are_jets, all_jets_one_array

    def compute_jet_props(self, season: str = None, force: bool = False) -> Tuple:
        if self.variable != "Wind":
            print("Only valid for wind experiment")
            raise RuntimeError
        all_jets, _, _ = self.find_jets(season=season, force=force)
        all_props, is_double, is_single, polys = compute_all_jet_props(all_jets)
        return all_props, is_double, is_single, polys

    def track_jets(self, season: str = None, force: bool = False) -> Tuple:
        if self.variable != "Wind":
            print("Only valid for wind experiment")
            raise RuntimeError

        all_jets, where_are_jets, all_jets_one_array = self.find_jets(
            season=season, force=force
        )
        ofile_ajot = self.path.joinpath("all_jets_over_time.pkl")
        ofile_flags = self.path.joinpath("flags.npy")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]) and not force:
            all_jets_over_time = load_pickle(ofile_ajot)
            flags = np.load(ofile_flags)

            return (
                all_jets,
                where_are_jets,
                all_jets_one_array,
                all_jets_over_time,
                flags,
            )

        all_jets_over_time, flags = track_jets(all_jets_one_array, where_are_jets)

        save_pickle(all_jets_over_time, ofile_ajot)
        np.save(ofile_flags, flags)
        return all_jets, where_are_jets, all_jets_one_array, all_jets_over_time, flags

    def other(
        self,
        variable: str,
        level: str,
        smallname: str = None,
        clim_type: str = "same",
        clim_smoothing: Mapping | str = "same",
        smoothing: Mapping | str = "same",
    ) -> "Experiment":
        args = [clim_type, clim_smoothing, smoothing]
        defaults = [self.clim_type, self.clim_smoothing, self.smoothing]
        new_args = []
        for arg, default in zip(args, defaults):
            new_args.append(default if arg == "same" else arg)

        return Experiment(
            self.dataset,
            variable,
            level,
            self.region,
            smallname,
            None,
            None,
            None,
            None,
            *new_args,
        )
        
class ClusteringExperiment(Experiment):
    def __init__(
        self,
        dataset: str,
        variable: str,
        level: int | str,
        region: Optional[str] = None,
        smallname: Optional[str] = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        midfix: str = "anomaly",
        season: list | str = None,
        clim_type: str = None,
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
        inner_norm: int = None,
    ) -> None:
        super().__init__(
            dataset,
            variable,
            level,
            region,
            smallname,
            minlon,
            maxlon,
            minlat,
            maxlat,
            clim_type,
            clim_smoothing,
            smoothing,
        )
        self.midfix = midfix
        self.season = season
        self.inner_norm = inner_norm
        filename = []
        for strin in (midfix, season, inner_norm):
            if strin is not None:
                filename.append(str(strin))
        self.filename = "_".join(filename)

    def prepare_for_clustering(self) -> Tuple[NDArray, xr.DataArray]:
        da = self.open_da(self.midfix, self.season)
        norm_path = self.path.joinpath(f"norm_{self.filename}.nc")
        if norm_path.is_file():
            norm_da = xr.open_dataarray(norm_path.as_posix())
        else:
            norm_da = np.sqrt(degcos(da.lat))

            if self.inner_norm and self.inner_norm == 1:  # Grams et al. 2017
                stds = (
                    (da * norm_da)
                    .chunk({"time": -1, "lon": 20, "lat": 20})
                    .rolling({"time": 30}, center=True, min_periods=1)
                    .std()
                    .mean(dim=["lon", "lat"])
                    .compute(n_workers=N_WORKERS, memory_limit=MEMORY_LIMIT)
                )
                norm_da = norm_da * (1 / stds)
            elif self.inner_norm and self.inner_norm == 2:
                stds = (da * norm_da).std(dim="time")
                norm_da = norm_da * (1 / stds)
            if self.inner_norm and self.inner_norm not in [1, 2]:
                raise NotImplementedError()
            norm_da.to_netcdf(norm_path.as_posix())
        da_weighted = da * norm_da
        X = da_weighted.values.reshape(len(da_weighted.time), -1)
        return X, da

    def compute_pcas(self, n_pcas: int, force: bool = False) -> str:
        glob_string = f"pca_*_{self.filename}.pkl"
        potential_paths = [
            Path(path) for path in glob.glob(self.path.joinpath(glob_string).as_posix())
        ]
        potential_paths = {
            path: int(path.parts[-1].split("_")[1]) for path in potential_paths
        }
        found = False
        for key, value in potential_paths.items():
            if value >= n_pcas:
                found = True
                break
        if found and not force:
            return key
        logging.debug(f"Computing {n_pcas} pcas")
        X, _ = self.prepare_for_clustering()
        pca_path = self.path.joinpath(f"pca_{n_pcas}_{self.filename}.pkl")
        results = pca(n_components=n_pcas, whiten=True).fit(X)
        save_pickle(results, pca_path)
        return pca_path

    def pca_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
    ) -> NDArray[Shape["*, *"], Float]:
        if not n_pcas:
            return X
        pca_path = self.compute_pcas(n_pcas)
        pca_results = load_pickle(pca_path)
        X = pca_results.transform(X)[:, :n_pcas]
        return X

    def pca_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
    ) -> NDArray[Shape["*, *"], Float]:
        if not n_pcas:
            return X
        pca_path = self.compute_pcas(n_pcas)
        pca_results = load_pickle(pca_path)
        diff_n_pcas = pca_results.n_components - X.shape[1]
        X = np.pad(X, [[0, 0], [0, diff_n_pcas]])
        X = pca_results.inverse_transform(X)
        return X.reshape(X.shape[0], -1)

    def _compute_opps_T1(
        self,
        X: NDArray,
        lag_max: int,
    ) -> dict:
        autocorrs = compute_autocorrs(X, lag_max)
        M = np.trapz(autocorrs + autocorrs.transpose((0, 2, 1)), axis=0)

        invC0 = linalg.inv(autocorrs[0])
        eigenvals, eigenvecs = linalg.eigh(0.5 * invC0 @ M)
        OPPs = autocorrs[0] @ eigenvecs.T
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        OPPs = OPPs[idx]
        T1s = np.sum(
            OPPs.reshape(OPPs.shape[0], 1, 1, OPPs.shape[1])
            @ autocorrs
            @ OPPs.reshape(OPPs.shape[0], 1, OPPs.shape[1], 1),
            axis=1,
        ).squeeze()
        T1s /= (
            OPPs.reshape(OPPs.shape[0], 1, OPPs.shape[1])
            @ autocorrs[0]
            @ OPPs.reshape(OPPs.shape[0], OPPs.shape[1], 1)
        ).squeeze()
        return {
            "T": T1s,
            "OPPs": OPPs,
        }

    def _compute_opps_T2(self, X: NDArray, lag_max: int) -> dict:
        autocorrs = compute_autocorrs(X, lag_max)
        C0sqrt = linalg.sqrtm(autocorrs[0])
        C0minushalf = linalg.inv(C0sqrt)
        basis = linalg.orth(C0minushalf)

        def minus_T2(x) -> float:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            return -2 * np.trapz(factor1**2) / normxsq**2

        def minus_T2_gradient(x) -> NDArray[Shape["*"], Float]:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            factor2 = (
                C0minushalf @ (autocorrs + autocorrs.transpose((0, 2, 1))) @ C0minushalf
            ) @ x
            numerator = 4 * np.trapz((factor1)[:, None] * factor2, axis=0)
            return -numerator / normxsq**2 - 4 * minus_T2(x) * x / normxsq**3

        def norm0(x) -> float:
            return 10 - linalg.norm(x) ** 2

        def jac_norm0(x) -> NDArray[Shape["*"], Float]:
            return -2 * x

        Id = np.eye(X.shape[1])
        proj = Id.copy()
        OPPs = []
        T2s = []
        numsuc = 0
        while numsuc < 10:
            xmin, xmax = np.amin(basis, axis=0), np.amax(basis, axis=0)
            x0 = xmin + (xmax - xmin) * np.random.rand(len(xmax))
            res = minimize(
                minus_T2,
                x0,
                jac=minus_T2_gradient,
                method="SLSQP",
                constraints={"type": "ineq", "fun": norm0, "jac": jac_norm0},
            )
            if res.success:
                unit_x = res.x / linalg.norm(res.x)
                OPPs.append(C0sqrt @ unit_x)
                T2s.append(-res.fun)
                proj = Id - np.outer(unit_x, unit_x)
                autocorrs = proj @ autocorrs @ proj
                C0minushalf = proj @ C0minushalf @ proj
                numsuc += 1
        return {
            "T": np.asarray(T2s),
            "OPPs": np.asarray(OPPs),
        }

    def compute_opps(
        self,
        n_pcas: int = None,
        lag_max: int = 90,
        type: int = 1,
        return_realspace: bool = False,
    ) -> Tuple[Path, dict] | Tuple[NDArray, xr.DataArray, Path]:
        if type not in [1, 2]:
            raise ValueError(f"Wrong OPP type, pick 1 or 2")
        X, da = self.prepare_for_clustering()
        if n_pcas:
            X = self.pca_transform(X, n_pcas)
        X = X.reshape((X.shape[0], -1))
        n_pcas = X.shape[1]
        opp_path: Path = self.path.joinpath(f"opp_{n_pcas}_{self.filename}_T{type}.pkl")
        results = None
        if not opp_path.is_file():
            if type == 1:
                logging.debug("Computing T1 OPPs")
                results = self._compute_opps_T1(X, lag_max)
            if type == 2:
                logging.debug("Computing T2 OPPs")
                results = self._compute_opps_T2(X, lag_max)
            save_pickle(results, opp_path)
        if results is None:
            results = load_pickle(opp_path)
        if not return_realspace:
            return opp_path, results
        OPPs = results["OPPs"]
        eigenvals = results["T"]
        OPPs = centers_as_dataarray(OPPs, X, da, "OPP")
        return eigenvals, OPPs, opp_path

    def opp_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int,
        cutoff: int = None,
        type: int = 1,
    ) -> NDArray[Shape["*, *"], Float]:
        _, results = self.compute_opps(n_pcas, type=type)
        if not X.shape[1] == n_pcas:
            X = self.pca_transform(X, n_pcas)
        if cutoff is None:
            cutoff = n_pcas if type == 1 else 10
        OPPs = results["OPPs"][:cutoff]
        X = X @ OPPs.T
        return X

    def opp_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
        cutoff: int = None,
        to_realspace=False,
        type: int = 1,
    ) -> NDArray[Shape["*, *"], Float]:
        _, results = self.compute_opps(n_pcas, type=type)
        if cutoff is None:
            cutoff = n_pcas if type == 1 else 10
        OPPs = results["OPPs"][:cutoff]
        X = X @ OPPs
        if to_realspace:
            return self.pca_inverse_transform(X)
        return X

    def inverse_transform_centers(
        self,
        centers: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
        coord: str = "eof",
    ) -> xr.DataArray:
        logging.debug("Transforming to dataarray")
        da = self.open_da("clim")
        coords = {
            coord: np.arange(centers.shape[0]),
            "lat": da.lat.values,
            "lon": da.lon.values,
        }
        shape = [len(coord) for coord in coords.values()]
        if n_pcas:  # 0 pcas is also no transform
            centers = self.pca_inverse_transform(centers, n_pcas)
        centers = xr.DataArray(centers.reshape(shape), coords=coords)
        norm_path = self.path.joinpath(f"norm_{self.filename}.nc")
        norm_da = xr.open_dataarray(norm_path.as_posix())
        if "time" in norm_da:
            norm_da = norm_da.mean(dim="time")
        return centers / norm_da

    def center_output(
        self,
        centers,
        labels,
        medoids: NDArray = None,
        return_type: int = RAW,
        da: xr.DataArray = None,
        X: NDArray = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        unique_labels, counts = np.unique(labels, return_counts=True)
        counts = counts / len(labels)
        labels = xr.DataArray(labels, coords={"time": da.time.values})

        if return_type == ADJUST_RAW:
            centers = np.stack([np.mean(X[labels == i], axis=0) for i in unique_labels])

        if return_type in [RAW, ADJUST_RAW, RAW_ADJUST_LABELS]:
            centers = xr.DataArray(
                centers,
                coords={
                    "cluster": np.arange(centers.shape[0]),
                    "pca": np.arange(centers.shape[1]),
                },
            )

        elif return_type in [REALSPACE_FROM_LABELS, ADJUST_REALSPACE]:
            centers = labels_to_centers(labels, da, coord="cluster")

        elif return_type in [REALSPACE_INV_TRANS, REALSPACE_INV_TRANS_ADJUST_LABELS]:
            centers = self.inverse_transform_centers(
                centers, centers.shape[1], coord="cluster"
            )

        elif return_type in [DIRECT_REALSPACE, ADJUST_DIRECT_REALSPACE]:
            centers = da.isel(time=medoids)
            centers = centers.assign_coords({"time": np.arange(centers.shape[0])})
            centers = centers.rename({"time": "medoid"})

        else:
            raise ValueError(f"Wrong return type")

        try:
            centers = centers.assign_coords({"ratios": ("cluster", counts)})
        except AttributeError:
            pass

        return centers, labels

    def cluster(
        self,
        n_clu: int,
        n_pcas: int,
        kind: str = "kmeans",
        return_type: int = RAW,
    ) -> str | Tuple[xr.DataArray, xr.DataArray, str]:
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        if CIequal(kind, "kmeans"):
            results = KMeans(n_clu)
            suffix = ""
        elif CIequal(kind, "kmedoids"):
            results = KMedoids(n_clu)
            suffix = "med"
        else:
            raise NotImplementedError(
                f"{kind} clustering not implemented. Options are kmeans and kmedoids"
            )

        results_path = self.path.joinpath(
            f"k{suffix}_{n_clu}_{n_pcas}_{self.filename}.pkl"
        )
        if results_path.is_file():
            results = load_pickle(results_path)
        else:
            logging.debug(f"Fitting {kind} clustering with {n_clu} clusters")
            results = results.fit(X)
            save_pickle(results, results_path)

        if return_type is None:
            return results_path

        direct = return_type in [DIRECT_REALSPACE, ADJUST_DIRECT_REALSPACE]
        if direct and not CIequal(kind, "kmedoids"):
            warnings.warn(f"Return type {return_type} only compatible with kmedoids")
            return_type = (
                REALSPACE_FROM_LABELS
                if return_type == DIRECT_REALSPACE
                else ADJUST_REALSPACE
            )

        centers = results.cluster_centers_
        try:
            medoids = results.medoid_indices_
        except AttributeError:
            medoids = None

        if (
            return_type
            in [  # Has to be here if center_output is to be able to accept both OPP clustering and regular clustering. Absolutely dirty, might change later
                RAW_ADJUST_LABELS,
                ADJUST_RAW,
                REALSPACE_INV_TRANS_ADJUST_LABELS,
                ADJUST_REALSPACE,
                ADJUST_DIRECT_REALSPACE,
            ]
        ):
            projection = project_onto_clusters(X, centers)
            labels = cluster_from_projs(projection, neg=False)

        else:
            labels = results.labels_

        centers, labels = self.center_output(
            centers, labels, medoids, return_type, da, X
        )
        return centers, labels, results_path

    def opp_cluster(
        self,
        n_clu: int,
        n_pcas: int,
        type: int = 1,
        return_type: int = RAW,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        adjust = return_type in [
            ADJUST_RAW,
            RAW_ADJUST_LABELS,
            REALSPACE_INV_TRANS_ADJUST_LABELS,
            ADJUST_REALSPACE,
        ]

        if type == 3:
            OPPs = np.empty((2 * n_clu, n_pcas))
            OPPs[::2] = self.compute_opps(n_pcas, type=1)[1]["OPPs"][:n_clu]
            OPPs[1::2] = self.compute_opps(n_pcas, type=2)[1]["OPPs"][:n_clu]
            X1 = self.opp_transform(X, n_pcas, cutoff=n_clu, type=1)
            X2 = self.opp_transform(X, n_pcas, cutoff=n_clu, type=2)
        elif type in [1, 2]:
            OPPs = self.compute_opps(n_pcas, type=type)[1]["OPPs"][:n_clu]
            X1 = self.opp_transform(X, n_pcas, cutoff=n_clu, type=type)
            X2 = None

        labels = cluster_from_projs(X1, X2, cutoff=n_clu, neg=True, adjust=adjust)
        return self.center_output(OPPs, labels, None, return_type, da, X)

    def compute_som(
        self,
        nx: int,
        ny: int,
        n_pcas: int = None,
        OPP: int = 0,
        GPU: bool = False,
        return_type: int = RAW,
        train_kwargs: dict = None,
        **kwargs,
    ) -> Tuple[SOMNet, xr.DataArray, NDArray]:
        if n_pcas is None and OPP:
            logging.warning("OPP flag will be ignored because n_pcas is set to None")
        opp_suffix = ""

        if OPP == 1:
            opp_suffix = "_T1"
        elif OPP == 2:
            opp_suffix = "_T2"

        output_path = self.path.joinpath(
            f"som_{nx}_{ny}_{n_pcas}_{self.filename}{opp_suffix}.npy"
        )

        if train_kwargs is None:
            train_kwargs = {}

        if OPP:
            X, da = self.prepare_for_clustering()
            X = self.opp_transform(X, n_pcas=n_pcas, type=OPP)
        else:
            X, da = self.prepare_for_clustering()
            X = self.pca_transform(X, n_pcas=n_pcas)

        if GPU:
            try:
                X = cp.asarray(X)
            except NameError:
                GPU = False

        if X.shape[1] > 1000:
            init = "random"
        else:
            init = "pca"

        if output_path.is_file():
            net = SOMNet(nx, ny, X, GPU=GPU, PBC=True, load_file=output_path.as_posix())
        else:
            net = SOMNet(
                nx,
                ny,
                X,
                PBC=True,
                GPU=GPU,
                init=init,
                **kwargs,
                # output_path=self.path.as_posix(),
            )
            net.train(**train_kwargs)
            net.save_map(output_path.as_posix())

        if (
            return_type
            in [  # Has to be here if center_output is to be able to accept both OPP clustering and regular clustering. Absolutely dirty, might change later
                RAW_ADJUST_LABELS,
                ADJUST_RAW,
                REALSPACE_INV_TRANS_ADJUST_LABELS,
                ADJUST_REALSPACE,
                ADJUST_DIRECT_REALSPACE,
            ]
        ):
            projection = project_onto_clusters(X, net.weights)
            labels = cluster_from_projs(projection, neg=False)
        else:
            labels = net.bmus

        centers, labels = self.center_output(
            net.weights, labels, None, return_type, da, X
        )
        return net, centers, labels

    def other(
        self,
        variable: str,
        level: str,
        smallname: str = None,
        midfix: str = "same",
        season: list | tuple | str = "same",
        clim_type: str = "same",
        clim_smoothing: Mapping | str = "same",
        smoothing: Mapping | str = "same",
        inner_norm: int | str = "same",
    ) -> "ClusteringExperiment":
        args = [midfix, season, clim_type, clim_smoothing, smoothing, inner_norm]
        defaults = [
            self.midfix,
            self.season,
            self.clim_type,
            self.clim_smoothing,
            self.smoothing,
            self.inner_norm,
        ]
        new_args = []
        for arg, default in zip(args, defaults):
            new_args.append(default if arg == "same" else arg)

        return ClusteringExperiment(
            self.dataset,
            variable,
            level,
            self.region,
            smallname,
            None,
            None,
            None,
            None,
            *new_args,
        )
        


def meandering(lines):
    m = 0
    for line in lines:  # typically few so a lopp is fine
        m += np.sum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1))) / 360
    return m


def one_ts(lon, lat, da):  # can't really vectorize this, have to parallelize
    m = []
    gen = contourpy.contour_generator(x=lon, y=lat, z=da)
    for lev in range(4900, 6205, 5):
        m.append(meandering(gen.lines(lev)))
    return np.amax(m)


@dataclass(init=False)
class ZooExperiment(object):
    def __init__(
        self,
        dataset: str,
        region: Optional[str],
        minlon: Optional[int | float],
        maxlon: Optional[int | float],
        minlat: Optional[int | float],
        maxlat: Optional[int | float],
    ):
        self.dataset = dataset
        self.exp_u = Experiment(
            dataset,
            "Wind",
            "Low",
            region,
            "u",
            minlon,
            maxlon,
            minlat,
            maxlat,
            "doy",
            clim_smoothing={"dayofyear": ("fft", 150)},
            smoothing={"lon": ("win", 60)},
        )
        self.exp_z = Experiment(
            dataset,
            "Geopotential",
            "500",
            region,
            None,
            minlon,
            maxlon,
            minlat,
            maxlat,
            clim_type="doy",
            clim_smoothing={},
            smoothing={},
        )
        self.region = self.exp_u.region
        self.minlon = self.exp_u.minlon
        self.maxlon = self.exp_u.maxlon
        self.minlat = self.exp_u.minlat
        self.maxlat = self.exp_u.maxlat
        self.da_wind = self.exp_u.open_da("anom").squeeze().load()
        file_zonal_mean = self.exp_u.file("zonal_mean")
        if file_zonal_mean.is_file():
            self.da_wind_zonal_mean = xr.open_dataarray(file_zonal_mean)
        else:
            self.da_wind_zonal_mean = (
                xr.open_dataset(self.exp_u.ifile())[self.exp_u.smallname]
                .sel(lon=self.da_wind.lon, lat=self.da_wind.lat)
                .mean(dim="lon")
            )
            self.da_wind_zonal_mean.to_netcdf(file_zonal_mean)
        self.da_z = self.exp_z.open_da().squeeze()

    def compute_JLI(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Computes the Jet Latitude Index (also called Lat) as well as the wind speed at the JLI (Int)

        Args:

        Returns:
            Lat (xr.DataArray): Jet Latitude Index (see Woollings et al. 2010, Barriopedro et al. 2022)
            Int (xr.DataArray): Wind speed at the JLI (see Woollings et al. 2010, Barriopedro et al. 2022)
        """
        da_Lat = self.da_wind_zonal_mean
        LatI = da_Lat.argmax(dim="lat", skipna=True)
        self.Lat = xr.DataArray(
            da_Lat.lat[LatI.values.flatten()].values, coords={"time": da_Lat.time}
        ).rename("Lat")
        self.Lat.attrs["units"] = "degree_north"
        self.Int = da_Lat.isel(lat=LatI).reset_coords("lat", drop=True).rename("Int")
        self.Int.attrs["units"] = "m/s"
        return self.Lat, self.Int

    def compute_Shar(self) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Computes sharpness and south + north latitudinal extent of the jet

        Args:

        Returns:
            Shar (xr.DataArray): Sharpness (see Woollings et al. 2010, Barriopedro et al. 2022)
            Lats (xr.DataArray): Southward latitudinal extent of the jet (see Woollings et al. 2010, Barriopedro et al. 2022)
            Latn (xr.DataArray): Northward latitudinal extent of the jet (see Woollings et al. 2010, Barriopedro et al. 2022)
        """
        da_Lat = self.da_wind_zonal_mean
        self.Shar = (self.Int - da_Lat.mean(dim="lat")).rename("Shar")
        self.Shar.attrs["units"] = self.Int.attrs["units"]
        difference_with_shar = da_Lat - self.Shar / 2
        roots = np.where(
            difference_with_shar.values[:, 1:] * difference_with_shar.values[:, :-1] < 0
        )
        hist = np.histogram(roots[0], bins=np.arange(len(da_Lat.time) + 1))[0]
        cumsumhist = np.append([0], np.cumsum(hist)[:-1])
        self.Lats = xr.DataArray(
            da_Lat.lat.values[roots[1][cumsumhist]],
            coords={"time": da_Lat.time},
            name="Lats",
        )
        self.Latn = xr.DataArray(
            da_Lat.lat.values[roots[1][cumsumhist + hist - 1]],
            coords={"time": da_Lat.time},
            name="Latn",
        )
        self.Latn[self.Latn < self.Lat] = da_Lat.lat[-1]
        self.Lats[self.Lats > self.Lat] = da_Lat.lat[0]
        self.Latn.attrs["units"] = "degree_north"
        self.Lats.attrs["units"] = "degree_north"
        return self.Shar, self.Lats, self.Latn

    def compute_Tilt(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """Computes tilt and also returns the tracked latitudes

        Args:

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        self.trackedLats = (
            self.da_wind.isel(lat=0)
            .copy(data=np.zeros(self.da_wind.shape[::2]))
            .reset_coords("lat", drop=True)
            .rename("Tracked Latitudes")
        )
        self.trackedLats.attrs["units"] = "degree_north"
        lats = self.da_wind.lat.values
        twodelta = lats[2] - lats[0]
        midpoint = int(len(self.da_wind.lon) / 2)
        self.trackedLats[:, midpoint] = self.Lat
        iterator = zip(
            reversed(range(midpoint)), range(midpoint + 1, len(self.da_wind.lon))
        )
        for lonw, lone in iterator:
            for k, thislon in enumerate((lonw, lone)):
                otherlon = thislon - (
                    2 * k - 1
                )  # previous step in the iterator for either east (k=1, otherlon=thislon-1) or west (k=0, otherlon=thislon+1)
                mask = (
                    np.abs(
                        self.trackedLats[:, otherlon].values[:, None] - lats[None, :]
                    )
                    > twodelta
                )
                # mask = where not to look for a maximum. The next step (forward for east or backward for west) needs to be within twodelta of the previous (otherlon)
                da_wind_at_thislon = self.da_wind.isel(lon=thislon).values
                here = np.ma.argmax(
                    np.ma.array(da_wind_at_thislon, mask=mask),
                    axis=1,
                )
                self.trackedLats[:, thislon] = lats[here]
        self.Tilt = (
            self.trackedLats.polyfit(dim="lon", deg=1)
            .sel(degree=1)["polyfit_coefficients"]
            .reset_coords("degree", drop=True)
            .rename("Tilt")
        )
        self.Tilt.attrs["units"] = "degree_north/degree_east"
        return self.trackedLats, self.Tilt

    def compute_Lon(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """_summary_

        Args:

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        self.Intlambda = self.da_wind.sel(lat=self.trackedLats).reset_coords(
            "lat", drop=True
        )
        Intlambdasq = self.Intlambda * self.Intlambda
        lons = xr.DataArray(
            self.da_wind.lon.values[None, :] * np.ones(len(self.da_wind.time))[:, None],
            coords={"time": self.da_wind.time, "lon": self.da_wind.lon},
        )
        self.Lon = (lons * Intlambdasq).sum(dim="lon") / Intlambdasq.sum(dim="lon")
        self.Lon.attrs["units"] = "degree_east"
        self.Lon = self.Lon.rename("Lon")
        return self.Intlambda, self.Lon

    def compute_Lonew(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """_summary_

        Args:

        Returns:
            tuple[xr.DataArray, xr.DataArray]: _description_
        """
        Intlambda = self.Intlambda.values
        Mean = np.mean(Intlambda, axis=1)
        lon = self.da_wind.lon.values
        iLon = np.argmax(lon[None, :] - self.Lon.values[:, None] > 0, axis=1)
        basearray = Intlambda - Mean[:, None] < 0
        iLonw = (
            np.ma.argmin(
                np.ma.array(basearray, mask=lon[None, :] > self.Lon.values[:, None]),
                axis=1,
            )
            - 1
        )
        iLone = (
            np.ma.argmax(
                np.ma.array(basearray, mask=lon[None, :] <= self.Lon.values[:, None]),
                axis=1,
            )
            - 1
        )
        self.Lonw = xr.DataArray(
            lon[iLonw], coords={"time": self.da_wind.time}, name="Lonw"
        )
        self.Lone = xr.DataArray(
            lon[iLone], coords={"time": self.da_wind.time}, name="Lone"
        )
        self.Lonw.attrs["units"] = "degree_east"
        self.Lone.attrs["units"] = "degree_east"
        return self.Lonw, self.Lone

    def compute_Dep(self) -> xr.DataArray:
        """_summary_

        Args:

        Returns:
            xr.DataArray: _description_
        """
        phistarl = xr.DataArray(
            self.da_wind.lat.values[self.da_wind.argmax(dim="lat").values],
            coords={"time": self.da_wind.time.values, "lon": self.da_wind.lon.values},
        )
        self.Dep = (
            np.sqrt((phistarl - self.trackedLats) ** 2).sum(dim="lon").rename("Dep")
        )
        self.Dep.attrs["units"] = "degree_north"
        return self.Dep

    def compute_Mea(self, njobs: int = 8) -> xr.DataArray:
        lon = self.da_z.lon.values
        lat = self.da_z.lat.values
        self.Mea = Parallel(
            n_jobs=njobs, backend="loky", max_nbytes=1e6, verbose=0, batch_size=50
        )(
            delayed(one_ts)(lon, lat, self.da_z.sel(time=t).values)
            for t in self.da_z.time[:]
        )
        self.Mea = xr.DataArray(self.Mea, coords={"time": self.da_z.time}, name="Mea")
        return self.Mea

    def get_Zoo_path(self) -> str:
        return self.exp_u.path.joinpath("Zoo.nc")

    def compute_Zoo(self, detrend=False) -> str:
        logging.debug("Lat")
        _ = self.compute_JLI()
        logging.debug("Shar")
        _ = self.compute_Shar()
        logging.debug("Tilt")
        _ = self.compute_Tilt()
        logging.debug("Lon")
        _ = self.compute_Lon()
        logging.debug("Lonew")
        _ = self.compute_Lonew()
        logging.debug("Dep")
        _ = self.compute_Dep()
        logging.debug("Mea")
        _ = self.compute_Mea()

        Zoo = xr.Dataset(
            {
                "Lat": self.Lat,
                "Int": self.Int,
                "Shar": self.Shar,
                "Lats": self.Lats,
                "Latn": self.Latn,
                "Tilt": self.Tilt,
                "Lon": self.Lon,
                "Lonw": self.Lonw,
                "Lone": self.Lone,
                "Dep": self.Dep,
                "Mea": self.Mea,
            }
        ).dropna(
            dim="time"
        )  # dropna if time does not match between z and u (happens for NCEP)
        self.Zoopath = self.get_Zoo_path()
        if not detrend:
            Zoo.to_netcdf(self.Zoopath)
            return self.Zoopath
        for key, value in Zoo.data_vars.items():
            Zoo[f"{key}_anomaly"], Zoo[f"{key}_climatology"] = compute_anomaly(
                value, return_clim=1, smooth_kmax=3
            )
            Zoo[f"{key}_detrended"] = xrft.detrend(
                Zoo[f"{key}_anomaly"], dim="time", detrend_type="linear"
            )
        Zoo.to_netcdf(self.Zoopath)
        return self.Zoopath





def open_pvs(da_template: xr.DataArray, q: float = 0.9) -> Tuple[xr.DataArray]:
    ofile = Path(f"{DATADIR}/ERA5/plev/pvs/6H")
    ofile1 = ofile.joinpath("full.nc")
    ofile2 = ofile.joinpath("anom.nc")
    ofile3 = ofile.joinpath("anom_normd.nc")
    try:
        da_pvs = xr.open_dataarray(ofile1).load()
        da_pvs_anom = xr.open_dataarray(ofile2).load()
        da_pvs_anom_normd = xr.open_dataarray(ofile3).load()
    except FileNotFoundError:
        print("Events to xarray")
        events = gpd.read_parquet(f"{DATADIR}/ERA5/RWB_index/era5_pv_streamers_350K_1959-2022.parquet")
        events = events[events.event_area >= events.event_area.quantile(q)]
        events = events[np.isin(events.date.dt.month, [6, 7, 8])]
        mask_anti = events.intensity >= 0
        mask_cycl = events.intensity < 0
        mask_tropo = events.mean_var < events.level
        events["flag"] = events.index
        events_anti = events[mask_anti & mask_tropo]
        events_cycl = events[mask_cycl & mask_tropo]
        from wavebreaking import to_xarray
        da_template = da_template.sel(time=da_template.time.dt.year>=1959)
        da_pvs_anti = to_xarray(da_template, events_anti, flag="flag")
        da_pvs_cycl = to_xarray(da_template, events_cycl, flag="flag")
        da_pvs = {"anti": da_pvs_anti, "cycl": da_pvs_cycl}
        da_pvs = xr.Dataset(da_pvs)
        da_pvs = da_pvs.to_array(dim="type").transpose("time", "type", "lat", "lon").chunk({"lon": 10, "lat": 10})
        clim = compute_clim(da_pvs, "hourofyear")
        da_pvs_anom = compute_anom(da_pvs, clim, "hourofyear")
        da_pvs_anom_normd = compute_anom(da_pvs, clim, "hourofyear", True) 
        return da_pvs, da_pvs_anom, da_pvs_anom_normd
    #     da_pvs.astype(int).to_netcdf(ofile1)
    #     da_pvs_anom.astype(np.float32).to_netcdf(ofile2)
    #     da_pvs_anom_normd.astype(np.float32).to_netcdf(ofile3)
    # return da_pvs, da_pvs_anom, da_pvs_anom_normd
    
import xarray as xr
from typing import Tuple


def compute_JLI_JSI(da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """Computes the Jet Latitude Index (also called Lat) as well as the wind speed at the JLI (Int)

    Args:
        da: (xr.DataArray): DataArray containing wind speed
    Returns:
        Lat (xr.DataArray): Jet Latitude Index (see Woollings et al. 2010, Barriopedro et al. 2022)
        Int (xr.DataArray): Wind speed at the JLI (see Woollings et al. 2010, Barriopedro et al. 2022)
    """
    if "lev" in da.dims:
        da = da.sel(lev=slice(900, 700)).mean(dim="lev")
    
    da = da.sel(lon=slice(0, 60), lat=slice(25, 85)).mean(dim=["lon"])
    LatI = da.argmax(dim="lat", skipna=True)
    JLI = xr.DataArray(
        da.lat[LatI.values.flatten()].values, coords={"time": da.time}
    ).rename("Lat")
    JLI.attrs["units"] = "degree_north"
    JSI = da.isel(lat=LatI).reset_coords("lat", drop=True).rename("Int")
    JSI.attrs["units"] = "m/s"
    return JLI, JSI