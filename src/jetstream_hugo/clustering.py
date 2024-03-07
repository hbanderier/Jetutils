import warnings
from functools import wraps, partial
from typing import Sequence, Tuple, Literal, Mapping, Optional, Callable

from sklearn.inspection import partial_dependence
from nptyping import NDArray, Float, Shape
import logging
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError:
    pass
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from numba_progress import ProgressBar as NumbaProgress
from kmedoids import KMedoids
import scipy.linalg as linalg
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from simpsom import SOMNet

from jetstream_hugo.definitions import (
    revert_zero_one,
    save_pickle,
    load_pickle,
    N_WORKERS,
    COMPUTE_KWARGS,
    degcos,
    case_insensitive_equal,
    labels_to_mask,
    to_zero_one,
)

from jetstream_hugo.stats import compute_autocorrs
from jetstream_hugo.data import (
    data_path,
    open_da,
    unpack_levels,
    get_land_mask,
    extract_season,
)

RAW_REALSPACE: int = 0
RAW_PCSPACE: int = 1
ADJUST: int = 2
ADJUST_TWOSIDED: int = 3

def time_mask(time_da: xr.DataArray, filename: str) -> NDArray:
    if filename == "full.nc":
        return np.ones(len(time_da)).astype(bool)

    filename = int(filename.rstrip(".nc"))
    try:
        t1, t2 = pd.to_datetime(filename, format="%Y%M"), pd.to_datetime(
            filename + 1, format="%Y%M"
        )
    except ValueError:
        t1, t2 = pd.to_datetime(filename, format="%Y"), pd.to_datetime(
            filename + 1, format="%Y"
        )
    return ((time_da >= t1) & (time_da < t2)).values


def centers_realspace(centers: NDArray, feature_dims: Mapping) -> xr.DataArray:
    coords = {"cluster": np.arange(centers.shape[0])} | feature_dims
    shape = [len(coord) for coord in coords.values()]
    return xr.DataArray(centers.reshape(shape), coords=coords)


def get_feature_dims(da: xr.DataArray) -> Mapping:
    excluded = ["time", "member", "cluster"]
    return {key: da[key].values for key in da.dims if key not in excluded}


def centers_realspace_from_da(centers: NDArray, da: xr.DataArray) -> xr.DataArray:
    return centers_realspace(centers, get_feature_dims(da))


def labels_from_projs(
    X1: NDArray,
    X2: NDArray = None,
    cutoff: int = None,
    neg: bool = True,
    adjust: bool = True,
) -> NDArray:
    if cutoff is None:
        if X2 is None:
            cutoff = X1.shape[1]
        else:
            cutoff = min(X1.shape[1], X2.shape[1])
    X1 = X1[:, :cutoff]

    if X2 is not None:
        X2 = X2[:, :cutoff]
        X = np.empty((X1.shape[0], X1.shape[1] + X2.shape[1]))
        X[:, ::2] = X1
        X[:, 1::2] = X2
    else:
        X = X1
    sigma = np.std(X, ddof=1)
    if neg:
        max_weight = np.argmax(np.abs(X), axis=1)
        Xmax = np.take_along_axis(X, max_weight[:, None], axis=1)
        sign = np.sign(Xmax)
    else:
        max_weight = np.argmax(X, axis=1)
        Xmax = np.take_along_axis(X, max_weight[:, None], axis=1)
        sign = np.ones(Xmax.shape, dtype=int)
    offset = 0
    if adjust:
        sign[np.abs(Xmax) < sigma] = 0
        offset = 1
    return sign.flatten() * (offset + max_weight)


def labels_to_centers(
    labels: list | NDArray | xr.DataArray, da: xr.DataArray, coord: str = "cluster"
) -> xr.DataArray:
    if isinstance(labels, xr.DataArray):
        labels = labels.values
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts = counts / float(len(labels))
    dims = list(get_feature_dims(da))
    centers = [da.isel(time=(labels == i)).mean(dim=dims) for i in unique_labels]
    centers = xr.concat(centers, dim=coord)
    centers = centers.assign_coords(
        {"ratio": (coord, counts), "label": (coord, unique_labels)}
    )
    return centers.set_xindex("label")


def timeseries_on_map(timeseries: NDArray, labels: list | NDArray):
    timeseries = np.atleast_2d(timeseries)
    mask = labels_to_mask(labels)
    return np.asarray(
        [[np.nanmean(timeseries_[mas]) for mas in mask.T] for timeseries_ in timeseries]
    )


def quantile_exceedence(
    da: xr.DataArray, q: float = 0.95, dim: str = "time"
) -> xr.DataArray:
    return da > da.quantile(q, dim=dim)


def spatial_agglomerative_clustering(
    da: xr.DataArray,
    condition_function: Callable = lambda x: x,
    mask: xr.DataArray | Literal["land"] | None = None,
    season: str | list | None = "JJA",
    metric: str = "jaccard",
) -> NDArray:
    lon, lat = da.lon.values, da.lat.values
    if mask and mask == "land":
        mask = get_land_mask()
    if mask is not None:
        mask = mask.sel(lon=lon, lat=lat)
    da = extract_season(da, season)
    to_cluster = condition_function(da)
    stack_dims = {"lat_lon": ("lat", "lon")}
    to_cluster_flat = to_cluster.stack(stack_dims)
    if mask is not None:
        mask_flat = mask.stack(stack_dims)
        to_cluster_flat = to_cluster_flat.values[:, mask_flat.values]
    return pairwise_distances(to_cluster_flat.T, metric=metric, n_jobs=N_WORKERS)


def select_cluster(
    Z: NDArray,
    da: xr.DataArray,
    n_clusters: int,
    i_cluster: int,
    mask: xr.DataArray | Literal["land"] | None = None,
) -> xr.DataArray:
    lon, lat = da.lon.values, da.lat.values
    if mask and mask == "land":
        mask = get_land_mask()
    if mask is not None:
        mask = mask.sel(lon=lon, lat=lat)
        stack_dims = {"lat_lon": ("lat", "lon")}
        mask_flat = mask.stack(stack_dims)
    clusters = cut_tree(Z, n_clusters=n_clusters)[:, 0]
    clusters_da = np.zeros(mask_flat.shape, dtype=float)
    clusters_da[:] = np.nan
    clusters_da = mask_flat.copy(data=clusters_da)
    clusters_da[mask_flat] = clusters
    this_region = clusters_da.where(clusters_da == i_cluster).unstack()
    return da.where(this_region).mean(dim=["lon", "lat"]).copy()


class Experiment(object):
    def __init__(
        self,
        dataset: str,
        level_type: Literal["plev"] | Literal["thetalev"] | Literal["surf"],
        varname: str,
        resolution: str,
        period: list | tuple | Literal["all"] | int | str = "all",
        season: list | str = None,
        minlon: Optional[int | float] = None,
        maxlon: Optional[int | float] = None,
        minlat: Optional[int | float] = None,
        maxlat: Optional[int | float] = None,
        levels: int | str | tuple | list | Literal["all"] = "all",
        clim_type: str = None,
        clim_smoothing: Mapping = None,
        smoothing: Mapping = None,
        inner_norm: int = None,
        reduce_da: bool = True,
    ) -> None:
        self.path = data_path(
            dataset, level_type, varname, resolution, clim_type, clim_smoothing, smoothing, False
        ).joinpath("results")
        self.path.mkdir(exist_ok=True)
        self.open_da_args = (
            dataset,
            level_type,
            varname,
            resolution,
            period,
            season,
            minlon,
            maxlon,
            minlat,
            maxlat,
            levels,
            clim_type,
            clim_smoothing,
            smoothing,
        )

        self.varname = varname
        self.region = (minlon, maxlon, minlat, maxlat)
        self.clim_type = clim_type
        if levels != 'all':
            self.levels, self.level_names = unpack_levels(levels)
        else: 
            self.levels = 'all'
        self.inner_norm = inner_norm

        self.metadata = {
            "period": period,
            "season": season,
            "region": (minlon, maxlon, minlat, maxlat),
            "levels": self.levels,
            "inner_norm": inner_norm,
        }

        found = False
        for dir in self.path.iterdir():
            if not dir.is_dir():
                continue
            try:
                other_mda = load_pickle(dir.joinpath("metadata.pkl"))
            except FileNotFoundError:
                continue
            if self.metadata == other_mda:
                self.path = self.path.joinpath(dir.name)
                found = True
                break

        if not found:
            seq = [int(dir.name) for dir in self.path.iterdir() if dir.is_dir()]
            id = max(seq) + 1 if len(seq) != 0 else 1
            self.path = self.path.joinpath(str(id))
            self.path.mkdir()
            save_pickle(self.metadata, self.path.joinpath("metadata.pkl"))

        da_path = self.path.joinpath("da.nc")
        if da_path.is_file():
            self.da = xr.open_dataarray(da_path).load()
        else:
            self.da = open_da(*self.open_da_args)
            with ProgressBar():
                self.da = self.da.load()
            self.da.to_netcdf(da_path, format="NETCDF4")

        if reduce_da:
            self.da = self.da.max("lev")
        self.samples_dims = {"time": self.da.time.values}
        try:
            self.samples_dims["member"] = self.da.member.values
        except AttributeError:
            pass
        self.lon, self.lat = self.da.lon.values, self.da.lat.values
        try:
            self.feature_dims = {"lev": self.da.lev.values}
        except AttributeError:
            self.feature_dims = {}
        self.feature_dims["lat"] = self.lat
        self.feature_dims["lon"] = self.lon
        self.flat_shape = (
            np.prod([len(dim) for dim in self.samples_dims.values()]),
            np.prod([len(dim) for dim in self.feature_dims.values()])
        )

    def prepare_for_clustering(self) -> Tuple[NDArray, xr.DataArray]:
        norm_path = self.path.joinpath(f"norm.nc")
        if norm_path.is_file():
            norm_da = xr.open_dataarray(norm_path)
        else:
            norm_da = np.sqrt(degcos(self.da.lat))  # lat as da to simplify mult

            if self.inner_norm and self.inner_norm == 1:  # Grams et al. 2017
                with ProgressBar():
                    stds = (
                        (self.da * norm_da)
                        .rolling({"time": 30}, center=True, min_periods=1)
                        .std()
                        .mean(dim=["lon", "lat"])
                        .compute(**COMPUTE_KWARGS)
                    )
                norm_da = norm_da * (1 / stds)
            elif self.inner_norm and self.inner_norm == 2:
                stds = (self.da * norm_da).std(dim="time")
                norm_da = norm_da * (1 / stds)
            elif self.inner_norm and self.inner_norm not in [1, 2]:
                raise NotImplementedError()
            norm_da.to_netcdf(norm_path)

        da_weighted = self.da * norm_da
        X = da_weighted.values.reshape(self.flat_shape)
        return X, self.da
    
    def _pca_file(self, n_pcas: int) -> str | None:
        potential_paths = list(self.path.glob("pca_*.pkl"))
        potential_paths = {
            path: int(path.stem.split("_")[1]) for path in potential_paths
        }
        for key, value in potential_paths.items():
            if value >= n_pcas:
                return key
        return None

    def compute_pcas(self, n_pcas: int, force: bool = False) -> str:
        path = self._pca_file(n_pcas)
        if path is not None:
            return path
        logging.debug(f"Computing {n_pcas} pcas")
        X, _ = self.prepare_for_clustering()
        pca_path = self.path.joinpath(f"pca_{n_pcas}.pkl")
        results = PCA(n_components=n_pcas, whiten=False).fit(X)
        save_pickle(results, pca_path)
        return pca_path

    def pca_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
    ) -> NDArray[Shape["*, *"], Float]:
        if not n_pcas:
            X, self.Xmin, self.Xmax = to_zero_one(X)
        pca_path = self.compute_pcas(n_pcas)
        pca_results = load_pickle(pca_path)
        transformed_file = self.path.joinpath(f"pca_{n_pcas}.npy")
        if transformed_file.is_file():
            return np.load(transformed_file)
        X = pca_results.transform(X)[:, :n_pcas]
        np.save(transformed_file, X)
        return X

    def pca_inverse_transform(
        self,
        X: NDArray[Shape["*, *"], Float],
        n_pcas: int = None,
    ) -> NDArray[Shape["*, *"], Float]:
        if not n_pcas:
            return revert_zero_one(X, self.Xmin, self.Xmax)
        pca_path = self.compute_pcas(n_pcas)
        pca_results = load_pickle(pca_path)
        diff_n_pcas = pca_results.n_components - X.shape[1]
        X = np.pad(X, [[0, 0], [0, diff_n_pcas]])
        X = pca_results.inverse_transform(X)
        return X.reshape(X.shape[0], -1) # why reshape ?
    
    def labels_as_da(self, labels: NDArray) -> xr.DataArray:
        shape = [len(dim) for dim in self.samples_dims.values()]
        labels = labels.reshape(shape)
        return xr.DataArray(labels, coords=self.samples_dims).rename("labels")
    
    def _centers_realspace(self, centers: NDArray):
        n_pcas_tentative = centers.shape[1]
        pca_path = self._pca_file(n_pcas_tentative)
        if pca_path is not None:
            centers = self.pca_inverse_transform(centers, n_pcas_tentative)
        centers = centers_realspace(centers, self.feature_dims)
        norm_path = self.path.joinpath(f"norm.nc")
        norm_da = xr.open_dataarray(norm_path.as_posix())
        if "time" in norm_da.dims:
            norm_da = norm_da.mean(dim="time")
        return centers / norm_da
    
    def _cluster_output(
        self,
        centers: NDArray,
        labels: NDArray,
        return_type: int = RAW_REALSPACE,
        da: xr.DataArray = None,
        X: NDArray = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        All the clustering methods are responsible for producing their centers in pca space and their labels in sample space. This function handles the rest
        """
        if return_type == RAW_PCSPACE:
            unique, counts = np.unique(labels, return_counts=True)
            counts = counts / float(len(labels))
            labels = self.labels_as_da(labels)
            coords_centers = {
                "cluster": np.arange(centers.shape[0]),
                "pc": np.arange(centers.shape[1])
            }
            centers = xr.DataArray(centers, coords=coords_centers)
            centers = centers.assign_coords({"ratios": ("cluster", counts)})
        
        elif return_type == RAW_REALSPACE:
            unique, counts = np.unique(labels, return_counts=True)
            counts = counts / float(len(labels))
            labels = self.labels_as_da(labels)
            centers = self._centers_realspace(centers)
            centers = centers.assign_coords({"ratios": ("cluster", counts)})
            
        elif return_type in [ADJUST, ADJUST_TWOSIDED]:
            projection = np.tensordot(X, centers.T, axes=X.ndim - 1)
            neg = return_type == ADJUST_TWOSIDED
            newlabels = labels_from_projs(projection, neg=neg, adjust=True)
            centers = labels_to_centers(newlabels, self.da, coord="cluster")
            
        else:
            print("Wrong return specifier")
            raise ValueError

        return centers, labels

    def cluster(
        self,
        n_clu: int,
        n_pcas: int,
        kind: str = "kmeans",
        return_type: int = RAW_REALSPACE,
    ) -> str | Tuple[xr.DataArray, xr.DataArray, str]:
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        if case_insensitive_equal(kind, "kmeans"):
            results = KMeans(n_clu)
            suffix = ""
        elif case_insensitive_equal(kind, "kmedoids"):
            results = KMedoids(n_clu)
            suffix = "med"
        else:
            raise NotImplementedError(
                f"{kind} clustering not implemented. Options are kmeans and kmedoids"
            )

        results_path = self.path.joinpath(f"k{suffix}_{n_clu}_{n_pcas}.pkl")
        if results_path.is_file():
            results = load_pickle(results_path)
        else:
            logging.debug(f"Fitting {kind} clustering with {n_clu} clusters")
            results = results.fit(X)
            save_pickle(results, results_path)
            
        centers = results.cluster_centers_
        labels = results.labels_
        
        return self._cluster_output(centers, labels, return_type, X, da)
    
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

        def minus_T2_gradient(x) -> NDArray:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            factor2 = (
                C0minushalf @ (autocorrs + autocorrs.transpose((0, 2, 1))) @ C0minushalf
            ) @ x
            numerator = 4 * np.trapz(factor1[:, None] * factor2, axis=0)
            return -numerator / normxsq**2 - 4 * minus_T2(x) * x / normxsq**3

        def norm0(x) -> float:
            return 10 - linalg.norm(x) ** 2

        def jac_norm0(x) -> NDArray:
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
    ) -> Tuple[Path, dict]:
        if type not in [1, 2]:
            raise ValueError(f"Wrong OPP type, pick 1 or 2")
        X, da = self.prepare_for_clustering()
        if n_pcas:
            X = self.pca_transform(X, n_pcas)
        X = X.reshape((X.shape[0], -1))
        n_pcas = X.shape[1]
        opp_path: Path = self.path.joinpath(f"opp_{n_pcas}_T{type}.pkl")
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

    def opp_cluster(
        self,
        n_clu: int,
        n_pcas: int,
        type: int = 1,
        return_type: int = RAW_REALSPACE,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)

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

        labels = labels_from_projs(X1, X2, cutoff=n_clu, neg=False, adjust=False)
        return self._cluster_output(OPPs, labels, return_type, da, X)

    def som_cluster(
        self,
        nx: int,
        ny: int,
        n_pcas: int = 0,
        PBC: bool = True,
        return_type: int = RAW_REALSPACE,
        force: bool = False,
        train_kwargs: dict = None,
        **kwargs,
    ) -> Tuple[SOMNet, xr.DataArray, NDArray]:
        pbc_flag = "_pbc" if PBC else ""
        output_path = self.path.joinpath(f"som_{nx}_{ny}{pbc_flag}_{n_pcas}.npy")

        if train_kwargs is None:
            train_kwargs = {}
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas=n_pcas)
        
        init = "random" if X.shape[1] > 1000 else "pca"
        if output_path.is_file() and not force:
            net = SOMNet(nx, ny, X, GPU=False, PBC=PBC, load_file=output_path.as_posix())
        else:
            net = SOMNet(
                nx,
                ny,
                X,
                PBC=PBC,
                init=init,
                **kwargs,
            )
            net.train(**train_kwargs)
            net.save_map(output_path.as_posix())

        labels = net.bmus

        return net, *self._cluster_output(net.weights, labels, return_type, da, X)


    def _only_temp(func):
        @wraps(func)
        def wrapper_decorator(self, *args, **kwargs):
            if self.varname != "t":
                print("Only valid for temperature, single pressure level")
                print(self.varname, self.clim_type, self.levels)
                raise RuntimeError
            value = func(self, *args, **kwargs)

            return value

        return wrapper_decorator

    def linkage(
        self,
        condition_function: Callable = lambda x: x,
        mask: xr.DataArray | Literal["land"] | None = None,
        season: str | list | None = "JJA",
        metric: str = "jaccard",
    ):
        distance_path = self.path.joinpath("distances.npy")
        try:
            distances = np.load(distance_path)
        except FileNotFoundError:
            distances = spatial_agglomerative_clustering(
                self.da, condition_function, mask, season=season, metric=metric
            )
            np.save(distance_path, distances)
        return linkage(squareform(distances), method="average")

    @_only_temp
    def heat_wave_linkage(self):
        condition_function = partial(quantile_exceedence, q=0.95, dim="time")
        mask = "land"
        season = [7, 8]
        metric = "jaccard"
        return self.linkage(condition_function, mask, season, metric)

    @_only_temp
    def select_heat_wave_cluster(self, n_clusters: int = 9, i_cluster: int = 5):
        Z = self.heat_wave_linkage()
        return select_cluster(Z, self.da, n_clusters, i_cluster, "land")
