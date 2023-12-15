import warnings
from functools import wraps, partial
from typing import Sequence, Tuple, Literal, Mapping, Optional
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kmedoids import KMedoids
import scipy.linalg as linalg
from scipy.optimize import minimize
from simpsom import SOMNet

from jetstream_hugo.definitions import (
    save_pickle,
    load_pickle,
    DATADIR,
    DATERANGEPL_EXT,
    DATERANGEPL_EXT_SUMMER,
    YEARSPL,
    YEARSPL_EXT,
    N_WORKERS,
    COMPUTE_KWARGS,
    degcos,
    degsin,
    case_insensitive_equal,
    labels_to_mask,
)

from jetstream_hugo.stats import compute_autocorrs
from jetstream_hugo.data import data_path, open_da, unpack_levels
from jetstream_hugo.jet_finding import (
    find_all_jets, all_jets_to_one_array, props_to_ds, compute_all_jet_props, track_jets, add_persistence_to_props, categorize_ds_jets
)

RAW = 0
RAW_ADJUST_LABELS = 1
ADJUST_RAW = 2
REALSPACE_INV_TRANS = 3
REALSPACE_INV_TRANS_ADJUST_LABELS = 4
REALSPACE_FROM_LABELS = 5
ADJUST_REALSPACE = 6

DIRECT_REALSPACE = 7  # KMEDOIDS only
ADJUST_DIRECT_REALSPACE = 8

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


def project_onto_clusters(
    X: NDArray | xr.DataArray, centers: NDArray | xr.DataArray, weighs: tuple = None
) -> NDArray | xr.DataArray:
    if isinstance(X, NDArray):
        if isinstance(centers, xr.DataArray):  # always cast to type of X
            centers = centers.values
        if weighs is not None:
            X = np.swapaxes(
                np.swapaxes(X, weighs[0], -1) * weighs[1], -1, weighs[0]
            )  # https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
        return np.tensordot(X, centers.T, axes=X.ndim - 1)

    if isinstance(X, xr.DataArray):
        if isinstance(centers, NDArray):
            if centers.ndim == 4:
                centers = centers.reshape(
                    (
                        centers.shape[0] * centers.shape[1],
                        centers.shape[2],
                        centers.shape[3],
                    )
                )
            coords = dict(
                centers=np.arange(centers.shape[0]),
                **{key: val for key, val in X.coords if key != "time"},
            )
            centers = xr.DataArray(centers, coords=coords)
        try:
            weighs = np.sqrt(degcos(X.lat))
            X *= weighs
            denominator = np.sum(weighs.values)
        except AttributeError:
            denominator = 1
        return X.dot(centers) / denominator


def cluster_from_projs(
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
    labels: list | NDArray | xr.DataArray, da: xr.DataArray, coord: str = "center"
) -> xr.DataArray:
    if isinstance(labels, xr.DataArray):
        labels = labels.values
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts = counts / float(len(labels))
    centers = [da.isel(time=(labels == i)).mean(dim="time") for i in unique_labels]
    centers = xr.concat(centers, dim=coord)
    centers = centers.assign_coords({'ratio': (coord, counts), 'label': (coord, unique_labels)})
    return centers.set_xindex('label')


def centers_as_dataarray(
    centers: NDArray[Shape["*, *"], Float] | Tuple[NDArray, NDArray],
    X: NDArray[Shape["*, *"], Float],
    da: xr.DataArray,
    coord: str = "center",
) -> xr.DataArray:
    logging.debug("Projecting on dataarray")
    neg = coord.lower() in ["opp", "opps", "pca", "pcas", "eof", "eofs"]
    if isinstance(centers, tuple):
        proj1 = project_onto_clusters(X, centers[0])
        proj2 = project_onto_clusters(X, centers[1])
        labels = cluster_from_projs(proj1, proj2, neg=neg)
    else:
        projection = project_onto_clusters(X, centers)
        labels = cluster_from_projs(projection, neg=neg)
    return labels_to_centers(labels, da, coord)


def timeseries_on_map(timeseries: NDArray, labels: list | NDArray):
    timeseries = np.atleast_2d(timeseries)
    mask = labels_to_mask(labels)
    return np.asarray(
        [[np.nanmean(timeseries_[mas]) for mas in mask.T] for timeseries_ in timeseries]
    )


class Experiment(object):
    def __init__(
        self,
        dataset: str,
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
    ) -> None:
        self.path = data_path(
            dataset, varname, resolution, clim_type, clim_smoothing, smoothing, False
        ).joinpath("results")
        self.path.mkdir(exist_ok=True)
        self.open_da_args = (
            dataset,
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
        self.levels, self.level_names = unpack_levels(levels)
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
        
        da_path = self.path.joinpath('da.nc')
        if da_path.is_file():
            self.da = xr.open_dataarray(da_path).load()
        else:
            self.da = open_da(*self.open_da_args)
            with ProgressBar():
                self.da = self.da.load()
            self.da.to_netcdf(da_path, format='NETCDF4')

        self.lon, self.lat = self.da.lon.values, self.da.lat.values
        self.time = self.da.time
        
    def add_data(self, new_year: list | tuple | int):
        open_args = self.open_da_args.copy()
        open_args[3] = new_year
        other_da = open_da(open_args)
        self.da = xr.concat([self.da, other_da], dim='time')
        self.da.to_netcdf(self.path.joinpath('da.nc'), format='NETCDF4')
        
    def prepare_for_clustering(self) -> Tuple[NDArray, xr.DataArray]:
        norm_path = self.path.joinpath(f"norm.nc")
        if norm_path.is_file():
            norm_da = xr.open_dataarray(norm_path)
        else:
            norm_da = np.sqrt(degcos(self.da.lat)) # lat as da to simplify mult

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
        X = da_weighted.values.reshape(len(da_weighted.time), -1)
        return X, self.da

    def compute_pcas(self, n_pcas: int, force: bool = False) -> str:
        potential_paths = list(self.path.glob("pca_*.pkl"))
        potential_paths = {
            path: int(path.stem.split("_")[1]) for path in potential_paths
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
        pca_path = self.path.joinpath(f"pca_{n_pcas}.pkl")
        results = PCA(n_components=n_pcas, whiten=True).fit(X)
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
        coords = {
            coord: np.arange(centers.shape[0]),
            "lat": self.lat,
            "lon": self.lon,
        }
        shape = [len(coord) for coord in coords.values()]
        if n_pcas:  # 0 pcas is also no transform
            centers = self.pca_inverse_transform(centers, n_pcas)
        centers = xr.DataArray(centers.reshape(shape), coords=coords)
        norm_path = self.path.joinpath(f"norm.nc")
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

        if return_type is None:
            return results_path

        direct = return_type in [DIRECT_REALSPACE, ADJUST_DIRECT_REALSPACE]
        if direct and not case_insensitive_equal(kind, "kmedoids"):
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

        output_path = self.path.joinpath(f"som_{nx}_{ny}_{n_pcas}_{opp_suffix}.npy")

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

    def _only_windspeed(func):
        @wraps(func)
        def wrapper_decorator(self, *args, **kwargs):
            if (self.varname != "s") or (self.clim_type is not None) or (len(self.levels) > 1):
                print("Only valid for absolute wind speed, single pressure level")
                print(self.varname, self.clim_type, self.levels)
                raise RuntimeError
            value = func(self, *args, **kwargs)

            return value

        return wrapper_decorator

    @_only_windspeed
    def find_jets(self, processes: int = N_WORKERS, chunksize=2) -> Tuple:
        ofile_aj = self.path.joinpath("all_jets.pkl")
        ofile_waj = self.path.joinpath("where_are_jets.npy")
        ofile_ajoa = self.path.joinpath("all_jets_one_array.npy")

        if all([ofile.is_file() for ofile in (ofile_aj, ofile_waj, ofile_ajoa)]):
            all_jets = load_pickle(ofile_aj)
            where_are_jets = np.load(ofile_waj)
            all_jets_one_array = np.load(ofile_ajoa)
            return all_jets, where_are_jets, all_jets_one_array

        all_jets = find_all_jets(self.da, processes, chunksize)
        where_are_jets, all_jets_one_array = all_jets_to_one_array(all_jets)
        save_pickle(all_jets, ofile_aj)
        np.save(ofile_waj, where_are_jets)
        np.save(ofile_ajoa, all_jets_one_array)
        return all_jets, where_are_jets, all_jets_one_array

    @_only_windspeed
    def compute_jet_props(self, processes: int = N_WORKERS, chunksize=2) -> Tuple:
        all_jets, _, _ = self.find_jets(processes, chunksize)
        return compute_all_jet_props(all_jets, self.da, processes, chunksize)

    @_only_windspeed
    def track_jets(self, processes: int = N_WORKERS, chunksize=2) -> Tuple:
        all_jets, where_are_jets, all_jets_one_array = self.find_jets(processes, chunksize)
        ofile_ajot = self.path.joinpath("all_jets_over_time.pkl")
        ofile_flags = self.path.joinpath("flags.npy")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]):
            all_jets_over_time = load_pickle(ofile_ajot)
            flags = np.load(ofile_flags)

            return (
                all_jets,
                where_are_jets,
                all_jets_one_array,
                all_jets_over_time,
                flags,
            )
        with NumbaProgress(total=where_are_jets.shape[0] - 1) as progress:
            all_jets_over_time, flags = track_jets(all_jets_one_array, where_are_jets, progress_proxy=progress)

        save_pickle(all_jets_over_time, ofile_ajot)
        np.save(ofile_flags, flags)
        
        return all_jets, where_are_jets, all_jets_one_array, all_jets_over_time, flags
    
    @_only_windspeed
    def props_as_ds(self, categorize: bool = True, processes: int = N_WORKERS, chunksize=2) -> xr.Dataset:
        _, where_are_jets, _, _, flags = self.track_jets()
        all_props = self.compute_jet_props(processes, chunksize)
        props_as_ds = props_to_ds(all_props, self.time, where_are_jets.shape[1])
        props_as_ds = add_persistence_to_props(props_as_ds, flags)
        if categorize:
            return categorize_ds_jets(props_as_ds)
        return props_as_ds
    