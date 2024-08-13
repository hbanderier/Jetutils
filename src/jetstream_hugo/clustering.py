from typing import Sequence, Tuple, Literal, Mapping, Optional, Callable

from nptyping import NDArray, Float, Shape
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
import scipy.linalg as linalg
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from simpsom import SOMNet

from jetstream_hugo.definitions import (
    coarsen_da,
    revert_zero_one,
    save_pickle,
    load_pickle,
    COMPUTE_KWARGS,
    degcos,
    labels_to_mask,
    to_zero_one,
)

from jetstream_hugo.stats import compute_autocorrs
from jetstream_hugo.data import DataHandler

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


def get_sample_dims(da: xr.DataArray) -> Mapping:
    included = ["time", "member"]
    return {key: da[key].values for key in da.dims if key in included}


def get_feature_dims(da: xr.DataArray) -> Mapping:
    excluded = ["time", "member", "cluster"]
    return {key: da[key].values for key in da.dims if key not in excluded}


def centers_realspace_from_da(centers: NDArray, da: xr.DataArray) -> xr.DataArray:
    return centers_realspace(centers, get_feature_dims(da))


def labels_from_projs(
    X1: NDArray,
    X2: NDArray | None = None,
    cutoff: int | None = None,
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
    labels: list | NDArray | xr.DataArray, da: xr.DataArray | xr.Dataset, coord: str = "cluster"
) -> xr.DataArray:
    if isinstance(labels, xr.DataArray):
        labels = labels.values
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts = counts / float(len(labels))
    dims = list(get_sample_dims(da))
    extra_dims = [coord for coord in da.coords if coord not in da.dims]
    centers = [da.isel(time=(labels == i)).mean(dim=dims) for i in unique_labels]
    for extra_dim in extra_dims:
        for i, center in enumerate(centers):
            centers[i] = center.assign_coords({extra_dim: da[extra_dim].isel(time=(labels == i)).mean(dim=dims)})
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


class Experiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
        inner_norm: str | None = None,
    ) -> None:
        self.data_handler = data_handler
        self.inner_norm = inner_norm
        self.da = self.data_handler.get_da()
        self.path = self.data_handler.get_path()

    def prepare_for_clustering(self) -> Tuple[NDArray, xr.DataArray]:
        try:
            self.da = self.da.load()
        except AttributeError:
            pass
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
        X = da_weighted.values.reshape(self.data_handler.get_flat_shape())
        return X, da_weighted
    
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
        n_pcas: int | None = None,
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
        n_pcas: int | None = None,
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
        sample_dims = self.data_handler.get_sample_dims()
        shape = [len(dim) for dim in sample_dims.values()]
        labels = labels.reshape(shape)
        return xr.DataArray(labels, coords=sample_dims).rename("labels")
    
    def _centers_realspace(self, centers: NDArray):
        feature_dims = self.data_handler.get_feature_dims()
        extra_dims = self.data_handler.get_extra_dims()
        n_pcas_tentative = centers.shape[1]
        pca_path = self._pca_file(n_pcas_tentative)
        if pca_path is not None:
            centers = self.pca_inverse_transform(centers, n_pcas_tentative)
        centers = centers_realspace(centers, feature_dims, extra_dims)
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
        X: NDArray | None = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        All the clustering methods are responsible for producing their centers in pca space and their labels in sample space. This function handles the rest
        """
        if return_type == RAW_PCSPACE:
            _, counts = np.unique(labels, return_counts=True)
            counts = counts / float(len(labels))
            labels = self.labels_as_da(labels)
            coords_centers = {
                "cluster": np.arange(centers.shape[0]),
                "pc": np.arange(centers.shape[1])
            }
            centers = xr.DataArray(centers, coords=coords_centers)
            centers = centers.assign_coords({"ratios": ("cluster", counts)})
        
        elif return_type == RAW_REALSPACE:
            centers = labels_to_centers(labels, self.da, coord="cluster")
            
        elif return_type in [ADJUST, ADJUST_TWOSIDED]:
            projection = np.tensordot(X, centers.T, axes=X.ndim - 1)
            neg = return_type == ADJUST_TWOSIDED
            newlabels = labels_from_projs(projection, neg=neg, adjust=True)
            centers = labels_to_centers(newlabels, self.da, coord="cluster")
            
        else:
            print("Wrong return specifier")
            raise ValueError

        return centers, labels

    def do_kmeans(
        self,
        n_clu: int,
        n_pcas: int,
        return_type: int = RAW_REALSPACE,
    ) -> str | Tuple[xr.DataArray, xr.DataArray, str]:
        X, _ = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        results = KMeans(n_clu)

        results_path = self.path.joinpath(f"k_{n_clu}_{n_pcas}.pkl")
        if results_path.is_file():
            results = load_pickle(results_path)
        else:
            logging.debug(f"Fitting KMeans clustering with {n_clu} clusters")
            results = results.fit(X)
            save_pickle(results, results_path)
            
        centers = results.cluster_centers_
        labels = results.labels_
        
        return self._cluster_output(centers, labels, return_type, X)
    
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
        n_pcas: int | None = None,
        lag_max: int = 90,
        type: int = 1,
        return_realspace: bool = False,
    ) -> Tuple[Path, dict]:
        if type not in [1, 2]:
            raise ValueError(f"Wrong OPP type, pick 1 or 2")
        X, _ = self.prepare_for_clustering()
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
        X, _ = self.prepare_for_clustering()
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
        return self._cluster_output(OPPs, labels, return_type, X)

    def som_cluster(
        self,
        nx: int,
        ny: int,
        n_pcas: int = 0,
        PBC: bool = True,
        metric: str = "ssim",
        return_type: int = RAW_REALSPACE,
        force: bool = False,
        train_kwargs: dict | None = None,
        **kwargs,
    ) -> Tuple[SOMNet, xr.DataArray, NDArray]:
        pbc_flag = "_pbc" if PBC else ""

        if train_kwargs is None:
            train_kwargs = {}
        X, da_weighted = self.prepare_for_clustering()
        if metric in ["ssim", "euclidean"]:
            da_weighted = coarsen_da(da_weighted, 1.5)
            X, Xmin, Xmax = to_zero_one(da_weighted.values)
            output_path = self.path.joinpath(f"som_{nx}_{ny}{pbc_flag}_{metric}.npy")
        else:
            X = self.pca_transform(X, n_pcas)
            output_path = self.path.joinpath(f"som_{nx}_{ny}{pbc_flag}_{n_pcas}.npy")
        init = "random" if np.prod(X.shape[1:]) > 1000 else "pca"
        if output_path.is_file() and not force:
            net = SOMNet(nx, ny, X, metric=metric, PBC=PBC, load_file=output_path.as_posix())
        else:
            net = SOMNet(
                nx,
                ny,
                X,
                PBC=PBC,
                metric=metric,
                init=init,
                **kwargs,
            )
            net.train(**train_kwargs)
            net.save_map(output_path.as_posix())

        labels = net.bmus
        if metric in ["euclidean", "ssim"]: 
            weights = revert_zero_one(net.weights, Xmin, Xmax)
        else:
            weights = net.weights

        return net, *self._cluster_output(weights, labels, return_type)
