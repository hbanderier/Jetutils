from typing import Sequence, Tuple, Literal, Mapping, Optional, Callable

from matplotlib.pylab import norm
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from dask.array import Array as DaArray, from_array
import scipy.linalg as linalg
from scipy.optimize import minimize

from dask_ml.decomposition import PCA as da_PCA
from dask_ml.cluster import KMeans as da_KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# from simpsom import SOMNet
from xpysom_dask.xpysom import XPySom

from jetstream_hugo.definitions import (
    coarsen_da,
    save_pickle,
    load_pickle,
    COMPUTE_KWARGS,
    degcos,
    labels_to_mask,
    normalize,
    revert_normalize,
    compute,
)

from jetstream_hugo.stats import compute_autocorrs
from jetstream_hugo.data import DataHandler, determine_feature_dims, determine_sample_dims

RAW_REALSPACE: int = 0
RAW_PCSPACE: int = 1
ADJUST: int = 2
ADJUST_TWOSIDED: int = 3


def time_mask(time_da: xr.DataArray, filename: str) -> np.ndarray:
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


def centers_realspace(centers: np.ndarray, feature_dims: Mapping) -> xr.DataArray:
    coords = {"cluster": np.arange(centers.shape[0])} | feature_dims
    shape = [len(coord) for coord in coords.values()]
    return xr.DataArray(centers.reshape(shape), coords=coords)


def centers_realspace_from_da(centers: np.ndarray, da: xr.DataArray) -> xr.DataArray:
    return centers_realspace(centers, determine_feature_dims(da))


def labels_from_projs(
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    cutoff: int | None = None,
    neg: bool = True,
    adjust: bool = True,
) -> np.ndarray:
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
    labels: xr.DataArray,
    da: xr.DataArray | xr.Dataset,
    expected_nclu: int | None = None,
    coord: str = "cluster",
) -> xr.DataArray:
    if expected_nclu is not None:
        unique_labels = np.arange(expected_nclu)
        counts = np.zeros(expected_nclu)
        unique_labels_, counts_ = np.unique(labels, return_counts=True)
        counts[unique_labels_] = counts_
    else:
        unique_labels, counts = np.unique(labels, return_counts=True)
    counts = counts / float(np.prod(labels.shape))
    dims = list(determine_sample_dims(da))
    extra_dims = [coord for coord in da.coords if (coord not in da.dims) and ("time" in da[coord].dims)]
    centers = [
        compute(da.where(labels == i).mean(dim=dims)) for i in tqdm(unique_labels)
    ]
    for extra_dim in extra_dims:
        for i, center in enumerate(centers):
            centers[i] = center.assign_coords(
                {extra_dim: da[extra_dim].isel(time=(labels == i)).mean(dim=dims)}
            )
    centers = xr.concat(centers, dim=coord)
    centers = centers.assign_coords(
        {"ratio": (coord, counts), "label": (coord, unique_labels)}
    )
    return centers.set_xindex("label")


def timeseries_on_map(timeseries: np.ndarray, labels: list | np.ndarray):
    timeseries = np.atleast_2d(timeseries)
    mask = labels_to_mask(labels)
    return np.asarray(
        [[np.nanmean(timeseries_[mas]) for mas in mask.T] for timeseries_ in timeseries]
    )


class Experiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
    ) -> None:
        self.data_handler = data_handler
        self.da = self.data_handler.da
        self.path = self.data_handler.get_path()
        
    def load_da(self, **kwargs):
        self.da = compute(self.da, **kwargs)

    def get_norm_da(self):
        norm_path = self.path.joinpath(f"norm.nc")
        if norm_path.is_file():
            return xr.open_dataarray(norm_path)

        norm_da = np.sqrt(degcos(self.da.lat))  # lat as da to simplify mult

        norm_da = compute(norm_da, progress_flag=True)
        norm_da.to_netcdf(norm_path)
        return norm_da

    def prepare_for_clustering(self) -> Tuple[np.ndarray | DaArray, xr.DataArray]:
        norm_da = self.get_norm_da()

        da_weighted = self.da * norm_da
        X = da_weighted.data.reshape(self.data_handler.get_flat_shape())
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
        if path is not None and not force:
            return path
        logging.debug(f"Computing {n_pcas} pcas")
        X, _ = self.prepare_for_clustering()
        pca_path = self.path.joinpath(f"pca_{n_pcas}.pkl")
        results = PCA(n_components=n_pcas, whiten=False).fit(X)
        results = compute(results, progress_flag=True)
        save_pickle(results, pca_path)
        return pca_path

    def pca_transform(
        self,
        X: np.ndarray | DaArray,
        n_pcas: int | None = None,
        compute: bool = False,
    ) -> np.ndarray:
        if n_pcas is None:
            return X
        transformed_file = self.path.joinpath(f"pca_{n_pcas}.npy")
        if transformed_file.is_file():
            return np.load(transformed_file)
        pca_path = self.compute_pcas(n_pcas)
        pca_results = load_pickle(pca_path)
        X = pca_results.transform(X)[:, :n_pcas]
        if not compute:
            return X
        X = compute(X, progress_flag=True)
        np.save(transformed_file, X)
        return X

    def pca_inverse_transform(
        self,
        X: np.ndarray | DaArray,
        n_pcas: int | None = None,
        compute: bool = False,
    ) -> np.ndarray:
        if n_pcas is None:
            return X
        pca_path = self.compute_pcas(n_pcas)
        pca_results = load_pickle(pca_path)
        diff_n_pcas = pca_results.n_components - X.shape[1]
        X = np.pad(X, [[0, 0], [0, diff_n_pcas]])
        X = pca_results.inverse_transform(X)
        if not compute:
            return X
        return compute(X, progress_flag=True)

    def labels_as_da(self, labels: np.ndarray) -> xr.DataArray:
        sample_dims = determine_sample_dims(self.da)
        shape = [len(dim) for dim in sample_dims.values()]
        labels = labels.reshape(shape)
        return xr.DataArray(labels, coords=sample_dims).rename("labels")

    def _centers_realspace(self, centers: np.ndarray):
        feature_dims = self.data_handler.get_feature_dims()
        extra_dims = self.data_handler.get_extra_dims()
        n_pcas_tentative = centers.shape[1]
        pca_path = self._pca_file(n_pcas_tentative)
        if pca_path is not None:
            centers = self.pca_inverse_transform(
                centers, n_pcas_tentative, compute=True
            )
        centers = centers_realspace(centers, feature_dims, extra_dims)
        norm_path = self.path.joinpath(f"norm.nc")
        norm_da = xr.open_dataarray(norm_path.as_posix())
        if "time" in norm_da.dims:
            norm_da = norm_da.mean(dim="time")
        return centers / norm_da

    def _cluster_output(
        self,
        centers: np.ndarray,
        labels: np.ndarray,
        return_type: int = RAW_REALSPACE,
        X: np.ndarray | None = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        All the clustering methods are responsible for producing their centers in pca space and their labels in sample space. This function handles the rest
        """
        n_clu = centers.shape[0]
        counts = np.zeros(n_clu)
        if return_type == RAW_PCSPACE:
            clusters_, counts_ = np.unique(labels, return_counts=True)
            counts[clusters_] = counts_
            counts = counts / float(len(labels))
            coords_centers = {
                "cluster": np.arange(centers.shape[0]),
                "pc": np.arange(centers.shape[1]),
            }
            centers = xr.DataArray(centers, coords=coords_centers)
            centers = centers.assign_coords({"ratios": ("cluster", counts)})
            labels = self.labels_as_da(labels)

        elif return_type == RAW_REALSPACE:
            labels = self.labels_as_da(labels)
            centers = labels_to_centers(labels, self.da, expected_nclu=n_clu, coord="cluster")

        elif return_type in [ADJUST, ADJUST_TWOSIDED]:
            projection = np.tensordot(X, centers.T, axes=X.ndim - 1)
            neg = return_type == ADJUST_TWOSIDED
            newlabels = labels_from_projs(projection, neg=neg, adjust=True)
            centers = labels_to_centers(newlabels, self.da, coord="cluster")
            labels = self.labels_as_da(labels)

        else:
            print("Wrong return specifier")
            raise ValueError
        
        return centers, labels

    def do_kmeans(
        self,
        n_clu: int,
        n_pcas: int,
        weigh_grams: bool = False,
        return_type: int = RAW_REALSPACE,
    ) -> str | Tuple[xr.DataArray, xr.DataArray, str]:
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        if weigh_grams:
            roll_std = da.rolling({"time": 30 * 4}, min_periods=3, center=False).std()
            roll_std = compute(roll_std.chunk({"time": -1, "lon": 1}).mean(["lon", "lat"]).interpolate_na("time", "nearest", fill_value="extrapolate"))
            X = X / roll_std.values[:, None]
            suffix = "_grams"
        else:
            suffix = ""
        results = KMeans(n_clu, n_init=20)

        results_path = self.path.joinpath(f"k_{n_clu}_{n_pcas}{suffix}.pkl")
        if results_path.is_file():
            results = load_pickle(results_path)
        else:
            logging.debug(f"Fitting KMeans clustering with {n_clu} clusters")
            results = results.fit(X)
            save_pickle(results, results_path)

        centers = results.cluster_centers_
        labels = results.labels_

        return self._cluster_output(centers, labels, return_type, X)

    def som_cluster(
        self,
        nx: int,
        ny: int,
        n_pcas: int = 0,
        PBC: bool = True,
        activation_distance: str = "euclidean",
        return_type: int = RAW_REALSPACE,
        force: bool = False,
        train_kwargs: dict | None = None,
        **kwargs,
    ) -> Tuple[XPySom, xr.DataArray, np.ndarray]:
        pbc_flag = "_pbc" if PBC else ""
        net = XPySom(
            nx,
            ny,
            PBC=PBC,
            activation_distance=activation_distance,
            **kwargs,
        )
        if n_pcas:
            output_file_stem = f"som_{nx}_{ny}{pbc_flag}_{n_pcas}"
        else:
            output_file_stem = f"som_{nx}_{ny}{pbc_flag}_{activation_distance}"
        output_path_weights = self.path.joinpath(f"{output_file_stem}.npy")
        output_path_centers = self.path.joinpath(f"centers_{output_file_stem}.nc")
        output_path_labels = self.path.joinpath(f"labels_{output_file_stem}.nc")
        if all([ofile.is_file() for ofile in [output_path_weights, output_path_labels, output_path_centers]]) and not force:
            net.load_weights(output_path_weights)
            centers = xr.open_dataarray(output_path_centers)
            labels = xr.open_dataarray(output_path_labels)
            net.latest_bmus = labels.values
            return net, centers, labels
        if train_kwargs is None:
            train_kwargs = {"num_epochs": 15}
        train_kwargs["out_path"] = output_path_weights
        X, da_weighted = self.prepare_for_clustering()
        if n_pcas:
            X = self.pca_transform(X, n_pcas)
        else:
            if (da_weighted.lon[1] - da_weighted.lon[0]).item() < 1:
                da_weighted = coarsen_da(da_weighted, 1.5)
                X = da_weighted.data.reshape(self.data_handler.get_flat_shape()[0], -1)
            X, meanX, stX = normalize(X)
        if not force and output_path_weights.is_file():
            net.load_weights(output_path_weights)
        else:
            X = compute(X, progress_flag=True)
            net.train(X, **train_kwargs)

        labels = net.predict(X)
        if n_pcas:
            weights = net.weights
        else:
            weights = revert_normalize(net.weights, meanX, stX)
            
        X = compute(X, progress_flag=True)
        centers, labels = self._cluster_output(weights, labels, return_type, X)
        centers.to_netcdf(output_path_centers)
        labels.to_netcdf(output_path_labels)
        return net, centers, labels
    
    def project_on_other_som(
        self,
        other_exp: Path,  
        **kwargs,
    ) -> Tuple[XPySom, xr.DataArray, np.ndarray]:
        nx, ny = kwargs["nx"], kwargs["ny"]
        pbc = "pbc" in kwargs and kwargs["pbc"]
        pbc_flag = "_pbc" if pbc else ""
        net, centers, labels = other_exp.som_cluster(**kwargs)
        
        if "n_pcas" in kwargs and kwargs["n_pcas"]:
            n_pcas = kwargs["n_pcas"]
            output_file = f"othersom_labels_{nx}_{ny}{pbc_flag}_{n_pcas}.nc"
        else:
            activation_distance = kwargs["activation_distance"]
            output_file = f"othersom_labels_{nx}_{ny}{pbc_flag}_{activation_distance}.nc"
        output_file = self.path.joinpath(output_file)
        if output_file.is_file():
            return net, centers, xr.open_dataarray(output_file)
        other_da = other_exp.da
        _, da_weighted = self.prepare_for_clustering()
                
        if "n_pcas" in kwargs and kwargs["n_pcas"]:
            X = self.pca_transform(X, kwargs["n_pcas"])
        else:
            if (other_da.lon[1] - other_da.lon[0]).item() < 1:
                other_da = coarsen_da(other_da, 1.5)
            da_weighted = da_weighted.interp(
                lon=other_da.lon.values, 
                lat=other_da.lat.values,
                kwargs={"fill_value": "extrapolate"}
            )
            X = normalize(
                da_weighted.data.reshape(
                    np.prod(da_weighted.shape[:2]), 
                    np.prod(da_weighted.shape[2:])
                )
            )[0]
        
        sample_shape = [len(co) for co in self.data_handler.sample_dims.values()]
        X = compute(X, progress_flag=True)
        labels = net.predict(X).reshape(sample_shape)
        labels = xr.DataArray(labels, coords=self.data_handler.sample_dims)
        labels.attrs["som_from_exp"] = other_exp.path.as_posix()
        for key, val in kwargs.items():
            if isinstance(val, dict):
                continue
            if isinstance(val, bool):
                val = int(val)
            labels.attrs[key] = val
        labels.to_netcdf(output_file)
        return net, centers, labels, X

    # TODO maybe: OPPs are untested with Dask input
    def _compute_opps_T1(
        self,
        X: np.ndarray,
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

    def _compute_opps_T2(self, X: np.ndarray, lag_max: int) -> dict:
        autocorrs = compute_autocorrs(X, lag_max)
        C0sqrt = linalg.sqrtm(autocorrs[0])
        C0minushalf = linalg.inv(C0sqrt)
        basis = linalg.orth(C0minushalf)

        def minus_T2(x) -> float:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            return -2 * np.trapz(factor1**2) / normxsq**2

        def minus_T2_gradient(x) -> np.ndarray:
            normxsq = linalg.norm(x) ** 2
            factor1 = x.T @ C0minushalf @ autocorrs @ C0minushalf @ x
            factor2 = (
                C0minushalf @ (autocorrs + autocorrs.transpose((0, 2, 1))) @ C0minushalf
            ) @ x
            numerator = 4 * np.trapz(factor1[:, None] * factor2, axis=0)
            return -numerator / normxsq**2 - 4 * minus_T2(x) * x / normxsq**3

        def norm0(x) -> float:
            return 10 - linalg.norm(x) ** 2

        def jac_norm0(x) -> np.ndarray:
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
        type_: int = 1,
        return_realspace: bool = False,
    ) -> Tuple[Path, dict]:
        if type_ not in [1, 2]:
            raise ValueError(f"Wrong OPP type, pick 1 or 2")
        X, _ = self.prepare_for_clustering()
        if n_pcas:
            X = self.pca_transform(X, n_pcas)
        X = X.reshape((X.shape[0], -1))
        n_pcas = X.shape[1]
        opp_path: Path = self.path.joinpath(f"opp_{n_pcas}_T{type_}.pkl")
        results = None
        if not opp_path.is_file():
            if type_ == 1:
                logging.debug("Computing T1 OPPs")
                results = self._compute_opps_T1(X, lag_max)
            if type_ == 2:
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
            OPPs[::2] = self.compute_opps(n_pcas, type_=1)[1]["OPPs"][:n_clu]
            OPPs[1::2] = self.compute_opps(n_pcas, type_=2)[1]["OPPs"][:n_clu]
            X1 = self.opp_transform(X, n_pcas, cutoff=n_clu, type=1)
            X2 = self.opp_transform(X, n_pcas, cutoff=n_clu, type=2)
        elif type in [1, 2]:
            OPPs = self.compute_opps(n_pcas, type_=type)[1]["OPPs"][:n_clu]
            X1 = self.opp_transform(X, n_pcas, cutoff=n_clu, type=type)
            X2 = None

        labels = labels_from_projs(X1, X2, cutoff=n_clu, neg=False, adjust=False)
        return self._cluster_output(OPPs, labels, return_type, X)
