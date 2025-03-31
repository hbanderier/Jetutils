# coding: utf-8
from typing import Tuple, Mapping

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm
from dask.array import Array as DaArray
import scipy.linalg as linalg
from scipy.optimize import minimize

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from simpsom_dask.simpsom import Simpsom

from .definitions import (
    coarsen_da,
    save_pickle,
    load_pickle,
    degcos,
    labels_to_mask,
    normalize,
    revert_normalize,
    compute,
)

from .stats import compute_autocorrs
from .data import (
    DataHandler,
    determine_feature_dims,
    open_dataarray,
    determine_sample_dims,
    to_netcdf,
)

RAW_REALSPACE: int = 0
RAW_PCSPACE: int = 1
ADJUST: int = 2
ADJUST_TWOSIDED: int = 3


def centers_realspace(centers: np.ndarray, feature_dims: Mapping) -> xr.DataArray:
    """
    Transforms raw cluster centers, typically the output of sklearn clustering algorithms, to easily plottable xarray `DataArray`.
    
    Parameters
    ----------
    centers : array of size (n_samples, n_features)
        Raw cluster centers
    
    feature_dims : Mapping
        Dictionnary from whose keys are dimension names and values are the correponding coordinates  as `ndarray`'s. Typically lon and lat.
        
    Returns
    -------
    centers : DataArray
        Cluster centers reshaped to fit `feature_dims` and with named dimensions.
    """
    coords = {"cluster": np.arange(centers.shape[0])} | feature_dims
    shape = [len(coord) for coord in coords.values()]
    return xr.DataArray(centers.reshape(shape), coords=coords)


def centers_realspace_from_da(
    centers: np.ndarray, da: xr.DataArray | xr.Dataset
) -> xr.DataArray:    
    """
    Transforms raw cluster centers, typically the output of sklearn clustering algorithms, to
    easily plottable xarray `DataArray`, taking the coordinates from another `DataArray`
    
    Parameters
    ----------
    centers : array of size (n_samples, n_features)
        Raw cluster centers
        
    da : xr.DataArray or xr.Dataset
        DataArray or Dataset from which to infer the feature dimensions
        
    Returns
    -------
    centers : DataArray
        Cluster centers reshaped to fit the feature dimension of `da` and with named dimensions.
    """
    return centers_realspace(centers, determine_feature_dims(da))


def labels_from_projs(
    X1: np.ndarray,
    X2: np.ndarray | None = None,
    cutoff: int | None = None,
    neg: bool = True,
    adjust: bool = True,
) -> np.ndarray:
    """
    Generates hard assignments from one or two collections of projection timeseries onto patterns using configurable heuristics.
    Typically, the timeseries are projections on patterns like PCs or OPPs, where the first axis is time and
    the second one is patterns.
    The output is a timeseries `y` of winner patterns at each timestep, in the most basic form `y=np.argmax(X1, axis=1)`.
    If `X2` is not present, `y` can get values from `0` to `n_patterns - 1` if `neg=False` and `adjust=False`.

    With `neg=True`, the condition becomes `y=np.argmax(np.abs(X1), axis=1)`. That is, the largest absolute projection wins, whether positive or negative.
    For example, the NAO pattern wins if the timestep resembles its positive or its negative phase. If the largest absolute projection is negative, the label is also negative.
    Therefore, with `neg=True`, the output can have output from `-n_patterns + 1` to `n_patterns - 1`

    With `adjust=True`, the assignments are only set to a label if the projection is above one standard deviation of projections, and 0 otherwise. In this case,
    `y` can take values from `0` to `n_patterns` if `neg=False` and `-n_patterns` to `n_patterns` if `neg=True`.

    If two timeseries are present and are of equal size (n_time, n_patterns), a large collection of timeseries is created with projections of `X1`
    in the even positions and the projections of `X2` in the odd positions.

    Parameters
    ----------
    X1 : array of shape (n_time, n_patterns)
        Projections on many patterns.
        
    X2 : array of shape (n_time, n_patterns), optional
        Additional projections on many patterns. If present, will be interleaved into `X1` before proceeding, with projections of `X1` in the even positions and the projections of `X2` in the odd positions. By default None.
        
    cutoff : int, optional
        Limits how many patterns to to perform the assignment on. All patterns if left to `None`, the default. By default all patterns
        
    neg : bool, optional
        If `False`, the timestep is assigned to a pattern if it has the highest projection of all the patterns for this timestep, if `True` the timestep is assigned to a pattern if it has the highest *absolute* projection of all the patterns for this timestep. By default True
        
    adjust : bool, optional
        Whether assignments are only valid if the projection is larger than one standard deviation of projections. By default True

    Returns
    -------
    labels : np.ndarray
        Integer ndarray of shape (n_time), assignments onto each pattern based on different rules.
    """    
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
    dim_name: str = "cluster",
) -> xr.DataArray:
    """
    Generates cluster centers by averaging the elements of `da` belonging to each cluster.

    Parameters
    ----------
    labels : xr.DataArray
        Array with one or several dimensions corresponding to the *sample* dimensions of the clustering that created them. Typically (time) or (member, time). Assignments from sample points (e.g. timesteps) to a cluster.
        
    da : xr.DataArray | xr.Dataset
        Array from which to create real space cluster centers. Does not have to be the data on which the clustering was performed.
        
    expected_nclu : int, optional
        Can be useful if not all clusters are present in `labels`. If `None`, the default, possible clusters are all clusters present in `labels`. If present, it is instead `np.arange(n_clu)`. By default None.
        
    dim_name : str, optional
        Name of the DataArray dimension name along the clusters, by default "cluster"

    Returns
    -------
    centers : Same as `da`
        Cluster center, dimensions are `dict(dim_name=clusters, **get_feature_dims(da))`.
    """    
    if expected_nclu is not None:
        unique_labels = np.arange(expected_nclu)
        counts = np.zeros(expected_nclu)
        unique_labels_, counts_ = np.unique(labels, return_counts=True)
        counts[unique_labels_] = counts_
    else:
        unique_labels, counts = np.unique(labels, return_counts=True)
    counts = counts / float(np.prod(labels.shape))
    if "megatime" in da.dims:
        centers = [
            compute(da.sel(megatime=labels == i).mean("megatime"))
            for i in tqdm(unique_labels)
        ]
    else:
        dims = list(determine_sample_dims(da))
        extra_dims = [
            coord
            for coord in da.coords
            if (coord not in da.dims) and ("time" in da[coord].dims)
        ]
        if len(dims) == 1:
            dim = dims[0]
            print(dim)
            centers = [
                compute(da.sel(**{dim: labels[dim][labels == i]}).mean(dim=dims))
                for i in tqdm(unique_labels)
            ]
        else:
            centers = [
                compute(da.where(labels == i).mean(dim=dims))
                for i in tqdm(unique_labels)
            ]
        for extra_dim in extra_dims:
            for i, center in enumerate(centers):
                centers[i] = center.assign_coords(
                    {extra_dim: da[extra_dim].isel(time=(labels == i)).mean(dim=dims)}
                )
    centers = xr.concat(centers, dim=dim_name)
    centers = centers.assign_coords(
        {"ratio": (dim_name, counts), "label": (dim_name, unique_labels)}
    )
    return centers.set_xindex("label")


def timeseries_on_map(timeseries: np.ndarray, labels: list | np.ndarray) -> np.ndarray:
    """
    From a timeseries of values and a timeseries of labels, assigning each timestep to a cluster, returns the clusterwise mean of the timeseries

    Parameters
    ----------
    timeseries : np.ndarray
        Any timeseries
        
    labels : list | np.ndarray
        Label assignment, must be of the same length as `timeseries`

    Returns
    -------
    np.ndarray
        _descrMeans of timeseries elements belonging to each cluster. As many elements as there are unique clusters in `labels`iption_
    """    
    timeseries = np.atleast_2d(timeseries)
    mask = labels_to_mask(labels)
    return np.asarray(
        [[np.nanmean(timeseries_[mas]) for mas in mask.T] for timeseries_ in timeseries]
    )


class Experiment(object):
    """
    Worker class for all the different clustering methods, handling various clustering tasks and pre- and post-processing.

    Attributes
    -------
    data_handler : DataHandler
        Provides the underlying `DataArray` and path in which to store results
    
    da : xr.DataArray
        shortcut to `self.data_handler.da`
        
    path : Path
        shortcut to `self.data_handler.path`
    """    
    def __init__(
        self,
        data_handler: DataHandler,
    ) -> None:
        """
        Creates instance of Experiment
        
        Parameters
        ----------
        data_handler: DataHandler 
            Provides underlying `DataArray` and path in which to store results
        """
        self.data_handler = data_handler
        self.da = self.data_handler.da
        self.path = self.data_handler.get_path()

    def load_da(self, **kwargs):
        """
        Coerces this Experiment's `DataArray` into memory.
        
        Parameters
        ----------
        kwargs
            Keyword arguments that get passed to `compute()`.
        """
        self.da = compute(self.da, **kwargs)

    def get_norm_da(self):
        """
        Computes, stores and returns the normalization factor 
        
        Returns
        -------
        xr.DataArray
            normalization factor, computed as the square root of the latitude.
        """
        norm_path = self.path.joinpath("norm.nc")
        if norm_path.is_file():
            return open_dataarray(norm_path)

        norm_da = np.sqrt(degcos(self.da.lat))  # lat as da to simplify mult

        norm_da = compute(norm_da, progress_flag=True)
        to_netcdf(norm_da, norm_path)
        return norm_da

    def prepare_for_clustering(self) -> Tuple[np.ndarray | DaArray, xr.DataArray]:
        """
        Normalizes and reshapes original data into a form ready for transformation and / or clustering tasks

        Returns
        -------
        np.ndarray shape (n_samples, n_features)
            Normalized and reshaped version of original data.
        
        xr.DataArray
            Normalized but not reshaped version of the original data.
        """
        norm_da = self.get_norm_da()

        da_weighted = self.da * norm_da
        X = da_weighted.data.reshape(self.data_handler.get_flat_shape())
        return X, da_weighted

    def _pca_file(self, n_pcas: int) -> str | None:
        """
        Tries to find the `.pkl` file containing the `sklearn.PCA` object that was trained on this Experiment's data with at least `n_pca` components

        Parameters
        ----------
        n_pcas : int
            Number of components
        
        Returns
        -------
        str or None
            posix path to file if it exsits, otherwise `None`
        """
        potential_paths = list(self.path.glob("pca_*.pkl"))
        potential_paths = {
            path: int(path.stem.split("_")[1]) for path in potential_paths
        }
        for key, value in potential_paths.items():
            if value >= n_pcas:
                return key
        return None

    def compute_pcas(self, n_pcas: int, force: bool = False) -> str:
        """
        Preprocess own data, trains scikit-learn `PCA` object, saves it and returns path to it. If a fitting PCA object is already stored, don't train and return path to it instead.
        
        Parameters
        ----------
        n_pcas : int
            Number of components
            
        force : bool, optional
            Trains PCA object even if a fitting one exists, by default False

        Returns
        -------
        str
            Posix path to `.pkl` file 
        """
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
        """
        Potentially fits `PCA` object on this object's own data, and transforms input data with trained `PCA` object.

        Parameters
        ----------
        X : np.ndarray | DaArray
            Data to transform, not necessarily the one on which `PCA` was trained.
            
        n_pcas : int | None, optional
            Number of components. If `None`, returns `X` unmodified. By default None
            
        compute : bool, optional
            If input was a Dask Array, whether or not to coerce output to memory, by default False

        Returns
        -------
        np.ndarray
            Transformed `X`
        """
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
        """
        Performs inverse PCA transform on `X`, based on PCA trained on this object's data.

        Parameters
        ----------
        X : np.ndarray | dask.Array
            Data to inverse transform, not necessarily the one on which `PCA` was trained.
            
        n_pcas : int | None, optional
            Number of components. If `None`, returns `X` unmodified. By default None
            
        compute : bool, optional
            If input was a Dask Array, whether or not to coerce output to memory, by default False

        Returns
        -------
        np.ndarray
            Inverse transformed `X`
        """        
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
        """
        Transforms a labels array into a `DataArray` with named dimensions, inferred from this object's data's sample dimensions.

        Parameters
        ----------
        labels : np.ndarray
            Labels, output from clustering methods

        Returns
        -------
        xr.DataArray
            `labels` with named sample dimensions.
        """        
        sample_dims = determine_sample_dims(self.da)
        shape = [len(dim) for dim in sample_dims.values()]
        labels = labels.reshape(shape)
        return xr.DataArray(labels, coords=sample_dims).rename("labels")

    def _centers_realspace(self, centers: np.ndarray) -> xr.DataArray:
        """
        Transforms the centers of clusters, as directly output by the various clustering methods, into the same space as this object's data.
        Tries to guess whether it was PCA transformed by checking for a PCA file with `n_pcas=centers.shape[1]`
        Also undoes the normalization.

        Parameters
        ----------
        centers : np.ndarray
            Raw cluster centers

        Returns
        -------
        xr.DataArray
            `centers` transformed back to a `DataArray` in the same space as this object's data.
        """        
        feature_dims = self.data_handler.get_feature_dims()
        extra_dims = self.data_handler.get_extra_dims()
        n_pcas_tentative = centers.shape[1]
        pca_path = self._pca_file(n_pcas_tentative)
        if pca_path is not None:
            centers = self.pca_inverse_transform(
                centers, n_pcas_tentative, compute=True
            )
        centers = centers_realspace(centers, feature_dims, extra_dims)
        norm_path = self.path.joinpath("norm.nc")
        norm_da = open_dataarray(norm_path.as_posix())
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
        All the clustering methods are responsible for producing their centers, potentially in in pca space and their labels in sample space. This function handles the rest. Potentially transforms `centers`, and turns both `centers` and `labels` into `DataArray`s with appropriate coordinates inferred from the original data.
        
        Parameters
        ----------
        centers : array of shape (n_centers, n_features)
            Cluster centers, potentially in PC space.
            
        labels : array of shape (n_samples)
            Timeseries Cluster labels.
        
        return_type : int, optional
            four options:
            
            - RAW_REALSPACE: the default, transforms centers into the same space as this object's data
            
            - RAW_PCSPACE: leaves centers in original training space
            
            - ADJUST: projects the training data onto the original clusters, and re-compute the cluster labels based on the adjusted projection winners. A sample is assigned to the "0 cluster" if the maximum projection on the cluster centers is below one standard deviation of all projections. See `labels_from_projs` for more details.

            - ADJUST_TWOSIDED: same as `ADJUST` but labels are computed with the *absolute values* of the projections of `X` onto the centers. This allows for negative cluster assignments: `labels[0]=-1` means `X[0]` had the highest absolute projection on the first cluster center, but this projection was negative. This is rarely useful.
            
        X : `ndarray` of shape (n_samples, n_features), optional
            Original training data, only necessary if the projections need to be recomputed, i.e. if `return_type` is either `ADJUST` or `ADJUST_TWOSIDED`.

        Returns
        -------
        centers: DataArray
            Transformed centers, with appropriate coordinates and dimensions
            
        labels: DataArray
            Potentially recomputed labels, with appropriate coordinates and dimensions

        Raises
        ------
        ValueError
            If a wrong return specifier is given
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
            centers = labels_to_centers(
                labels, self.da, expected_nclu=n_clu, dim_name="cluster"
            )

        elif return_type in [ADJUST, ADJUST_TWOSIDED]:
            projection = np.tensordot(X, centers.T, axes=X.ndim - 1)
            neg = return_type == ADJUST_TWOSIDED
            newlabels = labels_from_projs(projection, neg=neg, adjust=True)
            centers = labels_to_centers(newlabels, self.da, dim_name="cluster")
            labels = self.labels_as_da(labels)

        else:
            print("Wrong return specifier")
            raise ValueError

        return centers, labels

    def do_kmeans(
        self,
        n_clu: int,
        n_pcas: int | None = None,
        weigh_grams: bool = False,
        return_type: int = RAW_REALSPACE,
        force: bool = False,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Performs K-means clustering by wrapping the scikit-learn KMeans object, pre- and post-processing this object's data. Stores the underlying trained scikit-learn KMeans object.
        If a fitting KMeans object is already stored, use it instead unless `force=True`

        Parameters
        ----------
        n_clu : int
            Number of k-means cluster
            
        n_pcas : int | None, optional
            Number of principal components. If above 0 and not `None`, transforms the data into PC space, if 0 or None (the default), the data is left in real space.
            
        weigh_grams : bool, optional
            Performs special weighing recommended by Grams et al. 2017, by default False
            
        return_type : int, optional
            How to transform the output centers and labels, by default RAW_REALSPACE
            
        force : bool, optional
            Whether to re-train a KMeans object even if a fitting one is found, by default False

        Returns
        -------
        centers: DataArray
            Transformed centers, with appropriate coordinates and dimensions
            
        labels: DataArray
            Potentially recomputed labels, with appropriate coordinates and dimensions
        """
        output_file_stem = f"kmeans_{n_clu}_{n_pcas}"
        output_path_centers = self.path.joinpath(f"centers_{output_file_stem}.nc")
        output_path_labels = self.path.joinpath(f"labels_{output_file_stem}.nc")
        if (
            all(
                [ofile.is_file() for ofile in [output_path_labels, output_path_centers]]
            )
            and not force
        ):
            centers = open_dataarray(output_path_centers)
            labels = open_dataarray(output_path_labels)
            return centers, labels
        X, da = self.prepare_for_clustering()
        X = self.pca_transform(X, n_pcas)
        if weigh_grams:
            roll_std = da.rolling({"time": 30 * 4}, min_periods=3, center=False).std()
            roll_std = compute(
                roll_std.chunk({"time": -1, "lon": 1})
                .mean(["lon", "lat"])
                .interpolate_na("time", "nearest", fill_value="extrapolate")
            )
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

        centers, labels = self._cluster_output(centers, labels, return_type, X)
        to_netcdf(centers, output_path_centers)
        to_netcdf(labels, output_path_labels)
        return centers, labels

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
    ) -> Tuple[Simpsom, xr.DataArray, np.ndarray]:
        """
        Performs SOM clustering by wrapping the Simpsom object, pre- and post-processing this object's data. Stores the underlying trained object.
        If a fitting Simpsom object is already stored, use it instead unless `force=True`

        Parameters
        ----------
        nx : int
            SOM grid size in the x direction
            
        ny : int
            SOM grid size in the y direction
            
        n_pcas : int, optional
            Number of principal components. If any above 0, transforms the data into PC space, if 0 or None (the default), the data is left in real space.
            
        PBC : bool, optional
            Whether to use Periodic Boundary Conditions in the SOM grid, by default True
            
        activation_distance : str, optional
            SOM real space distance, by default "euclidean"
            
        return_type : int, optional
            How to transform the output centers and labels, by default RAW_REALSPACE
            
        force : bool, optional
            Whether to re-train a Simpsom object even if a fitting one is found, by default False
            
        train_kwargs : dict | None, optional
            arguments passed to `net.train()`, by default None

        Returns
        -------
        net: Simpsom
            Simpsom object.
            
        centers: DataArray
            Transformed centers, with appropriate coordinates and dimensions
            
        labels: DataArray
            Potentially recomputed labels, with appropriate coordinates and dimensions
            
        """        
        pbc_flag = "_pbc" if PBC else ""
        net = Simpsom(
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
        if (
            all(
                [
                    ofile.is_file()
                    for ofile in [
                        output_path_weights,
                        output_path_labels,
                        output_path_centers,
                    ]
                ]
            )
            and not force
        ):
            net.load_weights(output_path_weights)
            centers = open_dataarray(output_path_centers)
            labels = open_dataarray(output_path_labels)
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
        to_netcdf(centers, output_path_centers)
        to_netcdf(labels, output_path_labels)
        return net, centers, labels

    def project_on_other_som(
        self,
        other_exp: "Experiment",
        nx: int,
        ny: int,
        n_pcas: int = 0,
        PBC: bool = True,
        activation_distance: str = "euclidean",
        return_type: int = RAW_REALSPACE,
    ) -> Tuple[Simpsom, xr.DataArray, np.ndarray]:
        """
        Projects this object's data onto a SOM trained by another `Experiment`.

        Parameters
        ----------
        other_exp : Experiment
            _description_
        nx : int
            _description_
        ny : int
            _description_
        n_pcas : int, optional
            _description_, by default 0
        PBC : bool, optional
            _description_, by default True
        activation_distance : str, optional
            _description_, by default "euclidean"
        return_type : int, optional
            _description_, by default RAW_REALSPACE

        Returns
        -------
        net: Simpsom
            Original Simpsom object, whose `latest_bmus` new correspond to this data
            
        centers: DataArray
            SOM centers computed from this data, transformed and with appropriate coordinates and dimensions
            
        labels: DataArray
            Clustering labels corresponding to this data with appropriate coordinates and dimensions
        """        
        pbc_flag = "_pbc" if PBC else ""
        net, _, _ = other_exp.som_cluster(nx=nx, ny=ny, n_pcas=n_pcas, PBC=PBC)

        if n_pcas:
            output_file_stem = f"othersom_{nx}_{ny}{pbc_flag}_{n_pcas}"
        else:
            output_file_stem = f"othersom_{nx}_{ny}{pbc_flag}_{activation_distance}"
        output_path_centers = self.path.joinpath(f"centers_{output_file_stem}.nc")
        output_path_labels = self.path.joinpath(f"labels_{output_file_stem}.nc")
        if all(
            [ofile.is_file() for ofile in [output_path_labels, output_path_centers]]
        ):
            centers = open_dataarray(output_path_centers)
            labels = open_dataarray(output_path_labels)
            net.latest_bmus = labels.values
            return net, centers, labels
        X, da_weighted = self.prepare_for_clustering()

        if n_pcas:
            X = self.pca_transform(X, n_pcas)
        else:
            if (da_weighted.lon[1] - da_weighted.lon[0]).item() < 1:
                da_weighted = coarsen_da(da_weighted, 1.5)
                X = da_weighted.data.reshape(self.data_handler.get_flat_shape()[0], -1)
            X, meanX, stX = normalize(X)

        labels = net.predict(X)
        if n_pcas:
            weights = net.weights
        else:
            weights = revert_normalize(net.weights, meanX, stX)

        X = compute(X, progress_flag=True)
        centers, labels = self._cluster_output(weights, labels, return_type, X)
        to_netcdf(centers, output_path_centers)
        to_netcdf(labels, output_path_labels)
        return net, centers, labels

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
        """
        Compute Optimally Persistent Patters. Deprecated and probably broken.

        Parameters
        ----------
        n_pcas : int | None, optional
            _description_, by default None
        lag_max : int, optional
            _description_, by default 90
        type_ : int, optional
            _description_, by default 1
        return_realspace : bool, optional
            _description_, by default False

        Returns
        -------
        Tuple[Path, dict]
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if type_ not in [1, 2]:
            raise ValueError("Wrong OPP type, pick 1 or 2")
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
        """
        Cluster on type T1, type T2 or both types of Optimally Persistent Patterns. Deprecated and probably broken.

        Parameters
        ----------
        n_clu : int
            _description_
        n_pcas : int
            _description_
        type : int, optional
            _description_, by default 1
        return_type : int, optional
            _description_, by default RAW_REALSPACE

        Returns
        -------
        Tuple[xr.DataArray, xr.DataArray]
            _description_
        """
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
