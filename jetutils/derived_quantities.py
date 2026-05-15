import numpy as np
import dask.array as darr
import xarray as xr
from jetutils.definitions import RADIUS, OMEGA, degcos, degsin


def convolve_dask(in1, in2, mode="full", method="fft", axes=None):
    """
    I did not write this. Faster scipy.convolve using dask.

    Parameters
    ----------
    in1 : _type_
        _description_
    in2 : _type_
        _description_
    mode : str, optional
        _description_, by default "full"
    method : str, optional
        _description_, by default "fft"
    axes : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """
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


def find_axis(ds: xr.DataArray | xr.Dataset, name: str):
    if isinstance(ds, xr.Dataset):
        ds = ds[list(ds.data_vars)[0]]
    return np.where(np.array(ds.dims) == name)[0].item()


def setup_derivatives(ds: xr.Dataset):
    lon, lat = ds["lon"].values, ds["lat"].values
    xlon, ylat = np.meshgrid(lon, lat)

    dlaty, _ = np.gradient(ylat)
    _, dlonx = np.gradient(xlon)

    dx = RADIUS * np.radians(dlonx) * degcos(ylat)
    dy = RADIUS * np.radians(dlaty) * degcos(ylat)
    
    axis_x = find_axis(ds, "lon")
    axis_y = find_axis(ds, "lat")
    
    def this_expand_dims(arr):
        return np.expand_dims(arr, tuple(np.arange(min(axis_x, axis_y))))
    return dx, dy, axis_x, axis_y, this_expand_dims, degcos(ylat)


def compute_norm_derivative(ds: xr.Dataset, of: str = "s"):
    dx, dy, axis_x, axis_y, this_expand_dims, coslat = setup_derivatives(ds)
    u = ds["u"]
    v = ds["v"]
    s = np.hypot(u, v)
    da = ds[of]
    
    ddady = da.copy(data=np.gradient(da * this_expand_dims(coslat), axis=axis_y)) / this_expand_dims(dy)
    ddadx = da.copy(data=np.gradient(da, axis=axis_x)) / this_expand_dims(dx)
    
    return (- u * ddady + v * ddadx) / s


def compute_relative_vorticity(ds: xr.Dataset):
    dx, dy, axis_x, axis_y, this_expand_dims, coslat = setup_derivatives(ds)
    
    dvdx = ds["v"].copy(data=np.gradient(ds["v"], axis=axis_x)) / this_expand_dims(dx)
    dudy = ds["u"].copy(data=np.gradient(ds["u"] * this_expand_dims(coslat), axis=axis_y)) / this_expand_dims(dy)
    return dvdx - dudy


def compute_absolute_vorticity(ds: xr.Dataset):
    _, _, _, _, this_expand_dims, _ = setup_derivatives(ds)
    
    lat = ds["lat"].values
    zeta = compute_relative_vorticity(ds)
    f = 2 * OMEGA * degsin(lat)
    return this_expand_dims(f[:, None]) + zeta


def compute_2d_conv(ds: xr.Dataset, u: str | None = None, v: str | None = None):
    dx, dy, axis_x, axis_y, this_expand_dims, coslat = setup_derivatives(ds)
    
    if u is not None:
        dudx = ds[u].copy(data=np.gradient(ds[u], axis=axis_x)) / this_expand_dims(dx)
    else:
        da = ds[list(ds.data_vars)[0]]
        dudx = da.copy(data=np.zeros(da.shape))
    if v is not None:
        dvdy = ds[v].copy(data=np.gradient(ds[v] * this_expand_dims(coslat), axis=axis_y)) / this_expand_dims(dy)
    else:
        da = ds[list(ds.data_vars)[0]]
        dvdy = da.copy(data=np.zeros(da.shape))
    return dudx + dvdy


def compute_emf_2d_conv(ds: xr.Dataset):
    dx, dy, axis_x, axis_y, this_expand_dims, coslat = setup_derivatives(ds)
    
    e1 = 0.5 * (ds["vp"] ** 2 - ds["up"] ** 2)
    e2 = - ds["up"] * ds["vp"]
    de1dx = ds["up"].copy(data=np.gradient(e1, axis=axis_x)) / this_expand_dims(dx)
    de2dy = ds["vp"].copy(data=np.gradient(e2 * this_expand_dims(coslat), axis=axis_y)) / this_expand_dims(dy)
    return de1dx + de2dy