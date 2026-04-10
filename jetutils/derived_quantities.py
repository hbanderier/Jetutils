import numpy as np
import xarray as xr
from jetutils.definitions import RADIUS, OMEGA, degcos, degsin


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
    dy = RADIUS * np.radians(dlaty)
    
    axis_x = find_axis(ds, "lon")
    axis_y = find_axis(ds, "lat")
    
    def this_expand_dims(arr):
        return np.expand_dims(arr, tuple(np.arange(min(axis_x, axis_y))))
    return dx, dy, axis_x, axis_y, this_expand_dims


def compute_norm_derivative(ds: xr.Dataset, of: str = "s"):
    dx, dy, axis_x, axis_y, this_expand_dims = setup_derivatives(ds)
    u = ds["u"]
    v = ds["v"]
    s = np.hypot(u, v)
    da = ds[of]
    
    ddady = da.copy(data=np.gradient(da, axis=axis_y)) / this_expand_dims(dy)
    ddadx = da.copy(data=np.gradient(da, axis=axis_x)) / this_expand_dims(dx)
    
    return (- u * ddady + v * ddadx) / s


def compute_relative_vorticity(ds: xr.Dataset):
    dx, dy, axis_x, axis_y, this_expand_dims = setup_derivatives(ds)
    
    dvdx = ds["v"].copy(data=np.gradient(ds["v"], axis=axis_x)) / this_expand_dims(dx)
    dudy = ds["u"].copy(data=np.gradient(ds["u"], axis=axis_y)) / this_expand_dims(dy)
    return dvdx - dudy


def compute_absolute_vorticity(ds: xr.Dataset):
    lat = ds["lat"].values
    zeta = compute_relative_vorticity(ds)
    f = 2 * OMEGA * degsin(lat)
    return f[None, :, None] + zeta


def compute_2d_conv(ds: xr.Dataset, u: str | None = None, v: str | None = None):
    dx, dy, axis_x, axis_y, this_expand_dims = setup_derivatives(ds)
    
    if u is not None:
        dudx = ds[u].copy(data=np.gradient(ds[u], axis=axis_x)) / this_expand_dims(dx)
    else:
        da = ds[list(ds.data_vars)[0]]
        dudx = da.copy(data=np.zeros(da.shape))
    if v is not None:
        dvdy = ds[v].copy(data=np.gradient(ds[v], axis=axis_y)) / this_expand_dims(dy)
    else:
        da = ds[list(ds.data_vars)[0]]
        dvdy = da.copy(data=np.zeros(da.shape))
    return dudx + dvdy


def compute_emf_2d_conv(ds: xr.Dataset):
    dx, dy, axis_x, axis_y, this_expand_dims = setup_derivatives(ds)
    
    e1 = 0.5 * (ds["vp"] ** 2 - ds["up"] ** 2)
    e2 = - ds["up"] * ds["vp"]
    de1dx = ds["up"].copy(data=np.gradient(e1, axis=axis_x)) / this_expand_dims(dx)
    de2dy = ds["vp"].copy(data=np.gradient(e2, axis=axis_y)) / this_expand_dims(dy)
    return de1dx + de2dy