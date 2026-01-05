"""
This module has been shamefully stolen in its entirety from https://github.com/joaofig/discrete-frechet. I have not found anything better. Looks maybe somewhat polarizable if i had a bigger brain.
"""

from numba import jit, types, prange, int32, int64
import math
import numpy as np
from typing import Callable
from .definitions import RADIUS


@jit(nopython=True)
def _bresenham_pairs(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Generates the diagonal coordinates

    Parameters
    ----------
    x0 : int
        Origin x value
    y0 : int
        Origin y value
    x1 : int
        Target x value
    y1 : int
        Target y value

    Returns
    -------
    np.ndarray
        Array with the diagonal coordinates
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs


@jit(nopython=True)
def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1], f_mat[i, j - 1], f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a


@jit(nopython=True)
def _fast_distance_matrix(p, q, diag, dist_func):
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # Create the distance array
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # Fill in the diagonal with the seed distance values
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist


@jit(nopython=True)
def _fast_frechet_matrix(
    dist: np.ndarray, diag: np.ndarray, p: np.ndarray, q: np.ndarray
) -> np.ndarray:

    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist


@jit(nopython=True)
def fdfd_matrix(
    p: np.ndarray, q: np.ndarray, dist_func: Callable[[np.array, np.array], float]
) -> float:
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_matrix(p, q, diagonal, dist_func)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca[-1, -1]


@jit(nopython=True, fastmath=True)
def haversine_numba(p: np.ndarray, q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in radians
    :q: Final location in radians
    :return: Distance
    """
    d = q - p
    a = (
        math.sin(d[0] / 2.0) ** 2
        + math.cos(p[0]) * math.cos(q[0]) * math.sin(d[1] / 2.0) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c


@jit(nopython=True, fastmath=True)
def earth_haversine_numba(p: np.ndarray, q: np.ndarray) -> float:
    """
    Vectorized haversine distance calculation
    :p: Initial location in degrees [lat, lon]
    :q: Final location in degrees [lat, lon]
    :return: Distances in meters
    """
    earth_radius = RADIUS
    return haversine_numba(np.radians(p), np.radians(q)) * earth_radius


fdfd_matrix(np.array([[1, 0]]), np.array([[1, 0]]), earth_haversine_numba)


def polars_frechet_wip_broken(df):
    """
    wip, does not work, cannot figure out a good expr_ij
    """
    predicate_00 = (pl.col("index") == 0) & (pl.col("index_right") == 0)
    predicate_i0 = (pl.col("index") > 0) & (pl.col("index_right") == 0)
    predicate_0j = (pl.col("index") == 0) & (pl.col("index_right") > 0)
    predicate_ij = (pl.col("index") > 0) & (pl.col("index_right") > 0)

    expr_00 = pl.col("dist")
    expr_i0 = pl.when(predicate_i0).then("dist").otherwise(0).cum_max()
    expr_0j = pl.when(predicate_0j).then("dist").otherwise(0).cum_max()
    n_index = pl.col("index").n_unique().cast(pl.Int32())
    
    expr_ij = pl.col("frechet")
    expr_ij = pl.min_horizontal(expr_ij.shift(-1), expr_ij.shift(-n_index - 1), expr_ij.shift(-n_index))
    expr_ij = expr_ij.cum_max() # this is wrong
    
    frechet_1 = (
        pl
        .when(predicate_00)
        .then(expr_00)   
        .when(predicate_i0)
        .then(expr_i0)
        .when(predicate_0j)
        .then(expr_0j)
        .otherwise(pl.col("dist"))
        .alias("frechet")
    )
    frechet_2 = (
        pl
        .when(predicate_ij)
        .then(expr_ij)
        .otherwise("frechet")
        .alias("frechet")
    )

    df[["index", "index_right", "dist"]].with_columns(frechet_1).with_columns(frechet_2)