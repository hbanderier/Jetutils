# coding: utf-8
from itertools import product
from typing import Mapping, Sequence, Tuple, Union, Iterable, Callable
from math import log10, floor

import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import LinearNDInterpolator
import xarray as xr
from xarray import DataArray
import polars as pl
import polars_ds as pds
from contourpy import contour_generator
from tqdm import tqdm, trange
from string import ascii_lowercase
from pathlib import Path

import matplotlib as mpl
from matplotlib import path as mpath
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import PathPatch
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    ListedColormap,
    LinearSegmentedColormap,
    to_rgb,
    to_hex,
    rgb_to_hsv,
    hsv_to_rgb,
)
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.dates import MonthLocator, DateFormatter
import colormaps
import cartopy.crs as ccrs
import cartopy.feature as feat
from IPython.display import clear_output
import datetime

from .definitions import (
    FIGURES,
    PRETTIER_VARNAME,
    UNITS,
    SEASONS,
    JJADOYS,
    maybe_circular_mean,
    get_index_columns,
    infer_direction,
    polars_to_xarray,
    xarray_to_polars,
)
from .jet_finding import gather_normal_da_jets
from .stats import field_significance
from .data import periodic_rolling_pl
from .anyspell import extend_spells
import jetutils

TEXTWIDTH_IN = 0.0138889 * 503.61377
STYLE_SHEET = Path(jetutils.__path__[0]).joinpath("matplotlibrc").as_posix()

plt.style.use(STYLE_SHEET)

# mpl.rcParams.update({
#     "font.size": 15,
#     "font.family": "cmu serif",
#     "xtick.labelsize": 15,
#     "ytick.labelsize": 15,
#     "axes.titlesize": 15,
#     "axes.labelsize": 15,
#     "figure.titlesize": 17,
#     "axes.titlepad": 10,
#     "figure.dpi": 100,
#     "savefig.dpi": 300,
#     "savefig.bbox": "tight",
#     "text.usetex": True,
# })


COLORS5 = [
    "#167e1b",
    "#8d49c5",
    "#d2709c",
    "#c48b45",
    "#5ccc99",
]

COLORS10 = [  # https://coolors.co/palette/f94144-f3722c-f8961e-f9844a-f9c74f-90be6d-43aa8b-4d908e-577590-277da1
    "#F94144",  # Vermilion
    "#F3722C",  # Orange
    "#F8961E",  # Atoll
    "#F9844A",  # Cadmium orange
    "#F9C74F",  # Caramel
    "#90BE6D",  # Lettuce green
    "#43AA8B",  # Bright Parrot Green
    "#4D908E",  # Abyss Green
    "#577590",  # Night Blue
    "#277DA1",  # Night Blue
]

COLORS = np.asarray(
    [
        colormaps.cet_l_bmw(0.47)[:3],
        to_rgb("#7a1cfe"),
        to_rgb("#ff2ec0"),
        to_rgb("#CE1C66"),
    ]
)
# COLORS = np.append(colormaps.cet_l_bmw([0.2, 0.47])[:, :3], np.asarray([to_rgb("#ff2ec0"), to_rgb("#CE1C66")]), axis=0)
# Dark Blue
# Purple
# Pink
# Pinkish red
COLORS_EXT = np.repeat(COLORS, 3, axis=0)
for i in range(len(COLORS)):
    for sign in [1, -1]:
        newcol_hsv = rgb_to_hsv(COLORS[i][:3]) * (
            1 + sign * np.asarray([0.0, 0.4, -0.4])
        )
        newcol_hsv[1] = np.clip(newcol_hsv[1], 0, 1)
        newcol_hsv[2] = np.clip(newcol_hsv[2], 0, 1)
        COLORS_EXT[3 * i + 1 + sign, :3] = hsv_to_rgb(newcol_hsv)

COLORS = [to_hex(c) for c in COLORS]
COLORS_EXT = [to_hex(c) for c in COLORS_EXT]

MYBLUES = LinearSegmentedColormap.from_list(
    "myblues", ["#f2f2f2", COLORS[0], COLORS_EXT[2]]
)
MYPURPLES = LinearSegmentedColormap.from_list(
    "mypurples", ["#f2f2f2", COLORS[1], COLORS_EXT[5]]
)
MYPINKS = LinearSegmentedColormap.from_list(
    "mypinks", ["#f2f2f2", COLORS[2], COLORS_EXT[8]]
)
MYREDS = LinearSegmentedColormap.from_list(
    "myreds", ["#f2f2f2", COLORS[3], COLORS_EXT[11]]
)
PINKPURPLE = LinearSegmentedColormap.from_list("pinkpurple", [COLORS[2], COLORS[1]])
BLUEWHITERED = LinearSegmentedColormap.from_list(
    "bluewhitered", [COLORS_EXT[11], COLORS[3], "#f2f2f2", COLORS[0], COLORS_EXT[2]]
)


COASTLINE = feat.NaturalEarthFeature(
    "physical", "coastline", "110m", edgecolor="black", facecolor="none"
)
BORDERS = feat.NaturalEarthFeature(
    "cultural",
    "admin_0_boundary_lines_land",
    "10m",
    edgecolor="grey",
    facecolor="none",
)

COLOR_JETS = colormaps.bold(np.arange(12))
DEFAULT_COLORMAP = colormaps.fusion_r


def num2tex(x: float, force: bool = False, ncomma: int = 1) -> str:
    float_str = f"{x:.{ncomma}e}" if force else f"{x:.{ncomma}g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def p_to_tex(c1: float, c0: float, no_intercept: bool = True) -> str:
    coef1 = num2tex(c1)
    if no_intercept:
        return rf"$y\sim {coef1}\cdot x$"
    coef0 = num2tex(c0)
    sign = "+" if np.sign(c0) else "-"
    return rf"$y={coef1}\cdot x {sign} {coef0}$"


def make_boundary_path(
    minlon: float, maxlon: float, minlat: float, maxlat: float, n: int = 50
) -> mpath.Path:
    """Creates path to be used by GeoAxes.

    Args:
        minlon (float): minimum longitude
        maxlon (float): maximum longitude
        minlat (float): minimum latitude
        maxlat (float): maximum latitude
        n (int, optional): Interpolation points for each segment. Defaults to 50.

    Returns:
        boundary_path (mpath.Path): Boundary Path in flat projection
    """

    boundary_path = []
    # North (E->W)
    if maxlon > minlon:
        linspace =         np.linspace(minlon, maxlon, n)
    else:
        linspace = (180 + np.linspace(minlon % 360, maxlon % 360, n)) % 360 - 180
    edge = [linspace, np.full(n, maxlat)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # West (N->S)
    edge = [np.full(n, maxlon), np.linspace(maxlat, minlat, n)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # South (W->E)
    edge = [linspace[::-1], np.full(n, minlat)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # East (S->N)
    edge = [np.full(n, minlon), np.linspace(minlat, maxlat, n)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    boundary_path = mpath.Path(boundary_path)

    return boundary_path


def figtitle(
    minlon: str,
    maxlon: str,
    minlat: str,
    maxlat: str,
    season: str,
) -> str:
    minlon, maxlon, minlat, maxlat = (
        float(minlon),
        float(maxlon),
        float(minlat),
        float(maxlat),
    )
    title = f'${np.abs(minlon):.1f}°$ {"W" if minlon < 0 else "E"} - '
    title += f'${np.abs(maxlon):.1f}°$ {"W" if maxlon < 0 else "E"}, '
    title += f'${np.abs(minlat):.1f}°$ {"S" if minlat < 0 else "N"} - '
    title += f'${np.abs(maxlat):.1f}°$ {"S" if maxlat < 0 else "N"} '
    title += season
    return title


def honeycomb_panel(
    nrow,
    ncol,
    ratio: float = 1.4,
    hspace: float = 0.0,
    wspace: float = 0.0,
    row_height: float = 2.0,
    **subplot_kw,
) -> Tuple[Figure, np.ndarray]:
    fig = plt.figure(figsize=(row_height * nrow, row_height * ratio * nrow))
    gs = GridSpec(nrow, 2 * ncol + 1, hspace=hspace, wspace=wspace)
    axes = np.empty((nrow, ncol), dtype=object)
    if subplot_kw is None:
        subplot_kw = {}
    for i, j in product(range(ncol), range(nrow)):
        if j % 2 == 1:
            slice_x = slice(2 * i, 2 * i + 2)
        else:
            slice_x = slice(2 * i + 1, 2 * i + 2 + 1)
        axes[j, i] = fig.add_subplot(gs[j, slice_x], **subplot_kw)
    return fig, axes


def make_transparent(
    cmap: str | Colormap,
    nlev: int = None,
    alpha_others: float = 1,
    n_transparent: int = 1,
    direction: int = 1,
) -> Colormap:
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    if nlev is None:
        nlev = cmap.N
    colorlist = cmap(np.linspace(0, 1, nlev + int(direction == 0)))
    if direction == 0:
        midpoint = int(np.ceil(nlev / 2))
        colorlist[midpoint - n_transparent + 1 : midpoint + n_transparent, -1] = 0
        colorlist[midpoint + n_transparent :, -1] = alpha_others
        colorlist[: midpoint - n_transparent + 1, -1] = alpha_others
    elif direction == 1:
        colorlist[:n_transparent, -1] = 0
        colorlist[n_transparent:, -1] = alpha_others
    else:
        colorlist[-n_transparent:, -1] = 0
        colorlist[:-n_transparent, -1] = alpha_others
    return ListedColormap(colorlist)


def create_levels(
    to_plot: list, levels: int | Sequence | None = None, q: float = 0.99
) -> Tuple[np.ndarray, np.ndarray, str, int]:
    if to_plot[0].dtype == bool:
        return np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), "neither", 1
    extend = {-1: "min", 0: "both", 1: "max"}
    direction = infer_direction(to_plot)
    extend = extend[direction] if q < 1 else "neither"
    if isinstance(levels, Sequence):
        levelsc = np.asarray(levels)
        if direction == 0:
            levelscf = np.delete(levelsc, np.nonzero(levelsc == 0)[0])
        else:
            levelscf = levelsc
        return levelsc, levelscf, extend, direction

    if levels is None:
        levels = 7 if direction is None else 4

    lowbound, highbound = np.nanquantile(to_plot, q=[1 - q, q])
    lowbound = 0 if direction == 1 else lowbound
    highbound = 0 if direction == -1 else highbound
    levelsc = MaxNLocator(levels, symmetric=(direction == 0)).tick_values(
        lowbound, highbound
    )
    if direction == 0:
        levelscf = np.delete(levelsc, len(levelsc) // 2)
    else:
        levelscf = levelsc
    return levelsc, levelscf, extend, direction


def doubleit(thing: list | str | None, length: int, default: str) -> list:
    if isinstance(thing, str):
        return [thing] * length
    elif isinstance(thing, list):
        lover2 = int(length / 2)
        if len(thing) == 3:
            return (
                lover2 * [thing[0]] + (length % 2) * [thing[1]] + (lover2) * [thing[2]]
            )
        else:
            return (
                lover2 * [thing[0]] + (length % 2) * [default] + (lover2) * [thing[1]]
            )
    else:
        return [default] * length


def setup_lon_lat(
    to_plot: list[xr.DataArray],
    lon: np.ndarray | None,
    lat: np.ndarray | None,
):
    if lon is None or lat is None:
        try:
            lon = to_plot[0].lon.values
            lat = to_plot[0].lat.values
        except AttributeError:
            print("Either provide lon / lat or make to_plot items dataArrays")
            raise
    lon = np.sort(lon)
    lon_ = np.unique(lon)
    dl = lon_[1] - lon_[0]
    if np.all(np.diff(lon) < 2 * dl):
        return to_plot, lon - maybe_circular_mean(to_plot[0].lon.values), lat
    to_plot_ = []
    for tplt in to_plot:
        to_plot_.append(tplt.assign_coords(lon=tplt.lon.values % 360))
    return to_plot_, to_plot_[0].lon.values - maybe_circular_mean(to_plot_[0].lon.values), lat


def to_prettier_order(n: int | np.ndarray, width: int = 6, height: int = 4):
    col, row = divmod(n, height)
    row = height - 1 - row
    return 1 + width * row + col


# def inv_prettier_order(n: int | np.ndarray, width: int = 6, height: int = 4):
#     col, row = divmod(n, height)
#     row = height - 1 - row
#     return 1 + width * row + col


class Clusterplot:
    def __init__(
        self,
        nrow: int,
        ncol: int,
        region: np.ndarray | list | tuple = None,
        lambert_projection: bool = False,
        honeycomb: bool = False,
        numbering: bool | Callable | Sequence = False,
        coastline: bool = True,
        row_height: float = 3,
    ) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.lambert_projection = lambert_projection
        if region is None:
            region = (-80, 20, 15, 80)  # Default region ?
        self.region = region
        self.minlon, self.maxlon, self.minlat, self.maxlat = region
        if self.maxlon > self.minlon:
            lon = np.linspace(self.minlon, self.maxlon)
        else:
            lon = (180 + np.linspace(self.maxlon % 360, self.minlon % 360)) % 360 - 180
        self.central_longitude = np.degrees(maybe_circular_mean(np.radians(lon)))
        self.central_longitude = np.round(self.central_longitude)
        if self.lambert_projection:
            projection = ccrs.LambertConformal(
                central_longitude=self.central_longitude
            )
            ratio = 0.6 * self.nrow / (self.ncol + (0.5 if honeycomb else 0))
            self.boundary = make_boundary_path(*region)
        else:
            projection = ccrs.PlateCarree(central_longitude=self.central_longitude)
            ratio = (
                (self.maxlat - self.minlat)
                / np.abs(self.maxlon - self.minlon)
                * self.nrow
                / (self.ncol + (0.5 if honeycomb else 0))
                * (0.8 if honeycomb else 1)
            )
        if honeycomb:
            self.fig, self.axes = honeycomb_panel(
                self.nrow, self.ncol, ratio, row_height=row_height, projection=projection
            )
        else:
            self.fig, self.axes = plt.subplots(
                self.nrow,
                self.ncol,
                figsize=(row_height * self.ncol, row_height * self.ncol * ratio),
                constrained_layout=not lambert_projection,
                subplot_kw={"projection": projection},
            )
        self.axes = np.atleast_1d(self.axes).flatten()
        for ax in self.axes:
            if coastline:
                ax.add_feature(COASTLINE)
            ax.set_extent(
                [self.minlon, self.maxlon, self.minlat, self.maxlat],
            )
            ax.set_xlim(self.minlon - self.central_longitude, self.maxlon - self.central_longitude)
            ax.set_ylim(self.minlat, self.maxlat)
            if self.lambert_projection:
                ax.set_boundary(self.boundary)
            # ax.add_feature(BORDERS, transform=ccrs.PlateCarree())
        if numbering:
            plt.draw()
            for i, ax in enumerate(self.axes):
                if isinstance(numbering, Callable):
                    j = str(numbering(i))
                elif isinstance(numbering, Sequence):
                    j = str(numbering[i])
                else:
                    j = str(i + 1)
                ax.annotate(
                    j,
                    (2.2, 4),
                    xycoords="axes points",
                    ha="left",
                    va="baseline",
                    fontweight="demi",
                    bbox={
                        "boxstyle": "square, pad=0.1",
                        "edgecolor": "none",
                        "facecolor": "white",
                    },
                    usetex=False,
                )

    def _add_gridlines(self, step: int | tuple = None) -> None:
        for ax in self.axes:
            gl = ax.gridlines(
                dms=False, x_inline=False, y_inline=False, draw_labels=True
            )
            if step is not None:
                if isinstance(step, int):
                    step = (step, step)
            else:
                step = (30, 20)
            gl.xlocator = mticker.FixedLocator(
                np.arange(self.minlon, self.maxlon + 1, step[0])
            )
            gl.ylocator = mticker.FixedLocator(
                np.arange(self.minlat, self.maxlat + 1, step[1])
            )
            # gl.xlines = (False,)
            # gl.ylines = False
            plt.draw()
            # for ea in gl.label_artists:
            #     current_pos = ea.get_position()
            #     if ea.get_text()[-1] in ["N", "S"]:
            #         ea.set_visible(True)
            #         continue
            #     if current_pos[1] > 4000000:
            #         ea.set_visible(False)
            #         continue
            #     ea.set_visible(True)
            #     ea.set_rotation(0)
            #     ea.set_position([current_pos[0], current_pos[1] - 200000])

    def _add_titles(self, titles: Iterable) -> None:
        if len(titles) > len(self.axes):
            titles = titles[: len(self.axes)]
        for title, ax in zip(titles, self.axes):
            if isinstance(title, float):
                title = f"{title:.2f}"
            ax.set_title(title)

    def resize_relative(self, ratios=Sequence[float]):
        self.fig.set_size_inches(self.fig.get_size_inches() * np.asarray(ratios))

    def resize_absolute(self, size=Sequence[float]):
        self.fig.set_size_inches(np.asarray(size))

    def add_contour(
        self,
        to_plot: list | xr.DataArray,
        lon: np.ndarray = None,
        lat: np.ndarray = None,
        levels: int | Sequence | None = None,
        clabels: Union[bool, list] = False,
        draw_gridlines: bool = False,
        titles: Iterable = None,
        colors: list | str = None,
        linestyles: list | str = None,
        q: float = 0.99,
        **kwargs,
    ) -> None:
        to_plot, lon, lat = setup_lon_lat(to_plot, lon, lat)  # d r y too much
        lon = lon + self.central_longitude

        levelsc, levelscf, _, direction = create_levels(to_plot, levels, q=q)

        if direction == 0 and linestyles is None:
            linestyles = ["dashed", "solid"]

        if direction == 0:
            colors = doubleit(colors, len(levelsc), "black")
            linestyles = doubleit(linestyles, len(levelsc), "solid")

        if direction != 0 and colors is None:
            colors = "black"
        if direction != 0 and linestyles is None:
            linestyles = "solid"
        if "cmap" in kwargs and kwargs["cmap"] is not None:
            colors = None

        for ax, toplt in zip(self.axes, to_plot):
            cs = ax.contour(
                lon,
                lat,
                toplt,
                transform=ccrs.PlateCarree(),
                levels=levelscf,
                colors=colors,
                linestyles=linestyles,
                **kwargs,
            )

            if isinstance(clabels, bool) and clabels:
                ax.clabel(cs)
            elif isinstance(clabels, list):
                ax.clabel(cs, levels=clabels)

            if self.lambert_projection and self.boundary is not None:
                ax.set_boundary(self.boundary, transform=ccrs.PlateCarree())

        if titles is not None:
            self._add_titles(titles)

        if draw_gridlines:
            self._add_gridlines()

    def setup_contourf(
        self,
        to_plot: list | xr.DataArray,
        levels: int | Sequence | None = None,
        cmap: str | Colormap = DEFAULT_COLORMAP,
        transparify: bool | float | int = False,
        contours: bool = False,
        clabels: Union[bool, list] = None,
        cbar_label: str = None,
        cbar_kwargs: Mapping = None,
        q: float = 0.99,
        **kwargs,
    ) -> Tuple[Mapping, Mapping, ScalarMappable, np.ndarray]:
        levelsc, levelscf, extend, direction = create_levels(to_plot, levels, q=q)

        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        if transparify:
            if isinstance(transparify, int):
                cmap = make_transparent(
                    cmap,
                    nlev=len(levelscf),
                    n_transparent=transparify,
                    direction=direction,
                )
            elif isinstance(transparify, float):
                cmap = make_transparent(
                    cmap,
                    nlev=len(levelscf),
                    alpha_others=transparify,
                    direction=direction,
                )
            else:
                cmap = make_transparent(cmap, nlev=len(levelscf), direction=direction)

        if cbar_kwargs is None:
            cbar_kwargs = {}

        if cbar_label is not None:  # backwards compat
            cbar_kwargs["label"] = cbar_label

        norm = BoundaryNorm(levelscf, cmap.N, extend=extend)
        im = ScalarMappable(norm=norm, cmap=cmap)

        if contours or clabels is not None:
            self.add_contour(to_plot, levels, clabels)

        return (
            dict(
                transform=ccrs.PlateCarree(central_longitude=self.central_longitude),
                levels=levelscf,
                cmap=cmap,
                norm=norm,
                extend=extend,
                **kwargs,
            ),
            cbar_kwargs,
            im,
            levelsc,
        )

    def add_contourf(
        self,
        to_plot: list | xr.DataArray,
        lon: np.ndarray = None,
        lat: np.ndarray = None,
        levels: int | Sequence | None = None,
        cmap: str | Colormap = DEFAULT_COLORMAP,
        transparify: bool | float | int = False,
        contours: bool = False,
        clabels: Union[bool, list] = None,
        draw_gridlines: bool = False,
        draw_cbar: bool = True,
        cbar_label: str = None,
        titles: Iterable = None,
        cbar_kwargs: Mapping = None,
        q: float = 0.99,
        **kwargs,
    ) -> Tuple[ScalarMappable, Mapping]:
        to_plot, lon, lat = setup_lon_lat(to_plot, lon, lat) 

        kwargs, cbar_kwargs, im, levelsc = self.setup_contourf(
            to_plot,
            levels,
            cmap,
            transparify,
            contours,
            clabels,
            cbar_label,
            cbar_kwargs,
            q=q,
            **kwargs,
        )

        for ax, toplt in zip(self.axes, to_plot):
            try:
                toplt = toplt.values
            except AttributeError:
                pass
            ax.contourf(lon, lat, toplt, **kwargs)

            if self.lambert_projection and self.boundary is not None:
                ax.set_boundary(self.boundary, transform=ccrs.PlateCarree(central_longitude=self.central_longitude))

        if titles is not None:
            self._add_titles(titles)

        if draw_gridlines:
            self._add_gridlines()

        if draw_cbar:
            self.cbar = self.fig.colorbar(
                im, ax=self.fig.axes, spacing="proportional", **cbar_kwargs
            )
        else:
            self.cbar = None

        return im, kwargs

    def add_stippling(
        self,
        da: DataArray,
        mask: np.ndarray,
        FDR: bool = False,
        color: str = "black",
        hatch: str = "xx",
    ) -> None:
        to_test = []
        for mas in mask.T:
            if np.sum(mas) < 1:
                to_test.append(da[:1].copy(data=np.zeros((1, *da.shape[1:]))))
                continue
            to_test.append(da.isel(time=mas))

        lon = da.lon.values
        lat = da.lat.values
        lon = lon - self.central_longitude
        significances = []
        da_ = da.values
        # da = np.sort(da, axis=0)
        for i in trange(mask.shape[1]):
            significances.append(
                field_significance(to_test[i], da_, 200, q=0.1)[int(FDR)]
            )

        for ax, signif in zip(self.axes, significances):
            cs = ax.pcolor(
                lon,
                lat,
                da[0].where(signif),
                hatch=hatch,
                alpha=0.,
            )

            # cs.set_edgecolor(color)
            # cs.set_linewidth(0.0)

    def add_any_contour_from_mask(
        self,
        da: DataArray,
        mask: np.ndarray,
        type: str = "contourf",
        stippling: bool | str = False,
        reduction_function: Callable = np.mean,
        **kwargs,
    ) -> ScalarMappable | None:
        to_plot = []
        time_name = "time" if "time" in da.dims else da.dims[0]
        for mas in tqdm(mask.T, total=mask.shape[1]):
            if np.sum(mas) < 1:
                to_plot.append(da[0].copy(data=np.zeros(da.shape[1:])))
                continue
            to_plot.append(da.isel({time_name: mas}).reduce(reduction_function, dim=time_name))

        if type == "contourf":
            im = self.add_contourf(
                to_plot,
                contours=False,
                clabels=None,
                **kwargs,
            )
        elif type == "contour":
            self.add_contour(
                to_plot,
                **kwargs,
            )
            im = None
        elif type == "both":
            im = self.add_contourf(
                to_plot,
                contours=True,
                **kwargs,
            )
        else:
            raise ValueError(
                f'Wrong {type=}, choose among "contourf", "contour" or "both"'
            )
        if stippling:
            if isinstance(stippling, str):
                color = stippling
            else:
                color = "black"
            self.add_stippling(da, mask, color=color)
        return im

    def cluster_on_fig(
        self,
        coords: np.ndarray,
        clu_labs: np.ndarray,
        cmap: str | Colormap = None,
    ) -> None:
        unique_labs = np.unique(clu_labs)
        sym = np.any(unique_labs < 0)

        if cmap is None:
            cmap = "PiYG" if sym else "Greens"
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        nabove = np.sum(unique_labs > 0)
        if isinstance(cmap, list | np.ndarray):
            colors = cmap
        else:
            if sym:
                nbelow = np.sum(unique_labs < 0)
                cab = np.linspace(1, 0.66, nabove)
                cbe = np.linspace(0.33, 0, nbelow)
                if 0 in unique_labs:
                    zerocol = [0.5]
                else:
                    zerocol = []
                colors = [*cbe, *zerocol, *cab]
            else:
                if 0 in unique_labs:
                    zerocol = [0.0]
                else:
                    zerocol = []
                colors = np.linspace(1.0, 0.33, nabove)
                colors = [*zerocol, *colors]
            colors = cmap(colors)

        xmin, ymin = self.axes[0].get_position().xmin, self.axes[0].get_position().ymin
        xmax, ymax = (
            self.axes[-1].get_position().xmax,
            self.axes[-1].get_position().ymax,
        )
        x = np.linspace(xmin, xmax, 200)
        y = np.linspace(ymin, ymax, 200)
        newmin, newmax = np.asarray([xmin, ymin]), np.asarray([xmax, ymax])
        dx = coords[self.nrow, 0] - coords[0, 0]
        dy = (coords[2, 1] - coords[0, 1]) / 2
        for coord, val in zip(coords, clu_labs):
            newcs = [
                [coord[0] + sgnx * dx / 2.01, coord[1] + sgny * dy / 2.01]
                for sgnx, sgny in product([-1, 0, 1], [-1, 0, 1])
            ]
            coords = np.append(coords, newcs, axis=0)
            clu_labs = np.append(clu_labs, [val] * len(newcs))
        min, max = np.amin(coords, axis=0), np.amax(coords, axis=0)
        reduced_coords = (coords - min[None, :]) / (max - min)[None, :] * (
            newmax - newmin
        )[None, :] + newmin[None, :]
        for i, lab in enumerate(unique_labs):
            interp = LinearNDInterpolator(reduced_coords, clu_labs == lab)
            r = interp(*np.meshgrid(x, y))

            if lab == 0:
                iter = contour_generator(x, y, r).filled(0.99, 1)[0]
                ls = "none"
                fc = "black"
                alpha = 0.2
                ec = "none"
            else:
                iter = contour_generator(x, y, r).lines(0.99)
                ls = "solid" if lab >= 0 else "dashed"
                alpha = 1
                fc = "none"
                ec = colors[i]

            for p in iter:
                self.fig.add_artist(
                    PathPatch(
                        mpath.Path(p),
                        fc=fc,
                        alpha=alpha,
                        ec=ec,
                        lw=6,
                        ls=ls,
                    )
                )


def cdf(timeseries: Union[DataArray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the cumulative distribution function of a 1D DataArray

    Args:
        timeseries (xr.DataArray or npt.np.ndarray): will be cast to ndarray if DataArray.

    Returns:
        x (npt.np.ndarray): x values for plotting,
        y (npt.np.ndarray): cdf of the timeseries,
    """
    if isinstance(timeseries, DataArray):
        timeseries = timeseries.values
    idxs = np.argsort(timeseries)
    y = np.cumsum(idxs) / np.sum(idxs)
    x = timeseries[idxs]
    return x, y


def trends_and_pvalues(
    props_as_df: pl.DataFrame,
    data_vars: list,
    season: str | None = None,
    std: bool = False,
    bootstrap_len: int = 4,
    n_boostraps: int = 10000,
):
    ncat = props_as_df["jet"].n_unique()

    if season is not None and season != "Year":
        month_list = SEASONS[season]
        props_as_df = props_as_df.filter(pl.col("time").dt.month().is_in(month_list))
    else:
        season = "all_year"

    def agg_func(col):
        return pl.col(col).std() if std else pl.col(col).mean()

    aggs = [agg_func(col) for col in data_vars]
    props_as_df = props_as_df.group_by(
        pl.col("time").dt.year().alias("year"), pl.col("jet"), maintain_order=True
    ).agg(*aggs)

    x = props_as_df["year"].unique()
    n = len(x)
    num_blocks = n // bootstrap_len

    rng = np.random.default_rng()

    sample_indices = rng.choice(
        n - bootstrap_len, size=(n_boostraps, n // bootstrap_len)
    )
    sample_indices = sample_indices[..., None] + np.arange(bootstrap_len)[None, None, :]
    sample_indices = sample_indices.reshape(n_boostraps, num_blocks * bootstrap_len)
    sample_indices = np.append(
        sample_indices, np.arange(sample_indices.shape[1])[None, :], axis=0
    )
    sample_indices = ncat * np.repeat(sample_indices.flatten(), ncat)
    for k in range(ncat):
        sample_indices[k::ncat] = sample_indices[k::ncat] + k

    ts_bootstrapped = props_as_df[sample_indices]
    ts_bootstrapped = ts_bootstrapped.with_columns(
        sample_index=np.arange(len(ts_bootstrapped))
        // (ncat * num_blocks * bootstrap_len),
        inside_index=np.arange(len(ts_bootstrapped))
        % (ncat * num_blocks * bootstrap_len),
    )

    slopes = ts_bootstrapped.group_by(["sample_index", "jet"], maintain_order=True).agg(
        **{
            data_var: pds.lin_reg_report(
                    pl.int_range(0, pl.col("year").len()).alias("year"), 
                    target=pl.col(data_var), 
                    add_bias=True
                ).struct.field("beta").first()
            for data_var in data_vars
        }
    )

    constants = props_as_df.group_by("jet", maintain_order=True).agg(
        **{
            data_var: pds.lin_reg_report(
                pl.col("year"), 
                target=pl.col(data_var), 
                add_bias=True
            ).struct.field("beta").last()
            for data_var in data_vars
        }
    )

    pvals = slopes.group_by("jet", maintain_order=True).agg(
        **{
            data_var: pl.col(data_var)
            .head(n_boostraps)
            .sort()
            .search_sorted(pl.col(data_var).get(-1)).first()
            / n_boostraps
            for data_var in data_vars
        }
    )
    return x, props_as_df, slopes, constants, pvals


def clear(func):
    def wrapper(*args, **kwargs):
        clear = kwargs.get("clear", True)
        if clear:
            plt.ion()
            plt.show()
            clear_output()
        fig = func(*args, **kwargs)
        if clear:
            del fig
            plt.close()
            clear_output()
            return
        else:
            plt.show()
        return fig

    return wrapper


@clear
def plot_trends(
    props_as_df: pl.DataFrame,
    data_vars: list,
    season: str | None = None,
    bootstrap_len: int = 4,
    n_boostraps: int = 10000,
    std: bool = False,
    nrows: int = 3,
    ncols: int = 4,
    suffix: str = "",
    numbering: bool = False,
    *,
    save: bool = False,
    clear: bool = True,
):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 2.4),
        tight_layout=True,
        sharex="all",
    )
    axes = axes.flatten()

    x, y, slopes, constants, pvals = trends_and_pvalues(
        props_as_df=props_as_df,
        data_vars=data_vars,
        season=season,
        std=std,
        bootstrap_len=bootstrap_len,
        n_boostraps=n_boostraps,
    )
    ncat = props_as_df["jet"].n_unique()

    for letter, varname, ax in zip(ascii_lowercase, data_vars, axes):
        dji = varname == "double_jet_index"
        if varname == "mean_lev":
            ax.invert_yaxis()
        if numbering:
            ax.set_title(
                f"{letter}) {PRETTIER_VARNAME.get(varname, varname)} [{UNITS.get(varname, '')}]"
            )
        else:
            ax.set_title(
                f"{PRETTIER_VARNAME.get(varname, varname)} [{UNITS.get(varname, '')}]"
            )

        for j, jet in enumerate(["STJ", "EDJ"]):
            c1 = slopes[ncat * n_boostraps + j, varname]
            c0 = constants[j, varname]
            p = pvals[j, varname]
            p = min(p, 1 - p) * 2
            this_da = y.filter(pl.col("jet") == jet)[varname]
            color = "black" if dji else COLORS[2 - j]
            ls = "dashed" if p < 0.05 else "dotted"
            if c1 is not None:
                if dji:
                    label = f"{p_to_tex(c1, c0, True)}, $p={p:.2f}$"
                else:
                    label = f"{jet}, {p_to_tex(c1, c0, True)}, $p={p:.2f}$"
            else:
                if dji:
                    label = ""
                else:
                    label = f"{jet}"
            ax.plot(x, this_da.to_numpy(), lw=2, color=color)
            ax.plot(
                x,
                c1 * x + c0,
                lw=1.5,
                color=color,
                ls=ls,
                label=label,
            )
            if dji:
                break
        ax.legend(ncol=1)
    if save:
        subtitle = "_std_" if std else "_"
        fig.savefig(
            f"{FIGURES}/jet_props{subtitle}trends/jet_props_{season}{suffix}.png"
        )
    return fig


@clear
def plot_seasonal(
    props_as_df: pl.DataFrame,
    data_vars: list,
    nrows: int = 3,
    ncols: int = 4,
    suffix: str = "",
    numbering: bool = False,
    *,
    save: bool = False,
    clear: bool = True,
):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.5, nrows * 2.4),
        tight_layout=True,
        sharex="all",
    )
    axes = axes.flatten()
    jets = props_as_df["jet"].unique().to_numpy()
    njets = len(jets)
    gb = props_as_df.group_by(
        [pl.col("time").dt.ordinal_day().alias("dayofyear"), pl.col("jet")],
    )
    means = gb.agg([pl.col(col).mean() for col in data_vars]).sort("dayofyear", "jet", descending=[False, True])
    means = periodic_rolling_pl(means, 15, data_vars)
    x = means["dayofyear"].unique()
    medians = gb.agg([pl.col(col).median() for col in data_vars]).sort("dayofyear", "jet", descending=[False, True])
    medians = periodic_rolling_pl(medians, 15, data_vars)
    q025 = gb.quantile(0.25).sort("dayofyear", "jet", descending=[False, True])
    q075 = gb.quantile(0.75).sort("dayofyear", "jet", descending=[False, True])
    if njets == 3:
        color_order = [2, 3, 1]
    else:
        color_order = [2, 1]
    for letter, varname, ax in zip(ascii_lowercase, data_vars, axes.ravel()):
        dji = varname == "double_jet_index"
        ys = means[varname].to_numpy().reshape(366, njets)
        qs = np.stack(
            [
                q025[varname].to_numpy().reshape(366, njets),
                q075[varname].to_numpy().reshape(366, njets),
            ],
            axis=2,
        )
        median = medians[varname].to_numpy().reshape(366, njets)
        for i in range(njets):
            color = "black" if dji else COLORS[color_order[i]]
            ax.fill_between(
                x, qs[:, i, 0], qs[:, i, 1], color=color, alpha=0.2, zorder=-10
            )
            ax.plot(x, median[:, i], lw=2, color=color, ls="dotted", zorder=0)
            ax.plot(x, ys[:, i], lw=3, color=color, label=jets[i], zorder=10)
            if dji:
                break
        if numbering:
            ax.set_title(
                f"{letter}) {PRETTIER_VARNAME.get(varname, varname)} [{UNITS.get(varname, '')}]"
            )
        else:
            ax.set_title(
                f"{PRETTIER_VARNAME.get(varname, varname)} [{UNITS.get(varname, '')}]"
            )
        ax.xaxis.set_major_locator(MonthLocator(range(0, 13, 3)))
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.set_xlim(min(x), max(x))
        if varname == "mean_lev":
            ax.invert_yaxis()
        ylim = ax.get_ylim()
        wherex = np.isin(x, JJADOYS)
        ax.fill_between(x, *ylim, where=wherex, alpha=0.1, color="black", zorder=-10)
        ax.set_ylim(ylim)
    axes.ravel()[0].legend().set_zorder(102)
    if save:
        plt.savefig(f"{FIGURES}/jet_props_misc/jet_props_seasonal{suffix}.png")
    return fig


@clear
def props_histogram(
    props_as_ds: xr.Dataset,
    data_vars: list,
    season: str | None = None,
    nrows: int = 3,
    ncols: int = 4,
    suffix: str = "",
    *,
    save: bool = False,
    clear: bool = True,
):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.5, nrows * 2.4), tight_layout=True
    )
    axes = axes.flatten()
    if season is not None:
        month_list = SEASONS[season]
        season_mask = np.isin(props_as_ds.time.dt.month.values, month_list)
        props_as_ds_ = props_as_ds.sel(time=season_mask)
    else:
        props_as_ds_ = props_as_ds

    for i, (varname, ax) in enumerate(zip(data_vars, axes)):
        if varname == "mean_lev":
            ax.invert_xaxis()
        try:
            ax.set_title(f"{PRETTIER_VARNAME[varname]} [{UNITS[varname]}]")
        except KeyError:
            ax.set_title(varname)
        maxx = {}
        for j, jet in enumerate(["subtropical", "polar"]):
            try:
                this_da = props_as_ds_[varname].sel(jet=jet)
                x = np.linspace(this_da.min(), this_da.max(), 1000)
                y = gaussian_kde(
                    this_da.interpolate_na("time", fill_value="extrapolate").values
                )(x) * len(this_da)
                ax.plot(x, y, color=COLORS[2 - j], linewidth=2)
                ax.fill_between(x, 0, y, color=COLORS[2 - j], linewidth=1, alpha=0.3)
                maxx_ = x[np.argmax(y)]
                maxx[jet] = round(maxx_, -int(floor(log10(abs(maxx_)))) + 2)
            except KeyError:
                this_da = props_as_ds_[varname]
                x = np.linspace(this_da.min(), this_da.max(), 1000)
                y = gaussian_kde(
                    this_da.interpolate_na("time", fill_value="extrapolate").values
                )(x) * len(this_da)
                ax.plot(x, y, color="black", linewidth=2)
                ax.fill_between(x, 0, y, color="black", linewidth=1, alpha=0.3)
                break
        if len(maxx) > 0:
            current_ticks = ax.get_xticks()
            xticks = ax.set_xticks(np.concatenate([current_ticks, list(maxx.values())]))
            xticks[-1]._apply_params(
                color=COLORS[1], labelcolor=COLORS[1], length=12, width=3, pad=14
            )
            pad = (
                14
                if varname in ["mean_lat", "mean_lev", "spe_star", "waviness1"]
                else 25
            )
            xticks[-2]._apply_params(
                color=COLORS[2], labelcolor=COLORS[2], length=12, width=3, pad=pad
            )
            ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if save:
        fig.savefig(f"{FIGURES}/jet_props_hist/{season}{suffix}.png")
    return fig


def interp_jets_to_zero_one(
    jets: pl.DataFrame, varnames: list[str] | str, n_interp: int = 30
):
    if isinstance(varnames, str):
        varnames = [varnames]
    index_columns = get_index_columns(jets)
    if "relative_index" in index_columns and "time" in index_columns:
        index_columns.remove("time")
        varnames.append("time")
    jets = jets.with_columns(
        norm_index=jets.group_by(index_columns, maintain_order=True)
        .agg(pl.col("index") / pl.col("index").max())["index"]
        .explode()
    )
    jets = jets.group_by(
        [*index_columns, ((pl.col("norm_index") * n_interp) // 1) / n_interp, "n"],
        maintain_order=True,
    ).agg([pl.col(varname).mean() for varname in varnames])
    return jets


def _gather_normal_da_jets_wrapper(
    jets: pl.DataFrame,
    times: pl.DataFrame,
    da: xr.DataArray,
    n_interp: int = 30,
    clim: xr.DataArray | None = None,
    half_length: float = 20,
    dn: float = 1,
):
    jets = times.join(jets, on="time", how="left")
    jets = gather_normal_da_jets(jets, da, half_length=half_length, dn=dn)
    varname = da.name + "_interp"

    jets = interp_jets_to_zero_one(jets, [varname, "is_polar"], n_interp=n_interp)
    if clim is None:
        return jets
    clim = xarray_to_polars(clim)
    jets = jets.with_columns(
        dayofyear=pl.col("time").dt.ordinal_day(), is_polar=pl.col("is_polar") > 0.5
    ).cast({"n": pl.Float64})
    jets = (
        jets.join(clim, on=["dayofyear", "is_polar", "norm_index", "n"])
        .with_columns(pl.col(varname) - pl.col(f"{varname}_right"))
        .drop(f"{varname}_right", "dayofyear")
    )
    return jets


def gather_normal_da_jets_wrapper(
    jets: pl.DataFrame,
    times: pl.DataFrame,
    da: xr.DataArray,
    n_interp: int = 30,
    n_bootstraps: int = 0,
    half_length: float = 20,
    dn: float = 1,
    clim: xr.DataArray | None = None,
    time_before: datetime.timedelta = datetime.timedelta(0),
    time_after: datetime.timedelta = datetime.timedelta(0),
):
    times = extend_spells(times, time_before=time_before, time_after=time_after)
    varname = da.name + "_interp"
    if not n_bootstraps:
        jets = _gather_normal_da_jets_wrapper(jets, times, da, n_interp, clim=clim, half_length=half_length, dn=dn)
        jets = jets.group_by(
            [pl.col("is_polar") > 0.5, "norm_index", "n"], maintain_order=True
        ).agg(pl.col(varname).mean())
        return polars_to_xarray(jets, index_columns=["is_polar", "norm_index", "n"])
    rng = np.random.default_rng()
    orig_times = times.clone()
    boostrap_len = orig_times["time"].n_unique()
    all_times = jets["time"].unique().clone()
    times = (
        times[["time"]]
        .with_row_index("inside_index")
        .with_columns(sample_index=pl.lit(n_bootstraps, dtype=pl.UInt32))
    )
    if "spell" in orig_times.columns:  # then block bootstrapping
        if (boostrap_len % orig_times["spell"].n_unique()) == 0:
            bootstrap_block = int(boostrap_len // orig_times["spell"].n_unique())
        else:
            bootstrap_block = 1
        boostraps = rng.choice(
            all_times.shape[0] - bootstrap_block,
            size=(n_bootstraps, boostrap_len // bootstrap_block),
        )
        boostraps = boostraps[..., None] + np.arange(bootstrap_block)[None, None, :]
        boostraps = boostraps.reshape(n_bootstraps, boostrap_len)
    else:
        boostraps = rng.choice(all_times.shape[0], size=(n_bootstraps, boostrap_len))
    ts_bootstrapped = all_times[boostraps.flatten()].to_frame()
    ts_bootstrapped = ts_bootstrapped.with_columns(
        sample_index=pl.Series(
            values=np.arange(len(ts_bootstrapped)) // boostrap_len, dtype=pl.UInt32
        ),
        inside_index=pl.Series(
            values=np.arange(len(ts_bootstrapped)) % boostrap_len, dtype=pl.UInt32
        ),
    )
    columns = ["sample_index", "inside_index", "time"]
    ts_bootstrapped = pl.concat(
        [
            ts_bootstrapped[columns],
            times[columns],
        ]
    )
    jets = _gather_normal_da_jets_wrapper(
        jets, ts_bootstrapped, da, n_interp=n_interp, clim=clim
    )
    jets = (
        jets.group_by(
            ["sample_index", pl.col("is_polar") > 0.5, "norm_index", "n"],
            maintain_order=True,
        )
        .agg(pl.col(varname).mean())
        .sort("sample_index", "is_polar", "norm_index", "n")
    )
    jets = (
        jets[["sample_index"]]
        .unique()
        .join(jets[["is_polar"]].unique(), how="cross")
        .join(jets[["norm_index"]].unique(), how="cross")
        .join(jets[["n"]].unique(), how="cross")
        .sort("sample_index", "is_polar", "norm_index", "n")
        .join(jets, on=["sample_index", "is_polar", "norm_index", "n"], how="left")
    )
    pvals = jets.group_by(
        [pl.col("is_polar") > 0.5, "norm_index", "n"], maintain_order=True
    ).agg(
        pl.col(varname).head(n_bootstraps).sort().search_sorted(pl.col(varname).get(-1))
        / n_bootstraps
    )
    jets = jets.filter(pl.col("sample_index") == n_bootstraps).drop("sample_index")
    jets = jets.with_columns(pvals=pvals[varname])
    return polars_to_xarray(jets, index_columns=["is_polar", "norm_index", "n"])
