from itertools import product
from typing import Any, Mapping, Sequence, Tuple, Union, Iterable
from nptyping import NDArray, Object

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import center_of_mass
import xarray as xr
from xarray import DataArray, Dataset
from contourpy import contour_generator
from tqdm import tqdm, trange

import matplotlib as mpl
from matplotlib import path as mpath
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.patches import PathPatch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import (
    BoundaryNorm,
    Colormap,
    Normalize,
    ListedColormap,
    LinearSegmentedColormap,
    to_rgb,
    to_hex,
    rgb_to_hsv,
    hsv_to_rgb,
)
from matplotlib.container import BarContainer
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import colormaps
import cartopy.crs as ccrs
import cartopy.feature as feat

from jetstream_hugo.definitions import (
    DATADIR,
    PRETTIER_VARNAME,
    infer_direction,
)
from jetstream_hugo.stats import field_significance, field_significance_v2

TEXTWIDTH_IN = 0.0138889 * 503.61377

mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["axes.titlepad"] = 2
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11
mpl.rcParams["figure.titlesize"] = 11
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["animation.ffmpeg_path"] = r"~/mambaforge/envs/env11/bin/ffmpeg"

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

COLORS = np.asarray([colormaps.cet_l_bmw(0.47)[:3], to_rgb("#7a1cfe"), to_rgb("#ff2ec0"), to_rgb("#CE1C66")])
# COLORS = np.append(colormaps.cet_l_bmw([0.2, 0.47])[:, :3], np.asarray([to_rgb("#ff2ec0"), to_rgb("#CE1C66")]), axis=0)
# Dark Blue
# Purple
# Pink
# Pinkish red
COLORS_EXT = np.repeat(COLORS, 3, axis=0)
for i in range(len(COLORS)):
    for sign in [1, -1]:
        newcol_hsv = rgb_to_hsv(COLORS[i][:3]) * (1 + sign * np.asarray([0.0, 0.4, -0.4]))
        newcol_hsv[1] = np.clip(newcol_hsv[1], 0, 1)
        newcol_hsv[2] = np.clip(newcol_hsv[2], 0, 1)
        COLORS_EXT[3 * i + 1 + sign, :3] = hsv_to_rgb(newcol_hsv)
        
COLORS = [to_hex(c) for c in COLORS]
COLORS_EXT = [to_hex(c) for c in COLORS_EXT]

MYBLUES = LinearSegmentedColormap.from_list("myblues", ["#f2f2f2", COLORS[0], COLORS_EXT[2]])
MYPURPLES = LinearSegmentedColormap.from_list("mypurples", ["#f2f2f2", COLORS[1], COLORS_EXT[5]])
MYPINKS = LinearSegmentedColormap.from_list("mypinks", ["#f2f2f2", COLORS[2], COLORS_EXT[8]])
MYREDS = LinearSegmentedColormap.from_list("myreds", ["#f2f2f2", COLORS[3], COLORS_EXT[11]])
PINKPURPLE = LinearSegmentedColormap.from_list("pinkpurple", [COLORS[2], COLORS[1]])
BLUEWHITERED = LinearSegmentedColormap.from_list("bluewhitered", [COLORS_EXT[11], COLORS[3], "#f2f2f2", COLORS[0], COLORS_EXT[2]])


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
    

def p_to_tex(c1: float, c0: float, no_intercept: bool=True) -> str:
    coef1 = num2tex(c1)
    if no_intercept:
        return f"$y\sim {coef1}\cdot x$"
    coef0 = num2tex(c0)
    sign = "+" if np.sign(c0) else "-"
    return f"$y={coef1}\cdot x {sign} {coef0}$"

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
    edge = [np.linspace(minlon, maxlon, n), np.full(n, maxlat)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # West (N->S)
    edge = [np.full(n, maxlon), np.linspace(maxlat, minlat, n)]
    boundary_path += [[i, j] for i, j in zip(*edge)]

    # South (W->E)
    edge = [np.linspace(maxlon, minlon, n), np.full(n, minlat)]
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
    ncol, nrow, ratio: float = 1.4, subplot_kw: dict = None
) -> Tuple[Figure, NDArray[Any, Object]]:
    fig = plt.figure(figsize=(4.5 * nrow, 4.5 * ratio * nrow))
    gs = GridSpec(nrow, 2 * ncol + 1, hspace=0, wspace=0)
    axes = np.empty((ncol, nrow), dtype=object)
    if subplot_kw is None:
        subplot_kw = {}
    for i, j in product(range(ncol), range(nrow)): # TODO: new SOM ordering
        if j % 2 == 0:
            slice_x = slice(2 * i, 2 * i + 2)
        else:
            slice_x = slice(2 * i + 1, 2 * i + 2 + 1)
        axes[i, j] = fig.add_subplot(gs[nrow - j - 1, slice_x], **subplot_kw)
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



# def infer_extent(
#     to_plot: list, direction: int, q: float=0.99
# ) -> Tuple[int, float, float]:  # A worse MaxNLocator
    
#     lowbound, highbound = np.nanquantile(to_plot, q=[1 - q, q])
#     try:
#         lmax = np.log10(max(np.abs(lowbound), np.abs(highbound)))
#         lmax = int(np.round(lmax))
#     except OverflowError:
#         return (6, 0, 1)

#     if direction == 0:
#         lowbound, highbound = 0, max(np.abs(lowbound), np.abs(highbound))
#         lowbound = 0
#     elif direction == -1:
#         lowbound, highbound = 0, np.abs(lowbound)

#     num_digits = 1000
#     for minus in [0.5, 1, 1.5, 2]:
#         if minus == int(minus) and direction == 0:
#             max_rounded = np.ceil(highbound * 10 ** (-lmax + minus)) * 10 ** (lmax - minus)
#             min_rounded = 0
#         else:
#             minus = int(np.ceil(minus))
#             min_rounded = (
#                 np.floor(lowbound * 10 ** (-lmax + minus) / 5) * 5 * 10 ** (lmax - minus)
#             )
#             max_rounded = (
#                 np.ceil(highbound * 10 ** (-lmax + minus) / 5) * 5 * 10 ** (lmax - minus)
#             )
#         extent = max_rounded - min_rounded
#         minnlev = 4 if direction == 0 else 6
#         maxnlev = 7 if direction == 0 else 9
#         distance = np.abs(highbound - max_rounded)
#         for nlev in range(minnlev, maxnlev):
#             try:
#                 firstlev = np.round(extent / (nlev - 1), decimals=6)
#                 if np.isclose(firstlev, np.round(firstlev, 0)):
#                     firstlev = int(np.round(firstlev, 0))
#                 cand_nd = len(str(firstlev).rstrip("0"))
#             except ZeroDivisionError:
#                 cand_nd = 1000
#             if cand_nd < num_digits or (
#                 cand_nd == num_digits
#                 and distance < winner_dist
#             ):
#                 winner = (nlev, min_rounded, max_rounded)
#                 num_digits = cand_nd
#                 winner_dist = distance
#     if direction == -1:
#         winner = (winner[0], -winner[2], winner[1])
#     return winner


def create_levels(
    to_plot: list, levels: int | Sequence | None = None, q: float = 0.99
) -> Tuple[NDArray, NDArray, str, int]:
    if to_plot[0].dtype == bool:
        return np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), 'neither', 1
    extend = {-1: "min", 0: "both", 1: "max"}
    direction = infer_direction(to_plot)
    extend = extend[direction] if q < 1 else "neither"
    if isinstance(levels, Sequence):
        levelsc = np.asarray(levels)
        if direction == 0:
            levelscf = np.delete(levelsc, len(levelsc) // 2)
        else:
            levelscf = levelsc
        return levelsc, levelscf, extend, direction
    
    if levels is None:
        levels = 7 if direction is None else 4
        
    lowbound, highbound = np.nanquantile(to_plot, q=[1 - q, q])
    lowbound = 0 if direction == 1 else lowbound
    highbound = 0 if direction == -1 else highbound
    levelsc = MaxNLocator(levels, symmetric=(direction==0)).tick_values(lowbound, highbound)
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
    to_plot: list,
    lon: NDArray | None,
    lat: NDArray | None,
):
    if lon is None or lat is None:
        try:
            lon = to_plot[0].lon.values
            lat = to_plot[0].lat.values
        except AttributeError:
            print("Either provide lon / lat or make to_plot items dataArrays")
            raise
    return lon, lat


def to_prettier_order(n: int | NDArray, width: int = 6, height: int = 4):
    col, row = divmod(n, height)
    row = height - 1 - row
    return 1 + width * row + col


# def inv_prettier_order(n: int | NDArray, width: int = 6, height: int = 4):
#     col, row = divmod(n, height)
#     row = height - 1 - row
#     return 1 + width * row + col


class Clusterplot:
    def __init__(
        self,
        nrow: int,
        ncol: int,
        region: NDArray | list | tuple = None,
        lambert_projection: bool = False,
        honeycomb: bool = False,
        numbering: bool = False,
        coastline: bool = True,
    ) -> None:
        self.nrow = nrow
        self.ncol = ncol
        self.lambert_projection = lambert_projection
        if region is None:
            region = (-60, 70, 20, 80)  # Default region ?
        self.region = region
        self.minlon, self.maxlon, self.minlat, self.maxlat = region
        if self.lambert_projection:
            self.central_longitude = (self.minlon + self.maxlon) / 2
            projection = ccrs.LambertConformal(
                central_longitude=self.central_longitude,
            )
            ratio = 0.6 * self.nrow / (self.ncol + (0.5 if honeycomb else 0))
            self.boundary = make_boundary_path(*region)
        else:
            projection = ccrs.PlateCarree()
            ratio = (
                (self.maxlat - self.minlat)
                / (self.maxlon - self.minlon)
                * self.nrow
                / (self.ncol + (0.5 if honeycomb else 0))
                * (0.8 if honeycomb else 1)
            )
        if honeycomb:
            self.fig, self.axes = honeycomb_panel(
                self.ncol, self.nrow, ratio, subplot_kw={"projection": projection}
            )
        else:
            self.fig, self.axes = plt.subplots(
                self.nrow,
                self.ncol,
                figsize=(6.5 * self.ncol, 6.5 * self.ncol * ratio),
                constrained_layout=not lambert_projection,
                subplot_kw={"projection": projection},
            )
        self.axes = np.atleast_1d(self.axes).flatten()
        for ax in self.axes:
            if self.lambert_projection:
                ax.set_boundary(self.boundary, transform=ccrs.PlateCarree())
            else:
                ax.set_extent(
                    [self.minlon, self.maxlon, self.minlat, self.maxlat],
                    crs=ccrs.PlateCarree(),
                )
            if coastline:    
                ax.add_feature(COASTLINE)
            # ax.add_feature(BORDERS, transform=ccrs.PlateCarree())
        if numbering:
            plt.draw()
            for i, ax in enumerate(self.axes):
                j = to_prettier_order(i, ncol, nrow)
                ax.annotate(str(j), (2.2, 4), xycoords='axes points', ha="left", va="baseline", fontweight="demi", fontsize=12, bbox={"boxstyle": "square, pad=0.1", "edgecolor": "none", "facecolor": "white"}, usetex=False)            

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
            ax.set_title(title, fontsize=16)
            
    def resize_relative(self, ratios=Sequence[float]):
        self.fig.set_size_inches(self.fig.get_size_inches() * np.asarray(ratios))
        
    def resize_absolute(self, size=Sequence[float]):
        self.fig.set_size_inches(np.asarray(size))

    def add_contour(
        self,
        to_plot: list,
        lon: NDArray = None,
        lat: NDArray = None,
        levels: int | Sequence | None = None,
        clabels: Union[bool, list] = False,
        draw_gridlines: bool = False,
        titles: Iterable = None,
        colors: list | str = None,
        linestyles: list | str = None,
        q: float=0.99,
        **kwargs,
    ) -> None:
        lon, lat = setup_lon_lat(to_plot, lon, lat)  # d r y too much

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
        if 'cmap' in kwargs and kwargs['cmap'] is not None:
            colors=None

        for ax, toplt in zip(self.axes, to_plot):
            cs = ax.contour(
                lon,
                lat,
                toplt,
                transform=ccrs.PlateCarree(),
                levels=levelscf,
                colors=colors,
                linestyles=linestyles,
                linewidths=2.0,
                **kwargs,
            )

            if isinstance(clabels, bool) and clabels:
                ax.clabel(cs, fontsize=11)
            elif isinstance(clabels, list):
                ax.clabel(cs, levels=clabels, fontsize=11)

            if self.lambert_projection and self.boundary is not None:
                ax.set_boundary(self.boundary, transform=ccrs.PlateCarree())

        if titles is not None:
            self._add_titles(titles)

        if draw_gridlines:
            self._add_gridlines()

    def setup_contourf(
        self,
        to_plot: list,
        levels: int | Sequence | None = None,
        cmap: str | Colormap = DEFAULT_COLORMAP,
        transparify: bool | float | int = False,
        contours: bool = False,
        clabels: Union[bool, list] = None,
        cbar_label: str = None,
        cbar_kwargs: Mapping = None,
        q: float=0.99,
        **kwargs,
    ) -> Tuple[Mapping, Mapping, ScalarMappable, NDArray]:
        levelsc, levelscf, extend, direction = create_levels(to_plot, levels, q=q)

        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        if transparify:
            if isinstance(transparify, int):
                cmap = make_transparent(
                    cmap, nlev=len(levelscf), n_transparent=transparify, direction=direction
                )
            elif isinstance(transparify, float):
                cmap = make_transparent(
                    cmap, nlev=len(levelscf), alpha_others=transparify, direction=direction
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
                transform=ccrs.PlateCarree(),
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
        to_plot: list,
        lon: NDArray = None,
        lat: NDArray = None,
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
        q: float=0.99,
        **kwargs,
    ) -> Tuple[ScalarMappable, Mapping]:
        lon, lat = setup_lon_lat(to_plot, lon, lat)

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
                ax.set_boundary(self.boundary, transform=ccrs.PlateCarree())

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
        mask: NDArray,
        FDR: bool = True,
        color: str | list = "black",
        hatch: str = "..",
    ) -> None:
        to_test = []
        for mas in mask.T:
            if np.sum(mas) < 1:
                to_test.append(da[:1].copy(data=np.zeros((1, *da.shape[1:]))))
                continue
            to_test.append(da.isel(time=mas))

        lon = da.lon.values
        lat = da.lat.values
        significances = []
        da = da.values
        # da = np.sort(da, axis=0)
        for i in trange(mask.shape[1]):
            significances.append(
                field_significance(to_test[i], da, 100, q=0.01)[int(FDR)]
            )

        for ax, signif in zip(self.axes, significances):
            cs = ax.contourf(
                lon,
                lat,
                signif,
                levels=3,
                hatches=["", hatch],
                colors="none",
            )

            for col in cs.collections:
                col.set_edgecolor(color)
                col.set_linewidth(0.0)

    def add_any_contour_from_mask(
        self,
        da: DataArray,
        mask: NDArray,
        type: str = "contourf",
        stippling: bool | str = False,
        **kwargs,
    ) -> ScalarMappable | None:
        to_plot = []
        time_name = "time" if "time" in da.dims else da.dims[0]
        for mas in tqdm(mask.T, total=mask.shape[1]):
            if np.sum(mas) < 1:
                to_plot.append(da[0].copy(data=np.zeros(da.shape[1:])))
                continue
            to_plot.append(da.isel({time_name: mas}).mean(dim=time_name))
            
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
        coords: NDArray,
        clu_labs: NDArray,
        cmap: str | Colormap = None,
    ) -> None:
        unique_labs = np.unique(clu_labs)
        sym = np.any(unique_labs < 0)

        if cmap is None:
            cmap = "PiYG" if sym else "Greens"
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        nabove = np.sum(unique_labs > 0)
        if isinstance(cmap, list | NDArray):
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


def cdf(timeseries: Union[DataArray, NDArray]) -> Tuple[NDArray, NDArray]:
    """Computes the cumulative distribution function of a 1D DataArray

    Args:
        timeseries (xr.DataArray or npt.NDArray): will be cast to ndarray if DataArray.

    Returns:
        x (npt.NDArray): x values for plotting,
        y (npt.NDArray): cdf of the timeseries,
    """
    if isinstance(timeseries, DataArray):
        timeseries = timeseries.values
    idxs = np.argsort(timeseries)
    y = np.cumsum(idxs) / np.sum(idxs)
    x = timeseries[idxs]
    return x, y
