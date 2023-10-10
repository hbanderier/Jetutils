from itertools import product
from typing import Any, Mapping, Tuple, Union, Iterable
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
)
from matplotlib.container import BarContainer
from matplotlib.gridspec import GridSpec
import colormaps as cmaps
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as feat

from definitions import (
    DATADIR,
    REGIONS,
    PRETTIER_VARNAME,
    LATBINS,
    infer_sym,
)
from stats import field_significance, field_significance_v2

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

MYPURPLES = LinearSegmentedColormap.from_list("mypruples", ["#FFFFFF", "#8338EC"])

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

COLOR_JETS = cmaps.bold(np.arange(12))

mpl.rcParams["font.size"] = 18
mpl.rcParams["animation.ffmpeg_path"] = r"~/mambaforge/envs/env11/bin/ffmpeg"


def num2tex(x: float, force: bool = False) -> str:
    float_str = f"{x:.2e}" if force else f"{x:.2g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def make_transparent(
    cmap: str | Colormap,
    nlev: int = None,
    alpha_others: float = 1,
    n_transparent: int = 1,
    sym: bool = False,
) -> Colormap:
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    if nlev is None:
        nlev = cmap.N
    colorlist = cmap(np.linspace(0, 1, nlev + sym))
    if sym:
        midpoint = int(np.ceil(nlev / 2))
        colorlist[midpoint - n_transparent + 1 : midpoint + n_transparent, -1] = 0
        colorlist[midpoint + n_transparent :, -1] = alpha_others
        colorlist[: midpoint - n_transparent + 1, -1] = alpha_others
    else:
        colorlist[:n_transparent, -1] = 0
        colorlist[n_transparent:, -1] = alpha_others

    return ListedColormap(colorlist)


def num2tex(x: float) -> str:
    float_str = f"{x:.2g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


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
    """Write extend of a region lon lat box in a nicer way, plus season

    Args:
        minlon (str): minimum longitude
        maxlon (str): maximum longitude
        minlat (str): minimum latitude
        maxlat (str): maximum latitude
        season (str): season

    Returns:
        str: Nice title
    """
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
    for i, j in product(range(ncol), range(nrow)):
        if j % 2 == 0:
            slice_x = slice(2 * i, 2 * i + 2)
        else:
            slice_x = slice(2 * i + 1, 2 * i + 2 + 1)
        axes[i, j] = fig.add_subplot(gs[nrow - j - 1, slice_x], **subplot_kw)
    return fig, axes


def infer_extent(
    to_plot: list, sym: bool, start: float = None
) -> Tuple[int, float]:  # I could market this
    max = np.nanquantile(to_plot, q=0.95)
    lmax = np.log10(max)
    lmax = int(np.sign(lmax) * np.round(np.abs(lmax)))

    if sym:
        min = 0
    elif start is not None:
        min = start
    else:
        min = np.nanmin(to_plot)

    num_digits = 1000
    for minus in [0.5, 1, 1.5, 2]:
        if minus == int(minus) and sym:
            max_rounded = np.ceil(max * 10 ** (-lmax + minus)) * 10 ** (lmax - minus)
            min_rounded = 0
        else:
            minus = int(np.ceil(minus))
            min_rounded = (
                np.floor(min * 10 ** (-lmax + minus) / 5) * 5 * 10 ** (lmax - minus)
            )
            max_rounded = (
                np.ceil(max * 10 ** (-lmax + minus) / 5) * 5 * 10 ** (lmax - minus)
            )
        extent = max_rounded - min_rounded
        minnlev = 4 if sym else 5
        maxnlev = 8 if sym else 10
        for nlev in range(minnlev, maxnlev):
            try:
                firstlev = np.round(extent / (nlev - 1), decimals=6)
                if np.isclose(firstlev, np.round(firstlev, 0)):
                    firstlev = int(np.round(firstlev, 0))
                cand_nd = len(str(firstlev))
            except ZeroDivisionError:
                cand_nd = 1000
            if cand_nd < num_digits or (
                cand_nd == num_digits
                and np.isclose((firstlev * 10 ** (-lmax + 1)) % 5, 0)
            ):
                winner = (nlev, min_rounded, max_rounded)
                num_digits = cand_nd
    return winner


def create_levels(
    to_plot: list, nlevels: int = None, sym: bool = False, start: float = None
) -> Tuple[NDArray, NDArray, str]:
    if sym is None:
        sym = infer_sym(to_plot)

    nlevels_cand, min_rounded, max_rounded = infer_extent(to_plot, sym, start)

    if nlevels is None:
        nlevels = nlevels_cand

    if sym:
        levels0 = np.delete(
            np.append(
                np.linspace(-max_rounded, 0, nlevels),
                np.linspace(0, max_rounded, nlevels),
            ),
            nlevels - 1,
        )
        levels = np.delete(levels0, nlevels - 1)
        extend = "both"
    else:
        levels0 = np.linspace(min_rounded, max_rounded, nlevels)
        levels = levels0
        extend = "max"
    return levels0, levels, extend, sym


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


class Clusterplot:
    def __init__(
        self,
        nrow: int,
        ncol: int,
        region: NDArray | list | tuple = None,
        lambert_projection: bool = False,
        honeycomb: bool = False,
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
                * 0.8  # for the colorbar padding
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
            ax.add_feature(COASTLINE)
            # ax.add_feature(BORDERS, transform=ccrs.PlateCarree())

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
            gl.xlines = (False,)
            gl.ylines = False
            plt.draw()
            for ea in gl.label_artists:
                current_pos = ea.get_position()
                if ea.get_text()[-1] in ["N", "S"]:
                    ea.set_visible(True)
                    continue
                if current_pos[1] > 4000000:
                    ea.set_visible(False)
                    continue
                ea.set_visible(True)
                ea.set_rotation(0)
                ea.set_position([current_pos[0], current_pos[1] - 200000])

    def _add_titles(self, titles: Iterable) -> None:
        if len(titles) > len(self.axes):
            titles = titles[: len(self.axes)]
        for title, ax in zip(titles, self.axes):
            if isinstance(title, float):
                title = f"{title:.2f}"
            ax.set_title(title)

    def add_contour(
        self,
        to_plot: list,
        lon: NDArray = None,
        lat: NDArray = None,
        nlevels: int = None,
        sym: bool = None,
        start: float = None,
        clabels: Union[bool, list] = False,
        draw_gridlines: bool = False,
        titles: Iterable = None,
        colors: list | str = None,
        linestyles: list | str = None,
        **kwargs,
    ) -> None:
        assert len(to_plot) <= len(self.axes)

        lon, lat = setup_lon_lat(to_plot, lon, lat)  # d r y too much

        levelsc, _, _, sym = create_levels(to_plot, nlevels, sym, start)

        if sym and linestyles is None:
            linestyles = ["dashed", "solid"]

        if sym:
            colors = doubleit(colors, len(levelsc), "black")
            linestyles = doubleit(linestyles, len(levelsc), "solid")

        if not sym and colors is None:
            colors = "black"
        if not sym and linestyles is None:
            linestyles = "solid"
        if 'cmap' in kwargs and kwargs['cmap'] is not None:
            colors=None

        for ax, toplt in zip(self.axes, to_plot):
            cs = ax.contour(
                lon,
                lat,
                toplt,
                transform=ccrs.PlateCarree(),
                levels=levelsc,
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
        nlevels: int = None,
        sym: bool = None,
        start: float = None,
        cmap: str | Colormap = None,
        transparify: bool | float | int = False,
        contours: bool = False,
        clabels: Union[bool, list] = None,
        cbar_label: str = None,
        cbar_kwargs: Mapping = None,
        **kwargs,
    ) -> Tuple[Mapping, Mapping, ScalarMappable, NDArray]:
        levelsc, levelscf, extend, sym = create_levels(to_plot, nlevels, sym, start)

        if cmap is None and sym:
            cmap = "seismic"
        elif cmap is None:
            cmap = MYPURPLES  # Just think it's neat
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        if transparify:
            if isinstance(transparify, int):
                cmap = make_transparent(
                    cmap, nlev=len(levelscf), n_transparent=transparify, sym=sym
                )
            elif isinstance(transparify, float):
                cmap = make_transparent(
                    cmap, nlev=len(levelscf), alpha_others=transparify, sym=sym
                )
            else:
                cmap = make_transparent(cmap, nlev=len(levelscf), sym=sym)

        if cbar_kwargs is None:
            cbar_kwargs = {}

        if cbar_label is not None:  # backwards compat
            cbar_kwargs["label"] = cbar_label

        norm = BoundaryNorm(levelscf, cmap.N, extend=extend)
        im = ScalarMappable(norm=norm, cmap=cmap)

        if contours or clabels is not None:
            self.add_contour(to_plot, nlevels, sym, clabels)

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
        nlevels: int = None,
        sym: bool = None,
        start: float = None,
        cmap: str | Colormap = None,
        transparify: bool | float | int = False,
        contours: bool = False,
        clabels: Union[bool, list] = None,
        draw_gridlines: bool = False,
        draw_cbar: bool = True,
        cbar_label: str = None,
        titles: Iterable = None,
        cbar_kwargs: Mapping = None,
        **kwargs,
    ) -> ScalarMappable:
        assert len(to_plot) <= len(self.axes)

        lon, lat = setup_lon_lat(to_plot, lon, lat)

        kwargs, cbar_kwargs, im, levelsc = self.setup_contourf(
            to_plot,
            nlevels,
            sym,
            start,
            cmap,
            transparify,
            contours,
            clabels,
            cbar_label,
            cbar_kwargs,
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
            self.cbar.ax.set_yticks(levelsc)
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
        to_test = [da.isel(time=mas) for mas in mask.T]

        lon = da.lon.values
        lat = da.lat.values
        significances = []
        da = da.values
        # da = np.sort(da, axis=0)
        for i in trange(mask.shape[1]):
            significances.append(
                field_significance(to_test[i], da, 200, q=0.02)[int(FDR)]
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
        to_plot = [da.isel(time=mas).mean(dim="time") for mas in mask.T]
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


# Create histogram
def compute_hist(
    timeseries: DataArray, season: str = None, bins: Union[NDArray, list] = LATBINS
) -> Tuple[NDArray, NDArray]:
    """small wrapper for np.histogram that extracts a season out of xr.DataArray

    Args:
        timeseries (xr.DataArray): _description_
        season (str): _description_
        bins (list or npt.NDArray): _description_

    Returns:
        bins (npt.NDArray): _description_
        counts (npt.NDArray): _description_
    """
    if season is not None and season != "Annual":
        timeseries = timeseries.isel(time=timeseries.time.dt.season == season)
    return np.histogram(timeseries, bins=bins)


def histogram(
    timeseries: DataArray,
    ax: Axes,
    season: str = None,
    bins: Union[NDArray, list] = LATBINS,
    **kwargs,
) -> BarContainer:
    """Small wrapper to plot a histogram out of a time series

    Args:
        timeseries (xr.DataArray): _description_
        ax (Axes): _description_
        season (str, optional): _description_. Defaults to None.
        bins (Union[NDArray, list], optional): _description_. Defaults to LATBINS.

    Returns:
        BarContainer: _description_
    """
    hist = compute_hist(timeseries, season, bins)
    midpoints = (hist[1][1:] + hist[1][:-1]) / 2
    bars = ax.bar(midpoints, hist[0], width=hist[1][1] - hist[1][0], **kwargs)
    return bars


def kde(
    timeseries: DataArray,
    season: str = None,
    bins: Union[NDArray, list] = LATBINS,
    scaled: bool = False,
    return_x: bool = False,
    **kwargs,
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    hist = compute_hist(timeseries, season, bins)
    midpoints = (hist[1][1:] + hist[1][:-1]) / 2
    norm = (hist[1][1] - hist[1][0]) * np.sum(hist[0])
    kde: NDArray = gaussian_kde(midpoints, weights=hist[0], **kwargs).evaluate(
        midpoints
    )
    if scaled:
        kde *= norm
    if return_x:
        return midpoints, kde
    return kde


def create_double_composite(
    ds: Dataset,
    props_as_ds: Dataset,
    min_hotspell_length: int = 4,
    max_hotspell_length: int = 5,
    subset: list = None,
    fig_kwargs: dict = None,
):
    duncan_mask = np.abs(xr.open_dataarray(f"{DATADIR}/ERA5/cluster_def.nc"))
    ds_masked = ds.where(ds.hotspell_length >= min_hotspell_length).where(
        ds.hotspell_length <= max_hotspell_length
    )
    lon, lat = duncan_mask.lon.values, duncan_mask.lat.values
    inverse_landmask = duncan_mask.copy()
    inverse_landmask[:] = 1
    inverse_landmask = inverse_landmask.where(duncan_mask.isnull())
    centers = {}
    for i, region in enumerate(REGIONS):
        com = center_of_mass((duncan_mask == i + 1).values)
        mean_lat = lat[int(com[0])] + (com[0] % 1) * (lat[1] - lat[0])
        mean_lon = lon[int(com[1])] + (com[1] % 1) * (lon[1] - lon[0])
        duncan_mask = duncan_mask.where(
            (duncan_mask != i + 1) | (duncan_mask.lat < mean_lat), duncan_mask + 0.1
        )
        centers[region] = [mean_lon, mean_lat]
    if subset is None:
        subset = list(props_as_ds.data_vars)
    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = {'nrows': 3, 'ncols': 4, 'figsize': (22, 8)}
    fig_kwargs = default_fig_kwargs | fig_kwargs
    fig, axes = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, **fig_kwargs
    )
    fig.subplots_adjust(wspace=0.03)
    axes = axes.ravel()
    cmap_polar = cmaps.BlueRed
    cmap_subtropical = cmaps.GreenYellow_r
    norm = Normalize(-1.1, 1.1)
    for i, varname in enumerate(subset):
        da_polar = duncan_mask.copy()
        da_subtropical = duncan_mask.copy()
        for ir, region in enumerate(ds_masked.region.values):
            val_polar = (
                ds_masked[varname]
                .sel(jet="polar", region=region, day_after_beg=np.arange(-3, 4))
                .mean()
            )
            val_polar = (
                val_polar - props_as_ds[varname].sel(jet="polar").mean()
            ) / props_as_ds[varname].sel(jet="polar").std()

            val_subtropical = (
                ds_masked[varname]
                .sel(jet="subtropical", region=region, day_after_beg=np.arange(-3, 4))
                .mean()
            )
            val_subtropical = (
                val_subtropical - props_as_ds[varname].sel(jet="subtropical").mean()
            ) / props_as_ds[varname].sel(jet="subtropical").std()

            da_polar = da_polar.where(duncan_mask != ir + 1.1, val_polar)
            da_polar = da_polar.where(duncan_mask != ir + 1, np.nan)
            da_subtropical = da_subtropical.where(
                duncan_mask != ir + 1, val_subtropical
            )
            da_subtropical = da_subtropical.where(duncan_mask != ir + 1.1, np.nan)
        im_polar = axes[i].pcolormesh(
            lon,
            lat,
            da_polar.values,
            shading="nearest",
            transform=ccrs.PlateCarree(),
            norm=norm,
            cmap=cmap_polar,
        )
        im_subtropical = axes[i].pcolormesh(
            lon,
            lat,
            da_subtropical.values,
            shading="nearest",
            transform=ccrs.PlateCarree(),
            norm=norm,
            cmap=cmap_subtropical,
        )
        axes[i].pcolormesh(
            lon,
            lat,
            inverse_landmask,
            shading="nearest",
            transform=ccrs.PlateCarree(),
            cmap="Greys",
            vmin=0.95,
            vmax=1.05,
        )
        axes[i].contour(
            lon,
            lat,
            duncan_mask.values / 6,
            levels=7,
            colors="black",
            lw=0.1,
        )
        axes[i].set_title(PRETTIER_VARNAME[varname], fontsize=30)
        axes[i].axis("off")
    if fig_kwargs['ncols'] * fig_kwargs['nrows'] > len(subset):
        N = 6
        cmap2 = LinearSegmentedColormap.from_list(
            "hehe", cmaps.agsunset(np.linspace(1 / N, 1 - 1 / N, N))
        )
        axes[-1].pcolormesh(
            lon,
            lat,
            duncan_mask.values / N,
            shading="nearest",
            transform=ccrs.PlateCarree(),
            cmap=cmap2,
        )

        for i, region in enumerate(REGIONS):
            axes[-1].text(
                *centers[region],
                region,
                transform=ccrs.PlateCarree(),
                va="bottom" if region == "Arctic" else "center",
                ha="center",
            )
        axes[-1].axis("off")

    cbar_polar = fig.colorbar(im_polar, ax=fig.axes, pad=0.015, fraction=0.04)
    cbar_polar.ax.set_ylabel("Norm. anomaly (polar jet)", fontsize=27)
    cbar_polar.ax.yaxis.set_label_position("left")
    cbar_polar.ax.set_yticks(np.linspace(-1, 1, 9), [""] * 9)
    cbar_polar.ax.yaxis.set_ticks_position("left")
    cbar_subtropical = fig.colorbar(
        im_subtropical, ax=fig.axes, pad=0.007, fraction=0.04
    )
    cbar_subtropical.ax.set_ylabel("Norm. anomaly (subtropical jet)", fontsize=27)
    return fig, axes, cbar_polar, cbar_subtropical
