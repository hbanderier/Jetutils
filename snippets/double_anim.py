r1 = len(da_s.lon) / len(da_s.lat)
r2 = 1
height = 4
wspace = 0.04
cbar_size = 0.05
fig = plt.figure(
    figsize=(height * (r1 + r2 + 2 * cbar_size + 3 * wspace), height), dpi=90
)
gs = GridSpec(
    1,
    5,
    width_ratios=(r1, cbar_size, 4 * wspace, r2, cbar_size),
    wspace=0.02,
    figure=fig,
)

fig.add_subplot(gs[2], visible=False)
ax_contourf = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax_contourf.set_extent(
    exp_s.region,
    crs=ccrs.PlateCarree(),
)
ax_contourf.add_feature(COASTLINE)
cs = ax_contourf.contourf(
    da_s.lon.values, da_s.lat.values, da_s[tsteps[0]].values, **kwargs_contourf
)
cax = fig.add_subplot(gs[1])
fig.colorbar(cs, cax=cax)

ax_trajectory = fig.add_subplot(gs[3])
# ax_trajectory.plot([1, 2])
kwargs_trajectory = dict(
    cmap=mpl.colormaps["gray_r"], norm=Normalize(np.amin(thesepops), np.amax(thesepops))
)
xlims = [
    np.amin(coords[~outermask][:, 0]) - 0.8,
    np.amax(coords[~outermask][:, 0]) + 0.8,
]
ylims = [np.amin(coords[~outermask][:, 1]) - 1, np.amax(coords[~outermask][:, 1]) + 1]
fig, ax_trajectory = splots.plot_map(
    coords,
    populations,
    "hexagons",
    draw_cbar=False,
    show=False,
    edgecolors="black",
    cmap="Greys",
    alphas=alphas,
    linewidths=0,
    fig=fig,
    ax=ax_trajectory,
)
cax = fig.add_axes([0.67, 0.84, 0.06, 0.04])
im = ScalarMappable(**kwargs_trajectory)
fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[])
cax.text(-5, 0.35, "0", ha="center", va="center")
max_pop = np.amax(populations)
cax.text(max_pop + 9, 0.35, f"{max_pop}", ha="center", va="center")

lc = LineCollection(segments, cmap="magma", norm=norm)
lc.set_array(np.repeat(np.arange(len(traj) - 1), repeats=reps))
lc.set_linewidth(3)
lc = ax_trajectory.add_collection(lc)
cax = fig.add_subplot(gs[4])
cbar = fig.colorbar(lc, label=f"Days of summer {YEARSPL[yearidx]}", cax=cax)
list_of_days = np.asarray([0, 14 * 4, 30 * 4, 44 * 4, 61 * 4, 75 * 4, 91 * 4 + 3])

pretty_list_of_days = da.time[yearidx * 92 * 4 + list_of_days].dt.strftime("%b %d").values
cbar.ax.set_yticks(list_of_days, labels=pretty_list_of_days)
cbar.ax.invert_yaxis()
ax_trajectory.set_xlim(xlims)
ax_trajectory.set_ylim(ylims)

jets = all_jets[tsteps[0]]
lines = []
for j in range(3):
    try:
        jet = jets[j]
        x, y, s = jet.T
        p = np.polyfit(x, y, w=s, deg=3, full=False)
        p = np.poly1d(p)
        newy = p(x)
    except IndexError:
        x, newy, s = [], [], []

    lines.append(ax_contourf.plot(x, newy, color='dimgray', lw=4)[0])


def animate(i):
    global cs
    global lines
    global lc
    for c in cs.collections:
        c.remove()
    cs = ax_contourf.contourf(
        da_s.lon.values, da_s.lat.values, da_s[tsteps[i]].values, **kwargs_contourf
    )
    jets = all_jets[tsteps[i]]
    for j in range(3):
        try:
            jet = jets[j]
            x, y, s = jet.T
        except IndexError:
            x, y, s = [], [], []
        lines[j].set_data(x, y)
    if i > 1:
        to_be_set_visible = np.sum(reps[:i])
        lws = np.zeros(len(segments))
        lws[:to_be_set_visible] = 4
        lc.set_lw(lws)
    else:
        lc.set_lw(0.0)
    fig.suptitle(titles[i])
    return cs, lines, lc


ani = FuncAnimation(fig, animate, frames=np.arange(len(tsteps)))
ani.save("Figures/double_anim.gif", dpi=200, fps=5)


# mask = ((times.dt.season == "JJA") & (times.dt.year == year)).values
# indices = np.where(mask)[0]
# jets = all_jets[indices[0] : indices[-1]]  # janky
# times = times[mask]
# flags_ = flags.loc[times].values
# minflag, maxflax = flags_.min(), flags_[flags_ < 13000].max()
# COLORS = colormaps.BlAqGrYeOrReVi200(np.linspace(0, 1, maxflax - minflag + 1))
# flags_ -= minflag

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