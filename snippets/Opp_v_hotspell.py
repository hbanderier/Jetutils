"""
A notebook cell that needs to be redone
"""


fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.subplots_adjust(wspace=0.03, hspace=0.25)
axes = axes.flatten()
means = OPP_mask_da_T1.isel(OPP=np.arange(6)).mean(dim='hotspell')
quantiles = OPP_mask_da_T1.isel(OPP=np.arange(6)).quantile([0.05, 0.95], dim='hotspell')
pops = (~OPP_mask_da_T1.isnull()).isel(OPP=0).sum(dim='hotspell')
pop_cmap = ListedColormap(mpl.colormaps['magma'].reversed()(np.linspace(0, 1, 11)))
for i, (ax, region) in enumerate(zip(axes, REGIONS)):
    for OPP, color in zip(range(6), COLORS5):
        this = means.sel(region=region).isel(OPP=OPP)
        maxidx = np.sum(~np.isnan(this.values))
        this = this[:maxidx]
        q1, q2 = quantiles.sel(region=region).isel(OPP=OPP)[:, :maxidx]
        ax.plot(this.day_after_beg, this, lw=2, color=color, label=f'OPP n°{OPP + 1}')
        ax.fill_between(this.day_after_beg, q1, q2, color=color, alpha=0.2)
    ax.set_title(region)
    im = ax.pcolormesh(np.arange(pops.shape[1] + 1)[:(maxidx + 1)] - 10.5, [-2.8, -2.6], pops.sel(region=region).values[:maxidx, None].T, cmap=pop_cmap, vmin=1, vmax=pops.max(), norm='log')
    ax.set_xlim(this.day_after_beg[[0, -1]].values)
    ax.set_ylim([-2.8, 2.5])
    if i % 3 == 0:
        ax.set_yticks([-2, -1, 0, 1, 2])
    else:
        ax.set_yticks([-2, -1, 0, 1, 2], [''] * 5)
    if i > 2:
        ax.set_xlabel('Days around beg.')
    ax.grid(True, axis='y')
leg = axes[2].legend(ncol=1, loc='upper left', bbox_to_anchor=(1.03, 1.02))

width = 0.02
xmin = 1 + width * 2 - axes[3].get_position().xmin
height = 0.5
ymin = axes[-1].get_position().ymin
cax = fig.add_axes([xmin, ymin, width, height])
fig.colorbar(im, cax=cax, label='Hotspells still alive')