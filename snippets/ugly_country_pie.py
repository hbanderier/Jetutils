from matplotlib.patches import Patch, Polygon
from matplotlib.collections import PolyCollection

clu = Clusterplot(1, 1, get_region(clusters_da), coastline=False)
ax0 = clu.axes[0]
cmap = colormaps.BlAqGrYeOrReVi200
unique_clusters = np.arange(n_clu)
norm = BoundaryNorm(np.concatenate([[-1], unique_clusters]) + 0.5, cmap.N)
clusters_da = exp_T.spatial_clusters_as_da(n_clu)
clusters_da.unstack().plot(
    ax=ax0,
    colors="gainsboro", 
    levels=[0.99, 1.5],
    # cmap=cmap,
    # norm=norm,
    add_colorbar=False,
    add_labels=False,
    alpha=0.5
)

score_name = "balanced_accuracy_score"
to_plot = [combination[0][score_name].item() for combination in best_combination.values()]
to_plot = assign_to_regions(clusters_da, to_plot)
cmap_score = colormaps.bubblegum_r
levels = MaxNLocator(6).tick_values(to_plot.min(), to_plot.max())
cbar_kwargs = {"shrink": 0.84, "label": score_name, "pad": 0.02}
norm_score = BoundaryNorm(levels, cmap_score.N)
clu.fig.colorbar(ScalarMappable(norm_score, cmap_score), ax=ax0, **cbar_kwargs)

all_preds = np.arange(len(predictors.predictor))
cmap = colormaps.bold
norm = BoundaryNorm(np.arange(cmap.N) - 0.5, cmap.N)
colors = cmap(norm(all_preds // 2))
colors = pd.Series(dict(zip(predictors.predictor.values, colors)))
hatches = np.asarray(["", "/////"])[all_preds % 2]
hatches = pd.Series(dict(zip(predictors.predictor.values, hatches)))
dx, dy = 20, 20
size = 1.8
pie_kwargs1 = {
    "wedgeprops": dict(width=size, edgecolor='w', clip_on=True),
    "radius": 2.,
}
pie_kwargs2 = {
    "radius": 2. - size, "wedgeprops": dict(clip_on=True),
}
patch = dict(boxstyle='round', facecolor='white', alpha=1.0)
class CustomPatchAxes(Axes):
    def __init__(
        self, 
        *args,
        custom_patch,
        **kwargs,
    ):
        self.custom_patch = custom_patch
        super().__init__(*args, **kwargs)
        
    def _gen_axes_patch(self):
        return self.custom_patch
patches = []
stuff = []
for regionkey, combination in best_combination.items():
    n = int(regionkey.split("=")[-1])
    a = ax0.contour(lon, lat, (clusters_da == n).astype(int), colors="grey", levels=[0.5], zorder=20, linewidths=3, transform=ccrs.PlateCarree())
    polys = a.get_paths()[0].to_polygons()
    imax = np.argmax([len(poly) for poly in polys])
    custom_patch = polys[imax]
    polymean = (custom_patch.max(axis=0) + custom_patch.min(axis=0)) / 2
    lo, la = polymean
    custom_patch = custom_patch - polymean
    ax0.text(lo, la, f"{n + 1}", ha="center", va="center_baseline", ma="center", fontweight="bold", color="white", zorder=14, usetex=False, fontsize=9)
    dx, dy = custom_patch.max(axis=0) - custom_patch.min(axis=0)
    dl = custom_patch.max() - custom_patch.min()
    custom_patch = (custom_patch - custom_patch.min()) / dl
    custom_patch = Polygon(custom_patch)
    stuff.append((lo, la, dx, dy, dl))
    thisax = ax0.inset_axes([lo - dl / 2, la - dl / 2, dl, dl], transform=ax0.transData, zorder=13, axes_class=CustomPatchAxes, custom_patch=custom_patch)
    patches.append(custom_patch)
    score = combination[0][score_name]
    to_app = combination[1]
    keys = combination[2]
    asdict = pd.Series(dict(zip(keys, np.around(to_app.sel(type="impurity").values, 2)))).sort_values(ascending=False)
    is_neg = to_app.sel(type="correlation").values < 0
    colors_ = colors.loc[keys].values
    hatches_ = hatches.loc[keys].values
    hatches_ = hatches_ + np.asarray(["", "..."])[is_neg.astype(int)]
    pred_imp_ = np.clip(asdict.values, 0, None)
    thisax.pie(pred_imp_, colors=colors_, hatch=hatches_, **pie_kwargs1)
    thisax.pie([1.], colors=[cmap_score(norm_score(score))], **pie_kwargs2)
    
legend_elements = [
    Patch(facecolor=cmap(norm(i)), edgecolor="black", label=PRETTIER_VARNAME[varname]) for i, varname in enumerate(subset)
]
jets = props_as_ds.jet.values
legend_elements.extend([
    Patch(facecolor="white", edgecolor="black", label=SHORTHAND[jet], hatch=hatch) for jet, hatch in zip(jets, ["", "/////"])
])
legend_elements.extend([
    Patch(facecolor="white", edgecolor="black", label=sign_, hatch=hatch) for sign_, hatch in zip(["pos. correlation", "neg. correlation"], ["", "..."])
])
clu.fig.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(0.9, 0.5))
clu.fig.set_tight_layout(False)
plt.savefig(f"{FIGURES}/jet_props_hotspells/predictor_importance.png")
clear_output()