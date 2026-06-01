from definitions import *
from xmca.array import MCA
from xmca.xarray import xMCA
from scipy.optimize import basinhopping, minimize, brute, shgo
from numpy.typing import ArrayLike

exp_wind = ClusteringExperiment('ERA5', 'Wind', '300', None, None, -60, 60, 20, 80, False, 'anomaly', 'JJA')
exp_T = ClusteringExperiment('ERA5', 'Temperature', '2m', 'box_-25_60_32_72', 't', None, None, None, None, False, 'anomaly', 'JJA')
da_wind = exp_wind.open_da()
da_T = exp_T.open_da()
X, da_wind = exp_wind.prepare_for_clustering()
Y, da_T = exp_T.prepare_for_clustering()

X = exp_wind.pca_transform(X, 150)
Y = exp_T.pca_transform(Y, 100)
pca = MCA(X, Y)
pca.normalize()
pca.solve()
svals_rule_n = pca.rule_n(n_runs=1000)
q99 = np.quantile(svals_rule_n, 0.99, axis=1)
np.argmax(svals < q99)
eofs = pca.eofs(n=80, scaling='max')
pcs = pca.pcs(n=80, scaling='max')
eofs['left'].shape
eofs_realspace = {
    'left': exp_wind.to_dataarray(eofs['left'].T, da_wind, 150),
    'right': exp_T.to_dataarray(eofs['right'].T, da_T, 100),
}
eofs_realspace
to_plot = {key : [eofs_realspace[key].isel(mode=i) for i in range(9)] for key in eofs}

fig, axes, cbar = clusterplot(3, 3, to_plot['left'], 11, 10, clabels=None, contours=False)
fig, axes, cbar = clusterplot(3, 3, to_plot['right'], 6, 3, clabels=None, contours=False)

# to_plot = [da.isel(time=mask_high_val[:, i]).mean(dim='time') for i in range(6)]
# levels = 12
# levels2 = int(levels / 2)
# colors = levels2 * ['sienna'] + (levels2 + 1) * ['green']
# linestyles = levels2 * ['dashed'] + (levels2 + 1) * ['solid']
# for ax, toplt, eigenval in zip(axes, to_plot, eigenvals):
#     toplt.plot.contour(ax=ax, add_colorbar=False, add_labels=False, levels=levels, vmin=-10, vmax=10, colors=colors, linestyles=linestyles)
#     ax.set_title(f'{eigenval:.2f}')