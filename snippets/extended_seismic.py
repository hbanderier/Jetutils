base = mpl.colormaps['seismic']
end = LinearSegmentedColormap.from_list('extended_seismic', [base(base.N), '#9f0fff'])(np.linspace(0, 1, 31))
yay = ListedColormap([*base(np.linspace(0, 1, 62)), *end])
net_mask.resample(time='1Y').sum().groupby(xr.DataArray(labels, coords={'cluster': net_mask.cluster})).sum().plot(cmap=yay, lw=.5, edgecolor='white')