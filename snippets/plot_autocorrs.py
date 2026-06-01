fig, axes = plt.subplots(3, 4, figsize=(20, 16), tight_layout=True)
axes = axes.flatten(order="F")
dataset = "ERA5"
datadir = f"{DATADIR}/{dataset}/processed"  # daint
autocorrs = xr.open_dataset(f"{datadir}/Zoo_autocorrs.nc")
howmany = len(autocorrs.coords["lag"])
newlist = []
for key in list(Zoo.data_vars.keys())[:11]:
    for suffix in ["_anomaly"]:
        newlist.append(f"{key}{suffix}")
telatex = r"$T^e_{\rho}$"
tdlatex = r"$T^d_{\rho}$"
tclatex = r"$T^c_{\rho}$"
lw = 2
for i, varname in enumerate(newlist):
    te = np.argmax(autocorrs[varname].values <= 1 / np.exp(1))
    td = 1 + 2 * np.sum(autocorrs[varname])
    tc = 1 + np.sum(
        autocorrs[varname] * (1 - np.arange(1, howmany + 1) / (howmany + 1))
    )
    axes[i].plot(np.arange(howmany), autocorrs[varname], color=COLORS5[0], lw=lw)
    axes[i].plot([te, te], [0, 1], label=telatex, color=COLORS5[2], lw=lw)
    axes[i].plot([tc, tc], [0, 1], label=tclatex, color=COLORS5[3], lw=lw)
    axes[i].plot([td, td], [0, 1], label=tdlatex, color=COLORS5[4], lw=lw)
    axes[i].grid()
    axes[i].legend()
    axes[i].set_title(
        f"{varname}, {telatex}={te}, {tdlatex}={td:.3f}, {tclatex}={tc:.3f}"
    )
    axes[i].set_ylabel("Autocorrelation")
    axes[i].set_xlabel("Lag time [days]")
plt.savefig("Figures/zoo_autocorrs.png")
