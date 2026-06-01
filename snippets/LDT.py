fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True, sharey=False)
bins = [np.arange(-10, 10.5, 0.25), np.arange(-8, 8.01, 0.25)]
for i, series in enumerate([JLI, JSI]):
    ax = axes[i]
    for j, n in enumerate(range(15, 121, 15)):
        coarsened = series.rolling(time=n).mean()

        x, gkde = kde(coarsened, bins=bins[i], return_x=True)
        y = -np.log(gkde) / n
        y -= np.amin(y)
        ax.plot(x, y, color=COLORS10[j], label=f"n={n}")
    ax.set_title(series.name)
    ax.legend()

coarsened = JSI.coarsen(time=60, boundary="trim").mean()
x, gkde = kde(coarsened, bins=np.arange(-10, 10.01, 0.1), return_x=True)
y = -np.log(gkde) / n
a, b, c = np.polyfit(x, y, deg=2, w=1 / y)

plt.plot(x, y)
plt.plot(x, a * x * x + b * x + c)

plt.plot(x, gkde)
x2 = np.linspace(-25, 25, 201)
plt.plot(x, np.exp(-n * (a * x * x + b * x + c)))
plt.plot(x, normal_dist(loc=np.mean(coarsened), scale=np.std(coarsened)).pdf(x))
