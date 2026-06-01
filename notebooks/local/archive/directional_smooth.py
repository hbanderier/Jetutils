def angle_dependent_smoothing(ds, variable: str, sigma_tangent: float = 2.0, sigma_normal: float = 2.0, num_angles: int = 20, radius: int | None = None):
    angle = np.atan2(ds["v"], ds["u"]).compute() + np.pi / 2
    angles = np.nanquantile(angle.where(ds["s"] > 30), np.linspace(0.01, 0.99, num_angles))
    angle = angle.values
    data = ds[variable].values
    
    if radius is None:
        radius = int(4 * max(sigma_tangent, sigma_normal))
    if radius % 2 == 0:
        radius = radius - 1
    X = np.arange(radius) - radius // 2
    Y = X.copy()
    X = X[None, :, None]
    Y = Y[None, None, :]
    
    angles_ = angles[:, None, None] 
    a = np.cos(angles_) ** 2 / (2 * sigma_tangent ** 2) + np.sin(angles_) ** 2 / (2 * sigma_normal ** 2)
    b = - np.sin(2 * angles_) / (4 * sigma_tangent ** 2) + np.sin(2 * angles_) / (4 * sigma_normal ** 2)
    c = np.sin(angles_) ** 2 / (2 * sigma_tangent ** 2) + np.cos(angles_) ** 2 / (2 * sigma_normal ** 2)
    g = (a * X ** 2) + (2 * b * X * Y) + (c * Y ** 2) # parentheses for readability
    g = np.transpose(g, (0, 2, 1))
    g = np.exp(-g) / (2 * np.pi * sigma_tangent * sigma_normal)
    
    fdata = np.fft.fft2(data)[None, :, :, :]
    fg = np.fft.fft2(g, s=data.shape[-2:])[:, None, :, :]
    result = np.real(np.fft.ifft2(fg * fdata))
    where = np.argmin(np.abs(angle[None, ...] - angles[:, None, None, None]), axis=0)
    result = np.take_along_axis(result, where[None, ...], axis=0)[0]
    return result


from numba import njit, prange

@njit
def create_gaussian(sigma_tangent, sigma_normal, x, y, angle):
    x = x[:, None]
    y = y[None, :]
    a = np.cos(angle) ** 2 / (2 * sigma_tangent ** 2) + np.sin(angle) ** 2 / (2 * sigma_normal ** 2)
    b = - np.sin(2 * angle) / (4 * sigma_tangent ** 2) + np.sin(2 * angle) / (4 * sigma_normal ** 2)
    c = np.sin(angle) ** 2 / (2 * sigma_tangent ** 2) + np.cos(angle) ** 2 / (2 * sigma_normal ** 2)
    f = (a * x ** 2) + (2 * b * x * y) + (c * y ** 2) # parentheses for readability
    return np.exp(-f) / (np.sqrt(2 * np.pi) * sigma_tangent * sigma_normal)

@njit(fastmath=True, parallel=True)
def conv_with_angle(data, sigma_tangent, sigma_normal, angles):
    N = int(max(sigma_tangent * 2 + 0.5, sigma_normal * 2 + 0.5))
    if N % 2 == 0:
        N = N - 1
    X = np.arange(N) - N // 2
    Y = X.copy()
    to_ret = np.zeros_like(data)
    for i in prange(data.shape[0]):
        for j in range(data.shape[1]):
            g = create_gaussian(sigma_tangent, sigma_normal, X, Y, angles[i, j])
            
            for n, x in enumerate(X):
                if i - x < 0 or i - x >= data.shape[0]:
                    continue
                for m, y in enumerate(Y):
                    if j - y < 0 or j - y >= data.shape[1]:
                        continue
                    to_ret[i, j] = to_ret[i, j] + data[i - x, j - y] * g[n, m]
    return to_ret


def directional_smooth(ds, of: str, sigma_tangent: float = 2.0, sigma_normal: float = 2.0, radius: int | None = None):
    if radius is None:
        radius = int(4 * max(sigma_tangent, sigma_normal))
    if radius % 2 == 0:
        radius = radius - 1
    x = np.arange(radius) - radius // 2
    pad = radius // 2
    y = x.copy()
    X = x[None, None, :, None]
    Y = y[None, None, None, :]

    angles = np.atan2(ds["v"], ds["u"]).values
    angles_ = angles[:, :, None, None]
    a = np.cos(angles_) ** 2 / (2 * sigma_tangent ** 2) + np.sin(angles_) ** 2 / (2 * sigma_normal ** 2)
    b = np.sin(2 * angles_) / (4 * sigma_tangent ** 2) - np.sin(2 * angles_) / (4 * sigma_normal ** 2)
    c = np.sin(angles_) ** 2 / (2 * sigma_tangent ** 2) + np.cos(angles_) ** 2 / (2 * sigma_normal ** 2)
    g = (a * X ** 2) + (2 * b * X * Y) + (c * Y ** 2) # parentheses for readability
    g = np.exp(-g) / (2 * np.pi * sigma_tangent * sigma_normal)
    
    data = ds[of]
    data_ = np.pad(data, (pad, pad))
    data_ = np.lib.stride_tricks.as_strided(data_, (radius, radius, *data.shape), strides=data_.strides * 2)
    data_ = np.einsum("ijkl,ijkl->kl", data_, g.transpose(3, 2, 0, 1)[::-1, ::-1, ...])
    return ds[of].copy(data=data_)


from string import ascii_lowercase, ascii_uppercase
def create_gaussians(x, y, angles, sigma_tangent, sigma_normal):
    a = np.cos(angles) ** 2 / (2 * sigma_tangent ** 2) + np.sin(angles) ** 2 / (2 * sigma_normal ** 2)
    b = np.sin(2 * angles) / (4 * sigma_tangent ** 2) - np.sin(2 * angles) / (4 * sigma_normal ** 2)
    c = np.sin(angles) ** 2 / (2 * sigma_tangent ** 2) + np.cos(angles) ** 2 / (2 * sigma_normal ** 2)
    g = (a * x ** 2) + (2 * b * x * y) + (c * y ** 2) # parentheses for readability
    g = np.exp(-g) / (2 * np.pi * sigma_tangent * sigma_normal)
    return g


def directional_smooth_dask(ds: xr.Dataset, of: str, sigma_tangent: float = 2.0, sigma_normal: float = 2.0, radius: int | None = None):
    ds = ds.chunk("auto")
    dims = ds[of].dims
    data = ds[of].data
    angles = np.atan2(ds["v"], ds["u"]).data
    axis_lon = find_axis(ds, "lon")
    axis_lat = find_axis(ds, "lat")
    num_index_axes = min(axis_lon, axis_lat)
    axis_y = len(dims)
    axis_x = axis_y + 1

    if radius is None:
        radius = int(3 * max(sigma_tangent, sigma_normal))
    if radius % 2 == 0:
        radius = radius - 1
    radius = max(radius, 1)
    x = np.arange(radius) - radius // 2
    pad = radius // 2
    y = x.copy()
    
    x = darr.expand_dims(x, tuple(np.arange(axis_y)) + (axis_x,))
    y = darr.expand_dims(y, tuple(np.arange(axis_x)))
    angles = darr.expand_dims(angles, (axis_x, axis_y))
    
    g = create_gaussians(x, y, angles, sigma_tangent, sigma_normal)
    transpose_args = [*list(range(num_index_axes + 2)), num_index_axes + 3, num_index_axes + 2]
    g = g.transpose(*transpose_args)
    chunks = tuple([1] * num_index_axes) + (20, -1, -1, -1, -1)
    
    padding = tuple([(pad, pad) if i in [axis_lon, axis_lat] else (0, 0) for i in range(data.ndim)])
    data_ = darr.pad(data, padding)
    data_ = sliding_window_view(data_, (radius, radius), axis=(-2, -1))
    data_ = data_.rechunk(chunks)
    g = g.rechunk(chunks)
    
    letters_index = ascii_uppercase[:num_index_axes]
    res = np.einsum(f"{letters_index}ijkl,{letters_index}ijkl->{letters_index}ij", data_, g)
    res = ds[of].copy(data=res)
    return res


from matplotlib.patches import Rectangle

ds = wind_sample_.isel(time=0)
radius = None
sigma_tangent, sigma_normal = 10, 3
ds = ds.chunk("auto")
of = "s"
dims = ds[of].dims
data = ds[of].data
angles = np.atan2(ds["v"], ds["u"]).data
axis_lon = find_axis(ds, "lon")
axis_lat = find_axis(ds, "lat")
axis_y = len(dims)
axis_x = axis_y + 1

if radius is None:
    radius = int(3 * max(sigma_tangent, sigma_normal))
if radius % 2 == 0:
    radius = radius - 1
x = np.arange(radius) - radius // 2
pad = radius // 2
y = x.copy()

x = darr.expand_dims(x, tuple(np.arange(axis_y)) + (axis_x,))
y = darr.expand_dims(y, tuple(np.arange(axis_x)))
angles = darr.expand_dims(angles, (axis_x, axis_y))

g = create_gaussians(x, y, angles, sigma_tangent, sigma_normal)
g = g.transpose(0, 1, 3, 2)

padding = tuple([(pad, pad) if i in [axis_lon, axis_lat] else (0, 0) for i in range(data.ndim)])
data_ = darr.pad(data, padding)
data_ = sliding_window_view(data_, (radius, radius), axis=(-2, -1))

j, i = np.random.choice(len(ds.lon)), np.random.choice(len(ds.lat))
plt.pcolormesh(x.squeeze(), y.squeeze(), g[i, j])
angle_ = angles[i, j].compute().item()
plt.arrow(0, 0, 0.2 * ds["u"][i, j], .2 * ds["v"][i, j], width=0.2)

# plt.figure()
# plt.pcolormesh(x.squeeze(), y.squeeze(), data_[i, j], vmin=0, vmax=60)
# plt.arrow(0, 0, 0.2 * ds["u"][i, j], .2 * ds["v"][i, j], width=0.2)

plt.figure()
lo, la = ds.lon.values, ds.lat.values
plt.pcolormesh(lo, la, ds[of], vmin=0, vmax=60)
plt.quiver(lo[::4], la[::4], ds["u"][::4, ::4], ds["v"][::4, ::4])

center = lo[j], la[i]
x0 = lo[j] - radius / 4
y0 = la[i] - radius / 4
plt.gca().add_artist(Rectangle((x0, y0), radius / 2, radius / 2, facecolor="none", edgecolor=COLORS[2], linewidth=3))