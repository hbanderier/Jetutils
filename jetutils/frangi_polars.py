import numpy as np
import math
from scipy import signal
import polars as pl
import polars.selectors as cs


def safe_gather_2d(expr: pl.Expr, nx: int, ny: int, x: pl.Series, y: pl.Series):
    two_d_to_one_d = x + nx * y
    return (
        pl.when(x < nx, x >= 0, y < ny, y >= 0)
        .then(expr.gather(two_d_to_one_d.clip(0, nx * ny - 1)))
        .otherwise(0.0)
    )


def pl_convolve_2d(col: pl.Expr, kernel: pl.Expr, x1, y1, x2, y2):
    indices_x = np.arange(-(x2 // 2), x2 // 2 + 1)
    indices_y = np.arange(-(y2 // 2), y2 // 2 + 1)
    image_indices = [
        np.tile(np.arange(x1), y1)[:, None]
        - np.tile(indices_x, len(indices_y))[None, :],
        np.repeat(np.arange(y1), x1)[:, None]
        - np.repeat(indices_y, len(indices_x))[None, :],
    ]
    print(np.asarray(image_indices).max())
    agg = [
        (
            safe_gather_2d(col, x1, y1, pl.Series(None, indx), pl.Series(None, indy))
            * kernel.slice(0, x2 * y2)
        ).sum()
        for indx, indy in zip(*image_indices)
    ]
    return pl.concat_arr(agg)


def maybe_extend(col: pl.Expr, n2: pl.Expr | int) -> pl.Expr:
    n1 = col.count()
    pl.when(n1 >= n2).then(col).otherwise(col.extend_constant(0, n2 - n1))


def add_gaussian_kernels(
    images: pl.DataFrame,
    col: pl.Expr,
    group_by: list[str | pl.Expr],
    x1: int,
    y1: int,
    sigma: pl.Expr = pl.col("sigma").first(),
) -> pl.DataFrame:
    base_grid = pl.int_range(-3 * sigma, 3 * sigma + 1)
    Y = base_grid.repeat_by(6 * sigma + 1).explode()
    X = base_grid.gather(pl.int_range(0, (6 * sigma + 1) * (6 * sigma + 1)) % (6 * sigma + 1))
    
    x2, y2 = 6 * sigma, 6 * sigma
    gauss_xx = (
        1
        / (2 * math.pi * sigma.pow(4))
        * (X.pow(2) / sigma.pow(2) - 1)
        * (-(X.pow(2) + Y.pow(2)) / (2 * sigma.pow(2))).exp()
    )
    gauss_xy = (
        1
        / (2 * math.pi * sigma.pow(6))
        * (X * Y)
        * (-(X.pow(2) + Y.pow(2)) / (2 * sigma.pow(2))).exp()
    )
    gauss_yy = (
        1
        / (2 * math.pi * sigma.pow(4))
        * (Y.pow(2) / sigma.pow(2) - 1)
        * (-(X.pow(2) + Y.pow(2)) / (2 * sigma.pow(2))).exp()
    )
    
    images = images.group_by(group_by, maintain_order=True).agg(
        image=maybe_extend(col, x2 * y2),
        gauss_xx=maybe_extend(gauss_xx, x1 * y1),
        gauss_xy=maybe_extend(gauss_xy, x1 * y1),
        gauss_yy=maybe_extend(gauss_yy, x1 * y1),
        sigma=sigma,
    ).explode(["image", "gauss_xx", "gauss_xy", "gauss_yy"])
    return images


def pl_hessian(
    images: pl.DataFrame,
    col: str | pl.Expr,
    group_by: list[str | pl.Expr],
    x1: int,
    y1: int,
    sigma: pl.Expr = pl.col("sigma").first(),
) -> pl.DataFrame:
    col = pl.col(col)
    images_with_kernels, x2, y2 = add_gaussian_kernels(images, col, group_by, sigma, x1, y1)
    kernels = ["gauss_xx", "gauss_xy", "gauss_yy"]
    aggs = {
        kernel: sigma.pow(2) * pl_convolve_2d(col, kernel, x1, y1, x2, y2) 
        for kernel in kernels
    }
    images_with_kernels = (
        images_with_kernels
        .group_by(group_by, maintain_order=True)
        .agg(**aggs)
    )
    return images_with_kernels


def pl_eig2image(images: pl.DataFrame, group_by: list[str | pl.Expr], Dxx: pl.Expr, Dxy: pl.Expr, Dyy: pl.Expr) -> pl.DataFrame:
    tmp = (Dxx - Dyy).pow(2) + 4 * Dxy.pow(2)
    v2x = 2 * Dxy
    v2y = Dyy - Dxx + tmp
    mag = (v2x ** 2 + v2y ** 2).sqrt()
    
    v2x = pl.when(mag != 0).then(v2x / mag).otherwise(v2x)
    v2y = pl.when(mag != 0).then(v2y / mag).otherwise(v2y)
    v1x = -v2y
    v1y = v2x
    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)
    check = mu1.abs() > mu2.abs()
    Lambda1 = pl.when(check).then(mu2).otherwise(mu1)
    Lambda2 = pl.when(check).then(mu1).otherwise(mu2)
    Ix = pl.when(check).then(v2x).otherwise(v1x)
    Iy = pl.when(check).then(v2y).otherwise(v1y)
    aggs = {
        "Lambda1": Lambda1,
        "Lambda1": Lambda2,
        "Ix": Ix,
        "Iy": Iy,
    }
    return images.group_by(group_by, maintain_order=True).agg(**aggs)


def FrangiFilter2D(image):
    image = np.array(image, dtype=float)
    defaultoptions = {
        "FrangiScaleRange": (1, 10),
        "FrangiScaleRatio": 2,
        "FrangiBetaOne": 0.5,
        "FrangiBetaTwo": 15,
        "verbose": True,
        "BlackWhite": True,
    }
    options = defaultoptions

    sigmas = np.arange(
        options["FrangiScaleRange"][0],
        options["FrangiScaleRange"][1],
        options["FrangiScaleRatio"],
    )
    sigmas.sort()

    beta = 2 * pow(options["FrangiBetaOne"], 2)
    c = 2 * pow(options["FrangiBetaTwo"], 2)

    shape = (image.shape[0], image.shape[1], len(sigmas))
    ALLfiltered = np.zeros(shape)
    ALLangles = np.zeros(shape)

    # Frangi filter for all sigmas
    Rb = 0
    S2 = 0
    for i in range(len(sigmas)):
        # Show progress
        if options["verbose"]:
            print("Current Frangi Filter Sigma: ", sigmas[i])

        # Make 2D hessian
        [Dxx, Dxy, Dyy] = Hessian2D(image, sigmas[i])

        # Correct for scale
        Dxx = pow(sigmas[i], 2) * Dxx
        Dxy = pow(sigmas[i], 2) * Dxy
        Dyy = pow(sigmas[i], 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        [Lambda2, Lambda1, Ix, Iy] = eig2image(Dxx, Dxy, Dyy)

        # Compute the direction of the minor eigenvector
        angles = np.arctan2(Ix, Iy)

        # Compute some similarity measures
        Lambda1[Lambda1 == 0] = np.spacing(1)

        Rb = (Lambda2 / Lambda1) ** 2
        S2 = Lambda1**2 + Lambda2**2

        # Compute the output image
        Ifiltered = np.exp(-Rb / beta) * (np.ones(image.shape) - np.exp(-S2 / c))

        # see pp. 45
        if options["BlackWhite"]:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        # store the results in 3D matrices
        ALLfiltered[:, :, i] = Ifiltered
        ALLangles[:, :, i] = angles

        # Return for every pixel the value of the scale(sigma) with the maximum
        # output pixel value
        if len(sigmas) > 1:
            outIm = ALLfiltered.max(2)
        else:
            outIm = (outIm.transpose()).reshape(image.shape)

    return outIm
