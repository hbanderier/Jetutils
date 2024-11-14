from pathlib import Path
from sys import stderr
from datetime import timedelta
from typing import Callable, Iterable, Mapping, Sequence, Tuple, Literal
from multiprocessing import Pool, current_process, get_context

import numpy as np
import polars as pl
from polars.exceptions import ColumnNotFoundError
import polars_ols as pls
import xarray as xr
from contourpy import contour_generator
from sklearn.mixture import GaussianMixture
from tqdm import tqdm, trange

from jetstream_hugo.definitions import (
    N_WORKERS,
    RADIUS,
    labels_to_mask,
    to_zero_one,
    compute,
)
from jetstream_hugo.data import (
    SEASONS,
    compute_extreme_climatology,
    DataHandler,
)


def get_index_columns(df, potentials: tuple = ("member", "time", "cluster", "jet ID")):
    index_columns = [ic for ic in potentials if ic in df.columns]
    return index_columns


def haversine(lon1: pl.Expr, lat1: pl.Expr, lon2: pl.Expr, lat2: pl.Expr) -> pl.Expr:
    lon1 = lon1.radians()
    lat1 = lat1.radians()
    lon2 = lon2.radians()
    lat2 = lat2.radians()

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def haversine_from_dl(lat: pl.Expr, dlon: pl.Expr, dlat: pl.Expr) -> pl.Expr:
    lat = lat.radians()
    dlon = dlon.radians()
    dlat = dlat.radians()

    a = (dlat / 2.0).sin().pow(2) * (dlon / 2.0).cos().pow(2) + lat.cos().pow(2) * (
        dlon / 2.0
    ).sin().pow(2)
    return 2 * a.sqrt().arcsin() * RADIUS


def jet_integral_haversine(
    lon: pl.Expr, lat: pl.Expr, s: pl.Expr | None = None, x_is_one: bool = False
) -> pl.Expr:
    ds = haversine(
        lon,
        lat,
        lon.shift(),
        lat.shift(),
    )
    if x_is_one:
        return ds.sum()
    return 0.5 * (ds * (s + s.shift())).sum()


def has_periodic_x(df) -> bool:
    lon = df["lon"].unique().sort().to_numpy()
    dx = lon[1] - lon[0]
    return (-180 in lon) and ((180 - dx) in lon)


def smooth_in_space(
    df: pl.DataFrame, winsize: int, to_smooth: str | Sequence = "all"
) -> pl.DataFrame:
    index_columns = get_index_columns(df)
    other_columns = [
        col for col in df.columns if col not in [*index_columns, "lat", "lon"]
    ]

    if to_smooth == "all":
        to_smooth = other_columns
        other_columns = []
    else:
        if isinstance(to_smooth, str):
            to_smooth = [to_smooth]
        other_columns = [col for col in other_columns if col not in to_smooth]
    keep = [pl.col(col) for col in other_columns]
    means = [
        pl.col(col).rolling_mean(winsize, min_periods=1, center=True)
        for col in to_smooth
    ]

    df = df.sort([*index_columns, "lat", "lon"])
    if has_periodic_x(df):
        # df = df.cast({dim: pl.Int32})
        halfwinsize = winsize // 2
        len_ = [df[col].unique().len() for col in [*index_columns, "lat"]]
        len_ = np.prod(len_)
        df = df.sort(["lon", *index_columns, "lat"])
        offset_along_dim = df[-1, "lon"] - df[0, "lon"] + 1
        df = pl.concat(
            [
                df.tail(halfwinsize * len_).with_columns(
                    pl.col("lon") - offset_along_dim
                ),
                df,
                df.head(halfwinsize * len_).with_columns(
                    pl.col("lon") + offset_along_dim
                ),
            ]
        )
        df = (
            df.group_by([*index_columns, "lat"], maintain_order=True)
            .agg(pl.col("lon"), *keep, *means)
            .explode(["lon", *other_columns, *to_smooth])
        )
        df = df.sort(["lon", *index_columns, "lat"])
        df = df.slice(halfwinsize * len_, df.shape[0] - 2 * len_ * halfwinsize)
    else:
        df = (
            df.group_by([*index_columns, "lat"], maintain_order=True)
            .agg(pl.col("lon"), *keep, *means)
            .explode(["lon", *other_columns, *to_smooth])
        )

    df = df.sort([*index_columns, "lon", "lat"])
    df = (
        df.group_by([*index_columns, "lon"], maintain_order=True)
        .agg(pl.col("lat"), *keep, *means)
        .explode(["lat", *other_columns, *to_smooth])
    )
    df = df.sort([*index_columns, "lat", "lon"])
    return df


def coarsen(df: pl.DataFrame, coarsen_map: Mapping[str, float]) -> pl.DataFrame:
    index_columns = get_index_columns(df)
    other_columns = [
        col for col in df.columns if col not in [*index_columns, *list(coarsen_map)]
    ]
    by = [
        *index_columns,
        *[pl.col(col).floordiv(val) for col, val in coarsen_map.items()],
    ]
    agg = [pl.col(col).mean() for col in other_columns]
    # agg = [*[pl.col(col).alias(f"{col}_").mean() for col, val in coarsen_map.items()], *agg]
    df = df.group_by(by, maintain_order=True).agg(*agg)
    # df = df.drop(list(coarsen_map)).rename({f"{col}_": col for col in coarsen_map})
    return df


def round_polars(col: str, factor: int = 2) -> pl.Expr:
    return (pl.col(col) * factor).round() / factor


def central_diff(by: str) -> pl.Expr:
    diff_2 = pl.col(by).diff(2, null_behavior="ignore").slice(2)
    diff_1 = pl.col(by).diff(1, null_behavior="ignore").gather([1, -1])
    return diff_1.get(0).append(diff_2).append(diff_1.get(-1))


def diff_maybe_periodic(by: str, periodic: bool = False) -> pl.Expr:
    if not periodic:
        return central_diff(by)
    max_by = pl.col(by).max() - pl.col(by).min()
    diff_by = central_diff(by).abs()
    return pl.when(diff_by > max_by / 2).then(max_by - diff_by).otherwise(diff_by)


def directional_diff(
    df: pl.DataFrame, col: str, by: str, periodic: bool = False
) -> pl.DataFrame:
    others = {
        "lon": "lat",
        "lat": "lon",
        "x": "y",
        "y": "x",
    }
    other = others[by]
    index_columns = get_index_columns(df)
    name = f"d{col}d{by}"
    diff_by = diff_maybe_periodic(by, periodic)
    agg = {name: central_diff(col) / diff_by, by: pl.col(by)}
    return (
        df.group_by([*index_columns, other], maintain_order=True)
        .agg(**agg)
        .explode(name, by)
    )


def compute_sigma(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(pl.col("lat").is_in([-90.0, 90.0]).not_())
    index_columns = get_index_columns(df, ("member", "time", "cluster"))
    periodic_x = has_periodic_x(df)
    x = pl.col("lon").radians() * RADIUS
    y = (
        (1 + pl.col("lat").radians().sin()) / pl.col("lat").radians().cos()
    ).log() * RADIUS
    df = df.with_columns(x=x, y=y)
    df = df.join(
        directional_diff(df, "s", "x", periodic_x), on=[*index_columns, "x", "y"]
    )
    df = df.join(directional_diff(df, "s", "y"), on=[*index_columns, "x", "y"])
    sigma = (pl.col("v") * pl.col("dsdx") - pl.col("u") * pl.col("dsdy")) / pl.col("s")
    df = df.with_columns(sigma=sigma)
    df = df.sort([*index_columns, "lat", "lon"])
    df = df.drop("x", "y", "dsdx", "dsdy")
    return df


def nearest_mapping(df1: pl.DataFrame, df2: pl.DataFrame, col: str):
    df1 = df1.select(col).unique().sort(col)
    df2 = df2.select(col).unique().sort(col).rename({col: f"{col}_"})
    return df1.join_asof(
        df2, left_on=pl.col(col), right_on=pl.col(f"{col}_"), strategy="nearest"
    )


def find_contours_maybe_periodic(x, y, z) -> Tuple[list, list]:
    dx = x[1] - x[0]
    if (-180 not in x) or ((180 - dx) not in x):
        print("not periodic")
        contours, types = contour_generator(
            x=x, y=y, z=z, line_type="SeparateCode", quad_as_tri=False
        ).lines(0.0)
        cyclic = [79 in types_ for types_ in types]
        return contours, cyclic

    xind = np.append(np.arange(len(x)), np.arange(len(x)))
    fake_x = np.append(x, x + 360)
    z_prime = z[:, xind]
    all_contours, all_types = contour_generator(
        x=fake_x, y=y, z=z_prime, line_type="SeparateCode", quad_as_tri=False
    ).lines(0.0)
    all_cyclic = [int(79 in types_) for types_ in all_types]
    all_lens = [len(contour) for contour in all_contours]
    
    sorted_order = np.argsort(all_lens)[::-1]
    all_contours = [all_contours[i] for i in sorted_order]
    all_cyclic = [all_cyclic[i] for i in sorted_order]

    periodics = []
    contours = []
    per_cyclic = []
    cyclic = []

    for i, contour in enumerate(all_contours):
        if len(contour) < 20:
            continue
        if all(contour[:, 0] < 180):
            contours.append(np.round(contour))
            cyclic.append(all_cyclic[i])
        if all(contour[:, 0] >= 180):
            continue
        contour = np.round(contour)
        x_real = (contour[:, 0] + 180) % 360 - 180
        max_jump = np.abs(np.diff(x_real)).max()
        if max_jump > 180:
            contour = np.stack([x_real, contour[:, 1]], axis=-1)
            unique, index, counts = np.unique(
                contour, return_counts=True, return_index=True, axis=0
            )
            this_cyclic = int(np.mean(counts) > 1.95) + all_cyclic[i]
            do_append = True
            for j, other_per in enumerate(periodics):
                overlap = (contour == other_per[:, None, :]).all(axis=-1).any(axis=-1)
                if np.mean(overlap) > 0.3:
                    do_append = False
            if do_append:
                per_cyclic.append(this_cyclic)
                periodics.append(contour[np.sort(index)])
    to_del = []
    for i, periodic_contour in enumerate(periodics):
        for j, contour in enumerate(contours):
            overlap = (periodic_contour == contour[:, None, :]).all(axis=-1).any(axis=-1)
            if np.mean(overlap) > 0.9:
                to_del.append(j)
    contours = [contour for i, contour in enumerate(contours) if i not in to_del]
    cyclic = [cyclic_ for i, cyclic_ in enumerate(cyclic) if i not in to_del]
    contours.extend(periodics)
    cyclic.extend(per_cyclic)
    return contours, cyclic


def innter_compute_contours(args):
    indexer, df = args
    index_columns = get_index_columns(df)
    lon = df["lon"].unique().sort().to_numpy()
    lat = df["lat"].unique().sort().to_numpy()
    indexer = dict(zip(index_columns, [pl.lit(ind) for ind in indexer]))
    sigma = df["sigma"].to_numpy().reshape(len(lat), len(lon))
    contours, cyclic = find_contours_maybe_periodic(lon, lat, sigma)
    valid_index = [i for i, contour in enumerate(contours) if len(contour) > 20]
    contours = [
        pl.DataFrame(contours[i], schema={"lon": pl.Float32, "lat": pl.Float32})
        .with_columns(**indexer)
        .with_columns(cyclic=pl.lit(cyclic[i]))
        .with_columns(contour=pl.lit(i))
        for i in valid_index
    ]
    return pl.concat(contours)


def compute_contours(df: pl.DataFrame):
    index_columns = get_index_columns(df)
    iterator = df.group_by(index_columns, maintain_order=True)
    len_ = iterator.first().shape[0]
    all_contours = map_maybe_parallel(
        iterator, innter_compute_contours, len_, processes=1
    )  # polars-sequential is much faster than 20 cores multiproc
    all_contours = pl.concat(all_contours)
    return all_contours
    # .with_columns(
    #     lon=round_polars("lon", 2), lat=round_polars("lat", 2)
    # )[*index_columns, "contour", "lon", "lat", "cyclic"]
    # lat_mapping = nearest_mapping(all_contours, df, "lat")
    # lon_mapping = nearest_mapping(all_contours, df, "lon")
    # all_contours = all_contours.join(lat_mapping, on="lat")
    # all_contours = all_contours.join(lon_mapping, on="lon")
    # ho = ["lon", "lat"]
    # all_contours = all_contours.drop(ho).rename({f"{col}_": col for col in ho})
    # all_contours = all_contours.unique(
    #     [*index_columns, "lon", "lat"], maintain_order=True
    # )
    # return all_contours


def compute_alignment(
    all_contours: pl.DataFrame, periodic: bool = False
) -> pl.DataFrame:
    index_columns = get_index_columns(all_contours, ("member", "time", "cluster"))
    dlon = diff_maybe_periodic("lon", periodic)
    dlat = central_diff("lat")
    ds = (dlon.pow(2) + dlat.pow(2)).sqrt()
    align_x = pl.col("u") / pl.col("s") * dlon / ds
    align_y = pl.col("v") / pl.col("s") * dlat / ds
    alignment = align_x + align_y
    alignment = (
        all_contours.group_by([*index_columns, "contour"], maintain_order=True)
        .agg(alignment=alignment)
        .explode("alignment")
    )
    return all_contours.with_columns(alignment=alignment["alignment"])


def do_rle(df: pl.DataFrame, cond_name: str = "condition") -> pl.DataFrame:
    by = get_index_columns(df, ("member", "time", "cluster", "contour"))
    conditional = (
        df.group_by(by, maintain_order=True)
        .agg(
            pl.col(cond_name).rle().alias("rle"),
        )
        .explode("rle")
        .unnest("rle")
    )
    conditional = (
        conditional.group_by(by, maintain_order=True)
        .agg(
            len=pl.col("len"),
            start=pl.lit(0).append(
                pl.col("len").cum_sum().slice(0, pl.col("len").len() - 1)
            ),
            value=pl.col("value"),
        )
        .explode(["len", "start", "value"])
    )
    return conditional


def do_rle_fill_hole(
    df: pl.DataFrame, condition_expr: pl.Expr, hole_size: int = 4
) -> pl.DataFrame:
    by = get_index_columns(df, ("member", "time", "cluster", "contour"))
    condition = (
        df.group_by(by, maintain_order=True)
        .agg(
            condition_expr.alias("condition"),
            index=pl.int_range(0, condition_expr.len()),
        )
        .explode("condition", "index")
    )

    conditional = do_rle(condition)

    conditional = conditional.filter(
        pl.col("len") <= hole_size, pl.col("value").not_(), pl.col("start") > 0
    )

    conditional = (
        conditional.with_columns(
            index=pl.int_ranges(pl.col("start"), pl.col("start") + pl.col("len"))
        )[*by, "index"]
        .explode("index")
        .with_columns(condition=pl.lit(True))
    )

    condition = condition.join(conditional, on=[*by, "index"], how="left")
    condition = condition.with_columns(
        condition=pl.when(pl.col("condition_right").is_not_null())
        .then(pl.col("condition_right"))
        .otherwise(pl.col("condition"))
    ).drop("condition_right")

    conditional = do_rle(condition)

    conditional = conditional.filter(pl.col("value"))

    conditional = conditional.with_columns(
        index=pl.int_ranges(pl.col("start"), pl.col("start") + pl.col("len"))
    )[*by, "index"].explode("index")
    return conditional


def separate_jets(
    jets: pl.DataFrame, ds_thresh: float = 5, periodic: bool = False
) -> pl.DataFrame:
    index_columns = get_index_columns(jets)
    dx = diff_maybe_periodic("lon", periodic)
    dy = diff_maybe_periodic("lat", False)
    diffs = (dx.pow(2) + dy.pow(2)).sqrt()
    cont_lens = jets.group_by([*index_columns, "contour"], maintain_order=True).len()
    cont_lens = cont_lens.filter(pl.col("len") > 3)
    jets = cont_lens.join(jets, on=[*index_columns, "contour"]).drop("len")
    jets = jets.with_columns(
        jets.group_by([*index_columns, "contour"], maintain_order=True)
        .agg(ds=diffs.fill_null(0.0))
        .explode("ds")
    )
    other_columns = [
        col for col in jets.columns if col not in [*index_columns, "contour"]
    ]

    def shifter(col):
        argmax = pl.col("ds").arg_max()
        len_ = pl.col("ds").len()
        cond1 = pl.col("cyclic").get(0) >= 2
        cond2 = (pl.col("ds") > ds_thresh).any()
        argmax = argmax * (cond1 & cond2).cast(pl.Int32)
        shift = pl.col(col).gather(pl.int_range(argmax, argmax + len_) % len_)
        return shift

    jets = (
        jets.group_by([*index_columns, "contour"], maintain_order=True)
        .agg(
            *[shifter(col) for col in other_columns],
            index_2=pl.int_range(0, pl.col("ds").len()),
        )
        .explode(*other_columns, "index_2")
    )
    separate_jets_expr = pl.col("ds") > ds_thresh
    separate_jets_expr = separate_jets_expr.fill_null(True)
    separate_jets_expr = (
        separate_jets_expr
        | (pl.col("index_2") == pl.col("index_2").min())
        | (pl.col("index_2") >= pl.col("index_2").max())
    )
    jets = jets.with_columns(
        condition=jets.group_by([*index_columns, "contour"], maintain_order=True)
        .agg(condition=separate_jets_expr.cum_sum())["condition"]
        .explode()
    )
    jets = jets.with_columns(
        **{"jet ID": (pl.col("condition") / 100 + pl.col("contour").rle_id()).rle_id()}
    ).drop("index_2")
    return jets


def find_all_jets(df: pl.DataFrame, thresholds: xr.DataArray | None = None):
    # process input
    if thresholds is not None:
        thresholds = (
            pl.from_pandas(thresholds.to_dataframe().reset_index())
            .drop("quantile")
            .cast({"s": pl.Float32})
            .rename({"s": "s_thresh"})
        )
        df = df.join(thresholds, on="time")
        df = df.with_columns(int_thresh=pl.col("s_thresh") * 1.2e8 / 25)
        condition_expr = (pl.col("s") > pl.col("s_thresh")) & (
            pl.col("alignment") > 0.4
        )
        condition_expr2 = pl.col("int") > pl.col("int_thresh")
        drop = [
            "contour",
            "index",
            "cyclic",
            "s_thresh",
            "int_thresh",
            "condition",
            "int",
            "index_2",
        ]
    else:
        condition_expr = (pl.col("s") > 30) & (pl.col("alignment") > 0.5)
        condition_expr2 = pl.col("int") > 3e8
        drop = ["contour", "index", "cyclic", "condition", "int"]

    # smooth, compute sigma
    x_periodic = has_periodic_x(df)
    index_columns = get_index_columns(df)
    df = coarsen(df, {"lon": 1, "lat": 1})
    df = smooth_in_space(df, 11)
    df = compute_sigma(df)
    df = smooth_in_space(df, 9)
    df = df.with_columns(
        lon=round_polars("lon").cast(pl.Float32),
        lat=round_polars("lat").cast(pl.Float32),
    )

    # contours
    all_contours = compute_contours(df)
    all_contours = all_contours.with_columns(
        index=all_contours.group_by([*index_columns, "contour"], maintain_order=True)
        .agg(index=pl.int_range(0, pl.col("lon").len()))
        .explode("index")["index"]
    )
    for index_column in index_columns:
        try:
            all_contours = all_contours.cast({index_column: df[index_column].dtype})
        except ColumnNotFoundError:
            pass
    all_contours = all_contours.join(df, on=[*index_columns, "lon", "lat"], how="left")
    all_contours = compute_alignment(all_contours, x_periodic)

    # jets from contours
    conditional = do_rle_fill_hole(all_contours, condition_expr, 8)
    jets = conditional.join(all_contours, on=[*index_columns, "contour", "index"])
    jets = separate_jets(jets, 4, x_periodic)
    jets = jets.with_columns(
        len=jets.group_by([*index_columns, "jet ID"], maintain_order=True)
        .agg(pl.col("jet ID").len().repeat_by(pl.col("jet ID").len()).alias("len"))[
            "len"
        ]
        .list.explode()
        .list.explode()
    )
    jets = jets.filter(pl.col("len") >= 5).drop("len")
    jets = jets.with_columns(
        jets.group_by([*index_columns, "jet ID"], maintain_order=True)
        .agg(
            jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s"))
            .alias("int")
            .repeat_by(pl.col("jet ID").len())
        )
        .explode("int")
        .explode("int")["int"]
    )
    jets = jets.filter(condition_expr2).drop(drop)
    jets = jets.with_columns(
        **{
            "jet ID": jets.group_by(index_columns, maintain_order=True)
            .agg(pl.col("jet ID").rle_id().alias("id2"))
            .explode("id2")["id2"]
        }
    )
    return jets


def compute_jet_props(df: pl.DataFrame, polar_cutoff: float = 0.2) -> pl.DataFrame:
    aggregations = [
        *[
            ((pl.col(col) * pl.col("s")).sum() / pl.col("s").sum()).alias(f"mean_{col}")
            for col in ["lon", "lat", "lev"]
        ],
        pl.col("s").mean().alias("mean_s"),
        *[
            pl.col(col).get(pl.col("s").arg_max()).alias(f"{col}_star")
            for col in ["lon", "lat", "s"]
        ],
        *[
            (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_ext")
            for col in ["lon", "lat"]
        ],
        pl.col("lat")
        .least_squares.ols(pl.col("lon"), mode="coefficients", add_intercept=True)
        .struct.field("lon")
        .alias("tilt"),
        (
            pl.col("lat")
            .least_squares.ols(pl.col("lon"), mode="residuals", add_intercept=True)
            .pow(2)
            .sum()
            / (pl.col("lat") - pl.col("lat").mean()).pow(2).sum()
        ).alias("waviness1"),
        (pl.col("lat") - pl.col("lat").mean()).pow(2).sum().alias("waviness2"),
        (
            pl.col("lat").gather(pl.col("lon").arg_sort()).diff().abs().sum()
            / (pl.col("lon").max() - pl.col("lon").min())
        ).alias("wavinessR16"),
        (
            jet_integral_haversine(pl.col("lon"), pl.col("lat"), x_is_one=True)
            / pl.lit(RADIUS)
            * pl.col("lat").mean().radians().cos()
        ).alias("wavinessDC16"),
        (
            ((pl.col("v") - pl.col("v").mean()) * pl.col("v").abs() / pl.col("s")).sum()
            / pl.col("s").sum()
        ).alias("wavinessFV15"),
        jet_integral_haversine(pl.col("lon"), pl.col("lat"), pl.col("s")).alias("int"),
        jet_integral_haversine(
            pl.col("lon").filter(pl.col("lon") > -10),
            pl.col("lat").filter(pl.col("lon") > -10),
            pl.col("s").filter(pl.col("lon") > -10),
        ).alias("int_over_europe"),
        pl.col("is_polar").mean(),
    ]

    df_lazy = df.lazy()
    index_columns = get_index_columns(df)
    if "member" not in get_index_columns(df):
        gb = df_lazy.group_by(index_columns, maintain_order=True)
        props_as_df = gb.agg(*aggregations)
        return props_as_df.collect()

    # streaming mode doesn't work well
    collected = []
    for member in tqdm(df["member"].unique(maintain_order=True).to_numpy()):
        gb = df_lazy.filter(pl.col("member") == member).group_by(
            get_index_columns(df), maintain_order=True
        )
        props_as_df = gb.agg(*aggregations)
        collected.append(props_as_df.collect())
    props_as_df = pl.concat(collected).sort("member")
    props_as_df = props_as_df.with_columns(
        pl.when(pl.col("is_polar") > 1 - polar_cutoff)
        .then(pl.lit("EDJ"))
        .when(pl.col("is_polar") < polar_cutoff)
        .then(pl.lit("STJ"))
        .otherwise(pl.lit("NA"))
        .alias("jet")
    )
    return props_as_df


def distances_to_coord(
    da_df: pl.DataFrame, jet: pl.DataFrame, coord: str, prefix: str = ""
):
    unique = da_df[coord].unique()
    ind = unique.search_sorted(jet[f"{prefix}{coord}"]).cast(pl.Int32)
    dx = unique.diff()[1]
    dist_to_next = unique.gather(ind) - jet[f"{prefix}{coord}"]
    dist_to_previous = dx - dist_to_next
    n = unique.len()
    return n, ind, dx, dist_to_previous, dist_to_next


def interp_from_other(jets: pl.DataFrame, da_df: pl.DataFrame):
    n_lon, ind_lon, dlon, dist_to_previous_lon, dist_to_next_lon = distances_to_coord(
        da_df, jets, "lon", "normal"
    )
    n_lat, ind_lat, dlat, dist_to_previous_lat, dist_to_next_lat = distances_to_coord(
        da_df, jets, "lat", "normal"
    )
    s_above_right = da_df[n_lon * ind_lat + ind_lon, "s"]
    s_below_right = da_df[n_lon * (ind_lat - 1) + ind_lon, "s"]
    s_above_left = da_df[n_lon * ind_lat + ind_lon - 1, "s"]
    s_below_left = da_df[n_lon * (ind_lat - 1) + ind_lon - 1, "s"]

    s_below = (
        s_below_right * dist_to_next_lon / dlon
        + s_below_left * dist_to_previous_lon / dlon
    )
    s_above = (
        s_above_right * dist_to_next_lon / dlon
        + s_above_left * dist_to_previous_lon / dlon
    )

    s_interp = s_below * dist_to_next_lat / dlat + s_above * dist_to_previous_lat / dlat
    return s_interp


def inner_compute_widths(args):
    jets, da = args

    dn = 1
    ns_df = pl.Series("n", np.delete(np.arange(-12, 12 + dn, dn), 12)).to_frame()

    # Expr theta
    theta = pl.arctan2(pl.col("v"), pl.col("u")).interpolate("linear") + np.pi / 2

    # Expr normals
    normallon = pl.col("lon") + pl.col("theta").cos() * pl.col("n")
    normallat = pl.col("lat") + pl.col("theta").sin() * pl.col("n")

    # Expr half_width
    below = pl.col("s_interp") <= pl.max_horizontal(pl.col("s") / 4 * 3, pl.lit(25))
    stop_up = below.arg_max()
    nlo_up = pl.col("normallon").gather(stop_up)
    nla_up = pl.col("normallat").gather(stop_up)
    half_width_up = haversine(
        nlo_up, nla_up, pl.col("lon").get(0), pl.col("lat").get(0)
    ).cast(pl.Float32)

    stop_down = below.len() - below.reverse().arg_max() - 1
    nlo_down = pl.col("normallon").gather(stop_down)
    nla_down = pl.col("normallat").gather(stop_down)
    half_width_down = haversine(
        nlo_down, nla_down, pl.col("lon").get(0), pl.col("lat").get(0)
    ).cast(pl.Float32)

    half_width = (
        pl.when(pl.col("side") == -1)
        .then(pl.col("half_width_down"))
        .otherwise(pl.col("half_width_up"))
        .list.first()
    )

    index_columns = get_index_columns(jets, ("member", "time", "cluster"))
    agg_out = {ic: pl.col(ic).first() for ic in [*index_columns, "lon", "lat", "s"]}

    first_agg_out = agg_out | {
        "half_width_up": half_width_up,
        "half_width_down": half_width_down,
    }
    second_agg_out = agg_out | {"half_width": pl.col("half_width").sum()}
    third_agg_out = agg_out | {
        "width": (pl.col("half_width") * pl.col("s")).sum() / pl.col("s").sum()
    }

    da_df = pl.from_pandas(da.to_dataframe().reset_index()).drop(index_columns)
    jets = jets[[*index_columns, "jet ID", "lon", "lat", "u", "v", "s"]]

    jets = jets.with_columns(
        jets.group_by("jet ID", maintain_order=True)
        .agg(theta=theta, index=pl.int_range(pl.len()))
        .explode(["index", "theta"])
    )
    jets = jets.join(ns_df, how="cross")

    jets = jets.with_columns(normallon=normallon, normallat=normallat)
    jets = jets[
        [
            *index_columns,
            "jet ID",
            "index",
            "lon",
            "lat",
            "s",
            "n",
            "normallon",
            "normallat",
        ]
    ]
    jets = jets.filter(
        pl.col("normallon") >= da_df["lon"].min(),
        pl.col("normallon") <= da_df["lon"].max(),
        pl.col("normallat") >= da_df["lat"].min(),
        pl.col("normallat") <= da_df["lat"].max(),
    )

    jets = jets.with_columns(s_interp=interp_from_other(jets, da_df))
    jets = jets.with_columns(side=pl.col("n").sign().cast(pl.Int8))

    jets = jets.group_by(["jet ID", "index", "side"], maintain_order=True).agg(
        **first_agg_out
    )

    jets = jets.with_columns(half_width=half_width).drop(
        ["half_width_up", "half_width_down", "side"]
    )
    jets = jets.group_by(["jet ID", "index"]).agg(**second_agg_out)
    jets = jets.group_by("jet ID", maintain_order=True).agg(**third_agg_out)
    return jets.drop("s").cast({"width": pl.Float32})


def map_maybe_parallel(
    iterator: Iterable,
    func: Callable,
    len_: int,
    processes: int = N_WORKERS,
    chunksize: int = 100,
    progress: bool = True,
    pool_kwargs: dict | None = None,
    ctx=None,
) -> list:
    processes = min(processes, len_)
    if processes == 1 and progress:
        return list(tqdm(map(func, iterator), total=len_))
    if processes == 1:
        return list(map(func, iterator))
    if pool_kwargs is None:
        pool_kwargs = {}
    pool_func = Pool if ctx is None else ctx.Pool
    if not progress:
        with pool_func(processes=processes, **pool_kwargs) as pool:
            to_ret = pool.imap(func, iterator, chunksize=chunksize)
            return list(to_ret)
    with pool_func(processes=processes, **pool_kwargs) as pool:
        to_ret = tqdm(
            pool.imap(func, iterator, chunksize=chunksize),
            total=len_,
        )
        return list(to_ret)


def create_mappable_iterator(
    df: pl.DataFrame,
    das: Sequence | None = None,
    others: Sequence | None = None,
    potentials: Tuple = ("member", "time", "cluster"),
) -> Tuple:
    if das is None:
        das = []
    if others is None:
        others = []
    iter_dims = []
    for potential in potentials:
        if potential in df.columns:
            iter_dims.append(potential)
    gb = df.group_by(iter_dims, maintain_order=True)
    len_ = len(gb.first())
    iterator = (
        (
            jets,
            *[
                compute(
                    da.sel(
                        {
                            dim: values
                            for dim, values in zip(iter_dims, index)
                            if dim in da.dims
                        }
                    )
                )
                for da in das
            ],
            *others,
        )
        for index, jets in gb
    )
    return len_, iterator


def compute_widths_parallel(
    all_jets_one_df: pl.DataFrame,
    da: xr.DataArray,
    processes: int = N_WORKERS,
    chunksize: int = 100,
):
    len_, iterator = create_mappable_iterator(all_jets_one_df, [da])
    print("Computing widths")
    all_widths = map_maybe_parallel(
        iterator,
        inner_compute_widths,
        len_=len_,
        processes=processes,
        chunksize=chunksize,
        ctx=get_context("spawn"),
    )
    return pl.concat(all_widths)


def round_half(x):
    return np.round(x * 2) / 2


def extract_season_from_df(
    df: pl.DataFrame,
    season: list | str | tuple | int | None = None,
) -> pl.DataFrame:
    if season is None:
        return df
    if isinstance(season, str):
        season = SEASONS[season]
    if isinstance(season, int):
        season = [season]
    return df.filter(pl.col("time").dt.month().is_in(season))


def extract_features(
    props_as_df: pl.DataFrame,
    feature_names: Sequence = None,
    season: list | str | tuple | int | None = None,
) -> Tuple[np.ndarray, float]:
    props_as_df = extract_season_from_df(props_as_df, season)
    if feature_names is None:
        feature_names = ["mean_lon", "mean_lat", "s_star"]

    X = props_as_df[feature_names].to_numpy()
    return X


def one_gmix(X, n_components=3):
    X, _, _ = to_zero_one(X)
    model = GaussianMixture(
        n_components=n_components
    )  # to help with class imbalance, 1 for sub 2 for polar
    labels = model.fit_predict(X)
    masks = labels_to_mask(labels)
    mls = []
    for mask in masks.T:
        mls.append(X[mask, 0].mean() - X[mask, 1].mean())
    return labels != np.argmin(mls)


def is_polar_gmix(
    props_as_df: pl.DataFrame,
    feature_names: list,
    mode: Literal["year"] | Literal["season"] | Literal["month"] = "year",
) -> xr.Dataset:
    # TODO: assumes at least one year of data, check for season / month actually existing in the data, figure out output
    if mode == "year":
        X = extract_features(props_as_df, feature_names, None)
        labels = one_gmix(X)
        return props_as_df.with_columns(is_polar=labels)
    index_columns = get_index_columns(props_as_df)
    to_concat = []
    if mode == "season":
        for season in tqdm(["DJF", "MAM", "JJA", "SON"]):
            X = extract_features(props_as_df, feature_names, season)
            n_components = 2 if season == "MAM" else 3
            labels = one_gmix(X, n_components=n_components)
            to_concat.append(
                extract_season_from_df(props_as_df, season).with_columns(
                    is_polar=labels
                )
            )
    elif mode == "month":
        for month in trange(1, 13):
            X = extract_features(props_as_df, feature_names, month)
            n_components = 2 if month in [3, 4, 5, 6] else 3
            labels = one_gmix(X, n_components=n_components)
            to_concat.append(
                extract_season_from_df(props_as_df, month).with_columns(is_polar=labels)
            )
    return pl.concat(to_concat).sort(index_columns)


def categorize_df_jets(props_as_df: pl.DataFrame):
    index_columns = get_index_columns(
        props_as_df, ("member", "time", "cluster", "jet ID")
    )
    other_columns = [
        col for col in props_as_df.columns if col not in [*index_columns, "is_polar", "jet"]
    ]
    agg = {
        col: (pl.col(col) * pl.col("int")).sum() / pl.col("int").sum()
        for col in other_columns
    }
    agg["int"] = pl.col("int").mean()
    agg["s_star"] = pl.col("s_star").max()
    agg["lon_ext"] = pl.col("lon_ext").max()
    agg["lat_ext"] = pl.col("lat_ext").max()
    agg["exists"] = pl.col("int").len().cast(pl.Boolean())
    
    gb_columns = get_index_columns(
        props_as_df, ("member", "time", "cluster", "jet")
    )
    props_as_df_cat = (
        props_as_df.group_by(gb_columns, maintain_order=True)
        .agg(**agg)
        .sort(gb_columns)
    )

    if "member" in index_columns:
        dummy_indexer = (
            props_as_df_cat["member"]
            .unique(maintain_order=True)
            .to_frame()
            .join(
                props_as_df_cat["time"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
            .join(
                props_as_df_cat["jet"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
        )
    elif "cluster" in index_columns:
        dummy_indexer = (
            props_as_df_cat["cluster"]
            .unique(maintain_order=True)
            .to_frame()
            .join(
                props_as_df_cat["jet"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
        )
    else:
        dummy_indexer = (
            props_as_df_cat["time"]
            .unique(maintain_order=True)
            .to_frame()
            .join(
                props_as_df_cat["jet"].unique(maintain_order=True).to_frame(),
                how="cross",
            )
        )
    new_index_columns = get_index_columns(
        props_as_df_cat, ("member", "time", "cluster", "jet")
    )

    sort_descending = [False] * len(new_index_columns)
    sort_descending[-1] = True
    props_as_df_cat = dummy_indexer.join(
        props_as_df_cat, on=[pl.col(col) for col in new_index_columns], how="left"
    ).sort(new_index_columns, descending=sort_descending)
    props_as_df_cat = props_as_df_cat.with_columns(pl.col("exists").cast(pl.UInt8).fill_null(0))
    return props_as_df_cat


def overlap_vert_dist_polars() -> Tuple[pl.Expr]:
    x1 = pl.col("lon").flatten()
    y1 = pl.col("lat").flatten()
    x2 = pl.col("lon_right").flatten()
    y2 = pl.col("lat_right").flatten()

    row = pl.first().cum_count()

    a1 = x1.arg_unique()
    a2 = x2.arg_unique()

    x1 = x1.gather(a1)
    y1 = y1.gather(a1)
    x2 = x2.gather(a2)
    y2 = y2.gather(a2)

    inter12 = x1.is_in(x2)
    inter21 = x2.is_in(x1)
    vert_dist = (y1.filter(inter12) - y2.filter(inter21)).abs().mean()
    overlap = (
        inter12.mean()
    )  # new needs to be in old. giving weight to inter21 would let short new jets swallow old big jets, it's weird i think
    return vert_dist.over(row), overlap.over(row)


def _track_jets(df: pl.DataFrame):
    index_columns = get_index_columns(df)
    df = df.select([*index_columns, "lon", "lat", "is_polar"])
    unique_times = (
        df.select("time")
        .with_row_index()
        .unique("time", keep="first", maintain_order=True)
    )
    time_index_df = unique_times["index"]
    unique_times = unique_times["time"]
    df = df.with_columns(df.select(pl.col(["lon", "lat"]).map_batches(round_half)))
    guess_nflags = max(50, len(unique_times) // 2)
    guess_len = 1000
    all_jets_over_time = np.zeros(
        (guess_nflags, guess_len), dtype=[("time", "datetime64[ms]"), ("jet ID", "i2")]
    )
    all_jets_over_time[:] = (np.datetime64("NaT"), -1)
    last_valid_index_rel = np.full(guess_nflags, fill_value=-1, dtype="int32")
    last_valid_index_abs = np.full(guess_nflags, fill_value=-1, dtype="int32")

    flags = df.group_by(["time", "jet ID"], maintain_order=True).first()
    flags = flags.select([*index_columns]).clone()
    flags = flags.insert_column(
        -1, pl.Series("flag", np.zeros(len(flags), dtype=np.uint32))
    )
    time_index_flags = (
        flags.select("time")
        .with_row_index()
        .unique("time", keep="first", maintain_order=True)["index"]
    )
    for last_flag, _ in df[: time_index_df[1]].group_by("jet ID", maintain_order=True):
        last_flag = last_flag[0]
        all_jets_over_time[last_flag, 0] = (unique_times[0], last_flag)
        last_valid_index_rel[last_flag] = 0
        last_valid_index_abs[last_flag] = 0
        flags[last_flag, "flag"] = last_flag
    current = current_process()
    if current.name == "MainProcess":
        iterator = (pbar := trange(1, len(unique_times), position=0, leave=True))
    else:
        iterator = range(1, len(unique_times))
    for it in iterator:
        # create working dataframes: current timestep, previous 4 timesteps
        last_time = (
            time_index_df[it + 1] if (it < (len(time_index_df) - 1)) else df.shape[0]
        )
        current_df = df[time_index_df[it] : last_time]
        t = unique_times[it]
        min_it = max(0, it - 5)
        previous_df = df[time_index_df[min_it] : time_index_df[it]]
        potential_flags = np.where(
            (last_valid_index_abs >= (it - 4)) & (last_valid_index_abs >= 0)
        )[0]
        if len(potential_flags) == 0:
            n_new = current_df["jet ID"].unique().len()
            for j in range(n_new):
                last_flag += 1
                last_valid_index_rel[last_flag] = 0
                last_valid_index_abs[last_flag] = it
                all_jets_over_time[last_flag, 0] = (t, j)
                flags[int(time_index_flags[it] + j), "flag"] = last_flag
            if current.name == "MainProcess":
                pbar.set_description(f"last_flag: {last_flag}")
            continue
        potentials = all_jets_over_time[
            potential_flags, last_valid_index_rel[potential_flags]
        ]

        # Cumbersome construction for pairwise operations in polars
        # 1. Put potential previous jets in one df

        potentials_df = pl.concat(
            [
                previous_df.filter(
                    pl.col("time") == jtt_idx[0], pl.col("jet ID") == jtt_idx[1]
                )
                for jtt_idx in potentials
            ]
        )
        potentials_df_gb = potentials_df.group_by(
            ["jet ID", "time"], maintain_order=True
        )

        # 2. Turn into lists
        potentials_df = potentials_df_gb.agg(
            pl.col("lon"), pl.col("lat"), pl.col("is_polar").mean()
        )
        current_df = current_df.group_by(["jet ID", "time"], maintain_order=True).agg(
            pl.col("lon"), pl.col("lat"), pl.col("is_polar").mean()
        )

        # 3. create expressions (see function)
        vert_dist, overlap = overlap_vert_dist_polars()

        # perform pairwise using cross-join
        result = potentials_df.join(current_df, how="cross").select(
            old_jet="jet ID",
            new_jet="jet ID_right",
            vert_dist=vert_dist,
            overlap=overlap,
        )

        n_old = potentials_df.shape[0]
        n_new = current_df.shape[0]
        dist_mat = result["vert_dist"].to_numpy().reshape(n_old, n_new)
        overlaps = result["overlap"].to_numpy().reshape(n_old, n_new)

        try:
            dist_mat[np.isnan(dist_mat)] = np.nanmax(dist_mat) + 1
        except ValueError:
            pass
        index_start_flags = time_index_flags[it]
        connected_mask = (overlaps > 0.5) & (dist_mat < 10)
        potentials_isp = potentials_df["is_polar"].to_numpy()
        current_isp = current_df["is_polar"].to_numpy()
        connected_mask = (
            np.abs(potentials_isp[:, None] - current_isp[None, :]) < 0.15
        ) & connected_mask
        flagged = np.zeros(n_new, dtype=np.bool_)
        for i, jtt_idx in enumerate(potentials):
            js = np.argsort(
                dist_mat[i] / dist_mat[i].max() - overlaps[i] / overlaps[i].max()
            )
            for j in js:
                if not connected_mask[i, j]:
                    break
                if flagged[j]:
                    continue
                this_flag = potential_flags[i]
                last_valid_index_rel[this_flag] = last_valid_index_rel[this_flag] + 1
                last_valid_index_abs[this_flag] = it
                all_jets_over_time[this_flag, last_valid_index_rel[this_flag]] = (t, j)
                flagged[j] = True
                flags[int(index_start_flags + j), "flag"] = this_flag
                break
        pass
        for j in range(n_new):
            if not flagged[j]:
                last_flag += 1
                last_valid_index_rel[last_flag] = 0
                last_valid_index_abs[last_flag] = it
                all_jets_over_time[last_flag, 0] = (t, j)
                flags[int(index_start_flags + j), "flag"] = last_flag
                flagged[j] = True
        if current.name == "MainProcess":
            pbar.set_description(f"last_flag: {last_flag}")
    ajot_df = []
    for j, ajot in enumerate(all_jets_over_time[: last_flag + 1]):
        times = ajot["time"]
        ajot = ajot[: np.argmax(np.isnat(times))]
        ajot = pl.DataFrame(ajot)
        ajot = ajot.insert_column(
            0, pl.Series("flag", np.full(len(ajot), j, dtype=np.uint32))
        )
        if "member" in index_columns:
            ajot = ajot.insert_column(
                0,
                pl.Series("member", np.full(len(ajot), df["member"][0], dtype=object)),
            )
        ajot_df.append(ajot)
    ajot_df = pl.concat(ajot_df)
    return ajot_df, flags


def track_jets(all_jets_one_df: pl.DataFrame, processes: int = N_WORKERS):
    inner = ["time", "jet ID", "orig_points"]
    index_indices = min(
        all_jets_one_df.columns.index("lat"), all_jets_one_df.columns.index("lon")
    )
    levels = all_jets_one_df.columns[:index_indices]
    outer = [level for level in levels if level not in inner]
    if len(outer) == 0:
        return _track_jets(all_jets_one_df)
    len_, iterator = create_mappable_iterator(all_jets_one_df, potentials=tuple(outer))
    iterator = (a[0] for a in iterator)

    ctx = get_context("spawn")
    lock = ctx.RLock()  # I had to create a fresh lock
    tqdm.set_lock(lock)
    pool_kwargs = dict(initializer=tqdm.set_lock, initargs=(lock,))
    res = map_maybe_parallel(
        iterator,
        _track_jets,
        len_=len_,
        processes=processes,
        chunksize=1,
        pool_kwargs=pool_kwargs,
        ctx=ctx,
    )

    ajots, all_flags = tuple(zip(*res))
    ajots = pl.concat(ajots)
    all_flags = pl.concat(all_flags)
    return ajots, all_flags


def add_persistence_to_props(props_as_df: pl.DataFrame, flags: pl.DataFrame):
    if "member" in flags.columns:
        unique_to_count = (
            flags.group_by("member", maintain_order=True)
            .agg(
                flag=pl.col("flag").unique(),
                flag_count=pl.col("flag").unique_counts(),
            )
            .explode(["flag", "flag_count"])
        )
        on = ["member", "flag"]
    else:
        unique_to_count = pl.concat(
            [
                flags["flag"].unique().alias("flag").to_frame(),
                flags["flag"].unique_counts().alias("flag_count").to_frame(),
            ],
            how="horizontal",
        )
        on = ["flag"]
    factor = flags["time"].unique().diff()[1] / timedelta(days=1)
    persistence = flags.join(unique_to_count, on=on)
    persistence = persistence["flag_count"] * factor
    props_as_df = props_as_df.with_columns(persistence=persistence)
    return props_as_df


def compute_prop_anomalies(ds_props: xr.Dataset) -> xr.Dataset:
    prop_anomalies = ds_props.copy()

    for varname in ds_props.data_vars:
        gb = ds_props[varname].groupby("time.year")
        prop_anomalies[varname] = gb - gb.mean(dim="time")
        prop_anomalies[varname] = prop_anomalies[varname] / ds_props[varname].std(
            dim="time"
        )
    return prop_anomalies


def inner_jet_pos_as_da(args: Tuple):
    (time, jets), are_polar, da_template = args
    jets = jets.droplevel(0)
    jet_names = da_template.jet.values
    for j, (_, jet) in enumerate(jets.groupby(level=0)):
        jet = jet.droplevel(0)
        is_polar = are_polar.sel(jet=j)
        x, y, s = jet[["lon", "lat", "s"]].to_numpy().T
        x_ = xr.DataArray(round_half(x), dims="points")
        y_ = xr.DataArray(round_half(y), dims="points")
        try:
            is_polar = int(is_polar)
        except ValueError:
            continue
        da_template.loc[jet_names[int(is_polar)], y_, x_] += s
    return da_template


def jet_position_as_da(
    all_jets_one_df: pl.DataFrame,
    basepath: Path | None = None,
) -> xr.DataArray:
    if basepath is not None:
        ofile = basepath.joinpath("jet_pos.nc")
        if ofile.is_file():
            return xr.open_dataarray(ofile).load()

    index_columns = get_index_columns(all_jets_one_df, ("member", "time", "cluster"))
    all_jets_pandas = (
        all_jets_one_df.group_by(
            [*index_columns, "lon", "lat", "is_polar"], maintain_order=True
        )
        .len()
        .to_pandas()
    )
    da_jet_pos = xr.Dataset.from_dataframe(
        all_jets_pandas.set_index([*index_columns, "lat", "lon", "is_polar"])
    )["len"].fillna(0)
    if basepath is not None:
        da_jet_pos.to_netcdf(ofile)
    return da_jet_pos


def get_double_jet_index(df: pl.DataFrame, pos_as_da: xr.DataArray):
    overlap = (pos_as_da > 0).any("lat").all("is_polar")
    dji = pl.concat(
        [
            df.select("time").unique(maintain_order=True),
            pl.Series(
                "double_jet_index",
                overlap.sel(lon=slice(-20, None, None)).mean("lon").values,
            ).to_frame(),
        ],
        how="horizontal",
    )
    df = df.join(dji, on="time", how="left")
    return df


# def wave_activity_flux(u, v, z, u_c=None, v_c=None, z_c=None):
#     lon, lat = z.lon.values, z.lat.values
#     cos_lat = degcos(lat[None, :])
#     f = 2 * OMEGA * degsin(lat[:, None])
#     dlon = np.gradient(lon) * np.pi / 180.0
#     dlat = np.gradient(lat) * np.pi / 180.0
#     psi_p = (z - z_c) / f  # Pertubation stream-function

#     # 5 partial differential terms
#     dpsi_dlon = np.gradient(psi_p, dlon[1])[1]
#     dpsi_dlat = np.gradient(psi_p, dlat[1])[0]
#     d2psi_dlon2 = np.gradient(dpsi_dlon, dlon[1])[1]
#     d2psi_dlat2 = np.gradient(dpsi_dlat, dlat[1])[0]
#     d2psi_dlondlat = np.gradient(dpsi_dlat, dlon[1])[1]

#     termxu = dpsi_dlon * dpsi_dlon - psi_p * d2psi_dlon2
#     termxv = dpsi_dlon * dpsi_dlat - psi_p * d2psi_dlondlat
#     termyv = dpsi_dlat * dpsi_dlat - psi_p * d2psi_dlat2

#     # coefficient
#     p_lev = 300.0  # unit in hPa
#     p = p_lev / 1000.0
#     s_c = np.sqrt(u_c**2 + v_c**2)
#     coeff = (p * degcos(lat[None, :])) / (2 * s_c)
#     # x-component of TN-WAF
#     px = (coeff / (RADIUS * RADIUS * cos_lat)) * (
#         ((u_c) / cos_lat) * termxu + v_c * termxv
#     )
#     # y-component of TN-WAF
#     py = (coeff / (RADIUS * RADIUS)) * (((u_c) / cos_lat) * termxv + v_c * termyv)


class JetFindingExperiment(object):
    def __init__(
        self,
        data_handler: DataHandler,
    ) -> None:
        self.ds = data_handler.da
        self.path = data_handler.path
        self.data_handler = data_handler
        self.time = data_handler.get_sample_dims()["time"]

    def find_jets(self, **kwargs) -> pl.DataFrame:
        ofile_ajdf = self.path.joinpath("all_jets_one_df.parquet")

        if ofile_ajdf.is_file():
            all_jets_one_df = pl.read_parquet(ofile_ajdf)
            return all_jets_one_df
        try:
            qs_path = self.path.parent.joinpath("s_q.nc")
            qs = xr.open_dataarray(qs_path).sel(quantile=0.65)
            kwargs["thresholds"] = qs.rename("s")
        except FileNotFoundError:
            pass

        all_jets_one_df = []

        if "member" not in self.data_handler.get_sample_dims():
            for indices in np.array_split(np.arange(len(self.ds.time)), 10):
                ds_ = compute(self.ds.isel(time=indices), progress_flag=True)
                df_ds = pl.from_pandas(ds_.to_dataframe().reset_index())
                all_jets_one_df.append(find_all_jets(df_ds, **kwargs))

        else:
            members = self.data_handler.get_sample_dims()["member"]
            for i, memb_list in enumerate(np.array_split(members, 10)):
                print(i, " ", memb_list[0], "to", memb_list[-1])
                ds_ = compute(self.ds.sel(member=memb_list), progress_flag=True)
                df_ds = pl.from_pandas(ds_.to_dataframe().reset_index())
                all_jets_one_df.append(find_all_jets(df_ds, **kwargs))
                print("tictac", file=stderr, end="\r")

        all_jets_one_df = pl.concat(all_jets_one_df)
        # all_jets_one_df = is_polar_gmix(all_jets_one_df, ("lat", "lon", "lev"), mode="month")
        # all_jets_one_df.write_parquet(ofile_ajdf)
        return all_jets_one_df

    def compute_jet_props(
        self, processes: int = N_WORKERS, chunksize=100
    ) -> xr.Dataset:
        jet_props_incomplete_path = self.path.joinpath("props_as_df_raw.parquet")
        if jet_props_incomplete_path.is_file():
            return pl.read_parquet(jet_props_incomplete_path)
        all_jets_one_df = self.find_jets(processes=processes, chunksize=chunksize)
        props_as_df = compute_jet_props(all_jets_one_df)
        print("Loading s")
        da_ = compute(self.ds["s"], progress_flag=True)
        width = compute_widths_parallel(
            all_jets_one_df, da_, processes=processes, chunksize=chunksize
        )
        props_as_df = props_as_df.with_columns(width=width["width"])
        props_as_df.write_parquet(jet_props_incomplete_path)
        return props_as_df

    def track_jets(self) -> Tuple:
        all_jets_one_df = self.find_jets()
        ofile_ajot = self.path.joinpath("all_jets_over_time.parquet")
        ofile_flags = self.path.joinpath("flags.parquet")

        if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]):
            all_jets_over_time = pl.read_parquet(ofile_ajot)
            flags = pl.read_parquet(ofile_flags)

            return (
                all_jets_one_df,
                all_jets_over_time,
                flags,
            )
        all_jets_over_time, flags = track_jets(all_jets_one_df)

        all_jets_over_time.write_parquet(ofile_ajot)
        flags.write_parquet(ofile_flags)

        return (
            all_jets_one_df,
            all_jets_over_time,
            flags,
        )

    def props_as_df(self, categorize: bool = True) -> xr.Dataset:
        ofile_padu = self.path.joinpath("props_as_df_uncat.parquet")
        ofile_pad = self.path.joinpath("props_as_df.parquet")
        if ofile_padu.is_file() and not categorize:
            return pl.read_parquet(ofile_padu)
        if ofile_pad.is_file() and categorize:
            return pl.read_parquet(ofile_pad)
        if ofile_padu.is_file() and categorize:
            props_as_df = categorize_df_jets(pl.read_parquet(ofile_padu))
            props_as_df.write_parquet(ofile_pad)
            return props_as_df
        _, all_jets_over_time, flags = self.track_jets()
        props_as_df = self.compute_jet_props()
        props_as_df = add_persistence_to_props(props_as_df, flags)
        props_as_df = self.add_com_speed(all_jets_over_time, props_as_df)
        props_as_df.write_parquet(ofile_padu)
        props_as_df_cat = categorize_df_jets(props_as_df)
        props_as_df_cat.write_parquet(ofile_pad)
        if categorize:
            props_as_df_cat
        return props_as_df

    def props_over_time(
        self,
        all_jets_over_time: pl.DataFrame,
        props_as_df_uncat: pl.DataFrame,
        save: bool = True,
    ) -> pl.DataFrame:
        out_path = self.path.joinpath("all_props_over_time.parquet")
        if out_path.is_file():
            return pl.read_parquet(out_path)
        index_columns = get_index_columns(props_as_df_uncat)
        props_as_df_uncat = props_as_df_uncat.cast(
            {
                "time": all_jets_over_time["time"].dtype,
                "jet ID": all_jets_over_time["jet ID"].dtype,
            }
        )
        all_props_over_time = all_jets_over_time.join(
            props_as_df_uncat, on=index_columns
        )
        sort_on = ["member"] if "member" in index_columns else []
        sort_on.extend(("flag", "time"))
        all_props_over_time = all_props_over_time.sort(sort_on)
        if save:
            all_props_over_time.write_parquet(out_path)
        return all_props_over_time

    def add_com_speed(
        self, all_jets_over_time: pl.DataFrame, props_as_df: pl.DataFrame
    ) -> pl.DataFrame:
        all_props_over_time = self.props_over_time(
            all_jets_over_time,
            props_as_df,
            save=False,
        )
        com_speed = haversine_from_dl(
            pl.col("mean_lat"),
            pl.col("mean_lat").diff(),
            pl.col("mean_lon").diff(),
        ) / (pl.col("time").cast(pl.Float32).diff() / 1e3)
        agg = {
            "time": pl.col("time"),
            "jet ID": pl.col("jet ID"),
            "com_speed": com_speed,
        }
        com_speed = (
            all_props_over_time.group_by(
                get_index_columns(all_props_over_time, ("member", "flag")),
                maintain_order=True,
            )
            .agg(**agg)
            .explode(["time", "jet ID", "com_speed"])
        )
        index_columns = get_index_columns(
            all_props_over_time, ("member", "time", "jet ID")
        )
        index_exprs = [pl.col(col) for col in index_columns]
        props_as_df = props_as_df.cast(
            {"time": com_speed["time"].dtype, "jet ID": com_speed["jet ID"].dtype}
        ).join(com_speed, on=index_exprs)
        return props_as_df.sort(get_index_columns(props_as_df))

    def compute_extreme_clim(self, varname: str, subsample: int = 5):
        da = self.ds[varname]
        time = pl.Series("time", self.time)
        years = time.dt.year.values
        mask = np.isin(years, np.unique(years)[::subsample])
        opath = self.path.joinpath(f"q{varname}_clim_{subsample}.nc")
        compute_extreme_climatology(da.isel(time=mask), opath)
        quantiles_clim = xr.open_dataarray(opath)
        quantiles = xr.DataArray(
            np.zeros((len(self.ds.time), quantiles_clim.shape[0])),
            coords={
                "time": self.ds.time.values,
                "quantile": quantiles_clim.coords["quantile"].values,
            },
        )
        for qcl in quantiles_clim.transpose():
            dayofyear = qcl.dayofyear
            quantiles[quantiles.time.dt.dayofyear == dayofyear, :] = qcl.values
        quantiles.to_netcdf(
            self.path.joinpath(f"q{varname}_{subsample}.nc")
        )  # TODO: wrong name, fix
