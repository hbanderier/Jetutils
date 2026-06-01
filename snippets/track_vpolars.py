def overlap_vert_dist_polars() -> Tuple[pl.Expr]:
    """
    Generates two expressions; the mean latitudinal distances between jets and the longitudinal overlaps.

    Returns
    -------
    pl. Expr
        Latitudinal mean distance between jets, for each jet pair
    pl.Expr
        Longitudinal overlap, for each jet pair
    """
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
    overlap = (inter12.mean())  
    # new needs to be in old. giving weight to inter21 would let short new jets swallow old big jets, it's weird i think
    return vert_dist.over(row), overlap.over(row)


def _track_jets(df: pl.DataFrame):
    """
    Big sequential worker function to track jets in a DataFrame that has a time column

    Parameters
    ----------
    df : pl.DataFrame
        _description_

    Returns
    -------
    pl.DataFrame
        All jets ordered by their flag first, relative time step second. Absolute time is also recorded.
    pl.DataFrame
        Flag of each jet in `df`
    """
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
    guess_nflags = max(50, len(unique_times))
    n_hemispheres = len(np.unique(np.sign([df["lat"].min(), df["lat"].max()])))
    guess_nflags = guess_nflags * n_hemispheres
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
            print("artificially filling")
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
        connected_mask = (overlaps > 0.4) & (dist_mat < 12)
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
                    continue
                if flagged[j]:
                    continue
                this_flag = potential_flags[i]
                last_valid_index_rel[this_flag] = last_valid_index_rel[this_flag] + 1
                last_valid_index_abs[this_flag] = it
                all_jets_over_time[this_flag, last_valid_index_rel[this_flag]] = (t, j)
                flagged[j] = True
                flags[int(index_start_flags + j), "flag"] = this_flag
                break
        for j in range(n_new):
            if flagged[j]:
                continue
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
    """
    Potentially parallel wrapper for `_track_jets()`. If "members" is a column in `all_jets_one_df`, then paralell-iterates over members. Otherwise just calls `_track_jets()`
    """
    inner = ["time", "jet ID", "orig_points"]
    index_indices = min(
        all_jets_one_df.columns.index("lat"), all_jets_one_df.columns.index("lon")
    )
    levels = all_jets_one_df.columns[:index_indices]
    outer = [level for level in levels if level not in inner]
    all_jets_one_df = coarsen_pl(all_jets_one_df, {"lon": 1, "lat": 1})
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
    """
    Compute the lifetime of jets by counting how many timesteps each flag is present. If `"member"` is a column of `props_as_df` does it independently for each member.

    Parameters
    ----------
    props_as_df : pl.DataFrame
        Jet properties
    flags : pl.DataFrame
        Flags, the output of `track_jets`

    Returns
    -------
    pl.DataFrame
        `props_as_df` with a new column named "persistence"
    """
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


def track_jets(self, force: bool=False) -> Tuple:
    """
    Wraps `track_jets()` for this object's jets
    """
    all_jets_one_df = self.find_jets()
    ofile_ajot = self.path.joinpath("all_jets_over_time.parquet")
    ofile_flags = self.path.joinpath("flags.parquet")

    if all([ofile.is_file() for ofile in (ofile_ajot, ofile_flags)]) and not force:
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
    
def add_com_speed(
    self,
    all_jets_over_time: pl.DataFrame,
    props_as_df: pl.DataFrame,
    force: bool = False,
) -> pl.DataFrame:
    """
    Computes the jets' center of mass (COM) speed as a new property in `props_as_df`.

    Parameters
    ----------
    all_jets_over_time : pl.DataFrame
        the first output of `track_jets()`
    props_as_df : pl.DataFrame
        jet properties
    force : bool, optional
        whether to recompute even if `com_speed` is already a column in `props_as_df`, by default False

    Returns
    -------
    pl.DataFrame
        `props_as_df` with a new column: `com_speed`.
    """
    all_props_over_time = self.props_over_time(
        all_jets_over_time,
        props_as_df,
        save=False,
        force=force,
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