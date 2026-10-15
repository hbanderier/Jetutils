date_format = {"time": "%Y%m%d_%H"}
schema = {
    "time": pl.Datetime("ms"),
    "winner": pl.Int8(),
    "winner2": pl.Int8(),
}
wr_winners = pl.from_pandas(pd.read_csv("/Users/bandelol/Documents/code_local/data/wr_data_package_V1.3/wr_data/WR_LCattribution.txt", parse_dates=["time"], date_format=date_format, skiprows=6, sep=r"\s+", header=None, names=list(schema), usecols=[1, 3, 4]), schema_overrides=schema)

date_format = {"time": "%Y%m%d_%H"}
names = ["AT", "ZO", "ScTr", "AR", "EuBL", "ScBL", "GL"]
schema = {"time": pl.Datetime("ms")} | {
    f"WR_{name}": pl.Float32() for name in names
}

wr_projs = pl.from_pandas(pd.read_csv("/Users/bandelol/Documents/code_local/data/wr_data_package_V1.3/wr_data/WRI_projections.txt", parse_dates=["time"], date_format=date_format, skiprows=5, sep=r"\s+", header=None, names=list(schema)), schema_overrides=schema)

wr_all = wr_projs.join(wr_winners, on="time")

wr_all.write_parquet(basepath.joinpath("grams_wr_2.parquet"))