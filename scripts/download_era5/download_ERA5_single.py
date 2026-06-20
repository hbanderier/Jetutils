import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cdsapi import Client
import xarray as xr
from jetutils.definitions import DATADIR, N_WORKERS
from jetutils.data import standardize

basepath = Path(f"{DATADIR}/ERA5/surf/tp/raw")
basepath.mkdir(parents=True, exist_ok=True)
suffix = ""
suffix = f"_{suffix}" if suffix != "" else ""


def retrieve(client: Client, request: dict, year: int, month: int | None = None):
    yearstr = str(year).zfill(4)
    if month is not None:
        month = str(month).zfill(2)
        ofile = basepath.joinpath(f"{yearstr}{month}{suffix}.nc")
        ofile2 = basepath.joinpath(f"../6H/{yearstr}{month}{suffix}.nc")
    else:
        month = [str(i).zfill(2) for i in range(1, 13)]
        ofile = basepath.joinpath(f"{yearstr}{suffix}.nc")
        ofile2 = basepath.joinpath(f"../6H/{yearstr}{suffix}.nc")
    if ofile.is_file() or ofile2.is_file():
        return
    print(ofile, ofile2)
    request.update({"year": yearstr, "month": month})
    client.retrieve("reanalysis-era5-single-levels", request, ofile)

    da = xr.open_dataset(ofile).chunk("auto")
    print("da.dims:", da.dims)
    if "step" in da.dims:
        da = da.stack(
            {"valid_time": ("time", "step")}, create_index=False
        ).reset_coords(["time", "step", "number", "surface"], drop=True)
    da = standardize(da).transpose("time", "lat", "lon")
    da = da.sel(time=da.time.dt.year == year)
    da = da.resample(time="6h").sum()
    da.to_netcdf(ofile2)

    # ds: xr.Dataset = standardize(xr.open_dataset(ofile, engine="cfgrib")).transpose("time", "lat", "lon")
    # ds = ds.resample(time="6h").sum()
    # ogroup = basepath.joinpath("full.zarr")
    # if ogroup.is_dir():
    #     kwargs = {"mode": "a", "append_dim": "time"}
    # else:
    #     kwargs = {"mode": "w"}

    # ogroup = basepath.joinpath("full.zarr")
    # ds.to_zarr(ogroup, **kwargs)
    os.remove(ofile)
    return f"Retrieved {yearstr}, {month}"


def main():
    request = {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": ["1959"],
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [90, -100, 0, 60],
        "grid": "0.5/0.5",
    }
    client = Client()
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year)
            for year in range(1959, 2023)
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except Exception as e:
                print("Error")
                print(
                    f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
                )


if __name__ == "__main__":
    main()
