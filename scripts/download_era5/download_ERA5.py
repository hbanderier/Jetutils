import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cdsapi import Client
import xarray as xr
from jetutils.definitions import DATADIR, N_WORKERS
from jetutils.data import standardize

basepath = Path(f"{DATADIR}/ERA5/plev/uv/6H")
basepath.mkdir(parents=True, exist_ok=True)
suffix = ""
suffix = f"_{suffix}" if suffix != "" else ""


def retrieve(client: Client, request: dict, year: int, month: int | None = None):
    year = str(year).zfill(4)
    if month is not None:
        month = str(month).zfill(2)
        ofile = basepath.joinpath(f"{year}{month}{suffix}.nc")
    else:
        month = [str(i).zfill(2) for i in range(1, 13)]
        ofile = basepath.joinpath(f"{year}{suffix}.nc")
    if Path(ofile).is_file():
        return
    request.update({"year": year, "month": month})
    client.retrieve("reanalysis-era5-pressure-levels", request, ofile)
    # ds = standardize(xr.open_dataset(ofile))
    # ogroup = basepath.joinpath("full.zarr")
    # if ogroup.is_dir():
    #     kwargs = {"mode": "a", "append_dim": "time"}
    # else:
    #     kwargs = {"mode": "w"}

    # ogroup = basepath.joinpath("full.zarr")
    # ds.to_zarr(ogroup, **kwargs)
    # os.remove(ofile)
    return f"Retrieved {year}, {month}"


def main():
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        ],
        "year": "2023",
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
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "pressure_level": ["175", "200", "225", "250", "300", "350", "850"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [80, -80, 15, 40],
        "grid": "0.5/0.5",
    }
    client = Client()
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year, month)
            for year in range(1959, 2025)
            for month in range(1, 13)  # modify this if needed
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
