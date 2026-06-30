#!/bin/python

from ecmwfapi import ECMWFDataServer
from calendar import monthrange
from pathlib import Path
from jetutils.definitions import DATADIR
server = ECMWFDataServer()

first_date = (5, 1)
last_date = (8, 31)
def date_string(year: int, month: int, day: int):
    year = str(year).zfill(4)
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    return f"{year}-{month}-{day}"

current_date = first_date
dates = [date_string(2025, *current_date)]
while not current_date == last_date:
    month, day = current_date
    this_month_range = monthrange(2025, month)[1]
    next_day = day + 2
    if next_day > this_month_range:
        next_day = 1
        month = month + 1
    current_date = (month, next_day)
    dates.append(date_string(2025, *current_date))
    
def hdates_from_date(date: str):
    year, month, day = date.split("-")
    years = range(2005, 2025)
    return "/".join([f"{year}-{month}-{day}" for year in years])

basepath = Path(DATADIR, "S2S", "plev", "wind", "daily")
basepath.mkdir(parents=True, exist_ok=True)

for date in dates:
    hdates = hdates_from_date(date)
    date_str = "-".join(date.split("-")[1:]) 
    opath = basepath.joinpath(date_str + ".nc")
    if opath.is_file():
        print("Already here: ", opath)
        continue
    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date,
        "expver": "prod",
        "hdate": hdates,
        "levtype": "pl",
        "levellist": "200/300",
        "model": "glob",
        "number": "1/2/3/4/5/6/7/8/9/10",
        "origin": "ecmf",
        "param": "130/131/132",
        "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104",
        "stream": "enfh",
        "time": "00:00:00",
        "type": "pf",
        "format": "netcdf",
        "grid": "1.5/1.5",
        "area": "75/-15/30/42.5",
        "target": opath.as_posix(),
    })
