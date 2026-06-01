import time
from itertools import pairwise, product
from pathlib import Path
from jetutils.definitions import DATADIR
import numpy as np 
import xarray as xr
from tqdm import tqdm
import socket
from urllib.request import urlretrieve
from urllib.error import HTTPError, ContentTooShortError, URLError
socket.setdefaulttimeout(180)

    
## functions:

def get_url(varname: str, experiment: str, member: str, timebounds: str, minlon: float, maxlon: float, minlat: float, maxlat: float, year: int):
    h = 6 if varname in ["U", "V", "T", "Q", "OMEGA", "Z3"] else 1 # and sometimes 7??

    base_url = f"https://tds.gdex.ucar.edu/thredds/ncss/grid/files/d651056/CESM2-LE/atm/proc/tseries/day_1/{varname}/b.e21.{experiment}.f09_g17.LE2-{member}.cam.h{h}.{varname}.{timebounds}.nc"
    specs = f"var={varname}&north={maxlat}&west={minlon}&east={maxlon}&south={minlat}&horizStride=1&time_start={year}-01-01T00:00:00Z&time_end={year}-12-31T00:00:00Z&&&accept=netcdf4ext"
    return f"{base_url}?{specs}"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
        
def _download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)
        

def download_url(url, output_path, retries = 10):
    while retries > 0:
        try:
            _download_url(url, output_path)
            return 
        except (HTTPError, ContentTooShortError, URLError, TimeoutError):
            retries = retries - 1
            time.sleep(1.0)
            continue
    _download_url(url, output_path)


## constants:
FORCING_VARIANTS = ["smbb", "cmip6"]
EXPERIMENTS = {
    variant: {
        "historical": f"BHIST{variant}",
        "ssp370": f"BSSP370{variant}",
    }
    for variant in FORCING_VARIANTS
}
YEARBOUNDS = {
    "historical": np.arange(1850, 2021, 10),
    "ssp370": np.arange(2015, 2106, 10),
}
YEARBOUNDS["historical"][-1] = YEARBOUNDS["historical"][-1] - 5
YEARBOUNDS["ssp370"][-1] = YEARBOUNDS["ssp370"][-1] - 4
TIMEBOUNDS = {key: [f"{year1}0101-{year2 - 1}1231" for year1, year2 in pairwise(val)] for key, val in YEARBOUNDS.items()}
YEARS = {key: [list(range(max(year1, 2060) if key == "future" else year1, year2)) for year1, year2 in pairwise(val)] for key, val in YEARBOUNDS.items()}

MEMBERS1 = [f"{year}.{str(number).zfill(3)}" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    MEMBERS1.extend(f"{startyear}.{str(number).zfill(3)}" for number in range(1, 11))
    
MEMBERS2 = [f"r{number}i{year}p1f1" for year, number in zip(range(1001, 1201, 20), range(1, 11))]
for startyear in [1231, 1251, 1281, 1301]:
    MEMBERS2.extend(f"r{number}i{startyear}p1f1" for number in range(1, 11))
    
    
## config:
variables = ["U", "V", "T"] # your responsibility to make sure it exists
forcing_variant = "cmip6" # or smbb
assert forcing_variant in FORCING_VARIANTS
minlon = -80
maxlon = 40
minlat = 15
maxlat = 80

years = {
    "historical": np.arange(1960, 2011),
    "ssp370": np.arange(2015, 2100),
}

periods_to_do = ["historical", "ssp370"]
members_to_do = list(range(50))
# levels = "all" # cannot specify multiple levels with netcdf-select, only one or all
basepath = Path(f"{DATADIR}/CESM2/high_wind")
    
## combine into arguments
timebounds_to_do = {}
years_split = {}
for key, val in TIMEBOUNDS.items():
    these_years = YEARS[key]
    years_to_dl = years[key]
    grr = np.unique([i for i, y in enumerate(these_years) if np.isin(years_to_dl, y).any()])
    timebounds_to_do[key] = [TIMEBOUNDS[key][i] for i in grr]
    years_split[key] = [np.intersect1d(these_years[i], years_to_dl) for i in grr]
    
experiment_dict = EXPERIMENTS[forcing_variant]
members1 = np.array(MEMBERS1)[members_to_do]
members2 = np.array(MEMBERS2)[members_to_do]
iterator = product(periods_to_do, variables, zip(members1, members2))    

for period, varname, (member1, member2) in iterator:
    scratchdir = basepath.joinpath(period).joinpath("raw")
    scratchdir.mkdir(exist_ok=True, parents=True)
    experiment = EXPERIMENTS[forcing_variant][period]
    for years_, tb in zip(years_split[period], timebounds_to_do[period]):
        for year in years_:
            url = get_url(varname, experiment, member1, tb, minlon, maxlon, minlat, maxlat, year)
            scratchpath = scratchdir.joinpath(f"{varname}-{member2}-{year}.nc")
            try:
                xr.open_dataset(scratchpath)
            except (FileNotFoundError, ValueError):
                pass
            else:
                continue
            download_url(url, scratchpath)