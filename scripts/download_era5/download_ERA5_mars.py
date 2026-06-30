from pathlib import Path
from jetutils.definitions import DATADIR, YEARS
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import cdsapi

basepath = Path(f"{DATADIR}/ERA5/thetalev/PV_and_wind/6H")
basepath.mkdir(parents=True, exist_ok=True)

def retrieve(client, request, year, month: int | None = None):
    yearstr = str(year).zfill(4)
    if month is not None:
        monthstr = str(month).zfill(2)
        ofile = basepath.joinpath(f"{yearstr}{monthstr}.nc")
        last_day = calendar.monthrange(year, month)[1]
        last_day = str(last_day).zfill(2)
        date = f'{yearstr}-{monthstr}-01/to/{yearstr}-{monthstr}-{last_day}'
    else:
        ofile = basepath.joinpath(f"{yearstr}.nc")
        date = f'{yearstr}-01-01/to/{yearstr}-12-31'
    if Path(ofile).is_file():
        return
    request.update({"date": date})
    client.retrieve('reanalysis-era5-complete', request, ofile)
    return f"Retrieved {year}"
    
    
def main():
    request = { 
        "class"   : "ea",
        "date"    : "1959-01-01",
        "expver"  : "1",
        'levtype' : 'pt',
        "levelist": '320/330/350',
        'param'   : '54/60/131/132',                  
        'stream'  : 'oper',                  
        'time'    : '00/to/23/by/6', 
        'type'    : 'an',
        'area'    : '90/-90/0/40',
        'grid'    : '0.5/0.5', 
        'format'  : 'netcdf',
    }
    client = cdsapi.Client()
    # client.retrieve('reanalysis-era5-complete', request, basepath.joinpath("full.nc"))
    ## mars only allows on concurrent request, soo this is useless here. Keeping the Threading because Im too lazy to change it
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), year) for year in YEARS for month in range(1, 13)
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
            except:
                print("could not retrieve")
    
        

if __name__ == "__main__":
    main()
