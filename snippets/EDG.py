import numpy as np
import xarray as xr
import xrft
from scipy.stats import gaussian_kde
from definitions import degsin, degcos, RADIUS


def main():
    datadir = "TODO"
    ds_EDG = xr.open_dataset(f"{datadir}/EDG.nc").isel(level=0)
    ### Filter variables
    for varname in ["u", "v"]:
        da_fft_bgrnd = xrft.fft(ds_EDG[f"{varname}wnd"], dim="time")
        da_fft_trans = da_fft_bgrnd.copy()
        freq = np.abs(da_fft_bgrnd.freq_time)
        da_fft_bgrnd[freq > 1 / 10 / 24 / 3600] = 0
        da_fft_trans[
            np.logical_or(freq > 1 / 2 / 24 / 3600, freq < 1 / 6 / 24 / 3600)
        ] = 0
        ds_EDG[f"{varname}bgrnd"] = (
            xrft.ifft(
                da_fft_bgrnd, dim="freq_time", true_phase=True, true_amplitude=True
            )
            .real.assign_coords(time=ds_EDG.time)
            .rename(f"{varname}bgrnd")
        )
        ds_EDG[f"{varname}trans"] = (
            xrft.ifft(
                da_fft_trans, dim="freq_time", true_phase=True, true_amplitude=True
            )
            .real.assign_coords(time=ds_EDG.time)
            .rename(f"{varname}trans")
        )
        ds_EDG.to_netcdf(f"{datadir}/EDG3.nc")
    ### Compute quantities
    ds_EDG["E1"] = (ds_EDG["vtrans"] ** 2 - ds_EDG["utrans"] ** 2) / 2
    ds_EDG["E2"] = -ds_EDG["utrans"] * ds_EDG["vtrans"]
    ### D vector in spherical coordinates, see Obsidian page for this
    ds_EDG["D1"] = (
        ds_EDG["ubgrnd"].differentiate("lon") / RADIUS
        - 1 / degsin(ds_EDG["lat"]) / RADIUS * ds_EDG["vbgrnd"].differentiate("lat")
        - ds_EDG["ubgrnd"] * degcos(ds_EDG["lat"]) / degsin(ds_EDG["lat"]) / RADIUS
    )
    ds_EDG["D2"] = 0.5 * (
        degsin(ds_EDG["lat"])
        / RADIUS
        * (ds_EDG["vbgrnd"] / degsin(ds_EDG["lat"])).differentiate("lon")
        + 1 / degsin(ds_EDG["lat"]) / RADIUS * ds_EDG["ubgrnd"].differentiate("lat")
    )
    ### Generation rate
    ds_EDG["G"] = ds_EDG["E1"] * ds_EDG["D1"] + ds_EDG["E2"] * ds_EDG["D2"]
    ### while we're at it, let's compute the vorticity
    ds_EDG["omega"] = (
        1
        / RADIUS
        / degcos(ds_EDG["lat"])
        * (
            ds_EDG["vwnd"].differentiate("lon")
            - (ds_EDG["uwnd"] * degcos(ds_EDG["lat"])).differentiate("lat")
        )
    )
    ds_EDG["EKE"] = 0.5 * (ds_EDG["utrans"] ** 2 + ds_EDG["vtrans"] ** 2)
    for key in ds_EDG.data_vars:
        ds_EDG[key].to_netcdf(f"{datadir}/{key}.nc")


if __name__ == "__main__":
    main()
