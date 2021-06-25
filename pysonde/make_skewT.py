#!/bin/env python

"""
Plot skew-T diagram of converted soundings
"""
import argparse
import logging

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from metpy.plots import Hodograph, SkewT
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--inputfile",
        metavar="INPUT_FILE",
        help="Converted sounding file (netCDF)",
        default=None,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--outputfile",
        metavar="/some/example/path/filename.pdf",
        help="Output filename for skewT diagram (all file endings"
        " of matplotlib are supported. Formats can be used as well"
        " ({platform}, {direction}, {resolution}, %%Y, %%m, %%d, ...",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        metavar="DEBUG",
        help="Set the level of verbosity [DEBUG, INFO," " WARNING, ERROR]",
        required=False,
        default="INFO",
    )

    parsed_args = vars(parser.parse_args())

    return parsed_args


def setup_logging(verbose):
    assert verbose in ["DEBUG", "INFO", "WARNING", "ERROR"]
    logging.basicConfig(
        level=logging.getLevelName(verbose),
        format="%(levelname)s - %(name)s - %(funcName)s - %(message)s",
        handlers=[
            logging.FileHandler("{}.log".format(__file__)),
            logging.StreamHandler(),
        ],
    )


def find_direction_of_sounding(ascent_rate):
    """
    Calculate direction of sounding by calculating mean
    ascend rate

    Return
    ------
    direction : str
        Direction of sounding (ascending or descending)
    """
    if np.nanmedian(ascent_rate) > 0:
        direction = "ascending"
    elif np.nanmedian(ascent_rate) < 0:
        direction = "descending"
    return direction


def main():
    args = get_args()
    setup_logging(args["verbose"])

    # Define input file
    file = args["inputfile"]
    output = args["outputfile"]

    ds = xr.open_dataset(file)

    ds_sel = ds.isel({"sounding": 0})
    ds_sel = ds_sel.sortby(ds_sel.p, ascending=False)
    attrs = ds_sel.attrs
    ds_sel = ds_sel.metpy.quantify()

    p = ds_sel.p
    T = ds_sel.ta
    Td = ds_sel.dp
    wind_speed = ds_sel.wspd
    wind_dir = ds_sel.wdir
    ascend_rate = ds_sel.dz

    launch_time = attrs["time_of_launch_HHmmss"]
    platform = attrs["platform"]
    resolution = attrs["resolution"].replace(" ", "")

    # Filter nans
    idx = np.where(
        (
            np.isnan(T)
            + np.isnan(Td)
            + np.isnan(p)
            + np.isnan(wind_speed)
            + np.isnan(wind_dir)
        )
        is False,
        True,
        False,
    )
    p = p[idx].metpy.convert_units("hPa")
    T = T[idx].metpy.convert_units("degC")
    Td = Td[idx].metpy.convert_units("degC")
    wind_speed = wind_speed[idx].metpy.convert_units("meter / second")
    wind_dir = wind_dir[idx]

    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

    parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])
    parcel_prof = parcel_prof.metpy.convert_units("degC")

    direction = find_direction_of_sounding(ascend_rate)

    # Create a new figure. The dimensions here give a good aspect ratio
    fig = plt.figure(figsize=(9, 10))
    skew = SkewT(fig, rotation=30)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, "r")
    skew.plot(p, Td, "g")
    # Plot only specific barbs to increase visibility
    pressure_levels_barbs = np.logspace(0.1, 1, 50) * 100

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    # Search for levels by providing pressures
    # (levels is the coordinate not pressure)
    pres_vals = p.isel(level=idx)
    closest_pressure_levels = np.unique(
        [find_nearest(pres_vals, p_) for p_ in pressure_levels_barbs]
    )
    _, closest_pressure_levels_idx, _ = np.intersect1d(
        pres_vals, closest_pressure_levels, return_indices=True
    )

    p_barbs = p.isel({"level": closest_pressure_levels_idx})
    wind_speed_barbs = wind_speed.isel({"level": closest_pressure_levels_idx})
    wind_dir_barbs = wind_dir.isel({"level": closest_pressure_levels_idx})
    u_barbs, v_barbs = mpcalc.wind_components(wind_speed_barbs, wind_dir_barbs)

    # Find nans in pressure
    skew.plot_barbs(p_barbs, u_barbs, v_barbs, xloc=1.06)

    skew.ax.set_ylim(1020, 100)
    skew.ax.set_xlim(-50, 40)

    # Plot LCL as black dot
    skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

    # Plot the parcel profile as a black line
    skew.plot(pres_vals, parcel_prof, "k", linewidth=1.6)

    # Shade areas of CAPE and CIN
    skew.shade_cin(
        pres_vals.metpy.convert_units("hPa").values,
        T.metpy.convert_units("degC").values,
        parcel_prof.metpy.convert_units("degC").values,
    )
    skew.shade_cape(
        pres_vals.metpy.convert_units("hPa").values,
        T.metpy.convert_units("degC").values,
        parcel_prof.metpy.convert_units("degC").values,
    )

    # Plot a zero degree isotherm
    skew.ax.axvline(0, color="c", linestyle="--", linewidth=2)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    # Create a hodograph
    # Create an inset axes object that is 40% width and height of the
    # figure and put it in the upper right hand corner.
    ax_hod = inset_axes(skew.ax, "35%", "35%", loc=1)
    h = Hodograph(ax_hod, component_range=75.0)
    h.add_grid(increment=20)
    h.plot_colormapped(u, v, wind_speed)  # Plot a line colored by wind speed

    # Set title
    sounding_name = ds_sel.sounding.values
    sounding_name_str = str(sounding_name.astype("str"))
    skew.ax.set_title(
        "{sounding}_{direction}".format(sounding=sounding_name_str, direction=direction)
    )

    if output is None:
        filename_fmt = "{platform}_SoundingProfile_skew_{launch_time}_{res}.png".format(
            platform=platform, res=resolution, launch_time=launch_time
        )
        # filename_fmt = launch_time.strftime(filename_fmt)
        output = filename_fmt
    else:
        output = output.format(
            platform=platform, direction=direction, resolution=resolution
        )
        # output = launch_time.strftime(output)
    logging.info("Write output to {}".format(output))
    plt.savefig(output)


if __name__ == "__main__":
    main()
