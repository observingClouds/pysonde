#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot radiosonde level 1 data

Created on Sun Jan 29, 2023

@author: laura

run the script:
    python plot_radiosonde_data.py -i inputfilename -o outputdirectory
for level 1 radiosonde files. The inputfilename should include the direction (ascent or descent). If not, no SkewT diagram is made.

This script produces 3 plots:
    - trajectory
    - temperature T, dew point tau, relative humidity rh, water vapor mixing ratio, wind speed, and wind direction
    - SkewT diagram (only if direction, i.e. ascent or descent, is given in inputfilename)
"""

import argparse
import os
from decimal import ROUND_HALF_UP, Decimal

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pint_xarray  # noqa: F401
import xarray as xr
from metpy.plots import Hodograph, SkewT
from metpy.units import units


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
        "--altitude",
        metavar="ALTITUDE_VARIABLE",
        help="Altitude variable. Default: alt (geopotential height). Alternative alt_WGS84 (height above reference ellipsoid). Variable names might change depending on the configuration given in config/level1.yaml.",
        default="alt",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        metavar="/some/example/path/",
        help="Output directory for plots",
        default="/Users/laura/ownCloud/Campaigns/Radiosonden/plots",
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


def data4skewt(direction, data):
    if direction == "ascent":
        p = data["p"].isel({"sounding": 0}).pint.to(units.Pa).values * units.Pa
        T = data["ta"].isel({"sounding": 0}).pint.to(units.K).values * units.K
        Td = data["dp"].isel({"sounding": 0}).pint.to(units.K).values * units.K
        wind_speed = (
            data["wspd"]
            .isel({"sounding": 0})
            .pint.to(units.meter / units.second)
            .values
            * units.meter
            / units.second
        )
        wind_dir = (
            data["wdir"].isel({"sounding": 0}).pint.to(units.degrees).values
            * units.degrees
        )
        u, v = mpcalc.wind_components(wind_speed, wind_dir)

    elif direction == "descent":
        p = np.flip(data["p"].isel({"sounding": 0}).pint.to(units.Pa).values) * units.Pa
        T = np.flip(data["ta"].isel({"sounding": 0}).pint.to(units.K).values) * units.K
        Td = np.flip(data["dp"].isel({"sounding": 0}).pint.to(units.K).values) * units.K
        wind_speed = (
            np.flip(
                data["wspd"]
                .isel({"sounding": 0})
                .pint.to(units.meter / units.second)
                .values
            )
            * units.meter
            / units.second
        )
        wind_dir = (
            np.flip(data["wdir"].isel({"sounding": 0}).pint.to(units.degrees).values)
            * units.degrees
        )
        u, v = mpcalc.wind_components(wind_speed, wind_dir)
    return p, T, Td, wind_speed, wind_dir, u, v


def main():
    args = get_args()

    # Define input file
    filename = args["inputfile"]

    data = xr.open_dataset(filename)
    data = data.pint.quantify(unit_registry=units)
    for coord_name in data.coords:
        data.coords[coord_name] = data.coords[coord_name].pint.quantify()
    if "ascent" in filename:
        direction = "ascent"
    elif "descent" in filename:
        direction = "descent"
    else:
        direction = "unknown"

    quantities_color = "steelblue"

    # Check and create out directories
    outdir = args["outdir"]
    check_outdir = os.path.exists(f"{outdir}/Trajectories")
    if not check_outdir:
        os.makedirs(f"{outdir}/Trajectories")
    check_outdir = os.path.exists(f"{outdir}/Quantities")
    if not check_outdir:
        os.makedirs(f"{outdir}/Quantities")
    check_outdir = os.path.exists(f"{outdir}/SkewT")
    if not check_outdir:
        os.makedirs(f"{outdir}/SkewT")

    snd = 0  # sounding
    launch_time = np.datetime64(data.launch_time.values[0], "m")
    timestamp = str(np.datetime64(launch_time, "m")).replace(":", "")

    lats = data.isel(sounding=snd).lat.pint.to("degree_north")
    lons = data.isel(sounding=snd).lon.pint.to("degree_east")
    alts = (
        (data.isel(sounding=snd)[args["altitude"]])
        .pint.to(units.km)
        .pint.dequantify()
        .values
    )

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)

    # Trajectory of the radiosonde
    fig_track = plt.figure(figsize=(15, 5))
    ax = fig_track.add_subplot(projection=ccrs.PlateCarree())
    ax.coastlines(color="black")
    ax.stock_img()
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    traj = ax.scatter(
        lons,
        lats,
        c=alts,
        s=1,
        cmap="hot",
    )
    plt.colorbar(traj, label="altitude (km)", pad=0.06)
    lon_min = lons.min().values
    lon_max = lons.max().values
    ax.set_xlim([lon_min - 0.3, lon_max + 0.3])
    lat_min = lats.min().values
    lat_max = lats.max().values
    ax.set_ylim([lat_min - 0.3, lat_max + 0.3])
    ax.set_title(
        f"Radiosonde trajectory: launch time: {launch_time}, direction: {direction}",
        fontsize=13,
    )
    alt_max = alts.max().round(1)
    alt_max = Decimal(str(alt_max)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    ax.text(
        0.02,
        0.07,
        f"max. height: {alt_max} km",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )
    fig_track.savefig(
        f"{outdir}/Trajectories/track_{timestamp}_{direction}.png", bbox_inches="tight"
    )

    # measured quantites
    fig_quantities, ax = plt.subplots(
        1, 6, sharey=True, tight_layout=True, figsize=(15, 5.5)
    )

    ax[0].plot(
        data.isel(sounding=snd).ta.pint.to(units.K),
        alts,
        color=quantities_color,
    )
    ax[1].plot(
        data.isel(sounding=snd).dp.pint.to(units.K),
        alts,
        color=quantities_color,
    )
    ax[2].plot(
        data.isel(sounding=snd).rh.pint.to(units.percent),
        alts,
        color=quantities_color,
    )
    ax[3].plot(
        data.isel(sounding=snd).mr.pint.to(units.percent),
        alts,
        color=quantities_color,
    )
    ax[4].plot(
        data.isel(sounding=snd).wspd.pint.to(units.meter / units.second),
        alts,
        color=quantities_color,
    )
    ax[5].plot(
        data.isel(sounding=snd).wdir.pint.to(units.degrees),
        alts,
        color=quantities_color,
    )

    ax[0].tick_params(labelsize=11)
    ax[1].tick_params(labelsize=11)
    ax[2].tick_params(labelsize=11)
    ax[3].tick_params(labelsize=11)
    ax[4].tick_params(labelsize=11)
    ax[5].tick_params(labelsize=11)
    ax[0].set_ylabel("altitude (km)", fontsize=13)
    ax[0].set_xlabel("T (K)", fontsize=13)
    ax[2].set_xlabel("RH (%)", fontsize=13)
    ax[1].set_xlabel(r"$\tau$ (K)", fontsize=13)
    ax[3].set_xlabel(r"mixing ratio (%)", fontsize=13)
    ax[4].set_xlabel("wind speed (m/s)", fontsize=13)
    ax[5].set_xlabel("wind direction (degree)", fontsize=13)
    fig_quantities.suptitle(
        f"Radiosonde observations: launch time: {launch_time}, direction: {direction}",
        fontsize=14,
    )
    fig_quantities.savefig(
        f"{outdir}/Quantities/radiosonde_data_{timestamp}_{direction}.pdf",
        bbox_inches="tight",
    )

    # SkewT
    if direction == "unknown":
        print("No SkewT plot created since direction unknown.")
    else:
        p, T, Td, wind_speed, wind_dir, u, v = data4skewt(direction, data)

        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])

        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")

        elp, elt = mpcalc.el(p, T, Td, parcel_prof, which="most_cape")
        lfcp, lfct = mpcalc.lfc(p, T, Td, which="bottom")
        cape, cin = mpcalc.cape_cin(
            p, T, Td, parcel_prof, which_lfc="bottom", which_el="most_cape"
        )

        # Create a new figure. The dimensions here give a good aspect ratio
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=30)

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        skew.plot(p, T, "r")
        skew.plot(p, Td, "g")
        # Plot only specific barbs to increase visibility
        if direction == "ascent":
            pressure_levels_barbs = np.logspace(0.1, 1, 50) * 100
        elif direction == "descent":
            pressure_levels_barbs = (
                np.logspace(
                    0.1, data["p"].isel({"sounding": 0}).values.max() / 100000, 50
                )
                * 100
            )

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        # Search for levels by providing pressures
        # (levels is the coordinate not pressure)
        pres_vals = (
            data["p"].isel({"sounding": 0}).pint.to(units.hPa)
        ).values  # Convertion from Pa to hPa

        closest_pressure_levels = np.unique(
            [
                find_nearest(pres_vals[~np.isnan(pres_vals)], p_)
                for p_ in pressure_levels_barbs
            ]
        )
        closest_pressure_levels = closest_pressure_levels[
            ~np.isnan(closest_pressure_levels)
        ]
        _, closest_pressure_levels_idx, _ = np.intersect1d(
            pres_vals, closest_pressure_levels, return_indices=True
        )

        if direction == "ascent":
            p_barbs = (
                data["p"]
                .isel({"sounding": 0, "level": closest_pressure_levels_idx})
                .values
                * units.Pa
            )
            wind_speed_barbs = (
                data["wspd"]
                .isel({"sounding": 0, "level": closest_pressure_levels_idx})
                .values
                * units.knots
            )
            wind_dir_barbs = (
                data["wdir"]
                .isel({"sounding": 0, "level": closest_pressure_levels_idx})
                .values
                * units.degrees
            )
        elif direction == "descent":
            p_barbs = (
                np.flip(
                    data["p"]
                    .isel({"sounding": 0, "level": closest_pressure_levels_idx})
                    .values
                )
                * units.Pa
            )
            wind_speed_barbs = (
                np.flip(
                    data["wspd"]
                    .isel({"sounding": 0, "level": closest_pressure_levels_idx})
                    .values
                )
                * units.knots
            )
            wind_dir_barbs = (
                np.flip(
                    data["wdir"]
                    .isel({"sounding": 0, "level": closest_pressure_levels_idx})
                    .values
                )
                * units.degrees
            )
        u_barbs, v_barbs = mpcalc.wind_components(wind_speed_barbs, wind_dir_barbs)

        # Find nans in pressure
        p_non_nan_idx = np.where(~np.isnan(pres_vals))
        skew.plot_barbs(p_barbs, u_barbs, v_barbs)

        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-50, 40)

        # Plot LCL, LFC ab=nd EL as black dot
        skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")
        skew.plot(
            lcl_pressure,
            lcl_temperature + 5 * units.kelvin,
            label="LCL",
            markersize=5,
            markerfacecolor="black",
        )
        skew.plot(lfcp, lfct, "ko", markerfacecolor="black")
        skew.plot(elp, elt, "ko", markerfacecolor="black")

        # Plot the parcel profile as a black line
        if direction == "ascent":
            skew.plot(pres_vals[p_non_nan_idx], parcel_prof, "k", linewidth=2)
        elif direction == "descent":
            skew.plot(np.flip(pres_vals[p_non_nan_idx]), parcel_prof, "k", linewidth=2)

        # Shade areas of CAPE and CIN
        if direction == "ascent":
            skew.shade_cin(pres_vals[p_non_nan_idx], T[p_non_nan_idx], parcel_prof)
            skew.shade_cape(pres_vals[p_non_nan_idx], T[p_non_nan_idx], parcel_prof)
        elif direction == "descent":
            skew.shade_cin(
                np.flip(pres_vals[p_non_nan_idx]), T[p_non_nan_idx], parcel_prof
            )
            skew.shade_cape(
                np.flip(pres_vals[p_non_nan_idx]), T[p_non_nan_idx], parcel_prof
            )

        # Plot a zero degree isotherm
        skew.ax.axvline(0, color="c", linestyle="--", linewidth=2)

        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()

        # Create a hodograph
        ax_hod = fig.add_axes([0.55, 0.54, 0.29, 0.29])
        h = Hodograph(ax_hod, component_range=80.0)
        h.add_grid(increment=20)
        h.plot_colormapped(u, v, wind_speed)  # Plot a line colored by wind speed

        # Set title
        skew.ax.set_title(
            f"Launch time: {launch_time}, sounding: {snd}, direction: {direction}",
            fontsize=12,
        )
        skew.ax.text(
            0.02,
            0.16,
            "LCL: "
            + str(round(float(lcl_pressure.to("hPa").magnitude), 1))
            + " hPa"
            + ", "
            + str(round(float(lcl_temperature.to("degC").magnitude), 1))
            + "˚C"
            + "\nLFC: "
            + str(round(float(lfcp.to("hPa").magnitude), 1))
            + " hPa"
            + ", "
            + str(round(float(lfct.to("degC").magnitude), 1))
            + "˚C"
            + "\nEL: "
            + str(round(float(elp.to("hPa").magnitude), 1))
            + " hPa"
            + ", "
            + str(round(float(elt.to("degC").magnitude), 1))
            + "˚C"
            + "\nCAPE: "
            + str(round(float(cape.magnitude), 1))
            + " J/kg"
            + "\nCIN: "
            + str(round(float(cin.magnitude), 1))
            + " J/kg",
            transform=skew.ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=props,
        )
        fig_skewt = fig
        fig_skewt.savefig(
            f"{outdir}/SkewT/skewT_{timestamp}_{direction}.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )


if __name__ == "__main__":
    main()
