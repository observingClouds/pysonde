#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot radiosonde data in terminal

Created on Sun Jan 29, 2023

@author: laura

run in terminal via:
    python make_plots_radiosonde.py -i inputfilename -o outputdirectory
for inputfilenames of the form "{ship}_{date}_{time}_{direction}.nc" (e.g. MARIA_S_MERIAN_20221102_110535_ascent.nc),
the required inputfilename is "{ship}_{date}_{time}" (e.g. MARIA_S_MERIAN_20221102_110535).

This script produces 5 plots:
    - the trajectory of the radiosonde
    - Wind speed and direction of ascent and descent
    - temperature T, dew point tau, relative humidity rh, and water vapor mixing ratio for ascent and descent
    - SkewT diagram of ascent
    - SkewT diagram of descent
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import fsspec

import sys
import os.path
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", metavar="INPUT_FILE",
                        help="Converted sounding file (netCDF)", default=None,
                        required=True)

    parser.add_argument("-o", "--outdir", metavar="/some/example/path/",
                        help="Output directory for plots",
                        default="/Users/laura/ownCloud/Campaigns/Radiosonden/plots",
                        required=False)

    parser.add_argument('-v', '--verbose', metavar="DEBUG",
                        help='Set the level of verbosity [DEBUG, INFO,'
                        ' WARNING, ERROR]',
                        required=False, default="INFO")

    parsed_args = vars(parser.parse_args())

    return parsed_args

def main():
    args = get_args()

    # Define input file
    filename = args['inputfile']
    outdir = args['outdir']

    data_ascent = xr.open_dataset(f'{filename}_ascent.nc')
    data_descent = xr.open_dataset(f'{filename}_descent.nc')
    
    ascent_color = 'steelblue'
    descent_color = 'lightsteelblue'
    
    snd = 0 # sounding
    launch_time = np.datetime64(data_ascent.launch_time.values[0],'m')
    timestamp = str(np.datetime64(launch_time,'m')).replace(':','')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    
    # Trajectory of the radiosonde
    fig_track, ax = plt.subplots(figsize=(15,5))
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.coastlines(color="black")
    ax.stock_img()
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    plt.scatter(data_ascent.isel(sounding = snd).lon,data_ascent.isel(sounding = snd).lat, c = data_ascent.isel(sounding = snd).alt_WGS84.values, s =1, cmap = 'hot')
    plt.scatter(data_descent.isel(sounding = snd).lon,data_descent.isel(sounding = snd).lat, c = data_descent.isel(sounding = snd).alt_WGS84.values, s = 1, cmap = 'hot')
    plt.scatter(data_ascent.isel(sounding = snd).lon[0], data_ascent.isel(sounding = snd).lat[0], color = 'yellow', marker = 'x')
    lon_min = min(data_ascent.lon.min(),data_descent.lon.min())
    lon_max = max(data_ascent.lon.max(),data_descent.lon.max())
    ax.set_xlim([lon_min-0.3,lon_max+0.3])
    lat_min = min(data_ascent.lat.min(),data_descent.lat.min())
    lat_max = max(data_ascent.lat.max(),data_descent.lat.max())
    ax.set_ylim([lat_min-0.3,lat_max+0.3])
    ax.set_title(f'Maria S. Merian, launch time: {launch_time}', fontsize = 13)
    alt_max = round(data_ascent.isel(sounding = snd).alt_WGS84.values.max()/1000,1)
    ax.text(0.02, 0.07, f'max. height: {alt_max} km', transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox = props)
        
    # meaured quantites
    fig_quantities, ax = plt.subplots(1,6, sharey= True, tight_layout = True, figsize = (15,5.5))

    ax[0].plot(data_ascent.isel(sounding = snd).ta,data_ascent.isel(sounding = snd).alt_WGS84/1000, color = ascent_color, label = 'ascent')
    ax[0].plot(data_descent.isel(sounding = snd).ta,data_descent.isel(sounding = snd).alt_WGS84/1000, color = descent_color, label = 'descent')
    ax[2].plot(data_ascent.isel(sounding = snd).rh*100,data_ascent.isel(sounding = snd).alt_WGS84/1000, color = ascent_color)
    ax[2].plot(data_descent.isel(sounding = snd).rh*100,data_descent.isel(sounding = snd).alt_WGS84/1000, color = descent_color)
    ax[1].plot(data_ascent.isel(sounding = snd).dp,data_ascent.isel(sounding = snd).alt_WGS84/1000, color = ascent_color)
    ax[1].plot(data_descent.isel(sounding = snd).dp,data_descent.isel(sounding = snd).alt_WGS84/1000, color = descent_color)
    ax[3].plot(data_ascent.isel(sounding = snd).mr*100,data_ascent.isel(sounding = snd).alt_WGS84/1000, color = ascent_color)
    ax[3].plot(data_descent.isel(sounding = snd).mr*100,data_descent.isel(sounding = snd).alt_WGS84/1000, color = descent_color)
    ax[4].plot((data_ascent.isel(sounding = snd).wspd.values * units.knots).to(units.meter/units.second),data_ascent.isel(sounding = snd).alt_WGS84/1000, color = ascent_color)
    ax[4].plot((data_descent.isel(sounding = snd).wspd.values * units.knots).to(units.meter/units.second),data_descent.isel(sounding = snd).alt_WGS84/1000, color = descent_color)
    ax[5].scatter(data_ascent.isel(sounding = snd).wdir,data_ascent.isel(sounding = snd).alt_WGS84/1000, color = ascent_color, label = 'ascent', s = 0.05)
    ax[5].scatter(data_descent.isel(sounding = snd).wdir,data_descent.isel(sounding = snd).alt_WGS84/1000, color = descent_color, label = 'descent', s = 0.05)
    
    
    ax[0].tick_params(labelsize = 11)
    ax[1].tick_params(labelsize = 11)
    ax[2].tick_params(labelsize = 11)
    ax[3].tick_params(labelsize = 11)
    ax[4].tick_params(labelsize = 11)
    ax[5].tick_params(labelsize = 11)
    ax[0].set_ylabel('altitude (km)', fontsize = 13)
    ax[0].set_xlabel('T (K)', fontsize = 13)
    ax[2].set_xlabel('RH (%)', fontsize = 13)
    ax[1].set_xlabel(r'$\tau$ (K)', fontsize = 13)
    ax[3].set_xlabel(r'Mixing ratio (%)', fontsize = 13)
    ax[4].set_xlabel('wind speed (m/s)', fontsize = 13)
    ax[5].set_xlabel('wind direction (degrees)', fontsize = 13)
    ax[0].legend(frameon = False, fontsize = 11)
    fig_quantities.suptitle(f'Maria S. Merian, launch time: {launch_time}', fontsize = 14)
    
    #SkewT                        
    dirs = ['ascent', 'descent']

    for direction in dirs:
    
        if direction == 'ascent':
            data = data_ascent
            p = (data['p'].isel({'sounding': 0}).values * units.Pa)
            T = (data['ta'].isel({'sounding': 0}).values * units.K)
            Td = data['dp'].isel({'sounding': 0}).values * units.K
            wind_speed = data['wspd'].isel({'sounding': 0}).values * units.knots
            wind_speed = wind_speed.to(units.meter/units.second)
            wind_dir = data['wdir'].isel({'sounding': 0}).values * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)
    
        elif direction == 'descent':
            data = data_descent
            p = np.flip(data['p'].isel({'sounding': 0}).values) * units.Pa
            T = np.flip(data['ta'].isel({'sounding': 0}).values) * units.K
            Td = np.flip(data['dp'].isel({'sounding': 0}).values) * units.K
            wind_speed = np.flip(data['wspd'].isel({'sounding': 0}).values) * units.knots
            wind_speed = wind_speed.to(units.meter/units.second)
            wind_dir = np.flip(data['wdir'].isel({'sounding': 0}).values) * units.degrees
            u, v = mpcalc.wind_components(wind_speed, wind_dir)  
    
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])    
    
        parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        
        elp, elt = mpcalc.el(p, T, Td, parcel_prof, which = 'most_cape')
        lfcp, lfct = mpcalc.lfc(p, T, Td, which = 'bottom')
        cape, cin = mpcalc.cape_cin(p, T, Td, parcel_prof, which_lfc='bottom', which_el='most_cape')
        
        # Create a new figure. The dimensions here give a good aspect ratio
        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=30)
    
        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        skew.plot(p, T, 'r')
        skew.plot(p, Td, 'g')
        # Plot only specific barbs to increase visibility
        if direction == 'ascent':
            pressure_levels_barbs = np.logspace(0.1, 1, 50)*100
        elif direction == 'descent':
            pressure_levels_barbs = np.logspace(0.1, data['p'].isel({'sounding': 0}).values.max()/100000, 50)*100
    
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
    
        # Search for levels by providing pressures
        # (levels is the coordinate not pressure)
        pres_vals = data['p'].isel({'sounding': 0}).values/100 #Convertion from Pa to hPa
        closest_pressure_levels = np.unique([find_nearest(pres_vals[~np.isnan(pres_vals)], p_) for p_ in pressure_levels_barbs])
        closest_pressure_levels = closest_pressure_levels[~np.isnan(closest_pressure_levels)]
        _, closest_pressure_levels_idx, _ = np.intersect1d(pres_vals, closest_pressure_levels, return_indices=True)
    
        if direction == 'ascent':
            p_barbs = data['p'].isel({'sounding': 0, 'level': closest_pressure_levels_idx}).values * units.Pa
            wind_speed_barbs = data['wspd'].isel({'sounding': 0, 'level': closest_pressure_levels_idx}).values * units.knots
            wind_dir_barbs = data['wdir'].isel({'sounding': 0, 'level': closest_pressure_levels_idx}).values * units.degrees
        elif direction == 'descent':
            p_barbs = np.flip(data['p'].isel({'sounding': 0, 'level': closest_pressure_levels_idx}).values) * units.Pa
            wind_speed_barbs = np.flip(data['wspd'].isel({'sounding': 0, 'level': closest_pressure_levels_idx}).values) * units.knots
            wind_dir_barbs = np.flip(data['wdir'].isel({'sounding': 0, 'level': closest_pressure_levels_idx}).values) * units.degrees
        u_barbs, v_barbs = mpcalc.wind_components(wind_speed_barbs, wind_dir_barbs)
    
        # Find nans in pressure
        p_non_nan_idx = np.where(~np.isnan(pres_vals))
        skew.plot_barbs(p_barbs, u_barbs, v_barbs)
    
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-50, 40)
    
        # Plot LCL, LFC ab=nd EL as black dot
        skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')
        skew.plot(lcl_pressure, lcl_temperature + 5*units.kelvin, label = 'LCL', markersize = 5, markerfacecolor='black')
        skew.plot(lfcp, lfct, 'ko', markerfacecolor='black')
        skew.plot(elp, elt, 'ko', markerfacecolor='black')
    
        # Plot the parcel profile as a black line
        if direction == 'ascent':
            skew.plot(pres_vals[p_non_nan_idx], parcel_prof, 'k', linewidth=2)
        elif direction == 'descent':
            skew.plot(np.flip(pres_vals[p_non_nan_idx]), parcel_prof, 'k', linewidth=2)
    
        # Shade areas of CAPE and CIN
        if direction == 'ascent':
            skew.shade_cin(pres_vals[p_non_nan_idx], T[p_non_nan_idx], parcel_prof)
            skew.shade_cape(pres_vals[p_non_nan_idx], T[p_non_nan_idx], parcel_prof)
        elif direction == 'descent':
            skew.shade_cin(np.flip(pres_vals[p_non_nan_idx]), T[p_non_nan_idx], parcel_prof)
            skew.shade_cape(np.flip(pres_vals[p_non_nan_idx]), T[p_non_nan_idx], parcel_prof)
    
        # Plot a zero degree isotherm
        skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
    
        skew.plot_dry_adiabats()
        skew.plot_moist_adiabats()
        skew.plot_mixing_lines()
    
        # Create a hodograph
        # Create an inset axes object that is 40% width and height of the
        # figure and put it in the upper right hand corner.
        ax_hod = inset_axes(skew.ax, '47%', '40%', loc=1)
        h = Hodograph(ax_hod, component_range=80.)
        h.add_grid(increment=20)
        h.plot_colormapped(u, v, wind_speed)  # Plot a line colored by wind speed
    
        # Set title
        skew.ax.set_title(f'Maria S. Merian, launch time: {launch_time}, sounding: {snd}, direction: {direction}', fontsize = 12)
        skew.ax.text(0.02, 0.16, 'LCL: '+str(round(lcl_pressure.to('hPa'),1)).replace('hectopascal', 'hPa')+', '+str(round(lcl_temperature.to('degC'),1)).replace('degree_Celsius', '˚C')+'\nLFC: '+str(round(lfcp.to('hPa'),1)).replace('hectopascal', 'hPa')+', '+str(round(lfct.to('degC'),1)).replace('degree_Celsius', '˚C')+ '\nEL: '+str(round(elp.to('hPa'),1)).replace('hectopascal', 'hPa')+', '+str(round(elt.to('degC'),1)).replace('degree_Celsius', '˚C')  +'\nCAPE: '+str(round(cape,1)).replace('joule / kilogram', 'J/kg')+'\nCIN: '+str(round(cin,1)).replace('joule / kilogram', 'J/kg'), transform=skew.ax.transAxes, fontsize=11, verticalalignment='top', bbox = props)
    
        if direction == 'ascent':
            fig_ascent = fig
        elif direction == 'descent':
            fig_descent = fig
    
    check_outdir = os.path.exists(f'{outdir}/Trajectories')
    if check_outdir == False:
        os.mkdir(f'{outdir}/Trajectories')
    check_outdir = os.path.exists(f'{outdir}/Quantities')
    if check_outdir == False:
        os.mkdir(f'{outdir}/Quantities')
    check_outdir = os.path.exists(f'{outdir}/SkewT')
    if check_outdir == False:
        os.mkdir(f'{outdir}/SkewT')
    fig_track.savefig(f'{outdir}/Trajectories/track_{timestamp}.png', bbox_inches='tight')
    fig_quantities.savefig(f'{outdir}/Quantities/radiosonde_data_{timestamp}.pdf', bbox_inches='tight')
    fig_descent.savefig(f'{outdir}/SkewT/skewT_{timestamp}_descent.png', bbox_inches='tight')
    fig_ascent.savefig(f'{outdir}/SkewT/skewT_{timestamp}_ascent.png', bbox_inches='tight')

if __name__ == '__main__':
    main()

