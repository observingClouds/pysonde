#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert sounding files from different sources
to netCDF
"""
import argparse
import logging
import sys

import numpy as np
import xarray as xr
import tqdm
import pyproj
from omegaconf import OmegaConf
import metpy.calc as mpcalc
import os
import time
from metpy.units import units

from . import _helpers as h
from . import readers
from . import meteorology_helpers as mh
from . import thermodynamics as td


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--inputfile",
        metavar="INPUT_FILE",
        help="Single sonde file or file format\n" "including wildcards",
        default=None,
        required=False,
        nargs="+",
        type=h.unixpath,
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="/some/example/path/",
        help="Output folder for converted files (netCDF). You can\n"
        " although not recommended also define an output file\n"
        "(format). However, please share only those with the\n"
        " the default filename.\n"
        " The following formats can be used:\n"
        "\t {platform}\t platform name\n"
        "\t {location}\t platform location\n"
        "\t {direction}\t sounding direction\n"
        "\t\t date format with\n"
        "\t\t %%Y%%m%%d %%H%%M and so on\n"
        "\t\t and others to format the output folder dynamically.",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="MAIN_CONFIG.YML",
        help="Main config file with references\n" "to specific config files",
        default="../config/main.yaml",
        required=False,
        type=h.unixpath,
    )

    parser.add_argument('-m', '--method', metavar='METHOD',
                        help="Interpolation method ('bin' (default), 'linear')",
                        default='bin',
                        required=False)

    parser.add_argument(
        "-v",
        "--verbose",
        metavar="DEBUG",
        help="Set the level of verbosity [DEBUG, INFO," " WARNING, ERROR]",
        required=False,
        default="INFO",
    )

    parsed_args = vars(parser.parse_args())

    if parsed_args["inputfile"] is None:
        parser.error(
            "--inputfile must be defined. For several files"
            "enter the inputfile format with wildcards."
        )

    return parsed_args


def load_reader(filename):
    """
    Infer appropriate reader from filename
    """
    ending = filename.suffix
    if ending == ".mwx":
        from .readers.readers import MW41

        reader = MW41
    elif ending == ".nc":
        from .readers.readers import pysondeL1

        reader = pysondeL1
    else:
        raise h.ReaderNotImplemented(f"Reader for filetype {ending} not implemented")
    return reader


def main(args=None):
    if args is None:
        args = {}
        try:
            args = get_args()
        except ValueError:
            sys.exit()
    else:
        pass

    h.setup_logging(args["verbose"])

    # Combine all configurations
    main_cfg = OmegaConf.load(args["config"])
    cfg = h.combine_configs(main_cfg.configs)

    cfg = h.replace_placeholders_cfg(cfg)

    input_files = h.find_files(args["inputfile"])
    logging.info("Files to process {}".format([file.name for file in input_files]))

    logging.debug("Load reader. All files need to be of same type!")
    # Load correct reader class
    reader_class = load_reader(input_files[0])
    # Configure reader according to config file
    reader = reader_class(cfg)

    for ifile, file in enumerate(tqdm.tqdm(input_files)):
        logging.debug("Reading file number {}".format(ifile))
        sounding = reader.read(file)

        if isinstance(reader, readers.readers.MW41):
            # Split sounding into ascending and descending branch
            sounding_asc, sounding_dsc = sounding.split_by_direction()
            for snd in [sounding_asc, sounding_dsc]:
                if len(snd.profile) < 2:
                    logging.warning(
                        "Sounding ({}) does not contain data. "
                        "Skip sounding-direction of {}".format(
                            snd.meta_data["sounding_direction"], file
                        )
                    )
                    continue
                snd.calculate_additional_variables(cfg)
                snd.convert_sounding_df2ds()
                snd.create_dataset(cfg)
                snd.export(args["output"], cfg)
        elif isinstance(reader, readers.readers.pysondeL1):
            if len(sounding.profile.sounding) != 1:
                raise NotImplementedError(
                    "Level 1 files with more than one sounding are currently not supported"
                )
            ds = sounding.profile.isel({"sounding": 0})
            ds_input = ds.copy()

            # Check monotonic ascent/descent
            if np.all(np.diff(ds.isel(level=slice(20, -1)).alt.values) > 0) or np.all(
                np.diff(ds.isel(level=slice(20, -1)).alt.values) < 0
            ):
                logging.debug("Sounding is monotonic ascending/descending")
            else:
                logging.warning(
                    "Sounding is not monotonic ascending/descending. The ascent rate will be artificial"
                )

            # Geopotential height issue
            # the geopotential height is not a measured coordinate and
            # the same height can occur at different pressure levels
            # here the first occurrence is used
            _, uniq_altitude_idx = np.unique(ds.alt.values, return_index=True)
            ds = ds.isel({"level": uniq_altitude_idx})
            
            # Consistent platform test
            if ifile == 0:
                platform = ds.platform
            else:
                assert (
                    ds.platform == platform
                ), "The platform seems to change from {} to {}".format(
                    platform, ds.platform
                )

            # Unique levels test
            if len(ds.alt) != len(np.unique(ds.alt)):
                logging.error("Altitude levels are not unique of {}".format(file))
                break
            
            # Prepare some data that cannot be linearly interpolated
            u, v = mh.get_wind_components(ds.wdir, ds.wspd)
            ds['u'] = xr.DataArray(u.data, dims=['level'])
            ds['v'] = xr.DataArray(v.data, dims=['level'])

            if 'altitude_WGS84' in ds.keys():
                # Convert lat, lon, alt to cartesian coordinates
                ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
                lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
                x, y, z = pyproj.transform(lla, ecef,
                                           ds.lon.values,
                                           ds.lat.values,
                                           ds.altitude_WGS84.values,
                                           radians=False)
                reader.variable_name_mapping.update({'x': 'x', 'y': 'y', 'z': 'z'})
                for var, val in {'x': x, 'y': y, 'z': z}.items():
                    ds[var] = xr.DataArray(val, dims=['level'])
            else:
                logging.warning(
                    'No WGS84 altitude could be found. The averaging of the position might be faulty especially at the 0 meridian and close to the poles')

            td.metpy.units.units = sounding.unitregistry
            theta = td.calc_theta_from_T(ds['ta'], ds['p'])
            e_s = td.calc_saturation_pressure(ds['ta'])
            w_s = mpcalc.mixing_ratio(e_s, ds['p'].metpy.quantify())
            w = ds['rh'].data * w_s
            q = w / (1 + w)
            
            w['level'] = ds.alt.data
            w = w.rename({'level': 'alt'})
            w = w.expand_dims({"sounding":1})
            q['level'] = ds.alt.data
            q = q.rename({'level': 'alt'})
            q = q.expand_dims({"sounding": 1})
            theta['level'] = ds.alt.data
            theta = theta.rename({'level': 'alt'})
            theta = theta.expand_dims({"sounding": 1})
            ds = ds.rename_vars({"alt":"altitude"})
            ds = ds.rename({'level':'alt'})
            ds['alt'] = ds.altitude.data
            ds = ds.reset_coords()
            ds = ds.expand_dims({"sounding":1})
            # _ds_launch_time = ds["launch_time"].expand_dims({'sounding': 1})
            # del ds["launch_time"]
            # ds["launch_time"] = _ds_launch_time
            
            ds_new = xr.Dataset()  #ds.copy()
            ds_new['mr'] = w.reset_coords(drop=True)
            ds_new['theta'] = theta.reset_coords(drop=True)
            ds_new['specific_humidity'] = q.reset_coords(drop=True)
            
            for var in ds.data_vars:
                if var not in ds_new.data_vars and var not in ds_new.coords:
                    try:
                        ds_new[var] = ds[var]
                    except (NameError):
                        logging.warning(f"Variable {var} not found.")
                        pass
            
            for variable_name_in, variable_name_out in reader.variable_name_mapping_output.items():
                try:
                    ds_new = ds_new.rename({variable_name_in: variable_name_out})
                    ds_new[variable_name_out].attrs = ds[variable_name_in].attrs  # Copy attributes from input
                except (ValueError, KeyError):
                    logging.warning(f"Variable {variable_name_in} not found.")
                    pass
            
            # Interpolation
            interpolation_grid = np.arange(cfg.level2.setup.interpolation_grid_min,
                                           cfg.level2.setup.interpolation_grid_max,
                                           cfg.level2.setup.interpolation_grid_inc)
            if args['method'] == 'linear':
                ds_new = ds_new.dropna(dim='alt',
                                       subset=output_variables,
                                       how='any')
                ds_interp = ds_new.pint.interp(altitude=interpolation_grid)
            elif args['method'] == 'bin':
                interpolation_bins = np.arange(cfg.level2.setup.interpolation_grid_min - cfg.level2.setup.interpolation_grid_inc / 2,
                                               cfg.level2.setup.interpolation_grid_max + cfg.level2.setup.interpolation_grid_inc / 2,
                                               cfg.level2.setup.interpolation_grid_inc)
                ds_interp = ds_new.groupby_bins('altitude', interpolation_bins,
                                                labels=interpolation_grid,
                                                restore_coord_dims=True).mean()
                ds_interp = ds_interp.transpose()
                ds_interp = ds_interp.rename({'altitude_bins': 'alt'})
                
                # Create bounds variable
                ds_interp['alt_bnds'] = xr.DataArray(np.array([interpolation_bins[:-1], interpolation_bins[1:]]).T,
                                                     dims=['alt', 'nv'],
                                                     coords={'alt': ds_interp.alt.data}
                                                     )
                
                ds_interp['launch_time'] = ds_new['launch_time']
                
                ## Interpolation NaN
                ds_interp = ds_interp.interpolate_na('alt', max_gap=cfg.level2.setup.max_gap_fill,
                                                     use_coordinate=True)
                
                for var_in, var_out in reader.variable_name_mapping_output.items():
                    try:
                        ds_interp[var_out] = ds_interp[var_out].pint.quantify(ds_new[var_out].data.units)
                        ds_interp[var_out] = ds_interp[var_out].pint.to(cfg.level2.variables[var_in].attrs.units)
                    except:
                        pass
                    
                dims_2d = ['sounding', 'alt']
                dims_1d = ['alt']
                coords_1d = {'alt': ds_interp.alt.data}
                
                wind_u = ds_interp.isel({'sounding': 0})['wind_u']
                wind_v = ds_interp.isel({'sounding': 0})['wind_v']
                # umrechnen in m/s
                dir, wsp = mh.get_directional_wind(wind_u, wind_v)
                                
                # import pdb;
                # pdb.set_trace()
                
                ds_interp['wind_direction'] = xr.DataArray(dir.expand_dims({'sounding':1}).data,dims=dims_2d,coords=coords_1d)
                ds_interp['wind_speed'] = xr.DataArray(wsp.expand_dims({'sounding':1}).data, dims=dims_2d,coords=coords_1d)
                
                if 'altitude_WGS84' in ds.keys():
                    lon, lat, alt = pyproj.transform(ecef, lla,
                                                     ds_interp['x'].values[0],
                                                     ds_interp['y'].values[0],
                                                     ds_interp['z'].values[0],
                                                     radians=False)
                    for var, val in {'latitude': lat, 'longitude': lon, 'altitude_WGS84': alt}.items():
                        ds_interp[var] = xr.DataArray([val], dims=dims_2d, coords=coords_1d)

                    del ds_interp['x']
                    del ds_interp['y']
                    del ds_interp['z']
                    del ds_interp['altitude_WGS84']

                ds_input = ds_input.sortby('alt')
                ds_input.alt.load()
                ds_input.p.load()
                ds_input = ds_input.reset_coords()
                
                """
                interp_pres = mh.pressure_interpolation(ds_input.p.pint.to("hPa").values,
                                                      ds_input.alt.pint.to("m").values,
                                                      ds_interp.alt.values) * sounding.unitregistry('hPa')
                # * sounding.unitregistry(cfg.level2.variables['p'].attrs.units)
                # hier testen, ob ohne neu-interpolierten Druck die RH Daten anders sind
                ds_interp['pressure'] = xr.DataArray(interp_pres,
                                                     dims=dims_1d,
                                                     coords=coords_1d)
                """
                ds_interp['pressure'] = ds_interp['pressure'].pint.to(cfg.level2.variables['p'].attrs.units)
                ds_interp['pressure'] = ds_interp['pressure'].expand_dims({'sounding':1})
                
                ds_interp['launch_time'] = xr.DataArray([ds_interp.isel({'sounding': 0}).launch_time.item() / 1e9],
                                                        dims=['sounding'])
                # ds_interp['platform'] = xr.DataArray([ds.platform],
                #                                      dims=['sounding'])
                
                # Calculations after interpolation
                # Recalculate temperature and relative humidity from theta and q
                
                temperature = td.calc_T_from_theta(ds_interp.isel(sounding=0)['theta'].pint.to("K"),
                                                ds_interp.isel(sounding=0)['pressure'].pint.to("hPa"))
                ds_interp['temperature'] = xr.DataArray(temperature.data,
                                                        dims=dims_1d,
                                                        coords=coords_1d)
                ds_interp['temperature'] = ds_interp['temperature'].expand_dims({'sounding':1})

                w = (ds_interp.isel(sounding=0)['specific_humidity']) / (
                        1 - ds_interp.isel(sounding=0)['specific_humidity'])
                e_s = td.calc_saturation_pressure(ds_interp.isel(sounding=0)['temperature'])
                w_s = mpcalc.mixing_ratio(e_s,
                                          ds_interp.isel(sounding=0)['pressure'].data)
                relative_humidity = w / w_s * 100

                ds_interp['relative_humidity'] = xr.DataArray(relative_humidity.data,
                                                              dims=dims_1d,
                                                              coords=coords_1d)
                ds_interp['relative_humidity'] = ds_interp['relative_humidity'].expand_dims({'sounding': 1})
                ds_interp['relative_humidity'].data = ds_interp['relative_humidity'].data * units.percent
                # ds_interp = ds_interp.drop('humidity')
                
                dewpoint = td.convert_rh_to_dewpoint(ds_interp.isel(sounding=0)['temperature'],
                                                     ds_interp.isel(sounding=0)['relative_humidity'])
                ds_interp['dewpoint'] = xr.DataArray(dewpoint.data,
                                                        dims=dims_1d,
                                                        coords=coords_1d)
                ds_interp['dewpoint'] = ds_interp['dewpoint'].expand_dims({'sounding':1})
                ds_interp['dewpoint'].data = ds_interp['dewpoint'].data * units.K
                # ds_interp = ds_interp.drop('dew_point')
                
                # ds_interp = ds_interp.drop('altitude')
                
                ds_interp['mixing_ratio'].data = ds_interp['mixing_ratio'].data * units('g/g')
                ds_interp['specific_humidity'].data = ds_interp['specific_humidity'].data * units('g/g')
                
                for variable in ['mixing_ratio', 'theta',
                                  'specific_humidity', 'ascent_rate', 'altitude', 'latitude',
                                  'longitude', 'wind_u',
                                  'wind_v', 'alt_bnds'
                                  ]:
                    ds_interp[variable] = ds_interp[variable].expand_dims({'sounding':1})
                
                # Count number of measurements within each bin
                ds_interp['N_ptu'] = xr.DataArray(
                    ds_new.pressure.groupby_bins('alt', interpolation_bins, labels=interpolation_grid,
                                                 restore_coord_dims=True).count().values,
                    dims=dims_2d,
                    coords=coords_1d)
                ds_interp['N_gps'] = xr.DataArray(
                    ds_new.latitude.groupby_bins('alt', interpolation_bins, labels=interpolation_grid,
                                                 restore_coord_dims=True).count().values,
                    dims=dims_2d,
                    coords=coords_1d)

                # Cell method used
                data_exists = np.where(np.isnan(ds_interp.isel(sounding=0).pressure), False, True)
                data_mean = np.where(np.isnan(ds_interp.isel(sounding=0).N_ptu), False, True)  # no data or interp: nan; mean > 0
                data_method = np.zeros_like(data_exists, dtype='uint')
                data_method[np.logical_and(data_exists, data_mean)] = 2
                data_method[np.logical_and(data_exists, ~data_mean)] = 1
                ds_interp['m_ptu'] = xr.DataArray([data_method], dims=dims_2d, coords=coords_1d)
                ds_interp['N_ptu'].values[0,np.logical_and(~data_mean, data_method > 0)] = 0

                data_exists = np.where(np.isnan(ds_interp.isel(sounding=0).latitude), False, True)
                data_mean = np.where(np.isnan(ds_interp.isel(sounding=0).N_gps), False, True)  # no data or interp: nan; mean > 0
                data_method = np.zeros_like(data_exists, dtype='uint')
                data_method[np.logical_and(data_exists, data_mean)] = 2
                data_method[np.logical_and(data_exists, ~data_mean)] = 1
                ds_interp['m_gps'] = xr.DataArray([data_method], dims=dims_2d, coords=coords_1d)
                ds_interp['N_gps'].values[0,np.logical_and(~data_mean, data_method > 0)] = 0
                
                # direction = get_direction(ds_interp, ds)
                most_common_vertical_movement = np.argmax(
                    [np.count_nonzero(ds_interp.ascent_rate > 0), np.count_nonzero(ds_interp.ascent_rate < 0)]
                )
                ds_interp['ascent_flag'] = xr.DataArray([most_common_vertical_movement], dims=['sounding'])
                    
                # # Copy trajectory id from level1 dataset
                # ds_interp['sounding'] = xr.DataArray([ds['sounding'].values])#, dims=['sounding'])
                ds_interp.sounding.attrs = ds['sounding'].attrs
                
                import pandas as pd
                from netCDF4 import num2date
        
                def convert_num2_date_with_nan(num, format):
                    if not np.isnan(num):
                        return num2date(num, format, only_use_python_datetimes=True, only_use_cftime_datetimes=False)
                    else:
                        return pd.NaT
        
                convert_nums2date = np.vectorize(convert_num2_date_with_nan)
                
                # ds_interp['flight_time'].data = convert_nums2date(ds_interp.flight_time.data, "seconds since 1970-01-01")
                ds_interp['launch_time'].data = convert_nums2date(ds_interp.launch_time.data, "seconds since 1970-01-01")
                
                # merged_conf = OmegaConf.merge(config.level2, meta_data_cfg)
                # merged_conf._set_parent(OmegaConf.merge(config, meta_data_cfg))
                ds_interp.attrs = ds.attrs
                
                ds_interp = h.replace_global_attributes(ds_interp, cfg)
                ds_interp.attrs["source"] = str(file).split("/")[-1]
                
                variables = reader.variable_name_mapping_output.items()
                ds_interp = h.set_additional_var_attributes(ds_interp, cfg.level2.variables, variables)
                
                # Transpose dataset if necessary
                for variable in ds_interp.data_vars:
                     if variable == 'alt_bnds': continue
                     dims = ds_interp[variable].dims
                     if (len(dims) == 2) and (dims[0] != 'sounding'):
                         ds_interp[variable] = ds_interp[variable].T
        
                # time_dt = pd.Timestamp(np.datetime64(ds_interp.isel({'sounding': 0}).launch_time.data.astype("<M8[ns]")))
                
                if ds_interp.ascent_flag.values[0] == 0:
                    direction = 'AscentProfile'
                elif ds_interp.ascent_flag.values[0] == 1:
                    direction = 'DescentProfile'
                    
                # time_fmt = time_dt.strftime('%Y%m%dT%H%M')
                outfile = args['output']+'{platform}_Radiosonde_{level}_{date_YYYYMMDDTHHMM}_{location_coord}_{direction}.nc'.format(
                    platform=ds_interp.attrs['platform'],
                    level='Level2',
                    date_YYYYMMDDTHHMM=ds_interp.attrs['date_YYYYMMDDTHHMM'],
                    location_coord=ds_interp.attrs['location_coord'],
                    direction=direction
                    )
                
                logging.info('Write output to {}'.format(outfile))
                h.write_dataset(ds_interp, outfile)
                
if __name__ == "__main__":
    main()
