"""Readers for different sounding formats"""

import datetime as dt
import logging
import os
import sys
from functools import partial

import numpy as np
import pandas as pd  # noqa: F401
import pint
import pint_xarray  # noqa: F401
import xarray as xr
from metpy.units import units

sys.path.append(os.path.dirname(__file__))
import reader_helpers as rh  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import sounding as snd  # noqa: E402


class Level0:
    def __init__(self, cfg):
        """Configure reader"""
        # Configure, which values need to be read and how they are named
        self.sync_sounding_values = cfg.level0.get("sync_sounding_items", None)
        self.radiosondes_values = cfg.level0.get("radiosondes_sounding_items", None)
        # self.radiosondes_values = cfg.level0.radiosondes_sounding_items
        self.variable_name_mapping = cfg.level0.dictionary_input
        self.units = cfg.level0.input_units
        self.unitregistry = self._create_unitregistry()

    def _create_unitregistry(self):
        ureg = pint.UnitRegistry()
        ureg.define("fraction = [] = frac")
        ureg.define("percent = 1e-2 frac = pct")
        return ureg

    def read(self, mwx_file):
        raise NotImplementedError("This method needs to be implemented in the subclass")


class MW41(Level0):
    """
    Reader for MW41 mwx files
    """

    def read(self, mwx_file):
        def _get_flighttime(radio_time, start_time, launch_time):
            """
            f_flighttime = lambda radio_time: start_time + dt.timedelta(
                seconds=radio_time - float(launch_time)
            )
            """
            return start_time + dt.timedelta(seconds=radio_time - float(launch_time))

        with rh.MWX(mwx_file) as mwx:
            decompressed_files = mwx.decompressed_files

            # Get the files SynchronizedSoundingData.xml, Soundings.xml, ...
            a1, sync_filename = rh.check_availability(
                decompressed_files, "SynchronizedSoundingData.xml", True
            )
            a2, snd_filename = rh.check_availability(
                decompressed_files, "Soundings.xml", True
            )
            a3, radio_filename = rh.check_availability(
                decompressed_files, "Radiosondes.xml", True
            )
            if np.any([not a1, not a2, not a3]):
                logging.warning(
                    "No sounding data found in {}. Skipped".format(mwx_file)
                )
                return

            # Read Soundings.xml to get base time
            itemlist = rh.read_xml(snd_filename)
            for i, item in enumerate(itemlist):
                begin_time = item.attributes["BeginTime"].value
                launch_time = item.attributes["LaunchTime"].value
                station_altitude = item.attributes["Altitude"].value
            begin_time_dt = dt.datetime.strptime(begin_time, "%Y-%m-%dT%H:%M:%S.%f")

            # Read sounding data
            pd_snd = rh.get_sounding_profile(sync_filename, self.sync_sounding_values)

            # Read Radiosounding.xml to get sounding metadata
            sounding_meta_dict = rh.get_sounding_metadata(
                radio_filename, self.radiosondes_values
            )
            sounding_meta_dict["source"] = str(mwx_file)

        pd_snd = rh.rename_variables(pd_snd, self.variable_name_mapping)
        sounding_meta_dict = rh.rename_metadata(
            sounding_meta_dict, self.variable_name_mapping
        )

        # Attach units where provided
        pd_snd = rh.attach_units(pd_snd, self.units, self.unitregistry)

        # Create flight time
        f_flighttime = partial(
            _get_flighttime, start_time=begin_time_dt, launch_time=launch_time
        )
        pd_snd["flight_time"] = pd_snd.RadioRxTimePk.apply(f_flighttime)

        # Write to class
        sounding = snd.Sounding()
        sounding.profile = pd_snd
        sounding.meta_data = sounding_meta_dict
        sounding.meta_data["launch_time"] = launch_time
        sounding.meta_data["begin_time"] = begin_time_dt
        sounding.meta_data["station_altitude"] = station_altitude
        sounding.unitregistry = self.unitregistry
        sounding.source = mwx_file
        return sounding


class METEOMODEM(Level0):
    """
    Reader for level 0 data from Meteomodem sondes
    """

    def check_TU_sensor(self, snd):
        """
        Meteomodem soundings have occasionally
        a mismatch between T, Td and RH
        """
        idx = np.where(
            snd.Temperature == snd.Dewpoint
        )  # might need conversion to kelvin
        _snd = snd.iloc[idx]
        idx_pd = _snd.index
        if np.any(
            snd.loc[idx_pd, "Humidity"] != 100
        ):  # might need conversion to percent
            logging.warning("Humidity mismatch, setting Td to nan")
            snd.loc[idx_pd, "Dewpoint"] = np.nan
        return snd

    def read(self, cor_file, bufr_file=None, round_like_bufr=False):
        pd_snd = pd.read_csv(cor_file, delimiter="\t")

        def _get_date_information_from_filename(cor_file):
            basename = os.path.basename(cor_file)
            date_str = basename.split("_")[1]
            date_dt = dt.datetime.strptime(date_str, "%Y%m%d%H").date()
            first_time_hour = np.round(pd_snd.Time[0] / (60 * 60))
            if (first_time_hour > 12) and (
                dt.datetime.strptime(date_str, "%Y%m%d%H").hour == 0
            ):
                date_dt = (
                    date_dt - dt.timedelta(days=1)
                )  # hour is the forecast hour and therefor at midnight -1 need to be subtracted
            elif (first_time_hour < 12) and (
                dt.datetime.strptime(date_str, "%Y%m%d%H").hour == 0
            ):
                date_dt = date_dt
            else:
                date_dt = date_dt
            return date_dt

        def _get_flighttime(seconds, date_dt):
            return dt.datetime.combine(
                date_dt, dt.time(hour=0, minute=0)
            ) + dt.timedelta(seconds=seconds)

        date_dt = _get_date_information_from_filename(cor_file)
        pd_snd["flight_time"] = pd_snd.Time.apply(_get_flighttime, date_dt=date_dt)

        # Rename variables
        pd_snd = rh.rename_variables(pd_snd, self.variable_name_mapping)

        # Convert radians to degree
        pd_snd["Latitude"] = np.rad2deg(pd_snd["Latitude"])
        pd_snd["Longitude"] = np.rad2deg(pd_snd["Longitude"])

        # Attach units where provided
        pd_snd = rh.attach_units(pd_snd, self.units, self.unitregistry)

        if round_like_bufr:
            logging.debug("Data is rounded similar to BUFR message output")
            pd_snd = pd_snd.round(
                decimals={
                    "humidity": 2,
                    "wind_direction": 0,
                    "temperature": 2,
                    "latitude": 5,
                    "longitude": 5,
                    "wind_speed": 1,
                    "pressure": 2,
                    "height": 1,
                }
            )

        # Quality check
        pd_snd = self.check_TU_sensor(pd_snd)

        pd_snd["Dropping"] = np.where(pd_snd["ascent_rate"].rolling(10).sum() < 0, 1, 0)

        launch_time = pd_snd.flight_time.values[0]

        # Write to class
        sounding = snd.Sounding()
        sounding.profile = pd_snd
        # sounding.meta_data = sounding_meta_dict
        sounding.meta_data["launch_time"] = launch_time
        # sounding.meta_data["begin_time"] = begin_time_dt
        # sounding.meta_data["station_altitude"] = station_altitude
        sounding.unitregistry = self.unitregistry
        sounding.source = cor_file
        if bufr_file is not None:
            sounding.source += ", " + bufr_file
        return sounding


class pysondeL1:
    """
    Reader for level 1 data created with pysonde
    """

    def __init__(self, cfg):
        """Configure reader"""
        # Configure, which values need to be read and how they are named
        self.variable_name_mapping_input = self._get_variable_name_mapping(
            cfg["level1"]
        )
        self.variable_name_mapping_output = self._get_variable_name_mapping(
            cfg["level2"]
        )

    def _get_variable_name_mapping(self, cfg_lev):
        variables = cfg_lev["variables"]

        mapping_dict = {}
        for var_ext, var_dict in variables.items():
            try:
                var_int = var_dict["internal_varname"]
            except KeyError:
                logging.error("Internal varname is not defined for var {var_ext}.")
            mapping_dict[var_ext] = var_int
        return mapping_dict

    def read(self, L1_file):
        """Read level 1 file"""
        ureg = units  # pint.UnitRegistry()
        ureg.force_ndarray_like = True
        # ureg.define("fraction = [] = frac")
        # ureg.define("percent = 1e-2 frac = pct")
        # ureg.define("1 = pct")
        # ureg.define("degrees_east = degree")
        # ureg.define("degrees_north = degree")
        pint_xarray.unit_Registry = ureg

        ds = xr.open_dataset(L1_file)

        if ds.rh.attrs["units"] == "1":
            ds.rh.attrs["units"] = "dimensionless"

        ds = ds.pint.quantify(
            unit_registry=ureg
        )  # Apply units to dataset (requires currently pip install git+https://github.com/xarray-contrib/pint-xarray@7518c844a034361c1f8921d74bc5f9a96fec1910 --ignore-requires-python

        rh.rename_variables(ds, self.variable_name_mapping_input)

        # Write to class
        sounding = snd.Sounding()
        sounding.profile = ds
        sounding.unitregistry = ureg
        sounding.source = L1_file

        return sounding
