"""Sounding class
"""
import copy
import logging
import os
from pathlib import Path

import _dataset_creator as dc
import _helpers as h
import numpy as np
import pandas as pd
import pint_pandas
import pint_xarray
import thermodynamics as td
import xarray as xr
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

logging.debug(f"Pint_xarray version:{pint_xarray.__version__}")


class SondeTypeNotImplemented(Exception):
    pass


class Sounding:
    """Sounding class with processing functions"""

    def __init__(self, profile=None, meta_data={}, config=None, ureg=None):
        if profile is None:
            self.profile = None
        else:
            self.profile = profile.copy(deep=True)
        self.meta_data = meta_data
        self.config = config
        self.unitregistry = ureg

    def split_by_direction(self, method="maxHeight"):
        """Split sounding into ascending and descending branch"""
        # Simple approach
        sounding_ascent = copy.deepcopy(self)
        sounding_descent = copy.deepcopy(self)
        sounding_ascent.profile = self.profile.loc[self.profile.Dropping == 0]
        sounding_descent.profile = self.profile.loc[self.profile.Dropping == 1]

        # Bugfix 17
        if method == "maxHeight":
            for s, (sounding, func) in enumerate(
                zip(
                    (sounding_ascent.profile, sounding_descent.profile),
                    (np.greater_equal, np.less_equal),
                )
            ):
                if len(sounding) < 2:
                    continue
                window_size = 5
                smoothed_heights = np.convolve(
                    sounding.height, np.ones((window_size,)) / window_size, mode="valid"
                )
                if not np.all(func(np.gradient(smoothed_heights), 0)):
                    total = len(sounding.height)
                    nb_diff = total - np.sum(func(np.gradient(sounding.height), 0))
                    logging.warning(
                        "Of {} observations, {} observations have an inconsistent "
                        "sounding direction".format(total, nb_diff)
                    )
                    # Find split time for ascending and descending sounding by maximum height
                    # instead of relying on Dropping variable
                    logging.warning(
                        "Calculate bursting of balloon from maximum geopotential height"
                    )
                    idx_max_hgt = np.argmax(self.profile.height)

                    sounding_ascent.profile = self.profile.iloc[0 : idx_max_hgt + 1]
                    sounding_descent.profile = self.profile.iloc[idx_max_hgt + 1 :]
        sounding_ascent.meta_data["sounding_direction"] = "ascent"
        sounding_descent.meta_data["sounding_direction"] = "descent"

        return sounding_ascent, sounding_descent

    def convert_sounding_df2ds(self):
        unit_dict = {}
        for var in self.profile.columns:
            if type(self.profile[var].dtype) is pint_pandas.pint_array.PintType:
                unit_dict[var] = self.profile[var].pint.units
                self.profile[var] = self.profile[var].pint.magnitude

        self.profile = xr.Dataset.from_dataframe(self.profile)

        if self.unitregistry is not None:
            self.unitregistry.force_ndarray_like = True

        for var, unit in unit_dict.items():
            self.profile[var].attrs["units"] = unit.__str__()
        self.profile = self.profile.pint.quantify(unit_registry=self.unitregistry)

    def calc_ascent_rate(self):
        """
        Calculate ascent rate

        negative if sonde is falling
        """
        time_delta = np.diff(self.profile.flight_time) / np.timedelta64(1, "s")
        height_delta = np.diff(self.profile.height)
        ascent_rate = height_delta / time_delta
        ascent_rate_ = np.concatenate(([0], ascent_rate))  # 0 at first measurement
        self.profile.insert(10, "ascent_rate", ascent_rate_)

    def calc_temporal_resolution(self):
        """
        Calculate temporal resolution of sounding

        Returns the most common temporal resolution
        by calculating the temporal differences
        and returning the most common difference.

        Input
        -----
        sounding : obj
            sounding class containing flight time
            information

        Return
        ------
        temporal_resolution : float
            temporal resolution
        """
        time_differences = np.abs(
            np.diff(np.ma.compressed(self.profile.flight_time))
        ) / np.timedelta64(1, "s")
        time_differences_counts = np.bincount(time_differences.astype(int))
        most_common_diff = np.argmax(time_differences_counts)
        temporal_resolution = most_common_diff
        self.meta_data["temporal_resolution"] = temporal_resolution

    def generate_location_coord(self):
        """Generate unique id of sounding"""
        lat = self.profile.latitude.values[0]
        if lat > 0:
            lat = "{:04.1f}".format(lat)
        else:
            lat = "{:05.1f}".format(lat)

        lon = self.profile.longitude.values[0]
        if lon > 0:
            lon = "{:04.1f}".format(lon)
        else:
            lon = "{:05.1f}".format(lon)

        loc = str(lat) + "N" + str(lon) + "E"
        self.meta_data["location_coord"] = loc

    def generate_sounding_id(self, config):
        """Generate unique id of sounding"""
        id = config.level1.variables.sounding.format.format(
            direction=self.meta_data["sounding_direction"],
            lat=self.profile.latitude.values[0],
            lon=self.profile.longitude.values[0],
        )
        id = self.meta_data["launch_time_dt"].strftime(id)
        self.meta_data["sounding"] = id

    def get_sonde_type(self):
        """Get WMO sonde type"""
        if self.meta_data["SondeTypeName"] == "RS41-SGP":
            self.meta_data["sonde_type"] = "123"
        else:
            raise SondeTypeNotImplemented(
                "SondeTypeName {} is not implemented".format(
                    self.meta_data["SondeTypeName"]
                )
            )

    def calculate_additional_variables(self, config):
        """Calculation of additional variables"""
        # Ascent rate
        self.calc_ascent_rate()
        # Dew point temperature
        dewpoint = td.convert_rh_to_dewpoint(
            self.profile.temperature.values, self.profile.humidity.values
        )
        self.profile.insert(10, "dew_point", dewpoint)
        # Mixing ratio
        e_s = td.calc_saturation_pressure(self.profile.temperature.values)
        if "pint" in e_s.dtype.__str__():
            mixing_ratio = (
                td.calc_wv_mixing_ratio(self.profile, e_s)
                * self.profile.humidity.values
            )
        else:
            mixing_ratio = (
                td.calc_wv_mixing_ratio(self.profile, e_s)
                * self.profile.humidity.values
                / 100.0
            )
        self.profile.insert(10, "mixing_ratio", mixing_ratio)
        self.meta_data["launch_time_dt"] = self.profile.flight_time.iloc[0]
        # Resolution
        self.calc_temporal_resolution()
        # Location
        self.generate_location_coord()
        # Sounding ID
        self.generate_sounding_id(config)
        self.get_sonde_type()

    def collect_config(self, config, level):
        level_dims = {1: "flight_time", 2: "altitude"}
        runtime_cfg = OmegaConf.create(
            {
                "runtime": {
                    "sounding_dim": 1,
                    "level_dim": len(self.profile[level_dims[level]]),
                }
            }
        )
        meta_data_cfg = OmegaConf.create(
            {"meta_level0": h.remove_nontype_keys(self.meta_data, type("str"))}
        )

        merged_conf = OmegaConf.merge(
            config[f"level{level}"], meta_data_cfg, runtime_cfg
        )
        merged_conf._set_parent(OmegaConf.merge(config, meta_data_cfg, runtime_cfg))
        return merged_conf

    def isquantity(self, ds):
        return ds.pint.units is not None

    def set_unset_items(self, ds, unset_vars, config, level):
        for var_out, var_int in unset_vars.items():
            if var_int == "launch_time":
                ds[var_out].data = [self.meta_data["launch_time_dt"]]
            elif var_int == "sounding":
                try:
                    ds[var_out].data = [self.meta_data["sounding"]]
                except ValueError:
                    ds = ds.assign_coords({var_out: [self.meta_data["sounding"]]})
            elif var_int == "platform":
                ds[var_out].data = [config.main.platform_number]
        return ds

    def set_coordinate_data(self, ds, coords, config):
        unset_coords = {}
        for k in ds.coords.keys():
            try:
                int_var = config.coordinates[k].internal_varname
            except ConfigAttributeError:
                logging.debug(f"{k} does not seem to have an internal varname")
                continue
            if int_var not in self.profile:
                logging.warning(f"No data for output variable {k} found in input.")
                unset_coords[k] = int_var
                pass
            elif self.isquantity(
                self.profile[int_var]
            ):  # convert values to output unit
                ds = ds.assign_coords(
                    {
                        k: self.profile[int_var]
                        .pint.to(ds[k].attrs["units"])
                        .pint.magnitude
                    }
                )
            else:
                ds = ds.assign_coords({k: self.profile[int_var].pint.dequantify()})
            coord_dtype = config.coordinates[k].get("encodings")
            if coord_dtype is not None:
                coord_dtype = coord_dtype.get("dtype")
            if coord_dtype is not None:
                ds[k].encoding["dtype"] = coord_dtype
        return ds, unset_coords

    def create_dataset(self, config, level=1):
        merged_conf = self.collect_config(config, level)
        ds = dc.create_dataset(merged_conf)

        ds.flight_time.data = xr.DataArray(
            [self.profile.flight_time], dims=["sounding", "level"]
        )

        # Fill dataset with data
        unset_vars = {}

        for var in self.profile.data_vars:
            if var == "alt_bnds":
                continue
            if "sounding" not in self.profile[var].dims:
                self.profile[var] = self.profile[var].expand_dims({"sounding": 1})
        for k in ds.data_vars.keys():
            try:
                int_var = config[f"level{level}"].variables[k].internal_varname
            except ConfigAttributeError:
                logging.debug(f"{k} does not seem to have an internal varname")
                continue
            if int_var not in self.profile.keys():
                unset_vars[k] = int_var
                continue
            dims = ds[k].dims
            if k == "launch_time":
                try:
                    ds[k].data = self.profile[int_var].values
                except KeyError:
                    unset_vars[k] = int_var
                    pass
                continue
            elif k == "platform":
                continue  # will be set at later stage
            if self.isquantity(self.profile[int_var]):
                data = (
                    self.profile[int_var].pint.to(ds[k].attrs["units"]).pint.magnitude
                )
                if len(dims) > 1 and "sounding" == dims[1]:
                    ds[k].data = np.array(data).T
                else:
                    ds[k].data = data
            else:
                if len(dims) > 1 and "sounding" == dims[1]:
                    ds[k].data = np.array(self.profile[int_var].values).T
                else:
                    ds[k].data = self.profile[int_var].values
        ds, unset_coords = self.set_coordinate_data(
            ds, ds.coords, config[f"level{level}"]
        )
        unset_items = {**unset_vars, **unset_coords}
        ds = self.set_unset_items(ds, unset_items, config, level)
        merged_conf = h.replace_placeholders_cfg(self, merged_conf)

        logging.debug("Add global attributes")
        if "global_attrs" in merged_conf.keys():
            _cfg = h.remove_missing_cfg(merged_conf["global_attrs"])
            ds.attrs = _cfg

        self.dataset = ds

    def get_direction(self):
        if self.profile.ascent_flag.values[0] == 0:
            direction = "ascent"
        elif self.profile.ascent_flag.values[0] == 1:
            direction = "descent"
        self.meta_data["sounding_direction"] = direction

    def set_launchtime(self):
        first_idx_w_time = np.argwhere(
            ~np.isnan(self.profile.squeeze().flight_time.values)
        )[0][0]
        self.meta_data["launch_time_dt"] = pd.to_datetime(
            self.profile.squeeze().flight_time.values[first_idx_w_time]
        )

    def export(self, output_fmt, cfg):
        """
        Saves sounding to disk
        """
        output = output_fmt.format(
            platform=cfg.main.get("platform"),
            campaign=cfg.main.get("campaign"),
            campaign_id=cfg.main.get("campaign_id"),
            direction=self.meta_data["sounding_direction"],
            version=cfg.main.get("data_version"),
        )
        output = self.meta_data["launch_time_dt"].strftime(output)
        directory = os.path.dirname(output)
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.dataset.encoding["unlimited_dims"] = ["sounding"]
        self.dataset.to_netcdf(output)
        logging.info(f"Sounding written to {output}")
