"""Sounding class
"""
import copy
import logging
import os
from pathlib import Path

import _dataset_creator as dc
import _helpers as h
import numpy as np
import pint
import pint_pandas
import pint_xarray
import thermodynamics as td
import xarray as xr
from omegaconf import OmegaConf


class SondeTypeNotImplemented(Exception):
    pass


class Sounding:
    """Sounding class with processing functions"""

    def __init__(self, profile=None, meta_data={}, config=None, ureg=None):
        self.profile = profile
        self.meta_data = meta_data
        self.config = config
        self.unitregistry = ureg

    def split_by_direction(self, method="maxHeight"):
        """Split sounding into ascending and descending branch"""
        # Simple approach
        sounding_ascent = Sounding(
            self.profile.loc[self.profile.Dropping == 0],
            copy.deepcopy(self.meta_data),
            ureg=copy.deepcopy(self.unitregistry),
        )
        sounding_descent = Sounding(
            self.profile.loc[self.profile.Dropping == 1],
            copy.deepcopy(self.meta_data),
            ureg=copy.deepcopy(self.unitregistry),
        )

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
            if type(self.profile[var].dtype) == pint_pandas.pint_array.PintType:
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
        time_differences_counts = np.bincount(time_differences.astype(np.int))
        most_common_diff = np.argmax(time_differences_counts)
        temporal_resolution = most_common_diff
        self.meta_data["temporal_resolution"] = temporal_resolution

    def generate_sounding_id(self, config):
        """Generate unique id of sounding"""
        id = config.level1.variables.sounding.format.format(
            direction=self.meta_data["sounding_direction"],
            lat=self.profile.latitude.values[0],
            lon=self.profile.longitude.values[0],
            time=self.meta_data["launch_time_dt"].strftime("%Y%m%d%H%M"),
        )
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
        # Sounding ID
        self.generate_sounding_id(config)
        self.get_sonde_type()

    def create_dataset(self, config):
        # Create new dataset
        runtime_cfg = OmegaConf.create(
            {"runtime": {"sounding_dim": 1, "level_dim": len(self.profile.flight_time)}}
        )

        meta_data_cfg = OmegaConf.create(
            {"meta_level0": h.remove_nontype_keys(self.meta_data, type("str"))}
        )
        merged_conf = OmegaConf.merge(config.level1, meta_data_cfg, runtime_cfg)
        merged_conf._set_parent(OmegaConf.merge(config, meta_data_cfg, runtime_cfg))
        ds = dc.create_dataset(merged_conf)

        ds.flight_time.data = xr.DataArray(
            [self.profile.flight_time], dims=["sounding", "level"]
        )

        # Fill dataset with data
        unset_vars = {}
        for k in ds.data_vars.keys():
            try:
                isquantity = (
                    self.profile[config.level1.variables[k].internal_varname].pint.units
                    is not None
                )

                dims = ds[k].dims
                if "sounding" == dims[0]:
                    if isquantity:  # convert values to output unit
                        ds[k].data = [
                            self.profile[config.level1.variables[k].internal_varname]
                            .pint.to(ds[k].attrs["units"])
                            .pint.magnitude
                        ]
                    else:
                        ds[k].data = [
                            self.profile[
                                config.level1.variables[k].internal_varname
                            ].values
                        ]
                elif "sounding" == dims[1]:
                    if isquantity:  # convert values to output unit
                        ds[k].data = np.array(
                            [
                                self.profile[
                                    config.level1.variables[k].internal_varname
                                ]
                                .pint.to(ds[k].attrs["units"])
                                .pint.magnitude
                            ]
                        ).T
                    else:
                        ds[k].data = np.array(
                            [
                                self.profile[
                                    config.level1.variables[k].internal_varname
                                ].values
                            ]
                        ).T
                else:
                    if isquantity:
                        ds[k].data = (
                            self.profile[config.level1.variables[k].internal_varname]
                            .pint.to(ds[k].attrs["units"])
                            .pint.magnitude
                        )
                    else:
                        ds[k].data = self.profile[
                            config.level1.variables[k].internal_varname
                        ].values
            except KeyError:
                unset_vars[k] = config.level1.variables[k].internal_varname

        for var_out, var_int in unset_vars.items():
            if var_int == "launch_time":
                ds[var_out].data = [self.meta_data["launch_time_dt"]]
            elif var_int == "sounding":
                lat = self.profile["latitude"][0].values
                lon = self.profile["longitude"][0].values
                direction = self.meta_data["sounding_direction"]
                time = self.meta_data["launch_time_dt"]
                id_fmt = config.level1.variables[var_int].format
                id = id_fmt.format(lat=lat, lon=lon, direction=direction)
                id = time.strftime(id)
                ds[var_out].data = [id]
        self.dataset = ds

    def export(self, output_fmt, cfg):
        """
        Saves sounding to disk
        """
        output = output_fmt.format(
            platform=cfg.main.platform,
            campaign=cfg.main.campaign,
            campaign_id=cfg.main.campaign_id,
            direction=self.meta_data["sounding_direction"],
        )
        output = self.meta_data["launch_time_dt"].strftime(output)
        directory = os.path.dirname(output)
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.dataset.to_netcdf(output)
        logging.info(f"Sounding written to {output}")
