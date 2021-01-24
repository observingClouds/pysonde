"""Sounding class
"""
import logging
import os
import sys

import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(__file__))
import thermodynamics as td  # noqa: E402


class SondeTypeNotImplemented(Exception):
    pass


class Sounding:
    """Sounding class with processing functions"""

    def __init__(self, profile=None, meta_data={}, config=None):
        self.profile = profile
        self.meta_data = meta_data
        self.config = config

    def split_by_direction(self, method="maxHeight"):
        """Split sounding into ascending and descending branch"""
        # Simple approach
        sounding_ascent = Sounding(
            self.profile.loc[self.profile.Dropping == 0], self.meta_data
        )
        sounding_descent = Sounding(
            self.profile.loc[self.profile.Dropping == 1], self.meta_data
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

    def convert_sounding_pd2xr(self):
        self.profile = xr.Dataset.from_dataframe(self.profile)

    def calc_ascent_rate(self):
        """
        Calculate ascent rate

        negative if sonde is falling
        """
        time_delta = np.diff(self.profile.flight_time) / np.timedelta64(1, "s")
        height_delta = np.diff(self.profile.height)
        ascent_rate = height_delta / time_delta
        # ascent_rate = np.diff(self.profile.height) / (
        #    np.diff(self.profile.flight_time.astype(np.float) / 1e9)
        # )
        ascent_rate_ = np.concatenate(([0], ascent_rate))  # 0 at first measurement
        # self.profile["ascent_rate"] = ascent_rate_
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
        id = config.level1.variables.sounding_id.format.format(
            direction=self.meta_data["sounding_direction"],
            lat=self.profile.latitude.values[0],
            lon=self.profile.longitude.values[0],
            time=self.meta_data["launch_time_dt"].strftime("%Y%m%d%H%M"),
        )
        self.meta_data["sounding_id"] = id

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
        mixing_ratio = (
            td.calc_wv_mixing_ratio(self.profile, e_s)
            * self.profile.humidity.values
            / 100.0
        )
        # self.profile["mixing_ratio"] = mixing_ratio
        self.profile.insert(10, "mixing_ratio", mixing_ratio)
        # Launch time as type(datetime)
        # flight_time_unix = self.profile.flight_time.values.astype(np.float) / 1e9
        # launch_time_unix = flight_time_unix[0]
        # self.meta_data["launch_time_dt"] = num2date(
        #     launch_time_unix, "seconds since 1970-01-01"
        # )
        self.meta_data["launch_time_dt"] = self.profile.flight_time.iloc[0]
        # Resolution
        self.calc_temporal_resolution()
        # Sounding ID
        self.generate_sounding_id(config)
        self.get_sonde_type()
