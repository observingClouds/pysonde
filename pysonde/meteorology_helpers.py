"""
Functions for post-processing data
"""

import metpy.calc
import metpy.calc as mpcalc
import metpy.interpolate as mpinterp
import numpy as np
from metpy.units import units


def calc_saturation_pressure(temperature_K, method="hardy1998"):
    """
    Calculate saturation water vapor pressure

    Input
    -----
    temperature_K : array
        array of temperature in Kevlin or dew point temperature for actual vapor pressure
    method : str
        Formula used for calculating the saturation pressure
            'hardy1998' : ITS-90 Formulations for Vapor Pressure, Frostpoint Temperature,
                Dewpoint Temperature, and Enhancement Factors in the Range –100 to +100 C,
                Bob Hardy, Proceedings of the Third International Symposium on Humidity and Moisture,
                1998 (same as used in Aspen software after May 2018)

    Return
    ------
    e_sw : array
        saturation pressure (Pa)

    Examples
    --------
    >>> calc_saturation_pressure([273.15])
    array([ 611.2129107])

    >>> calc_saturation_pressure([273.15, 293.15, 253.15])
    array([  611.2129107 ,  2339.26239586,   125.58350529])
    """

    if method == "hardy1998":
        g = np.empty(8)
        g[0] = -2.8365744 * 10**3
        g[1] = -6.028076559 * 10**3
        g[2] = 1.954263612 * 10**1
        g[3] = -2.737830188 * 10 ** (-2)
        g[4] = 1.6261698 * 10 ** (-5)
        g[5] = 7.0229056 * 10 ** (-10)
        g[6] = -1.8680009 * 10 ** (-13)
        g[7] = 2.7150305

        e_sw = np.zeros_like(temperature_K)

        for t, temp in enumerate(temperature_K):
            ln_e_sw = np.sum([g[i] * temp ** (i - 2) for i in range(0, 7)]) + g[
                7
            ] * np.log(temp)
            e_sw[t] = np.exp(ln_e_sw)
        return e_sw


def calc_mixing_ratio_hardy(dew_point_K, pressure_Pa):
    e_s = calc_saturation_pressure(dew_point_K)
    mixing_ratio = metpy.calc.mixing_ratio(
        e_s * units.pascal, pressure_Pa * units.pascal
    )
    return mixing_ratio


def calc_q_from_rh(dp, p):
    """
    Input :
        dp : dew point (deg Celsius)
        p : pressure (hPa)
    Output :
        q : Specific humidity values

    Function to estimate specific humidity from the relative humidity,
    temperature and pressure in the given dataset. This function uses MetPy's
    functions to get q:
    (i) mpcalc.dewpoint_from_relative_humidity()
    (ii) mpcalc.specific_humidity_from_dewpoint()

    """
    q = mpcalc.specific_humidity_from_dewpoint(dp * units.degC, p * units.hPa).magnitude

    return q


def calc_theta_from_T(T, p):
    """
    Input :
        T : temperature (deg Celsius)
        p : pressure (hPa)
    Output :
        theta : Potential temperature values
    Function to estimate potential temperature from the
    temperature and pressure in the given dataset. This function uses MetPy's
    functions to get theta:
    (i) mpcalc.potential_temperature()

    """
    theta = mpcalc.potential_temperature(p * units.hPa, T * units.degC).magnitude

    return theta


def calc_T_from_theta(theta, p):
    """
    Input :
        theta : potential temperature (K)
        p : pressure (hPa)
    Output :
        T : Temperature values
    Function to estimate temperature from potential temperature and pressure,
    in the given dataset. This function uses MetPy's
    functions to get T:
    (i) mpcalc.temperature_from_potential_temperature()

    """
    T = (
        mpcalc.temperature_from_potential_temperature(
            p * units.hPa,
            theta * units.kelvin,
        ).magnitude
        - 273.15
    )

    return T


def calc_rh_from_q(q, T, p):
    """
    Input :
        q : specific humidity
        T : Temperature values estimated from interpolated potential_temperature;
        p : pressure (hPa)
    Output :
        rh : Relative humidity values
    Function to estimate relative humidity from specific humidity, temperature
    and pressure in the given dataset. This function uses MetPy's
    functions to get rh:
    (i) mpcalc.relative_humidity_from_specific_humidity()

    """
    rh = (
        mpcalc.relative_humidity_from_specific_humidity(
            q,
            T * units.degC,
            p * units.hPa,
        ).magnitude
        * 100
    )

    return rh


def get_wind_components(dir, spd):
    """
    Convert directional windspeed and winddirection
    to wind components
    Input
    -----
    dir : array-like
        wind from direction (deg)
    spd : array-like
        wind speed along dir
    Return
    ------
    u : array-like
       eastward wind component
       (positive when directed eastward)
    v : array-like
       northward wind component
       (positive when directed northward)
    """
    dir_rad = np.deg2rad(dir)
    u = -1 * spd * np.sin(dir_rad)
    v = -1 * spd * np.cos(dir_rad)
    return u, v


def get_directional_wind(u, v):
    """
    Convert wind-components to directional
    wind
    Input
    -----
    u : array-like
        eastward wind component
        (positive when directed eastward)
    v : array-like
        northward wind component
        (positive when directed northward)
    Return
    ------
    dir : array-like
        wind from direction (deg)
    spd : array-like
        wind speed along dir
    """
    # windu = wind_u.pint.dequantify().values
    # windv = wind_v.pint.dequantify().values

    # dir = np.rad2deg(np.arctan2(-1*u, -1*v)) % 360
    dir = np.rad2deg(np.arctan2(-1 * u, -1 * v))
    dir = np.mod(dir, 360)
    spd = np.sqrt(u**2 + v**2)

    return dir, spd


def set_global_attributes(ds, global_attrs_dict):
    """
    Set global attributes
    """
    for attribute, value in global_attrs_dict.items():
        ds.attrs[attribute] = value
    return ds


def compress_dataset(ds):
    """
    Apply internal netCDF4 compression
    """
    for var in ds.data_vars:
        ds[var].encoding["zlib"] = True
    return ds


def set_additional_var_attributes(ds, meta_data_dict):
    """
    Set further descriptive variable
    attributes and encoding.
    """
    for var in ds.variables:
        try:
            meta_data_var = meta_data_dict[var]
        except KeyError:
            continue
        for key, value in meta_data_var.items():
            if key not in ["_FillValue", "dtype"] and not ("time" in var):
                ds[var].attrs[key] = value
            elif (key not in ["_FillValue", "dtype", "units"]) and ("time" in var):
                ds[var].attrs[key] = value
            elif (key == "_FillValue") and (value is False):
                ds[var].attrs[key] = value
            else:
                ds[var].encoding[key] = value

    return ds


def write_dataset(ds, filename):
    ds = compress_dataset(ds)
    ds.to_netcdf(filename, unlimited_dims=["sounding"])


def get_direction(ds_interp, ds):
    # First source of direction
    direction_msg = None
    try:
        if "309057" in ds.source or "309052" in ds.source:
            direction_msg = "ascending"
        elif "309056" in ds.source or "309053" in ds.source:
            direction_msg = "descending"
    except AttributeError:
        print("No attribute source found")

    # Second source of direction
    median_ascent = np.nanmedian(ds.ascentRate.values)
    if median_ascent > 0:
        direction_data = "ascending"
    elif median_ascent < 0:
        direction_data = "descending"
    else:
        print("Direction not retrievable")

    if (direction_msg == direction_data) or (direction_msg is None):
        return direction_data
    else:
        print("Direction mismatch")


def calc_ascentrate(height, time):
    """
    Calculate the ascent rate
    Input
    -----
    sounding : obj
        sounding class containing gpm
        and flight time
    Return
    ------
    sounding : obj
        sounding including the ascent rate
    """
    ascent_rate = np.diff(height) / (np.diff(time))
    ascent_rate = np.ma.concatenate(([0], ascent_rate))  # 0 at first measurement
    return ascent_rate


def pressure_interpolation(
    pressures, altitudes, output_altitudes, convergence_error=0.05
):
    """
    Interpolates pressure on altitude grid
    The pressure is interpolated logarithmically.
    Input
    -----
    pressure : array
        pressures in hPa
    altitudes : array
        altitudes in m belonging to pressure values
    output_altitudes : array
        altitudes (m) on which the pressure should
        be interpolated to
    convergence_error : float
        Error that needs to be reached to stop
        convergence iteration
    Returns
    -------
    pressure_interpolated : array
        array of interpolated pressure values
        on altitudes
    """
    new_alt = output_altitudes
    pressure_interpolated = np.empty(len(output_altitudes))
    pressure_interpolated[:] = np.nan

    # Exclude heights outside of the intersection of measurements heights
    # and output_altitudes
    altitudes_above_measurements = new_alt > max(altitudes)
    range_of_alt_max = (
        np.min(np.where(altitudes_above_measurements | (new_alt == new_alt[-1]))) - 1
    )

    altitudes_below_measurements = new_alt < min(altitudes)
    range_of_alt_min = (
        np.max(np.where(altitudes_below_measurements | (new_alt == new_alt[0]))) + 1
    )

    for i in range(range_of_alt_min, range_of_alt_max):
        target_h = new_alt[i]

        lower_idx = np.nanmax(np.where(altitudes <= target_h))
        upper_idx = np.nanmin(np.where(altitudes >= target_h))

        p1 = np.float64(pressures[lower_idx])  # pressure at lower altitude
        p2 = np.float64(pressures[upper_idx])  # pressure at higher altitude
        a1 = altitudes[lower_idx]  # lower altitude
        a2 = altitudes[upper_idx]  # higher altitude

        if a1 == a2:  # Target height already reached
            pressure_interpolated[i] = p1
            continue

        xp = np.array([p1, p2])
        arr = np.array([a1, a2])

        err = 10

        if a2 - a1 < 100:
            while err > convergence_error:
                x = np.mean([p1, p2])
                ah = mpinterp.log_interpolate_1d(x, xp, arr, fill_value=np.nan)
                if ah > target_h:
                    p2 = x
                else:
                    p1 = x
                err = abs(ah - target_h)
            pressure_interpolated[i] = x

    return pressure_interpolated
