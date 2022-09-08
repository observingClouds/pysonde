"""
Thermodynamic functions
"""
import metpy
import metpy.calc as mpcalc
import numpy as np
import pint_pandas as pp
import xarray as xr

# ureg = pint.UnitRegistry()
# ureg.define("fraction = [] = frac")
# ureg.define("percent = 1e-2 frac = pct")
# pp.PintType.ureg = ureg


def convert_rh_to_dewpoint(temperature, rh):
    """
    Convert T and RH to dewpoint exactly
    following the formula used by the Vaisala
    M41 sounding system

    Input
    -----
    temperature : array
        temperature in Kevlin or supported quantity
    """
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    else:
        temperature_K = temperature.data.magnitude
    if isinstance(rh, pp.pint_array.PintArray):
        rh = rh.quantity.to("percent").magnitude
    else:
        rh = rh.data.magnitude
    assert np.any(temperature_K > 100), "Temperature seems to be not given in Kelvin"
    kelvin = 15 * np.log(100 / rh) - 2 * (temperature_K - 273.15) + 2711.5
    t_dew = temperature_K * 2 * kelvin / (temperature_K * np.log(100 / rh) + 2 * kelvin)
    if isinstance(temperature, pp.pint_array.PintArray):
        t_dew = t_dew * ureg("K")
    return t_dew


def calc_saturation_pressure(temperature, method="hardy1998"):
    """
    Calculate saturation water vapor pressure

    Input
    -----
    temperature : array
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
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    elif isinstance(temperature, xr.core.dataarray.DataArray) and hasattr(
        temperature.data, "_units"
    ):
        temperature_K = temperature.pint.to("K").metpy.magnitude
    else:
        temperature_K = temperature
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
        if isinstance(temperature, pp.pint_array.PintArray):
            pp.PintType.ureg = ureg
            e_sw = pp.PintArray(e_sw, dtype="Pa")
        elif isinstance(temperature, xr.core.dataarray.DataArray) and hasattr(
            temperature.data, "_units"
        ):
            e_sw = xr.DataArray(
                e_sw, dims=temperature.dims, coords=temperature.coords
            ) * metpy.units.units("Pa")
        return e_sw


def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Calculate water vapor mixing ratio
    """
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        ureg = vapor_pressure.units._REGISTRY
        vapor_pressure_Pa = vapor_pressure.quantity.to("Pa").magnitude
    else:
        vapor_pressure_Pa = vapor_pressure
    if "pint" in sounding.pressure.dtype.__str__():
        total_pressure = sounding.pressure.values.quantity.to("hPa").magnitude
    else:
        total_pressure = sounding.pressure.values
    wv_mix_ratio = 1000.0 * (
        (0.622 * vapor_pressure_Pa) / (100.0 * total_pressure - vapor_pressure_Pa)
    )
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        return wv_mix_ratio * ureg("g") / ureg("kg")
    else:
        return wv_mix_ratio


def calc_theta_from_T(T, p):
    """
    Input :
        T : temperature
        p : pressure
    Output :
        theta : Potential temperature values
    Function to estimate potential temperature from the
    temperature and pressure in the given dataset. This function uses MetPy's
    functions to get theta:
    (i) mpcalc.potential_temperature()

    """
    theta = mpcalc.potential_temperature(p.metpy.quantify(), T.metpy.quantify())

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
    T = mpcalc.temperature_from_potential_temperature(
        p,
        theta,
    )

    return T
