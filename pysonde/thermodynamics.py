"""
Thermodynamic functions
"""
import numpy as np
import pint
import pint_pandas as pp

ureg = pint.UnitRegistry()


def convert_rh_to_dewpoint(t_kelvin, rh):
    """
    Convert T and RH to dewpoint exactly
    following the formula used by the Vaisala
    M41 sounding system
    """
    if isinstance(t_kelvin, pp.pint_array.PintArray):
        t_kelvin = t_kelvin.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    if isinstance(rh, pp.pint_array.PintArray):
        rh = rh.quantity.to("dimensionless").magnitude
    assert np.any(t_kelvin > 100), "Temperature seems to be not given in Kelvin"
    kelvin = 15 * np.log(100 / rh) - 2 * (t_kelvin - 273.15) + 2711.5
    t_dew = t_kelvin * 2 * kelvin / (t_kelvin * np.log(100 / rh) + 2 * kelvin)
    if isinstance(t_kelvin, pp.pint_array.PintArray):
        t_dew = t_dew * ureg["K"]
    return t_dew


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
    if isinstance(temperature_K, pp.pint_array.PintArray):
        temperature_K = temperature_K.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    if method == "hardy1998":
        g = np.empty(8)
        g[0] = -2.8365744 * 10 ** 3
        g[1] = -6.028076559 * 10 ** 3
        g[2] = 1.954263612 * 10 ** 1
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
        if isinstance(temperature_K, pp.pint_array.PintArray):
            e_sw = pp.PintArray(e_sw, dtype="Pa")
        return e_sw


def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Calculate water vapor mixing ratio
    """
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        vapor_pressure = vapor_pressure.quantity.to("Pa").magnitude
    if "pint" in sounding.pressure.dtype.__str__():
        total_pressure = sounding.pressure.pint.quantity.to("hPa").magnitude
    else:
        total_pressure = sounding.pressure
    wv_mix_ratio = 1000.0 * (
        (0.622 * vapor_pressure) / (100.0 * total_pressure - vapor_pressure)
    )
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        return wv_mix_ratio * ureg("kg") / ureg("kg")
    else:
        return wv_mix_ratio
