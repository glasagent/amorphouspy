"""Module for extracting the coefficient of thermal expansion (CTE) from simulation data.

This module provides functions to calculate the CTE using data from molecular dynamics simulations,
specifically from constant pressure (NPT) simulations. The CTE can be computed based on
fluctuations in enthalpy and volume, or from volume-temperature data obtained from multiple
NPT simulations at different temperatures.

Author: Marcel Sadowski (github.com/Gitdowski)
"""

import numpy as np
from scipy.stats import linregress


def cte_from_npt_fluctuations(
    temperature: float | list | np.ndarray,
    enthalpy: list | np.ndarray,
    volume: list | np.ndarray,
    N_points: int = 1000,
    *,
    use_running_mean: bool = False,
) -> float:
    """Compute the CTE from enthalpy-volume cross-correlations.

    Data needs to be obtained from a constant pressure, constant temperature NPT simulation. Formula used
    can be found in See Tildesley, Computer Simulation of Liquids, Eq. 2.94.

    Args:
        temperature: Target temperature (defining the ensemble) in K. Can be specified as a single value (int/float).
            If provided as list or array-like, the mean will be used as target temperature.
        enthalpy: "Instantaneous enthalpy" in eV, list or np.ndarray. Not that the instantaneous enthalpy is given by
            H = U + pV, where U is the instantaneous internal energy (i.e., kinetic + potential of every timestep),
            p the pressure defining the ensemble and V the average volume. Note that this is not the same quantity
            as the enthalpy obtained from LAMMPS directly via the 'enthalpy' output from thermo_modify. The latter
            uses the instantaneous pressure at the given timestep instead of the average (or ensemble-defining)
            pressure.
        volume: Instantaneous volume in Ang^3, list or np.ndarray. Here, one can also feed individual cell lengths for
            anisotropic CTE calculations.
        N_points: Window size for running mean calculation if running_mean is True. Defaults to 1000.
        use_running_mean: Conventionally, fluctuations are calculated as difference from the mean of the whole
            trajectory. If use_running_mean is True, running mean values are used to determine fluctuations. This
            can be useful for non-stationary data where drift in volume and energy is still observed. Defaults to False.

    Returns:
        The calculated CTE value in 1/K.

    Example:
        >>> cte = cte_from_npt_fluctuations(
        ...     temperature=300.0,
        ...     enthalpy=enthalpy_array,
        ...     volume=volume_array
        ... )

    """

    def _running_mean(data: list | np.ndarray, N: int) -> np.ndarray:
        """Calculate running mean of an array-like dataset.

        The initial and final values of the returned array are NaN, as the running mean is not defined
        for those points.

        Args:
            data: Input data for which the running mean should be calculated.
            N: Width of the averaging window.

        Returns:
            Array of same size as input data containing the running mean values.

        """
        data = np.asarray(data)
        if N == 1:
            return data
        retArray = np.zeros(data.size) * np.nan
        padL = int(N / 2)
        padR = N - padL - 1
        retArray[padL:-padR] = np.convolve(data, np.ones((N,)) / N, mode="valid")
        return retArray

    kB = 8.617333262145e-5  # eV/K
    T_target = np.mean(temperature)

    if not use_running_mean:
        H_fluctuations = np.array(enthalpy) - np.mean(enthalpy)
        V_fluctuations = np.array(volume) - np.mean(volume)
    elif use_running_mean:
        H_fluctuations = np.array(enthalpy) - _running_mean(enthalpy, N_points)
        V_fluctuations = np.array(volume) - _running_mean(volume, N_points)
        # Remove NaN values at beginning and end that resulted from running_mean
        H_fluctuations = H_fluctuations[~np.isnan(H_fluctuations)]
        V_fluctuations = V_fluctuations[~np.isnan(V_fluctuations)]

    CTE = (np.mean(H_fluctuations * V_fluctuations)) / (np.mean(volume) * kB * T_target**2)
    return float(CTE)


def cte_from_volume_temperature_data(
    temperature: list | np.ndarray,
    volume: list | np.ndarray,
    reference_volume: float | None = None,
) -> float:
    """Compute the CTE from slope of a linear fit to volume-temperature data.

    This can be done from by performing various constant pressure, constant temperature NPT simulation
    at different temperatures. Afterwards, collect the averaged volume and fit linearly to the temperature.
    Divide slope by the volume belonging to the lowest temperature to obtain the CTE.
    If specified, the volume-temperature plot is shown and the fitted line is overlaid.

    Args:
        temperature: Temperature in K.
        volume: Volume in Angstrom^3 or other structural quantity of interest. E.g., for anisotropic CTE
            calculations, one can also use individual cell lengths in Angstroms here.
        reference_volume (optional): Reference of the volume (or structural quantity of interest) the CTE
            will be calculated from. If not otherwise specified, the volume corresponding to the lowest
            temperature is used as reference.

    Returns:
        The calculated CTE value in 1/K and the R^2 value of the linear fit.

    Note:
        - This function is currently not in use in the workflows. But it might be used for cross-checking.
        - If volume-temperature data is used, the CTE needs to be divided by 3 to obtain the linear CTE.

    Example:
        >>> cte = cte_from_volume_temperature_data(
        ...     temperature=[300, 400, 500],
        ...     volume=[100.0, 100.1, 100.2]
        ... )

    """
    # Set reference volume corresponding to the lowest temperature if not provided.
    if reference_volume is None:
        index_lowest_T = np.argmin(temperature)
        reference_volume = volume[index_lowest_T]

    # fit and calculate CTE
    slope, intercept, r_value, p_value, std_err = linregress(temperature, volume)
    CTE = slope / reference_volume
    R2 = r_value**2

    return float(CTE), float(R2)
