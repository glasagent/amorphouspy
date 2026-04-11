"""Module for extracting the coefficient of thermal expansion (CTE) from simulation data.

This module provides functions to calculate the CTE using data from molecular dynamics simulations,
specifically from constant pressure (NPT) simulations. The CTE can be computed based on
fluctuations in enthalpy and volume, or from volume-temperature data obtained from multiple
NPT simulations at different temperatures.

Author: Marcel Sadowski (github.com/Gitdowski)
"""

import numpy as np
from scipy.stats import linregress

from amorphouspy.shared import running_mean


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

    Notes:
        - If temperature is provided as int/float, it will be used as-is for the target temperature to compute the CTE.
        - If temperature is provided as list/array-like, the mean of the temperature data will be used as target
          temperature for the CTE calculation. See below how this is slightly changed if running mean is used.
        - If running_mean is set to True used, a part of the volume and enthalpy data at the beginning and end of the
          arrays cannot be used because it is required to calculate the running mean values. To ensure that the data for
          the calculation of the fluctuations is consistent with the mean volume and target temperature (in case this
          was provided as list/array-like), also this data will be trimmed correspondingly.
    """

    def _input_checks(
        enthalpy: list | np.ndarray,
        volume: list | np.ndarray,
        temperature: float | list | np.ndarray,
        N_points: int,
        *,
        use_running_mean: bool,
    ) -> tuple[int | None, int | None]:
        """Checks the input beyond simple type checks and determines the padding for the running mean if required.

        Args:
            enthalpy: See main function.
            volume: See main function.
            temperature: See main function.
            N_points: See main function.
            use_running_mean: See main function.

        Returns:
            padL: Number of points to be trimmed at the beginning of the enthalpy and volume arrays if use_running_mean
                is True. None otherwise. Will also be used for trimming the temperature if provided as list or
                array-like to ensure consistency with the mean volume and target temperature.
            padR: Number of points to be trimmed at the end of the enthalpy and volume arrays if use_running_mean is
                True. None otherwise. Will also be used for trimming the temperature if provided as list or array-like
                to ensure consistency with the mean volume and target temperature.
        """
        if isinstance(temperature, (int, float)):
            if len(enthalpy) != len(volume):
                msg = (
                    "If temperature is provided as int or float, enthalpy and volume arrays must have the same length."
                )
                raise ValueError(msg)
        elif len(temperature) != len(enthalpy) or len(enthalpy) != len(volume):
            msg = "If temperature is provided as list or array-like,"
            msg += " it must have the same length as the enthalpy and volume arrays."
            raise ValueError(msg)

        if use_running_mean:
            if N_points < 1:
                msg = "N_points must be a positive integer >= 1."
                raise ValueError(msg)
            if N_points >= len(enthalpy):
                msg = "N_points must be smaller than the length of the enthalpy and volume arrays."
                raise ValueError(msg)
            padL, padR = int(N_points / 2), N_points - int(N_points / 2) - 1
            return padL, padR
        return None, None

    # if use_running_mean = False, padL and padR will both be None
    padL, padR = _input_checks(enthalpy, volume, temperature, N_points, use_running_mean=use_running_mean)

    if use_running_mean:
        assert padR is not None
        assert padL is not None
        H_fluctuations = np.array(np.array(enthalpy) - running_mean(enthalpy, N_points))[padL:-padR]
        V_fluctuations = np.array(np.array(volume) - running_mean(volume, N_points))[padL:-padR]
    else:
        H_fluctuations = np.array(enthalpy) - np.mean(enthalpy)
        V_fluctuations = np.array(volume) - np.mean(volume)

    V_mean = np.mean(volume) if padL is None else np.mean(volume[padL:-padR])
    T_target = (
        temperature
        if isinstance(temperature, (int, float))
        else (np.mean(temperature) if padL is None else np.mean(temperature[padL:-padR]))
    )

    kB = 8.617333262145e-5  # eV/K
    CTE = (np.mean(H_fluctuations * V_fluctuations)) / (V_mean * kB * T_target**2)
    return float(CTE)


def cte_from_volume_temperature_data(
    temperature: list | np.ndarray,
    volume: list | np.ndarray,
    reference_volume: float | None = None,
) -> tuple[float, float]:
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
    slope, _intercept, r_value, _p_value, _std_err = linregress(temperature, volume)
    CTE = slope / reference_volume
    R2 = r_value**2

    return float(CTE), float(R2)
