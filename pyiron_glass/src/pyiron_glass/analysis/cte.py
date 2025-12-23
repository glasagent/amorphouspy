"""Module for extracting the coefficient of thermal expansion (CTE) from simulation data.

This module provides functions to calculate the CTE using data from molecular dynamics simulations,
specifically from constant pressure (NPT) simulations. The CTE can be computed based on
fluctuations in enthalpy and volume, or from volume-temperature data obtained from multiple
NPT simulations at different temperatures.

Author
------
Marcel Sadowski (github.com/Gitdowski)
"""

import matplotlib.pyplot as plt
import numpy as np


def CTE_from_NPT_fluctuations(
    T: float | list | np.ndarray,
    H: list | np.ndarray,
    V: list | np.ndarray,
    N: int = 1000,
    *,
    running_mean: bool = False,
) -> float:
    """Compute the CTE from enthalpy-volume cross-correlations.

    Data needs to be obtained from a constant pressure, constant temperature NPT simulation. Formula used
    can be found in See Tildesley, Computer Simulation of Liquids, Eq. 2.94.

    Parameters
    ----------
    T : int | float | list | np.ndarray
        Target temperature (defining the ensemble) in K. Can be specified as a single value (int/float).
        If provided as list or array-like, the mean will be used as target temperature.
    H : list | np.ndarray
        "Instantaneous enthalpy" in eV, list or np.ndarray. Not that the instantaneous enthalpy is given by
        H = U + pV, where U is the instantaneous internal energy (i.e., kinetic + potential of every timestep),
        p the pressure defining the ensemble and V the average volume. Note that this is not the same quantity
        as the enthalpy obtained from LAMMPS directly via the 'enthalpy' output from thermo_modify. The latter
        uses the instantaneous pressure at the given timestep instead of the average (or ensemble-defining) pressure.
    V : list | np.ndarray
        Instantaneous volume in Ang^3, list or np.ndarray. Here, one can also feed individual cell lengths for
        anisotropic CTE calculations.
    N: int
        Window size for running mean calculation if running_mean is True.
    running_mean: bool
        Conventionally, fluctuations are calculated as difference from the mean of the whole trajectory. If
        running_mean is True, running mean values are used to determine fluctuations. This can be useful for
        non-stationary data where drift in volume and energy is still observed.

    Returns
    -------
    float
        The calculated CTE value in 1/K.

    """

    def _running_mean(data: list | np.ndarray, N: int) -> np.ndarray:
        """Calculate running mean of an array-like dataset.

        The initial and final values of the returned array are NaN, as the running mean is not defined
        for those points.

        Parameters
        ----------
        data : list | np.ndarray
            Input data for which the running mean should be calculated.
        N : int
            Width of the averaging window

        Returns
        -------
        np.ndarray
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
    T_target = np.mean(T)

    if not running_mean:
        H_fluctuations = np.array(H) - np.mean(H)
        V_fluctuations = np.array(V) - np.mean(V)
    elif running_mean:
        H_fluctuations = np.array(H) - _running_mean(H, N)
        V_fluctuations = np.array(V) - _running_mean(V, N)
        # Remove NaN values at beginning and end that resulted from running_mean
        H_fluctuations = H_fluctuations[~np.isnan(H_fluctuations)]
        V_fluctuations = V_fluctuations[~np.isnan(V_fluctuations)]

    CTE = (np.mean(H_fluctuations * V_fluctuations)) / (np.mean(V) * kB * T_target**2)
    return float(CTE)


def CTE_from_V_T_data(
    T: list | np.ndarray,
    V: list | np.ndarray,
    *,
    show_plot: bool = True,
) -> float:
    """Compute the CTE from slope of volume-temperature.

    This can be done from by performing various constant pressure, constant temperature NPT simulation
    at different temperatures. Afterwards, collect the averaged volume and fit linearly to the temperature.
    Divide slope by the volume belonging to the lowest temperature to obtain the CTE.
    If specified, the volume-temperature plot is shown and the fitted line is overlaid.

    Parameters
    ----------
    T : list | np.ndarray
        Target or averaged temperatures in K of different NPT simulations. One entry for every simulation.
    V : list | np.ndarray
        Averaged volumes in Ang^3 of different NPT simulations. One entry for every simulation.
    show_plot : bool, optional
        Whether to show the volume-temperature plot with fitted line, by default True.

    Returns
    -------
    float
        The calculated CTE value in 1/K.

    Note
    ----
    - This function is currently not in use in the workflows. But it might be used for cross-checking.

    """
    # make sure to order lists by increasing temperature
    sorted_indices = np.argsort(T)
    T = np.array(T)[sorted_indices]
    V = np.array(V)[sorted_indices]

    # fit and calculate CTE
    slope, intercept = np.polyfit(T, V, 1)
    CTE = slope / V[0]

    # plotting
    if show_plot:
        plt.close("all")
        plt.figure()
        plt.plot(T, V, label="Data", color="blue")
        plt.plot(T, slope * T + intercept, color="red", label="Fitted", linestyle="--")
        plt.xlabel("Temperature / K")
        plt.ylabel(r"Volume / $\AA^3$")
        plt.legend()
        plt.show()

    return float(CTE)
