"""Tests for pure CTE calculation functions in amorphouspy.analysis.cte."""

import numpy as np
import pytest
from amorphouspy.analysis.cte import cte_from_npt_fluctuations, cte_from_volume_temperature_data

# ---------------------------------------------------------------------------
# cte_from_npt_fluctuations
# ---------------------------------------------------------------------------


def test_cte_fluctuations_constant_signals_give_zero():
    """Constant H and V have zero fluctuations, so CTE is zero."""
    n = 500
    H = np.ones(n) * 100.0
    V = np.ones(n) * 50.0
    cte = cte_from_npt_fluctuations(temperature=300.0, enthalpy=H, volume=V)
    assert pytest.approx(cte, abs=1e-20) == 0.0


def test_cte_fluctuations_returns_float():
    """Return value is a Python float."""
    rng = np.random.default_rng(0)
    n = 200
    cte = cte_from_npt_fluctuations(
        temperature=300.0,
        enthalpy=rng.normal(100.0, 1.0, n),
        volume=rng.normal(50.0, 0.5, n),
    )
    assert isinstance(cte, float)


def test_cte_fluctuations_positive_correlation_positive_cte():
    """Perfectly correlated H and V fluctuations yield a positive CTE."""
    rng = np.random.default_rng(1)
    n = 1000
    noise = rng.normal(0, 1.0, n)
    H = 100.0 + noise
    V = 50.0 + noise  # same fluctuation direction
    cte = cte_from_npt_fluctuations(temperature=300.0, enthalpy=H, volume=V)
    assert cte > 0.0


def test_cte_fluctuations_temperature_as_array():
    """Temperature provided as an array uses its mean."""
    rng = np.random.default_rng(2)
    n = 300
    T = np.full(n, 300.0)
    H = rng.normal(100.0, 1.0, n)
    V = rng.normal(50.0, 0.5, n)
    cte_scalar = cte_from_npt_fluctuations(temperature=300.0, enthalpy=H, volume=V)
    cte_array = cte_from_npt_fluctuations(temperature=T, enthalpy=H, volume=V)
    assert pytest.approx(cte_scalar) == cte_array


def test_cte_fluctuations_running_mean():
    """Running mean mode returns a float without errors."""
    rng = np.random.default_rng(3)
    n = 500
    H = rng.normal(100.0, 1.0, n)
    V = rng.normal(50.0, 0.5, n)
    cte = cte_from_npt_fluctuations(temperature=300.0, enthalpy=H, volume=V, N_points=50, use_running_mean=True)
    assert isinstance(cte, float)


def test_cte_fluctuations_raises_mismatched_lengths():
    """Raises ValueError if H and V have different lengths (scalar T)."""
    with pytest.raises(ValueError, match="same length"):
        cte_from_npt_fluctuations(temperature=300.0, enthalpy=np.ones(100), volume=np.ones(200))


def test_cte_fluctuations_raises_mismatched_array_temp():
    """Raises ValueError if T, H, V have different lengths (array T)."""
    with pytest.raises(ValueError, match="same length"):
        cte_from_npt_fluctuations(
            temperature=np.ones(100),
            enthalpy=np.ones(200),
            volume=np.ones(200),
        )


def test_cte_fluctuations_raises_n_points_too_large():
    """Raises ValueError if N_points >= len(enthalpy) when use_running_mean=True."""
    n = 50
    with pytest.raises(ValueError, match="smaller than"):
        cte_from_npt_fluctuations(
            temperature=300.0,
            enthalpy=np.ones(n),
            volume=np.ones(n),
            N_points=n,
            use_running_mean=True,
        )


def test_cte_fluctuations_raises_n_points_zero():
    """Raises ValueError if N_points < 1 when use_running_mean=True."""
    n = 100
    with pytest.raises(ValueError, match="positive integer"):
        cte_from_npt_fluctuations(
            temperature=300.0,
            enthalpy=np.ones(n),
            volume=np.ones(n),
            N_points=0,
            use_running_mean=True,
        )


# ---------------------------------------------------------------------------
# cte_from_volume_temperature_data
# ---------------------------------------------------------------------------


def test_cte_volume_temp_perfect_linear():
    """Perfect linear V-T data yields the exact CTE and R2=1."""
    T = np.array([300.0, 400.0, 500.0, 600.0])
    slope = 0.001  # Å³/K
    V0 = 10.0
    V = V0 + slope * T
    cte, r2 = cte_from_volume_temperature_data(T, V)
    assert pytest.approx(cte, rel=1e-6) == slope / V[0]
    assert pytest.approx(r2) == 1.0


def test_cte_volume_temp_returns_two_floats():
    """Returns a tuple of two floats."""
    T = np.array([300.0, 400.0, 500.0])
    V = np.array([10.0, 10.1, 10.2])
    result = cte_from_volume_temperature_data(T, V)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_cte_volume_temp_custom_reference_volume():
    """Using a custom reference volume changes the CTE value."""
    T = np.array([300.0, 400.0, 500.0])
    V = np.array([10.0, 10.1, 10.2])
    cte_default, _ = cte_from_volume_temperature_data(T, V)
    cte_custom, _ = cte_from_volume_temperature_data(T, V, reference_volume=20.0)
    assert pytest.approx(cte_custom) == cte_default / 2.0


def test_cte_volume_temp_r2_below_one_for_noisy_data():
    """Noisy data gives R² < 1."""
    rng = np.random.default_rng(42)
    T = np.linspace(300, 600, 20)
    V = 10.0 + 0.001 * T + rng.normal(0, 0.05, len(T))
    _, r2 = cte_from_volume_temperature_data(T, V)
    assert r2 < 1.0
