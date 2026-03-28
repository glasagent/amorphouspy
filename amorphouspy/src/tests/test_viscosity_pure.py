"""Tests for pure functions in amorphouspy.workflows.viscosity."""

import numpy as np
import pytest

from amorphouspy.workflows.viscosity import fit_vft, vft_model

# ---------------------------------------------------------------------------
# vft_model
# ---------------------------------------------------------------------------


def test_vft_model_scalar_input():
    """Returns a scalar for scalar temperature input."""
    result = vft_model(1000.0, log10_eta0=-3.0, B=2000.0, T0=300.0)
    assert isinstance(float(result), float)


def test_vft_model_array_input():
    """Returns an array of the same shape as T."""
    T = np.array([600.0, 800.0, 1000.0, 1200.0])
    result = vft_model(T, log10_eta0=-3.0, B=2000.0, T0=300.0)
    assert result.shape == T.shape


def test_vft_model_known_value():
    """At T = T0 + B*log10(e), result equals log10_eta0 + 1."""
    log10_eta0, B, T0 = -3.0, 2000.0, 300.0
    T = T0 + B * np.log10(np.e)
    result = vft_model(T, log10_eta0=log10_eta0, B=B, T0=T0)
    assert pytest.approx(result, rel=1e-6) == log10_eta0 + 1.0


def test_vft_model_increases_with_decreasing_temperature():
    """Viscosity increases as temperature decreases (above T0)."""
    T = np.linspace(800.0, 600.0, 10)
    result = vft_model(T, log10_eta0=-3.0, B=2000.0, T0=300.0)
    assert np.all(np.diff(result) > 0)


def test_vft_model_at_eta0_limit():
    """As T → ∞, log10(η) → log10_eta0."""
    log10_eta0 = -3.0
    result = vft_model(1e10, log10_eta0=log10_eta0, B=2000.0, T0=300.0)
    assert pytest.approx(float(result), abs=1e-4) == log10_eta0


# ---------------------------------------------------------------------------
# fit_vft
# ---------------------------------------------------------------------------


def test_fit_vft_recovers_parameters():
    """fit_vft recovers the true VFT parameters from noise-free data."""
    true_params = (-3.0, 2000.0, 300.0)
    T_data = np.linspace(600.0, 1500.0, 30)
    log10_eta_data = vft_model(T_data, *true_params)

    popt, _ = fit_vft(T_data, log10_eta_data)

    assert pytest.approx(popt[0], abs=0.01) == true_params[0]
    assert pytest.approx(popt[1], abs=1.0) == true_params[1]
    assert pytest.approx(popt[2], abs=1.0) == true_params[2]


def test_fit_vft_returns_tuple_of_two():
    """fit_vft returns (popt, pcov)."""
    T_data = np.linspace(600.0, 1500.0, 20)
    log10_eta_data = vft_model(T_data, -3.0, 2000.0, 300.0)
    result = fit_vft(T_data, log10_eta_data)
    assert len(result) == 2


def test_fit_vft_popt_has_three_params():
    """Popt contains three parameters: log10_eta0, B, T0."""
    T_data = np.linspace(600.0, 1500.0, 20)
    log10_eta_data = vft_model(T_data, -3.0, 2000.0, 300.0)
    popt, _ = fit_vft(T_data, log10_eta_data)
    assert len(popt) == 3


def test_fit_vft_custom_initial_guess():
    """fit_vft works with a custom initial guess."""
    T_data = np.linspace(600.0, 1500.0, 20)
    log10_eta_data = vft_model(T_data, -2.0, 1500.0, 200.0)
    popt, _ = fit_vft(T_data, log10_eta_data, initial_guess=(-2.0, 1500.0, 200.0))
    assert pytest.approx(popt[0], abs=0.01) == -2.0
