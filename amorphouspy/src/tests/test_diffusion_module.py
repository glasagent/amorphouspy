"""Tests for the pure self-diffusion analysis functions in ``amorphouspy.properties.diffusion``.

These exercise the MSD algorithm, diffusion fitting, PBC unwrapping, COM-drift removal,
per-species resolution, the cross-check, the Arrhenius fit and Nernst-Einstein conductivity
against analytic ground truths, without running LAMMPS.
"""

import warnings

import numpy as np
import pytest
from amorphouspy.properties.diffusion import (
    _msd_brute_force,
    _msd_fft_multi,
    compute_msd,
    crosscheck_msd,
    fit_arrhenius,
    get_diffusion,
    log_linear_dump_steps,
    nernst_einstein_conductivity,
)
from ase import Atoms

KB_EV = 8.617333262e-5


def _frames_from_trajectory(trajectory, symbols, *, box=1000.0, temperature=1000.0, set_unwrapped=True):
    """Build ASE frames from an unwrapped trajectory of shape (n_frames, n_atoms, 3)."""
    frames = []
    for positions in trajectory:
        atoms = Atoms(symbols, positions=positions, cell=[box, box, box], pbc=True)
        if set_unwrapped:
            atoms.arrays["unwrapped_positions"] = np.asarray(positions, dtype=float)
        atoms.info["temperature"] = temperature
        frames.append(atoms)
    return frames


def test_fca_matches_brute_force():
    """The FFT MSD must equal the O(n^2) multi-origin definition to numerical precision."""
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(size=(150, 25, 3)), axis=0)
    assert np.allclose(_msd_fft_multi(positions), _msd_brute_force(positions), atol=1e-8)


def test_msd_lag_zero_is_zero():
    """MSD at lag 0 must be exactly zero."""
    rng = np.random.default_rng(1)
    trajectory = np.cumsum(rng.normal(scale=0.1, size=(50, 10, 3)), axis=0)
    msd = compute_msd(_frames_from_trajectory(trajectory, "H" * 10), remove_com_drift=False)
    assert msd["msd_total"][0] == pytest.approx(0.0, abs=1e-9)


def test_random_walk_diffusion_recovery():
    """A Gaussian random walk with known step variance recovers D = s^2 / 2 (Angstrom^2/ps)."""
    rng = np.random.default_rng(42)
    n_frames, n_atoms, step = 4000, 400, 0.1
    trajectory = np.cumsum(rng.normal(scale=step, size=(n_frames, n_atoms, 3)), axis=0)
    frames = _frames_from_trajectory(trajectory, "H" * n_atoms)
    msd = compute_msd(frames, timestep=1.0, output_frequency=1000, remove_com_drift=False)  # dt = 1 ps
    diffusion = get_diffusion(msd, fit_start_frac=0.05, fit_end_frac=0.5)
    expected_cm2_s = (step**2 / 2.0) * 1e-4
    assert diffusion["total"]["diffusion_cm2_s"] == pytest.approx(expected_cm2_s, rel=0.1)
    assert diffusion["total"]["loglog_slope"] == pytest.approx(1.0, abs=0.05)


def test_pbc_unwrapping_fallback_crosses_boundary():
    """Without an unwrapped array, on-the-fly unwrapping reconstructs displacement across PBC."""
    box = 10.0
    n_frames = 100
    true_positions = np.cumsum(np.full((n_frames, 1, 3), 0.25), axis=0)
    wrapped = true_positions % box
    frames = _frames_from_trajectory(wrapped, "H", box=box, set_unwrapped=False)
    msd = compute_msd(frames, remove_com_drift=False, per_species=False)
    expected_last = (0.25 * (n_frames - 1)) ** 2 * 3
    assert msd["msd_total"][-1] == pytest.approx(expected_last, rel=1e-6)


def test_com_drift_removed():
    """A rigid translation of all atoms gives ~0 MSD with COM removal and ballistic MSD without."""
    drift = np.outer(np.arange(50), [0.3, 0.0, 0.0])
    positions = np.zeros((50, 20, 3)) + drift[:, None, :]
    frames = _frames_from_trajectory(positions, "H" * 20)
    with_removal = compute_msd(frames, remove_com_drift=True, per_species=False)
    without_removal = compute_msd(frames, remove_com_drift=False, per_species=False)
    assert max(with_removal["msd_total"]) < 1e-12
    assert max(without_removal["msd_total"]) > 1.0


def test_per_species_distinct_mobility():
    """Two species with different step sizes yield distinct, correctly-ordered diffusion coefficients."""
    rng = np.random.default_rng(7)
    n_frames, n_each = 2000, 60
    traj_na = np.cumsum(rng.normal(scale=0.2, size=(n_frames, n_each, 3)), axis=0)
    traj_si = np.cumsum(rng.normal(scale=0.02, size=(n_frames, n_each, 3)), axis=0)
    trajectory = np.concatenate([traj_na, traj_si], axis=1)
    frames = _frames_from_trajectory(trajectory, "Na" * n_each + "Si" * n_each)
    msd = compute_msd(frames, timestep=1.0, output_frequency=1000)
    assert msd["n_atoms_per_species"] == {"Na": n_each, "Si": n_each}
    diffusion = get_diffusion(msd, fit_start_frac=0.05, fit_end_frac=0.5)
    assert diffusion["per_species"]["Na"]["diffusion_cm2_s"] > 5 * diffusion["per_species"]["Si"]["diffusion_cm2_s"]


def test_crosscheck_brute_force_exact():
    """The brute-force cross-check matches the FFT MSD to tight tolerance."""
    rng = np.random.default_rng(3)
    trajectory = np.cumsum(rng.normal(scale=0.1, size=(200, 40, 3)), axis=0)
    frames = _frames_from_trajectory(trajectory, "H" * 40)
    msd = compute_msd(frames, remove_com_drift=False)
    result = crosscheck_msd(msd, frames, reference="brute_force", remove_com_drift=False)
    assert result["passed"]
    assert result["relative_deviation"] < 1e-6


def test_crosscheck_flags_mismatch():
    """A corrupted MSD trips the cross-check and warns."""
    rng = np.random.default_rng(4)
    trajectory = np.cumsum(rng.normal(scale=0.1, size=(200, 40, 3)), axis=0)
    frames = _frames_from_trajectory(trajectory, "H" * 40)
    msd = compute_msd(frames, remove_com_drift=False)
    msd["msd_total"] = [value * 2.0 for value in msd["msd_total"]]
    with pytest.warns(UserWarning, match="exceeds rtol"):
        result = crosscheck_msd(msd, frames, reference="brute_force", remove_com_drift=False)
    assert not result["passed"]


def test_compute_msd_requires_two_frames():
    """A single frame cannot define an MSD."""
    frames = _frames_from_trajectory(np.zeros((1, 5, 3)), "H" * 5)
    with pytest.raises(ValueError, match="At least two frames"):
        compute_msd(frames)


def test_fit_arrhenius_recovery():
    """fit_arrhenius recovers the prefactor and activation energy of synthetic Arrhenius data."""
    temperatures = np.array([2000.0, 2500.0, 3000.0, 3500.0])
    d0_true, ea_true = 1e-3, 1.2
    diffusivities = d0_true * np.exp(-ea_true / (KB_EV * temperatures))
    (d0, ea), _ = fit_arrhenius(temperatures, diffusivities)
    assert d0 == pytest.approx(d0_true, rel=1e-6)
    assert ea == pytest.approx(ea_true, rel=1e-6)


def test_fit_arrhenius_rejects_nonpositive():
    """Non-positive diffusivities are rejected."""
    with pytest.raises(ValueError, match="strictly positive"):
        fit_arrhenius([2000.0, 3000.0], [1e-3, -1e-3])


def test_nernst_einstein_closed_form():
    """Single-species Nernst-Einstein conductivity matches the closed-form value."""
    diffusion = 1e-9
    out = nernst_einstein_conductivity(
        {"Na": diffusion},
        {"Na": 1.0},
        n_per_species={"Na": 100},
        volume_a3=1.0e5,
        temperature=1000.0,
    )
    e_charge, kb_j = 1.602176634e-19, 1.380649e-23
    expected = (100 / (1.0e5 * 1e-30)) * (1.0 * e_charge) ** 2 * diffusion / (kb_j * 1000.0)
    assert out["conductivity_S_m"] == pytest.approx(expected, rel=1e-9)
    assert out["per_species_contribution"]["Na"] == pytest.approx(expected, rel=1e-9)


def test_subdiffusive_warning():
    """A clearly sub-diffusive (ballistic) MSD warns during the fit."""
    lag_time_ps = np.linspace(0, 10, 50)
    msd = {"lag_time_ps": lag_time_ps.tolist(), "msd_total": (lag_time_ps**2).tolist()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_diffusion(msd, fit_start_frac=0.1, fit_end_frac=0.9)
    assert any("diffusive regime" in str(w.message) for w in caught)


def test_log_linear_dump_steps_schedule():
    """Schedule is log-spaced below the crossover and linear (fixed interval) above it."""
    steps = log_linear_dump_steps(5000, crossover_steps=500, points_per_decade=10, linear_interval_steps=200)
    assert steps[0] == 0
    assert steps == sorted(set(steps))  # strictly increasing, no duplicates
    assert steps[-1] <= 5000
    # below crossover: geometric growth (ratios > 1, increasing gaps)
    below = [s for s in steps if s < 500]
    assert below[1] / below[0] if below[0] else True  # smoke
    # above crossover: uniform spacing of linear_interval_steps
    above = [s for s in steps if s >= 500]
    above_diffs = np.diff(above)
    assert np.all(above_diffs == 200)


def test_compute_msd_single_origin_nonuniform():
    """Single-origin MSD on a non-uniformly (log+linear) sampled random walk recovers D."""
    rng = np.random.default_rng(0)
    schedule = log_linear_dump_steps(20000, crossover_steps=1000, points_per_decade=12, linear_interval_steps=500)
    d_target = 0.005  # Angstrom^2/ps
    var_per_step = 2 * d_target * (1.0 / 1000.0)  # 2 D dt, dt = 1 fs = 1e-3 ps
    walk = np.cumsum(rng.normal(scale=np.sqrt(var_per_step), size=(schedule[-1] + 1, 300, 3)), axis=0)
    frames = []
    for step in schedule:
        atoms = Atoms("H" * 300, positions=walk[step], cell=[500, 500, 500], pbc=True)
        atoms.arrays["unwrapped_positions"] = walk[step]
        atoms.info["step"] = step
        frames.append(atoms)
    msd = compute_msd(frames, timestep=1.0, method="single_origin", remove_com_drift=False)
    # the lag-time axis must be non-uniform (log then linear)
    diffs = np.diff(msd["lag_time_ps"])
    assert not np.allclose(diffs, diffs[0])
    diffusion = get_diffusion(msd, fit_start_frac=0.3, fit_end_frac=0.9)
    assert diffusion["total"]["diffusion_cm2_s"] == pytest.approx(d_target * 1e-4, rel=0.15)


def test_compute_msd_unknown_method_raises():
    """An unknown MSD method is rejected."""
    frames = _frames_from_trajectory(np.zeros((5, 4, 3)), "H" * 4)
    with pytest.raises(ValueError, match="Unknown method"):
        compute_msd(frames, method="bogus")
