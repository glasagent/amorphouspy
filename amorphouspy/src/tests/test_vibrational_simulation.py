"""Tests for ``vdos_simulation`` using a mocked LAMMPS runner.

The LAMMPS MD call is patched to return a synthetic parsed block, so the orchestration (stages, frame handling, groups
and spectra) is exercised without a real LAMMPS installation.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from amorphouspy.properties.vibrational import (
    classify_vibrational_groups,
    compute_partial_vacf,
    compute_partial_vdos,
    vdos_simulation,
)
from ase import Atoms

SI_O_BOND = 1.6
GROUP_KWARGS = {"cutoff": 2.0, "former_types": [14]}
# 0.2 ps at the default 1 fs timestep, dumped every 5 steps: 200 steps, 40 kept frames plus the step-0 frame,
# 5 fs between frames, 5 THz resolution and a 100 THz Nyquist frequency.
BLOCK = {"block_ps": 0.2}
N_FRAMES = 40
DT_FS = 5.0


def _si2o7_dimer() -> Atoms:
    """Two Si sharing one bridging O (position 2), each with three terminal O, in a large box, with velocities."""
    si1 = np.array([10.0, 10.0, 10.0])
    si2 = si1 + np.array([2 * SI_O_BOND, 0.0, 0.0])
    bridging = si1 + np.array([SI_O_BOND, 0.0, 0.0])
    terminals_1 = [si1 + d for d in ([-SI_O_BOND, 0, 0], [0, SI_O_BOND, 0], [0, 0, SI_O_BOND])]
    terminals_2 = [si2 + d for d in ([SI_O_BOND, 0, 0], [0, SI_O_BOND, 0], [0, 0, SI_O_BOND])]
    positions = [si1, si2, bridging, *terminals_1, *terminals_2]
    atoms = Atoms("Si2O7", positions=positions, cell=[30.0, 30.0, 30.0], pbc=True)
    atoms.set_velocities(np.random.default_rng(0).normal(scale=0.01, size=(len(atoms), 3)))
    return atoms


def _generic(n_atoms, n_frames=N_FRAMES + 1, seed=1):
    """Parsed NVE output: random velocities in Å/fs and per-frame thermo."""
    return {
        "velocities": np.random.default_rng(seed).normal(scale=0.01, size=(n_frames, n_atoms, 3)),
        "temperature": 300.0 + np.arange(n_frames, dtype=float),
        "energy_tot": -1000.0 + np.arange(n_frames, dtype=float),
    }


def _potential():
    """Return a minimal non-empty potential DataFrame."""
    return pd.DataFrame({"Name": ["pmmcs"], "Config": [["pair_style buck/coul/long 8.0"]]})


def _run(structure, generic, *, potential=None, side_effect=None, **kwargs):
    """Call vdos_simulation with the MD runner mocked; by default every stage returns the input structure."""
    arguments = {"thermostat": "nose_hoover", "equilibration_steps": 50, **BLOCK, **GROUP_KWARGS, **kwargs}
    with patch("amorphouspy.properties.vibrational._run_lammps_md", autospec=True) as mock_md:
        mock_md.return_value = (structure, {"generic": generic})
        mock_md.side_effect = side_effect
        out = vdos_simulation(structure, _potential() if potential is None else potential, **arguments)
    return out, mock_md


def _assert_layout(result, partial):
    """``result`` has the vdos_simulation layout; ``partial`` is the library output with the all-atoms group."""
    assert list(result) == ["total", "element", "oxygen", "qn"]
    np.testing.assert_array_equal(result["total"], partial["total"]["all"])
    for grouping in ("element", "oxygen", "qn"):
        assert partial[grouping]
        assert result[grouping].keys() == partial[grouping].keys()
        for label, values in partial[grouping].items():
            np.testing.assert_array_equal(result[grouping][label], values)


def _same_groups(a, b):
    return a.keys() == b.keys() and all(
        a[g].keys() == b[g].keys() and all(np.array_equal(a[g][label], b[g][label]) for label in a[g]) for g in a
    )


@pytest.mark.parametrize(
    ("thermostat", "equilibration_steps", "ensembles", "steps"),
    [
        ("nose_hoover", 50, ["nvt_langevin", "nvt", "nve"], [10_000, 50, 200]),
        ("langevin", 50, ["nvt_langevin", "nvt_langevin", "nve"], [10_000, 50, 200]),
        ("nose_hoover", 0, ["nvt_langevin", "nve"], [10_000, 200]),
    ],
)
def test_vdos_simulation_stages(thermostat, equilibration_steps, ensembles, steps, tmp_path):
    """Langevin, NVT (skipped for 0 steps) and NVE run in order, each on the output structure of the stage before."""
    structure = _si2o7_dimer()
    outputs = [_si2o7_dimer() for _ in ensembles]
    generic = _generic(len(structure))
    potential = _potential()
    out, mock_md = _run(
        structure,
        generic,
        potential=potential,
        side_effect=[(output, {"generic": generic}) for output in outputs],
        thermostat=thermostat,
        equilibration_steps=equilibration_steps,
        seed=7,
        server_kwargs={"cores": 2},
        tmp_working_directory=tmp_path,
    )

    calls = [c.kwargs for c in mock_md.call_args_list]
    assert [c["ensemble"] for c in calls] == ensembles
    assert [c["n_ionic_steps"] for c in calls] == steps
    for c, previous in zip(calls, [structure, *outputs[:-1]], strict=True):
        assert c["structure"] is previous
        assert c["potential"] is potential
        assert (c["seed"], c["server_kwargs"], c["tmp_working_directory"]) == (7, {"cores": 2}, tmp_path)
    for c in calls[:-1]:
        assert (c["temperature"], c["initial_temperature"]) == (300.0, 300.0)
    nve = {key: value for key, value in calls[-1].items() if key not in ("structure", "potential")}
    assert nve == {
        "tmp_working_directory": tmp_path,
        "temperature": None,
        "initial_temperature": None,
        "n_ionic_steps": 200,
        "timestep": 1.0,
        "n_dump": 5,
        "ensemble": "nve",
        "seed": 7,
        "server_kwargs": {"cores": 2},
    }
    assert out["structure"] is outputs[-1]


@pytest.mark.parametrize(
    ("kwargs", "window", "integral"),
    [
        ({}, "hann", 1.0),
        ({"normalization": "3N", "window": ("gaussian", 5.0)}, ("gaussian", 5.0), 27.0),
        ({"window": None}, None, 1.0),
    ],
    ids=["unit-default-hann", "3N-gaussian", "unit-no-window-only"],
)
def test_vdos_simulation_drops_first_frame(kwargs, window, integral):
    """Frame 0 (the NVT end state) is dropped: every output equals the library functions on frames 1 to N."""
    structure = _si2o7_dimer()
    generic = _generic(len(structure))
    generic["velocities"][0] = 1e6
    out, _ = _run(structure, generic, **kwargs)
    normalization = kwargs.get("normalization", "unit")

    kept = generic["velocities"][1:]
    masses = structure.get_masses()
    groups = classify_vibrational_groups(structure, **GROUP_KWARGS)
    spectral_groups = {"total": {"all": np.arange(len(structure))}, **groups}
    windows = {"no_window": None} if window is None else {"no_window": None, "windowed": window}
    assert list(out["vdos"]) == list(windows)
    for key, key_window in windows.items():
        frequencies_thz, partial = compute_partial_vdos(
            kept, masses, DT_FS, spectral_groups, normalization=normalization, window=key_window
        )
        np.testing.assert_array_equal(out["frequencies_thz"], frequencies_thz)
        _assert_layout(out["vdos"][key], partial)
        assert np.trapezoid(out["vdos"][key]["total"], frequencies_thz) == pytest.approx(integral, rel=1e-12)
    lag_ps, partial = compute_partial_vacf(kept, masses, DT_FS, spectral_groups)
    np.testing.assert_array_equal(out["lag_ps"], lag_ps)
    _assert_layout(out["vacf"], partial)

    assert set(out) == {
        "frequencies_thz",
        "vdos",
        "lag_ps",
        "vacf",
        "groups",
        "structure",
        "temperature",
        "energy_tot",
        "dt_fs",
        "n_frames",
        "resolution_thz",
        "nyquist_thz",
        "normalization",
        "window",
    }
    np.testing.assert_array_equal(out["temperature"], generic["temperature"][1:])
    np.testing.assert_array_equal(out["energy_tot"], generic["energy_tot"][1:])
    assert (out["dt_fs"], out["n_frames"], out["normalization"], out["window"]) == (
        DT_FS,
        N_FRAMES,
        normalization,
        window,
    )
    assert out["resolution_thz"] == pytest.approx(5.0)
    assert out["resolution_thz"] == pytest.approx(out["frequencies_thz"][1])
    assert out["nyquist_thz"] == pytest.approx(100.0)
    assert out["nyquist_thz"] == pytest.approx(out["frequencies_thz"][-1])


def test_vdos_simulation_cosine_peak():
    """A cosine at an on-grid frequency (20 THz, bin 4 of the 5 THz grid) peaks there with and without the window."""
    structure = _si2o7_dimer()
    masses = structure.get_masses()
    amplitudes = np.random.default_rng(2).normal(size=len(structure))
    amplitudes -= masses @ amplitudes / masses.sum()  # no centre-of-mass motion
    t_ps = np.arange(N_FRAMES + 1) * DT_FS * 1e-3
    velocities = np.zeros((N_FRAMES + 1, len(structure), 3))
    velocities[:, :, 0] = np.cos(2 * np.pi * 20.0 * t_ps)[:, None] * amplitudes
    out, _ = _run(structure, {**_generic(len(structure)), "velocities": velocities})

    for key in ("no_window", "windowed"):
        assert out["frequencies_thz"][np.argmax(out["vdos"][key]["total"])] == pytest.approx(20.0)


def test_vdos_simulation_groups_from_input_structure():
    """The groups are classified on the input structure, not on a structure returned by the MD."""
    structure = _si2o7_dimer()
    moved = _si2o7_dimer()
    moved.positions[3] += [0.0, 0.0, 8.0]  # a terminal O leaves its Si and becomes free
    generic = _generic(len(structure))
    out, _ = _run(structure, generic, side_effect=lambda **_: (moved, {"generic": generic}))

    assert _same_groups(out["groups"], classify_vibrational_groups(structure, **GROUP_KWARGS))
    assert not _same_groups(out["groups"], classify_vibrational_groups(moved, **GROUP_KWARGS))


@pytest.mark.parametrize(("key", "n_parsed"), [("velocities", N_FRAMES), ("energy_tot", N_FRAMES + 2)])
def test_vdos_simulation_frame_count_mismatch_raises(key, n_parsed):
    """A parsed frame count other than the block frames plus the step-0 frame raises; nothing is trimmed."""
    structure = _si2o7_dimer()
    generic = _generic(len(structure))
    generic[key] = _generic(len(structure), n_frames=n_parsed)[key]
    with pytest.raises(RuntimeError, match=rf"{n_parsed} frames of '{key}', expected {N_FRAMES + 1}"):
        _run(structure, generic)


def test_vdos_simulation_nyquist_warning():
    """A Nyquist frequency below 50 THz warns and names the timestep and dump_every; the defaults do not warn."""
    structure = _si2o7_dimer()
    # 2 fs every 10 steps: 20 fs between frames, Nyquist 25 THz; 0.2 ps is 100 steps and 10 frames.
    with pytest.warns(UserWarning, match=r"25 THz \(timestep 2.0 fs, dump_every 10\)"):
        _run(structure, _generic(len(structure), n_frames=11), timestep=2.0, dump_every=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _run(structure, _generic(len(structure)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"potential": pd.DataFrame()}, "No matching potential"),
        ({"timestep": 0.0}, "timestep must be positive and finite"),
        ({"block_ps": -1.0}, "block_ps must be positive and finite"),
        ({"temperature_sim": float("nan")}, "temperature_sim must be positive and finite"),
        ({"dump_every": 0}, "dump_every must be a positive integer"),
        ({"dump_every": 2.5}, "dump_every must be a positive integer"),
        ({"dump_every": True}, "dump_every must be a positive integer"),
        ({"equilibration_steps": -1}, "equilibration_steps must not be negative"),
        ({"normalization": "none"}, "Unknown normalization"),
        ({"window": "gaussian"}, "Unknown window 'gaussian'"),
        ({"timestep": 3.0}, "not a whole number of 3.0 fs timesteps"),
        ({"dump_every": 3}, "not a multiple of dump_every=3"),
        ({"block_ps": 0.005}, "at least 2 frames"),
    ],
    ids=[
        "potential-empty",
        "timestep-zero",
        "block-negative",
        "temperature-nan",
        "dump-zero",
        "dump-float",
        "dump-bool",
        "equilibration-negative",
        "normalization-none",
        "window-gaussian-without-std",
        "block-not-whole-steps",
        "block-not-multiple",
        "block-one-frame",
    ],
)
def test_vdos_simulation_invalid_inputs_raise(kwargs, match):
    """Invalid arguments raise before any MD stage is launched."""
    arguments = {"potential": _potential(), "thermostat": "nose_hoover", **BLOCK, **GROUP_KWARGS, **kwargs}
    with (
        patch("amorphouspy.properties.vibrational._run_lammps_md", autospec=True) as mock_md,
        pytest.raises(ValueError, match=match),
    ):
        vdos_simulation(_si2o7_dimer(), **arguments)
    assert mock_md.call_count == 0
