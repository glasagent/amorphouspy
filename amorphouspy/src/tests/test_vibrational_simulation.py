"""Tests for ``vdos_simulation`` and ``harmonic_vdos_simulation`` using mocked LAMMPS runners.

The LAMMPS MD call is patched to return a synthetic parsed block, so the orchestration (stages, frame handling, groups
and spectra) is exercised without a real LAMMPS installation. The dynamical-matrix runner is patched to write a known
mass-weighted matrix, and everything after it runs for real.
"""

import gc
import shutil
import tempfile
import warnings
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from amorphouspy.properties.vibrational import (
    classify_vibrational_groups,
    compute_harmonic_vdos,
    compute_partial_vacf,
    compute_partial_vdos,
    compute_participation_ratio,
    compute_vibrational_thermodynamics,
    diagonalize_dynamical_matrix,
    harmonic_simulation,
    harmonic_vdos_simulation,
    read_dynamical_matrix,
    vdos_simulation,
)
from ase import Atoms

SI_O_BOND = 1.6
GROUP_KWARGS = {"cutoff": 2.0, "former_types": [14]}
# One block of 0.2 ps at the default 1 fs timestep, dumped every 5 steps: 200 steps, 40 kept frames plus the step-0
# frame, 5 fs between frames, 5 THz resolution and a 100 THz Nyquist frequency.
BLOCK = {"block_ps": 0.2, "production_ps": 0.2}
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


def _outlier_blocks(n_atoms, n_blocks):
    """Parsed outputs of consecutive blocks with different velocities and thermo, and an outlier in each frame 0."""
    generics = [_generic(n_atoms, seed=block + 1) for block in range(n_blocks)]
    for block, generic in enumerate(generics):
        generic["velocities"][0] = 1e6
        generic["temperature"] += 100.0 * block
        generic["energy_tot"] += 100.0 * block
    return generics


def _cosine_velocities(structure):
    """Velocities of an on-grid cosine: 20 THz is bin 4 of the 5 THz grid, and 4 whole periods fit in the block."""
    masses = structure.get_masses()
    amplitudes = np.random.default_rng(2).normal(size=len(structure))
    amplitudes -= masses @ amplitudes / masses.sum()  # no centre-of-mass motion
    t_ps = np.arange(N_FRAMES + 1) * DT_FS * 1e-3
    velocities = np.zeros((N_FRAMES + 1, len(structure), 3))
    velocities[:, :, 0] = np.cos(2 * np.pi * 20.0 * t_ps)[:, None] * amplitudes
    return velocities


def _potential():
    """Return a minimal non-empty potential DataFrame."""
    return pd.DataFrame({"Name": ["pmmcs"], "Config": [["pair_style buck/coul/long 8.0"]]})


def _run(structure, generic, *, potential=None, side_effect=None, **kwargs):
    """Call vdos_simulation with the MD runner mocked; by default every stage returns the input structure."""
    arguments = {"thermostat": "nose_hoover", "equilibration_steps": 50, **BLOCK, **GROUP_KWARGS, **kwargs}
    with patch("amorphouspy.properties.vibrational.simulation._run_lammps_md", autospec=True) as mock_md:
        mock_md.return_value = (structure, {"generic": generic})
        mock_md.side_effect = side_effect
        out = vdos_simulation(structure, _potential() if potential is None else potential, **arguments)
    return out, mock_md


def _run_blocks(structure, generics, **kwargs):
    """Call vdos_simulation for ``len(generics)`` blocks, each returning its own parsed output."""
    outputs = [(structure, {"generic": generic}) for generic in [generics[0], generics[0], *generics]]
    return _run(structure, None, side_effect=outputs, production_ps=0.2 * len(generics), **kwargs)[0]


def _fake_runner(structure, generics, run_dirs):
    """Mimic the runner: a run given a directory makes its own sub-directory there and writes a dump and a log."""
    outputs = iter(generics)

    def run_md(**kwargs):
        if kwargs["tmp_working_directory"] is not None:
            run_dir = Path(tempfile.mkdtemp(dir=kwargs["tmp_working_directory"]))
            for name in ("dump.out", "log.lammps"):
                (run_dir / name).write_text(name)
            run_dirs.append(run_dir)
        return structure, {"generic": next(outputs)}

    return run_md


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


def _leaves(curves):
    """Yield the arrays of nested dicts of curves, in order."""
    for value in curves.values():
        if isinstance(value, dict):
            yield from _leaves(value)
        else:
            yield value


def _over_blocks(blocks, statistic):
    """Apply ``statistic`` to each curve of a list of library outputs that have the all-atoms group."""
    return {
        group: {label: statistic(np.array([block[group][label] for block in blocks])) for label in blocks[0][group]}
        for group in blocks[0]
    }


@pytest.mark.parametrize(
    ("thermostat", "equilibration_steps", "ensembles", "steps"),
    [
        ("nose_hoover", 50, ["nvt_langevin", "nvt", "nve"], [10_000, 50, 200]),
        ("langevin", 50, ["nvt_langevin", "nvt_langevin", "nve"], [10_000, 50, 200]),
        ("nose_hoover", 0, ["nvt_langevin", "nve"], [10_000, 200]),
        ("nose_hoover", 50, ["nvt_langevin", "nvt", "nve", "nve", "nve"], [10_000, 50, 200, 200, 200]),
    ],
)
def test_vdos_simulation_stages(thermostat, equilibration_steps, ensembles, steps, tmp_path):
    """Langevin, NVT (skipped for 0 steps) and the NVE blocks run in order, each on the output of the stage before."""
    structure = _si2o7_dimer()
    outputs = [_si2o7_dimer() for _ in ensembles]
    generic = _generic(len(structure))
    potential = _potential()
    n_blocks = ensembles.count("nve")
    # The mocked runs write no dump, so every block warns that it found nothing to delete, and still returns.
    with pytest.warns(UserWarning, match=r"No dump.out found under .* for NVE block") as record:
        out, mock_md = _run(
            structure,
            generic,
            potential=potential,
            side_effect=[(output, {"generic": generic}) for output in outputs],
            thermostat=thermostat,
            equilibration_steps=equilibration_steps,
            production_ps=0.2 * n_blocks,
            seed=7,
            server_kwargs={"cores": 2},
            tmp_working_directory=tmp_path,
        )
    assert len(record) == n_blocks

    assert mock_md.call_count == (2 if equilibration_steps else 1) + n_blocks
    calls = [c.kwargs for c in mock_md.call_args_list]
    assert [c["ensemble"] for c in calls] == ensembles
    assert [c["n_ionic_steps"] for c in calls] == steps
    for c, previous in zip(calls, [structure, *outputs[:-1]], strict=True):
        assert c["structure"] is previous
        assert c["potential"] is potential
        assert (c["seed"], c["server_kwargs"]) == (7, {"cores": 2})
    for c in calls[:-n_blocks]:
        assert (c["temperature"], c["initial_temperature"], c["tmp_working_directory"]) == (300.0, 300.0, tmp_path)
    for block, c in enumerate(calls[-n_blocks:], start=1):
        block_dir = Path(c["tmp_working_directory"])
        assert block_dir.parent == tmp_path
        assert block_dir.name.startswith(f"vdos_nve_block_{block:03d}_")
        nve = {key: value for key, value in c.items() if key not in ("structure", "potential", "tmp_working_directory")}
        assert nve == {
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
        "vdos_sem",
        "lag_ps",
        "vacf",
        "vacf_sem",
        "groups",
        "structure",
        "temperature",
        "energy_tot",
        "nve_drift",
        "dt_fs",
        "n_blocks",
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


@pytest.mark.parametrize("window", ["hann", None], ids=["hann", "no-window-only"])
def test_vdos_simulation_block_statistics(window):
    """Every curve is the mean and the standard error over blocks of the library output on frames 1 to N of a block."""
    structure = _si2o7_dimer()
    generics = _outlier_blocks(len(structure), 3)
    out = _run_blocks(structure, generics, window=window)

    masses = structure.get_masses()
    groups = classify_vibrational_groups(structure, **GROUP_KWARGS)
    spectral_groups = {"total": {"all": np.arange(len(structure))}, **groups}
    kept = [generic["velocities"][1:] for generic in generics]

    def mean(curves):
        return np.mean(curves, axis=0)

    def sem(curves):
        return np.std(curves, axis=0, ddof=1) / np.sqrt(3)

    windows = {"no_window": None} if window is None else {"no_window": None, "windowed": window}
    assert list(out["vdos"]) == list(windows)
    assert list(out["vdos_sem"]) == list(windows)
    for key, key_window in windows.items():
        blocks = [compute_partial_vdos(v, masses, DT_FS, spectral_groups, window=key_window)[1] for v in kept]
        _assert_layout(out["vdos"][key], _over_blocks(blocks, mean))
        _assert_layout(out["vdos_sem"][key], _over_blocks(blocks, sem))
        total = out["vdos"][key]["total"]
        assert np.trapezoid(total, out["frequencies_thz"]) == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(sum(out["vdos"][key]["element"].values()), total, atol=1e-12)
    blocks = [compute_partial_vacf(v, masses, DT_FS, spectral_groups)[1] for v in kept]
    _assert_layout(out["vacf"], _over_blocks(blocks, mean))
    _assert_layout(out["vacf_sem"], _over_blocks(blocks, sem))
    assert out["vacf"]["total"][0] == pytest.approx(1.0, rel=1e-12)

    np.testing.assert_array_equal(out["temperature"], np.concatenate([g["temperature"][1:] for g in generics]))
    np.testing.assert_array_equal(out["energy_tot"], np.concatenate([g["energy_tot"][1:] for g in generics]))
    assert (out["n_blocks"], out["n_frames"], len(out["temperature"])) == (3, N_FRAMES, 3 * N_FRAMES)


def test_vdos_simulation_stationary_signal():
    """Identical blocks of an on-grid cosine give the single-block result, with a standard error at rounding level."""
    structure = _si2o7_dimer()
    generic = {**_generic(len(structure)), "velocities": _cosine_velocities(structure)}
    single, _ = _run(structure, generic)
    out = _run_blocks(structure, [generic] * 3)

    for key in ("vdos", "vacf"):
        for mean, one, sem in zip(_leaves(out[key]), _leaves(single[key]), _leaves(out[f"{key}_sem"]), strict=True):
            rounding = 1e-12 * np.abs(one).max()
            np.testing.assert_allclose(mean, one, rtol=0.0, atol=rounding)
            assert np.isfinite(sem).all()
            assert sem.max() <= rounding


def test_vdos_simulation_one_block_sem_is_nan():
    """With one block the standard error is NaN in the layout of the curves, without any warning."""
    structure = _si2o7_dimer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, _ = _run(structure, _generic(len(structure)))

    for key in ("vdos", "vacf"):
        assert out[f"{key}_sem"].keys() == out[key].keys()
        for curve, sem in zip(_leaves(out[key]), _leaves(out[f"{key}_sem"]), strict=True):
            assert sem.shape == curve.shape
            assert np.isnan(sem).all()
    assert out["nve_drift"]["boundary_energy_jump_mev_per_atom"].shape == (0,)


@pytest.mark.parametrize("keep_trajectory", [False, True], ids=["delete", "keep"])
def test_vdos_simulation_dump_cleanup(keep_trajectory, tmp_path):
    """Each block runs in its own directory; unless kept, its dump goes after its spectra, before the next block."""
    structure = _si2o7_dimer()
    run_dirs, dumps_at_launch = [], []
    fake = _fake_runner(structure, [_generic(len(structure))] * 5, run_dirs)

    def run_md(**kwargs):
        dumps_at_launch.append(len(list(tmp_path.glob("vdos_nve_block_*/*/dump.out"))))
        return fake(**kwargs)

    checked = []

    def vacf_after_dump_check(*args, **kwargs):
        checked.append((run_dirs[-1] / "dump.out").exists())
        return compute_partial_vacf(*args, **kwargs)

    with patch("amorphouspy.properties.vibrational.simulation.compute_partial_vacf", side_effect=vacf_after_dump_check):
        _run(
            structure,
            None,
            side_effect=run_md,
            production_ps=0.6,
            tmp_working_directory=tmp_path,
            keep_trajectory=keep_trajectory,
        )

    assert checked == [True] * 3  # the patch took effect: each block's dump existed when its VACF was computed
    block_dirs = [run_dir.parent for run_dir in run_dirs[2:]]
    assert [d.parent for d in block_dirs] == [tmp_path] * 3
    assert all(d.name.startswith(f"vdos_nve_block_{block:03d}_") for block, d in enumerate(block_dirs, start=1))
    assert dumps_at_launch[2:] == ([0, 1, 2] if keep_trajectory else [0, 0, 0])
    assert len(list(tmp_path.glob("vdos_nve_block_*/*/dump.out"))) == (3 if keep_trajectory else 0)
    assert all((run_dir / "log.lammps").exists() for run_dir in run_dirs)
    assert all((run_dir / "dump.out").exists() for run_dir in run_dirs[:2])  # Langevin and NVT are untouched


def test_vdos_simulation_nve_drift():
    """Every drift key on a linear energy and temperature over 3 blocks, with known step-0 energies (boundary jumps)."""
    structure = _si2o7_dimer()
    n_atoms, h_ps, n_kept = len(structure), DT_FS * 1e-3, 3 * N_FRAMES
    slope_ev_ps, slope_k_ps, jumps_ev = 0.9, 2.0, [0.018, -0.027]
    generics = []
    for block in range(3):
        generic = _generic(n_atoms, seed=block + 1)
        t_ps = (block * N_FRAMES + np.arange(1, N_FRAMES + 1)) * h_ps
        generic["energy_tot"][1:] = -1000.0 + slope_ev_ps * t_ps
        generic["temperature"][1:] = 300.0 + slope_k_ps * t_ps
        generic["temperature"][0] = 1e4  # the step-0 frame never enters the statistics
        previous_last = generics[-1]["energy_tot"][-1] if generics else -950.0
        generic["energy_tot"][0] = previous_last + (jumps_ev[block - 1] if block else 0.0)
        generics.append(generic)
    drift = _run_blocks(structure, generics)["nve_drift"]

    mev_per_atom = 1e3 / n_atoms
    assert drift["energy_drift_mev_per_atom_ps"] == pytest.approx(slope_ev_ps * mev_per_atom)
    np.testing.assert_allclose(drift["block_energy_drift_mev_per_atom_ps"], [slope_ev_ps * mev_per_atom] * 3)
    np.testing.assert_allclose(drift["boundary_energy_jump_mev_per_atom"], np.array(jumps_ev) * mev_per_atom)
    assert drift["max_energy_deviation_mev_per_atom"] == pytest.approx(slope_ev_ps * (n_kept - 1) * h_ps * mev_per_atom)
    assert drift["temperature_mean_k"] == pytest.approx(300.0 + slope_k_ps * h_ps * (n_kept + 1) / 2)
    assert drift["temperature_std_k"] == pytest.approx(slope_k_ps * h_ps * np.sqrt((n_kept**2 - 1) / 12))
    assert drift["temperature_drift_k_per_ps"] == pytest.approx(slope_k_ps)


def test_vdos_simulation_frees_each_block_before_the_next():
    """The velocities of a block are dead when the next block is launched, not only after the function returns."""
    structure = _si2o7_dimer()
    block_velocities = []

    def run_md(**kwargs):
        generic = _generic(len(structure))
        if kwargs["ensemble"] == "nve":
            assert all(velocities() is None for velocities in block_velocities)
            block_velocities.append(weakref.ref(generic["velocities"]))
        return structure, {"generic": generic}

    _run(structure, None, side_effect=run_md, production_ps=0.6)
    assert len(block_velocities) == 3


def test_vdos_simulation_cosine_peak():
    """A cosine at an on-grid frequency (20 THz, bin 4 of the 5 THz grid) peaks there with and without the window."""
    structure = _si2o7_dimer()
    velocities = _cosine_velocities(structure)
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


def test_vdos_simulation_frame_count_mismatch_stops_at_named_block(tmp_path):
    """A wrong frame count in block 2 of 3 names that block and its kept dump, and no third block is launched."""
    structure = _si2o7_dimer()
    good, short = _generic(len(structure)), _generic(len(structure), n_frames=N_FRAMES)
    run_dirs = []
    generics = [good, good, good, short, good]
    match = rf"block 2 of 3 returned {N_FRAMES} frames of 'velocities', expected 41"
    with pytest.raises(RuntimeError, match=match) as e:
        _run(
            structure,
            None,
            side_effect=_fake_runner(structure, generics, run_dirs),
            production_ps=0.6,
            tmp_working_directory=tmp_path,
        )
    assert len(run_dirs) == 4  # Langevin, NVT and blocks 1 and 2
    dump = run_dirs[-1] / "dump.out"
    assert dump.exists()
    assert str(e.value).endswith(f"Its dump is kept in {dump}.")


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
        ({"timestep": None}, "timestep must be positive and finite"),
        ({"temperature_sim": True}, "temperature_sim must be positive and finite"),
        ({"block_ps": -1.0}, "block_ps must be positive and finite"),
        ({"temperature_sim": float("nan")}, "temperature_sim must be positive and finite"),
        ({"dump_every": 0}, "dump_every must be a positive integer"),
        ({"dump_every": 2.5}, "dump_every must be a positive integer"),
        ({"dump_every": True}, "dump_every must be a positive integer"),
        ({"equilibration_steps": -1}, "equilibration_steps must not be negative"),
        ({"equilibration_steps": 1.5}, "equilibration_steps must not be negative and must be an integer"),
        ({"normalization": "none"}, "Unknown normalization"),
        ({"window": "gaussian"}, "Unknown window 'gaussian'"),
        ({"timestep": 3.0}, "not a whole number of 3.0 fs timesteps"),
        ({"dump_every": 3}, "not a multiple of dump_every=3"),
        ({"block_ps": 0.005}, "at least 2 frames"),
        ({"production_ps": float("inf")}, "production_ps must be positive and finite"),
        ({"production_ps": 0.2005}, "production_ps=0.2005 ps is not a whole number of 1.0 fs timesteps"),
        ({"production_ps": 0.5}, "production of 500 steps is not a multiple of the block of 200 steps"),
        ({"production_ps": 0.1}, "production of 100 steps is not a multiple of the block of 200 steps"),
        ({"keep_trajectory": True}, "keep_trajectory=True needs a tmp_working_directory"),
    ],
    ids=[
        "potential-empty",
        "timestep-zero",
        "timestep-none",
        "temperature-bool",
        "block-negative",
        "temperature-nan",
        "dump-zero",
        "dump-float",
        "dump-bool",
        "equilibration-negative",
        "equilibration-float",
        "normalization-none",
        "window-gaussian-without-std",
        "block-not-whole-steps",
        "block-not-multiple",
        "block-one-frame",
        "production-inf",
        "production-not-whole-steps",
        "production-not-multiple",
        "production-shorter-than-block",
        "keep-trajectory-without-directory",
    ],
)
def test_vdos_simulation_invalid_inputs_raise(kwargs, match):
    """Invalid arguments raise before any MD stage is launched."""
    arguments = {"potential": _potential(), "thermostat": "nose_hoover", **BLOCK, **GROUP_KWARGS, **kwargs}
    with (
        patch("amorphouspy.properties.vibrational.simulation._run_lammps_md", autospec=True) as mock_md,
        pytest.raises(ValueError, match=match),
    ):
        vdos_simulation(_si2o7_dimer(), **arguments)
    assert mock_md.call_count == 0


# ---------------------------------------------------------------------------
# harmonic_vdos_simulation
# ---------------------------------------------------------------------------

HARMONIC_TEMPERATURES_K = [0.0, 300.0, 3000.0]
RUNNER_SUMMARY = {
    "matrix_bytes": 9 * 9**2 * 8,
    "style": "regular",
    "max_force_ev_per_a": 1e-7,
    "force_norm_ev_per_a": 2e-7,
    "minimization": {"converged": True, "n_iterations": 42, "mean_pressure_gpa": 0.1},
    "wall_time_s": 0.5,
}


def _minimised_dimer() -> Atoms:
    """The dimer as the runner returns it: terminal O 3 pulled from 1.6 to 2.2 Å of its Si, past the 2 Å cutoff."""
    atoms = _si2o7_dimer()
    atoms.positions[3] += np.array([-0.6, 0.0, 0.0])
    atoms.set_velocities(np.zeros((len(atoms), 3)))
    return atoms


def _dynamical_matrix(masses):
    """D = Q diag(eigenvalues) Q^T in eV/Å²/amu: three zero modes along the mass-weighted translations, then 0.3-3."""
    n_dof = 3 * len(masses)
    translations = np.zeros((n_dof, 3))
    for alpha in range(3):
        translations[alpha::3, alpha] = np.sqrt(masses)
    q, _ = np.linalg.qr(np.hstack([translations, np.random.default_rng(3).normal(size=(n_dof, n_dof - 3))]))
    matrix = (q * np.concatenate([np.zeros(3), np.linspace(0.3, 3.0, n_dof - 3)])) @ q.T
    return (matrix + matrix.T) / 2


def _fake_dynamical_matrix_runner(minimised, matrix):
    """Mimic the runner: write ``matrix`` to ``matrix_path``, a run directory in tmp_working_directory, a summary."""

    def run(_structure, _potential, matrix_path, **kwargs):
        matrix.tofile(matrix_path)
        if kwargs["tmp_working_directory"] is not None:
            run_dir = Path(kwargs["tmp_working_directory"]) / "lammps_run"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "log.lammps").touch()
        summary = {
            "matrix_path": Path(matrix_path),
            "n_atoms": len(minimised),
            "displacement_a": kwargs["displacement"],
            **RUNNER_SUMMARY,
        }
        return minimised, summary

    return run


def _run_harmonic(**kwargs):
    """Call harmonic_vdos_simulation on the dimer with the runner mocked; return result, mock, matrix and structure."""
    minimised = _minimised_dimer()
    matrix = _dynamical_matrix(minimised.get_masses())
    with patch(
        "amorphouspy.properties.vibrational.harmonic_simulation._run_lammps_dynamical_matrix", autospec=True
    ) as runner:
        runner.side_effect = _fake_dynamical_matrix_runner(minimised, matrix)
        out = harmonic_vdos_simulation(_si2o7_dimer(), _potential(), **{**GROUP_KWARGS, **kwargs})
    return out, runner, matrix, minimised


def _assert_same(actual, expected):
    """Nested dicts with identical keys and exactly equal arrays and values."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_same(actual[key], value)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def test_harmonic_vdos_simulation_matches_manual_chain(tmp_path):
    """The result is exactly read -> diagonalize -> VDOS -> participation ratio -> thermodynamics on the matrix."""
    with patch.object(
        harmonic_simulation, "diagonalize_dynamical_matrix", wraps=harmonic_simulation.diagonalize_dynamical_matrix
    ) as diagonalize:
        out, runner, matrix, minimised = _run_harmonic(temperatures_k=HARMONIC_TEMPERATURES_K, keep_eigenvectors=True)

    structure = runner.call_args.args[0]
    assert runner.call_count == 1
    assert runner.call_args.kwargs == {
        "displacement": 1e-5,
        "minimize": True,
        "minimize_kwargs": {"min_style": "fire", "pressure": None, "force_tolerance": 1e-6, "max_iter": 100_000},
        "server_kwargs": None,
        "tmp_working_directory": None,
    }
    assert diagonalize.call_args.kwargs["overwrite_matrix"] is True
    assert diagonalize.call_args.kwargs["eigenvectors"] is True

    path = tmp_path / "dynmat.bin"
    matrix.tofile(path)
    masses = minimised.get_masses()
    groups = classify_vibrational_groups(minimised, **GROUP_KWARGS)
    modes = diagonalize_dynamical_matrix(read_dynamical_matrix(path, len(minimised)), masses=masses)
    vdos = compute_harmonic_vdos(modes["frequencies_thz"], modes["eigenvectors"], groups=groups)
    positive = modes["frequencies_thz"][modes["frequencies_thz"] > 0]
    thermodynamics = compute_vibrational_thermodynamics(
        vdos["frequencies_thz"], vdos["vdos"]["total"], HARMONIC_TEMPERATURES_K, lowest_frequency_thz=positive.min()
    )
    diagonalization_keys = [
        "asymmetry",
        "asymmetry_atoms",
        "translational_frequencies_thz",
        "translational_overlap",
        "n_imaginary",
        "min_frequency_thz",
        "driver",
    ]
    expected = {
        "frequencies_thz": vdos["frequencies_thz"],
        "vdos": vdos["vdos"],
        "groups": groups,
        "normalization": "unit",
        "broadening_thz": 0.2,
        "mode_frequencies_thz": modes["frequencies_thz"],
        "participation_ratio": compute_participation_ratio(modes["eigenvectors"], masses),
        **{key: vdos[key] for key in ("mode_weights", "n_imaginary_excluded", "imaginary_weight", "grid_integral")},
        "eigenvectors": modes["eigenvectors"],
        "diagonalization": {key: modes[key] for key in diagonalization_keys},
        "thermodynamics": thermodynamics,
        "displacement_a": 1e-5,
    }
    _assert_same({key: out[key] for key in expected}, expected)
    assert out.keys() == {*expected, "structure", "dynamical_matrix", "minimization", "wall_time_s"}
    assert out["structure"] is minimised
    assert out["minimization"] == RUNNER_SUMMARY["minimization"]
    assert out["dynamical_matrix"] == {
        "matrix_path": None,
        "n_atoms": 9,
        "displacement_a": 1e-5,
        **RUNNER_SUMMARY,
    }
    assert out["wall_time_s"].keys() == {"lammps", "diagonalization", "analysis"}
    assert not _same_groups(out["groups"], classify_vibrational_groups(structure, **GROUP_KWARGS))


def test_harmonic_vdos_simulation_atom_guard():
    """N == max_atoms runs; one atom more raises before the runner, with the matrix size and the expected peak."""
    assert _run_harmonic(max_atoms=9)[1].call_count == 1
    match = r"9 atoms, more than max_atoms=8.*9 N² x 8 bytes = 5.83e-06 GB.*2 matrices, 1.17e-05 GB. Raise max_atoms"
    with (
        patch(
            "amorphouspy.properties.vibrational.harmonic_simulation._run_lammps_dynamical_matrix", autospec=True
        ) as runner,
        pytest.raises(ValueError, match=match),
    ):
        harmonic_vdos_simulation(_si2o7_dimer(), _potential(), max_atoms=8, **GROUP_KWARGS)
    assert runner.call_count == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_atoms": 0}, "max_atoms must be a positive integer"),
        ({"max_atoms": True}, "max_atoms must be a positive integer"),
        ({"max_atoms": 10.0}, "max_atoms must be a positive integer"),
        ({"max_atoms": "10"}, "max_atoms must be a positive integer"),
        ({"potential": pd.DataFrame()}, "No matching potential"),
        ({"displacement": 0.0}, "displacement must be positive and finite"),
        ({"displacement": float("nan")}, "displacement must be positive and finite"),
        ({"displacement": None}, "displacement must be positive and finite"),
        ({"force_tolerance": -1e-6}, "force_tolerance must be finite and non-negative"),
        ({"force_tolerance": 0.0}, "cannot both be 0"),
        ({"max_iter": 0}, "max_iter must be a positive integer"),
        ({"broadening_thz": 0.0}, "broadening_thz must be positive and finite"),
        ({"normalization": "none"}, "Unknown normalization"),
        ({"frequency_grid_thz": [2.0, 1.0]}, "frequency_grid_thz must be"),
        ({"temperatures_k": [300.0, -1.0]}, "temperatures_k must be"),
        ({"temperatures_k": True}, "temperatures_k must be.*not booleans"),
        ({"driver": "evz"}, "Unknown driver"),
        ({"imaginary_threshold_thz": -0.01}, "imaginary_threshold_thz must be >= 0"),
        ({"keep_dynamical_matrix": True}, "keep_dynamical_matrix=True needs a tmp_working_directory"),
    ],
    ids=[
        "max-atoms-zero",
        "max-atoms-bool",
        "max-atoms-float",
        "max-atoms-str",
        "potential-empty",
        "displacement-zero",
        "displacement-nan",
        "displacement-none",
        "force-tolerance-negative",
        "force-tolerance-zero",
        "max-iter-zero",
        "broadening-zero",
        "normalization-none",
        "grid-decreasing",
        "temperature-negative",
        "temperature-bool",
        "driver",
        "threshold-negative",
        "keep-matrix-without-directory",
    ],
)
def test_harmonic_vdos_simulation_invalid_inputs_raise(kwargs, match):
    """Invalid arguments raise before the LAMMPS runner is called."""
    arguments = {"potential": _potential(), **GROUP_KWARGS, **kwargs}
    with (
        patch(
            "amorphouspy.properties.vibrational.harmonic_simulation._run_lammps_dynamical_matrix", autospec=True
        ) as runner,
        pytest.raises(ValueError, match=match),
    ):
        harmonic_vdos_simulation(_si2o7_dimer(), **arguments)
    assert runner.call_count == 0


@pytest.mark.parametrize(
    ("directory", "keep_dynamical_matrix"),
    [(True, True), (True, False), (False, False)],
    ids=["directory-keep", "directory-delete", "temporary"],
)
def test_harmonic_vdos_simulation_matrix_file(directory, keep_dynamical_matrix, tmp_path):
    """Only the matrix's own directory is removed after reading, unless kept; the LAMMPS run directories stay."""
    tmp_working_directory = tmp_path if directory else None
    out, runner, matrix, _ = _run_harmonic(
        tmp_working_directory=tmp_working_directory, keep_dynamical_matrix=keep_dynamical_matrix
    )
    path = runner.call_args.args[2]
    assert path.name == "dynamical_matrix.bin"
    assert path.parent.name.startswith("harmonic_vdos_")
    assert runner.call_args.kwargs["tmp_working_directory"] == tmp_working_directory
    assert path.parent.exists() is keep_dynamical_matrix
    assert out["dynamical_matrix"]["matrix_path"] == (path if keep_dynamical_matrix else None)
    if keep_dynamical_matrix:
        assert np.array_equal(np.fromfile(path), matrix.ravel())
    if directory:
        assert path.parent.parent == tmp_path
        assert (tmp_path / "lammps_run" / "log.lammps").exists()


@pytest.mark.parametrize("directory", [False, True], ids=["system-temp", "tmp_working_directory"])
def test_harmonic_vdos_simulation_checks_free_space(monkeypatch, directory, tmp_path):
    """Too little free space where the matrix file goes raises before the runner; exactly enough space runs."""
    matrix_bytes = 9 * 9**2 * 8
    parent = str(tmp_path) if directory else tempfile.gettempdir()
    kwargs = {"tmp_working_directory": tmp_path} if directory else {}
    checked = []

    def usage(free):
        def disk_usage(path):
            checked.append(str(path))
            return shutil._ntuple_diskusage(2 * free, free, free)  # noqa: SLF001

        return disk_usage

    monkeypatch.setattr(harmonic_simulation.shutil, "disk_usage", usage(matrix_bytes - 1))
    with (
        patch(
            "amorphouspy.properties.vibrational.harmonic_simulation._run_lammps_dynamical_matrix", autospec=True
        ) as runner,
        pytest.raises(ValueError, match=r"9 N² x 8 bytes = 5.83e-06 GB, but .* has only 5.83e-06 GB free"),
    ):
        harmonic_vdos_simulation(_si2o7_dimer(), _potential(), **GROUP_KWARGS, **kwargs)
    assert runner.call_count == 0
    monkeypatch.setattr(harmonic_simulation.shutil, "disk_usage", usage(matrix_bytes))
    assert _run_harmonic(**kwargs)[1].call_count == 1
    assert checked == [parent, parent]


@pytest.mark.parametrize("keep_eigenvectors", [False, True], ids=["drop", "keep"])
def test_harmonic_vdos_simulation_eigenvectors(keep_eigenvectors):
    """The eigenvectors are returned only when kept; otherwise their buffer is freed and every other entry is equal."""
    buffers = []

    def spy(*args, **kwargs):
        modes = diagonalize_dynamical_matrix(*args, **kwargs)
        root = modes["eigenvectors"]
        while root.base is not None:
            root = root.base
        buffers.append(weakref.ref(root))
        return modes

    with patch.object(harmonic_simulation, "diagonalize_dynamical_matrix", side_effect=spy):
        out, _, matrix, minimised = _run_harmonic(keep_eigenvectors=keep_eigenvectors)
    gc.collect()
    assert (buffers[0]() is not None) is keep_eigenvectors
    if keep_eigenvectors:
        modes = diagonalize_dynamical_matrix(np.array(matrix), masses=minimised.get_masses())
        assert np.array_equal(out["eigenvectors"], modes["eigenvectors"])
    else:
        assert out["eigenvectors"] is None
    kept = _run_harmonic(keep_eigenvectors=True)[0]
    other_keys = out.keys() - {"eigenvectors", "wall_time_s"}
    assert out.keys() == kept.keys()
    _assert_same({key: out[key] for key in other_keys}, {key: kept[key] for key in other_keys})


def test_harmonic_vdos_simulation_thermodynamics():
    """No thermodynamics without temperatures; "unit" and "3N" give the same values."""
    assert _run_harmonic()[0]["thermodynamics"] is None
    unit, three_n = (
        _run_harmonic(temperatures_k=HARMONIC_TEMPERATURES_K, normalization=normalization)[0]["thermodynamics"]
        for normalization in ("unit", "3N")
    )
    assert unit.keys() == three_n.keys()
    for key, value in unit.items():
        if isinstance(value, str):
            assert three_n[key] == value
        else:
            np.testing.assert_allclose(three_n[key], value, rtol=1e-12)


def test_harmonic_vdos_simulation_without_positive_mode_skips_thermodynamics():
    """With every kept mode imaginary, the result is still returned, with a warning and no thermodynamics."""
    minimised = _minimised_dimer()
    matrix = -_dynamical_matrix(minimised.get_masses())  # translations stay at 0, every other mode turns imaginary
    with (
        patch(
            "amorphouspy.properties.vibrational.harmonic_simulation._run_lammps_dynamical_matrix",
            autospec=True,
            side_effect=_fake_dynamical_matrix_runner(minimised, matrix),
        ),
        pytest.warns(UserWarning, match="24 imaginary mode"),
        pytest.warns(UserWarning, match="No mode has a positive frequency"),
    ):
        out = harmonic_vdos_simulation(
            _si2o7_dimer(), _potential(), temperatures_k=HARMONIC_TEMPERATURES_K, **GROUP_KWARGS
        )
    assert out["thermodynamics"] is None
    assert out["n_imaginary_excluded"] == len(out["mode_frequencies_thz"])
