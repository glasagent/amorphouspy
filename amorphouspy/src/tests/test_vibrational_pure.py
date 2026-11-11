"""Tests for pure functions in amorphouspy.properties.vibrational."""

import math
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.fft
import scipy.linalg
from amorphouspy.atoms.mass import get_atomic_mass
from amorphouspy.properties.structural.qn import compute_qn
from amorphouspy.properties.vibrational import (
    _common,
    classify_vibrational_groups,
    compute_debye_level,
    compute_dielectric_from_velocities,
    compute_dielectric_function,
    compute_harmonic_vdos,
    compute_ir_oscillator_strengths,
    compute_optical_response,
    compute_partial_vacf,
    compute_partial_vdos,
    compute_participation_ratio,
    compute_reduced_vdos,
    compute_vacf_from_velocities,
    compute_vdos_from_velocities,
    compute_vibrational_thermodynamics,
    convert_frequency,
    diagonalize_dynamical_matrix,
    read_dynamical_matrix,
)
from ase import Atoms
from ase.io import read
from scipy.constants import N_A, Boltzmann, atomic_mass, c, e, epsilon_0, h
from scipy.integrate import quad

DATA_DIR = Path(__file__).parent / "data"
SI_O_BOND = 1.6

UNITS = ["THz", "cm-1", "meV", "rad/ps"]

# ---------------------------------------------------------------------------
# convert_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("unit", "expected"), [("cm-1", 33.35641), ("meV", 4.135668), ("rad/ps", 6.283185)])
def test_convert_frequency_reference_values(unit, expected):
    """1 THz matches the reference value in each unit."""
    assert convert_frequency(1.0, "THz", unit) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("from_unit", "to_unit"), [("THz", unit) for unit in UNITS[1:]])
def test_convert_frequency_round_trip(from_unit, to_unit):
    """Converting from THz to each other unit and back returns the input."""
    values = np.array([-3.7, 0.0, 0.25, 12.5, 40.0])
    back = convert_frequency(convert_frequency(values, from_unit, to_unit), to_unit, from_unit)
    np.testing.assert_allclose(back, values, rtol=1e-12)
    scalar_back = convert_frequency(convert_frequency(12.5, from_unit, to_unit), to_unit, from_unit)
    assert scalar_back == pytest.approx(12.5, rel=1e-12)


def test_convert_frequency_array_keeps_shape_and_float64():
    """Array input returns a float64 array of the same shape."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = convert_frequency(values, "THz", "meV")
    assert result.shape == (2, 3)
    assert result.dtype == np.float64


def test_convert_frequency_integer_input():
    """Integer scalar returns float; integer array returns float64."""
    scalar = convert_frequency(2, "THz", "cm-1")
    assert type(scalar) is float
    array = convert_frequency(np.array([1, 2, 3]), "THz", "cm-1")
    assert array.dtype == np.float64


def test_convert_frequency_negative_sign_preserving():
    """Negative frequencies (imaginary modes) keep their sign."""
    assert convert_frequency(-1.0, "THz", "cm-1") == pytest.approx(-33.35641, rel=1e-6)
    result = convert_frequency(np.array([-2.0, 2.0]), "THz", "meV")
    assert result[0] == pytest.approx(-result[1])
    assert result[0] < 0


def test_convert_frequency_input_unchanged():
    """The input array is not modified in place."""
    values = np.array([-1.0, 1.0, 5.0])
    original = values.copy()
    convert_frequency(values, "THz", "rad/ps")
    np.testing.assert_array_equal(values, original)


def test_convert_frequency_same_unit_returns_copy():
    """from_unit == to_unit returns an equal array that is not the input object."""
    values = np.array([1.0, 2.0])
    result = convert_frequency(values, "meV", "meV")
    np.testing.assert_array_equal(result, values)
    assert result is not values
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize(("from_unit", "to_unit"), [("Hz", "THz"), ("THz", "cm^-1")])
def test_convert_frequency_unknown_unit_raises(from_unit, to_unit):
    """An unknown unit raises ValueError naming the allowed units."""
    with pytest.raises(ValueError, match="allowed units"):
        convert_frequency(1.0, from_unit, to_unit)


# ---------------------------------------------------------------------------
# classify_vibrational_groups
# ---------------------------------------------------------------------------


def _si2o7_dimer() -> Atoms:
    """Two Si sharing one bridging O (position 2), each with three terminal O, in a large box."""
    si1 = np.array([10.0, 10.0, 10.0])
    si2 = si1 + np.array([2 * SI_O_BOND, 0.0, 0.0])
    bridging = si1 + np.array([SI_O_BOND, 0.0, 0.0])
    terminals_1 = [si1 + d for d in ([-SI_O_BOND, 0, 0], [0, SI_O_BOND, 0], [0, 0, SI_O_BOND])]
    terminals_2 = [si2 + d for d in ([SI_O_BOND, 0, 0], [0, SI_O_BOND, 0], [0, 0, SI_O_BOND])]
    positions = [si1, si2, bridging, *terminals_1, *terminals_2]
    return Atoms("Si2O7", positions=positions, cell=[30.0, 30.0, 30.0], pbc=True)


def _assert_partition(groups: dict[str, np.ndarray], expected: np.ndarray) -> None:
    """The groups are sorted int64 arrays that together hold every expected position exactly once."""
    for idx in groups.values():
        assert idx.dtype == np.int64
        assert len(idx) > 0
        np.testing.assert_array_equal(idx, np.sort(idx))
    combined = np.concatenate(list(groups.values())) if groups else np.array([], dtype=np.int64)
    np.testing.assert_array_equal(np.sort(combined), np.sort(expected))


def _assert_all_partitions(atoms: Atoms, groups: dict[str, dict[str, np.ndarray]], former_types: list[int]) -> None:
    numbers = atoms.get_atomic_numbers()
    _assert_partition(groups["element"], np.arange(len(atoms)))
    _assert_partition(groups["oxygen"], np.flatnonzero(numbers == 8))
    _assert_partition(groups["qn"], np.flatnonzero(np.isin(numbers, former_types)))


def test_classify_vibrational_groups_si2o7_dimer():
    """Isolated Si2O7: one BO, six NBO, both Si are Q1."""
    groups = classify_vibrational_groups(_si2o7_dimer(), cutoff=2.0, former_types=[14])
    assert set(groups["element"]) == {"Si", "O"}
    np.testing.assert_array_equal(groups["element"]["Si"], [0, 1])
    np.testing.assert_array_equal(groups["element"]["O"], np.arange(2, 9))
    assert set(groups["oxygen"]) == {"O_BO", "O_NBO"}
    np.testing.assert_array_equal(groups["oxygen"]["O_BO"], [2])
    np.testing.assert_array_equal(groups["oxygen"]["O_NBO"], [3, 4, 5, 6, 7, 8])
    assert set(groups["qn"]) == {"Si_Q1"}
    np.testing.assert_array_equal(groups["qn"]["Si_Q1"], [0, 1])
    _assert_all_partitions(_si2o7_dimer(), groups, [14])


def test_classify_vibrational_groups_sio2_glass_defaults_match_compute_qn():
    """Default cutoffs and formers partition the glass and reproduce compute_qn's totals."""
    atoms = read(DATA_DIR / "SiO2_glass_300_atoms.xyz")
    groups = classify_vibrational_groups(atoms)
    _assert_all_partitions(atoms, groups, [14])
    assert set(groups["element"]) == {"Si", "O"}

    # Same cutoff as the default derivation (first Si-O RDF minimum), checked for consistency below.
    explicit = classify_vibrational_groups(atoms, cutoff={(14, 8): 1.816}, former_types=[14])
    for key in ("oxygen", "qn"):
        assert explicit[key].keys() == groups[key].keys()
        for label in groups[key]:
            np.testing.assert_array_equal(explicit[key][label], groups[key][label])

    total_qn, _partial_qn = compute_qn(atoms, cutoff={(14, 8): 1.816}, former_types=[14], o_type=8)
    sizes = {n: len(groups["qn"].get(f"Si_Q{n}", [])) for n in total_qn}
    assert sizes == {n: int(count) for n, count in total_qn.items()}


def test_classify_vibrational_groups_shuffled_ids_give_positions():
    """A shuffled, non-contiguous "id" array still yields positions, not IDs."""
    atoms = _si2o7_dimer()
    reference = classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])
    rng = np.random.default_rng(42)
    atoms.arrays["id"] = rng.permutation(np.arange(1, len(atoms) + 1)) * 7 + 100
    groups = classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])
    for key in ("element", "oxygen", "qn"):
        assert groups[key].keys() == reference[key].keys()
        for label in reference[key]:
            np.testing.assert_array_equal(groups[key][label], reference[key][label])


def test_classify_vibrational_groups_duplicate_ids_raise():
    """Duplicate IDs cannot be mapped to positions."""
    atoms = _si2o7_dimer()
    atoms.arrays["id"] = np.ones(len(atoms), dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])


def test_classify_vibrational_groups_no_oxygen():
    """Without oxygen only the element grouping is filled."""
    atoms = Atoms("Si2Na", positions=[[1, 1, 1], [4, 4, 4], [7, 7, 7]], cell=[20, 20, 20], pbc=True)
    groups = classify_vibrational_groups(atoms)
    assert groups["oxygen"] == {}
    assert groups["qn"] == {}
    _assert_partition(groups["element"], np.arange(3))


def test_classify_vibrational_groups_no_formers():
    """With oxygen but no formers, qn is empty and every O is free."""
    atoms = Atoms("Na2O", positions=[[5, 5, 5], [9, 5, 5], [7, 5, 5]], cell=[20, 20, 20], pbc=True)
    groups = classify_vibrational_groups(atoms)
    assert groups["qn"] == {}
    assert set(groups["oxygen"]) == {"O_free"}
    np.testing.assert_array_equal(groups["oxygen"]["O_free"], [2])
    _assert_partition(groups["element"], np.arange(3))


# ---------------------------------------------------------------------------
# compute_vdos_from_velocities and compute_partial_vdos
# ---------------------------------------------------------------------------

DT_FS = 5.0
M_SI = get_atomic_mass("Si")
M_O = get_atomic_mass("O")


def _pair_velocities(n_frames: int, f0_thz: float, amplitude: float = 1.0) -> np.ndarray:
    """Two atoms oscillating with opposite x velocities, so the pair's centre of mass is at rest."""
    t_ps = np.arange(n_frames) * DT_FS * 1e-3
    velocities = np.zeros((n_frames, 2, 3))
    velocities[:, 0, 0] = amplitude * np.cos(2 * np.pi * f0_thz * t_ps)
    velocities[:, 1, 0] = -velocities[:, 0, 0]
    return velocities


def _grid_frequency(n_frames: int, k: int) -> float:
    """Frequency of bin k on the rfft grid, in THz."""
    return k / (n_frames * DT_FS * 1e-3)


def _random_system(n_frames: int, n_atoms: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_frames, n_atoms, 3)), rng.uniform(1.0, 30.0, size=n_atoms)


def _count_calls(monkeypatch, module, name: str) -> list:
    """Wrap ``module.name`` so that the positional arguments of every call are recorded; return the record."""
    calls = []
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


def _centred_kinetic_sum(velocities: np.ndarray, masses: np.ndarray) -> float:
    """Time average of sum_i m_i |v_i - v_com|^2, computed directly."""
    com = np.einsum("tia,i->ta", velocities, masses) / masses.sum()
    centred = velocities - com[:, None, :]
    return float(np.einsum("tia,tia,i->", centred, centred, masses)) / velocities.shape[0]


@pytest.mark.parametrize("n_frames", [400, 401])
def test_vdos_single_pair_peak_and_normalisation(n_frames):
    """A cosine on the frequency grid peaks at its bin and the VDOS integrates to 1."""
    k = 60
    f0 = _grid_frequency(n_frames, k)
    freqs, vdos = compute_vdos_from_velocities(_pair_velocities(n_frames, f0), np.full(2, M_SI), DT_FS)
    assert np.argmax(vdos) == k
    assert freqs[k] == pytest.approx(f0, rel=1e-12)
    assert np.trapezoid(vdos, freqs) == pytest.approx(1.0, rel=1e-12)


def test_vdos_mass_weighting():
    """Two pairs with equal amplitudes give peak heights in the ratio of their masses."""
    n_frames, k_si, k_o = 400, 40, 90
    velocities = np.concatenate(
        [
            _pair_velocities(n_frames, _grid_frequency(n_frames, k_si)),
            _pair_velocities(n_frames, _grid_frequency(n_frames, k_o)),
        ],
        axis=1,
    )
    masses = np.array([M_SI, M_SI, M_O, M_O])
    _, vdos = compute_vdos_from_velocities(velocities, masses, DT_FS)
    assert vdos[k_si] / vdos[k_o] == pytest.approx(M_SI / M_O, rel=1e-10)


def test_vdos_scale_invariant():
    """Scaling all velocities by a constant leaves the result unchanged."""
    velocities = np.random.default_rng(0).normal(size=(64, 5, 3))
    masses = np.array([M_SI, M_O, M_O, M_SI, M_O])
    freqs, vdos = compute_vdos_from_velocities(velocities, masses, DT_FS)
    freqs_scaled, vdos_scaled = compute_vdos_from_velocities(1e5 * velocities, masses, DT_FS)
    np.testing.assert_array_equal(freqs_scaled, freqs)
    np.testing.assert_allclose(vdos_scaled, vdos, rtol=1e-12)


def test_vdos_uniform_translation_removed():
    """A uniform translation added to an oscillation leaves the oscillation's spectrum unchanged."""
    n_frames = 400
    oscillation = _pair_velocities(n_frames, _grid_frequency(n_frames, 50))
    masses = np.array([M_SI, M_SI])
    _, vdos = compute_vdos_from_velocities(oscillation, masses, DT_FS)
    _, vdos_drift = compute_vdos_from_velocities(oscillation + np.array([0.3, -0.2, 0.7]), masses, DT_FS)
    np.testing.assert_allclose(vdos_drift, vdos, rtol=0, atol=1e-12 * vdos.max())


def test_vdos_without_com_removal_keeps_translation():
    """With remove_com=False a uniform drift shows up at zero frequency and a pure translation is allowed."""
    n_frames = 400
    oscillation = _pair_velocities(n_frames, _grid_frequency(n_frames, 50))
    masses = np.array([M_SI, M_SI])
    _, vdos = compute_vdos_from_velocities(oscillation, masses, DT_FS, remove_com=False)
    _, vdos_drift = compute_vdos_from_velocities(
        oscillation + np.array([0.3, 0.0, 0.0]), masses, DT_FS, remove_com=False
    )
    assert vdos[0] == pytest.approx(0.0, abs=1e-12 * vdos.max())
    assert vdos_drift[0] > 0.1 * vdos_drift.max()
    translation = np.broadcast_to(np.array([0.1, 0.2, 0.3]), (n_frames, 2, 3))
    _, vdos_translation = compute_vdos_from_velocities(translation, masses, DT_FS, remove_com=False)
    assert np.argmax(vdos_translation) == 0


@pytest.mark.parametrize("n_frames", [64, 65])
def test_vdos_frequency_axis_and_dtype(n_frames):
    """Frequencies equal rfftfreq in THz; both outputs are float64 with length n_frames // 2 + 1."""
    velocities = np.random.default_rng(1).normal(size=(n_frames, 4, 3)).astype(np.float32)
    freqs, vdos = compute_vdos_from_velocities(velocities, np.full(4, M_O), DT_FS)
    np.testing.assert_array_equal(freqs, np.fft.rfftfreq(n_frames, d=DT_FS * 1e-3))
    assert freqs.shape == vdos.shape == (n_frames // 2 + 1,)
    assert freqs.dtype == vdos.dtype == np.float64


@pytest.mark.parametrize("n_frames", [64, 65])
@pytest.mark.parametrize("remove_com", [True, False])
def test_vdos_normalisations(n_frames, remove_com):
    """'unit' integrates to 1, '3N' to 3 n_atoms, and 'none' to the time-averaged sum of m v^2 (even n_frames)."""
    velocities, masses = _random_system(n_frames, 7, seed=5)
    freqs, unit = compute_vdos_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    _, dof = compute_vdos_from_velocities(velocities, masses, DT_FS, normalization="3N", remove_com=remove_com)
    _, psd = compute_vdos_from_velocities(velocities, masses, DT_FS, normalization="none", remove_com=remove_com)
    assert np.trapezoid(unit, freqs) == pytest.approx(1.0, rel=1e-12)
    assert np.trapezoid(dof, freqs) == pytest.approx(21.0, rel=1e-12)
    np.testing.assert_allclose(dof, 21.0 * unit, rtol=1e-12)
    np.testing.assert_allclose(psd / np.trapezoid(psd, freqs), unit, rtol=1e-12)
    if remove_com:
        kinetic = _centred_kinetic_sum(velocities, masses)
    else:
        kinetic = float(np.einsum("tia,tia,i->", velocities, velocities, masses)) / n_frames
    if n_frames % 2 == 0:
        assert np.trapezoid(psd, freqs) == pytest.approx(kinetic, rel=1e-12)
    else:
        # The trapezoid rule counts the last (non-Nyquist) bin with half weight.
        last_bin = 0.5 * psd[-1] * freqs[1]
        assert np.trapezoid(psd, freqs) + last_bin == pytest.approx(kinetic, rel=1e-12)


def test_vdos_none_is_independent_of_run_length():
    """The 'none' spectral density of a stationary cosine does not depend on the number of frames."""
    f0 = _grid_frequency(400, 40)
    freqs_short, short = compute_vdos_from_velocities(
        _pair_velocities(400, f0), np.full(2, M_SI), DT_FS, normalization="none"
    )
    freqs_long, long = compute_vdos_from_velocities(
        _pair_velocities(800, f0), np.full(2, M_SI), DT_FS, normalization="none"
    )
    assert np.trapezoid(short, freqs_short) == pytest.approx(np.trapezoid(long, freqs_long), rel=1e-12)
    assert np.trapezoid(short, freqs_short) == pytest.approx(M_SI, rel=1e-12)  # 2 atoms * m * <cos^2> = m


def test_vdos_float32_input_close_to_float64():
    """float32 input is transformed in single precision and agrees with float64 input to single precision."""
    velocities, masses = _random_system(200, 6, seed=6)
    freqs, vdos64 = compute_vdos_from_velocities(velocities, masses, DT_FS)
    _, vdos32 = compute_vdos_from_velocities(velocities.astype(np.float32), masses, DT_FS)
    assert vdos32.dtype == np.float64
    np.testing.assert_allclose(vdos32, vdos64, rtol=1e-4, atol=1e-6 * vdos64.max())
    assert np.trapezoid(vdos32, freqs) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("remove_com", [True, False])
def test_vdos_integer_input_matches_float64(remove_com):
    """Integer velocities are transformed in double precision and match the same values given as float64."""
    velocities = np.random.default_rng(13).integers(-5, 6, size=(40, 4, 3))
    masses = np.array([M_SI, M_O, M_O, M_SI])
    _, from_int = compute_vdos_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    _, from_float = compute_vdos_from_velocities(velocities.astype(np.float64), masses, DT_FS, remove_com=remove_com)
    np.testing.assert_allclose(from_int, from_float, rtol=1e-12)


@pytest.mark.parametrize("window", [None, "hann"])
@pytest.mark.parametrize("normalization", ["unit", "3N", "none"])
def test_partial_vdos_sums_to_total(window, normalization):
    """Partials over a partition of the atoms add up to the total, via compute_partial_vdos and via indices."""
    velocities, masses = _random_system(65, 9, seed=7)
    groups = {"element": {"A": np.array([0, 4, 8]), "B": np.array([1, 2, 3]), "C": np.array([5, 6, 7])}}
    kwargs = {"normalization": normalization, "window": window}
    _, total = compute_vdos_from_velocities(velocities, masses, DT_FS, **kwargs)
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups, **kwargs)
    from_indices = [
        compute_vdos_from_velocities(velocities, masses, DT_FS, indices=indices, **kwargs)[1]
        for indices in groups["element"].values()
    ]
    np.testing.assert_allclose(sum(partial["element"].values()), total, rtol=1e-12, atol=1e-14 * total.max())
    np.testing.assert_allclose(sum(from_indices), total, rtol=1e-12, atol=1e-14 * total.max())


def test_partial_vdos_3n_integral_is_kinetic_share():
    """With '3N' a partial integrates to 3 N times the group's share of the internal kinetic energy."""
    velocities, masses = _random_system(128, 6, seed=8)
    indices = np.array([1, 4])
    freqs, partial = compute_vdos_from_velocities(velocities, masses, DT_FS, indices=indices, normalization="3N")
    com = np.einsum("tia,i->ta", velocities, masses) / masses.sum()
    centred = velocities - com[:, None, :]
    share = np.einsum("tia,tia,i->", centred[:, indices], centred[:, indices], masses[indices]) / np.einsum(
        "tia,tia,i->", centred, centred, masses
    )
    assert np.trapezoid(partial, freqs) == pytest.approx(18.0 * share, rel=1e-12)


def test_partial_vdos_matches_indices_and_all_atoms_matches_total():
    """compute_partial_vdos equals compute_vdos_from_velocities with indices; all atoms as indices give the total."""
    velocities, masses = _random_system(80, 5, seed=9)
    groups = {"g1": {"x": np.array([3, 0])}, "g2": {"y": np.array([2]), "all": np.arange(5)}}
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups, normalization="3N")
    for grouping, labels in groups.items():
        for label, indices in labels.items():
            _, single = compute_vdos_from_velocities(velocities, masses, DT_FS, indices=indices, normalization="3N")
            np.testing.assert_allclose(partial[grouping][label], single, rtol=1e-12)
    _, total = compute_vdos_from_velocities(velocities, masses, DT_FS, normalization="3N")
    np.testing.assert_allclose(partial["g2"]["all"], total, rtol=1e-12)
    assert list(partial) == ["g1", "g2"]
    assert list(partial["g2"]) == ["y", "all"]


def test_partial_vdos_from_classified_groups():
    """The output of classify_vibrational_groups can be passed directly; element partials sum to the total."""
    atoms = _si2o7_dimer()
    groups = classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])
    velocities = np.random.default_rng(10).normal(size=(60, len(atoms), 3))
    masses = atoms.get_masses()
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups)
    _, total = compute_vdos_from_velocities(velocities, masses, DT_FS)
    np.testing.assert_allclose(sum(partial["element"].values()), total, rtol=1e-12)
    np.testing.assert_allclose(sum(partial["oxygen"].values()), partial["element"]["O"], rtol=1e-12)
    np.testing.assert_allclose(sum(partial["qn"].values()), partial["element"]["Si"], rtol=1e-12)


def test_vdos_memmap_input(tmp_path):
    """A read-only memory-mapped trajectory gives the same result as the in-memory array."""
    velocities, masses = _random_system(40, 6, seed=11)
    np.save(tmp_path / "v.npy", velocities)
    mapped = np.load(tmp_path / "v.npy", mmap_mode="r")
    _, expected = compute_vdos_from_velocities(velocities, masses, DT_FS, indices=np.array([0, 2, 5]))
    _, vdos = compute_vdos_from_velocities(mapped, masses, DT_FS, indices=np.array([0, 2, 5]))
    np.testing.assert_array_equal(vdos, expected)


def _vdos_invalid_cases() -> list:
    good = np.random.default_rng(3).normal(size=(10, 2, 3))
    masses = np.array([M_SI, M_O])
    nan_velocities = good.copy()
    nan_velocities[4, 1, 2] = np.nan
    translation = np.broadcast_to(np.array([0.1, 0.2, 0.3]), (10, 2, 3))
    first_frame_only = np.zeros((10, 2, 3))  # the periodic Hann window is exactly 0 at frame 0
    first_frame_only[0] = good[0]
    return [
        pytest.param(good[:, :, 0], masses, DT_FS, {}, "shape", id="2d-velocities"),
        pytest.param(good[:, :, :2], masses, DT_FS, {}, "shape", id="two-components"),
        pytest.param(np.zeros((10, 0, 3)), np.zeros(0), DT_FS, {}, "n_atoms >= 1", id="zero-atoms"),
        pytest.param(good, np.array([M_SI, M_O, M_O]), DT_FS, {}, "masses must have shape", id="masses-length"),
        pytest.param(good[:1], masses, DT_FS, {}, "At least 2 frames", id="one-frame"),
        pytest.param(good, masses, 0.0, {}, "dt_fs must be positive", id="dt-zero"),
        pytest.param(good, masses, np.nan, {}, "dt_fs must be positive", id="dt-nan"),
        pytest.param(good, masses, None, {}, "dt_fs must be positive", id="dt-none"),
        pytest.param(good, np.array([M_SI, 0.0]), DT_FS, {}, "positive and finite", id="mass-zero"),
        pytest.param(good, np.array([-M_SI, M_O]), DT_FS, {}, "positive and finite", id="mass-negative"),
        pytest.param(nan_velocities, masses, DT_FS, {}, "NaN or inf", id="velocity-nan"),
        pytest.param(good.astype(np.complex128), masses, DT_FS, {}, "real numeric", id="velocity-complex"),
        pytest.param(good.astype(object), masses, DT_FS, {}, "real numeric", id="velocity-object"),
        pytest.param(
            first_frame_only, masses, DT_FS, {"window": "hann"}, "window removes all motion", id="window-removes-motion"
        ),
        pytest.param(translation, masses, DT_FS, {}, "zero after removing", id="uniform-translation"),
        pytest.param(np.zeros((10, 2, 3)), masses, DT_FS, {}, "all velocities are zero", id="all-zero"),
        pytest.param(good, masses, DT_FS, {"normalization": "per-atom"}, "Unknown normalization", id="normalization"),
        pytest.param(good, masses, DT_FS, {"window": "kaiser"}, "Unknown window", id="window"),
        pytest.param(good, masses, DT_FS, {"window": "gaussian"}, "Unknown window", id="window-gaussian-no-std"),
        pytest.param(good, masses, DT_FS, {"window": ("gaussian",)}, "Unknown window", id="window-gaussian-short"),
        pytest.param(good, masses, DT_FS, {"window": ("gaussian", 0.0)}, "Unknown window", id="window-gaussian-zero"),
        pytest.param(good, masses, DT_FS, {"window": ("gaussian", np.nan)}, "Unknown window", id="window-gaussian-nan"),
        pytest.param(good, masses, DT_FS, {"window": ("gaussian", True)}, "Unknown window", id="window-gaussian-bool"),
        pytest.param(good, masses, DT_FS, {"window": ("hann", 3.0)}, "Unknown window", id="window-hann-tuple"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([], dtype=int)}, "non-empty", id="indices-empty"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([[0]])}, "1-D", id="indices-2d"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([0.0])}, "integer", id="indices-float"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([2])}, "outside", id="indices-range"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([-1])}, "outside", id="indices-negative"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([1, 1])}, "duplicate", id="indices-duplicate"),
    ]


@pytest.mark.parametrize(("velocities", "masses", "dt_fs", "kwargs", "match"), _vdos_invalid_cases())
def test_vdos_invalid_input_raises(velocities, masses, dt_fs, kwargs, match):
    """Every invalid input raises ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        compute_vdos_from_velocities(velocities, masses, dt_fs, **kwargs)


def test_partial_vdos_invalid_groups_raise():
    """Empty groups and invalid group indices raise ValueError naming the group."""
    velocities, masses = _random_system(10, 3, seed=12)
    with pytest.raises(ValueError, match="no labels"):
        compute_partial_vdos(velocities, masses, DT_FS, {"element": {}})
    with pytest.raises(ValueError, match=r"groups\['element'\]\['O'\] contains positions outside"):
        compute_partial_vdos(velocities, masses, DT_FS, {"element": {"O": np.array([3])}})


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("window", [None, "hann"])
def test_vdos_input_unchanged(dtype, window):
    """The inputs are not modified, also with remove_com=False, where a contiguous selection is a view of the input."""
    velocities = np.random.default_rng(4).normal(size=(32, 4, 3)).astype(dtype)
    masses = np.array([M_SI, M_O, M_O, M_SI])
    velocities_before, masses_before = velocities.copy(), masses.copy()
    groups = {"g": {"a": np.array([0, 1]), "b": np.array([2, 3])}}
    for remove_com in (True, False):
        kwargs = {"remove_com": remove_com, "window": window}
        compute_vdos_from_velocities(velocities, masses, DT_FS, **kwargs)
        compute_vdos_from_velocities(velocities, masses, DT_FS, indices=np.array([1, 2]), **kwargs)
        compute_vdos_from_velocities(velocities, masses, DT_FS, indices=np.array([3, 0]), **kwargs)
        compute_partial_vdos(velocities, masses, DT_FS, groups, **kwargs)
    np.testing.assert_array_equal(velocities, velocities_before)
    np.testing.assert_array_equal(masses, masses_before)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

# Cosine-sum coefficients a_j of w_t = sum_j (-1)^j a_j cos(2 pi j t / N).
COSINE_SUM_COEFFICIENTS = {
    "hann": [0.5, 0.5],
    "hamming": [0.54, 0.46],
    "blackmanharris": [0.35875, 0.48829, 0.14128, 0.01168],
}


@pytest.mark.parametrize("window", list(COSINE_SUM_COEFFICIENTS))
def test_vdos_cosine_sum_window_on_grid_cosine(window):
    """An on-grid cosine spreads over bins k +- j with power (a_j / (2 a_0))^2 of the peak, and nothing else."""
    n_frames, k = 400, 60
    _, vdos = compute_vdos_from_velocities(
        _pair_velocities(n_frames, _grid_frequency(n_frames, k)), np.full(2, M_SI), DT_FS, window=window
    )
    coefficients = COSINE_SUM_COEFFICIENTS[window]
    assert np.argmax(vdos) == k
    for j, a in enumerate(coefficients[1:], start=1):
        expected = (a / (2 * coefficients[0])) ** 2
        assert vdos[k - j] / vdos[k] == pytest.approx(expected, rel=1e-10)
        assert vdos[k + j] / vdos[k] == pytest.approx(expected, rel=1e-10)
    spread = np.arange(k - len(coefficients) + 1, k + len(coefficients))
    assert np.abs(np.delete(vdos, spread)).max() < 1e-20 * vdos[k]


@pytest.mark.parametrize(
    ("window", "bound"),
    [("hann", 1e-6), ("hamming", 1e-4), ("blackmanharris", 1e-10), (("gaussian", 50.0), 1e-8)],
)
def test_vdos_window_suppresses_leakage(window, bound):
    """A cosine half-way between two bins leaks 20 bins away without a window; each window reduces that leakage."""
    n_frames, k = 400, 60
    velocities = _pair_velocities(n_frames, _grid_frequency(n_frames, k + 0.5))
    masses = np.full(2, M_SI)
    _, plain = compute_vdos_from_velocities(velocities, masses, DT_FS)
    _, windowed = compute_vdos_from_velocities(velocities, masses, DT_FS, window=window)
    assert plain[k + 20] / plain.max() > 1e-4
    assert windowed[k + 20] / windowed.max() < bound


@pytest.mark.parametrize("n_frames", [64, 65])
@pytest.mark.parametrize("remove_com", [True, False])
def test_vdos_window_normalisations(n_frames, remove_com):
    """With a window, 'unit' integrates to 1 and '3N' to 3 n_atoms, for even and odd n_frames."""
    velocities, masses = _random_system(n_frames, 7, seed=22)
    kwargs = {"remove_com": remove_com, "window": "hann"}
    freqs, unit = compute_vdos_from_velocities(velocities, masses, DT_FS, **kwargs)
    _, dof = compute_vdos_from_velocities(velocities, masses, DT_FS, normalization="3N", **kwargs)
    assert np.trapezoid(unit, freqs) == pytest.approx(1.0, rel=1e-12)
    assert np.trapezoid(dof, freqs) == pytest.approx(21.0, rel=1e-12)


@pytest.mark.parametrize("remove_com", [True, False])
def test_vdos_window_none_integral_is_weighted_kinetic_average(remove_com):
    """With a window, the 'none' integral is the w^2-weighted time average of sum_i m_i |v_i|^2 (even n_frames)."""
    n_frames, std = 64, 8.0
    velocities, masses = _random_system(n_frames, 7, seed=23)
    freqs, psd = compute_vdos_from_velocities(
        velocities, masses, DT_FS, normalization="none", remove_com=remove_com, window=("gaussian", std)
    )
    if remove_com:
        velocities = velocities - (np.einsum("tia,i->ta", velocities, masses) / masses.sum())[:, None, :]
    kinetic = np.einsum("tia,tia,i->t", velocities, velocities, masses)
    # The Gaussian weights written out; the cosine-sum windows are pinned by the on-grid cosine test.
    window_sq = np.exp(-0.5 * ((np.arange(n_frames) - n_frames / 2) / std) ** 2) ** 2
    assert np.trapezoid(psd, freqs) == pytest.approx(window_sq @ kinetic / window_sq.sum(), rel=1e-12)


@pytest.mark.parametrize("window", [None, "hann"])
@pytest.mark.parametrize("chunk_bytes", [1, 5000])
@pytest.mark.parametrize("n_frames", [50, 51])
def test_vdos_chunking_matches_single_chunk(monkeypatch, window, chunk_bytes, n_frames):
    """Total and partial results do not depend on how atoms are split into chunks, with and without a window."""
    velocities, masses = _random_system(n_frames, 7, seed=27)
    groups = {"a": {"odd": np.array([1, 3, 5]), "first": np.array([0, 1, 2])}}
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 2**40)
    _, reference = compute_vdos_from_velocities(velocities, masses, DT_FS, window=window)
    _, reference_partial = compute_partial_vdos(velocities, masses, DT_FS, groups, window=window)
    monkeypatch.setattr(_common, "_CHUNK_BYTES", chunk_bytes)
    transforms = _count_calls(monkeypatch, scipy.fft, "rfft")
    _, vdos = compute_vdos_from_velocities(velocities, masses, DT_FS, window=window)
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups, window=window)
    # One transform per chunk: the 7 atoms of the total, then the 5 atoms in the groups.
    atoms_per_chunk = max(1, chunk_bytes // (n_frames * 3 * velocities.itemsize))
    assert len(transforms) == math.ceil(7 / atoms_per_chunk) + math.ceil(5 / atoms_per_chunk)
    np.testing.assert_allclose(vdos, reference, rtol=1e-12)
    for label in groups["a"]:
        np.testing.assert_allclose(partial["a"][label], reference_partial["a"][label], rtol=1e-12)


# ---------------------------------------------------------------------------
# compute_vacf_from_velocities and compute_partial_vacf
# ---------------------------------------------------------------------------


def _brute_force_vacf(
    velocities: np.ndarray, masses: np.ndarray, indices: np.ndarray | None, *, remove_com: bool, normalization: str
) -> np.ndarray:
    """Direct double loop over lags and time origins (no FFT); 'unit' divides by the total lag-0 value."""
    if remove_com:
        velocities = velocities - (np.einsum("tia,i->ta", velocities, masses) / masses.sum())[:, None, :]
    n_frames = velocities.shape[0]
    selected = np.arange(velocities.shape[1]) if indices is None else indices
    vacf = np.zeros(n_frames)
    for k in range(n_frames):
        for t in range(n_frames - k):
            vacf[k] += np.sum(masses[selected, None] * velocities[t, selected] * velocities[t + k, selected])
        vacf[k] /= n_frames - k
    if normalization == "unit":
        vacf /= np.einsum("tia,tia,i->", velocities, velocities, masses) / n_frames
    return vacf


@pytest.mark.parametrize("n_frames", [36, 37])
@pytest.mark.parametrize("normalization", ["unit", "none"])
@pytest.mark.parametrize("remove_com", [True, False])
def test_vacf_matches_brute_force(n_frames, normalization, remove_com):
    """Total and partial VACF equal a direct sum over all time origins at every lag (this includes lag 0)."""
    velocities, masses = _random_system(n_frames, 5, seed=30)
    velocities += np.array([0.4, -0.3, 0.2])  # a drift, so that remove_com matters
    kwargs = {"max_lag_frames": n_frames - 1, "normalization": normalization, "remove_com": remove_com}
    scale = abs(_brute_force_vacf(velocities, masses, None, remove_com=remove_com, normalization=normalization)[0])
    for indices in (None, np.array([3, 0])):
        expected = _brute_force_vacf(velocities, masses, indices, remove_com=remove_com, normalization=normalization)
        lag_ps, vacf = compute_vacf_from_velocities(velocities, masses, DT_FS, indices=indices, **kwargs)
        assert vacf.shape == lag_ps.shape == (n_frames,)
        # FFT round-off is absolute (relative to the lag-0 value), hence the atol for lags where C(t) is near zero.
        np.testing.assert_allclose(vacf, expected, rtol=1e-12, atol=1e-14 * scale)


def test_vacf_unit_lag_zero_is_one_and_partial_is_kinetic_share():
    """With 'unit' the total C(0) is 1 and a partial's C(0) is its share of the centred kinetic sum."""
    velocities, masses = _random_system(64, 6, seed=31)
    indices = np.array([5, 0, 2])
    _, total = compute_vacf_from_velocities(velocities, masses, DT_FS)
    _, partial = compute_vacf_from_velocities(velocities, masses, DT_FS, indices=indices)
    com = np.einsum("tia,i->ta", velocities, masses) / masses.sum()
    centred = velocities - com[:, None, :]
    share = np.einsum("tia,tia,i->", centred[:, indices], centred[:, indices], masses[indices]) / np.einsum(
        "tia,tia,i->", centred, centred, masses
    )
    assert total[0] == pytest.approx(1.0, rel=1e-12)
    assert partial[0] == pytest.approx(share, rel=1e-12)


@pytest.mark.parametrize("normalization", ["unit", "none"])
def test_partial_vacf_sums_to_total(normalization):
    """Partials over a partition of the atoms add up to the total, through groups and through indices."""
    velocities, masses = _random_system(64, 9, seed=32)
    groups = {"element": {"A": np.array([0, 4, 8]), "B": np.array([1, 2, 3]), "C": np.array([5, 6, 7])}}
    lag_ps, total = compute_vacf_from_velocities(velocities, masses, DT_FS, normalization=normalization)
    lag_partial, partial = compute_partial_vacf(velocities, masses, DT_FS, groups, normalization=normalization)
    np.testing.assert_array_equal(lag_partial, lag_ps)
    atol = 1e-14 * abs(total[0])
    np.testing.assert_allclose(sum(partial["element"].values()), total, rtol=1e-12, atol=atol)
    for label, indices in groups["element"].items():
        _, single = compute_vacf_from_velocities(
            velocities, masses, DT_FS, indices=indices, normalization=normalization
        )
        np.testing.assert_allclose(partial["element"][label], single, rtol=1e-12, atol=atol)


def test_vacf_uniform_translation_removed():
    """A uniform translation added to every atom leaves the total and partial VACF unchanged."""
    velocities, masses = _random_system(60, 5, seed=34)
    for indices in (None, np.array([1, 3])):
        _, vacf = compute_vacf_from_velocities(velocities, masses, DT_FS, indices=indices)
        _, vacf_drift = compute_vacf_from_velocities(
            velocities + np.array([0.3, -0.2, 0.7]), masses, DT_FS, indices=indices
        )
        np.testing.assert_allclose(vacf_drift, vacf, rtol=0, atol=1e-12 * abs(vacf[0]))


def test_vacf_single_pair_follows_cosine():
    """An opposite-velocity pair oscillating at f0 gives C(t) = cos(2 pi f0 t) within the estimator's finite-N error."""
    n_frames, f0 = 2000, 7.37
    lag_ps, vacf = compute_vacf_from_velocities(_pair_velocities(n_frames, f0), np.full(2, M_SI), DT_FS)
    # The unbiased estimate of <cos(w t) cos(w (t + k))> is cos(w k) / 2 + E_k / (2 (N - k)), with
    # E_k = sum_{t < N - k} cos(w (2 t + k)) a geometric sum bounded by b = 1 / |sin w| (w: phase per frame).
    # Dividing by the lag-0 value 1/2 + E_0 / (2 N) gives |C(k) - cos(w k)| <= (b / (N - k) + b / N) / (1 - b / N),
    # at most about 0.007 here. The estimator is exact only for an infinite trajectory.
    omega = 2 * np.pi * f0 * DT_FS * 1e-3
    b = 1 / abs(np.sin(omega))
    k = np.arange(vacf.size)
    bound = (b / (n_frames - k) + b / n_frames) / (1 - b / n_frames)
    deviation = np.abs(vacf - np.cos(2 * np.pi * f0 * lag_ps))
    assert np.all(deviation <= bound + 1e-12)
    assert deviation.max() < 0.01


def test_vacf_max_lag_frames():
    """Lags run to n_frames // 2 by default; a shorter max_lag_frames returns the first lags of the full result."""
    velocities, masses = _random_system(37, 4, seed=35)
    lag_ps, vacf = compute_vacf_from_velocities(velocities, masses, DT_FS)
    assert lag_ps.shape == vacf.shape == (19,)
    assert lag_ps.dtype == vacf.dtype == np.float64
    np.testing.assert_array_equal(lag_ps, np.arange(19) * DT_FS * 1e-3)
    _, full = compute_vacf_from_velocities(velocities, masses, DT_FS, max_lag_frames=36)
    _, short = compute_vacf_from_velocities(velocities, masses, DT_FS, max_lag_frames=np.int64(10))
    np.testing.assert_array_equal(short, full[:11])


def test_vacf_transform_matches_vdos_psd():
    """The 'none' VDOS is the triangle-weighted cosine transform of the 'none' VACF over all lags."""
    n_frames = 37
    velocities, masses = _random_system(n_frames, 5, seed=37)
    freqs, psd = compute_vdos_from_velocities(velocities, masses, DT_FS, normalization="none")
    _, vacf = compute_vacf_from_velocities(velocities, masses, DT_FS, max_lag_frames=n_frames - 1, normalization="none")
    k = np.arange(1, n_frames)
    cosines = np.cos(2 * np.pi * np.outer(np.arange(freqs.size), k) / n_frames)
    expected = 2 * DT_FS * 1e-3 * (vacf[0] + 2 * cosines @ ((1 - k / n_frames) * vacf[1:]))
    np.testing.assert_allclose(psd, expected, rtol=1e-10, atol=1e-12 * psd.max())


@pytest.mark.parametrize("chunk_bytes", [1, 5000])
@pytest.mark.parametrize("remove_com", [True, False])
def test_vacf_chunking_matches_single_chunk(monkeypatch, chunk_bytes, remove_com):
    """Total and partial VACFs do not depend on how atoms are split into chunks."""
    velocities, masses = _random_system(50, 7, seed=38)
    groups = {"a": {"odd": np.array([1, 3, 5]), "first": np.array([0, 1, 2])}}
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 2**40)
    _, reference = compute_vacf_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    _, reference_partial = compute_partial_vacf(velocities, masses, DT_FS, groups, remove_com=remove_com)
    monkeypatch.setattr(_common, "_CHUNK_BYTES", chunk_bytes)
    transforms = _count_calls(monkeypatch, scipy.fft, "rfft")
    _, vacf = compute_vacf_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    _, partial = compute_partial_vacf(velocities, masses, DT_FS, groups, remove_com=remove_com)
    # One transform per chunk, sized for the 2N = 100 padded frames: the 7 atoms of the total, then the 5 in the groups.
    atoms_per_chunk = max(1, chunk_bytes // (100 * 3 * velocities.itemsize))
    assert len(transforms) == math.ceil(7 / atoms_per_chunk) + math.ceil(5 / atoms_per_chunk)
    np.testing.assert_allclose(vacf, reference, rtol=1e-12, atol=1e-14)
    for label in groups["a"]:
        np.testing.assert_allclose(partial["a"][label], reference_partial["a"][label], rtol=1e-12, atol=1e-14)


def test_vacf_float32_input_close_to_float64():
    """float32 input is transformed in single precision and agrees with float64 input to single precision."""
    velocities, masses = _random_system(200, 6, seed=39)
    _, vacf64 = compute_vacf_from_velocities(velocities, masses, DT_FS)
    _, vacf32 = compute_vacf_from_velocities(velocities.astype(np.float32), masses, DT_FS)
    assert vacf32.dtype == np.float64
    np.testing.assert_allclose(vacf32, vacf64, rtol=0, atol=1e-5)


def test_vacf_memmap_input(tmp_path):
    """A read-only memory-mapped trajectory gives the same result as the in-memory array."""
    velocities, masses = _random_system(40, 6, seed=40)
    np.save(tmp_path / "v.npy", velocities)
    mapped = np.load(tmp_path / "v.npy", mmap_mode="r")
    for indices in (None, np.array([0, 2, 5])):
        _, expected = compute_vacf_from_velocities(velocities, masses, DT_FS, indices=indices)
        _, vacf = compute_vacf_from_velocities(mapped, masses, DT_FS, indices=indices)
        np.testing.assert_array_equal(vacf, expected)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("remove_com", [True, False])
def test_vacf_input_unchanged(dtype, remove_com):
    """The velocity and mass arrays are not modified, for the total, contiguous and scattered indices, and groups."""
    velocities = np.random.default_rng(42).normal(size=(32, 4, 3)).astype(dtype)
    masses = np.array([M_SI, M_O, M_O, M_SI])
    velocities_before, masses_before = velocities.copy(), masses.copy()
    compute_vacf_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    compute_vacf_from_velocities(velocities, masses, DT_FS, indices=np.array([1, 2]), remove_com=remove_com)
    compute_vacf_from_velocities(velocities, masses, DT_FS, indices=np.array([3, 0]), remove_com=remove_com)
    compute_partial_vacf(
        velocities, masses, DT_FS, {"g": {"a": np.array([0, 1]), "b": np.array([2, 3])}}, remove_com=remove_com
    )
    np.testing.assert_array_equal(velocities, velocities_before)
    np.testing.assert_array_equal(masses, masses_before)


# One case per shared validator is enough to show the VACF uses it; the VDOS tests cover each validator in full.
_SHARED_INVALID_IDS = {
    "2d-velocities",
    "masses-length",
    "one-frame",
    "dt-nan",
    "dt-none",
    "mass-negative",
    "velocity-nan",
    "velocity-complex",
    "uniform-translation",
    "all-zero",
    "indices-range",
}


def _vacf_invalid_cases() -> list:
    good = np.random.default_rng(3).normal(size=(10, 2, 3))
    masses = np.array([M_SI, M_O])
    shared = [case for case in _vdos_invalid_cases() if case.id in _SHARED_INVALID_IDS]
    return [
        *shared,
        pytest.param(
            good,
            masses,
            DT_FS,
            {"normalization": "3N"},
            r"Unknown normalization '3N'; allowed values are 'unit' and 'none'",
            id="normalization-3n",
        ),
        pytest.param(good, masses, DT_FS, {"max_lag_frames": 0}, "max_lag_frames", id="max-lag-zero"),
        pytest.param(good, masses, DT_FS, {"max_lag_frames": -1}, "max_lag_frames", id="max-lag-negative"),
        pytest.param(good, masses, DT_FS, {"max_lag_frames": 10}, r"in \[1, 9\]", id="max-lag-n-frames"),
        pytest.param(good, masses, DT_FS, {"max_lag_frames": 2.5}, "max_lag_frames", id="max-lag-float"),
        pytest.param(good, masses, DT_FS, {"max_lag_frames": True}, "max_lag_frames", id="max-lag-bool"),
    ]


@pytest.mark.parametrize(("velocities", "masses", "dt_fs", "kwargs", "match"), _vacf_invalid_cases())
def test_vacf_invalid_input_raises(velocities, masses, dt_fs, kwargs, match):
    """Every invalid input raises ValueError with the same message as for the VDOS."""
    with pytest.raises(ValueError, match=match):
        compute_vacf_from_velocities(velocities, masses, dt_fs, **kwargs)


def test_partial_vacf_invalid_groups_raise():
    """Empty groups and invalid group indices raise ValueError naming the group."""
    velocities, masses = _random_system(10, 3, seed=12)
    with pytest.raises(ValueError, match="no labels"):
        compute_partial_vacf(velocities, masses, DT_FS, {"element": {}})
    with pytest.raises(ValueError, match=r"groups\['element'\]\['O'\] contains positions outside"):
        compute_partial_vacf(velocities, masses, DT_FS, {"element": {"O": np.array([3])}})


# ---------------------------------------------------------------------------
# compute_debye_level and compute_reduced_vdos
# ---------------------------------------------------------------------------

# Approximate literature values for fused silica: B and G in GPa, density in g/cm^3.
SILICA_BULK_GPA, SILICA_SHEAR_GPA, SILICA_DENSITY = 36.7, 31.2, 2.2
# 1.5 THz lies between bin 21 (1.47 THz) and bin 22 (1.54 THz).
GRID_THZ = np.arange(201) * 0.07
# The ends of the default 0.5-5 THz range are exactly bins 8 and 80.
SIXTEENTH_GRID_THZ = np.arange(161) / 16
SHEARED_CELL = np.array([[12.0, 0.0, 0.0], [8.0, 10.0, 0.0], [0.0, 0.0, 11.0]])


def _silica(shape: np.ndarray, n_units: int) -> Atoms:
    """n_units SiO2 at 2.2 g/cm^3 in a cell of the given shape (positions are irrelevant to the Debye level)."""
    atoms = Atoms(f"Si{n_units}O{2 * n_units}", pbc=True)
    volume_a3 = atoms.get_masses().sum() * atomic_mass * 1e3 / SILICA_DENSITY * 1e24
    atoms.set_cell(shape * (volume_a3 / abs(np.linalg.det(shape))) ** (1 / 3))
    return atoms


def _reduced_bump(freqs: np.ndarray) -> np.ndarray:
    """A VDOS whose reduced VDOS is a flat level plus a Gaussian excess centred at 1.5 THz."""
    return freqs**2 * (1.0 + np.exp(-0.5 * ((freqs - 1.5) / 0.5) ** 2))


def test_debye_level_fused_silica():
    """Fused-silica moduli and density give its sound velocities and Debye temperature."""
    atoms = _silica(np.eye(3), 1)
    debye = compute_debye_level(atoms, SILICA_BULK_GPA, SILICA_SHEAR_GPA)
    # Approximate literature values for fused silica.
    assert debye["longitudinal_velocity_m_s"] == pytest.approx(5950.0, rel=0.03)
    assert debye["transverse_velocity_m_s"] == pytest.approx(3770.0, rel=0.03)
    assert debye["debye_temperature_k"] == pytest.approx(495.0, rel=0.03)
    assert debye["density_g_cm3"] == pytest.approx(SILICA_DENSITY, rel=1e-12)
    assert debye["number_density_per_a3"] == pytest.approx(3 / atoms.get_volume(), rel=1e-12)


@pytest.mark.parametrize("normalization", ["unit", "3N"])
def test_reduced_vdos_of_debye_spectrum_is_flat(normalization):
    """A Debye VDOS has a flat reduced VDOS at the Debye level, NaN at nu = 0, and no point is dropped."""
    atoms = _silica(np.eye(3), 1)
    debye = compute_debye_level(atoms, SILICA_BULK_GPA, SILICA_SHEAR_GPA, normalization=normalization)
    nu_d = debye["debye_frequency_thz"]
    scale = 1.0 if normalization == "unit" else 3.0 * len(atoms)
    freqs = np.linspace(0.0, nu_d, 301)
    result = compute_reduced_vdos(freqs, scale * 3 * freqs**2 / nu_d**3, debye=debye)
    reduced = result["reduced_vdos"]
    assert reduced.shape == freqs.shape
    assert np.isnan(reduced[0])
    np.testing.assert_allclose(reduced[1:], scale * 3 / nu_d**3, rtol=1e-12)
    assert result["debye_level_thz-3"] == pytest.approx(scale * 3 / nu_d**3, rel=1e-12)
    assert result["boson_peak_over_debye"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize(
    ("freqs", "vdos", "peak_bin", "at_edge"),
    [
        pytest.param(GRID_THZ, _reduced_bump(GRID_THZ), 21, False, id="bump"),
        pytest.param(SIXTEENTH_GRID_THZ, SIXTEENTH_GRID_THZ, 8, True, id="decreasing"),
        pytest.param(SIXTEENTH_GRID_THZ, SIXTEENTH_GRID_THZ**3, 80, True, id="increasing"),
    ],
)
def test_reduced_vdos_boson_peak(freqs, vdos, peak_bin, at_edge):
    """The peak is the raw maximum within the range, ends included; a monotonic reduced VDOS peaks on a range edge."""
    result = compute_reduced_vdos(freqs, vdos)
    assert result["boson_peak_thz"] == freqs[peak_bin]
    assert result["boson_peak_height"] == result["reduced_vdos"][peak_bin]
    assert result["peak_at_range_edge"] is at_edge
    assert set(result) == {"reduced_vdos", "boson_peak_thz", "boson_peak_height", "peak_at_range_edge"}


@pytest.mark.parametrize(
    ("shape", "n_units", "peak_below"),
    [
        pytest.param(np.eye(3), 27, True, id="small-cubic"),  # L = 10.7 A, nu_min = 3.5 THz
        pytest.param(SHEARED_CELL, 1000, False, id="large-sheared"),  # largest height 35.8 A, nu_min = 1.05 THz
    ],
)
def test_reduced_vdos_box_limit(shape, n_units, peak_below):
    """The lowest box frequency is v_T over the largest perpendicular height; frequencies below it are unreliable."""
    atoms = _silica(shape, n_units)
    # c is perpendicular to a and b, so the height along c is cell[2, 2]. In the sheared cell it is the largest height,
    # while the longest edge (b) and the largest diagonal element (a) are longer.
    l_max = atoms.cell[2, 2]  # in A
    v_t = np.sqrt(SILICA_SHEAR_GPA / SILICA_DENSITY * 1e6)
    debye = compute_debye_level(atoms, SILICA_BULK_GPA, SILICA_SHEAR_GPA)
    assert debye["lowest_box_frequency_thz"] == pytest.approx(v_t / (l_max * 1e-10) * 1e-12, rel=1e-12)
    # The lowest box frequency itself is on the grid and counts as reliable.
    freqs = np.union1d(GRID_THZ, debye["lowest_box_frequency_thz"])
    result = compute_reduced_vdos(freqs, _reduced_bump(freqs), debye=debye)
    np.testing.assert_array_equal(result["reliable"], debye["lowest_box_frequency_thz"] <= freqs)
    assert result["peak_below_box_limit"] is peak_below


def _debye_invalid_cases() -> list:
    silica = _silica(np.eye(3), 1)
    open_cell = silica.copy()
    open_cell.pbc = [True, True, False]
    b, g = SILICA_BULK_GPA, SILICA_SHEAR_GPA
    return [
        pytest.param(silica, 0.0, g, {}, "bulk_modulus_gpa must be positive", id="bulk-zero"),
        pytest.param(silica, b, None, {}, "shear_modulus_gpa must be positive", id="shear-none"),
        pytest.param(Atoms(cell=[10.0, 10.0, 10.0], pbc=True), b, g, {}, "no atoms", id="empty"),
        pytest.param(open_cell, b, g, {}, "periodic in all three", id="non-periodic"),
        pytest.param(Atoms("SiO2", pbc=True), b, g, {}, "positive volume", id="zero-volume"),
        pytest.param(silica, b, g, {"normalization": "none"}, "no Debye level", id="normalization-none"),
    ]


@pytest.mark.parametrize(("structure", "bulk", "shear", "kwargs", "match"), _debye_invalid_cases())
def test_debye_level_invalid_input_raises(structure, bulk, shear, kwargs, match):
    """Invalid moduli, structures and normalisations raise ValueError."""
    with pytest.raises(ValueError, match=match):
        compute_debye_level(structure, bulk, shear, **kwargs)


def _reduced_invalid_cases() -> list:
    vdos = GRID_THZ**2
    nan_vdos = vdos.copy()
    nan_vdos[5] = np.nan
    return [
        pytest.param(GRID_THZ, vdos[:-1], {}, "1-D arrays of the same length", id="length-mismatch"),
        pytest.param(GRID_THZ, nan_vdos, {}, "NaN or inf", id="vdos-nan"),
        pytest.param(GRID_THZ - 0.07, vdos, {}, "negative", id="negative-frequency"),
        pytest.param(GRID_THZ[::-1], vdos, {}, "strictly increasing", id="unsorted"),
        pytest.param(GRID_THZ, vdos, {"peak_range_thz": (2.0, 2.0)}, "low < high", id="range-low-equals-high"),
        pytest.param(GRID_THZ, vdos, {"peak_range_thz": (1.48, 1.53)}, "no frequency bins", id="range-without-bins"),
    ]


@pytest.mark.parametrize(("freqs", "vdos", "kwargs", "match"), _reduced_invalid_cases())
def test_reduced_vdos_invalid_input_raises(freqs, vdos, kwargs, match):
    """Invalid spectra and peak ranges raise ValueError."""
    with pytest.raises(ValueError, match=match):
        compute_reduced_vdos(freqs, vdos, **kwargs)


# ---------------------------------------------------------------------------
# compute_vibrational_thermodynamics
# ---------------------------------------------------------------------------

THZ_TO_K = h * 1e12 / Boltzmann
THZ_TO_MEV = h * 1e12 / e * 1e3
KB_MEV = Boltzmann / e * 1e3
DEBYE_THZ = 10.0
DEBYE_GRID_THZ = np.linspace(0.0, DEBYE_THZ, 4001)
DEBYE_VDOS = 3 * DEBYE_GRID_THZ**2 / DEBYE_THZ**3
# Steps of 0.1 THz from -1 THz, with nu = 0 exactly on the grid.
SIGNED_GRID_THZ = np.arange(-10, 301) * 0.1
THERMO_ARRAYS = (
    "heat_capacity_kb_per_atom",
    "entropy_kb_per_atom",
    "internal_energy_mev_per_atom",
    "free_energy_mev_per_atom",
)


def test_thermodynamics_einstein_solid():
    """A narrow peak at nu_E gives the Einstein heat capacity, entropy and energy, and ZPE = 3 h nu_E / 2."""
    nu_e, sigma = 10.0, 0.05
    freqs = np.linspace(0.0, 20.0, 20001)
    vdos = np.exp(-0.5 * ((freqs - nu_e) / sigma) ** 2)
    vdos /= np.trapezoid(vdos, freqs)
    temps = np.array([100.0, 300.0, 1000.0])
    result = compute_vibrational_thermodynamics(freqs, vdos, temps)
    x = nu_e * THZ_TO_K / temps
    # Averaging over the Gaussian shifts each quantity by about q'' sigma^2 / 2, a relative deviation of order
    # (x sigma / nu_E)^2 <= 6e-4 here (x <= 4.8); the largest measured is 2e-4 (entropy at 100 K).
    rtol = 1e-3
    np.testing.assert_allclose(result["heat_capacity_kb_per_atom"], 3 * x**2 * np.exp(x) / np.expm1(x) ** 2, rtol=rtol)
    np.testing.assert_allclose(result["entropy_kb_per_atom"], 3 * (x / np.expm1(x) - np.log(1 - np.exp(-x))), rtol=rtol)
    np.testing.assert_allclose(
        result["internal_energy_mev_per_atom"], 3 * nu_e * THZ_TO_MEV * (0.5 + 1 / np.expm1(x)), rtol=rtol
    )
    assert result["zero_point_energy_mev_per_atom"] == pytest.approx(1.5 * nu_e * THZ_TO_MEV, rel=1e-12)


def test_thermodynamics_debye_solid():
    """A Debye VDOS gives the Debye heat capacity at low, intermediate and high temperature."""
    theta_d = DEBYE_THZ * THZ_TO_K
    temps = theta_d * np.array([0.05, 0.5, 5.0])
    result = compute_vibrational_thermodynamics(DEBYE_GRID_THZ, DEBYE_VDOS, temps)
    expected = [
        9 * (t / theta_d) ** 3 * quad(lambda y: y**4 * np.exp(y) / np.expm1(y) ** 2, 0.0, theta_d / t)[0] for t in temps
    ]
    # The trapezoid rule overestimates the normalisation of this grid by 3e-8.
    np.testing.assert_allclose(result["heat_capacity_kb_per_atom"], expected, rtol=1e-7)


@pytest.mark.parametrize("temperature", [5e-324, 1e-310, 1e-307])
def test_thermodynamics_subnormal_temperature(temperature):
    """H nu / (k_B T) overflows at a subnormal T; the T = 0 limits are returned, without NaN or a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_vibrational_thermodynamics(DEBYE_GRID_THZ, DEBYE_VDOS, temperature)
    zpe = result["zero_point_energy_mev_per_atom"]
    assert result["heat_capacity_kb_per_atom"][0] == 0.0
    assert result["entropy_kb_per_atom"][0] == 0.0
    assert result["internal_energy_mev_per_atom"][0] == zpe
    assert result["free_energy_mev_per_atom"][0] == zpe


def test_thermodynamics_limits_and_free_energy():
    """T = 0 gives the exact limits, high T the classical 3 k_B, F = U - T S throughout, and nothing overflows."""
    temps = np.array([0.0, 1e-3, 300.0, 1e6])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_vibrational_thermodynamics(DEBYE_GRID_THZ, DEBYE_VDOS, temps)
        scalar = compute_vibrational_thermodynamics(DEBYE_GRID_THZ, DEBYE_VDOS, 300.0)
    zpe = result["zero_point_energy_mev_per_atom"]
    cv, s = result["heat_capacity_kb_per_atom"], result["entropy_kb_per_atom"]
    u, f = result["internal_energy_mev_per_atom"], result["free_energy_mev_per_atom"]
    assert (cv[0], s[0], u[0], f[0]) == (0.0, 0.0, zpe, zpe)
    np.testing.assert_allclose([cv[1], s[1], u[1] - zpe, f[1] - zpe], 0.0, atol=1e-12)
    assert cv[-1] == pytest.approx(3.0, rel=1e-6)
    np.testing.assert_allclose(f, u - temps * s * KB_MEV, rtol=1e-12)
    for key in ("temperatures_k", "reliable", *THERMO_ARRAYS):
        np.testing.assert_array_equal(scalar[key], result[key][2:3])


def test_partial_thermodynamics_sum_to_total():
    """Every quantity of partial spectra sums to that of the total, including the excluded weight."""
    rng = np.random.default_rng(0)
    partials = rng.random((2, SIGNED_GRID_THZ.size))
    temps = [0.0, 50.0, 300.0, 1500.0]
    total = compute_vibrational_thermodynamics(SIGNED_GRID_THZ, partials.sum(axis=0), temps)
    parts = [compute_vibrational_thermodynamics(SIGNED_GRID_THZ, p, temps) for p in partials]
    for key in (*THERMO_ARRAYS, "zero_point_energy_mev_per_atom", "vdos_integral", "excluded_weight"):
        np.testing.assert_allclose(parts[0][key] + parts[1][key], total[key], rtol=1e-12)


def test_thermodynamics_weight_at_and_below_zero_frequency():
    """Weight at nu <= 0 is reported as excluded_weight and leaves the results finite and unchanged."""
    freqs = np.arange(-2, 101) * 0.1
    level = 0.1
    temps = [0.0, 300.0]
    result = compute_vibrational_thermodynamics(freqs, np.full(freqs.size, level), temps)
    cut = compute_vibrational_thermodynamics(freqs, np.where(freqs > 0, level, 0.0), temps)
    # The two bins below 0 and half of the bin at 0 are excluded.
    assert result["excluded_weight"] == pytest.approx(level * 2.5 * 0.1, rel=1e-12)
    assert result["vdos_integral"] == pytest.approx(level * 10.2, rel=1e-12)
    assert cut["excluded_weight"] == 0.0
    for key in THERMO_ARRAYS:
        assert np.all(np.isfinite(result[key]))
        np.testing.assert_allclose(result[key], cut[key], rtol=1e-12)


@pytest.mark.parametrize(
    ("lowest", "nu_low", "source"),
    [
        pytest.param(None, 0.1, "grid", id="grid"),  # the first positive frequency, not the first grid point
        pytest.param(2.0, 2.0, "caller", id="caller"),
    ],
)
def test_thermodynamics_temperature_bound(lowest, nu_low, source):
    """T_min = h nu_low / k_B; temperatures below it are flagged, not removed, and T_min itself is reliable."""
    bound = nu_low * THZ_TO_K
    temps = np.array([0.0, bound / 2, bound, 2 * bound])
    result = compute_vibrational_thermodynamics(
        SIGNED_GRID_THZ, np.ones(SIGNED_GRID_THZ.size), temps, lowest_frequency_thz=lowest
    )
    assert result["lowest_frequency_thz"] == nu_low
    assert result["lowest_frequency_source"] == source
    assert result["temperature_bound_k"] == bound
    np.testing.assert_array_equal(result["reliable"], [False, False, True, True])
    np.testing.assert_array_equal(result["temperatures_k"], temps)
    assert result["heat_capacity_kb_per_atom"].shape == temps.shape


def _thermodynamics_invalid_cases() -> list:
    freqs = DEBYE_GRID_THZ
    nan_vdos = DEBYE_VDOS.copy()
    nan_vdos[5] = np.nan
    negative_vdos = DEBYE_VDOS.copy()
    negative_vdos[5] = -1e-3
    return [
        pytest.param(freqs, DEBYE_VDOS[:-1], 300.0, {}, "1-D arrays of the same length", id="length-mismatch"),
        pytest.param([1.0], [1.0], 300.0, {}, "At least 2", id="too-few-points"),
        pytest.param(freqs, nan_vdos, 300.0, {}, "NaN or inf", id="vdos-nan"),
        pytest.param(freqs, negative_vdos, 300.0, {}, "must not be negative", id="vdos-negative"),
        pytest.param(freqs[::-1], DEBYE_VDOS, 300.0, {}, "strictly increasing", id="unsorted"),
        pytest.param([-1.0, 0.0], [1.0, 1.0], 300.0, {}, "above 0 THz", id="no-positive-frequency"),
        pytest.param(freqs, DEBYE_VDOS, [-1.0, 300.0], {}, "non-negative", id="temperature-negative"),
        pytest.param(freqs, DEBYE_VDOS, True, {}, "not booleans", id="temperature-bool"),
        pytest.param(freqs, DEBYE_VDOS, [True, False], {}, "not booleans", id="temperature-bool-list"),
        pytest.param(freqs, DEBYE_VDOS, 300.0, {"lowest_frequency_thz": 0.0}, "positive and finite", id="lowest-zero"),
        pytest.param(freqs, DEBYE_VDOS, 300.0, {"lowest_frequency_thz": "1"}, "positive and finite", id="lowest-str"),
        pytest.param(freqs, DEBYE_VDOS, 300.0, {"lowest_frequency_thz": True}, "positive and finite", id="lowest-bool"),
    ]


@pytest.mark.parametrize(("freqs", "vdos", "temps", "kwargs", "match"), _thermodynamics_invalid_cases())
def test_thermodynamics_invalid_input_raises(freqs, vdos, temps, kwargs, match):
    """Invalid spectra, temperatures and lowest frequencies raise ValueError."""
    with pytest.raises(ValueError, match=match):
        compute_vibrational_thermodynamics(freqs, vdos, temps, **kwargs)


# ---------------------------------------------------------------------------
# read_dynamical_matrix
# ---------------------------------------------------------------------------

# data/morse_si_o_dynmat.{bin,txt}: LAMMPS 29 Aug 2024 (lmp_mpi, 1 rank), one Si (type 1) and one O (type 2) at the
# minimum of a Morse pair, bond skewed along (1, 2, 3)/sqrt(14). Data file: box 0-20 Å on each axis, masses below,
#   1 1 10 10 10
#   2 2 10.427617987059879 10.855235974119758 11.282853961179637
# Input (the text file uses "file dynmat.txt binary no"):
#   units metal / atom_style atomic / atom_modify map array / boundary f f f / read_data data.pair
#   pair_style morse 5.0 / pair_coeff * * 1.0 2.0 1.6
#   dynamical_matrix all regular 1e-5 file dynmat.bin binary yes
# Text layout: 3 values per line in %.8f, each matrix row spans N lines, no header.
MORSE_MASSES = (28.0855, 15.9994)
# eskm / regular measured on LAMMPS 29 Aug 2024 (exact to 2e-16); CODATA e N_A / 10 is 3.4e-6 higher.
LAMMPS_ESKM_PER_REGULAR = 9648.5


def _morse_dynamical_matrix():
    u = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    block = 2 * 1.0 * 2.0**2 * np.outer(u, u)  # k = 2 D alpha² along the bond
    phi = np.block([[block, -block], [-block, block]])
    masses = np.repeat(MORSE_MASSES, 3)
    return phi / np.sqrt(np.outer(masses, masses))


@pytest.mark.parametrize(
    ("name", "binary", "atol"),
    [
        pytest.param("morse_si_o_dynmat.bin", True, 1e-9, id="binary"),
        # %.8f text: rounding to 8 decimals limits the precision to 5e-9.
        pytest.param("morse_si_o_dynmat.txt", False, 5.1e-9, id="text"),
    ],
)
def test_read_dynamical_matrix_lammps_fixture(name, binary, atol):
    """The LAMMPS files read back as Phi / sqrt(m_i m_j) of the Morse pair, Si block first."""
    matrix = read_dynamical_matrix(DATA_DIR / name, 2, binary=binary)
    np.testing.assert_allclose(matrix, _morse_dynamical_matrix(), rtol=0, atol=atol)


def test_read_dynamical_matrix_round_trip(tmp_path):
    """A non-symmetric matrix reads back bitwise equal as a writable, C-contiguous float64 array."""
    original = np.random.default_rng(7).normal(size=(12, 12))
    original.tofile(tmp_path / "dynmat.bin")
    matrix = read_dynamical_matrix(tmp_path / "dynmat.bin", 4)
    np.testing.assert_array_equal(matrix, original)
    assert matrix.dtype == np.float64
    assert matrix.flags.c_contiguous
    assert matrix.flags.writeable


def test_read_dynamical_matrix_eskm(tmp_path):
    """An eskm file (regular x 9648.5, as LAMMPS writes it) reads back as the regular matrix."""
    assert abs(LAMMPS_ESKM_PER_REGULAR / (e * N_A / 10) - 1) < 4e-6
    regular = np.random.default_rng(8).normal(size=(6, 6))
    (regular * LAMMPS_ESKM_PER_REGULAR).tofile(tmp_path / "dynmat.bin")
    matrix = read_dynamical_matrix(tmp_path / "dynmat.bin", 2, style="eskm")
    np.testing.assert_allclose(matrix, regular, rtol=1e-12)


def _matrix_with(row, column, value):
    matrix = np.ones((12, 12))
    matrix[row, column] = value
    return matrix


def _write_binary(values):
    return lambda path: np.asarray(values, dtype=np.float64).tofile(path)


def _write_text(values):
    return lambda path: np.savetxt(path, np.reshape(values, (-1, 3)), fmt="%.8f")


def _read_dynamical_matrix_invalid_cases():
    good = _write_binary(np.ones(36))
    return [
        pytest.param(good, 3, {}, ValueError, r"36 values, expected 9 N² = 81 .*imply n_atoms=2", id="size-binary"),
        pytest.param(
            lambda path: path.write_bytes(np.ones(36).tobytes() + b"abc"),
            2,
            {},
            ValueError,
            "291 bytes, not a whole number",
            id="size-partial-value",
        ),
        pytest.param(_write_text(np.ones(33)), 2, {"binary": False}, ValueError, r"33 values, .*=2\.$", id="size-text"),
        pytest.param(_write_binary(_matrix_with(11, 5, np.nan)), 4, {}, ValueError, "rows 11 to 11", id="nan-binary"),
        pytest.param(
            _write_text(_matrix_with(10, 0, -np.inf)), 4, {"binary": False}, ValueError, "rows 10 to 10", id="inf-text"
        ),
        pytest.param(good, 0, {}, ValueError, "positive integer", id="n-atoms-zero"),
        pytest.param(good, True, {}, ValueError, "positive integer", id="n-atoms-bool"),
        pytest.param(good, 2.0, {}, ValueError, "positive integer", id="n-atoms-float"),
        pytest.param(good, 2, {"style": "Regular"}, ValueError, "Unknown style", id="style"),
        pytest.param(None, 2, {}, FileNotFoundError, "dynmat", id="missing-file"),
    ]


@pytest.mark.parametrize(("write", "n_atoms", "kwargs", "error", "match"), _read_dynamical_matrix_invalid_cases())
def test_read_dynamical_matrix_invalid_raises(monkeypatch, tmp_path, write, n_atoms, kwargs, error, match):
    """Wrong sizes, non-finite values in a late row block, bad n_atoms or style, and a missing file raise."""
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)  # one row per chunk
    path = tmp_path / "dynmat"
    if write is not None:
        write(path)
    with pytest.raises(error, match=match):
        read_dynamical_matrix(path, n_atoms, **kwargs)


# ---------------------------------------------------------------------------
# diagonalize_dynamical_matrix
# ---------------------------------------------------------------------------

# An eigenvalue of 1 eV/Å²/amu is an angular frequency sqrt(e / (1e-20 atomic_mass)) rad/s, i.e. 15.633 THz.
THZ_PER_SQRT_EV_PER_A2_AMU = np.sqrt(e / (1e-20 * atomic_mass)) / (2 * np.pi) / 1e12


def _to_thz(eigenvalues):
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * THZ_PER_SQRT_EV_PER_A2_AMU


def _ring_matrix(n_atoms, k, mass):
    """N identical atoms on a periodic ring with isotropic nearest-neighbour springs: x, y and z decouple."""
    eye = np.eye(n_atoms)
    ring = 2 * eye - np.roll(eye, 1, axis=1) - np.roll(eye, -1, axis=1)
    return np.kron(ring, np.eye(3)) * k / mass


def _spectral_matrix(masses, eigenvalues, seed=0):
    """D = Q diag(eigenvalues) Q^T, Q orthogonal with its first three columns spanning the rigid translations."""
    n_dof = 3 * len(masses)
    translations = np.zeros((n_dof, 3))
    for alpha in range(3):
        translations[alpha::3, alpha] = np.sqrt(masses)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(np.hstack([translations, rng.normal(size=(n_dof, n_dof - 3))]))
    matrix = (q * eigenvalues) @ q.T
    return (matrix + matrix.T) / 2, q


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("driver", ["ev", "evd", "evr", "evx"])
def test_diagonalize_ring_matches_analytic(driver, overwrite):
    """A periodic spring ring gives (1/2pi) 2 sqrt(k/m) |sin(pi q/N)| three times per q != 0, with eigenvectors."""
    assert abs(THZ_PER_SQRT_EV_PER_A2_AMU - 15.633) < 5e-4
    n_atoms, k, mass = 10, 3.0, 28.0855
    matrix = _ring_matrix(n_atoms, k, mass)
    original = matrix.copy()
    result = diagonalize_dynamical_matrix(
        matrix, masses=np.full(n_atoms, mass), overwrite_matrix=overwrite, driver=driver
    )
    q = np.arange(1, n_atoms)
    expected = np.sort(np.repeat(2 * np.sqrt(k / mass) * np.abs(np.sin(np.pi * q / n_atoms)) / (2 * np.pi), 3))
    np.testing.assert_allclose(result["frequencies_thz"], expected * 2 * np.pi * THZ_PER_SQRT_EV_PER_A2_AMU, rtol=1e-10)
    vectors = result["eigenvectors"]
    eigenvalues = np.sign(result["frequencies_thz"]) * (result["frequencies_thz"] / THZ_PER_SQRT_EV_PER_A2_AMU) ** 2
    np.testing.assert_allclose(original @ vectors, vectors * eigenvalues, atol=1e-12)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3 * n_atoms - 3), atol=1e-12)
    np.testing.assert_allclose(result["translational_overlap"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["translational_frequencies_thz"], 0.0, atol=1e-5)
    assert (result["asymmetry"], result["asymmetry_atoms"], result["n_imaginary"]) == (0.0, None, 0)
    assert (result["n_atoms"], result["driver"]) == (n_atoms, driver)
    if not overwrite:
        np.testing.assert_array_equal(matrix, original)


def test_diagonalize_lammps_morse_pair():
    """The LAMMPS Morse pair has one nonzero frequency sqrt(2 D alpha^2 (1/m1 + 1/m2)) and near-zero rotations."""
    result = diagonalize_dynamical_matrix(read_dynamical_matrix(DATA_DIR / "morse_si_o_dynmat.bin", 2))
    stretch = np.sqrt(2 * 1.0 * 2.0**2 * (1 / MORSE_MASSES[0] + 1 / MORSE_MASSES[1])) * THZ_PER_SQRT_EV_PER_A2_AMU
    np.testing.assert_allclose(result["frequencies_thz"][-1], stretch, rtol=1e-9)
    assert np.abs(result["frequencies_thz"][:-1]).max() < 0.01
    assert result["translational_overlap"] is None
    assert result["eigenvectors"].shape == (6, 3)


@pytest.mark.parametrize("order", ["C", "F"])
def test_diagonalize_asymmetry_is_measured_before_symmetrising(monkeypatch, order):
    """Antisymmetric perturbations give the exact relative asymmetry and atom pair; LAPACK gets (D + D^T) / 2."""
    n_dof = 24
    rng = np.random.default_rng(3)
    a = rng.normal(size=(n_dof, n_dof))
    matrix = a @ a.T / n_dof
    matrix[3 * 4 + 1, 3 * 2 + 2] += 0.3  # atoms (4, 2), rows 13 and 8: middle blocks
    matrix[3 * 2 + 2, 3 * 4 + 1] -= 0.3
    matrix[3 * 7, 3 * 5 + 1] += 0.1  # atoms (7, 5), rows 21 and 16: smaller, partly in the last block
    matrix[3 * 5 + 1, 3 * 7] -= 0.1
    matrix = np.asarray(matrix, order=order)
    symmetrised = (matrix + matrix.T) / 2
    max_abs = np.abs(matrix).max()
    expected = np.abs(matrix - matrix.T).max() / max_abs
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 5 * n_dof * 8)  # 5 rows per block, 5 blocks
    received = []
    eigh = scipy.linalg.eigh

    def spy(a, **kwargs):
        received.append(np.array(a))
        return eigh(a, **kwargs)

    monkeypatch.setattr(scipy.linalg, "eigh", spy)
    result = diagonalize_dynamical_matrix(matrix, overwrite_matrix=True, eigenvectors=False)
    assert result["asymmetry"] == pytest.approx(expected, rel=1e-12)
    assert result["asymmetry"] == pytest.approx(0.6 / max_abs, rel=1e-12)
    assert result["asymmetry_atoms"] == (2, 4)
    np.testing.assert_array_equal(received[0], symmetrised)
    np.testing.assert_allclose(
        result["frequencies_thz"], _to_thz(np.linalg.eigvalsh(symmetrised)[3:]), rtol=1e-10, atol=1e-12
    )


def test_diagonalize_imaginary_modes_are_negative_and_warn():
    """Two negative eigenvalues orthogonal to the translations give two negative frequencies and a warning."""
    masses = np.array([15.9994, 28.0855, 15.9994, 15.9994, 28.0855])
    eigenvalues = np.concatenate([np.zeros(3), [-2.0, -0.5], np.linspace(0.3, 3.0, 10)])
    matrix, q = _spectral_matrix(masses, eigenvalues)
    with pytest.warns(UserWarning, match=r"2 imaginary mode\(s\) below -0.01 THz; the most negative is -22.11") as rec:
        result = diagonalize_dynamical_matrix(matrix, masses=masses)
    assert len(rec) == 1
    assert result["n_imaginary"] == 2
    np.testing.assert_allclose(result["frequencies_thz"], _to_thz(eigenvalues[3:]), rtol=1e-10)
    assert result["min_frequency_thz"] == result["frequencies_thz"][0]
    # The two imaginary modes precede the translations, so their columns were moved; each matches Q up to sign.
    np.testing.assert_allclose(np.abs((result["eigenvectors"] * q[:, 3:]).sum(axis=0)), 1.0, atol=1e-10)
    np.testing.assert_allclose(result["translational_overlap"], 1.0, atol=1e-12)


def test_diagonalize_warns_when_dropped_modes_are_not_translations():
    """A soft mode below translations lifted by a sum-rule violation is dropped instead, with a warning."""
    masses = np.array([15.9994, 28.0855, 15.9994, 15.9994, 28.0855])
    eigenvalues = np.concatenate([np.full(3, 1e-3), [1e-5], np.linspace(0.3, 3.0, 11)])
    matrix, _ = _spectral_matrix(masses, eigenvalues)
    with pytest.warns(UserWarning, match="overlap the rigid translations by only"):
        result = diagonalize_dynamical_matrix(matrix, masses=masses)
    assert result["translational_overlap"].min() < 1e-10
    assert result["translational_frequencies_thz"][0] == pytest.approx(_to_thz(1e-5), rel=1e-6)


def test_diagonalize_overwrite_without_eigenvectors_does_not_copy(monkeypatch):
    """overwrite_matrix=True and eigenvectors=False allocate well below one matrix beyond the input."""
    matrix = _ring_matrix(300, 3.0, 28.0855)  # n = 900, 6.5 MB
    # Real matrices are far above _CHUNK_BYTES; scale the blocks to 1/16 of the matrix so they stay small here too.
    # Expected peak: about 3 blocks (3/16) plus O(n) LAPACK workspace (~0.03), against >= 1 for any copy.
    monkeypatch.setattr(_common, "_CHUNK_BYTES", matrix.nbytes // 16)
    tracemalloc.start()
    try:
        diagonalize_dynamical_matrix(matrix, overwrite_matrix=True, eigenvectors=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 0.5 * matrix.nbytes


def _with_nan(matrix, row, column):
    matrix = matrix.copy()
    matrix[row, column] = np.nan
    return matrix


def _diagonalize_invalid_cases():
    good = np.eye(6)
    return [
        pytest.param(np.eye(6)[None], {}, "square 2-D", id="3d"),
        pytest.param(np.ones((6, 9)), {}, "square 2-D", id="not-square"),
        pytest.param(np.eye(4), {}, "N >= 2", id="not-multiple-of-3"),
        pytest.param(np.eye(3), {}, "N >= 2", id="one-atom"),
        pytest.param(np.eye(6, dtype=np.float32), {}, "float64", id="float32"),
        pytest.param(_with_nan(good, 4, 1), {}, "NaN or inf in rows or columns 1 to 1", id="nan"),
        pytest.param(np.eye(12)[::2, ::2], {"overwrite_matrix": True}, "C- or F-contiguous", id="overwrite-strided"),
        pytest.param(good, {"masses": [1.0]}, "2 positive finite", id="masses-shape"),
        pytest.param(good, {"masses": [1.0, 0.0]}, "2 positive finite", id="mass-zero"),
        pytest.param(good, {"masses": [1.0, np.nan]}, "2 positive finite", id="mass-nan"),
        pytest.param(good, {"masses": [1.0, 1.0], "eigenvectors": False}, "eigenvectors=True", id="masses-no-vectors"),
        pytest.param(good, {"driver": "evx2"}, "Unknown driver", id="driver"),
        pytest.param(good, {"imaginary_threshold_thz": -0.1}, ">= 0", id="threshold-negative"),
    ]


@pytest.mark.parametrize(("matrix", "kwargs", "match"), _diagonalize_invalid_cases())
def test_diagonalize_invalid_input_raises(monkeypatch, matrix, kwargs, match):
    """Bad shapes, dtypes, non-finite entries in a later block, masses, drivers and thresholds raise."""
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)  # one row per block
    with pytest.raises(ValueError, match=match):
        diagonalize_dynamical_matrix(matrix, **kwargs)


# ---------------------------------------------------------------------------
# compute_harmonic_vdos
# ---------------------------------------------------------------------------

HARMONIC_FREQUENCIES_THZ = np.array([-1.5, 0.8, 3.0, 7.2, 12.0, 12.4])  # M = 6 (N = 3), one imaginary mode
SPRING_MASSES = np.array([28.0855, 15.9994, 15.9994, 28.0855, 15.9994, 15.9994, 15.9994, 28.0855])
SPRING_SI, SPRING_O = np.array([0, 3, 7]), np.array([1, 2, 4, 5, 6])


@pytest.mark.parametrize(
    ("normalization", "c"), [pytest.param("unit", 1 / 6, id="unit"), pytest.param("3N", 9 / 6, id="3N")]
)
def test_harmonic_vdos_total_is_gaussian_sum(normalization, c):
    """Without eigenvectors the total is c times the unit Gaussians of the real modes; the imaginary one is missing."""
    sigma = 0.3
    grid = np.linspace(-5.0, 20.0, 2501)
    result = compute_harmonic_vdos(
        HARMONIC_FREQUENCIES_THZ, frequency_grid_thz=grid, broadening_thz=sigma, normalization=normalization
    )
    real = HARMONIC_FREQUENCIES_THZ[1:]
    expected = c * np.exp(-0.5 * ((grid[:, None] - real) / sigma) ** 2).sum(axis=1) / (sigma * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(result["vdos"]["total"], expected, rtol=1e-12, atol=1e-300)
    assert (result["n_modes"], result["n_imaginary_excluded"], result["imaginary_weight"]) == (6, 1, 1 / 6)
    assert result["grid_integral"] == pytest.approx(6 * c * (1 - 1 / 6), abs=1e-10)
    assert "mode_weights" not in result


def _spring_modes():
    """Kept modes of a random 8-atom spring network with Si and O masses, through diagonalize_dynamical_matrix."""
    eigenvalues = np.concatenate([np.zeros(3), np.linspace(0.3, 3.0, 21)])
    matrix, _ = _spectral_matrix(SPRING_MASSES, eigenvalues, seed=1)
    modes = diagonalize_dynamical_matrix(matrix, masses=SPRING_MASSES)
    groups = {
        "atom": {str(i): np.array([i]) for i in range(len(SPRING_MASSES))},
        "element": {"Si": SPRING_SI, "O": SPRING_O},
        "oxygen": {"O_BO": SPRING_O[:3], "O_NBO": SPRING_O[3:]},
    }
    return modes["frequencies_thz"], modes["eigenvectors"], groups


def test_harmonic_vdos_completeness_and_partials():
    """Each atom's weight over the kept modes is 3 - 3 m_i / m_total; partials sum to the total and to "O"."""
    frequencies, vectors, groups = _spring_modes()
    result = compute_harmonic_vdos(frequencies, vectors, groups=groups)
    atom_sums = [result["mode_weights"]["atom"][str(i)].sum() for i in range(len(SPRING_MASSES))]
    np.testing.assert_allclose(atom_sums, 3 - 3 * SPRING_MASSES / SPRING_MASSES.sum(), atol=1e-12)
    vdos = result["vdos"]
    np.testing.assert_allclose(vdos["element"]["Si"] + vdos["element"]["O"], vdos["total"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(vdos["oxygen"]["O_BO"] + vdos["oxygen"]["O_NBO"], vdos["element"]["O"], atol=1e-15)


def test_harmonic_vdos_lammps_morse_pair():
    """In the stretch of the LAMMPS Morse pair each atom carries the other's mass fraction (momentum conservation)."""
    modes = diagonalize_dynamical_matrix(
        read_dynamical_matrix(DATA_DIR / "morse_si_o_dynmat.bin", 2), masses=MORSE_MASSES
    )
    groups = {"element": {"Si": np.array([0]), "O": np.array([1])}}
    weights = compute_harmonic_vdos(modes["frequencies_thz"], modes["eigenvectors"], groups=groups)["mode_weights"]
    m_si, m_o = MORSE_MASSES
    assert weights["element"]["Si"][-1] == pytest.approx(m_o / (m_si + m_o), abs=1e-8)
    assert weights["element"]["O"][-1] == pytest.approx(m_si / (m_si + m_o), abs=1e-8)


def test_harmonic_vdos_chunks_give_same_result(monkeypatch):
    """One mode per block, for the weights and the Gaussians, gives the result of the default blocks."""
    frequencies, vectors, groups = _spring_modes()
    default = compute_harmonic_vdos(frequencies, vectors, groups=groups, normalization="3N")
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)
    einsums, exps = _count_calls(monkeypatch, np, "einsum"), _count_calls(monkeypatch, np, "exp")
    chunked = compute_harmonic_vdos(frequencies, vectors, groups=groups, normalization="3N")
    # One mode per block: a weight block per mode, and a block of Gaussians per mode with nu >= 0.
    weight_blocks = sum(isinstance(call[0], str) and call[0] == "iak,iak->ik" for call in einsums)
    gaussian_blocks = len(exps)
    assert weight_blocks == frequencies.size
    assert gaussian_blocks == np.count_nonzero(frequencies >= 0)
    np.testing.assert_allclose(chunked["vdos"]["total"], default["vdos"]["total"], rtol=1e-13, atol=1e-16)
    for grouping, labels in groups.items():
        for label in labels:
            for key in ("vdos", "mode_weights"):
                np.testing.assert_allclose(
                    chunked[key][grouping][label], default[key][grouping][label], rtol=1e-13, atol=1e-16
                )


def test_harmonic_vdos_caller_grid():
    """A grid coarser than sigma / 2 warns; the VACF grid does not, and matches the default grid where they meet."""
    frequencies, vectors, groups = _spring_modes()
    with pytest.warns(UserWarning, match="exceeds half the broadening"):
        compute_harmonic_vdos(frequencies, frequency_grid_thz=np.arange(0, 40, 0.15))
    default = compute_harmonic_vdos(frequencies, vectors, groups=groups)
    vacf_grid = np.fft.rfftfreq(2000, d=5e-3)  # vdos_simulation defaults: 10 ps blocks, 5 fs
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on_vacf = compute_harmonic_vdos(frequencies, vectors, groups=groups, frequency_grid_thz=vacf_grid)
    np.testing.assert_array_equal(on_vacf["frequencies_thz"], vacf_grid)
    i, j = np.nonzero(np.isclose(default["frequencies_thz"][:, None], vacf_grid[None, :], rtol=0, atol=1e-9))
    assert i.size > 100
    for key in (("total",), ("oxygen", "O_NBO")):
        a, b = default["vdos"], on_vacf["vdos"]
        for part in key:
            a, b = a[part], b[part]
        np.testing.assert_allclose(b[j], a[i], rtol=1e-9, atol=1e-15)


def _harmonic_invalid_cases():
    f, vectors = HARMONIC_FREQUENCIES_THZ, np.zeros((9, 6))
    groups = {"element": {"Si": np.array([0])}}
    return [
        pytest.param(f.reshape(2, 3), None, {}, "non-empty 1-D", id="frequencies-2d"),
        pytest.param([np.nan, 1.0, 2.0], None, {}, "non-empty 1-D", id="frequencies-nan"),
        pytest.param([], None, {}, "non-empty 1-D", id="frequencies-empty"),
        pytest.param(f, np.zeros(9), {}, "2-D float64", id="vectors-1d"),
        pytest.param(f, vectors.astype(np.float32), {}, "2-D float64", id="vectors-float32"),
        pytest.param(f, np.zeros((6, 6)), {}, r"\(3N, 3N - 3\)", id="vectors-rows"),
        pytest.param(f, np.zeros((9, 5)), {}, r"\(3N, 3N - 3\)", id="vectors-columns"),
        pytest.param(f, None, {"broadening_thz": 0.0}, "positive and finite", id="broadening-zero"),
        pytest.param(f, None, {"broadening_thz": None}, "positive and finite", id="broadening-none"),
        pytest.param(f, None, {"broadening_thz": True}, "positive and finite", id="broadening-bool"),
        pytest.param(f, None, {"groups": groups}, "groups need eigenvectors", id="groups-without-vectors"),
        pytest.param(f, vectors, {"groups": {}}, "no labels", id="groups-empty"),
        pytest.param(f, None, {"normalization": "none"}, "Unknown normalization", id="normalization"),
        pytest.param(f[:5], None, {"normalization": "3N"}, "not a multiple of 3", id="3n-without-vectors"),
        pytest.param(f, None, {"frequency_grid_thz": np.zeros((2, 2))}, "strictly increasing", id="grid-2d"),
        pytest.param(f, None, {"frequency_grid_thz": [0.0]}, "strictly increasing", id="grid-one-point"),
        pytest.param(f, None, {"frequency_grid_thz": [0.0, np.nan, 1.0]}, "strictly increasing", id="grid-nan"),
        pytest.param(f, None, {"frequency_grid_thz": [0.0, 1.0, 1.0]}, "strictly increasing", id="grid-repeated"),
    ]


@pytest.mark.parametrize(("frequencies", "vectors", "kwargs", "match"), _harmonic_invalid_cases())
def test_harmonic_vdos_invalid_input_raises(frequencies, vectors, kwargs, match):
    """Bad frequencies, eigenvectors, broadening, groups, normalisation and grids raise."""
    with pytest.raises(ValueError, match=match):
        compute_harmonic_vdos(frequencies, vectors, **kwargs)


# ---------------------------------------------------------------------------
# compute_participation_ratio
# ---------------------------------------------------------------------------


def _mode(masses, displacements):
    """One mode as a (3N, 1) mass-weighted column e = sqrt(m) u from per-atom (x, y, z) displacements u."""
    masses = np.asarray(masses, dtype=np.float64)
    return (np.sqrt(masses)[:, None] * np.array(displacements, dtype=np.float64)).reshape(-1, 1)


SIO2_MASSES = (28.0855, 15.9994, 15.9994)


@pytest.mark.parametrize(
    ("masses", "displacements", "expected"),
    [
        pytest.param(SIO2_MASSES, [(0.6, 0.8, 0)] * 3, 1.0, id="rigid-translation"),
        pytest.param(
            np.arange(1.0, 6.0), [(0, 0, 0), (0, 0, 0), (0, 0.3, 0), (0, 0, 0), (0, 0, 0)], 1 / 5, id="one-atom"
        ),
        # Equal |u| = 2 on atoms 0, 2 and 4 of 6, along different directions and with different masses.
        pytest.param(
            SIO2_MASSES * 2, [(2, 0, 0), (0, 0, 0), (0, 2, 0), (0, 0, 0), (1.2, 1.6, 0), (0, 0, 0)], 1 / 2, id="n-of-n"
        ),
        # Atom 0 split 0.36 / 0.64 over x and y, atom 1 all z: b = (1, 1, 0), P = 2² / (3 · 2).
        pytest.param(np.ones(3), [(0.6, 0.8, 0), (0, 0, 1), (0, 0, 0)], 2 / 3, id="uneven-xyz"),
    ],
)
def test_participation_ratio_analytic(masses, displacements, expected):
    """Translation, one-atom, n-atom and direction-split modes give 1, 1/N, n/N and 2/3 at any column scale and sign."""
    vectors = _mode(masses, displacements)
    for scaled in (vectors, -3.7 * vectors):
        np.testing.assert_allclose(compute_participation_ratio(scaled, masses), [expected], rtol=1e-14)


def test_participation_ratio_lammps_morse_pair():
    """In the stretch of the LAMMPS Morse pair |u_Si| / |u_O| = m_O / m_Si (momentum conservation)."""
    modes = diagonalize_dynamical_matrix(
        read_dynamical_matrix(DATA_DIR / "morse_si_o_dynmat.bin", 2), masses=MORSE_MASSES
    )
    m_si, m_o = MORSE_MASSES
    expected = (m_o**2 + m_si**2) ** 2 / (2 * (m_o**4 + m_si**4))
    assert compute_participation_ratio(modes["eigenvectors"], MORSE_MASSES)[-1] == pytest.approx(expected, abs=1e-8)


def test_participation_ratio_chunks_order_and_no_copy(monkeypatch):
    """F-ordered view and C copy, one mode per block, match the default blocks; no copy of the eigenvectors is made."""
    rng = np.random.default_rng(0)
    # (300, 400) view of a Fortran array, as diagonalize_dynamical_matrix returns it: 0.96 MB.
    vectors = np.asfortranarray(rng.normal(size=(300, 403)))[:, 3:]
    masses = rng.uniform(10.0, 30.0, size=100)
    tracemalloc.start()
    default = compute_participation_ratio(vectors, masses)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < vectors.nbytes / 2  # the atom weights are a third
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)
    einsums = _count_calls(monkeypatch, np, "einsum")
    orders = (vectors, np.ascontiguousarray(vectors))
    chunked = [compute_participation_ratio(ordered, masses) for ordered in orders]
    # One weight block per mode in each of the two calls.
    weight_blocks = sum(isinstance(call[0], str) and call[0] == "iak,iak->ik" for call in einsums)
    assert weight_blocks == len(orders) * vectors.shape[1]
    for ratios in chunked:  # einsum differs by ulps between memory orders
        np.testing.assert_allclose(ratios, default, rtol=1e-13)


def _participation_invalid_cases():
    nan_late, zero, masses = np.ones((6, 4)), np.ones((6, 4)), np.ones(2)
    nan_late[4, 3], zero[:, 2] = np.nan, 0.0
    return [
        pytest.param(np.ones(6), masses, "2-D float64", id="1d"),
        pytest.param(np.ones((6, 2), dtype=np.float32), masses, "2-D float64", id="float32"),
        pytest.param(np.ones((5, 2)), masses, "3N rows", id="rows-not-3n"),
        pytest.param(np.ones((0, 2)), masses, "3N rows", id="no-rows"),
        pytest.param(np.ones((6, 0)), masses, "3N rows", id="no-columns"),
        pytest.param(np.ones((6, 2)), np.ones(3), "2 positive finite", id="masses-length"),
        pytest.param(np.ones((6, 2)), np.array([1.0, 0.0]), "2 positive finite", id="masses-zero"),
        pytest.param(zero, masses, r"zero column \(mode 2\)", id="zero-column"),
        pytest.param(nan_late, masses, r"NaN or inf \(mode 3\)", id="nan-last-block"),
    ]


@pytest.mark.parametrize(("vectors", "masses", "match"), _participation_invalid_cases())
def test_participation_ratio_invalid_raises(monkeypatch, vectors, masses, match):
    """Bad shapes, dtypes and masses, a zero column and NaN in a late block raise."""
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)  # one mode per block
    with pytest.raises(ValueError, match=match):
        compute_participation_ratio(vectors, masses)


# ---------------------------------------------------------------------------
# compute_ir_oscillator_strengths
# ---------------------------------------------------------------------------


def _ir_system(n_atoms=8, seed=0, *, neutral=True, tensor=False):
    """Random masses, the 3N - 3 non-translational modes of a random orthogonal basis, and random charges."""
    rng = np.random.default_rng(seed)
    masses = rng.uniform(10.0, 30.0, size=n_atoms)
    _, basis = _spectral_matrix(masses, np.ones(3 * n_atoms), seed)
    charges = rng.normal(size=(n_atoms, 3, 3) if tensor else n_atoms)
    if neutral:
        charges -= charges.mean(axis=0)
    return masses, np.ascontiguousarray(basis[:, 3:]), charges


@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
@pytest.mark.parametrize("neutral", [True, False], ids=["neutral", "charged"])
def test_ir_strengths_completeness_sum_rule(neutral, tensor):
    """Over the 3N - 3 modes sum_m Z_m Z_m^T = sum_i Z_i Z_i^T / m_i - (sum_i Z_i)(sum_i Z_i)^T / M."""
    masses, vectors, charges = _ir_system(neutral=neutral, tensor=tensor)
    if neutral:
        result = compute_ir_oscillator_strengths(vectors, masses, charges)
    else:
        with pytest.warns(UserWarning, match="do not sum to zero"):
            result = compute_ir_oscillator_strengths(vectors, masses, charges)
    tensors = charges if tensor else charges[:, None, None] * np.eye(3)
    total = tensors.sum(axis=0)
    expected = np.einsum("iab,icb,i->ac", tensors, tensors, 1 / masses) - total @ total.T / masses.sum()
    mode_charges = result["mode_charges"]
    assert mode_charges.shape == (vectors.shape[1], 3)
    np.testing.assert_allclose(mode_charges.T @ mode_charges, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result["strengths"], np.sum(mode_charges**2, axis=1) / 3, rtol=1e-14)
    assert result["strengths"].sum() == pytest.approx(np.trace(expected) / 3, rel=1e-12)


def test_ir_strengths_pair_stretch_is_q2_over_reduced_mass():
    """The stretch of a +q / -q pair has |Z|^2 = q^2 / mu along the bond, and the rigid translation has none."""
    m_si, m_o = SIO2_MASSES[:2]
    charge = 2.4
    stretch = _mode([m_si, m_o], [(1 / m_si, 0, 0), (-1 / m_o, 0, 0)])
    translation = _mode([m_si, m_o], [(0, 1, 0), (0, 1, 0)])
    vectors = np.hstack([stretch / np.linalg.norm(stretch), translation / np.linalg.norm(translation)])
    result = compute_ir_oscillator_strengths(vectors, [m_si, m_o], [charge, -charge])
    reduced_mass = m_si * m_o / (m_si + m_o)
    np.testing.assert_allclose(result["strengths"], [charge**2 / (3 * reduced_mass), 0.0], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(np.abs(result["mode_charges"][0]), [charge / math.sqrt(reduced_mass), 0, 0], rtol=1e-13)


def test_ir_strengths_scalar_matches_isotropic_tensor():
    """Scalar charges q_i give the same result as the tensors q_i times the identity."""
    masses, vectors, charges = _ir_system()
    scalar = compute_ir_oscillator_strengths(vectors, masses, charges)
    tensor = compute_ir_oscillator_strengths(vectors, masses, charges[:, None, None] * np.eye(3))
    for key in ("mode_charges", "strengths"):
        np.testing.assert_allclose(tensor[key], scalar[key], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
def test_ir_partial_strengths_sum_to_total(tensor):
    """For a grouping holding every atom once the partial mode charges and the pair strengths sum to the totals."""
    masses, vectors, charges = _ir_system(n_atoms=9, tensor=tensor)
    groups = {
        "element": {"Si": np.arange(0, 3), "O": np.arange(3, 8), "Na": np.array([8])},
        "subset": {"first": np.array([0, 4])},
    }
    result = compute_ir_oscillator_strengths(vectors, masses, charges, groups=groups)
    assert list(result["partial_strengths"]["element"]) == ["Si-Si", "Si-O", "Si-Na", "O-O", "O-Na", "Na-Na"]
    assert list(result["partial_strengths"]["subset"]) == ["first-first"]
    np.testing.assert_allclose(
        sum(result["partial_mode_charges"]["element"].values()), result["mode_charges"], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        sum(result["partial_strengths"]["element"].values()), result["strengths"], rtol=1e-11, atol=1e-14
    )
    only_si = np.where(np.arange(9)[:, None, None] < 3, 1.0, 0.0) * (charges if tensor else charges[:, None, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        silicon = compute_ir_oscillator_strengths(vectors, masses, only_si if tensor else only_si[:, 0, 0])
    np.testing.assert_allclose(result["partial_strengths"]["element"]["Si-Si"], silicon["strengths"], rtol=1e-12)


def test_ir_strengths_chunks_and_memory_order(monkeypatch):
    """One mode per block and an F-ordered view give the default result."""
    masses, vectors, charges = _ir_system(tensor=True)
    groups = {"half": {"low": np.arange(4), "high": np.arange(4, 8)}}
    default = compute_ir_oscillator_strengths(vectors, masses, charges, groups=groups)
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)
    chunked = compute_ir_oscillator_strengths(np.asfortranarray(vectors), masses, charges, groups=groups)
    np.testing.assert_allclose(chunked["mode_charges"], default["mode_charges"], rtol=1e-13, atol=1e-15)
    for pair, strengths in default["partial_strengths"]["half"].items():
        np.testing.assert_allclose(chunked["partial_strengths"]["half"][pair], strengths, rtol=1e-12, atol=1e-15)


def _ir_strengths_invalid_cases():
    vectors, masses, charges = np.ones((6, 2)), np.ones(2), np.array([1.0, -1.0])
    nan = vectors.copy()
    nan[4, 1] = np.nan
    return [
        pytest.param(np.ones(6), masses, charges, {}, "2-D float64", id="vectors-1d"),
        pytest.param(vectors.astype(np.float32), masses, charges, {}, "2-D float64", id="vectors-float32"),
        pytest.param(np.ones((5, 2)), masses, charges, {}, "3N rows", id="rows-not-3n"),
        pytest.param(np.ones((6, 0)), masses, charges, {}, "3N rows", id="no-columns"),
        pytest.param(nan, masses, charges, {}, "NaN or inf", id="vectors-nan"),
        pytest.param(vectors, np.ones(3), charges, {}, "2 positive finite", id="masses-length"),
        pytest.param(vectors, np.array([1.0, -1.0]), charges, {}, "2 positive finite", id="masses-negative"),
        pytest.param(vectors, masses, np.ones(3), {}, r"shape \(2,\) or \(2, 3, 3\)", id="charges-length"),
        pytest.param(vectors, masses, np.ones((2, 3)), {}, r"shape \(2,\) or \(2, 3, 3\)", id="charges-2d"),
        pytest.param(vectors, masses, [1.0, np.nan], {}, r"shape \(2,\) or \(2, 3, 3\)", id="charges-nan"),
        pytest.param(vectors, masses, charges, {"groups": {"element": {}}}, "no labels", id="groups-empty"),
        pytest.param(
            vectors, masses, charges, {"groups": {"element": {"Si": np.array([2])}}}, "outside", id="groups-range"
        ),
    ]


@pytest.mark.parametrize(("vectors", "masses", "charges", "kwargs", "match"), _ir_strengths_invalid_cases())
def test_ir_strengths_invalid_input_raises(vectors, masses, charges, kwargs, match):
    """Bad eigenvectors, masses, charges and groups raise."""
    with pytest.raises(ValueError, match=match):
        compute_ir_oscillator_strengths(vectors, masses, charges, **kwargs)


# ---------------------------------------------------------------------------
# compute_dielectric_function and compute_optical_response
# ---------------------------------------------------------------------------

# e^2 / (4 pi^2 epsilon_0 amu A^3) in THz^2, from SI: (rad/s)^2 to THz^2 is 1e-24 / (2 pi)^2.
IR_THZ2_A3 = e**2 / (epsilon_0 * atomic_mass * 1e-30) * 1e-24 / (2 * math.pi) ** 2


def test_dielectric_single_oscillator_static_limit_and_peak():
    """One mode gives eps(0) = eps_inf + C f / (V nu0^2) and Im eps(nu0) = C f / (V gamma nu0)."""
    assert pytest.approx(44224.53, rel=1e-6) == IR_THZ2_A3
    nu0, strength, volume, eps_inf, damping = 30.0, 0.4, 45.0, 2.1, 0.5
    result = compute_dielectric_function(
        [nu0], [strength], volume, frequency_grid_thz=[0.0, nu0, 3 * nu0], eps_inf=eps_inf, damping_thz=damping
    )
    scale = IR_THZ2_A3 * strength / volume
    epsilon = result["epsilon"]
    assert epsilon[0] == pytest.approx(eps_inf + scale / nu0**2, rel=1e-13)
    assert result["epsilon_static"] == pytest.approx(eps_inf + scale / nu0**2, rel=1e-13)
    assert epsilon[1] == pytest.approx(eps_inf + 1j * scale / (damping * nu0), rel=1e-13)
    assert epsilon[2] == pytest.approx(eps_inf + scale / (nu0**2 - 9 * nu0**2 - 3j * damping * nu0), rel=1e-13)
    assert (result["n_modes_excluded"], result["eps_inf"], result["damping_thz"]) == (0, eps_inf, damping)


def test_dielectric_lyddane_sachs_teller_zero():
    """With vanishing damping eps crosses zero at nu_LO = nu_TO sqrt(eps(0) / eps_inf)."""
    nu_to, eps_inf = 13.5, 2.1
    static = compute_dielectric_function([nu_to], [0.3], 45.0, eps_inf=eps_inf)["epsilon_static"]
    nu_lo = nu_to * math.sqrt(static / eps_inf)
    grid = [0.5 * nu_lo, nu_lo, 2 * nu_lo]
    result = compute_dielectric_function(
        [nu_to], [0.3], 45.0, frequency_grid_thz=grid, eps_inf=eps_inf, damping_thz=1e-9
    )
    assert abs(result["epsilon"][1]) < 1e-8
    assert result["epsilon"][0].real > static
    assert 0 < result["epsilon"][2].real < eps_inf
    loss = compute_optical_response(grid, result["epsilon"])["loss_function"]
    assert loss[1] > 1e6 * max(loss[0], loss[2])


def test_dielectric_modes_add_and_default_grid(monkeypatch):
    """The modes add up, absorption is non-negative, modes at nu <= 0 are left out, and blocks do not matter."""
    rng = np.random.default_rng(0)
    frequencies = np.concatenate([[-2.0, 0.0], rng.uniform(3.0, 40.0, size=20)])
    strengths = rng.uniform(0.0, 0.5, size=22)
    result = compute_dielectric_function(frequencies, strengths, 300.0, damping_thz=0.4)
    grid = result["frequencies_thz"]
    assert grid[0] == 0.0
    assert np.diff(grid) == pytest.approx(0.4 / 5)
    assert grid[-1] >= frequencies.max() + 5 * 0.4
    assert result["n_modes_excluded"] == 2
    assert np.all(result["epsilon"].imag >= 0)
    singles = sum(
        compute_dielectric_function([nu], [f], 300.0, frequency_grid_thz=grid, damping_thz=0.4)["epsilon"] - 1.0
        for nu, f in zip(frequencies[2:], strengths[2:], strict=True)
    )
    np.testing.assert_allclose(result["epsilon"] - 1.0, singles, rtol=1e-11)
    assert result["epsilon"][0].real == pytest.approx(result["epsilon_static"], rel=1e-12)
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)
    chunked = compute_dielectric_function(frequencies, strengths, 300.0, damping_thz=0.4)
    np.testing.assert_allclose(chunked["epsilon"], result["epsilon"], rtol=1e-12)


def test_dielectric_kramers_kronig_static_sum_rule():
    """eps(0) - eps_inf = (2 / pi) int Im eps / nu dnu, the Kramers-Kronig relation at zero frequency."""
    grid = np.linspace(0.0, 4000.0, 2_000_001)
    result = compute_dielectric_function([12.0, 30.0], [0.2, 0.5], 80.0, frequency_grid_thz=grid, damping_thz=1.0)
    integrand = np.divide(result["epsilon"].imag, grid, out=np.zeros_like(grid), where=grid > 0)
    integrand[0] = 2 * integrand[1] - integrand[2]  # Im eps / nu tends to a constant at nu = 0
    assert 2 / math.pi * np.trapezoid(integrand, grid) == pytest.approx(result["epsilon_static"] - 1.0, rel=1e-4)


def _dielectric_invalid_cases():
    return [
        pytest.param([], [], 1.0, {}, "non-empty 1-D", id="no-modes"),
        pytest.param([[1.0]], [[1.0]], 1.0, {}, "non-empty 1-D", id="frequencies-2d"),
        pytest.param([1.0, np.inf], [1.0, 1.0], 1.0, {}, "non-empty 1-D", id="frequencies-inf"),
        pytest.param([1.0, 2.0], [1.0], 1.0, {}, "2 finite values", id="strengths-length"),
        pytest.param([1.0], [np.nan], 1.0, {}, "1 finite values", id="strengths-nan"),
        pytest.param([1.0], [1.0], 0.0, {}, "volume_a3 must be positive", id="volume-zero"),
        pytest.param([1.0], [1.0], np.nan, {}, "volume_a3 must be positive", id="volume-nan"),
        pytest.param([1.0], [1.0], 1.0, {"eps_inf": 0.0}, "eps_inf must be positive", id="eps-inf-zero"),
        pytest.param([1.0], [1.0], 1.0, {"eps_inf": True}, "eps_inf must be positive", id="eps-inf-bool"),
        pytest.param([1.0], [1.0], 1.0, {"damping_thz": -0.1}, "damping_thz must be positive", id="damping-negative"),
        pytest.param([1.0], [1.0], 1.0, {"frequency_grid_thz": [1.0, 0.0]}, "strictly increasing", id="grid-order"),
    ]


@pytest.mark.parametrize(("frequencies", "strengths", "volume", "kwargs", "match"), _dielectric_invalid_cases())
def test_dielectric_invalid_input_raises(frequencies, strengths, volume, kwargs, match):
    """Bad frequencies, strengths, volume, eps_inf, damping and grids raise."""
    with pytest.raises(ValueError, match=match):
        compute_dielectric_function(frequencies, strengths, volume, **kwargs)


def test_optical_response_analytic():
    """A real eps = 4 is transparent with R = 1/9; eps = (n + ik)^2 returns n, k, alpha = 4 pi k nu / c and loss."""
    index = 1.5 + 0.2j
    result = compute_optical_response([10.0, 30.0, 0.0], [4.0, index**2, np.nan])
    np.testing.assert_allclose(result["refractive_index"][:2], [2.0, 1.5], rtol=1e-14)
    np.testing.assert_allclose(result["extinction"][:2], [0.0, 0.2], rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(result["absorption_cm"][:2], [0.0, 4 * math.pi * 0.2 * 30e12 / (100 * c)], rtol=1e-13)
    np.testing.assert_allclose(result["reflectivity"][:2], [1 / 9, abs((index - 1) / (index + 1)) ** 2], rtol=1e-14)
    np.testing.assert_allclose(result["loss_function"][:2], [0.0, (-1 / index**2).imag], rtol=1e-14, atol=1e-16)
    assert all(np.isnan(values[2]) for values in result.values())


@pytest.mark.parametrize(
    ("frequencies", "epsilon", "match"),
    [
        pytest.param([], [], "non-empty 1-D", id="empty"),
        pytest.param([-1.0], [1.0], "non-empty 1-D", id="negative-frequency"),
        pytest.param([1.0, 2.0], [1.0], "2 values without inf", id="length"),
        pytest.param([1.0], [np.inf], "1 values without inf", id="inf"),
    ],
)
def test_optical_response_invalid_input_raises(frequencies, epsilon, match):
    """Bad frequencies and a mismatched or infinite dielectric function raise."""
    with pytest.raises(ValueError, match=match):
        compute_optical_response(frequencies, epsilon)


# ---------------------------------------------------------------------------
# compute_dielectric_from_velocities
# ---------------------------------------------------------------------------

# e^2 ps / (amu A^3) in S/m.
S_PER_M_A3 = e**2 * 1e-12 / (atomic_mass * 1e-30)


def _conductivity_sum(result):
    """Sum of Re sigma over the full FFT grid (end bins once, inner bins twice), in e^2 ps / (amu A^3)."""
    sigma = result["conductivity_s_per_m"].real
    return (sigma[0] + 2 * sigma[1:-1].sum() + sigma[-1]) / S_PER_M_A3


def _harmonic_trajectory(n_frames=4000, n_atoms=6, seed=0):
    """Velocities of a harmonic solid at equipartition: every mode a cosine of equal amplitude on the rfft grid."""
    masses, vectors, charges = _ir_system(n_atoms, seed)
    rng = np.random.default_rng(seed)
    n_modes = vectors.shape[1]
    frequencies = np.array([_grid_frequency(n_frames, 200 + 40 * (mode + 1)) for mode in range(n_modes)])
    t_ps = np.arange(n_frames) * DT_FS * 1e-3
    rates = np.cos(2 * np.pi * frequencies * t_ps[:, None] + rng.uniform(0, 2 * np.pi, size=n_modes))
    velocities = (rates @ vectors.T).reshape(n_frames, n_atoms, 3) / np.sqrt(masses)[None, :, None]
    return velocities, masses, vectors, charges, frequencies


@pytest.mark.parametrize("remove_com", [True, False], ids=["com-removed", "com-kept"])
@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
@pytest.mark.parametrize("window", [None, "hann", ("gaussian", 20.0)], ids=["none", "hann", "gaussian"])
def test_md_dielectric_f_sum_rule(window, tensor, remove_com):
    """The sum of Re sigma over the grid is n_fft dt <|J|^2> / (6 V kT), with kT from the same velocities."""
    n_frames, n_atoms, volume = 257, 7, 80.0
    velocities, masses = _random_system(n_frames, n_atoms, seed=3)
    rng = np.random.default_rng(4)
    charges = rng.normal(size=(n_atoms, 3, 3) if tensor else n_atoms)
    charges -= charges.mean(axis=0)
    result = compute_dielectric_from_velocities(
        velocities, masses, charges, DT_FS, volume, window=window, remove_com=remove_com
    )
    centred = velocities
    if remove_com:
        centred = velocities - (np.einsum("tia,i->ta", velocities, masses) / masses.sum())[:, None, :]
    thermal = np.einsum("tia,tia,i->", centred, centred, masses) / (n_frames * 3 * (n_atoms - remove_com))
    tensors = charges if tensor else charges[:, None, None] * np.eye(3)
    current = np.einsum("iab,tib->ta", tensors, centred)
    zero_lag = np.sum(current**2) / n_frames / (3 * thermal)
    n_lags = n_frames // 2
    assert result["n_lag_frames"] == n_lags
    assert result["frequencies_thz"] == pytest.approx(np.fft.rfftfreq(2 * n_lags, d=DT_FS * 1e-3))
    assert _conductivity_sum(result) * volume == pytest.approx(2 * n_lags * DT_FS * 1e-3 * zero_lag / 2, rel=1e-12)


def test_md_dielectric_matches_harmonic_route():
    """A harmonic trajectory at equipartition gives the strengths and the dielectric function of the modes."""
    velocities, masses, vectors, charges, frequencies = _harmonic_trajectory()
    volume = 100.0
    strengths = compute_ir_oscillator_strengths(vectors, masses, charges)["strengths"]
    result = compute_dielectric_from_velocities(velocities, masses, charges, DT_FS, volume)
    grid, n_lags = result["frequencies_thz"], result["n_lag_frames"]
    # f-sum rule: the zero-lag current correlation over 3 kT is the sum of the strengths.
    assert _conductivity_sum(result) * volume == pytest.approx(n_lags * DT_FS * 1e-3 * strengths.sum(), rel=1e-10)
    # Each mode holds a quarter of its strength under its peak of Re sigma on the positive axis.
    sigma = result["conductivity_s_per_m"].real * volume / S_PER_M_A3
    areas = [sigma[np.abs(grid - nu) < 0.6].sum() * (grid[1] - grid[0]) for nu in frequencies]
    np.testing.assert_allclose(areas, strengths / 4, rtol=0.02)
    # Below every mode the dielectric function is that of the undamped oscillators.
    harmonic = compute_dielectric_function(frequencies, strengths, volume, frequency_grid_thz=grid[1:40])
    np.testing.assert_allclose(result["epsilon"][1:40].real - 1, harmonic["epsilon"].real - 1, rtol=2e-3)
    assert np.isnan(result["epsilon"][0])
    # Far below the modes only a small leakage of the peaks is left in Re sigma (Im eps divides it by nu).
    assert np.abs(sigma[:40]).max() < 1e-4 * sigma.max()


def test_md_dielectric_pair_peak_at_mode_frequency():
    """A +q / -q pair oscillating at one frequency absorbs there, and 2.1 is added by eps_inf."""
    n_frames = 2000
    nu0 = _grid_frequency(n_frames, 300)
    velocities = _pair_velocities(n_frames, nu0)
    velocities[:, 1, 0] *= SIO2_MASSES[0] / SIO2_MASSES[1]  # zero total momentum
    kwargs = {"masses": SIO2_MASSES[:2], "charges": [2.4, -2.4], "dt_fs": DT_FS, "volume_a3": 50.0}
    result = compute_dielectric_from_velocities(velocities, **kwargs)
    shifted = compute_dielectric_from_velocities(velocities, **kwargs, eps_inf=2.1)
    assert result["frequencies_thz"][np.nanargmax(result["epsilon"].imag)] == pytest.approx(nu0, rel=1e-12)
    np.testing.assert_allclose(shifted["epsilon"][1:], result["epsilon"][1:] + 1.1, rtol=1e-13)
    assert (shifted["eps_inf"], result["eps_inf"]) == (2.1, 1.0)
    np.testing.assert_array_equal(shifted["conductivity_s_per_m"], result["conductivity_s_per_m"])


@pytest.mark.parametrize("tensor", [False, True], ids=["scalar", "tensor"])
def test_md_dielectric_partials_sum_to_total(tensor, monkeypatch):
    """Pairs of a grouping holding every atom once sum to eps - eps_inf; atom chunks do not matter."""
    n_atoms = 9
    velocities, masses = _random_system(200, n_atoms, seed=5)
    rng = np.random.default_rng(6)
    charges = rng.normal(size=(n_atoms, 3, 3) if tensor else n_atoms)
    charges -= charges.mean(axis=0)
    groups = {
        "element": {"Si": np.arange(0, 3), "O": np.arange(3, 8), "Na": np.array([8])},
        "subset": {"first": np.array([0, 4])},
    }
    result = compute_dielectric_from_velocities(
        velocities, masses, charges, DT_FS, 60.0, groups=groups, eps_inf=2.0, n_lag_frames=50
    )
    partial = result["partial_susceptibility"]
    assert list(partial["element"]) == ["Si-Si", "Si-O", "Si-Na", "O-O", "O-Na", "Na-Na"]
    assert list(partial["subset"]) == ["first-first"]
    assert result["epsilon"].shape == partial["element"]["Si-O"].shape == (51,)
    np.testing.assert_allclose(sum(partial["element"].values())[1:], result["epsilon"][1:] - 2.0, rtol=1e-9, atol=1e-9)
    monkeypatch.setattr(_common, "_CHUNK_BYTES", 1)  # one atom per chunk
    chunked = compute_dielectric_from_velocities(
        velocities, masses, charges, DT_FS, 60.0, groups=groups, eps_inf=2.0, n_lag_frames=50
    )
    np.testing.assert_allclose(chunked["epsilon"][1:], result["epsilon"][1:], rtol=1e-9, atol=1e-9)
    for pair, values in partial["element"].items():
        np.testing.assert_allclose(chunked["partial_susceptibility"]["element"][pair][1:], values[1:], atol=1e-9)


def test_md_dielectric_velocity_unit_dtype_and_tensor_form():
    """The velocity unit cancels, float32 input agrees to single precision, and q is the tensor q times identity."""
    velocities, masses = _random_system(300, 6, seed=7)
    charges = np.array([2.4, 2.4, -1.2, -1.2, -1.2, -1.2])
    reference = compute_dielectric_from_velocities(velocities, masses, charges, DT_FS, 70.0)["epsilon"][1:]
    scaled = compute_dielectric_from_velocities(1e-3 * velocities, masses, charges, DT_FS, 70.0)["epsilon"][1:]
    single = compute_dielectric_from_velocities(velocities.astype(np.float32), masses, charges, DT_FS, 70.0)
    tensor = compute_dielectric_from_velocities(velocities, masses, charges[:, None, None] * np.eye(3), DT_FS, 70.0)
    np.testing.assert_allclose(scaled, reference, rtol=1e-10)
    np.testing.assert_allclose(single["epsilon"][1:], reference, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(tensor["epsilon"][1:], reference, rtol=1e-10)


def _md_dielectric_invalid_cases():
    velocities, masses, charges = np.ones((10, 2, 3)), np.ones(2), np.array([1.0, -1.0])
    velocities[:, 1] = -1.0
    return [
        pytest.param(np.ones((10, 2)), masses, charges, 1.0, {}, "velocities", id="velocities-2d"),
        pytest.param(np.zeros((10, 2, 3)), masses, charges, 1.0, {}, "spectrum is zero", id="velocities-zero"),
        pytest.param(velocities, masses, np.ones(3), 1.0, {}, r"shape \(2,\) or \(2, 3, 3\)", id="charges-length"),
        pytest.param(velocities, masses, charges, 0.0, {}, "volume_a3 must be positive", id="volume-zero"),
        pytest.param(velocities, masses, charges, 1.0, {"eps_inf": -1.0}, "eps_inf must be positive", id="eps-inf"),
        pytest.param(velocities, masses, charges, 1.0, {"window": "boxcar"}, "Unknown window", id="window"),
        pytest.param(velocities, masses, charges, 1.0, {"n_lag_frames": 1}, r"integer in \[2, 9\]", id="lags-low"),
        pytest.param(velocities, masses, charges, 1.0, {"n_lag_frames": 10}, r"integer in \[2, 9\]", id="lags-high"),
        pytest.param(velocities, masses, charges, 1.0, {"n_lag_frames": 4.0}, r"integer in \[2, 9\]", id="lags-float"),
        pytest.param(velocities, masses, charges, 1.0, {"groups": {"element": {}}}, "no labels", id="groups-empty"),
    ]


@pytest.mark.parametrize(
    ("velocities", "masses", "charges", "volume", "kwargs", "match"), _md_dielectric_invalid_cases()
)
def test_md_dielectric_invalid_input_raises(velocities, masses, charges, volume, kwargs, match):
    """Bad trajectories, charges, volume, eps_inf, window, lag count and groups raise."""
    with pytest.raises(ValueError, match=match):
        compute_dielectric_from_velocities(velocities, masses, charges, DT_FS, volume, **kwargs)
