"""Tests for pure functions in amorphouspy.properties.vibrational."""

from pathlib import Path

import numpy as np
import pytest
from amorphouspy.atoms.mass import get_atomic_mass
from amorphouspy.properties import vibrational
from amorphouspy.properties.structural.qn import compute_qn
from amorphouspy.properties.vibrational import (
    classify_vibrational_groups,
    compute_partial_vacf,
    compute_partial_vdos,
    compute_vacf_from_velocities,
    compute_vdos_from_velocities,
    convert_frequency,
)
from ase import Atoms
from ase.io import read

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
    return [
        pytest.param(good[:, :, 0], masses, DT_FS, {}, "shape", id="2d-velocities"),
        pytest.param(good[:, :, :2], masses, DT_FS, {}, "shape", id="two-components"),
        pytest.param(np.zeros((10, 0, 3)), np.zeros(0), DT_FS, {}, "n_atoms >= 1", id="zero-atoms"),
        pytest.param(good, np.array([M_SI, M_O, M_O]), DT_FS, {}, "masses must have shape", id="masses-length"),
        pytest.param(good[:1], masses, DT_FS, {}, "At least 2 frames", id="one-frame"),
        pytest.param(good, masses, 0.0, {}, "dt_fs must be positive", id="dt-zero"),
        pytest.param(good, masses, np.nan, {}, "dt_fs must be positive", id="dt-nan"),
        pytest.param(good, np.array([M_SI, 0.0]), DT_FS, {}, "positive and finite", id="mass-zero"),
        pytest.param(good, np.array([-M_SI, M_O]), DT_FS, {}, "positive and finite", id="mass-negative"),
        pytest.param(nan_velocities, masses, DT_FS, {}, "NaN or inf", id="velocity-nan"),
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
    monkeypatch.setattr(vibrational, "_CHUNK_BYTES", 2**40)
    _, reference = compute_vdos_from_velocities(velocities, masses, DT_FS, window=window)
    _, reference_partial = compute_partial_vdos(velocities, masses, DT_FS, groups, window=window)
    monkeypatch.setattr(vibrational, "_CHUNK_BYTES", chunk_bytes)
    _, vdos = compute_vdos_from_velocities(velocities, masses, DT_FS, window=window)
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups, window=window)
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
    monkeypatch.setattr(vibrational, "_CHUNK_BYTES", 2**40)
    _, reference = compute_vacf_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    _, reference_partial = compute_partial_vacf(velocities, masses, DT_FS, groups, remove_com=remove_com)
    monkeypatch.setattr(vibrational, "_CHUNK_BYTES", chunk_bytes)
    _, vacf = compute_vacf_from_velocities(velocities, masses, DT_FS, remove_com=remove_com)
    _, partial = compute_partial_vacf(velocities, masses, DT_FS, groups, remove_com=remove_com)
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
    "mass-negative",
    "velocity-nan",
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
