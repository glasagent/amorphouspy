"""Tests for pure functions in amorphouspy.properties.vibrational."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from amorphouspy.atoms.mass import get_atomic_mass
from amorphouspy.properties import vibrational
from amorphouspy.properties.structural.qn import compute_qn
from amorphouspy.properties.vibrational import (
    classify_vibrational_groups,
    compute_partial_vdos,
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


@pytest.mark.parametrize(("from_unit", "to_unit"), list(itertools.product(UNITS, UNITS)))
def test_convert_frequency_round_trip(from_unit, to_unit):
    """Converting there and back returns the input for every unit pair."""
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


@pytest.mark.parametrize("chunk_bytes", [1, 5000, 2**40])
@pytest.mark.parametrize("n_frames", [50, 51])
def test_vdos_chunking_matches_single_chunk(monkeypatch, chunk_bytes, n_frames):
    """Total and partial results do not depend on how atoms are split into chunks."""
    velocities, masses = _random_system(n_frames, 7, seed=2)
    groups = {"a": {"odd": np.array([1, 3, 5]), "first": np.array([0, 1, 2])}}
    monkeypatch.setattr(vibrational, "_CHUNK_BYTES", 2**40)
    _, reference = compute_vdos_from_velocities(velocities, masses, DT_FS)
    _, reference_partial = compute_partial_vdos(velocities, masses, DT_FS, groups)
    monkeypatch.setattr(vibrational, "_CHUNK_BYTES", chunk_bytes)
    _, vdos = compute_vdos_from_velocities(velocities, masses, DT_FS)
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups)
    np.testing.assert_allclose(vdos, reference, rtol=1e-12)
    for label in groups["a"]:
        np.testing.assert_allclose(partial["a"][label], reference_partial["a"][label], rtol=1e-12)


@pytest.mark.parametrize("n_frames", [64, 65])
@pytest.mark.parametrize("normalization", ["unit", "3N", "none"])
def test_partial_vdos_sums_to_total(n_frames, normalization):
    """Partials over a partition of the atoms add up to the total VDOS for every normalisation."""
    velocities, masses = _random_system(n_frames, 9, seed=7)
    groups = {"element": {"A": np.array([0, 4, 8]), "B": np.array([1, 2, 3]), "C": np.array([5, 6, 7])}}
    _, total = compute_vdos_from_velocities(velocities, masses, DT_FS, normalization=normalization)
    _, partial = compute_partial_vdos(velocities, masses, DT_FS, groups, normalization=normalization)
    np.testing.assert_allclose(sum(partial["element"].values()), total, rtol=1e-12, atol=1e-14 * total.max())


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
    inf_velocities = good.copy()
    inf_velocities[0, 0, 0] = np.inf
    translation = np.broadcast_to(np.array([0.1, 0.2, 0.3]), (10, 2, 3))
    return [
        pytest.param(good[:, :, 0], masses, DT_FS, {}, "shape", id="2d-velocities"),
        pytest.param(good[:, :, :2], masses, DT_FS, {}, "shape", id="two-components"),
        pytest.param(np.zeros((10, 0, 3)), np.zeros(0), DT_FS, {}, "n_atoms >= 1", id="zero-atoms"),
        pytest.param(good, masses[None, :], DT_FS, {}, "masses must have shape", id="2d-masses"),
        pytest.param(good, np.array([M_SI, M_O, M_O]), DT_FS, {}, "masses must have shape", id="masses-length"),
        pytest.param(good[:1], masses, DT_FS, {}, "At least 2 frames", id="one-frame"),
        pytest.param(good, masses, 0.0, {}, "dt_fs must be positive", id="dt-zero"),
        pytest.param(good, masses, -1.0, {}, "dt_fs must be positive", id="dt-negative"),
        pytest.param(good, masses, np.nan, {}, "dt_fs must be positive", id="dt-nan"),
        pytest.param(good, np.array([M_SI, 0.0]), DT_FS, {}, "positive and finite", id="mass-zero"),
        pytest.param(good, np.array([-M_SI, M_O]), DT_FS, {}, "positive and finite", id="mass-negative"),
        pytest.param(nan_velocities, masses, DT_FS, {}, "NaN or inf", id="velocity-nan"),
        pytest.param(inf_velocities, masses, DT_FS, {}, "NaN or inf", id="velocity-inf"),
        pytest.param(translation, masses, DT_FS, {}, "zero after removing", id="uniform-translation"),
        pytest.param(np.zeros((10, 2, 3)), masses, DT_FS, {}, "all velocities are zero", id="all-zero"),
        pytest.param(
            np.zeros((10, 2, 3)), masses, DT_FS, {"remove_com": False}, "all velocities are zero", id="all-zero-no-com"
        ),
        pytest.param(good, masses, DT_FS, {"normalization": "per-atom"}, "Unknown normalization", id="normalization"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([], dtype=int)}, "non-empty", id="indices-empty"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([[0]])}, "1-D", id="indices-2d"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([0.0])}, "integer", id="indices-float"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([2])}, "outside", id="indices-range"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([-1])}, "outside", id="indices-negative"),
        pytest.param(good, masses, DT_FS, {"indices": np.array([1, 1])}, "duplicate", id="indices-duplicate"),
        pytest.param(
            good,
            np.float64(M_SI),
            DT_FS,
            {"indices": np.array([0])},
            "masses must have shape",
            id="scalar-masses-indices",
        ),
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
def test_vdos_input_unchanged(dtype):
    """The velocity and mass arrays are not modified in place, for the total and for partials."""
    velocities = np.random.default_rng(4).normal(size=(32, 3, 3)).astype(dtype)
    masses = np.array([M_SI, M_O, M_O])
    velocities_before, masses_before = velocities.copy(), masses.copy()
    compute_vdos_from_velocities(velocities, masses, DT_FS)
    compute_vdos_from_velocities(velocities, masses, DT_FS, indices=np.array([0, 2]))
    compute_partial_vdos(velocities, masses, DT_FS, {"g": {"a": np.array([1])}})
    np.testing.assert_array_equal(velocities, velocities_before)
    np.testing.assert_array_equal(masses, masses_before)
