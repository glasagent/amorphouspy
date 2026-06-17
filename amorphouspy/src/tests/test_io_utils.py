"""Tests for amorphouspy.io_utils."""

import textwrap
from unittest.mock import patch

import numpy as np
import pytest
from amorphouspy.atoms.io import (
    write_angle_distribution,
    write_distribution_to_file,
    write_xyz,
)
from amorphouspy.lammps.io import (
    frames_from_melt_quench_result,
    load_lammps_dump,
    structure_from_parsed_output,
)
from ase import Atoms

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _dump_frame(timestep: int, x: float) -> str:
    """Return one LAMMPS dump frame with a single Si atom at (x, 0, 0)."""
    return textwrap.dedent(f"""\
        ITEM: TIMESTEP
        {timestep}
        ITEM: NUMBER OF ATOMS
        1
        ITEM: BOX BOUNDS pp pp pp
        0.0 10.0
        0.0 10.0
        0.0 10.0
        ITEM: ATOMS id type x y z
        1 1 {x} 0.0 0.0
        """)


def _dump_frame_with_element(timestep: int, x: float) -> str:
    """Return one frame including the `element` column."""
    return textwrap.dedent(f"""\
        ITEM: TIMESTEP
        {timestep}
        ITEM: NUMBER OF ATOMS
        1
        ITEM: BOX BOUNDS pp pp pp
        0.0 10.0
        0.0 10.0
        0.0 10.0
        ITEM: ATOMS id type element x y z
        1 1 Si {x} 0.0 0.0
        """)


@pytest.fixture
def dump_5frames(tmp_path):
    """Write a 5-frame dump (type 1 = Si) and return the path."""
    p = tmp_path / "traj.lammpstrj"
    p.write_text("".join(_dump_frame(i, float(i)) for i in range(5)))
    return p


# ---------------------------------------------------------------------------
# load_lammps_dump
# ---------------------------------------------------------------------------


def test_load_full_trajectory_returns_list(dump_5frames):
    """Default call returns all 5 frames as a list."""
    result = load_lammps_dump(dump_5frames, type_map={1: "Si"})
    assert isinstance(result, list)
    assert len(result) == 5


def test_load_single_frame_returns_atoms(dump_5frames):
    """frame=2 returns a single Atoms object, not a list."""
    result = load_lammps_dump(dump_5frames, type_map={1: "Si"}, frame=2)
    assert isinstance(result, Atoms)
    assert len(result) == 1


def test_load_single_frame_correct_position(dump_5frames):
    """frame=3 has the atom at x=3 (matching the fixture)."""
    atoms = load_lammps_dump(dump_5frames, type_map={1: "Si"}, frame=3)
    assert isinstance(atoms, Atoms)
    np.testing.assert_allclose(atoms.get_positions()[0, 0], 3.0)


def test_load_frame_range(dump_5frames):
    """start=1, stop=4 returns frames 1, 2, 3 (3 frames)."""
    result = load_lammps_dump(dump_5frames, type_map={1: "Si"}, start=1, stop=4)
    assert isinstance(result, list)
    assert len(result) == 3


def test_load_frame_stride(dump_5frames):
    """start=0, stop=5, step=2 returns frames 0, 2, 4 (3 frames)."""
    result = load_lammps_dump(dump_5frames, type_map={1: "Si"}, start=0, stop=5, step=2)
    assert isinstance(result, list)
    assert len(result) == 3
    # atom x positions should be 0.0, 2.0, 4.0
    xs = [a.get_positions()[0, 0] for a in result]
    np.testing.assert_allclose(xs, [0.0, 2.0, 4.0])


def test_load_frame_and_range_raises(dump_5frames):
    """Combining frame with start/stop/step raises ValueError."""
    with pytest.raises(ValueError, match="cannot be combined"):
        load_lammps_dump(dump_5frames, type_map={1: "Si"}, frame=0, start=1)


@patch("amorphouspy.lammps.io.ase.io.read")
def test_load_dump_uses_element_column_when_type_map_missing(mock_read):
    """When type_map is None, symbols are assigned from an existing `element` array."""
    atoms = Atoms(numbers=[14], positions=[[1.5, 0.0, 0.0]], cell=np.diag([10.0, 10.0, 10.0]), pbc=True)
    atoms.set_array("type", np.array([1]))
    atoms.set_array("element", np.array(["Si"], dtype="U2"))
    mock_read.return_value = atoms

    out = load_lammps_dump("dummy.lammpstrj", type_map=None, frame=0)
    assert isinstance(out, Atoms)
    assert out.get_chemical_symbols() == ["Si"]


def test_load_dump_raises_without_type_map_and_without_element_column(dump_5frames):
    """A dump without `element` requires explicit type_map."""
    with pytest.raises(ValueError, match="type_map is required"):
        load_lammps_dump(dump_5frames, type_map=None, frame=0)


def test_load_full_trajectory_with_atoms_dict(dump_5frames):
    """return_atoms_dict=True with full trajectory returns list of (Atoms, dict)."""
    result = load_lammps_dump(dump_5frames, type_map={1: "Si"}, return_atoms_dict=True)
    assert isinstance(result, list)
    assert len(result) == 5
    atoms, d = result[0]
    assert isinstance(atoms, Atoms)
    assert "atoms" in d
    assert "box" in d
    assert "total_atoms" in d


def test_load_single_frame_with_atoms_dict(dump_5frames):
    """return_atoms_dict=True with frame= returns a single (Atoms, dict) tuple."""
    result = load_lammps_dump(dump_5frames, type_map={1: "Si"}, frame=1, return_atoms_dict=True)
    assert isinstance(result, tuple)
    atoms, d = result
    assert isinstance(atoms, Atoms)
    assert d["total_atoms"] == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_atoms() -> Atoms:
    """Two-atom Si-O structure in a 5 Å cubic box."""
    coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    numbers = np.array([14, 8])
    cell = np.diag([5.0, 5.0, 5.0])
    return Atoms(numbers=numbers, positions=coords, cell=cell, pbc=True)


def _parsed_output(atoms: Atoms) -> dict:
    """Minimal parsed_output dict matching the structure expected by structure_from_parsed_output."""
    pos = atoms.get_positions()
    vel = np.zeros_like(pos)
    cell = np.array(atoms.get_cell())
    indices = np.arange(len(atoms))
    return {
        "generic": {
            "positions": [pos],
            "velocities": [vel],
            "cells": [cell],
            "indices": [indices],
        }
    }


# ---------------------------------------------------------------------------
# write_distribution_to_file
# ---------------------------------------------------------------------------


def test_write_distribution_creates_header_and_row(tmp_path):
    """New file gets a header line followed by a data row."""
    fp = tmp_path / "dist.txt"
    write_distribution_to_file(0.5, str(fp), {0: 10, 1: 5}, "Q")
    lines = fp.read_text().splitlines()
    assert lines[0].startswith("Composition Q_0 Q_1 Q_tot")
    assert lines[1] == "0.5 10 5 15"


def test_write_distribution_append_skips_header(tmp_path):
    """Appending to an existing file adds a row without a second header."""
    fp = tmp_path / "dist.txt"
    write_distribution_to_file(0.1, str(fp), {0: 2}, "Q")
    write_distribution_to_file(0.2, str(fp), {0: 3}, "Q", append=True)
    lines = fp.read_text().splitlines()
    assert sum(1 for line in lines if line.startswith("Composition")) == 1
    assert len(lines) == 3


def test_write_distribution_append_to_nonexistent_writes_header(tmp_path):
    """Appending to a non-existent file still writes a header."""
    fp = tmp_path / "new.txt"
    write_distribution_to_file(0.3, str(fp), {0: 1}, "Si", append=True)
    lines = fp.read_text().splitlines()
    assert lines[0].startswith("Composition")


# ---------------------------------------------------------------------------
# write_angle_distribution
# ---------------------------------------------------------------------------


def test_write_angle_distribution_creates_header_and_row(tmp_path):
    """New file gets a header of bin centers followed by a data row."""
    fp = tmp_path / "angles.txt"
    centers = np.array([90.0, 120.0])
    hist = np.array([0.3, 0.7])
    write_angle_distribution(centers, hist, 0.5, str(fp))
    lines = fp.read_text().splitlines()
    assert lines[0] == "Composition 90.0 120.0"
    assert lines[1].startswith("0.5 0.300000 0.700000")


def test_write_angle_distribution_append_skips_header(tmp_path):
    """Appending to an existing file adds a row without a second header."""
    fp = tmp_path / "angles.txt"
    centers = np.array([90.0])
    write_angle_distribution(centers, np.array([1.0]), 0.1, str(fp))
    write_angle_distribution(centers, np.array([1.0]), 0.2, str(fp), append=True)
    lines = fp.read_text().splitlines()
    assert sum(1 for line in lines if line.startswith("Composition")) == 1
    assert len(lines) == 3


def test_write_angle_distribution_append_to_nonexistent_writes_header(tmp_path):
    """Appending to a non-existent file still writes a header."""
    fp = tmp_path / "new_angles.txt"
    write_angle_distribution(np.array([60.0]), np.array([1.0]), 0.0, str(fp), append=True)
    lines = fp.read_text().splitlines()
    assert lines[0].startswith("Composition")


# ---------------------------------------------------------------------------
# structure_from_parsed_output
# ---------------------------------------------------------------------------


def test_structure_from_parsed_output_positions():
    """Positions from parsed_output are set on the returned atoms."""
    atoms = _simple_atoms()
    new_pos = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    parsed = _parsed_output(atoms)
    parsed["generic"]["positions"] = [new_pos]
    result = structure_from_parsed_output(atoms, parsed)
    np.testing.assert_allclose(result.get_positions(), new_pos)


def test_structure_from_parsed_output_pbc_set():
    """PBC is always set to True on the returned atoms."""
    atoms = _simple_atoms()
    result = structure_from_parsed_output(atoms, _parsed_output(atoms))
    assert all(result.get_pbc())


def test_structure_from_parsed_output_wrap_false_preserves_positions():
    """With wrap=False, out-of-cell positions are kept as-is."""
    atoms = _simple_atoms()
    out_pos = np.array([[-1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    parsed = _parsed_output(atoms)
    parsed["generic"]["positions"] = [out_pos]
    result = structure_from_parsed_output(atoms, parsed, wrap=False)
    np.testing.assert_allclose(result.get_positions(), out_pos)


def test_structure_from_parsed_output_wrap_true_wraps_positions():
    """With wrap=True, out-of-cell positions are folded into the cell."""
    atoms = _simple_atoms()
    out_pos = np.array([[-1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    parsed = _parsed_output(atoms)
    parsed["generic"]["positions"] = [out_pos]
    result = structure_from_parsed_output(atoms, parsed, wrap=True)
    assert (result.get_positions() >= 0.0).all()


# ---------------------------------------------------------------------------
# frames_from_melt_quench_result
# ---------------------------------------------------------------------------


def test_frames_from_melt_quench_result_populates_info_and_arrays():
    """Trajectory conversion attaches thermo metadata and optional per-atom arrays."""
    initial = _simple_atoms()
    n_atoms = len(initial)
    positions = [
        np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
        np.array([[0.1, 0.0, 0.0], [2.6, 0.0, 0.0]]),
    ]
    cells = [np.diag([5.0, 5.0, 5.0]), np.diag([5.1, 5.1, 5.1])]
    velocities = [np.zeros((n_atoms, 3)), np.ones((n_atoms, 3))]
    indices = [np.array([0, 1]), np.array([0, 1])]
    forces = [np.zeros((n_atoms, 3)), np.full((n_atoms, 3), 2.0)]
    unwrapped = [np.array(positions[0]), np.array(positions[1]) + 5.0]

    stage = {
        "positions": positions,
        "cells": cells,
        "velocities": velocities,
        "indices": indices,
        "temperature": [300.0, 400.0],
        "energy_pot": [-10.0, -9.0],
        "energy_tot": [-9.0, -8.0],
        "volume": [125.0, 132.0],
        "pressures": [np.zeros((3, 3)), np.ones((3, 3))],
        "steps": [0, 1],
        "forces": forces,
        "unwrapped_positions": unwrapped,
    }
    result = {"result": [stage]}

    frames = frames_from_melt_quench_result(result, initial, stage=0, stride=1)
    assert len(frames) == 2
    assert frames[1].info["temperature"] == 400.0
    assert frames[1].info["step"] == 1
    assert "forces" in frames[1].arrays
    assert "unwrapped_positions" in frames[1].arrays
    assert "indices" in frames[1].arrays


def test_frames_from_melt_quench_result_stride_and_optional_keys_absent():
    """Stride sub-samples frames and missing optional arrays are handled gracefully."""
    initial = _simple_atoms()
    stage = {
        "positions": [
            np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
            np.array([[0.1, 0.0, 0.0], [2.6, 0.0, 0.0]]),
            np.array([[0.2, 0.0, 0.0], [2.7, 0.0, 0.0]]),
        ],
        "cells": [np.diag([5.0, 5.0, 5.0])] * 3,
        "temperature": [300.0, 350.0, 400.0],
        "energy_pot": [-10.0, -9.5, -9.0],
        "energy_tot": [-9.0, -8.5, -8.0],
        "volume": [125.0, 125.0, 125.0],
        "pressures": [np.zeros((3, 3))] * 3,
    }
    result = {"result": [stage]}

    frames = frames_from_melt_quench_result(result, initial, stage=0, stride=2)
    assert len(frames) == 2
    assert frames[0].info["temperature"] == 300.0
    assert frames[1].info["temperature"] == 400.0
    assert "forces" not in frames[0].arrays
    assert "unwrapped_positions" not in frames[0].arrays


# ---------------------------------------------------------------------------
# write_xyz
# ---------------------------------------------------------------------------


def test_write_xyz_raises_without_type_dict(tmp_path):
    """Passing type_dict=None raises ValueError."""
    fp = tmp_path / "out.xyz"
    with pytest.raises(ValueError, match="type_dict must be provided"):
        write_xyz(str(fp), np.zeros((1, 3)), np.array([14]))


def test_write_xyz_raises_on_unknown_type(tmp_path):
    """An atomic type absent from type_dict raises ValueError."""
    fp = tmp_path / "out.xyz"
    with pytest.raises(ValueError, match="Unknown atomic type"):
        write_xyz(str(fp), np.zeros((1, 3)), np.array([99]), type_dict={14: "Si"})


def test_write_xyz_with_box_size(tmp_path):
    """When box_size is provided the comment line starts with CUB."""
    fp = tmp_path / "out.xyz"
    coords = np.array([[1.0, 2.0, 3.0]])
    write_xyz(str(fp), coords, np.array([14]), box_size=np.array([5.0, 5.0, 5.0]), type_dict={14: "Si"})
    lines = fp.read_text().splitlines()
    assert lines[0] == "1"
    assert lines[1].startswith("CUB 5.00000000")
    assert lines[2].startswith("Si")


def test_write_xyz_without_box_size(tmp_path):
    """When box_size is None the comment line is blank."""
    fp = tmp_path / "out.xyz"
    coords = np.array([[0.0, 0.0, 0.0]])
    write_xyz(str(fp), coords, np.array([8]), type_dict={8: "O"})
    lines = fp.read_text().splitlines()
    assert lines[1] == ""


def test_write_xyz_atom_count(tmp_path):
    """First line is the number of atoms and the file has the correct number of lines."""
    fp = tmp_path / "out.xyz"
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    write_xyz(str(fp), coords, np.array([14, 8, 14]), type_dict={14: "Si", 8: "O"})
    lines = fp.read_text().splitlines()
    assert lines[0] == "3"
    assert len(lines) == 5  # count + comment + 3 atoms
