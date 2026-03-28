"""Tests for amorphouspy.io_utils."""

import numpy as np
import pytest
from ase import Atoms

from amorphouspy.io_utils import (
    get_properties_for_structure_analysis,
    structure_from_parsed_output,
    write_angle_distribution,
    write_distribution_to_file,
    write_xyz,
)

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
# get_properties_for_structure_analysis
# ---------------------------------------------------------------------------


def test_get_properties_ids_sequential():
    """IDs are 1-based sequential integers."""
    atoms = _simple_atoms()
    ids, _, _, _ = get_properties_for_structure_analysis(atoms)
    assert list(ids) == [1, 2]


def test_get_properties_types():
    """Atomic numbers are returned unchanged."""
    atoms = _simple_atoms()
    _, types, _, _ = get_properties_for_structure_analysis(atoms)
    np.testing.assert_array_equal(types, [14, 8])


def test_get_properties_cell_diagonal():
    """Cell is returned as the diagonal of the cell matrix."""
    atoms = _simple_atoms()
    _, _, _, cell = get_properties_for_structure_analysis(atoms)
    np.testing.assert_allclose(cell, [5.0, 5.0, 5.0])


def test_get_properties_does_not_modify_original():
    """Original atoms object is not modified by the wrap operation."""
    atoms = _simple_atoms()
    atoms.positions[0] = [-1.0, 0.0, 0.0]
    original_pos = atoms.get_positions().copy()
    get_properties_for_structure_analysis(atoms)
    np.testing.assert_array_equal(atoms.get_positions(), original_pos)


def test_get_properties_coords_are_wrapped():
    """Returned coordinates are wrapped inside the simulation cell."""
    atoms = _simple_atoms()
    atoms.positions[0] = [-1.0, 0.0, 0.0]  # outside the box
    _, _, coords, _ = get_properties_for_structure_analysis(atoms)
    assert (coords >= 0.0).all()
    assert (coords < 5.0).all()


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
