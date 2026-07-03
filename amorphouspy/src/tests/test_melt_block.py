"""Tests for the shared melt pre-equilibration block helpers.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import amorphouspy.fabrication.meltquench as mq_module
import amorphouspy.lammps.md as md_module
import amorphouspy.properties.elastic as elastic_module
import pytest
from amorphouspy.lammps.potentials._melt_block import (
    melt_block_lines,
    set_melt_block_temperature,
    strip_melt_block,
)
from amorphouspy.lammps.potentials.bjp_potential import generate_bjp_potential
from amorphouspy.lammps.potentials.bmp_potential import generate_bmp_potential
from amorphouspy.lammps.potentials.du_teter_potential import generate_du_teter_potential
from amorphouspy.lammps.potentials.pmmcs_potential import generate_pmmcs_potential
from amorphouspy.lammps.potentials.shik_potential import generate_shik_potential
from amorphouspy.lammps.potentials.yang_potential import generate_yang2026_potential
from ase import Atoms

_POTENTIAL_NAMES = ("bjp", "bmp", "du_teter", "pmmcs", "shik", "yang2026")


def _sio2_atoms_dict() -> dict:
    """Minimal SiO2 atoms_dict for potential generation."""
    return {"atoms": [{"element": "Si"}, {"element": "O"}, {"element": "O"}]}


def _cas_atoms_dict() -> dict:
    """CaAlSiO atoms_dict for BJP potential (needs all four BJP elements)."""
    return {"atoms": [{"element": "Ca"}, {"element": "Al"}, {"element": "Si"}, {"element": "O"}]}


def _generate(name: str, tmp_path, *, melt: bool):
    """Generate the named potential, writing any table files under *tmp_path*."""
    if name == "shik":
        return generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=melt)
    if name == "pmmcs":
        return generate_pmmcs_potential(_sio2_atoms_dict(), melt=melt)
    if name == "bjp":
        return generate_bjp_potential(_cas_atoms_dict(), melt=melt)
    if name == "bmp":
        return generate_bmp_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=melt)
    if name == "du_teter":
        return generate_du_teter_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=melt, use_three_body=False)
    return generate_yang2026_potential(_sio2_atoms_dict(), melt=melt)


@pytest.mark.parametrize("name", _POTENTIAL_NAMES)
def test_strip_melt_block_matches_melt_false(name, tmp_path):
    """Stripping a melt=True config yields exactly the melt=False config."""
    with_melt = _generate(name, tmp_path, melt=True)
    without_melt = _generate(name, tmp_path, melt=False)
    assert strip_melt_block(with_melt).loc[0, "Config"] == without_melt.loc[0, "Config"]


def test_strip_melt_block_does_not_mutate_input(tmp_path):
    """strip_melt_block returns a copy and leaves the caller's DataFrame untouched."""
    potential = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=True)
    before = list(potential.loc[0, "Config"])
    strip_melt_block(potential)
    assert potential.loc[0, "Config"] == before


def test_set_melt_block_temperature_retunes(tmp_path):
    """set_melt_block_temperature replaces the block with one at the requested temperature."""
    potential = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=True)
    retuned = set_melt_block_temperature(potential, 4500.0)
    config = "".join(retuned.loc[0, "Config"])
    assert "fix langevinnve all langevin 4500 4500 0.01 48279" in config
    assert "4000 4000" not in config
    assert retuned.loc[0, "Config"][-5:] == melt_block_lines(4500.0)


def test_melt_quench_simulation_retunes_melt_block(tmp_path, monkeypatch):
    """Stage 1 runs the melt block at the requested melt temperature; later stages have none."""
    captured = []

    def fake_runner(structure, potential, **_kwargs):
        captured.append(list(potential.loc[0, "Config"]))
        return structure, {"generic": {"steps": [0], "temperature": [4500.0]}}

    monkeypatch.setattr(mq_module, "_run_lammps_md", fake_runner)

    potential = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=True)
    structure = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]], cell=[10, 10, 10], pbc=True)

    mq_module.melt_quench_simulation(
        structure=structure,
        potential=potential,
        temperature_high=4500.0,
        temperature_low=300.0,
        heating_rate=1e15,
        cooling_rate=1e15,
    )

    assert len(captured) > 1
    first_stage = "".join(captured[0])
    assert "fix langevinnve all langevin 4500 4500 0.01 48279" in first_stage
    assert "4000 4000" not in first_stage
    for config in captured[1:]:
        assert "langevinnve" not in "".join(config)


def test_md_simulation_strips_melt_block(tmp_path, monkeypatch):
    """md_simulation strips the melt block and leaves the caller's potential untouched."""
    captured = []

    def fake_runner(structure, potential, **_kwargs):
        captured.append("".join(potential.loc[0, "Config"]))
        return structure, {"generic": {"steps": [0]}}

    monkeypatch.setattr(md_module, "_run_lammps_md", fake_runner)

    potential = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=True)
    before = list(potential.loc[0, "Config"])
    structure = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]], cell=[10, 10, 10], pbc=True)

    md_module.md_simulation(structure=structure, potential=potential, temperature_sim=300.0)

    assert captured
    assert "langevinnve" not in captured[0]
    assert potential.loc[0, "Config"] == before


def test_elastic_simulation_strips_melt_block(tmp_path, monkeypatch):
    """elastic_simulation strips the melt block before its first MD stage."""
    captured = []

    def fake_runner(**kwargs):
        captured.append("".join(kwargs["potential"].loc[0, "Config"]))
        msg = "abort after first stage"
        raise RuntimeError(msg)

    monkeypatch.setattr(elastic_module, "_run_lammps_md", fake_runner)

    potential = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=True)
    structure = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]], cell=[10, 10, 10], pbc=True)

    with pytest.raises(RuntimeError, match="abort after first stage"):
        elastic_module.elastic_simulation(structure=structure, potential=potential)

    assert captured
    assert "langevinnve" not in captured[0]
