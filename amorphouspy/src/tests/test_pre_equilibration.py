"""Tests for the melt-quench pre-equilibration block helpers.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import amorphouspy.fabrication.meltquench as mq_module
import pandas as pd
from amorphouspy.fabrication.pre_equilibration import append_melt_block, melt_block_lines
from amorphouspy.lammps.potentials.pmmcs_potential import generate_pmmcs_potential
from ase import Atoms


def _sio2_atoms_dict() -> dict:
    """Minimal SiO2 atoms_dict for potential generation."""
    return {"atoms": [{"element": "Si"}, {"element": "O"}, {"element": "O"}]}


def test_melt_block_lines_contains_langevin_at_requested_temperature():
    """melt_block_lines renders the Langevin + nve/limit block at the given temperature."""
    joined = "".join(melt_block_lines(4500.0))
    assert "fix langevinnve all langevin 4500 4500 0.01 48279" in joined
    assert "nve/limit" in joined
    assert "run 10000" in joined


def test_append_melt_block_appends_block_at_end():
    """append_melt_block places the block after the existing Config lines."""
    potential = pd.DataFrame({"Name": ["pmmcs"], "Config": [["line1", "line2"]]})
    appended = append_melt_block(potential, 4321.0)
    assert appended.loc[0, "Config"][:2] == ["line1", "line2"]
    assert appended.loc[0, "Config"][2:] == melt_block_lines(4321.0)


def test_append_melt_block_does_not_mutate_input():
    """append_melt_block returns a copy and leaves the caller's DataFrame untouched."""
    potential = pd.DataFrame({"Name": ["pmmcs"], "Config": [["line1"]]})
    append_melt_block(potential, 4000.0)
    assert potential.loc[0, "Config"] == ["line1"]


def _run_melt_quench_with_fake_runner(monkeypatch, **kwargs):
    """Run a PMMCS melt_quench_simulation with a fake runner; return per-stage Config lines."""
    captured = []

    def fake_runner(structure, potential, **_kwargs):
        captured.append(list(potential.loc[0, "Config"]))
        return structure, {"generic": {"steps": [0], "temperature": [4500.0]}}

    monkeypatch.setattr(mq_module, "_run_lammps_md", fake_runner)
    potential = generate_pmmcs_potential(_sio2_atoms_dict())
    structure = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]], cell=[10, 10, 10], pbc=True)
    mq_module.melt_quench_simulation(
        structure=structure,
        potential=potential,
        temperature_high=4500.0,
        temperature_low=300.0,
        heating_rate=1e15,
        cooling_rate=1e15,
        **kwargs,
    )
    return captured


def test_melt_quench_simulation_runs_block_in_first_stage_only(monkeypatch):
    """Stage 1 runs the block once at temperature_high; later stages have none."""
    captured = _run_melt_quench_with_fake_runner(monkeypatch)
    assert len(captured) > 1
    first_stage = "".join(captured[0])
    assert "fix langevinnve all langevin 4500 4500 0.01 48279" in first_stage
    assert first_stage.count("run 10000") == 1
    for config in captured[1:]:
        assert "langevinnve" not in "".join(config)


def test_melt_quench_simulation_pre_equilibrate_false_omits_block(monkeypatch):
    """pre_equilibrate=False leaves every stage without the block."""
    captured = _run_melt_quench_with_fake_runner(monkeypatch, pre_equilibrate=False)
    assert len(captured) > 1
    for config in captured:
        assert "langevinnve" not in "".join(config)
