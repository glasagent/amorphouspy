"""Tests for the melt-quench pre-equilibration fix override.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import amorphouspy.fabrication.meltquench as mq_module
from amorphouspy.fabrication.pre_equilibration import pre_equilibration_fix_override
from amorphouspy.lammps.potentials.pmmcs_potential import generate_pmmcs_potential
from ase import Atoms


def _sio2_atoms_dict() -> dict:
    """Minimal SiO2 atoms_dict for potential generation."""
    return {"atoms": [{"element": "Si"}, {"element": "O"}, {"element": "O"}]}


def test_fix_override_contains_langevin_at_requested_temperature():
    """The override renders the Langevin + nve/limit fix pair at the given temperature."""
    override = pre_equilibration_fix_override(4500.0)
    assert override.startswith("langevinnve all langevin 4500 4500 0.01 48279")
    assert "fix ensemblenve all nve/limit 0.5" in override


def _run_melt_quench_with_fake_runner(monkeypatch, **kwargs):
    """Run a PMMCS melt_quench_simulation with a fake runner; return per-stage call kwargs."""
    captured = []

    def fake_runner(structure, potential, **_kwargs):
        captured.append({"config": list(potential.loc[0, "Config"]), **_kwargs})
        return structure, {"generic": {"steps": [0], "temperature": [4500.0]}}

    monkeypatch.setattr(mq_module, "_run_lammps_md", fake_runner)
    potential = generate_pmmcs_potential(_sio2_atoms_dict())
    structure = Atoms("SiO2", positions=[[0, 0, 0], [1.6, 0, 0], [0, 1.6, 0]], cell=[10, 10, 10], pbc=True)
    mq_module.melt_quench_simulation(
        structure=structure,
        potential=potential,
        temperature_high=4500.0,
        temperature_low=300.0,
        cooling_rate=1e15,
        **kwargs,
    )
    return captured


def test_melt_quench_simulation_runs_pre_equilibration_as_stage0(monkeypatch):
    """Stage 0 carries the fix override at temperature_high; every stage Config stays clean."""
    captured = _run_melt_quench_with_fake_runner(monkeypatch)
    assert len(captured) > 1
    stage0 = captured[0]
    assert stage0["input_control_file"]["fix"] == pre_equilibration_fix_override(4500.0)
    assert stage0["n_ionic_steps"] == 10_000
    assert stage0["langevin"] is False
    for stage in captured[1:]:
        assert "input_control_file" not in stage
    for stage in captured:
        assert not any("langevinnve" in line for line in stage["config"])


def test_melt_quench_simulation_pre_equilibrate_false_omits_stage0(monkeypatch):
    """pre_equilibrate=False drops the stage-0 call; no stage carries the override."""
    captured = _run_melt_quench_with_fake_runner(monkeypatch, pre_equilibrate=False)
    with_stage0 = _run_melt_quench_with_fake_runner(monkeypatch)
    assert len(captured) == len(with_stage0) - 1
    for stage in captured:
        assert "input_control_file" not in stage
