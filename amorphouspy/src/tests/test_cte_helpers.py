"""Tests for amorphouspy.workflows.cte_helpers — full coverage of all helper functions."""

import logging

import numpy as np
import pytest

from amorphouspy.workflows.cte_helpers import (
    _collect_sim_data,
    _create_logger,
    _fluctuation_simulation_cte_calculation,
    _fluctuation_simulation_input_checker,
    _fluctuation_simulation_merge_results,
    _fluctuation_simulation_uncertainty_check,
    _initialize_datadict,
    _sanity_check_sim_data,
    _temperature_scan_input_checker,
    _temperature_scan_merge_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOGGER_NAME = "amorphouspy.workflows.cte"


def _clean_logger_handlers() -> None:
    """Remove all handlers from the CTE logger singleton to avoid leaks."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _make_sim_data(n: int = 50, run_index: int = 0) -> dict:
    """Build a minimal sim_data dict matching what _collect_sim_data produces."""
    rng = np.random.default_rng(42)
    return {
        "run_index": np.array([run_index]),
        "steps": np.arange(n, dtype=float),
        "T": np.full(n, 300.0) + rng.normal(0, 1, n),
        "E_tot": rng.normal(-100.0, 1.0, n),
        "ptot": np.zeros(n),
        "pxx": np.zeros(n),
        "pyy": np.zeros(n),
        "pzz": np.zeros(n),
        "V": np.full(n, 1000.0) + rng.normal(0, 5, n),
        "Lx": np.full(n, 10.0) + rng.normal(0, 0.1, n),
        "Ly": np.full(n, 10.0) + rng.normal(0, 0.1, n),
        "Lz": np.full(n, 10.0) + rng.normal(0, 0.1, n),
    }


def _make_parsed_output(n: int = 50) -> dict:
    """Build a dict that mimics LAMMPS parsed output for _collect_sim_data."""
    rng = np.random.default_rng(7)
    pressures = np.zeros((n, 3, 3))
    pressures[:, 0, 0] = rng.normal(0, 0.01, n)
    pressures[:, 1, 1] = rng.normal(0, 0.01, n)
    pressures[:, 2, 2] = rng.normal(0, 0.01, n)
    return {
        "generic": {
            "steps": np.arange(n, dtype=float).tolist(),
            "temperature": (np.full(n, 300.0) + rng.normal(0, 1, n)).tolist(),
            "energy_tot": rng.normal(-100.0, 1.0, n).tolist(),
            "volume": (np.full(n, 1000.0) + rng.normal(0, 5, n)).tolist(),
            "pressures": pressures,
        },
        "lammps": {
            "Lx": (np.full(n, 10.0) + rng.normal(0, 0.05, n)).tolist(),
            "Ly": (np.full(n, 10.0) + rng.normal(0, 0.05, n)).tolist(),
            "Lz": (np.full(n, 10.0) + rng.normal(0, 0.05, n)).tolist(),
        },
    }


# ---------------------------------------------------------------------------
# _create_logger
# ---------------------------------------------------------------------------


def test_create_logger_returns_logger(tmp_path, monkeypatch) -> None:
    """_create_logger returns a Logger instance named correctly."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == LOGGER_NAME
    finally:
        _clean_logger_handlers()


def test_create_logger_creates_cte_log_file(tmp_path, monkeypatch) -> None:
    """_create_logger creates a 'cte.log' file in the current directory."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        _create_logger()
        assert (tmp_path / "cte.log").exists()
    finally:
        _clean_logger_handlers()


def test_create_logger_adds_file_handler(tmp_path, monkeypatch) -> None:
    """The returned logger has at least one FileHandler attached."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1
    finally:
        _clean_logger_handlers()


# ---------------------------------------------------------------------------
# _initialize_datadict
# ---------------------------------------------------------------------------


def test_initialize_datadict_without_cte_keys() -> None:
    """Default call returns dict with expected keys and empty numpy arrays."""
    d = _initialize_datadict()
    expected_keys = {"run_index", "steps", "T", "E_tot", "ptot", "pxx", "pyy", "pzz", "V", "Lx", "Ly", "Lz"}
    assert set(d.keys()) == expected_keys
    for v in d.values():
        assert isinstance(v, np.ndarray)
        assert len(v) == 0


def test_initialize_datadict_with_cte_keys() -> None:
    """with_CTE_keys=True adds the four CTE entries."""
    d = _initialize_datadict(with_CTE_keys=True)
    for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
        assert key in d
        assert isinstance(d[key], np.ndarray)


def test_initialize_datadict_without_cte_keys_excludes_cte() -> None:
    """with_CTE_keys=False must NOT include CTE keys."""
    d = _initialize_datadict(with_CTE_keys=False)
    for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
        assert key not in d


# ---------------------------------------------------------------------------
# _collect_sim_data
# ---------------------------------------------------------------------------


def test_collect_sim_data_keys() -> None:
    """Returned dict has all expected keys."""
    parsed = _make_parsed_output()
    data = _collect_sim_data(parsed, counter_production_run=3)
    expected_keys = {"run_index", "steps", "T", "E_tot", "ptot", "pxx", "pyy", "pzz", "V", "Lx", "Ly", "Lz"}
    assert set(data.keys()) == expected_keys


def test_collect_sim_data_run_index() -> None:
    """run_index reflects counter_production_run."""
    parsed = _make_parsed_output()
    data = _collect_sim_data(parsed, counter_production_run=7)
    assert data["run_index"][0] == 7


def test_collect_sim_data_pressure_averaging() -> None:
    """Ptot is the average of pxx, pyy, pzz."""
    parsed = _make_parsed_output()
    data = _collect_sim_data(parsed, counter_production_run=0)
    expected_ptot = (data["pxx"] + data["pyy"] + data["pzz"]) / 3
    np.testing.assert_allclose(data["ptot"], expected_ptot)


def test_collect_sim_data_array_lengths_match() -> None:
    """All data arrays (except run_index) have the same length."""
    n = 30
    parsed = _make_parsed_output(n=n)
    data = _collect_sim_data(parsed, counter_production_run=0)
    lengths = {k: len(v) for k, v in data.items() if k != "run_index"}
    assert len(set(lengths.values())) == 1, f"Inconsistent lengths: {lengths}"


# ---------------------------------------------------------------------------
# _sanity_check_sim_data
# ---------------------------------------------------------------------------


def test_sanity_check_no_warning_when_on_target(tmp_path, monkeypatch, caplog) -> None:
    """No warnings when T and p are within tolerance."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        sim_data = _make_sim_data()
        T_target = float(np.mean(sim_data["T"]))
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _sanity_check_sim_data(sim_data, T_target=T_target, p_target=0.0, logger=logger)
        assert not caplog.records
    finally:
        _clean_logger_handlers()


def test_sanity_check_warns_on_temperature_deviation(tmp_path, monkeypatch, caplog) -> None:
    """Warning is emitted when temperature deviates significantly."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        sim_data = _make_sim_data()
        # Force a large T deviation
        sim_data["T"] = np.full(len(sim_data["T"]), 500.0)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _sanity_check_sim_data(sim_data, T_target=300.0, p_target=0.0, logger=logger)
        assert any("Temperature" in r.message for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_sanity_check_warns_on_pressure_deviation(tmp_path, monkeypatch, caplog) -> None:
    """Warnings are emitted when pressure components deviate significantly."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        sim_data = _make_sim_data()
        # Set large pressure deviations on all components
        large_p = 10.0  # GPa — far from target 0 GPa
        for key in ("ptot", "pxx", "pyy", "pzz"):
            sim_data[key] = np.full(len(sim_data["ptot"]), large_p)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _sanity_check_sim_data(sim_data, T_target=300.0, p_target=0.0, logger=logger)
        # At least one pressure warning should be present
        assert any("pressure" in r.message.lower() or "Pressure" in r.message for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_sanity_check_warns_on_individual_pressure_components(tmp_path, monkeypatch, caplog) -> None:
    """Warnings for pxx, pyy, pzz each trigger separately."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        sim_data = _make_sim_data()
        large_p = 5.0  # GPa
        for key in ("ptot", "pxx", "pyy", "pzz"):
            sim_data[key] = np.full(len(sim_data["ptot"]), large_p)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _sanity_check_sim_data(sim_data, T_target=300.0, p_target=0.0, logger=logger)
        messages = " ".join(r.message for r in caplog.records)
        # Expect at least two pressure-related warnings
        assert messages.count("pressure") + messages.count("Pressure") >= 2
    finally:
        _clean_logger_handlers()


# ---------------------------------------------------------------------------
# _fluctuation_simulation_input_checker
# ---------------------------------------------------------------------------


def _make_dummy_input_checker_logger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    return _create_logger()


def test_input_checker_no_warnings_good_params(tmp_path, monkeypatch, caplog) -> None:
    """No warnings when all parameters are fine."""
    logger = _make_dummy_input_checker_logger(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            result = _fluctuation_simulation_input_checker(
                production_steps=100_000,
                min_production_runs=3,
                max_production_runs=10,
                n_log=10,
                timestep=1.0,
                n_dump=50_000,
                logger=logger,
            )
        assert not caplog.records
        assert len(result) == 5
    finally:
        _clean_logger_handlers()


def test_input_checker_corrects_min_production_runs(tmp_path, monkeypatch, caplog) -> None:
    """min_production_runs < 2 is corrected to 2 with a warning."""
    logger = _make_dummy_input_checker_logger(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _prod_steps, min_runs, *_ = _fluctuation_simulation_input_checker(
                production_steps=100_000,
                min_production_runs=1,
                max_production_runs=5,
                n_log=10,
                timestep=1.0,
                n_dump=50_000,
                logger=logger,
            )
        assert min_runs == 2
        assert any("2 individual" in r.message or "min_production_runs" in r.message for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_input_checker_corrects_max_less_than_min(tmp_path, monkeypatch, caplog) -> None:
    """max_production_runs < min_production_runs is corrected."""
    logger = _make_dummy_input_checker_logger(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _, min_runs, max_runs, *_ = _fluctuation_simulation_input_checker(
                production_steps=100_000,
                min_production_runs=5,
                max_production_runs=3,
                n_log=10,
                timestep=1.0,
                n_dump=50_000,
                logger=logger,
            )
        assert max_runs >= min_runs
        assert any("max_production_runs" in r.message for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_input_checker_corrects_short_production_steps(tmp_path, monkeypatch, caplog) -> None:
    """Too-short production_steps is increased with a warning."""
    logger = _make_dummy_input_checker_logger(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            prod_steps, *_ = _fluctuation_simulation_input_checker(
                production_steps=100,  # very short
                min_production_runs=2,
                max_production_runs=5,
                n_log=10,
                timestep=1.0,
                n_dump=50,
                logger=logger,
            )
        assert prod_steps > 100
        assert any(
            "production_steps" in r.message or "production runs are too short" in r.message for r in caplog.records
        )
    finally:
        _clean_logger_handlers()


def test_input_checker_warns_low_averaging_points(tmp_path, monkeypatch, caplog) -> None:
    """Warning when N_for_averaging < 1000 (large n_log or large timestep)."""
    logger = _make_dummy_input_checker_logger(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _fluctuation_simulation_input_checker(
                production_steps=500_000,
                min_production_runs=2,
                max_production_runs=5,
                n_log=500,  # high n_log → few averaging points
                timestep=1.0,
                n_dump=100_000,
                logger=logger,
            )
        # At least warning about running mean data points
        assert any("data points" in r.message or "running mean" in r.message.lower() for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_input_checker_corrects_n_dump_too_large(tmp_path, monkeypatch, caplog) -> None:
    """n_dump > production_steps is corrected to production_steps."""
    logger = _make_dummy_input_checker_logger(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            prod_steps, _, _, _, n_dump = _fluctuation_simulation_input_checker(
                production_steps=100_000,
                min_production_runs=2,
                max_production_runs=5,
                n_log=10,
                timestep=1.0,
                n_dump=999_999,  # larger than production_steps
                logger=logger,
            )
        assert n_dump == prod_steps
        assert any("n_dump" in r.message for r in caplog.records)
    finally:
        _clean_logger_handlers()


# ---------------------------------------------------------------------------
# _fluctuation_simulation_cte_calculation
# ---------------------------------------------------------------------------


def test_cte_calculation_returns_four_keys() -> None:
    """Returns dict with keys CTE_V, CTE_x, CTE_y, CTE_z."""
    sim_data = _make_sim_data(n=200)
    result = _fluctuation_simulation_cte_calculation(sim_data, temperature=300.0, p=0.0)
    assert set(result.keys()) == {"CTE_V", "CTE_x", "CTE_y", "CTE_z"}


def test_cte_calculation_values_are_floats() -> None:
    """All returned CTE values are Python floats."""
    sim_data = _make_sim_data(n=200)
    result = _fluctuation_simulation_cte_calculation(sim_data, temperature=300.0, p=0.0)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_cte_calculation_running_mean_mode() -> None:
    """use_running_mean=True also returns four float CTE keys."""
    sim_data = _make_sim_data(n=500)
    result = _fluctuation_simulation_cte_calculation(
        sim_data, temperature=300.0, p=0.0, N_points=100, use_running_mean=True
    )
    assert set(result.keys()) == {"CTE_V", "CTE_x", "CTE_y", "CTE_z"}
    for val in result.values():
        assert isinstance(val, float)


def test_cte_calculation_nonzero_pressure() -> None:
    """Nonzero pressure is incorporated correctly (no crash, float returned)."""
    sim_data = _make_sim_data(n=200)
    result = _fluctuation_simulation_cte_calculation(sim_data, temperature=300.0, p=1.0)
    assert set(result.keys()) == {"CTE_V", "CTE_x", "CTE_y", "CTE_z"}


# ---------------------------------------------------------------------------
# _fluctuation_simulation_uncertainty_check
# ---------------------------------------------------------------------------


def test_uncertainty_check_raises_on_single_value() -> None:
    """Raises ValueError when any CTE array has <= 1 value."""
    data = _initialize_datadict(with_CTE_keys=True)
    data["CTE_V"] = np.array([1e-6])
    data["CTE_x"] = np.array([1e-6])
    data["CTE_y"] = np.array([1e-6])
    data["CTE_z"] = np.array([1e-6])
    with pytest.raises(ValueError, match="more than one value"):
        _fluctuation_simulation_uncertainty_check(data, criterion=1e-7)


def test_uncertainty_check_converged_when_identical() -> None:
    """All identical CTE values give zero uncertainty → converged."""
    data = _initialize_datadict(with_CTE_keys=True)
    for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
        data[key] = np.full(5, 5e-6)
    converged, result = _fluctuation_simulation_uncertainty_check(data, criterion=1e-7)
    assert converged is True
    assert "CTE_V_mean" in result
    assert "CTE_x_uncertainty" in result


def test_uncertainty_check_not_converged_with_large_variance() -> None:
    """Noisy CTE values with large spread return converged=False."""
    rng = np.random.default_rng(0)
    data = _initialize_datadict(with_CTE_keys=True)
    for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
        data[key] = rng.normal(5e-6, 1e-5, 5)  # huge variance relative to criterion
    converged, result = _fluctuation_simulation_uncertainty_check(data, criterion=1e-12)
    assert converged is False
    assert isinstance(result, dict)


def test_uncertainty_check_result_keys() -> None:
    """Result dict contains mean and uncertainty for each CTE component."""
    data = _initialize_datadict(with_CTE_keys=True)
    for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
        data[key] = np.array([5e-6, 6e-6, 4e-6])
    _, result = _fluctuation_simulation_uncertainty_check(data, criterion=1e-5)
    expected = {
        "CTE_V_mean",
        "CTE_x_mean",
        "CTE_y_mean",
        "CTE_z_mean",
        "CTE_V_uncertainty",
        "CTE_x_uncertainty",
        "CTE_y_uncertainty",
        "CTE_z_uncertainty",
    }
    assert set(result.keys()) == expected


def test_uncertainty_check_raises_for_each_cte_key() -> None:
    """ValueError is raised separately for each CTE key that has <= 1 value."""
    for bad_key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
        data = _initialize_datadict(with_CTE_keys=True)
        for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
            if key == bad_key:
                data[key] = np.array([1e-6])  # only 1 value
            else:
                data[key] = np.array([1e-6, 2e-6, 3e-6])
        with pytest.raises(ValueError, match="more than one value"):
            _fluctuation_simulation_uncertainty_check(data, criterion=1e-7)


# ---------------------------------------------------------------------------
# _fluctuation_simulation_merge_results
# ---------------------------------------------------------------------------


def test_merge_results_increases_array_length() -> None:
    """Merging doubles the number of data points for averaged keys."""
    base = _initialize_datadict(with_CTE_keys=True)
    sim_data = _make_sim_data(n=50, run_index=0)
    cte_data = {"CTE_V": 5e-6, "CTE_x": 4e-6, "CTE_y": 4e-6, "CTE_z": 4e-6}

    merged1 = _fluctuation_simulation_merge_results(base, sim_data, cte_data)
    assert len(merged1["CTE_V"]) == 1
    assert len(merged1["T"]) == 1
    assert len(merged1["run_index"]) == 1

    # Merge a second time
    sim_data2 = _make_sim_data(n=50, run_index=1)
    cte_data2 = {"CTE_V": 6e-6, "CTE_x": 5e-6, "CTE_y": 5e-6, "CTE_z": 5e-6}
    merged2 = _fluctuation_simulation_merge_results(merged1, sim_data2, cte_data2)
    assert len(merged2["CTE_V"]) == 2
    assert len(merged2["T"]) == 2


def test_merge_results_steps_takes_last_value() -> None:
    """The 'steps' key appends the last value of the new simulation."""
    base = _initialize_datadict(with_CTE_keys=True)
    sim_data = _make_sim_data(n=50)
    cte_data = {"CTE_V": 5e-6, "CTE_x": 4e-6, "CTE_y": 4e-6, "CTE_z": 4e-6}
    merged = _fluctuation_simulation_merge_results(base, sim_data, cte_data)
    assert merged["steps"][0] == sim_data["steps"][-1]


def test_merge_results_contains_all_keys() -> None:
    """Merged result contains all sim and CTE keys."""
    base = _initialize_datadict(with_CTE_keys=True)
    sim_data = _make_sim_data(n=10)
    cte_data = {"CTE_V": 1e-6, "CTE_x": 1e-6, "CTE_y": 1e-6, "CTE_z": 1e-6}
    merged = _fluctuation_simulation_merge_results(base, sim_data, cte_data)
    expected_sim_keys = {"run_index", "steps", "T", "E_tot", "ptot", "pxx", "pyy", "pzz", "V", "Lx", "Ly", "Lz"}
    expected_cte_keys = {"CTE_V", "CTE_x", "CTE_y", "CTE_z"}
    assert expected_sim_keys.issubset(set(merged.keys()))
    assert expected_cte_keys.issubset(set(merged.keys()))


# ---------------------------------------------------------------------------
# _temperature_scan_input_checker
# ---------------------------------------------------------------------------


def test_temperature_scan_raises_for_single_temperature(tmp_path, monkeypatch) -> None:
    """Raises ValueError when fewer than 2 unique temperatures are given."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        with pytest.raises(ValueError, match="two different temperatures"):
            _temperature_scan_input_checker(
                temperature=[300.0],
                production_steps=10_000,
                n_dump=5_000,
                logger=logger,
            )
    finally:
        _clean_logger_handlers()


def test_temperature_scan_raises_for_all_duplicates(tmp_path, monkeypatch) -> None:
    """Raises ValueError when all temperatures are the same."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        with pytest.raises(ValueError, match="two different temperatures"):
            _temperature_scan_input_checker(
                temperature=[300.0, 300.0, 300.0],
                production_steps=10_000,
                n_dump=5_000,
                logger=logger,
            )
    finally:
        _clean_logger_handlers()


def test_temperature_scan_warns_on_duplicates(tmp_path, monkeypatch, caplog) -> None:
    """Warning when temperatures contain duplicates but at least 2 are unique."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _temperature_scan_input_checker(
                temperature=[300.0, 300.0, 400.0],
                production_steps=10_000,
                n_dump=5_000,
                logger=logger,
            )
        assert any("duplicates" in r.message.lower() or "duplicate" in r.message.lower() for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_temperature_scan_corrects_n_dump(tmp_path, monkeypatch, caplog) -> None:
    """n_dump > production_steps is corrected to production_steps."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            n_dump = _temperature_scan_input_checker(
                temperature=[300.0, 400.0],
                production_steps=1000,
                n_dump=9999,
                logger=logger,
            )
        assert n_dump == 1000
        assert any("n_dump" in r.message for r in caplog.records)
    finally:
        _clean_logger_handlers()


def test_temperature_scan_no_warning_good_params(tmp_path, monkeypatch, caplog) -> None:
    """No warning when two unique temps and n_dump <= production_steps."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        logger = _create_logger()
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            n_dump = _temperature_scan_input_checker(
                temperature=[300.0, 400.0],
                production_steps=10_000,
                n_dump=5_000,
                logger=logger,
            )
        assert n_dump == 5_000
        assert not caplog.records
    finally:
        _clean_logger_handlers()


# ---------------------------------------------------------------------------
# _temperature_scan_merge_results
# ---------------------------------------------------------------------------


def test_temperature_scan_merge_results_grows() -> None:
    """Merging adds one entry per averaged key."""
    base = _initialize_datadict(with_CTE_keys=False)
    sim_data = _make_sim_data(n=40)
    merged1 = _temperature_scan_merge_results(base, sim_data)
    assert len(merged1["T"]) == 1
    assert len(merged1["V"]) == 1

    sim_data2 = _make_sim_data(n=40, run_index=1)
    merged2 = _temperature_scan_merge_results(merged1, sim_data2)
    assert len(merged2["T"]) == 2


def test_temperature_scan_merge_results_steps_takes_last() -> None:
    """'steps' appends the last step of new simulation data."""
    base = _initialize_datadict(with_CTE_keys=False)
    sim_data = _make_sim_data(n=40)
    merged = _temperature_scan_merge_results(base, sim_data)
    assert merged["steps"][0] == sim_data["steps"][-1]


def test_temperature_scan_merge_results_keys_unchanged() -> None:
    """Merged result has the same set of keys as the base dict."""
    base = _initialize_datadict(with_CTE_keys=False)
    sim_data = _make_sim_data(n=10)
    merged = _temperature_scan_merge_results(base, sim_data)
    assert set(merged.keys()) == set(base.keys())
