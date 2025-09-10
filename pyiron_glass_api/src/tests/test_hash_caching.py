"""Test hash-based caching functionality for meltquench simulations.

This module tests that:
1. Identical requests produce identical hashes
2. Different requests produce different hashes
3. The caching logic can be imported and executed without errors
"""

from pyiron_glass_api.app import get_meltquench_hash
from pyiron_glass_api.models import MeltquenchRequest


def test_hash_consistency() -> None:
    """Test that identical requests produce identical hashes."""
    # Create two identical requests
    request1 = MeltquenchRequest(
        components=["SiO2", "Na2O"],
        values=[75.0, 25.0],
        unit="wt",
        heating_rate=int(1e14),
        cooling_rate=int(1e14),
        n_print=1000,
    )

    request2 = MeltquenchRequest(
        components=["SiO2", "Na2O"],
        values=[75.0, 25.0],
        unit="wt",
        heating_rate=int(1e14),
        cooling_rate=int(1e14),
        n_print=1000,
    )

    hash1 = get_meltquench_hash(request1)
    hash2 = get_meltquench_hash(request2)

    assert hash1 == hash2, f"Identical requests should have identical hashes: {hash1} != {hash2}"
    assert len(hash1) == 16, f"Hash should be 16 characters long, got {len(hash1)}"


def test_hash_differentiation() -> None:
    """Test that different requests produce different hashes."""
    # Create requests with different parameters
    request1 = MeltquenchRequest(components=["SiO2", "Na2O"], values=[75.0, 25.0], unit="wt")

    request2 = MeltquenchRequest(
        components=["SiO2", "CaO"],  # Different component
        values=[75.0, 25.0],
        unit="wt",
    )

    request3 = MeltquenchRequest(
        components=["SiO2", "Na2O"],
        values=[80.0, 20.0],  # Different values
        unit="wt",
    )

    request4 = MeltquenchRequest(
        components=["SiO2", "Na2O"],
        values=[75.0, 25.0],
        unit="wt",
        heating_rate=int(1e15),  # Different heating rate
    )

    hash1 = get_meltquench_hash(request1)
    hash2 = get_meltquench_hash(request2)
    hash3 = get_meltquench_hash(request3)
    hash4 = get_meltquench_hash(request4)

    assert hash1 != hash2, "Different components should have different hashes"
    assert hash1 != hash3, "Different values should have different hashes"
    assert hash1 != hash4, "Different heating rates should have different hashes"


def test_component_order_independence() -> None:
    """Test that different compositions produce different hashes."""
    request1 = MeltquenchRequest(
        components=["SiO2", "Na2O", "CaO"],
        values=[60.0, 20.0, 20.0],  # 60% SiO2, 20% Na2O, 20% CaO
        unit="wt",
    )

    request2 = MeltquenchRequest(
        components=["SiO2", "Na2O", "CaO"],  # Same order
        values=[50.0, 25.0, 25.0],  # Different composition: 50% SiO2, 25% Na2O, 25% CaO
        unit="wt",
    )

    hash1 = get_meltquench_hash(request1)
    hash2 = get_meltquench_hash(request2)

    # These should be different because they have different compositions
    assert hash1 != hash2, "Different component proportions should have different hashes"


def test_component_order_with_same_composition() -> None:
    """Test that truly identical compositions produce the same hash regardless of order."""
    request1 = MeltquenchRequest(components=["SiO2", "Na2O"], values=[75.0, 25.0], unit="wt")

    request2 = MeltquenchRequest(
        components=["Na2O", "SiO2"],  # Different order
        values=[25.0, 75.0],  # Same composition but values match the reordered components
        unit="wt",
    )

    hash1 = get_meltquench_hash(request1)
    hash2 = get_meltquench_hash(request2)

    # These should be the same because they represent identical compositions
    assert hash1 == hash2, "Identical compositions should have the same hash regardless of component order"
