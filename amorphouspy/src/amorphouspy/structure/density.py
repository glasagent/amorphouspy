"""Glass density models for oxide glass systems.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from amorphouspy.structure.composition import extract_composition

# Predefined list of trace oxides for the "Remainder" term
TRACE_OXIDES = {
    "Ag2O",
    "Bi2O3",
    "Br",
    "CoxOy",
    "Cr2O3",
    "Cs2O",
    "CuO",
    "Ga2O3",
    "Gd2O3",
    "I",
    "MoO3",
    "Nb2O5",
    "PdO",
    "PrxOy",
    "Rb2O",
    "RexOy",
    "RhxOy",
    "RuO2",
    "SeO2",
    "Sm2O3",
    "SnO2",
    "TeO2",
    "Tl2O3",
    "WO3",
    "Y2O3",
}


def get_glass_density_from_model(composition: dict[str, float]) -> float:  # noqa: C901, PLR0912
    """Calculate the room-temperature glass density using Fluegel's empirical model.

    The model uses a polynomial expansion based on mole percentages of oxides.
    Source: Fluegel, A. "Global Model for Calculating Room-Temperature Glass Density from the Composition",
    J. Am. Ceram. Soc., 90 [8] 2622-2635 (2007).

    Args:
        composition: A dictionary mapping oxide formulas to their fractions.

    Returns:
        The calculated density in g/cm^3.

    Raises:
        ValueError: If the composition contains components unsupported by the model
            or if the format is invalid.

    Example:
        >>> density = get_glass_density_from_model({"SiO2": 0.75, "Na2O": 0.25})

    """
    COEFFICIENTS = {
        "b0": 2.121560704,
        "Al2O3": 0.010525974,
        "Al2O3_2": -0.000076924,
        "B2O3": 0.00579283,
        "B2O3_2": 0.000129174,
        "B2O3_3": -0.000019887,
        "Li2O": 0.012848733,
        "Li2O_2": -0.000276404,
        "Li2O_3": 0.00000259,
        "Na2O": 0.018129123,
        "Na2O_2": -0.000264838,
        "Na2O_3": 0.000001614,
        "K2O": 0.019177312,
        "K2O_2": -0.000319863,
        "K2O_3": 0.00000191,
        "MgO": 0.01210604,
        "MgO_2": -0.000061159,
        "CaO": 0.017992367,
        "CaO_2": -0.00005478,
        "SrO": 0.034630735,
        "SrO_2": -0.000086939,
        "BaO": 0.049879597,
        "BaO_2": -0.000168063,
        "ZnO": 0.025221567,
        "ZnO_2": 0.000099961,
        "PbO": 0.070020298,
        "PbO_2": 0.000214424,
        "PbO_3": -0.000001502,
        "FexOy": 0.036995747,
        "MnxOy": 0.016648722,
        "TiO2": 0.018820343,
        "ZrO2": 0.043059714,
        "ZrO2_2": -0.000779078,
        "CexOy": 0.061277268,
        "CdO": 0.052945783,
        "La2O3": 0.10643194,
        "Nd2O3": 0.090134135,
        "NiO": 0.024289113,
        "ThO2": 0.090253734,
        "UxOy": 0.063297404,
        "SbxOy": 0.044258719,
        "SO3": -0.044488661,
        "F": 0.00109839,
        "Cl": -0.006092537,
        "Remainder": 0.02514614,
        "Na2O_K2O": -0.000395491,
        "Na2O_Li2O": -0.00031449,
        "K2O_Li2O": -0.000329725,
        "Na2O_B2O3": 0.000242157,
        "K2O_B2O3": 0.000259927,
        "Li2O_B2O3": 0.000106359,
        "MgO_B2O3": -0.000206488,
        "CaO_B2O3": -0.000032258,
        "PbO_B2O3": -0.000186195,
        "FexOy_B2O3": -0.000720268,
        "ZrO2_B2O3": -0.000697195,
        "Al2O3_B2O3": -0.000735749,
        "Li2O_Al2O3": -0.000116227,
        "Na2O_Al2O3": -0.000253454,
        "K2O_Al2O3": -0.000371858,
        "MgO_CaO": 0.000057248,
        "MgO_Al2O3": 0.000167218,
        "MgO_ZnO": 0.000220766,
        "Li2O_CaO": -0.00008792,
        "Na2O_MgO": -0.000300745,
        "Na2O_CaO": -0.000228249,
        "Na2O_SrO": -0.00023137,
        "Na2O_BaO": -0.000171693,
        "K2O_MgO": -0.000337747,
        "K2O_CaO": -0.000349578,
        "K2O_SrO": -0.000425589,
        "K2O_BaO": -0.000392897,
        "Al2O3_CaO": -0.000102444,
        "Al2O3_PbO": -0.000651745,
        "Al2O3_TiO2": -0.000563594,
        "Al2O3_BaO": -0.000273835,
        "Al2O3_SrO": -0.000177761,
        "Al2O3_ZnO": -0.000109968,
        "Al2O3_ZrO2": -0.002381651,
        "Na2O_PbO": -0.000036455,
        "Na2O_TiO2": -0.00014331,
        "Na2O_ZnO": -0.000155275,
        "Na2O_ZrO2": -0.000126728,
        "Na2O_FexOy": -0.000371343,
        "K2O_PbO": -0.000525213,
        "K2O_TiO2": -0.000386587,
        "K2O_ZnO": -0.000329812,
        "CaO_PbO": -0.00084145,
        "ZnO_FexOy": -0.001536804,
        "Na2O_K2O_B2O3": -0.000032967,
        "Na2O_MgO_CaO": -0.000009143,
        "Na2O_MgO_Al2O3": -0.000012286,
        "Na2O_CaO_Al2O3": -0.000005106,
        "Na2O_CaO_PbO": 0.000100796,
        "K2O_MgO_CaO": -0.00001217,
        "K2O_MgO_Al2O3": -0.000041908,
        "K2O_CaO_Al2O3": -0.000012421,
        "K2O_CaO_PbO": 0.000125759,
        "MgO_CaO_Al2O3": -0.000011236,
        "CaO_Al2O3_Li2O": -0.000016177,
        "Al2O3_B2O3_PbO": 0.000030116,
    }
    try:
        mole_fractions = extract_composition(composition)
    except (ValueError, TypeError) as e:
        error_msg = f"Invalid composition {composition!r}: {e}"
        raise ValueError(error_msg) from e

    concentrations = {oxide: frac * 100 for oxide, frac in mole_fractions.items()}
    remainder_conc = 0.0
    main_components = []

    for oxide, conc in concentrations.items():
        if conc < 0:
            error_msg = f"Negative concentration for '{oxide}': {conc}"
            raise ValueError(error_msg)

        if oxide == "SiO2":
            continue

        if oxide in TRACE_OXIDES:
            remainder_conc += conc
        else:
            main_components.append(oxide)

    if remainder_conc > 0 and "Remainder" not in COEFFICIENTS:
        error_msg = "Trace oxides present but 'Remainder' coefficient missing"
        raise ValueError(error_msg)

    valid_components = set(COEFFICIENTS.keys()) - {"b0", "Remainder"}
    for comp in main_components:
        if comp not in valid_components:
            error_msg = f"Component '{comp}' not in density model coefficients"
            raise ValueError(error_msg)

    density = COEFFICIENTS.get("b0", 0)
    if remainder_conc > 0:
        density += COEFFICIENTS.get("Remainder", 0) * remainder_conc

    for comp in main_components:
        conc = concentrations[comp]
        if comp in COEFFICIENTS:
            density += COEFFICIENTS[comp] * conc
        quad_key = f"{comp}_2"
        if quad_key in COEFFICIENTS:
            density += COEFFICIENTS[quad_key] * (conc**2)
        cube_key = f"{comp}_3"
        if cube_key in COEFFICIENTS:
            density += COEFFICIENTS[cube_key] * (conc**3)

    TWO_WAY_KEY_PARTS = 1
    THREE_WAY_KEY_PARTS = 2
    two_way_keys = {k for k in COEFFICIENTS if k.count("_") == TWO_WAY_KEY_PARTS and not k.endswith(("_2", "_3"))}
    three_way_keys = {k for k in COEFFICIENTS if k.count("_") == THREE_WAY_KEY_PARTS}

    for key in two_way_keys:
        comp1, comp2 = key.split("_")
        if comp1 in main_components and comp2 in main_components:
            density += COEFFICIENTS[key] * concentrations[comp1] * concentrations[comp2]

    for key in three_way_keys:
        comp1, comp2, comp3 = key.split("_")
        if comp1 in main_components and comp2 in main_components and comp3 in main_components:
            density += COEFFICIENTS[key] * concentrations[comp1] * concentrations[comp2] * concentrations[comp3]

    return density
