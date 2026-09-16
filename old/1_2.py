"""
Composition range scan for equilibrium melting assessment with MELTS.

Edit COMPOSITION_MODE, OXIDE_RANGES, FE3_FET_LIQ and temperature settings.

The script calculates first liquid, T10, T50, T90 and T99 for every valid
composition. All reported and exported temperatures are degC.

Each temperature is evaluated as an independent fixed bulk composition and
fixed pressure MELTS equilibrium calculation. It is not a fractional
crystallisation calculation.
"""

from pathlib import Path
from itertools import product
import csv
import math
import multiprocessing

import numpy as np
import pandas as pd
import petthermotools as ptt


# ============================================================================
# USER INPUT
# ============================================================================

COMPOSITION_MODE = "BALANCE_COMPONENT"
BALANCE_OXIDE = "SiO2"

# Format:
# "oxide": (minimum_wt_pct, maximum_wt_pct, increment_wt_pct)
#
# Enter FeO and Fe2O3 ranges here manually.
# MELTS receives their combined iron content as FeOt.
OXIDE_RANGES = {
    "CaO": (15.0, 30.0, 5.0),
    "SiO2": (45.0, 80.0, 5.0),
    "Al2O3": (0.0, 10.0, 5.0),
    "Fe2O3": (0.0, 5.0, 2.5),
    "FeO": (0.0, 0.0, 1.0),
    "MgO": (0.0, 5.0, 2.5),
}

PRESSURE_BAR = 1.0
MELTS_MODEL = "MELTSv1.0.2"

# This is the only redox setting sent to MELTS.
#
# It is constant for every scanned composition.
#
# Example values:
# 0.00 means all Fe treated as Fe2+
# 0.20 means 20% of total Fe treated as Fe3+
# 0.80 means 80% of total Fe treated as Fe3+
FE3_FET_LIQ = 0.80

T_MIN_C = 1000.0
T_MAX_C = 1800.0
COARSE_STEP_C = 50.0
REFINEMENT_TOLERANCE_C = 1.0

PHASE_FRACTION_TOLERANCE = 1.0e-6
LIQUID_FRACTION_MILESTONES = (0.10, 0.50, 0.90, 0.99)
REPORTING_PHASE_FRACTION_MIN = 1.0e-4

# Keep refine equal to zero because this script performs its own threshold
# refinement using bisection.
PHASE_DIAGRAM_I_MAX = 15
PHASE_DIAGRAM_REFINE = 0

TOP_RESULTS_TO_PRINT = 15

OUTPUT_FILE = Path(__file__).with_name(
    "melts_composition_melting_scan_results.csv"
)

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.6882


# ============================================================================
# MELTS INPUTS
# ============================================================================

MELTS_INPUT_OXIDES = {
    "SiO2",
    "TiO2",
    "Al2O3",
    "Fe2O3",
    "FeO",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
    "H2O",
    "CO2",
}

PHASE_LABELS = {
    "liq": "LIQUID [MELTS liquid]",
    "liquid": "LIQUID [MELTS liquid]",
    "ol": "OLIVINE [(Mg,Fe)2SiO4 solution]",
    "olivine": "OLIVINE [(Mg,Fe)2SiO4 solution]",
    "opx": "ORTHOPYROXENE [(Mg,Fe)SiO3 solution]",
    "orthopyroxene": "ORTHOPYROXENE [(Mg,Fe)SiO3 solution]",
    "cpx": "CLINOPYROXENE [Ca(Mg,Fe)Si2O6 solution]",
    "clinopyroxene": "CLINOPYROXENE [Ca(Mg,Fe)Si2O6 solution]",
    "plag": "PLAGIOCLASE [NaAlSi3O8 to CaAl2Si2O8 solution]",
    "plagioclase": "PLAGIOCLASE [NaAlSi3O8 to CaAl2Si2O8 solution]",
    "fsp": "ALKALI FELDSPAR [alkali feldspar solution]",
    "quartz": "QUARTZ [SiO2]",
    "tridymite": "TRIDYMITE [SiO2]",
    "cristobalite": "CRISTOBALITE [SiO2]",
    "sp": "SPINEL [spinel type oxide]",
    "spinel": "SPINEL [spinel type oxide]",
    "rhm": "RHOMBOHEDRAL OXIDE [hematite ilmenite type]",
    "ilmenite": "ILMENITE [FeTiO3 type]",
    "garnet": "GARNET [garnet solid solution]",
    "melilite": "MELILITE [melilite solid solution]",
    "nepheline": "NEPHELINE [NaAlSiO4 type]",
    "leucite": "LEUCITE [KAlSi2O6]",
    "kalsilite": "KALSILITE [KAlSiO4]",
}


# ============================================================================
# COMPOSITION FUNCTIONS
# ============================================================================

def inclusive_range(start, stop, step):
    if step <= 0:
        raise ValueError("Each composition increment must be greater than zero.")

    if stop < start:
        raise ValueError(
            "Each composition maximum must not be below its minimum."
        )

    count = int(math.floor((stop - start) / step + 1.0e-10))

    return [
        round(start + index * step, 10)
        for index in range(count + 1)
    ]


def generate_compositions():
    unknown_oxides = set(OXIDE_RANGES) - MELTS_INPUT_OXIDES

    if unknown_oxides:
        raise ValueError(
            "Unsupported MELTS input oxides: "
            + ", ".join(sorted(unknown_oxides))
        )

    if COMPOSITION_MODE not in {"BALANCE_COMPONENT", "FULL_GRID"}:
        raise ValueError(
            "COMPOSITION_MODE must be BALANCE_COMPONENT or FULL_GRID."
        )

    if (
        COMPOSITION_MODE == "BALANCE_COMPONENT"
        and BALANCE_OXIDE not in OXIDE_RANGES
    ):
        raise ValueError(
            "BALANCE_OXIDE must appear in OXIDE_RANGES."
        )

    oxides = list(OXIDE_RANGES)

    values = {
        oxide: inclusive_range(*OXIDE_RANGES[oxide])
        for oxide in oxides
    }

    compositions = []
    tolerance = 1.0e-8

    if COMPOSITION_MODE == "FULL_GRID":
        for combination in product(*(values[oxide] for oxide in oxides)):
            composition = dict(zip(oxides, combination))

            if abs(sum(composition.values()) - 100.0) <= tolerance:
                compositions.append(composition)

    else:
        scanned_oxides = [
            oxide
            for oxide in oxides
            if oxide != BALANCE_OXIDE
        ]

        balance_min, balance_max, _ = OXIDE_RANGES[BALANCE_OXIDE]

        for combination in product(
            *(values[oxide] for oxide in scanned_oxides)
        ):
            composition = dict(zip(scanned_oxides, combination))

            balance_value = 100.0 - sum(composition.values())

            if (
                balance_min - tolerance
                <= balance_value
                <= balance_max + tolerance
            ):
                composition[BALANCE_OXIDE] = round(balance_value, 10)

                compositions.append(
                    {
                        oxide: composition[oxide]
                        for oxide in oxides
                    }
                )

    if not compositions:
        raise ValueError(
            "No valid compositions were generated. "
            "Check ranges and balance oxide."
        )

    return compositions, oxides


# ============================================================================
# MELTS COMPOSITION CONVERSION
# ============================================================================

def convert_fe_oxides_to_feot(composition):
    """
    Convert manually entered FeO and Fe2O3 amounts to FeOt as FeO equivalent.
    """
    feo_wt_pct = composition.get("FeO", 0.0)
    fe2o3_wt_pct = composition.get("Fe2O3", 0.0)

    fe2o3_as_feo = fe2o3_wt_pct * (
        2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3
    )

    return feo_wt_pct + fe2o3_as_feo


def create_melts_bulk(composition):
    """
    Build a normalized MELTS liquid input composition.

    FeO and Fe2O3 are converted to one FeOt_Liq input because MELTS does not
    accept FeO_Liq and Fe2O3_Liq as separate bulk inputs.
    """
    bulk = {
        "SiO2_Liq": composition.get("SiO2", 0.0),
        "TiO2_Liq": composition.get("TiO2", 0.0),
        "Al2O3_Liq": composition.get("Al2O3", 0.0),
        "FeOt_Liq": convert_fe_oxides_to_feot(composition),
        "MnO_Liq": composition.get("MnO", 0.0),
        "MgO_Liq": composition.get("MgO", 0.0),
        "CaO_Liq": composition.get("CaO", 0.0),
        "Na2O_Liq": composition.get("Na2O", 0.0),
        "K2O_Liq": composition.get("K2O", 0.0),
        "P2O5_Liq": composition.get("P2O5", 0.0),
        "H2O_Liq": composition.get("H2O", 0.0),
        "CO2_Liq": composition.get("CO2", 0.0),
    }

    total_before_normalisation = sum(bulk.values())

    if total_before_normalisation <= 0.0:
        raise ValueError(
            "Composition contains no positive oxide amount."
        )

    normalised_bulk = {
        oxide: 100.0 * value / total_before_normalisation
        for oxide, value in bulk.items()
    }

    return normalised_bulk, total_before_normalisation


# ============================================================================
# MELTS OUTPUT PROCESSING
# ============================================================================

def phase_label(phase_identifier):
    identifier = str(phase_identifier).strip().lower()

    return PHASE_LABELS.get(
        identifier,
        f"{phase_identifier} [MELTS phase identifier]"
    )


def find_column(frame, preferred_name, alternative_names=()):
    for name in (preferred_name, *alternative_names):
        if name in frame.columns:
            return name

    lower_case_columns = {
        str(column).lower(): column
        for column in frame.columns
    }

    for name in (preferred_name, *alternative_names):
        if name.lower() in lower_case_columns:
            return lower_case_columns[name.lower()]

    return None


def select_temperature_row(result, requested_temperature_c):
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "MELTS did not return a pandas DataFrame."
        )

    if result.empty:
        raise RuntimeError(
            "MELTS returned no equilibrium calculation rows."
        )

    temperature_column = find_column(
        result,
        "T_C",
        alternative_names=("T_Celsius", "Temperature_C"),
    )

    if temperature_column is None:
        raise RuntimeError(
            "MELTS output has no recognised temperature column."
        )

    output_temperatures = pd.to_numeric(
        result[temperature_column],
        errors="coerce",
    )

    distance = (
        output_temperatures - requested_temperature_c
    ).abs()

    if distance.isna().all():
        raise RuntimeError(
            "MELTS output temperature column contains no numeric values."
        )

    selected_index = distance.idxmin()

    if float(distance.loc[selected_index]) > 1.0e-4:
        raise RuntimeError(
            f"MELTS did not return requested temperature "
            f"{requested_temperature_c:.6f} degC."
        )

    return result.loc[selected_index]


def phase_fractions_from_row(row):
    total_mass_column = next(
        (
            name
            for name in ("mass_g", "Mass_g", "mass")
            if name in row.index
        ),
        None,
    )

    liquid_mass_column = next(
        (
            name
            for name in ("mass_g_Liq", "mass_g_liq")
            if name in row.index
        ),
        None,
    )

    if total_mass_column is None:
        raise RuntimeError(
            "MELTS output lacks total system mass column mass_g."
        )

    if liquid_mass_column is None:
        raise RuntimeError(
            "MELTS output lacks liquid mass column mass_g_Liq."
        )

    total_mass = float(row[total_mass_column])

    if pd.isna(row[liquid_mass_column]):
        liquid_mass = 0.0
    else:
        liquid_mass = float(row[liquid_mass_column])

    if not np.isfinite(total_mass) or total_mass <= 0.0:
        raise RuntimeError(
            "MELTS returned an invalid total system mass."
        )

    if not np.isfinite(liquid_mass):
        raise RuntimeError(
            "MELTS returned an invalid liquid mass."
        )

    liquid_fraction = liquid_mass / total_mass

    if liquid_fraction < -1.0e-8:
        raise RuntimeError(
            f"MELTS returned negative liquid fraction "
            f"{liquid_fraction:.8f}."
        )

    if liquid_fraction > 1.0 + 1.0e-6:
        raise RuntimeError(
            f"MELTS returned liquid fraction above one "
            f"{liquid_fraction:.8f}."
        )

    liquid_fraction = max(0.0, min(1.0, liquid_fraction))

    stable_phases = []

    for column, value in row.items():
        column_name = str(column)

        if not column_name.startswith("mass_g_"):
            continue

        if column_name.lower() == "mass_g_liq":
            continue

        if pd.isna(value):
            continue

        phase_mass = float(value)

        if not np.isfinite(phase_mass):
            continue

        if phase_mass <= 0.0:
            continue

        phase_identifier = column_name[len("mass_g_"):]

        phase_fraction = phase_mass / total_mass

        stable_phases.append(
            (phase_identifier, phase_fraction)
        )

    if liquid_mass > 0.0:
        stable_phases.append(("Liq", liquid_fraction))

    stable_phases.sort(
        key=lambda item: item[0].lower()
    )

    return liquid_fraction, stable_phases


def format_phases(stable_phases):
    phase_text = [
        f"{phase_label(name)}: {fraction:.5f}"
        for name, fraction in stable_phases
        if fraction >= REPORTING_PHASE_FRACTION_MIN
    ]

    if phase_text:
        return "; ".join(phase_text)

    return "No phase above reporting threshold"


# ============================================================================
# MELTS EQUILIBRIUM CALCULATION
# ============================================================================

def equilibrium_at_temperature(
    melts_bulk,
    requested_temperature_c,
):
    """
    Evaluate MELTS at a requested temperature.

    Two nearly identical temperatures are passed intentionally because some
    PetThermoTools phaseDiagram_calc versions raise UnboundLocalError when only
    one temperature is supplied.

    Only the row corresponding to requested_temperature_c is extracted.
    """
    requested_temperature_c = float(requested_temperature_c)

    calculation_temperatures_c = np.asarray(
        [
            requested_temperature_c,
            requested_temperature_c + 0.01,
        ]
    )

    result = ptt.phaseDiagram_calc(
        Model=MELTS_MODEL,
        bulk=melts_bulk,
        P_bar=PRESSURE_BAR,
        T_C=calculation_temperatures_c,
        Fe3Fet_Liq=FE3_FET_LIQ,
        i_max=PHASE_DIAGRAM_I_MAX,
        refine=PHASE_DIAGRAM_REFINE,
    )

    row = select_temperature_row(
        result=result,
        requested_temperature_c=requested_temperature_c,
    )

    return phase_fractions_from_row(row)


# ============================================================================
# THRESHOLD SEARCH
# ============================================================================

def find_first_crossing(scan_rows, threshold):
    for previous, current in zip(scan_rows[:-1], scan_rows[1:]):
        if previous[1] < threshold and current[1] >= threshold:
            return previous[0], current[0]

    return None


def refine_first_crossing(
    threshold,
    lower_c,
    upper_c,
    evaluate,
):
    lower = float(lower_c)
    upper = float(upper_c)

    while upper - lower > REFINEMENT_TOLERANCE_C:
        midpoint = 0.5 * (lower + upper)

        liquid_fraction, _ = evaluate(midpoint)

        if liquid_fraction >= threshold:
            upper = midpoint
        else:
            lower = midpoint

    return upper


# ============================================================================
# ONE COMPOSITION CALCULATION
# ============================================================================

def calculate_composition(oxide_composition):
    melts_bulk, total_before_normalisation = create_melts_bulk(
        oxide_composition
    )

    cache = {}

    def evaluate(temperature_c):
        key = round(float(temperature_c), 8)

        if key not in cache:
            cache[key] = equilibrium_at_temperature(
                melts_bulk=melts_bulk,
                requested_temperature_c=temperature_c,
            )

        return cache[key]

    temperatures = np.arange(
        T_MIN_C,
        T_MAX_C + 0.5 * COARSE_STEP_C,
        COARSE_STEP_C,
    )

    scan_rows = []

    for temperature_c in temperatures:
        liquid_fraction, stable_phases = evaluate(temperature_c)

        scan_rows.append(
            (
                temperature_c,
                liquid_fraction,
                stable_phases,
            )
        )

    result = {
        "FeOt_before_normalisation_wt_pct":
            convert_fe_oxides_to_feot(oxide_composition),

        "MELTS_bulk_total_before_normalisation_wt_pct":
            total_before_normalisation,

        "Fe3Fet_Liq_input":
            FE3_FET_LIQ,
    }

    thresholds = {
        "first_liquid_degC": PHASE_FRACTION_TOLERANCE,
        "T10_degC": 0.10,
        "T50_degC": 0.50,
        "T90_degC": 0.90,
        "T99_degC": 0.99,
    }

    for temperature_column, threshold in thresholds.items():
        fraction_column = temperature_column.replace(
            "degC",
            "liquid_fraction",
        )

        phases_column = temperature_column.replace(
            "degC",
            "phases",
        )

        bracket = find_first_crossing(
            scan_rows,
            threshold,
        )

        if bracket is None:
            result[temperature_column] = None
            result[fraction_column] = None
            result[phases_column] = (
                "Not bracketed in selected range"
            )

            continue

        threshold_temperature_c = refine_first_crossing(
            threshold=threshold,
            lower_c=bracket[0],
            upper_c=bracket[1],
            evaluate=evaluate,
        )

        liquid_fraction, stable_phases = evaluate(
            threshold_temperature_c
        )

        result[temperature_column] = threshold_temperature_c
        result[fraction_column] = liquid_fraction
        result[phases_column] = format_phases(stable_phases)

    maximum_liquid_row = max(
        scan_rows,
        key=lambda row: row[1],
    )

    result["maximum_liquid_fraction"] = maximum_liquid_row[1]
    result["maximum_liquid_temperature_degC"] = maximum_liquid_row[0]
    result["maximum_liquid_phases"] = format_phases(
        maximum_liquid_row[2]
    )

    return result


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def write_results(output_file, oxide_names, results):
    metric_columns = [
        "status",
        "message",

        "FeOt_before_normalisation_wt_pct",
        "MELTS_bulk_total_before_normalisation_wt_pct",
        "Fe3Fet_Liq_input",

        "first_liquid_degC",
        "first_liquid_liquid_fraction",
        "first_liquid_phases",

        "T10_degC",
        "T10_liquid_fraction",
        "T10_phases",

        "T50_degC",
        "T50_liquid_fraction",
        "T50_phases",

        "T90_degC",
        "T90_liquid_fraction",
        "T90_phases",

        "T99_degC",
        "T99_liquid_fraction",
        "T99_phases",

        "maximum_liquid_fraction",
        "maximum_liquid_temperature_degC",
        "maximum_liquid_phases",
    ]

    field_names = oxide_names + metric_columns

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(results)


def print_risk_summary(oxide_names, results):
    successful = [
        row
        for row in results
        if (
            row["status"] == "OK"
            and row.get("first_liquid_degC") is not None
        )
    ]

    successful.sort(
        key=lambda row: row["first_liquid_degC"]
    )

    print("\nMOST MELT SENSITIVE COMPOSITIONS")
    print(
        "Composition is analysed oxide wt%. "
        "T99 is the practical liquidus criterion."
    )

    composition_header = "  ".join(
        f"{oxide:>8}"
        for oxide in oxide_names
    )

    print(
        f"{composition_header}  "
        f"{'First liquid degC':>18}  "
        f"{'T50 degC':>10}  "
        f"{'T99 degC':>10}"
    )

    print(
        "=" * max(
            100,
            len(composition_header) + 46,
        )
    )

    for row in successful[:TOP_RESULTS_TO_PRINT]:
        composition_text = "  ".join(
            f"{row.get(oxide, 0.0):8.2f}"
            for oxide in oxide_names
        )

        t50 = row.get("T50_degC")
        t99 = row.get("T99_degC")

        t50_display = (
            t50
            if t50 is not None
            else float("nan")
        )

        t99_display = (
            t99
            if t99 is not None
            else float("nan")
        )

        print(
            f"{composition_text}  "
            f"{row['first_liquid_degC']:18.2f}  "
            f"{t50_display:10.2f}  "
            f"{t99_display:10.2f}"
        )

    failures = [
        row
        for row in results
        if row["status"] != "OK"
    ]

    print(f"\nSuccessful compositions: {len(successful)}")
    print(f"Failed compositions: {len(failures)}")

    if failures:
        print(
            "Failed rows are retained in the CSV with the error message."
        )


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    if FE3_FET_LIQ < 0.0 or FE3_FET_LIQ > 1.0:
        raise ValueError(
            "FE3_FET_LIQ must be between 0.0 and 1.0."
        )

    compositions, oxide_names = generate_compositions()

    print(
        f"Generated valid compositions: {len(compositions)}"
    )

    print(f"MELTS model: {MELTS_MODEL}")

    print(
        f"Temperature range: "
        f"{T_MIN_C:.0f} to {T_MAX_C:.0f} degC"
    )

    print(
        f"Temperature increment: "
        f"{COARSE_STEP_C:.1f} degC"
    )

    print(
        f"Pressure: {PRESSURE_BAR:.4f} bar"
    )

    print(
        f"Fixed Fe3+ to Fet ratio sent to MELTS: "
        f"{FE3_FET_LIQ:.5f}"
    )

    results = []

    for index, oxide_composition in enumerate(
        compositions,
        start=1,
    ):
        composition_text = ", ".join(
            f"{oxide}={value:.2f}"
            for oxide, value in oxide_composition.items()
        )

        print(
            f"Calculating {index} of {len(compositions)}: "
            f"{composition_text}"
        )

        row = dict(oxide_composition)

        try:
            row.update(
                calculate_composition(oxide_composition)
            )

            row["status"] = "OK"
            row["message"] = ""

        except Exception as error:
            row["status"] = "FAILED"

            row["message"] = (
                f"{type(error).__name__}: {error}"
            )

        results.append(row)

    write_results(
        output_file=OUTPUT_FILE,
        oxide_names=oxide_names,
        results=results,
    )

    print_risk_summary(
        oxide_names=oxide_names,
        results=results,
    )

    print(
        f"\nFull results written to: "
        f"{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()