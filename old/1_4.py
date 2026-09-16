"""Composition range scan for equilibrium liquid fraction assessment with MELTS.

Edit COMPOSITION_MODE, OXIDE_RANGES, redox settings and the temperature settings.
For every valid bulk composition, the script calculates first liquid, T10, T50,
T90 and T99. All temperatures shown and exported are degC.

The calculation uses PetThermoTools phaseDiagram_calc at fixed pressure and
therefore evaluates independent equilibrium states at each temperature. It does
not calculate a fractional crystallisation path.
"""

from pathlib import Path
from itertools import product
import csv
import math
import multiprocessing

import numpy as np
import pandas as pd
import petthermotools as ptt


# USER INPUT
OUTPUT_FILE = Path(__file__).with_name("melts_composition_melting_scan_results.csv")

# Use BALANCE_COMPONENT for the recommended workflow. All oxide ranges except
# BALANCE_OXIDE are scanned independently. BALANCE_OXIDE is calculated as the
# remainder to give exactly 100 wt%. Its range is still enforced.
#
# Use FULL_GRID only when every listed range and increment deliberately gives
# combinations that sum to exactly 100 wt%. This mode can easily yield no rows.
COMPOSITION_MODE = "BALANCE_COMPONENT"
BALANCE_OXIDE = "SiO2"

# Use the conventional analysed oxide names here. Fe2O3 and FeO are converted
# internally to MELTS FeOt_Liq and Fe3Fet_Liq.
#
# Keep the initial grid small. Calculation time rises rapidly with every added
# component value because every composition needs a complete temperature scan.
OXIDE_RANGES = {
    "CaO": (15.0, 30.0, 5.0),
    "SiO2": (45.0, 80.0, 5.0),
    "Al2O3": (0.0, 10.0, 5.0),
    "Fe2O3": (0.0, 5.0, 2.5),
    "FeO": (0.0, 0.0, 1.0),
    "MgO": (0.0, 5.0, 2.5),
    "TiO2": (0.0, 0.0, 1.0),
    "MnO": (0.0, 0.0, 1.0),
    "Na2O": (0.0, 0.0, 1.0),
    "K2O": (0.0, 0.0, 1.0),
    "P2O5": (0.0, 0.0, 1.0),
    "H2O": (0.0, 0.0, 1.0),
    "CO2": (0.0, 0.0, 1.0),
}

PRESSURE_BAR = 1.0
MELTS_MODEL = "MELTSv1.0.2"
T_MIN_C = 1000.0
T_MAX_C = 1800.0
COARSE_STEP_C = 50.0
REFINEMENT_TOLERANCE_C = 1.0
PHASE_FRACTION_TOLERANCE = 1.0e-6

# T99 is the practical liquidus for deposit and refractory risk screening.
PRACTICAL_LIQUIDUS_FRACTION = 0.99
LIQUID_FRACTION_MILESTONES = (0.10, 0.50, 0.90, 0.99)
REPORTING_PHASE_FRACTION_MIN = 1.0e-4
TOP_RESULTS_TO_PRINT = 15

# MELTS accepts total iron as FeOt and one Fe3+/Fet value. FROM_INPUT_OXIDES
# derives Fe3+/Fet from the entered Fe2O3 and FeO. FIXED uses FIXED_FE3_FET.
# FULLY_FERRIC and FULLY_FERROUS inputs are clipped very slightly because some
# MELTS installations are more stable with an interior redox ratio.
REDOX_MODE = "FROM_INPUT_OXIDES"  # FROM_INPUT_OXIDES or FIXED
FIXED_FE3_FET = 0.80
FE3_FET_MIN = 1.0e-6
FE3_FET_MAX = 1.0 - 1.0e-6

# Some alphaMELTS installations fail before solving a zero total iron bulk.
# TRY_WITHOUT_RED0X first preserves exactly zero iron and omits Fe3Fet_Liq.
# TRACE_FEOT_RETRY retries only failed zero iron rows using this negligible
# FeOt amount, then records that fact in the CSV. FAIL reports the error.
ZERO_IRON_POLICY = "TRACE_FEOT_RETRY"  # TRY_WITHOUT_RED0X, TRACE_FEOT_RETRY, FAIL
TRACE_FEOT_WT_PCT = 1.0e-8

# phaseDiagram_calc can refine phase boundaries by inserting interpolated and
# additional calculated points. Set this to zero because this script performs
# its own crossing refinement and requires only explicitly requested T values.
PHASE_DIAGRAM_I_MAX = 8
PHASE_DIAGRAM_REFINE = 0

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.6882

MELTS_OXIDE_COLUMNS = {
    "SiO2": "SiO2_Liq",
    "TiO2": "TiO2_Liq",
    "Al2O3": "Al2O3_Liq",
    "MnO": "MnO_Liq",
    "MgO": "MgO_Liq",
    "CaO": "CaO_Liq",
    "Na2O": "Na2O_Liq",
    "K2O": "K2O_Liq",
    "P2O5": "P2O5_Liq",
    "H2O": "H2O_Liq",
    "CO2": "CO2_Liq",
}


def inclusive_range(start, stop, step):
    if step <= 0:
        raise ValueError("Each composition increment must be greater than zero.")
    if stop < start:
        raise ValueError("Each composition maximum must not be below its minimum.")
    count = int(math.floor((stop - start) / step + 1.0e-10))
    return [round(start + index * step, 10) for index in range(count + 1)]


def generate_compositions():
    unknown = set(OXIDE_RANGES) - (set(MELTS_OXIDE_COLUMNS) | {"FeO", "Fe2O3"})
    if unknown:
        raise ValueError("Unsupported oxides: " + ", ".join(sorted(unknown)))
    if COMPOSITION_MODE not in {"BALANCE_COMPONENT", "FULL_GRID"}:
        raise ValueError("COMPOSITION_MODE must be BALANCE_COMPONENT or FULL_GRID.")
    if COMPOSITION_MODE == "BALANCE_COMPONENT" and BALANCE_OXIDE not in OXIDE_RANGES:
        raise ValueError("BALANCE_OXIDE must appear in OXIDE_RANGES.")

    oxides = list(OXIDE_RANGES)
    values = {oxide: inclusive_range(*OXIDE_RANGES[oxide]) for oxide in oxides}
    tolerance = 1.0e-8
    compositions = []

    if COMPOSITION_MODE == "FULL_GRID":
        for combination in product(*(values[oxide] for oxide in oxides)):
            composition = dict(zip(oxides, combination))
            if abs(sum(composition.values()) - 100.0) <= tolerance:
                compositions.append(composition)
    else:
        scanned_oxides = [oxide for oxide in oxides if oxide != BALANCE_OXIDE]
        lower, upper, _ = OXIDE_RANGES[BALANCE_OXIDE]
        for combination in product(*(values[oxide] for oxide in scanned_oxides)):
            composition = dict(zip(scanned_oxides, combination))
            balance_value = 100.0 - sum(composition.values())
            if lower - tolerance <= balance_value <= upper + tolerance:
                composition[BALANCE_OXIDE] = round(balance_value, 10)
                compositions.append({oxide: composition[oxide] for oxide in oxides})

    if not compositions:
        raise ValueError("No valid compositions were generated. Check ranges and balance oxide.")
    return compositions, oxides


def feot_as_feo(composition):
    return composition.get("FeO", 0.0) + composition.get("Fe2O3", 0.0) * (
        2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3
    )


def fe3_fet_from_input(composition):
    moles_fe2 = composition.get("FeO", 0.0) / MOLAR_MASS_FEO
    moles_fe3 = 2.0 * composition.get("Fe2O3", 0.0) / MOLAR_MASS_FE2O3
    total_iron_moles = moles_fe2 + moles_fe3
    if total_iron_moles <= 0:
        return None
    return moles_fe3 / total_iron_moles


def create_melts_bulk(composition, add_trace_iron=False):
    bulk = {column: composition.get(oxide, 0.0) for oxide, column in MELTS_OXIDE_COLUMNS.items()}
    feot = feot_as_feo(composition)
    if add_trace_iron and feot <= 0:
        feot = TRACE_FEOT_WT_PCT
    bulk["FeOt_Liq"] = feot

    total = sum(bulk.values())
    if total <= 0:
        raise ValueError("Composition contains no positive MELTS oxide amount.")
    normalised = {name: 100.0 * value / total for name, value in bulk.items()}

    if feot <= 0:
        return normalised, None, total
    if REDOX_MODE == "FIXED":
        fe3_fet = FIXED_FE3_FET
    elif REDOX_MODE == "FROM_INPUT_OXIDES":
        fe3_fet = fe3_fet_from_input(composition)
        if fe3_fet is None:
            fe3_fet = FIXED_FE3_FET
    else:
        raise ValueError("REDOX_MODE must be FROM_INPUT_OXIDES or FIXED.")
    return normalised, min(FE3_FET_MAX, max(FE3_FET_MIN, float(fe3_fet))), total


def extract_phase_fractions(result_row):
    total_mass = float(result_row.get("mass_g", np.nan))
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise RuntimeError("MELTS result has no valid total system mass.")

    phases = []
    for column, value in result_row.items():
        if not column.startswith("mass_g_"):
            continue
        phase = column.removeprefix("mass_g_")
        if phase.lower() in {"liq", "liquid"}:
            continue
        if pd.notna(value) and float(value) > 0:
            phases.append((phase, float(value) / total_mass))

    liquid_mass = 0.0
    for liquid_column in ("mass_g_Liq", "mass_g_liq", "mass_g_Liquid"):
        if liquid_column in result_row and pd.notna(result_row[liquid_column]):
            liquid_mass = float(result_row[liquid_column])
            break
    liquid_fraction = liquid_mass / total_mass
    if liquid_fraction > 0:
        phases.append(("Liq", liquid_fraction))
    return liquid_fraction, phases


def format_phases(phases):
    shown = [
        f"{name}: {fraction:.5f}"
        for name, fraction in sorted(phases)
        if fraction >= REPORTING_PHASE_FRACTION_MIN
    ]
    return "; ".join(shown) if shown else "No phase above reporting threshold"


def equilibrium_at_temperatures(melts_bulk, fe3_fet, temperatures_c):
    temperatures_c = np.asarray(temperatures_c, dtype=float)
    kwargs = {
        "Model": MELTS_MODEL,
        "bulk": melts_bulk,
        "P_bar": np.full(temperatures_c.size, PRESSURE_BAR),
        "T_C": temperatures_c,
        "i_max": PHASE_DIAGRAM_I_MAX,
        "refine": PHASE_DIAGRAM_REFINE,
    }
    if fe3_fet is not None:
        kwargs["Fe3Fet_Liq"] = np.full(temperatures_c.size, fe3_fet)

    result = ptt.phaseDiagram_calc(**kwargs)
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise RuntimeError("phaseDiagram_calc did not return a nonempty pandas DataFrame.")
    if "T_C" not in result.columns:
        raise RuntimeError("MELTS result does not contain the T_C column.")

    rows = []
    returned_temperatures = result["T_C"].to_numpy(dtype=float)
    for temperature_c in temperatures_c:
        matching = np.flatnonzero(np.isclose(returned_temperatures, temperature_c, atol=1.0e-7))
        if matching.size == 0:
            raise RuntimeError(f"MELTS did not return requested temperature {temperature_c:.6f} degC.")
        row = result.iloc[int(matching[0])]
        liquid_fraction, phases = extract_phase_fractions(row)
        if not np.isfinite(liquid_fraction) or liquid_fraction < -1.0e-8 or liquid_fraction > 1.0 + 1.0e-8:
            raise RuntimeError(f"Invalid liquid fraction at {temperature_c:.6f} degC: {liquid_fraction}")
        rows.append((float(temperature_c), max(0.0, min(1.0, liquid_fraction)), phases))
    return rows


def find_first_crossing(scan_rows, threshold):
    for previous, current in zip(scan_rows[:-1], scan_rows[1:]):
        if previous[1] < threshold and current[1] >= threshold:
            return previous[0], current[0]
    return None


def calculate_composition(oxide_composition):
    melts_bulk, fe3_fet, total_before_normalisation = create_melts_bulk(oxide_composition)
    zero_iron_note = ""
    if fe3_fet is None and ZERO_IRON_POLICY == "FAIL":
        raise ValueError("This composition contains zero total iron and ZERO_IRON_POLICY is FAIL.")

    temperatures = np.arange(T_MIN_C, T_MAX_C + 0.5 * COARSE_STEP_C, COARSE_STEP_C)
    cache = {}

    def evaluate(temperature_c):
        key = round(float(temperature_c), 8)
        if key not in cache:
            try:
                cache[key] = equilibrium_at_temperatures(melts_bulk, fe3_fet, [key])[0]
            except Exception as first_error:
                if fe3_fet is not None or ZERO_IRON_POLICY != "TRACE_FEOT_RETRY":
                    raise
                trace_bulk, trace_fe3_fet, _ = create_melts_bulk(oxide_composition, add_trace_iron=True)
                try:
                    cache[key] = equilibrium_at_temperatures(trace_bulk, trace_fe3_fet, [key])[0]
                except Exception as trace_error:
                    raise RuntimeError(
                        f"Zero iron solve failed without redox: {first_error}; "
                        f"trace iron retry failed: {trace_error}"
                    ) from trace_error
        return cache[key]

    try:
        scan_rows = equilibrium_at_temperatures(melts_bulk, fe3_fet, temperatures)
        for row in scan_rows:
            cache[round(row[0], 8)] = row
    except Exception as first_error:
        if fe3_fet is not None or ZERO_IRON_POLICY != "TRACE_FEOT_RETRY":
            raise
        trace_bulk, trace_fe3_fet, _ = create_melts_bulk(oxide_composition, add_trace_iron=True)
        scan_rows = equilibrium_at_temperatures(trace_bulk, trace_fe3_fet, temperatures)
        cache = {round(row[0], 8): row for row in scan_rows}
        melts_bulk, fe3_fet = trace_bulk, trace_fe3_fet
        zero_iron_note = f"Trace FeOt retry used: {TRACE_FEOT_WT_PCT:.3e} wt% before normalisation."

    result = {
        "melts_bulk_total_before_normalisation_wt_pct": total_before_normalisation,
        "FeOt_before_normalisation_wt_pct": feot_as_feo(oxide_composition),
        "Fe3Fet_Liq_used": "" if fe3_fet is None else fe3_fet,
        "zero_iron_handling": zero_iron_note or ("No total iron, no redox parameter passed." if fe3_fet is None else ""),
    }
    thresholds = {"first_liquid_degC": PHASE_FRACTION_TOLERANCE}
    thresholds.update({f"T{int(round(value * 100)):02d}_degC": value for value in LIQUID_FRACTION_MILESTONES})

    for column, threshold in thresholds.items():
        bracket = find_first_crossing(scan_rows, threshold)
        if bracket is None:
            result[column] = None
            result[column.replace("degC", "phases")] = "Not bracketed in selected range"
            continue

        lower, upper = map(float, bracket)
        while upper - lower > REFINEMENT_TOLERANCE_C:
            midpoint = 0.5 * (lower + upper)
            if evaluate(midpoint)[1] >= threshold:
                upper = midpoint
            else:
                lower = midpoint
        temperature_c, liquid_fraction, phases = evaluate(upper)
        result[column] = temperature_c
        result[column.replace("degC", "liquid_fraction")] = liquid_fraction
        result[column.replace("degC", "phases")] = format_phases(phases)

    maximum = max(scan_rows, key=lambda row: row[1])
    result["maximum_liquid_fraction"] = maximum[1]
    result["maximum_liquid_temperature_degC"] = maximum[0]
    result["maximum_liquid_phases"] = format_phases(maximum[2])
    return result


def write_results(output_file, oxide_names, results):
    metric_columns = [
        "status", "message", "melts_bulk_total_before_normalisation_wt_pct",
        "FeOt_before_normalisation_wt_pct", "Fe3Fet_Liq_used", "zero_iron_handling",
        "first_liquid_degC", "first_liquid_liquid_fraction", "first_liquid_phases",
        "T10_degC", "T10_liquid_fraction", "T10_phases",
        "T50_degC", "T50_liquid_fraction", "T50_phases",
        "T90_degC", "T90_liquid_fraction", "T90_phases",
        "T99_degC", "T99_liquid_fraction", "T99_phases",
        "maximum_liquid_fraction", "maximum_liquid_temperature_degC", "maximum_liquid_phases",
    ]
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=oxide_names + metric_columns)
        writer.writeheader()
        writer.writerows(results)


def print_risk_summary(oxide_names, results):
    successful = [row for row in results if row["status"] == "OK" and row.get("first_liquid_degC") is not None]
    successful.sort(key=lambda row: row["first_liquid_degC"])

    print("\nMOST MELT SENSITIVE COMPOSITIONS")
    print("Composition is analysed wt%. T99 is the practical liquidus criterion.")
    print(" | ".join([f"{oxide:>8}" for oxide in oxide_names] + ["First liquid degC", "T50 degC", "T99 degC"]))
    print("=" * max(100, 12 * len(oxide_names) + 48))
    for row in successful[:TOP_RESULTS_TO_PRINT]:
        oxide_text = " | ".join(f"{row.get(oxide, 0.0):8.2f}" for oxide in oxide_names)
        t50 = row.get("T50_degC")
        t99 = row.get("T99_degC")
        print(f"{oxide_text} | {row['first_liquid_degC']:17.2f} | {t50 if t50 is not None else float('nan'):8.2f} | {t99 if t99 is not None else float('nan'):8.2f}")

    failures = [row for row in results if row["status"] != "OK"]
    print(f"\nSuccessful compositions: {len(successful)}")
    print(f"Failed compositions: {len(failures)}")
    if failures:
        print("Failed rows are retained in the CSV with the error message.")


def main():
    compositions, oxide_names = generate_compositions()
    print(f"Generated valid compositions: {len(compositions)}")
    print(f"MELTS model: {MELTS_MODEL}")
    print(f"Temperature range: {T_MIN_C:.0f} to {T_MAX_C:.0f} degC")
    print(f"Temperature increment: {COARSE_STEP_C:.1f} degC")
    print(f"Pressure: {PRESSURE_BAR:.4f} bar")
    print(f"Redox mode: {REDOX_MODE}")
    print(f"Zero iron policy: {ZERO_IRON_POLICY}")

    results = []
    for index, oxide_composition in enumerate(compositions, start=1):
        composition_text = ", ".join(f"{oxide}={value:.2f}" for oxide, value in oxide_composition.items())
        print(f"Calculating {index} of {len(compositions)}: {composition_text}")
        row = dict(oxide_composition)
        try:
            row.update(calculate_composition(oxide_composition))
            row["status"] = "OK"
            row["message"] = ""
        except Exception as error:
            row["status"] = "FAILED"
            row["message"] = f"{type(error).__name__}: {error}"
        results.append(row)

    write_results(OUTPUT_FILE, oxide_names, results)
    print_risk_summary(oxide_names, results)
    print(f"\nFull results written to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
