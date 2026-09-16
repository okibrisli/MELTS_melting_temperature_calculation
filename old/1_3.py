"""Composition range scan for equilibrium melting assessment with MELTS.

Mirrors the pycalphad composition_melting_scan.py workflow, but drives
petthermotools/alphaMELTS instead of a CALPHAD database. For every valid
composition the script runs one equilibrium (non fractionating) isobaric
temperature path with MELTS and extracts first liquid, T10, T50, T90 and T99
from the liquid mass fraction curve. All temperatures shown and exported are
degC.

Important MELTS specific notes, different from the pycalphad script:

1. MELTS solves along a continuous T path, not point by point. Re-querying
   an arbitrary single temperature (as the pycalphad bisection refinement
   does) is slow and can break the solver's initial guess continuity.
   Instead this script runs one dense path per composition (T_STEP_C) and
   linearly interpolates the exact crossing temperature between the two
   bracketing grid points. Lower T_STEP_C for a more precise crossing at
   the cost of run time.
2. MELTS (alphaMELTS) does not tolerate a bulk composition with zero total
   iron. If a scanned/balanced composition ends up with FeOt = 0, a small
   trace amount (MIN_FEOT_WT_PCT) is injected before normalisation so the
   solver does not fail outright. This substitution is flagged in the CSV.
3. PetThermoTools raises a Warning (a Warning is also an Exception subclass
   in Python, so a bare "except Exception" catches it) whenever the bulk
   contains zero H2O. A small trace amount (MIN_H2O_WT_PCT) is injected
   before normalisation whenever H2O is zero, mirroring the iron
   correction, and is also flagged in the CSV.
4. The 'All' result table mixes several column families: run metadata
   (T_C, P_bar, ...), per-phase oxide compositions named "<oxide>_<phase
   suffix>" (for example SiO2_Liq), and per-phase bulk amount columns
   named after the raw MELTS phase key itself (for example "liquid1"),
   not "mass_liquid1". Phase amount columns are therefore identified by
   elimination: anything that is not known run metadata and does not look
   like an "<oxide>_<phase>" composition column. If this still cannot find
   a liquid column on your installed version, the raised error lists every
   column so the exact naming can be fixed in one pass.
"""

from pathlib import Path
from itertools import product
import contextlib
import io
import math
import multiprocessing
import csv
import numpy as np
import pandas as pd
import petthermotools as ptt


# USER INPUT
MELTS_MODEL = "MELTSv1.0.2"
PRESSURE_BAR = 1.0
FE3_FET_LIQ = 0.80

# Set to an integer to only run the first N generated compositions (useful
# for a quick smoke test). Set to None to run the full grid.
LIMIT_COMPOSITIONS = 10

# Use BALANCE_COMPONENT for the recommended workflow. All oxide ranges except
# BALANCE_OXIDE are scanned independently. BALANCE_OXIDE is calculated as the
# remainder to give exactly 100 wt%. Its range is still enforced.
#
# Use FULL_GRID only when every listed range and increment deliberately gives
# combinations that sum to exactly 100 wt%. This mode can easily yield no rows.
COMPOSITION_MODE = "BALANCE_COMPONENT"
BALANCE_OXIDE = "SiO2"

# Format: "oxide": (minimum_wt_pct, maximum_wt_pct, increment_wt_pct)
# Keep the initial grid small. Calculation time rises rapidly with every added
# component value because every composition needs a complete MELTS T path.
OXIDE_RANGES = {
    "CaO": (15.0, 30.0, 5.0),
    "SiO2": (45.0, 80.0, 5.0),
    "Al2O3": (0.0, 10.0, 5.0),
    "Fe2O3": (0.0, 5.0, 2.5),
    "FeO": (0.0, 0.0, 1.0),
    "MgO": (0.0, 5.0, 2.5),
}

# Any oxide accepted by MELTS that is not listed in OXIDE_RANGES defaults to
# zero wt% for every composition (handled by .get() in create_melts_bulk).
ALLOWED_OXIDES = {
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "H2O", "CO2",
}

T_MIN_C = 1000.0
T_MAX_C = 1800.0
T_STEP_C = 25.0
TIMEOUT_SECONDS = 300

# T99 is the practical liquidus for deposit and refractory risk screening.
PRACTICAL_LIQUIDUS_FRACTION = 0.99
LIQUID_FRACTION_MILESTONES = (0.10, 0.50, 0.90, 0.99)
FIRST_LIQUID_FRACTION_THRESHOLD = 1.0e-4
REPORTING_PHASE_FRACTION_MIN = 1.0e-4

# MELTS refuses a bulk composition with exactly zero total iron, and issues a
# blocking warning (raised as an exception by petthermotools) when H2O is
# exactly zero. These trace amounts (wt%, added before normalisation) are
# injected only when the composition's calculated value is zero.
MIN_FEOT_WT_PCT = 0.01
MIN_H2O_WT_PCT = 0.05

# Phases excluded from the liquid fraction denominator and from the phase
# assemblage listing (vapour is not part of the silicate melt assessment).
EXCLUDED_PHASES = {"fluid1", "fluid2", "water1"}

# Run metadata columns that must never be mistaken for a phase amount column.
METADATA_COLUMNS = {
    "t_c", "p_bar", "pressure", "t_liq_c", "fe3fet_liq", "h", "s", "v",
    "dvdp", "dvdt", "time", "index", "mass", "logfo2", "delta_nnO",
    "delta_nno", "delta_fmq", "fo2_buffer", "fo2_offset", "h2o_melt",
    "co2_melt", "viscosity", "density",
}

OXIDE_PREFIXES = (
    "SiO2", "TiO2", "Al2O3", "Cr2O3", "Fe2O3", "FeOt", "FeO", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "H2O", "CO2",
)

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

# Results are sorted by this field for the terminal risk summary.
TOP_RESULTS_TO_PRINT = 15

OUTPUT_FILE = Path("melts_composition_melting_scan_results.csv")

PHASE_LABELS = {
    "liquid1": "LIQUID [silicate melt]",
    "clinopyroxene1": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "clinopyroxene2": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "orthopyroxene1": "Orthopyroxene, (Mg,Fe)SiO3 type",
    "olivine1": "Olivine, (Mg,Fe)2SiO4 type",
    "plagioclase1": "Plagioclase feldspar, NaAlSi3O8 to CaAl2Si2O8",
    "alkali feldspar1": "Alkali feldspar, KAlSi3O8 to NaAlSi3O8",
    "quartz1": "Quartz, SiO2",
    "tridymite1": "Tridymite, SiO2",
    "cristobalite1": "Cristobalite, SiO2",
    "spinel1": "Spinel type oxide",
    "rhm oxide1": "Rhombohedral oxide, hematite ilmenite type",
    "ilmenite1": "Ilmenite type oxide",
    "garnet1": "Garnet solid solution",
    "melilite1": "Melilite solid solution",
    "nepheline1": "Nepheline, NaAlSiO4 type",
    "leucite1": "Leucite, KAlSi2O6",
    "kalsilite1": "Kalsilite, KAlSiO4",
    "whitlockite1": "Whitlockite, Ca3(PO4)2 type",
    "apatite1": "Apatite, Ca5(PO4)3(OH,F,Cl) type",
}


def inclusive_range(start, stop, step):
    if step <= 0:
        raise ValueError("Each composition increment must be greater than zero.")
    if stop < start:
        raise ValueError("Each composition maximum must not be below its minimum.")
    count = int(math.floor((stop - start) / step + 1.0e-10))
    return [round(start + index * step, 10) for index in range(count + 1)]


def generate_compositions():
    unknown = set(OXIDE_RANGES) - ALLOWED_OXIDES
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


def convert_fe_oxides_to_feot(composition):
    return composition.get("FeO", 0.0) + composition.get("Fe2O3", 0.0) * (
        2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3
    )


def create_melts_bulk(composition):
    feot = convert_fe_oxides_to_feot(composition)
    fe_correction_applied = feot <= 0.0
    if fe_correction_applied:
        feot = MIN_FEOT_WT_PCT

    h2o = composition.get("H2O", 0.0)
    h2o_correction_applied = h2o <= 0.0
    if h2o_correction_applied:
        h2o = MIN_H2O_WT_PCT

    bulk = {
        "SiO2_Liq": composition.get("SiO2", 0.0),
        "TiO2_Liq": composition.get("TiO2", 0.0),
        "Al2O3_Liq": composition.get("Al2O3", 0.0),
        "FeOt_Liq": feot,
        "MnO_Liq": composition.get("MnO", 0.0),
        "MgO_Liq": composition.get("MgO", 0.0),
        "CaO_Liq": composition.get("CaO", 0.0),
        "Na2O_Liq": composition.get("Na2O", 0.0),
        "K2O_Liq": composition.get("K2O", 0.0),
        "P2O5_Liq": composition.get("P2O5", 0.0),
        "H2O_Liq": h2o,
        "CO2_Liq": composition.get("CO2", 0.0),
    }
    total = sum(bulk.values())
    if total <= 0:
        raise ValueError("The entered composition contains no positive oxide values.")
    normalised = {oxide: 100.0 * value / total for oxide, value in bulk.items()}
    return normalised, total, fe_correction_applied, h2o_correction_applied


def display_phase_name(name):
    return PHASE_LABELS.get(name.lower(), f"{name} [MELTS phase, not mapped]")


def format_phases(phase_masses, total_mass):
    if total_mass <= 0:
        return "No phase above reporting threshold"
    values = [
        f"{display_phase_name(name)}: {mass / total_mass:.5f}"
        for name, mass in phase_masses.items()
        if mass / total_mass >= REPORTING_PHASE_FRACTION_MIN
    ]
    return "; ".join(values) if values else "No phase above reporting threshold"


@contextlib.contextmanager
def suppress_melts_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def run_temperature_path(melts_bulk):
    with suppress_melts_output():
        results = ptt.isobaric_crystallisation(
            Model=MELTS_MODEL,
            bulk=melts_bulk,
            P_bar=PRESSURE_BAR,
            Fe3Fet_Liq=FE3_FET_LIQ,
            T_start_C=T_MAX_C,
            T_end_C=T_MIN_C,
            dt_C=T_STEP_C,
            Frac_solid=False,
            Frac_fluid=False,
            find_liquidus=False,
            timeout=TIMEOUT_SECONDS,
        )

    if isinstance(results, dict) and "All" in results:
        table = results["All"]
    else:
        table = results

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise RuntimeError("MELTS returned no usable temperature path for this composition.")
    if "T_C" not in table.columns:
        raise RuntimeError(
            "MELTS output is missing the T_C column. Columns present: " + ", ".join(table.columns)
        )
    return table


def is_composition_column(column_name):
    return any(
        column_name == prefix or column_name.startswith(prefix + "_")
        for prefix in OXIDE_PREFIXES
    )


def identify_phase_amount_columns(table):
    mass_prefixed = [column for column in table.columns if column.lower().startswith("mass_")]
    if mass_prefixed:
        return {column[len("mass_"):]: column for column in mass_prefixed}

    candidates = {}
    for column in table.columns:
        lowered = column.lower()
        if lowered in METADATA_COLUMNS:
            continue
        if is_composition_column(column):
            continue
        if not pd.api.types.is_numeric_dtype(table[column]):
            continue
        candidates[column] = column
    return candidates


def build_scan_rows(table):
    phase_columns = identify_phase_amount_columns(table)
    if not phase_columns:
        raise RuntimeError(
            "Could not identify any phase amount columns. Columns present: "
            + ", ".join(table.columns)
        )

    liquid_key = None
    for phase_key in phase_columns:
        if phase_key.lower() == "liquid1":
            liquid_key = phase_key
            break
    if liquid_key is None:
        for phase_key in phase_columns:
            if "liquid" in phase_key.lower() or phase_key.lower() == "liq":
                liquid_key = phase_key
                break
    if liquid_key is None:
        raise RuntimeError(
            "MELTS output contains no recognisable liquid phase column. "
            "Detected phase-like columns: " + ", ".join(phase_columns)
            + " | Full column list: " + ", ".join(table.columns)
        )

    included_keys = [key for key in phase_columns if key not in EXCLUDED_PHASES]

    ordered = table.sort_values("T_C", ascending=True).reset_index(drop=True)
    scan_rows = []
    for _, row in ordered.iterrows():
        phase_masses = {}
        for key in included_keys:
            value = row[phase_columns[key]]
            if pd.notna(value) and value > 0.0:
                phase_masses[key] = float(value)
        total_mass = sum(phase_masses.values())
        liquid_value = row[phase_columns[liquid_key]]
        liquid_mass = float(liquid_value) if pd.notna(liquid_value) else 0.0
        liquid_fraction = (liquid_mass / total_mass) if total_mass > 0.0 else 0.0
        scan_rows.append((float(row["T_C"]), liquid_fraction, phase_masses, total_mass))
    return scan_rows


def find_first_crossing(scan_rows, threshold):
    for previous, current in zip(scan_rows[:-1], scan_rows[1:]):
        if previous[1] < threshold <= current[1]:
            return previous, current
    return None


def interpolate_crossing_temperature(threshold, lower_row, upper_row):
    t_lo, frac_lo = lower_row[0], lower_row[1]
    t_hi, frac_hi = upper_row[0], upper_row[1]
    if frac_hi == frac_lo:
        return t_hi
    fraction_along = (threshold - frac_lo) / (frac_hi - frac_lo)
    return t_lo + fraction_along * (t_hi - t_lo)


def calculate_composition(oxide_composition):
    melts_bulk, total_before_normalisation, fe_correction_applied, h2o_correction_applied = create_melts_bulk(
        oxide_composition
    )
    table = run_temperature_path(melts_bulk)
    scan_rows = build_scan_rows(table)

    result = {
        "fe_correction_applied": fe_correction_applied,
        "h2o_correction_applied": h2o_correction_applied,
        "fe3fet_liq_used": FE3_FET_LIQ,
        "total_before_normalisation_wt_pct": total_before_normalisation,
    }

    thresholds = {"first_liquid_degC": FIRST_LIQUID_FRACTION_THRESHOLD}
    thresholds.update({f"T{int(round(value * 100)):02d}_degC": value for value in LIQUID_FRACTION_MILESTONES})

    for column, threshold in thresholds.items():
        bracket = find_first_crossing(scan_rows, threshold)
        if bracket is None:
            result[column] = None
            result[column.replace("degC", "phases")] = "Not bracketed in selected range"
        else:
            lower_row, upper_row = bracket
            temperature_c = interpolate_crossing_temperature(threshold, lower_row, upper_row)
            result[column] = temperature_c
            result[column.replace("degC", "liquid_fraction")] = upper_row[1]
            result[column.replace("degC", "phases")] = format_phases(upper_row[2], upper_row[3])

    maximum = max(scan_rows, key=lambda row: row[1])
    result["maximum_liquid_fraction"] = maximum[1]
    result["maximum_liquid_temperature_degC"] = maximum[0]
    result["maximum_liquid_phases"] = format_phases(maximum[2], maximum[3])
    return result


def write_results(output_file, oxide_names, results):
    metric_columns = [
        "status", "message", "fe_correction_applied", "h2o_correction_applied", "fe3fet_liq_used",
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

    print("\nMOST MELT SENSITIVE COMPOSITIONS (MELTS)")
    print("Composition is wt%. T99 is the practical liquidus criterion.")
    header = "  ".join(f"{oxide:>7}" for oxide in oxide_names)
    print(f"{header}   First liquid degC   T50 degC   T99 degC   Fe fix  H2O fix")
    print("=" * (len(header) + 60))
    for row in successful[:TOP_RESULTS_TO_PRINT]:
        values = "  ".join(f"{row.get(oxide, 0.0):7.2f}" for oxide in oxide_names)
        print(
            f"{values}   {row['first_liquid_degC']:17.2f}  "
            f"{row.get('T50_degC') or float('nan'):8.2f}  "
            f"{row.get('T99_degC') or float('nan'):8.2f}  "
            f"{'yes' if row.get('fe_correction_applied') else 'no':>6}  "
            f"{'yes' if row.get('h2o_correction_applied') else 'no':>6}"
        )

    failures = [row for row in results if row["status"] != "OK"]
    print(f"\nSuccessful compositions: {len(successful)}")
    print(f"Failed compositions: {len(failures)}")
    if failures:
        print("Failed rows are retained in the CSV with the error message.")


def main():
    compositions, oxide_names = generate_compositions()
    if LIMIT_COMPOSITIONS is not None:
        compositions = compositions[:LIMIT_COMPOSITIONS]

    print(f"Generated valid compositions (after LIMIT_COMPOSITIONS): {len(compositions)}")
    print(f"MELTS model: {MELTS_MODEL}")
    print(f"Pressure: {PRESSURE_BAR:.4f} bar")
    print(f"Fe3+/FeT (liquid) used for every composition: {FE3_FET_LIQ:.4f}")
    print(f"Temperature range: {T_MIN_C:.0f} to {T_MAX_C:.0f} degC")
    print(f"Temperature increment: {T_STEP_C:.1f} degC")

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
            row["message"] = str(error)
        results.append(row)

    write_results(OUTPUT_FILE, oxide_names, results)
    print_risk_summary(oxide_names, results)
    print(f"\nFull results written to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
