from pathlib import Path
import csv
import multiprocessing
import re
import shutil
import time

import numpy as np
import pandas as pd
import petthermotools as ptt


ANALYSED_OXIDE_WT_PCT = {
    "SiO2": 77.0,
    "TiO2": 0.0,
    "Al2O3": 5.0,
    "Fe2O3": 1.0,
    "FeO": 0.0,
    "MnO": 0.0,
    "MgO": 2.0,
    "CaO": 15.0,
    "Na2O": 0.0,
    "K2O": 0.0,
    "P2O5": 0.0,
    "H2O": 0.0,
    "CO2": 0.0,
}

MELTS_MODEL = "MELTSv1.0.2"
PRESSURE_BAR = 1.0
FE3_FET_LIQ = 0.80

T_MIN_C = 600.0
T_MAX_C = 1200.0
TEMPERATURE_STEP_C = 20.0
PHASE_DIAGRAM_CORES = 1

PATH_H2O_WT_PCT = 0.001
PRACTICAL_LIQUIDUS_FRACTION = 0.99
LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
REPORTING_PHASE_FRACTION_MIN = 1.0e-4
FULL_LIQUID_TOLERANCE = 1.0e-6

FINDINGS_OUTPUT_FILE = Path("old_working_scripts/melts_melting_findings.csv")
SCAN_OUTPUT_FILE = Path("old_working_scripts/melts_melting_scan.csv")
TBL_OUTPUT_DIRECTORY = Path("old")

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

PHASE_LABELS = {
    "clinopyroxene1": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "clinopyroxene2": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "cpx": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "orthopyroxene1": "Orthopyroxene, (Mg,Fe)SiO3 type",
    "opx": "Orthopyroxene, (Mg,Fe)SiO3 type",
    "olivine1": "Olivine, (Mg,Fe)2SiO4 type",
    "ol": "Olivine, (Mg,Fe)2SiO4 type",
    "plagioclase1": "Plagioclase feldspar, NaAlSi3O8 to CaAl2Si2O8",
    "plag": "Plagioclase feldspar, NaAlSi3O8 to CaAl2Si2O8",
    "alkali feldspar1": "Alkali feldspar, KAlSi3O8 to NaAlSi3O8",
    "quartz1": "Quartz, SiO2",
    "qtz": "Quartz, SiO2",
    "tridymite1": "Tridymite, SiO2",
    "cristobalite1": "Cristobalite, SiO2",
    "spinel1": "Spinel type oxide",
    "sp": "Spinel type oxide",
    "rhm oxide1": "Rhombohedral oxide, hematite ilmenite type",
    "rhm": "Rhombohedral oxide, hematite ilmenite type",
    "ilmenite1": "Ilmenite type oxide",
    "ilm": "Ilmenite type oxide",
    "garnet1": "Garnet solid solution",
    "grt": "Garnet solid solution",
    "melilite1": "Melilite solid solution",
    "nepheline1": "Nepheline, NaAlSiO4 type",
    "leucite1": "Leucite, KAlSi2O6",
    "kalsilite1": "Kalsilite, KAlSiO4",
}


def convert_fe_oxides_to_feot(composition):
    return composition.get("FeO", 0.0) + composition.get("Fe2O3", 0.0) * (
        2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3
    )


def create_melts_bulk(composition):
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

    total = sum(bulk.values())
    if total <= 0.0:
        raise ValueError("The entered composition contains no positive oxide values.")

    return {oxide: 100.0 * value / total for oxide, value in bulk.items()}, total


def phase_name(raw_name):
    if pd.isna(raw_name):
        return "Not returned"

    text = str(raw_name)
    return PHASE_LABELS.get(text.lower(), text)


def normalised_column_name(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def phase_identifier_from_mass_column(column):
    return re.sub(r"^mass[_ ]?g[_ ]?", "", str(column), flags=re.IGNORECASE)


def numeric_value(value):
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(number) else float(number)


def temperature_values():
    span = T_MAX_C - T_MIN_C

    if span <= 0.0:
        raise ValueError("T_MAX_C must be greater than T_MIN_C.")
    if TEMPERATURE_STEP_C <= 0.0:
        raise ValueError("TEMPERATURE_STEP_C must be greater than zero.")

    number_of_intervals = span / TEMPERATURE_STEP_C
    rounded_intervals = round(number_of_intervals)

    if not np.isclose(number_of_intervals, rounded_intervals, atol=1.0e-9):
        raise ValueError(
            "The selected temperature range is not exactly divisible by "
            "TEMPERATURE_STEP_C. Adjust one of these settings."
        )

    return np.linspace(T_MIN_C, T_MAX_C, int(rounded_intervals) + 1)


def tbl_file_snapshot():
    return {
        path.resolve(): path.stat().st_mtime_ns
        for path in Path.cwd().glob("*_tbl.txt")
    }


def move_new_tbl_files(snapshot, calculation_start_time_ns):
    TBL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    moved_files = []

    for source_file in Path.cwd().glob("*_tbl.txt"):
        source_key = source_file.resolve()
        modified_time = source_file.stat().st_mtime_ns
        existing_time = snapshot.get(source_key)

        if existing_time == modified_time and modified_time < calculation_start_time_ns:
            continue

        destination_file = TBL_OUTPUT_DIRECTORY / source_file.name

        if destination_file.exists():
            destination_file.unlink()

        shutil.move(str(source_file), str(destination_file))
        moved_files.append(destination_file)

    return moved_files


def prepare_scan_dataframe(result):
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise TypeError("phaseDiagram_calc did not return a nonempty pandas DataFrame.")

    required_columns = {"T_C", "P_bar", "mass_g", "mass_g_Liq"}
    missing_columns = required_columns - set(result.columns)

    if missing_columns:
        raise KeyError(
            "The MELTS result does not contain required columns: "
            + ", ".join(sorted(missing_columns))
        )

    scan = result.copy()
    scan["temperature_degC"] = pd.to_numeric(scan["T_C"], errors="coerce")
    scan["pressure_bar"] = pd.to_numeric(scan["P_bar"], errors="coerce")
    scan["total_mass_g"] = pd.to_numeric(scan["mass_g"], errors="coerce")
    scan["liquid_mass_g"] = pd.to_numeric(scan["mass_g_Liq"], errors="coerce").fillna(0.0)

    solid_mass_columns = [
        column
        for column in scan.columns
        if normalised_column_name(column).startswith("massg")
        and normalised_column_name(column) not in {"massg", "massgliq"}
    ]

    if not solid_mass_columns:
        raise KeyError("No crystalline phase mass columns were found in the MELTS output.")

    scan["solid_mass_g"] = (
        scan[solid_mass_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )

    valid_total_mass = scan["total_mass_g"] > 0.0
    scan["liquid_fraction"] = np.nan
    scan.loc[valid_total_mass, "liquid_fraction"] = (
        scan.loc[valid_total_mass, "liquid_mass_g"]
        / scan.loc[valid_total_mass, "total_mass_g"]
    )
    scan["liquid_fraction"] = scan["liquid_fraction"].clip(0.0, 1.0)
    scan["solid_fraction"] = (1.0 - scan["liquid_fraction"]).clip(0.0, 1.0)

    scan = scan.dropna(subset=["temperature_degC", "pressure_bar", "liquid_fraction"])
    scan = scan.sort_values(["pressure_bar", "temperature_degC"])
    scan = scan.drop_duplicates(subset=["pressure_bar", "temperature_degC"])
    scan = scan.reset_index(drop=True)

    if scan.empty:
        raise ValueError("No valid temperature points were returned by phaseDiagram_calc.")

    return scan, solid_mass_columns


def stable_phase_text(row, solid_mass_columns):
    total_mass = float(row["total_mass_g"])
    phase_entries = []

    for column in solid_mass_columns:
        phase_fraction = numeric_value(row[column]) / total_mass

        if phase_fraction >= REPORTING_PHASE_FRACTION_MIN:
            phase_identifier = phase_identifier_from_mass_column(column)
            phase_entries.append(
                f"{phase_name(phase_identifier)}: {phase_fraction:.5f}"
            )

    liquid_fraction = float(row["liquid_fraction"])
    if liquid_fraction >= REPORTING_PHASE_FRACTION_MIN:
        phase_entries.append(
            f"LIQUID [MELTS silicate liquid]: {liquid_fraction:.5f}"
        )

    return "; ".join(phase_entries) if phase_entries else "No phase above reporting threshold"


def interpolate_crossing(scan, threshold):
    temperatures = scan["temperature_degC"].to_numpy(dtype=float)
    fractions = scan["liquid_fraction"].to_numpy(dtype=float)

    for index in range(len(scan) - 1):
        lower_fraction = fractions[index]
        upper_fraction = fractions[index + 1]

        if lower_fraction < threshold <= upper_fraction:
            lower_temperature = temperatures[index]
            upper_temperature = temperatures[index + 1]

            if np.isclose(lower_fraction, upper_fraction):
                return upper_temperature

            position = (threshold - lower_fraction) / (upper_fraction - lower_fraction)
            return lower_temperature + position * (upper_temperature - lower_temperature)

    return None


def state_at_temperature(scan, solid_mass_columns, temperature_c):
    index = (scan["temperature_degC"] - float(temperature_c)).abs().idxmin()
    row = scan.loc[index]

    return {
        "temperature_c": float(temperature_c),
        "liquid_fraction": float(row["liquid_fraction"]),
        "solid_fraction": float(row["solid_fraction"]),
        "phases": stable_phase_text(row, solid_mass_columns),
    }


def make_finding(label, temperature_c, scan, solid_mass_columns):
    if temperature_c is None:
        return {
            "finding": label,
            "temperature_c": None,
            "liquid_fraction": None,
            "solid_fraction": None,
            "phases": "Not bracketed in selected range",
        }

    result = state_at_temperature(scan, solid_mass_columns, temperature_c)
    result["finding"] = label
    return result


def print_key_findings(findings):
    print("\nKEY FINDINGS")
    print(
        f"{'Finding':<31} {'Temperature degC':>18} "
        f"{'Liquid fraction':>17} {'Solid fraction':>16} Stable phases"
    )
    print("=" * 180)

    for finding in findings:
        temperature = (
            "Not found"
            if finding["temperature_c"] is None
            else f"{finding['temperature_c']:.2f}"
        )
        liquid = (
            ""
            if finding["liquid_fraction"] is None
            else f"{finding['liquid_fraction']:.6f}"
        )
        solid = (
            ""
            if finding["solid_fraction"] is None
            else f"{finding['solid_fraction']:.6f}"
        )
        print(
            f"{finding['finding']:<31} {temperature:>18} "
            f"{liquid:>17} {solid:>16} {finding['phases']}"
        )


def write_csv_files(findings, scan, solid_mass_columns):
    with FINDINGS_OUTPUT_FILE.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "finding",
                "temperature_degC",
                "liquid_fraction",
                "solid_fraction",
                "stable_phases",
            ),
        )
        writer.writeheader()

        for finding in findings:
            writer.writerow(
                {
                    "finding": finding["finding"],
                    "temperature_degC": finding["temperature_c"],
                    "liquid_fraction": finding["liquid_fraction"],
                    "solid_fraction": finding["solid_fraction"],
                    "stable_phases": finding["phases"],
                }
            )

    with SCAN_OUTPUT_FILE.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "temperature_degC",
                "pressure_bar",
                "liquid_fraction",
                "solid_fraction",
                "stable_phases",
            ),
        )
        writer.writeheader()

        for _, row in scan.iterrows():
            writer.writerow(
                {
                    "temperature_degC": f"{row['temperature_degC']:.4f}",
                    "pressure_bar": f"{row['pressure_bar']:.6f}",
                    "liquid_fraction": f"{row['liquid_fraction']:.10f}",
                    "solid_fraction": f"{row['solid_fraction']:.10f}",
                    "stable_phases": stable_phase_text(row, solid_mass_columns),
                }
            )


def main():
    temperatures = temperature_values()
    path_composition = dict(ANALYSED_OXIDE_WT_PCT)
    path_composition["H2O"] = PATH_H2O_WT_PCT
    melts_bulk, total_before_normalisation = create_melts_bulk(path_composition)
    feot_before_normalisation = convert_fe_oxides_to_feot(
        ANALYSED_OXIDE_WT_PCT
    )

    print("\nInput oxide composition, wt%")
    for oxide, value in ANALYSED_OXIDE_WT_PCT.items():
        print(f" {oxide:>6s} : {value:10.5f}")
    print(f" Total : {sum(ANALYSED_OXIDE_WT_PCT.values()):10.5f}")
    print(f" Pressure: {PRESSURE_BAR:.5f} bar")
    print(f" MELTS model: {MELTS_MODEL}")
    print(f" Input Fe3+ to total Fe: {FE3_FET_LIQ:.5f}")
    print(f" FeOt before normalisation: {feot_before_normalisation:.5f} wt%")
    print(f" Path calculation H2O workaround: {PATH_H2O_WT_PCT:.3f} wt%")
    print(f" Practical liquidus criterion: liquid fraction >= {PRACTICAL_LIQUIDUS_FRACTION:.4f}")
    print(f" Temperature range: {T_MIN_C:.2f} to {T_MAX_C:.2f} degC")
    print(f" Temperature step: {TEMPERATURE_STEP_C:.2f} degC")
    print(f" Number of temperature points: {len(temperatures)}")
    print(f" CPU cores: {PHASE_DIAGRAM_CORES}")

    tbl_snapshot = tbl_file_snapshot()
    calculation_start_time_ns = time.time_ns()

    print("\nCalculating MELTS phase diagram temperature scan")
    result = ptt.phaseDiagram_calc(
        cores=PHASE_DIAGRAM_CORES,
        Model=MELTS_MODEL,
        bulk=melts_bulk,
        T_C=temperatures,
        P_bar=np.array([PRESSURE_BAR], dtype=float),
        Fe3Fet_Liq=FE3_FET_LIQ,
    )

    moved_tbl_files = move_new_tbl_files(tbl_snapshot, calculation_start_time_ns)
    scan, solid_mass_columns = prepare_scan_dataframe(result)

    print("\nPHASEDIAGRAM_CALC TEMPERATURE REPORT")
    for _, row in scan.iterrows():
        print(
            f" {row['temperature_degC']:8.1f} degC   "
            f"Liquid fraction = {row['liquid_fraction']:.8f}   "
            f"{stable_phase_text(row, solid_mass_columns)}"
        )

    findings = []
    first_liquid_temperature = interpolate_crossing(
        scan,
        REPORTING_PHASE_FRACTION_MIN,
    )
    findings.append(
        make_finding(
            "First liquid boundary",
            first_liquid_temperature,
            scan,
            solid_mass_columns,
        )
    )

    for milestone in LIQUID_FRACTION_MILESTONES:
        temperature = interpolate_crossing(scan, milestone)
        findings.append(
            make_finding(
                f"T{int(round(milestone * 100)):02d}, liquid fraction",
                temperature,
                scan,
                solid_mass_columns,
            )
        )

    practical_temperature = interpolate_crossing(
        scan,
        PRACTICAL_LIQUIDUS_FRACTION,
    )
    findings.append(
        make_finding(
            "Practical liquidus",
            practical_temperature,
            scan,
            solid_mass_columns,
        )
    )

    full_liquid_temperature = interpolate_crossing(
        scan,
        1.0 - FULL_LIQUID_TOLERANCE,
    )
    findings.append(
        make_finding(
            "Full liquid grid boundary",
            full_liquid_temperature,
            scan,
            solid_mass_columns,
        )
    )

    maximum_liquid_row = scan.loc[scan["liquid_fraction"].idxmax()]
    findings.append(
        {
            "finding": "Maximum scan liquid",
            "temperature_c": float(maximum_liquid_row["temperature_degC"]),
            "liquid_fraction": float(maximum_liquid_row["liquid_fraction"]),
            "solid_fraction": float(maximum_liquid_row["solid_fraction"]),
            "phases": stable_phase_text(maximum_liquid_row, solid_mass_columns),
        }
    )

    print_key_findings(findings)
    write_csv_files(findings, scan, solid_mass_columns)

    print("\nCSV files written")
    print(f" {FINDINGS_OUTPUT_FILE}")
    print(f" {SCAN_OUTPUT_FILE}")

    if moved_tbl_files:
        print("\nNew alphaMELTS tbl files moved to")
        print(f" {TBL_OUTPUT_DIRECTORY}")
        for tbl_file in moved_tbl_files:
            print(f" {tbl_file.name}")
    else:
        print("\nNo new alphaMELTS tbl files were found in the current folder.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
