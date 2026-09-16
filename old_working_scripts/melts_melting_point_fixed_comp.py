from pathlib import Path
import csv
import multiprocessing
import re

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

PRESSURE_BAR = 1.0
MELTS_MODEL = "MELTSv1.0.2"
FE3_FET_LIQ = 0.80

T_MIN_C = 1000.0
T_MAX_C = 2000.0
PATH_STEP_C = 1.0
REPORT_TEMPERATURES_C = tuple(range(1000, 2001, 100))

PRACTICAL_LIQUIDUS_FRACTION = 0.99
LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
PHASE_MASS_TOLERANCE_G = 1.0e-6
REPORTING_PHASE_FRACTION_MIN = 1.0e-4

PATH_H2O_WT_PCT = 0.001
OUTPUT_DIRECTORY = Path("../melts_melting_assessment")

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


def find_temperature_column(frame):
    for column in frame.columns:
        if normalised_column_name(column) in {"tc", "temperaturec"}:
            return column
    raise KeyError("No temperature column was found in the MELTS path output.")


def find_total_mass_column(frame):
    for column in frame.columns:
        if normalised_column_name(column) == "massg":
            return column
    raise KeyError("No total mass column named mass_g or massg was found.")


def find_phase_mass_columns(frame):
    return [
        column
        for column in frame.columns
        if normalised_column_name(column).startswith("massg")
        and normalised_column_name(column) != "massg"
    ]


def find_liquid_mass_columns(mass_columns):
    return [
        column for column in mass_columns
        if "liq" in normalised_column_name(column)
    ]


def phase_identifier_from_mass_column(column):
    text = str(column)
    text = re.sub(r"^mass[_ ]?g[_ ]?", "", text, flags=re.IGNORECASE)
    return text


def get_all_results(path_result):
    if isinstance(path_result, dict):
        for key in ("All", "all"):
            if key in path_result:
                return path_result[key]

    if isinstance(path_result, pd.DataFrame):
        return path_result

    raise TypeError("Could not obtain the All DataFrame from the MELTS path result.")


def get_first_value(frame, column, default="Not returned"):
    if not isinstance(frame, pd.DataFrame) or frame.empty or column not in frame.columns:
        return default

    value = frame[column].iloc[0]
    return default if pd.isna(value) else value


def format_direct_liquidus(result):
    temperature = get_first_value(result, "T_Liq_C")
    phase = get_first_value(result, "liquidus_phase")

    temperature_text = (
        f"{float(temperature):.2f}"
        if isinstance(temperature, (int, float))
        else str(temperature)
    )
    return temperature_text, phase_name(phase)


def prepare_path_scan(all_results, direct_liquidus_c):
    temperature_column = find_temperature_column(all_results)
    total_mass_column = find_total_mass_column(all_results)
    phase_mass_columns = find_phase_mass_columns(all_results)
    liquid_mass_columns = find_liquid_mass_columns(phase_mass_columns)

    if not liquid_mass_columns:
        raise KeyError("No liquid mass column was found in the MELTS path output.")

    solid_mass_columns = [
        column for column in phase_mass_columns
        if column not in liquid_mass_columns
    ]

    scan = all_results.copy()
    scan["temperature_degC"] = pd.to_numeric(
        scan[temperature_column], errors="coerce"
    )
    scan["total_mass_g"] = pd.to_numeric(
        scan[total_mass_column], errors="coerce"
    )
    scan["liquid_mass_g"] = (
        scan[liquid_mass_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )
    scan["solid_mass_g"] = (
        scan[solid_mass_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )

    valid_total = scan["total_mass_g"] > 0.0
    scan["liquid_fraction"] = 0.0
    scan.loc[valid_total, "liquid_fraction"] = (
        scan.loc[valid_total, "liquid_mass_g"]
        / scan.loc[valid_total, "total_mass_g"]
    )
    scan["solid_fraction"] = 1.0 - scan["liquid_fraction"]
    scan["liquid_fraction"] = scan["liquid_fraction"].clip(0.0, 1.0)
    scan["solid_fraction"] = scan["solid_fraction"].clip(0.0, 1.0)
    scan = scan.dropna(subset=["temperature_degC"]).sort_values(
        "temperature_degC"
    ).reset_index(drop=True)

    if scan.empty:
        raise ValueError("The MELTS path calculation returned no temperature rows.")

    if scan["temperature_degC"].max() < direct_liquidus_c - 2.0 * PATH_STEP_C:
        raise RuntimeError(
            "The path scan does not extend to the direct MELTS liquidus temperature."
        )

    return scan, solid_mass_columns


def phases_at_row(row, solid_mass_columns):
    total_mass = float(row["total_mass_g"])
    if total_mass <= 0.0:
        return "No phase information"

    phase_entries = []
    for column in solid_mass_columns:
        mass = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        mass = 0.0 if pd.isna(mass) else float(mass)
        fraction = mass / total_mass
        if fraction >= REPORTING_PHASE_FRACTION_MIN:
            identifier = phase_identifier_from_mass_column(column)
            phase_entries.append(f"{phase_name(identifier)}: {fraction:.5f}")

    liquid_fraction = float(row["liquid_fraction"])
    if liquid_fraction >= REPORTING_PHASE_FRACTION_MIN:
        phase_entries.append(f"LIQUID [MELTS silicate liquid]: {liquid_fraction:.5f}")

    return "; ".join(phase_entries) if phase_entries else "No phase above reporting threshold"


def evaluate_temperature(scan, solid_mass_columns, temperature_c, direct_liquidus_c):
    temperature_c = float(temperature_c)

    if temperature_c >= direct_liquidus_c:
        return {
            "temperature_c": temperature_c,
            "liquid_fraction": 1.0,
            "solid_fraction": 0.0,
            "phases": "LIQUID [MELTS silicate liquid]: 1.00000",
        }

    index = (scan["temperature_degC"] - temperature_c).abs().idxmin()
    row = scan.loc[index]

    return {
        "temperature_c": float(row["temperature_degC"]),
        "liquid_fraction": float(row["liquid_fraction"]),
        "solid_fraction": float(row["solid_fraction"]),
        "phases": phases_at_row(row, solid_mass_columns),
    }


def interpolate_crossing(scan, threshold):
    temperatures = scan["temperature_degC"].to_numpy(dtype=float)
    fractions = scan["liquid_fraction"].to_numpy(dtype=float)

    for index in range(len(scan) - 1):
        lower_fraction = fractions[index]
        upper_fraction = fractions[index + 1]

        if lower_fraction < threshold <= upper_fraction:
            lower_temperature = temperatures[index]
            upper_temperature = temperatures[index + 1]

            if upper_fraction == lower_fraction:
                return upper_temperature

            fraction_position = (
                (threshold - lower_fraction)
                / (upper_fraction - lower_fraction)
            )
            return lower_temperature + fraction_position * (
                upper_temperature - lower_temperature
            )

    return None


def make_finding(label, temperature_c, scan, solid_mass_columns, direct_liquidus_c):
    if temperature_c is None:
        return {
            "finding": label,
            "temperature_c": None,
            "liquid_fraction": None,
            "solid_fraction": None,
            "phases": "Not bracketed in selected range",
        }

    result = evaluate_temperature(
        scan,
        solid_mass_columns,
        temperature_c,
        direct_liquidus_c,
    )
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
        liquid_fraction = (
            ""
            if finding["liquid_fraction"] is None
            else f"{finding['liquid_fraction']:.6f}"
        )
        solid_fraction = (
            ""
            if finding["solid_fraction"] is None
            else f"{finding['solid_fraction']:.6f}"
        )
        print(
            f"{finding['finding']:<31} {temperature:>18} "
            f"{liquid_fraction:>17} {solid_fraction:>16} {finding['phases']}"
        )


def write_csv_files(findings, scan):
    findings_file = OUTPUT_DIRECTORY / "melts_melting_findings.csv"
    scan_file = OUTPUT_DIRECTORY / "melts_melting_scan.csv"

    with findings_file.open("w", newline="", encoding="utf_8") as file:
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

    scan.to_csv(scan_file, index=False, encoding="utf_8")
    return findings_file, scan_file


def main():
    if not 0.0 < PRACTICAL_LIQUIDUS_FRACTION <= 1.0:
        raise ValueError("PRACTICAL_LIQUIDUS_FRACTION must be greater than zero and at most one.")
    if PATH_STEP_C <= 0.0:
        raise ValueError("PATH_STEP_C must be greater than zero.")
    if T_MIN_C >= T_MAX_C:
        raise ValueError("T_MIN_C must be lower than T_MAX_C.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    path_composition = dict(ANALYSED_OXIDE_WT_PCT)
    path_composition["H2O"] = PATH_H2O_WT_PCT

    dry_bulk, dry_total = create_melts_bulk(ANALYSED_OXIDE_WT_PCT)
    path_bulk, path_total = create_melts_bulk(path_composition)
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
    print(f" Practical liquidus criterion: liquid fraction >= {PRACTICAL_LIQUIDUS_FRACTION:.4f}")
    print(f" Path calculation H2O workaround: {PATH_H2O_WT_PCT:.3f} wt%")
    print(f" Dry bulk total after FeOt conversion: {dry_total:.5f} wt%")
    print(f" Path bulk total after FeOt conversion: {path_total:.5f} wt%")

    print("\nStep 1 of 2: Direct MELTS liquidus search with dry input")
    direct_liquidus = ptt.findLiq_multi(
        Model=MELTS_MODEL,
        bulk=dry_bulk,
        P_bar=PRESSURE_BAR,
        Fe3Fet_Liq=FE3_FET_LIQ,
    )
    if not isinstance(direct_liquidus, pd.DataFrame) or direct_liquidus.empty:
        raise TypeError("findLiq_multi did not return a nonempty pandas DataFrame.")

    direct_liquidus.to_csv(
        OUTPUT_DIRECTORY / "melts_direct_liquidus.csv",
        index=False,
        encoding="utf_8",
    )

    direct_liquidus_c = get_first_value(direct_liquidus, "T_Liq_C")
    if not isinstance(direct_liquidus_c, (int, float)):
        raise TypeError("MELTS did not return a numeric T_Liq_C value.")
    direct_liquidus_c = float(direct_liquidus_c)
    direct_temperature_text, direct_phase_text = format_direct_liquidus(
        direct_liquidus
    )

    print("\nStep 2 of 2: Equilibrium melting scan")
    print("MELTS calculates its liquidus, then follows an equilibrium cooling path to T_MIN_C.")
    print("Temperatures above the direct MELTS liquidus are reported as liquid only.")
    print("Frac_solid=False retains crystals and liquid in mutual equilibrium.")

    path_result = ptt.isobaric_crystallisation(
        Model=MELTS_MODEL,
        bulk=path_bulk,
        find_liquidus=True,
        P_bar=PRESSURE_BAR,
        T_end_C=T_MIN_C,
        dt_C=PATH_STEP_C,
        Fe3Fet_init=FE3_FET_LIQ,
        Frac_solid=False,
        Frac_fluid=False,
    )

    path_all_results = get_all_results(path_result)
    if not isinstance(path_all_results, pd.DataFrame) or path_all_results.empty:
        raise TypeError("isobaric_crystallisation did not return a nonempty All DataFrame.")

    scan, solid_mass_columns = prepare_path_scan(
        path_all_results,
        direct_liquidus_c,
    )

    scan_rows = []
    print("\nCalculating MELTS equilibrium temperature scan")
    for temperature_c in REPORT_TEMPERATURES_C:
        state = evaluate_temperature(
            scan,
            solid_mass_columns,
            temperature_c,
            direct_liquidus_c,
        )
        scan_rows.append(state)
        print(
            f" {state['temperature_c']:8.1f} degC   "
            f"Liquid fraction = {state['liquid_fraction']:.8f}   "
            f"{state['phases']}"
        )

    findings = []

    first_liquid_temperature = interpolate_crossing(
        scan,
        PHASE_MASS_TOLERANCE_G / 100.0,
    )
    findings.append(
        make_finding(
            "First liquid boundary",
            first_liquid_temperature,
            scan,
            solid_mass_columns,
            direct_liquidus_c,
        )
    )

    for milestone in LIQUID_FRACTION_MILESTONES:
        temperature = interpolate_crossing(scan, milestone)
        label = f"T{int(round(milestone * 100)):02d}, liquid fraction"
        findings.append(
            make_finding(
                label,
                temperature,
                scan,
                solid_mass_columns,
                direct_liquidus_c,
            )
        )

    findings.append(
        {
            "finding": "Direct MELTS liquidus",
            "temperature_c": direct_liquidus_c,
            "liquid_fraction": 1.0,
            "solid_fraction": 0.0,
            "phases": f"Primary phase on cooling: {direct_phase_text}",
        }
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
            direct_liquidus_c,
        )
    )

    highest_scan_state = evaluate_temperature(
        scan,
        solid_mass_columns,
        float(scan["temperature_degC"].max()),
        direct_liquidus_c,
    )
    highest_scan_state["finding"] = "Maximum path scan temperature"
    findings.append(highest_scan_state)

    print_key_findings(findings)

    print("\nDIRECT MELTS RESULT")
    print(f" Direct dry liquidus: {direct_temperature_text} degC")
    print(f" Direct primary phase: {direct_phase_text}")

    print("\nNORMALISED DRY BULK SENT TO findLiq_multi")
    for oxide, value in dry_bulk.items():
        print(f" {oxide:<12} : {value:10.5f} wt%")

    print("\nNORMALISED NEAR DRY BULK SENT TO EQUILIBRIUM PATH")
    for oxide, value in path_bulk.items():
        print(f" {oxide:<12} : {value:10.5f} wt%")

    print("\nPhase mass columns detected in the equilibrium path")
    for column in solid_mass_columns:
        identifier = phase_identifier_from_mass_column(column)
        print(f" {column:<30} : {phase_name(identifier)}")

    findings_file, raw_scan_file = write_csv_files(findings, scan)

    scan_report_file = OUTPUT_DIRECTORY / "melts_temperature_report.csv"
    with scan_report_file.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "temperature_degC",
                "liquid_fraction",
                "solid_fraction",
                "stable_phases",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "temperature_degC": f"{row['temperature_c']:.4f}",
                "liquid_fraction": f"{row['liquid_fraction']:.10f}",
                "solid_fraction": f"{row['solid_fraction']:.10f}",
                "stable_phases": row["phases"],
            }
            for row in scan_rows
        )

    print("\nCSV files written")
    print(f" {findings_file}")
    print(f" {raw_scan_file}")
    print(f" {scan_report_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
