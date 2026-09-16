from pathlib import Path
import csv
import multiprocessing
import pickle

import numpy as np
import pandas as pd
import petthermotools as ptt

import melts_melting_point_fixed_comp as base


RUN_EQUILIBRATE_MULTI = True
RUN_PHASE_DIAGRAM_CALC = True

EQUILIBRATE_TEMPERATURES_C = tuple(range(1000, 2001, 100))

PHASE_DIAGRAM_T_MIN_C = 700.0
PHASE_DIAGRAM_T_MAX_C = 1200.0
PHASE_DIAGRAM_T_NUM = 101
PHASE_DIAGRAM_PRESSURE_BAR = base.PRESSURE_BAR
PHASE_DIAGRAM_CORES = 1

OUTPUT_DIRECTORY = Path("../melts_optional_diagnostics")


def get_all_results(result):
    if isinstance(result, dict):
        for key in ("All", "all"):
            if key in result:
                return result[key]

    if isinstance(result, pd.DataFrame):
        return result

    raise TypeError("No All pandas DataFrame was returned by equilibrate_multi.")


def prepare_near_dry_bulk():
    composition = dict(base.ANALYSED_OXIDE_WT_PCT)
    composition["H2O"] = base.PATH_H2O_WT_PCT
    return base.create_melts_bulk(composition)[0]


def numeric_value(value):
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(result) else float(result)


def phase_mass_columns(frame):
    return [
        column
        for column in frame.columns
        if str(column).lower().startswith("mass_g_")
        and str(column).lower() != "mass_g_liq"
    ]


def stable_phase_text(row, total_mass, solid_columns, liquid_fraction):
    phase_entries = []

    for column in solid_columns:
        phase_fraction = numeric_value(row[column]) / total_mass

        if phase_fraction >= base.REPORTING_PHASE_FRACTION_MIN:
            identifier = base.phase_identifier_from_mass_column(column)
            phase_entries.append(
                f"{base.phase_name(identifier)}: {phase_fraction:.5f}"
            )

    if liquid_fraction >= base.REPORTING_PHASE_FRACTION_MIN:
        phase_entries.append(
            f"LIQUID [MELTS silicate liquid]: {liquid_fraction:.5f}"
        )

    return "; ".join(phase_entries) if phase_entries else "No phase above reporting threshold"


def summarise_equilibrium_row(frame, requested_temperature_c):
    total_mass_column = base.find_total_mass_column(frame)
    all_phase_mass_columns = base.find_phase_mass_columns(frame)
    liquid_mass_columns = base.find_liquid_mass_columns(all_phase_mass_columns)
    solid_columns = [
        column
        for column in all_phase_mass_columns
        if column not in liquid_mass_columns
    ]

    row = frame.iloc[0]
    total_mass = numeric_value(row[total_mass_column])

    if total_mass <= 0.0:
        raise ValueError("equilibrate_multi returned a nonpositive total mass.")

    liquid_mass = sum(numeric_value(row[column]) for column in liquid_mass_columns)
    liquid_fraction = min(max(liquid_mass / total_mass, 0.0), 1.0)

    return {
        "temperature_degC": requested_temperature_c,
        "liquid_fraction": liquid_fraction,
        "solid_fraction": 1.0 - liquid_fraction,
        "stable_phases": stable_phase_text(
            row,
            total_mass,
            solid_columns,
            liquid_fraction,
        ),
    }


def run_equilibrate_multi(near_dry_bulk):
    print("\nEQUILIBRATE_MULTI DIAGNOSTIC")
    print("Each temperature is solved independently at fixed bulk composition.")
    print("This differs from isobaric_crystallisation, which follows one continuous cooling path.")

    records = []
    raw_frames = []

    for temperature_c in EQUILIBRATE_TEMPERATURES_C:
        print(f" Solving equilibrium at {temperature_c:.1f} degC")

        result = ptt.equilibrate_multi(
            Model=base.MELTS_MODEL,
            bulk=near_dry_bulk,
            T_C=float(temperature_c),
            P_bar=base.PRESSURE_BAR,
            Fe3Fet_init=base.FE3_FET_LIQ,
        )

        all_results = get_all_results(result).copy()
        all_results["requested_temperature_degC"] = temperature_c
        raw_frames.append(all_results)

        record = summarise_equilibrium_row(all_results, temperature_c)
        records.append(record)

        print(
            f" {temperature_c:8.1f} degC   "
            f"Liquid fraction = {record['liquid_fraction']:.8f}   "
            f"{record['stable_phases']}"
        )

    raw_output = pd.concat(raw_frames, ignore_index=True)
    raw_file = OUTPUT_DIRECTORY / "equilibrate_multi_raw.csv"
    report_file = OUTPUT_DIRECTORY / "equilibrate_multi_report.csv"

    raw_output.to_csv(raw_file, index=False, encoding="utf_8")

    with report_file.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print("\nEquilibrate multi CSV files written")
    print(f" {raw_file}")
    print(f" {report_file}")


def run_phase_diagram_calc(near_dry_bulk):
    print("\nPHASEDIAGRAM_CALC DIAGNOSTIC")
    print("This constructs a constant pressure temperature phase map.")
    print(
        f" Temperature range: {PHASE_DIAGRAM_T_MIN_C:.1f} to "
        f"{PHASE_DIAGRAM_T_MAX_C:.1f} degC"
    )
    print(f" Temperature grid points: {PHASE_DIAGRAM_T_NUM}")
    print(f" Pressure: {PHASE_DIAGRAM_PRESSURE_BAR:.5f} bar")
    print(f" CPU cores: {PHASE_DIAGRAM_CORES}")

    temperature_values = np.linspace(
        PHASE_DIAGRAM_T_MIN_C,
        PHASE_DIAGRAM_T_MAX_C,
        int(PHASE_DIAGRAM_T_NUM),
    )

    pressure_values = np.array(
        [PHASE_DIAGRAM_PRESSURE_BAR],
        dtype=float,
    )

    result = ptt.phaseDiagram_calc(
        cores=PHASE_DIAGRAM_CORES,
        Model=base.MELTS_MODEL,
        bulk=near_dry_bulk,
        T_C=temperature_values,
        P_bar=pressure_values,
        Fe3Fet_Liq=base.FE3_FET_LIQ,
    )

    pickle_file = OUTPUT_DIRECTORY / "phaseDiagram_calc_return.pkl"
    with pickle_file.open("wb") as file:
        pickle.dump(result, file)

    print("\nPHASEDIAGRAM_CALC RETURN INFORMATION")
    print(f" Returned Python type: {type(result).__name__}")
    print(f" Raw return object saved to: {pickle_file}")

    if not isinstance(result, pd.DataFrame):
        print("phaseDiagram_calc did not return a pandas DataFrame.")
        print("Inspect the saved pickle object for this PetThermoTools version.")
        return

    phase_dataframe = result.copy()
    raw_file = OUTPUT_DIRECTORY / "phaseDiagram_calc_raw.csv"
    phase_dataframe.to_csv(raw_file, index=False, encoding="utf_8")

    required_columns = {"T_C", "P_bar", "mass_g", "mass_g_Liq"}
    missing_columns = required_columns - set(phase_dataframe.columns)

    if missing_columns:
        print("phaseDiagram_calc returned a DataFrame but required columns are missing.")
        print("Missing columns: " + ", ".join(sorted(missing_columns)))
        print(f"Raw DataFrame saved to: {raw_file}")
        return

    solid_columns = phase_mass_columns(phase_dataframe)

    phase_dataframe["liquid_fraction"] = (
        pd.to_numeric(phase_dataframe["mass_g_Liq"], errors="coerce")
        / pd.to_numeric(phase_dataframe["mass_g"], errors="coerce")
    ).clip(0.0, 1.0)

    phase_dataframe["solid_fraction"] = (
        1.0 - phase_dataframe["liquid_fraction"]
    ).clip(0.0, 1.0)

    phase_dataframe["stable_phases"] = phase_dataframe.apply(
        lambda row: stable_phase_text(
            row,
            numeric_value(row["mass_g"]),
            solid_columns,
            numeric_value(row["liquid_fraction"]),
        ),
        axis=1,
    )

    phase_dataframe = phase_dataframe.sort_values(
        ["P_bar", "T_C"]
    ).reset_index(drop=True)

    print(" phaseDiagram_calc returned a pandas DataFrame.")
    print(f" DataFrame shape: {phase_dataframe.shape}")
    print(f" Raw DataFrame saved to: {raw_file}")

    print("\nPHASEDIAGRAM_CALC TEMPERATURE REPORT")
    for _, row in phase_dataframe.iterrows():
        print(
            f" {row['T_C']:8.1f} degC   "
            f"Liquid fraction = {row['liquid_fraction']:.8f}   "
            f"{row['stable_phases']}"
        )

    report_file = OUTPUT_DIRECTORY / "phaseDiagram_calc_temperature_report.csv"
    phase_dataframe[
        [
            "T_C",
            "P_bar",
            "liquid_fraction",
            "solid_fraction",
            "stable_phases",
        ]
    ].to_csv(report_file, index=False, encoding="utf_8")

    print("\nPhase diagram CSV files written")
    print(f" {raw_file}")
    print(f" {report_file}")


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    near_dry_bulk = prepare_near_dry_bulk()

    print("MELTS OPTIONAL DIAGNOSTICS")
    print(f"Model: {base.MELTS_MODEL}")
    print(f"Pressure: {base.PRESSURE_BAR:.5f} bar")
    print(f"Fe3+ to total Fe: {base.FE3_FET_LIQ:.5f}")
    print(f"H2O used in these diagnostics: {base.PATH_H2O_WT_PCT:.3f} wt%")

    if RUN_EQUILIBRATE_MULTI:
        run_equilibrate_multi(near_dry_bulk)

    if RUN_PHASE_DIAGRAM_CALC:
        run_phase_diagram_calc(near_dry_bulk)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
