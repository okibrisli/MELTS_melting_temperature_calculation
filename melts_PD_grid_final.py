from pathlib import Path
import csv
from decimal import Decimal
from itertools import product
import multiprocessing
import re
import shutil
import time

import numpy as np
import pandas as pd
import petthermotools as ptt


COMPOSITION_RANGES = {
    "SiO2": (70.0, 77.0, 1.0),
    "TiO2": (0.0, 0.0, 0.0),
    "Al2O3": (5.0, 5.0, 0.0),
    "Fe2O3": (1.0, 1.0, 0.0),
    "FeO": (0.0, 0.0, 0.0),
    "MnO": (0.0, 0.0, 0.0),
    "MgO": (2.0, 2.0, 0.0),
    "CaO": (15.0, 15.0, 0.0),
    "Na2O": (0.0, 0.0, 0.0),
    "K2O": (0.0, 0.0, 0.0),
    "P2O5": (0.0, 0.0, 0.0),
    "H2O": (0.0, 0.0, 0.0),
    "CO2": (0.0, 0.0, 0.0),
}

MELTS_MODEL = "MELTSv1.0.2"
PRESSURE_BAR = 1.0
FE3_FET_LIQ = 0.80

T_MIN_C = 800.0
T_MAX_C = 1200.0
TEMPERATURE_STEP_C = 5.0
PHASE_DIAGRAM_CORES = 1

PATH_H2O_WT_PCT = 0.001
FIRST_LIQUID_FRACTION = 1.0e-4
PRACTICAL_LIQUIDUS_FRACTION = 0.99
FULL_LIQUID_TOLERANCE = 1.0e-6
LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
REPORTING_PHASE_FRACTION_MIN = 1.0e-4
MAX_CALCULATIONS = 500

OUTPUT_DIRECTORY = Path("melts_pd_grid_results")
TBL_OUTPUT_DIRECTORY = Path("old")
SUMMARY_FILE = OUTPUT_DIRECTORY / "melts_pd_grid_summary.csv"
SCAN_FILE = OUTPUT_DIRECTORY / "melts_pd_grid_scan.csv"

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

PHASE_LABELS = {
    "cpx": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "opx": "Orthopyroxene, (Mg,Fe)SiO3 type",
    "ol": "Olivine, (Mg,Fe)2SiO4 type",
    "plag": "Plagioclase feldspar, NaAlSi3O8 to CaAl2Si2O8",
    "qtz": "Quartz, SiO2",
    "grt": "Garnet solid solution",
    "sp": "Spinel type oxide",
    "rhm": "Rhombohedral oxide, hematite ilmenite type",
    "ilm": "Ilmenite type oxide",
}


def make_oxide_values(start, end, step, oxide):
    start = Decimal(str(start))
    end = Decimal(str(end))
    step = Decimal(str(step))

    if start == end:
        return [float(start)]
    if step == 0:
        raise ValueError(f"{oxide} varies, so its step cannot be zero.")
    if (end - start) * step <= 0:
        raise ValueError(f"{oxide} has inconsistent start, end, and step values.")

    count = (end - start) / step
    if count != count.to_integral_value():
        raise ValueError(f"{oxide} range is not exactly divisible by its step.")

    return [float(start + index * step) for index in range(int(count) + 1)]


def make_composition_grid(ranges):
    oxides = list(ranges)
    values = [make_oxide_values(*ranges[oxide], oxide) for oxide in oxides]
    for combination in product(*values):
        yield dict(zip(oxides, combination))


def composition_grid_size(ranges):
    total = 1
    for oxide, values in ranges.items():
        total *= len(make_oxide_values(*values, oxide))
    return total


def temperature_values():
    if T_MAX_C <= T_MIN_C or TEMPERATURE_STEP_C <= 0:
        raise ValueError("Temperature limits and step must be positive and ordered.")

    intervals = (T_MAX_C - T_MIN_C) / TEMPERATURE_STEP_C
    rounded_intervals = round(intervals)
    if not np.isclose(intervals, rounded_intervals, atol=1.0e-9):
        raise ValueError("Temperature range must be exactly divisible by TEMPERATURE_STEP_C.")

    return np.linspace(T_MIN_C, T_MAX_C, int(rounded_intervals) + 1)


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
        raise ValueError("Composition contains no positive oxide values.")
    return {oxide: 100.0 * value / total for oxide, value in bulk.items()}


def normalise_column_name(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def phase_identifier_from_column(column):
    return re.sub(r"^mass[_ ]?g[_ ]?", "", str(column), flags=re.IGNORECASE)


def phase_name(identifier):
    text = str(identifier)
    return PHASE_LABELS.get(text.lower(), text)


def numeric_value(value):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(value) else float(value)


def tbl_snapshot():
    return {path.resolve(): path.stat().st_mtime_ns for path in Path.cwd().glob("*_tbl.txt")}


def move_new_tbl_files(snapshot, start_time_ns, composition_id):
    TBL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for source in Path.cwd().glob("*_tbl.txt"):
        old_time = snapshot.get(source.resolve())
        new_time = source.stat().st_mtime_ns
        if old_time == new_time and new_time < start_time_ns:
            continue
        destination = TBL_OUTPUT_DIRECTORY / f"grid_{composition_id:04d}_{source.name}"
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))


def prepare_scan(result):
    required = {"T_C", "P_bar", "mass_g", "mass_g_Liq"}
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise TypeError("phaseDiagram_calc did not return a nonempty pandas DataFrame.")
    if missing := required - set(result.columns):
        raise KeyError("Missing MELTS columns: " + ", ".join(sorted(missing)))

    scan = result.copy()
    scan["temperature_degC"] = pd.to_numeric(scan["T_C"], errors="coerce")
    scan["pressure_bar"] = pd.to_numeric(scan["P_bar"], errors="coerce")
    scan["total_mass_g"] = pd.to_numeric(scan["mass_g"], errors="coerce")
    scan["liquid_mass_g"] = pd.to_numeric(scan["mass_g_Liq"], errors="coerce").fillna(0.0)

    solid_columns = [
        column for column in scan.columns
        if normalise_column_name(column).startswith("massg")
        and normalise_column_name(column) not in {"massg", "massgliq"}
    ]

    valid_mass = scan["total_mass_g"] > 0.0
    scan["liquid_fraction"] = np.nan
    scan.loc[valid_mass, "liquid_fraction"] = (
        scan.loc[valid_mass, "liquid_mass_g"] / scan.loc[valid_mass, "total_mass_g"]
    )
    scan["liquid_fraction"] = scan["liquid_fraction"].clip(0.0, 1.0)
    scan["solid_fraction"] = (1.0 - scan["liquid_fraction"]).clip(0.0, 1.0)

    scan = scan.dropna(subset=["temperature_degC", "liquid_fraction"])
    scan = scan.sort_values("temperature_degC")
    scan = scan.drop_duplicates(subset="temperature_degC").reset_index(drop=True)
    return scan, solid_columns


def phase_text(row, solid_columns):
    entries = []
    total_mass = float(row["total_mass_g"])
    for column in solid_columns:
        fraction = numeric_value(row[column]) / total_mass
        if fraction >= REPORTING_PHASE_FRACTION_MIN:
            entries.append(f"{phase_name(phase_identifier_from_column(column))}: {fraction:.5f}")
    if float(row["liquid_fraction"]) >= REPORTING_PHASE_FRACTION_MIN:
        entries.append(f"LIQUID [MELTS silicate liquid]: {float(row['liquid_fraction']):.5f}")
    return "; ".join(entries) if entries else "No phase above reporting threshold"


def interpolate_crossing(scan, target):
    temperatures = scan["temperature_degC"].to_numpy(dtype=float)
    fractions = scan["liquid_fraction"].to_numpy(dtype=float)
    for index in range(len(scan) - 1):
        lower_fraction = fractions[index]
        upper_fraction = fractions[index + 1]
        if lower_fraction < target <= upper_fraction:
            position = (target - lower_fraction) / (upper_fraction - lower_fraction)
            return temperatures[index] + position * (temperatures[index + 1] - temperatures[index])
    return None


def phase_text_near_temperature(scan, solid_columns, temperature):
    index = (scan["temperature_degC"] - temperature).abs().idxmin()
    return phase_text(scan.loc[index], solid_columns)


def make_summary(composition_id, composition, scan, solid_columns):
    milestones = {
        f"T{int(round(fraction * 100)):02d}_degC": interpolate_crossing(scan, fraction)
        for fraction in LIQUID_FRACTION_MILESTONES
    }
    first_liquid = interpolate_crossing(scan, FIRST_LIQUID_FRACTION)
    t99 = milestones["T99_degC"]
    full_liquid = interpolate_crossing(scan, 1.0 - FULL_LIQUID_TOLERANCE)

    return {
        "composition_id": composition_id,
        **{oxide: composition.get(oxide, 0.0) for oxide in COMPOSITION_RANGES},
        "raw_oxide_total_wt_pct": sum(composition.values()),
        "first_liquid_boundary_degC": first_liquid,
        "scan_min_liquid_fraction": float(scan["liquid_fraction"].iloc[0]),
        **milestones,
        "practical_liquidus_degC": t99,
        "full_liquid_grid_boundary_degC": full_liquid,
        "scan_max_liquid_fraction": float(scan["liquid_fraction"].max()),
        "first_liquid_phase_assemblage": (
            phase_text_near_temperature(scan, solid_columns, first_liquid)
            if first_liquid is not None else "Not bracketed below scan range"
        ),
        "full_liquid_phase_assemblage": (
            phase_text_near_temperature(scan, solid_columns, full_liquid)
            if full_liquid is not None else "Not bracketed above scan range"
        ),
    }


def make_scan_records(composition_id, composition, scan, solid_columns):
    records = []
    for _, row in scan.iterrows():
        records.append({
            "composition_id": composition_id,
            **{oxide: composition.get(oxide, 0.0) for oxide in COMPOSITION_RANGES},
            "temperature_degC": float(row["temperature_degC"]),
            "pressure_bar": float(row["pressure_bar"]),
            "liquid_fraction": float(row["liquid_fraction"]),
            "solid_fraction": float(row["solid_fraction"]),
            "stable_phases": phase_text(row, solid_columns),
        })
    return records


def print_summary_table(summaries):
    columns = [
        "composition_id", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "first_liquid_boundary_degC", "scan_min_liquid_fraction",
        "T01_degC", "T10_degC", "T25_degC", "T50_degC", "T75_degC",
        "T90_degC", "T95_degC", "T99_degC", "full_liquid_grid_boundary_degC",
    ]
    headers = [
        "Point", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "First liquid", "Min liquid", "T01", "T10", "T25", "T50", "T75",
        "T90", "T95", "T99", "Full liquid",
    ]

    rows = []
    for summary in summaries:
        formatted = []
        for column in columns:
            value = summary.get(column)
            if column == "composition_id":
                formatted.append(str(value))
            elif value is None:
                formatted.append("Not found")
            elif "fraction" in column:
                formatted.append(f"{value:.5f}")
            else:
                formatted.append(f"{value:.2f}")
        rows.append(formatted)

    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    separator = "+" + "+".join("=" * (width + 2) for width in widths) + "+"

    print("\nMELTS PHASE DIAGRAM GRID SUMMARY")
    print(separator)
    print("|" + "|".join(f" {header:^{width}} " for header, width in zip(headers, widths)) + "|")
    print(separator)
    for row in rows:
        print("|" + "|".join(f" {value:>{width}} " for value, width in zip(row, widths)) + "|")
    print(separator)
    print("Composition values are raw input oxide wt%. Temperatures are degC.")
    print("If first liquid is not found, use the reported T25, T50, T75, T90, T95, or T99 values.")


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def failed_summary(composition_id, composition, error):
    return {
        "composition_id": composition_id,
        **{oxide: composition.get(oxide, 0.0) for oxide in COMPOSITION_RANGES},
        "raw_oxide_total_wt_pct": sum(composition.values()),
        "first_liquid_boundary_degC": None,
        "scan_min_liquid_fraction": None,
        **{f"T{int(round(fraction * 100)):02d}_degC": None for fraction in LIQUID_FRACTION_MILESTONES},
        "practical_liquidus_degC": None,
        "full_liquid_grid_boundary_degC": None,
        "scan_max_liquid_fraction": None,
        "first_liquid_phase_assemblage": f"FAILED: {type(error).__name__}: {error}",
        "full_liquid_phase_assemblage": "Calculation failed",
    }


def main():
    compositions = list(make_composition_grid(COMPOSITION_RANGES))
    temperatures = temperature_values()

    if len(compositions) > MAX_CALCULATIONS:
        raise ValueError(f"Grid contains {len(compositions)} points, above MAX_CALCULATIONS.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nMELTS PHASE DIAGRAM COMPOSITION GRID")
    print(f"MELTS model: {MELTS_MODEL}")
    print(f"Pressure: {PRESSURE_BAR:.5f} bar")
    print(f"Fe3+ to total Fe: {FE3_FET_LIQ:.5f}")
    print(f"Composition points: {len(compositions)}")
    print(f"Temperature range: {T_MIN_C:.2f} to {T_MAX_C:.2f} degC")
    print(f"Temperature step: {TEMPERATURE_STEP_C:.2f} degC")
    print(f"Temperature points per composition: {len(temperatures)}")

    summaries = []
    scan_records = []

    for composition_id, composition in enumerate(compositions, start=1):
        print("\n" + "=" * 76)
        print(f"COMPOSITION POINT {composition_id} OF {len(compositions)}")
        print("=" * 76)
        print(" ".join(f"{oxide}={value:.3f}" for oxide, value in composition.items() if value != 0.0))

        calculation_composition = dict(composition)
        calculation_composition["H2O"] = PATH_H2O_WT_PCT
        bulk = create_melts_bulk(calculation_composition)
        snapshot = tbl_snapshot()
        start_time_ns = time.time_ns()

        try:
            result = ptt.phaseDiagram_calc(
                cores=PHASE_DIAGRAM_CORES,
                Model=MELTS_MODEL,
                bulk=bulk,
                T_C=temperatures,
                P_bar=np.array([PRESSURE_BAR], dtype=float),
                Fe3Fet_Liq=FE3_FET_LIQ,
            )
            scan, solid_columns = prepare_scan(result)
            summaries.append(make_summary(composition_id, composition, scan, solid_columns))
            scan_records.extend(make_scan_records(composition_id, composition, scan, solid_columns))
        except Exception as error:
            summaries.append(failed_summary(composition_id, composition, error))
            print(f"Calculation failed: {type(error).__name__}: {error}")
        finally:
            move_new_tbl_files(snapshot, start_time_ns, composition_id)

    print_summary_table(summaries)
    write_csv(SUMMARY_FILE, summaries)
    write_csv(SCAN_FILE, scan_records)

    print("\nCSV files written")
    print(f" {SUMMARY_FILE}")
    print(f" {SCAN_FILE}")
    print(f"New alphaMELTS tbl files, if produced, were moved to {TBL_OUTPUT_DIRECTORY}.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
