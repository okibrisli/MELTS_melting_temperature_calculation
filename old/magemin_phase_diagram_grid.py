from pathlib import Path
import csv
from decimal import Decimal
from itertools import product
import multiprocessing
import re

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

MAGEMIN_MODEL = "Weller2024"
PRESSURE_BAR = 1.0
FE3_FET_LIQ = 0.80

T_MIN_C = 800.0
T_MAX_C = 1200.0
TEMPERATURE_STEP_C = 5.0
MAGEMIN_CORES = 1

FIRST_LIQUID_FRACTION = 1.0e-4
PRACTICAL_LIQUIDUS_FRACTION = 0.99
FULL_LIQUID_TOLERANCE = 1.0e-6
LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
REPORTING_PHASE_FRACTION_MIN = 1.0e-4
MAX_CALCULATIONS = 500

OUTPUT_DIRECTORY = Path("../magemin_pd_grid_results")
SUMMARY_FILE = OUTPUT_DIRECTORY / "magemin_pd_grid_summary.csv"
SCAN_FILE = OUTPUT_DIRECTORY / "magemin_pd_grid_scan.csv"

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688


def oxide_values(start, end, step, oxide):
    start = Decimal(str(start))
    end = Decimal(str(end))
    step = Decimal(str(step))

    if start == end:
        return [float(start)]
    if step == 0 or (end - start) * step <= 0:
        raise ValueError(f"Invalid start, end, or step for {oxide}.")

    count = (end - start) / step
    if count != count.to_integral_value():
        raise ValueError(f"Range for {oxide} is not exactly divisible by its step.")

    return [float(start + index * step) for index in range(int(count) + 1)]


def composition_grid():
    oxides = list(COMPOSITION_RANGES)
    values = [oxide_values(*COMPOSITION_RANGES[oxide], oxide) for oxide in oxides]
    for combination in product(*values):
        yield dict(zip(oxides, combination))


def temperatures():
    if T_MAX_C <= T_MIN_C or TEMPERATURE_STEP_C <= 0.0:
        raise ValueError("Temperature limits and step must be positive and ordered.")

    intervals = (T_MAX_C - T_MIN_C) / TEMPERATURE_STEP_C
    rounded = round(intervals)
    if not np.isclose(intervals, rounded, atol=1.0e-9):
        raise ValueError("Temperature range must be divisible by TEMPERATURE_STEP_C.")

    return np.linspace(T_MIN_C, T_MAX_C, int(rounded) + 1)


def feot(composition):
    return composition.get("FeO", 0.0) + composition.get("Fe2O3", 0.0) * (
        2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3
    )


def magemin_bulk(composition):
    bulk = {
        "SiO2_Liq": composition.get("SiO2", 0.0),
        "Al2O3_Liq": composition.get("Al2O3", 0.0),
        "CaO_Liq": composition.get("CaO", 0.0),
        "MgO_Liq": composition.get("MgO", 0.0),
        "FeOt_Liq": feot(composition),
        "K2O_Liq": composition.get("K2O", 0.0),
        "Na2O_Liq": composition.get("Na2O", 0.0),
        "TiO2_Liq": composition.get("TiO2", 0.0),
        "Cr2O3_Liq": 0.0,
    }

    total = sum(bulk.values())
    if total <= 0.0:
        raise ValueError("Composition contains no positive oxide values.")

    bulk = {name: 100.0 * value / total for name, value in bulk.items()}
    bulk["Fe3Fet_Liq"] = FE3_FET_LIQ
    bulk["O"] = FE3_FET_LIQ * (
        ((159.59 / 2.0) / MOLAR_MASS_FEO) * bulk["FeOt_Liq"]
        - bulk["FeOt_Liq"]
    )
    return bulk


def clean_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def numeric(value):
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(value) else float(value)


def phase_id(column):
    return re.sub(r"^mass[_ ]?g[_ ]?", "", str(column), flags=re.IGNORECASE)


def phase_label(identifier):
    aliases = {
        "liq": "LIQUID [MAGEMin melt solution]",
        "liquid": "LIQUID [MAGEMin melt solution]",
        "cpx": "Clinopyroxene",
        "opx": "Orthopyroxene",
        "plag": "Plagioclase feldspar",
        "qtz": "Quartz",
        "grt": "Garnet",
        "ol": "Olivine",
        "sp": "Spinel",
    }
    return aliases.get(str(identifier).lower(), str(identifier))


def result_dataframe(result):
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, dict):
        for key in ("All", "all", "Combined", "combined"):
            if key in result and isinstance(result[key], pd.DataFrame):
                return result[key].copy()
    raise TypeError("MAGEMin did not return a supported pandas DataFrame.")


def find_column(frame, names):
    for column in frame.columns:
        if clean_name(column) in names:
            return column
    return None


def prepare_scan(result):
    scan = result_dataframe(result)
    temperature_column = find_column(scan, {"tc", "temperaturec"})
    pressure_column = find_column(scan, {"pbar", "pressurebar"})
    total_mass_column = find_column(scan, {"massg", "mass"})

    phase_mass_columns = [
        column for column in scan.columns
        if clean_name(column).startswith("massg") and clean_name(column) != "massg"
    ]
    liquid_mass_columns = [
        column for column in phase_mass_columns
        if "liq" in clean_name(column) or "liquid" in clean_name(column)
    ]
    solid_columns = [column for column in phase_mass_columns if column not in liquid_mass_columns]

    if temperature_column is None or pressure_column is None or total_mass_column is None:
        raise KeyError("MAGEMin output lacks recognizable T, P, or total mass columns.")
    if not liquid_mass_columns:
        raise KeyError("MAGEMin output lacks a recognizable liquid mass column.")

    scan["temperature_degC"] = pd.to_numeric(scan[temperature_column], errors="coerce")
    scan["pressure_bar"] = pd.to_numeric(scan[pressure_column], errors="coerce")
    scan["total_mass_g"] = pd.to_numeric(scan[total_mass_column], errors="coerce")
    scan["liquid_mass_g"] = scan[liquid_mass_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)

    valid_mass = scan["total_mass_g"] > 0.0
    scan["liquid_fraction"] = np.nan
    scan.loc[valid_mass, "liquid_fraction"] = scan.loc[valid_mass, "liquid_mass_g"] / scan.loc[valid_mass, "total_mass_g"]
    scan["liquid_fraction"] = scan["liquid_fraction"].clip(0.0, 1.0)
    scan["solid_fraction"] = (1.0 - scan["liquid_fraction"]).clip(0.0, 1.0)

    scan = scan.dropna(subset=["temperature_degC", "pressure_bar", "liquid_fraction"])
    scan = scan.sort_values("temperature_degC").drop_duplicates("temperature_degC").reset_index(drop=True)
    if scan.empty:
        raise ValueError("MAGEMin returned no valid temperature rows.")
    return scan, solid_columns


def phase_text(row, solid_columns):
    total_mass = float(row["total_mass_g"])
    phases = []
    for column in solid_columns:
        fraction = numeric(row[column]) / total_mass
        if fraction >= REPORTING_PHASE_FRACTION_MIN:
            phases.append(f"{phase_label(phase_id(column))}: {fraction:.5f}")
    if float(row["liquid_fraction"]) >= REPORTING_PHASE_FRACTION_MIN:
        phases.append(f"LIQUID [MAGEMin melt solution]: {float(row['liquid_fraction']):.5f}")
    return "; ".join(phases) if phases else "No phase above reporting threshold"


def crossing(scan, target):
    temperatures_array = scan["temperature_degC"].to_numpy(dtype=float)
    fractions = scan["liquid_fraction"].to_numpy(dtype=float)
    for index in range(len(scan) - 1):
        low = fractions[index]
        high = fractions[index + 1]
        if low < target <= high:
            part = (target - low) / (high - low)
            return temperatures_array[index] + part * (temperatures_array[index + 1] - temperatures_array[index])
    return None


def phase_near(scan, solid_columns, temperature):
    index = (scan["temperature_degC"] - temperature).abs().idxmin()
    return phase_text(scan.loc[index], solid_columns)


def make_summary(composition_id, composition, scan, solid_columns):
    milestone_data = {
        f"T{int(round(fraction * 100)):02d}_degC": crossing(scan, fraction)
        for fraction in LIQUID_FRACTION_MILESTONES
    }
    first_liquid = crossing(scan, FIRST_LIQUID_FRACTION)
    full_liquid = crossing(scan, 1.0 - FULL_LIQUID_TOLERANCE)

    return {
        "composition_id": composition_id,
        **{oxide: composition[oxide] for oxide in COMPOSITION_RANGES},
        "raw_oxide_total_wt_pct": sum(composition.values()),
        "first_liquid_boundary_degC": first_liquid,
        "scan_min_liquid_fraction": float(scan["liquid_fraction"].iloc[0]),
        **milestone_data,
        "practical_liquidus_degC": milestone_data["T99_degC"],
        "full_liquid_grid_boundary_degC": full_liquid,
        "scan_max_liquid_fraction": float(scan["liquid_fraction"].max()),
        "first_liquid_phase_assemblage": phase_near(scan, solid_columns, first_liquid) if first_liquid is not None else "Not bracketed below scan range",
        "full_liquid_phase_assemblage": phase_near(scan, solid_columns, full_liquid) if full_liquid is not None else "Not bracketed above scan range",
    }


def scan_records(composition_id, composition, scan, solid_columns):
    records = []
    for _, row in scan.iterrows():
        records.append({
            "composition_id": composition_id,
            **{oxide: composition[oxide] for oxide in COMPOSITION_RANGES},
            "temperature_degC": float(row["temperature_degC"]),
            "pressure_bar": float(row["pressure_bar"]),
            "liquid_fraction": float(row["liquid_fraction"]),
            "solid_fraction": float(row["solid_fraction"]),
            "stable_phases": phase_text(row, solid_columns),
        })
    return records


def failed_summary(composition_id, composition, error):
    return {
        "composition_id": composition_id,
        **{oxide: composition[oxide] for oxide in COMPOSITION_RANGES},
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


def print_summary(summaries):
    fields = [
        "composition_id", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "first_liquid_boundary_degC", "scan_min_liquid_fraction", "T01_degC",
        "T10_degC", "T25_degC", "T50_degC", "T75_degC", "T90_degC",
        "T95_degC", "T99_degC", "full_liquid_grid_boundary_degC",
    ]
    labels = [
        "Point", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO", "First liquid",
        "Min liquid", "T01", "T10", "T25", "T50", "T75", "T90", "T95",
        "T99", "Full liquid",
    ]
    rows = []
    for summary in summaries:
        row = []
        for field in fields:
            value = summary[field]
            if field == "composition_id":
                row.append(str(value))
            elif value is None:
                row.append("Not found")
            elif "fraction" in field:
                row.append(f"{value:.5f}")
            else:
                row.append(f"{value:.2f}")
        rows.append(row)

    widths = [max(len(label), *(len(row[index]) for row in rows)) for index, label in enumerate(labels)]
    separator = "+" + "+".join("=" * (width + 2) for width in widths) + "+"

    print("\nMAGEMIN PHASE DIAGRAM GRID SUMMARY")
    print(separator)
    print("|" + "|".join(f" {label:^{width}} " for label, width in zip(labels, widths)) + "|")
    print(separator)
    for row in rows:
        print("|" + "|".join(f" {value:>{width}} " for value, width in zip(row, widths)) + "|")
    print(separator)


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    print("\nActivating PetThermoTools MAGEMin environment")
    try:
        ptt.activate_petthermotools_env()
    except ModuleNotFoundError as error:
        if error.name == "juliacall":
            raise ModuleNotFoundError(
                "juliacall is not installed in this virtual environment. Run: "
                "python -m pip install juliacall"
            ) from error
        raise

    compositions = list(composition_grid())
    temperature_grid = temperatures()
    if len(compositions) > MAX_CALCULATIONS:
        raise ValueError(f"Grid contains {len(compositions)} points, above MAX_CALCULATIONS.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nMAGEMIN PHASE DIAGRAM COMPOSITION GRID")
    print(f"MAGEMin model: {MAGEMIN_MODEL}")
    print(f"Pressure: {PRESSURE_BAR:.5f} bar")
    print(f"Fe3+ to total Fe: {FE3_FET_LIQ:.5f}")
    print(f"Composition points: {len(compositions)}")
    print(f"Temperature range: {T_MIN_C:.2f} to {T_MAX_C:.2f} degC")
    print(f"Temperature step: {TEMPERATURE_STEP_C:.2f} degC")
    print(f"Temperature points per composition: {len(temperature_grid)}")
    print(f"MAGEMin calculation cores: {MAGEMIN_CORES}")

    summaries = []
    all_scan_records = []

    for composition_id, composition in enumerate(compositions, start=1):
        print("\n" + "=" * 76)
        print(f"COMPOSITION POINT {composition_id} OF {len(compositions)}")
        print("=" * 76)
        print(" ".join(f"{oxide}={value:.3f}" for oxide, value in composition.items() if value != 0.0))

        try:
            result = ptt.phaseDiagram_calc(
                cores=MAGEMIN_CORES,
                Model=MAGEMIN_MODEL,
                bulk=magemin_bulk(composition),
                T_C=temperature_grid,
                P_bar=np.array([PRESSURE_BAR], dtype=float),
                Fe3Fet_Liq=FE3_FET_LIQ,
            )
            scan, solid_columns = prepare_scan(result)
            summaries.append(make_summary(composition_id, composition, scan, solid_columns))
            all_scan_records.extend(scan_records(composition_id, composition, scan, solid_columns))
        except Exception as error:
            summaries.append(failed_summary(composition_id, composition, error))
            print(f"Calculation failed: {type(error).__name__}: {error}")

    print_summary(summaries)
    write_csv(SUMMARY_FILE, summaries)
    write_csv(SCAN_FILE, all_scan_records)

    print("\nCSV files written")
    print(f" {SUMMARY_FILE}")
    print(f" {SCAN_FILE}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
