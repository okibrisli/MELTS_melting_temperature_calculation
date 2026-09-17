"""Add direct findLiq_multi results to melts_csv_batch_calc.py.

Keep this file in the same folder as melts_csv_batch_calc.py, the documented
CSV driven MELTS phase diagram batch script. It preserves that script's input
handling, phase diagram scan, composition audit, settings log, explanatory
notes, and existing CSV reports. It adds a direct MELTS liquidus search for
each composition and extends the summary reporting accordingly.
"""

from pathlib import Path
import csv

import numpy as np
import pandas as pd
import petthermotools as ptt

import melt_PD_csv_comp_final as base


# ADDITIONAL USER SETTINGS
FIND_LIQUIDUS_ENABLED = True
# Run ptt.findLiq_multi for every composition before the phase diagram scan.
# Set False to run only the original phase diagram calculation.

FIND_LIQUIDUS_RAW_FILE = (
    base.OUTPUT_DIRECTORY / "melts_csv_batch_findliq_raw.csv"
)
# Raw unformatted DataFrame returned by ptt.findLiq_multi for every point.


ORIGINAL_PHASE_DIAGRAM_CALC = base.ptt.phaseDiagram_calc
ORIGINAL_MAKE_SUMMARY = base.make_summary
ORIGINAL_FAILED_SUMMARY = base.failed_summary
DIRECT_RESULT_QUEUE = []
DIRECT_RAW_RECORDS = []


def get_dataframe_value(frame, column, default=None):
    if not isinstance(frame, pd.DataFrame) or frame.empty or column not in frame.columns:
        return default
    value = frame[column].iloc[0]
    return default if pd.isna(value) else value


def direct_liquidus_result(model, bulk, pressure_bar, fe3_fet):
    if not FIND_LIQUIDUS_ENABLED:
        return {
            "status": "Disabled",
            "temperature_c": None,
            "phase_raw": None,
            "phase_text": "Direct liquidus search disabled",
            "fluid_saturated": None,
            "error": None,
            "raw": None,
        }

    try:
        result = ptt.findLiq_multi(
            Model=model,
            bulk=bulk,
            P_bar=pressure_bar,
            Fe3Fet_Liq=fe3_fet,
        )

        if not isinstance(result, pd.DataFrame) or result.empty:
            raise TypeError("findLiq_multi did not return a nonempty pandas DataFrame.")

        temperature = get_dataframe_value(result, "T_Liq_C")
        phase_raw = get_dataframe_value(result, "liquidus_phase")
        fluid_saturated = get_dataframe_value(result, "fluid_saturated")

        phase_text = (
            base.phase_name(phase_raw)
            if phase_raw is not None else "Not returned by MELTS"
        )

        return {
            "status": "Success",
            "temperature_c": float(temperature) if temperature is not None else None,
            "phase_raw": phase_raw,
            "phase_text": phase_text,
            "fluid_saturated": fluid_saturated,
            "error": None,
            "raw": result.copy(),
        }

    except Exception as error:
        return {
            "status": "Failed",
            "temperature_c": None,
            "phase_raw": None,
            "phase_text": "Not available",
            "fluid_saturated": None,
            "error": f"{type(error).__name__}: {error}",
            "raw": None,
        }


def phase_diagram_with_direct_liquidus(*args, **kwargs):
    direct = direct_liquidus_result(
        model=kwargs.get("Model", base.MELTS_MODEL),
        bulk=kwargs.get("bulk"),
        pressure_bar=kwargs.get("P_bar", base.PRESSURE_BAR),
        fe3_fet=kwargs.get("Fe3Fet_Liq", base.FE3_FET_LIQ),
    )
    DIRECT_RESULT_QUEUE.append(direct)
    return ORIGINAL_PHASE_DIAGRAM_CALC(*args, **kwargs)


def add_direct_fields(summary, direct, composition_id):
    summary["findliq_status"] = direct["status"]
    summary["findliq_liquidus_degC"] = direct["temperature_c"]
    summary["findliq_primary_phase"] = direct["phase_text"]
    summary["findliq_phase_identifier"] = direct["phase_raw"]
    summary["findliq_fluid_saturated"] = direct["fluid_saturated"]
    summary["findliq_error"] = direct["error"]

    if direct["raw"] is not None:
        raw = direct["raw"].copy()
        raw.insert(0, "composition_id", composition_id)
        DIRECT_RAW_RECORDS.append(raw)

    return summary


def make_summary_with_direct(composition_id, composition, extras, scan, solid_columns):
    summary = ORIGINAL_MAKE_SUMMARY(
        composition_id,
        composition,
        extras,
        scan,
        solid_columns,
    )
    direct = DIRECT_RESULT_QUEUE.pop(0) if DIRECT_RESULT_QUEUE else {
        "status": "Not run",
        "temperature_c": None,
        "phase_text": "Not run",
        "phase_raw": None,
        "fluid_saturated": None,
        "error": "No direct liquidus result was queued",
        "raw": None,
    }
    return add_direct_fields(summary, direct, composition_id)


def failed_summary_with_direct(composition_id, composition, extras, error):
    summary = ORIGINAL_FAILED_SUMMARY(composition_id, composition, extras, error)
    direct = DIRECT_RESULT_QUEUE.pop(0) if DIRECT_RESULT_QUEUE else {
        "status": "Not run",
        "temperature_c": None,
        "phase_text": "Not run",
        "phase_raw": None,
        "fluid_saturated": None,
        "error": "Phase diagram failed before direct liquidus result was available",
        "raw": None,
    }
    return add_direct_fields(summary, direct, composition_id)


def print_summary_table_with_direct(summaries):
    columns = [
        "composition_id", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "first_liquid_boundary_degC", "T25_degC", "T50_degC", "T75_degC",
        "T90_degC", "T95_degC", "T99_degC",
        "findliq_liquidus_degC", "findliq_status",
        "full_liquid_grid_boundary_degC",
    ]
    headers = [
        "Point", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "First liquid", "T25", "T50", "T75", "T90", "T95", "T99",
        "findLiq", "findLiq status", "Full liquid",
    ]

    rows = []
    for summary in summaries:
        row = []
        for column in columns:
            value = summary.get(column)
            if column == "composition_id":
                row.append(str(value))
            elif column == "findliq_status":
                row.append(str(value))
            elif value is None:
                row.append("Not found")
            else:
                row.append(f"{float(value):.2f}")
        rows.append(row)

    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    separator = "+" + "+".join("=" * (width + 2) for width in widths) + "+"

    print("\nMELTS CSV BATCH SUMMARY WITH DIRECT LIQUIDUS")
    print(separator)
    print("|" + "|".join(f" {header:^{width}} " for header, width in zip(headers, widths)) + "|")
    print(separator)
    for row in rows:
        print("|" + "|".join(f" {value:>{width}} " for value, width in zip(row, widths)) + "|")
    print(separator)
    print("findLiq is the direct MELTS liquidus: first solid phase saturated on cooling.")
    print("Full liquid is the phase diagram scan based near 100% liquid boundary.")


def write_direct_raw_csv():
    if not DIRECT_RAW_RECORDS:
        return
    combined = pd.concat(DIRECT_RAW_RECORDS, ignore_index=True)
    combined.to_csv(FIND_LIQUIDUS_RAW_FILE, index=False, encoding="utf_8")


def main():
    base.ptt.phaseDiagram_calc = phase_diagram_with_direct_liquidus
    base.make_summary = make_summary_with_direct
    base.failed_summary = failed_summary_with_direct
    base.print_summary_table = print_summary_table_with_direct

    base.main()
    write_direct_raw_csv()

    print("\nAdditional direct liquidus CSV written")
    print(f" {FIND_LIQUIDUS_RAW_FILE}")


if __name__ == "__main__":
    main()
