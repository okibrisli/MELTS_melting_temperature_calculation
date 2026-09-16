from pathlib import Path
import multiprocessing

import pandas as pd

import melts_liquidus_single as calculator
from composition_generator import linear_composition_path


START_COMPOSITION = {
    "SiO2": 70.0,
    "TiO2": 0.0,
    "Al2O3": 0.0,
    "Fe2O3": 1.0,
    "FeO": 0.0,
    "MnO": 0.0,
    "MgO": 0.0,
    "CaO": 1.0,
    "Na2O": 0.0,
    "K2O": 0.0,
    "P2O5": 0.0,
    "H2O": 0.0,
    "CO2": 0.0,
}

END_COMPOSITION = {
    "SiO2": 40.0,
    "TiO2": 0.0,
    "Al2O3": 10.0,
    "Fe2O3": 3.0,
    "FeO": 0.0,
    "MnO": 2.0,
    "MgO": 2.0,
    "CaO": 5.0,
    "Na2O": 2.0,
    "K2O": 2.0,
    "P2O5": 2.0,
    "H2O": 0.0,
    "CO2": 0.0,
}

POINT_COUNT = 11
BATCH_OUTPUT_DIRECTORY = Path("../batch_melts_results")
REPORT_OXIDES = ("SiO2", "Al2O3", "Fe2O3", "FeO", "MgO", "CaO")


def validate_composition(composition, composition_number):
    total = sum(float(value) for value in composition.values())

    if total <= 0.0:
        raise ValueError(f"Composition {composition_number} has a nonpositive total")

    negative_oxides = [
        oxide for oxide, value in composition.items() if float(value) < 0.0
    ]
    if negative_oxides:
        names = ", ".join(negative_oxides)
        raise ValueError(f"Composition {composition_number} has negative oxides: {names}")


def read_calculation_summary(composition_number, composition, output_file):
    record = {
        "Point": str(composition_number),
        "Temperature": "Failed",
        "Primary phase": "No result file",
    }

    for oxide in REPORT_OXIDES:
        record[oxide] = f"{float(composition.get(oxide, 0.0)):.3f}"

    if not output_file.exists():
        return record

    try:
        result = pd.read_csv(output_file)

        if result.empty:
            record["Primary phase"] = "Empty result file"
            return record

        first_row = result.iloc[0]
        liquidus_temperature = first_row.get("T_Liq_C")
        liquidus_phase = first_row.get("liquidus_phase")

        if pd.notna(liquidus_temperature):
            record["Temperature"] = f"{float(liquidus_temperature):.2f} degC"

        if pd.notna(liquidus_phase):
            record["Primary phase"] = calculator.phase_name(liquidus_phase)
        else:
            record["Primary phase"] = "Not returned"

    except Exception as error:
        record["Primary phase"] = f"Could not read result: {type(error).__name__}"

    return record


def print_batch_summary(summary_records):
    columns = ["Point", *REPORT_OXIDES, "Temperature", "Primary phase"]

    display_rows = []
    for record in summary_records:
        display_rows.append([str(record.get(column, "")) for column in columns])

    widths = []
    for column_index, column in enumerate(columns):
        content_width = max(
            [len(row[column_index]) for row in display_rows],
            default=0,
        )
        widths.append(max(len(column), content_width))

    separator = "+" + "+".join("=" * (width + 2) for width in widths) + "+"
    header = "|" + "|".join(
        f" {column:^{width}} " for column, width in zip(columns, widths)
    ) + "|"

    print("\nBATCH LIQUIDUS SUMMARY")
    print(separator)
    print(header)
    print(separator)

    for row in display_rows:
        formatted_row = "|" + "|".join(
            f" {value:>{width}} " for value, width in zip(row, widths)
        ) + "|"
        print(formatted_row)

    print(separator)
    print("Composition columns are analysed input oxide values in wt%.")


def run_batch():
    compositions = linear_composition_path(
        start_composition=START_COMPOSITION,
        end_composition=END_COMPOSITION,
        point_count=POINT_COUNT,
    )

    BATCH_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    original_composition = calculator.ANALYSED_OXIDE_WT_PCT
    original_output_file = calculator.OUTPUT_FILE
    summary_records = []

    try:
        for composition_number, composition in enumerate(compositions, start=1):
            validate_composition(composition, composition_number)

            output_file = BATCH_OUTPUT_DIRECTORY / (
                f"melts_liquidus_point_{composition_number:03d}.csv"
            )

            if output_file.exists():
                output_file.unlink()

            calculator.ANALYSED_OXIDE_WT_PCT = composition
            calculator.OUTPUT_FILE = output_file

            print("\n" + "=" * 72)
            print(
                f"BATCH POINT {composition_number} OF {len(compositions)} "
                f"  OUTPUT: {output_file}"
            )
            print("=" * 72)

            calculator.main()

            summary_records.append(
                read_calculation_summary(
                    composition_number=composition_number,
                    composition=composition,
                    output_file=output_file,
                )
            )

    finally:
        calculator.ANALYSED_OXIDE_WT_PCT = original_composition
        calculator.OUTPUT_FILE = original_output_file

    print_batch_summary(summary_records)


def main():
    run_batch()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
