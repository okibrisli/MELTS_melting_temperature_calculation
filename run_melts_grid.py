from pathlib import Path
import multiprocessing

import melts_liquidus_single_final as calculator
from composition_grid_generator import composition_grid_size, make_composition_grid
from old.run_melts_batch import (
    print_batch_summary,
    read_calculation_summary,
    validate_composition,
)


COMPOSITION_RANGES = {
    "SiO2": (77.0, 77.0, 5.0),
    "TiO2": (0.0, 0.0, 0.0),
    "Al2O3": (5.0, 5.0, 0.0),
    "Fe2O3": (1.0, 1.0, 0.0),
    "FeO": (0.0, 0.0, 0.0),
    "MnO": (0.0, 0.0, 0.0),
    "MgO": (2.0, 2.0, 0.0),
    "CaO": (15.0, 15.0, 5.0),
    "Na2O": (0.0, 0.0, 0.0),
    "K2O": (0.0, 0.0, 0.0),
    "P2O5": (0.0, 0.0, 0.0),
    "H2O": (0.0, 0.0, 0.0),
    "CO2": (0.0, 0.0, 0.0),
}

BATCH_OUTPUT_DIRECTORY = Path("old/batch_melts_grid_results")
MAX_CALCULATIONS = 500


def run_grid():
    calculation_count = composition_grid_size(COMPOSITION_RANGES)

    if calculation_count > MAX_CALCULATIONS:
        raise ValueError(
            f"The defined grid contains {calculation_count} calculations. "
            f"Increase MAX_CALCULATIONS only after checking the range settings."
        )

    BATCH_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nCOMPOSITION GRID")
    print(f"Total calculations: {calculation_count}")
    print("Each generated composition is normalised by the original MELTS script.")

    original_composition = calculator.ANALYSED_OXIDE_WT_PCT
    original_output_file = calculator.OUTPUT_FILE
    summary_records = []

    try:
        for composition_number, composition in enumerate(
            make_composition_grid(COMPOSITION_RANGES),
            start=1,
        ):
            validate_composition(composition, composition_number)

            output_file = BATCH_OUTPUT_DIRECTORY / (
                f"melts_liquidus_grid_{composition_number:05d}.csv"
            )

            if output_file.exists():
                output_file.unlink()

            calculator.ANALYSED_OXIDE_WT_PCT = composition
            calculator.OUTPUT_FILE = output_file

            print("\n" + "=" * 72)
            print(
                f"GRID POINT {composition_number} OF {calculation_count} "
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
    run_grid()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
