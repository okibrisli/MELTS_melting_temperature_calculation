from pathlib import Path
import multiprocessing

import pandas as pd
import petthermotools as ptt


DRY_ANALYSED_OXIDE_WT_PCT = {
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

SCAN_H2O_WT_PCT = 0.001
MELTS_MODEL = "MELTSv1.0.2"
PRESSURE_BAR = 1.0
FE3_FET_LIQ = 0.80
SCAN_END_TEMPERATURE_C = 900.0
TEMPERATURE_STEP_C = 1.0
SOLID_MASS_TOLERANCE_G = 1.0e-8
OUTPUT_DIRECTORY = Path("../melts_single_equilibrium_check")

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

PHASE_LABELS = {
    "clinopyroxene1": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "clinopyroxene2": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "cpx": "Clinopyroxene",
    "orthopyroxene1": "Orthopyroxene, (Mg,Fe)SiO3 type",
    "opx": "Orthopyroxene",
    "olivine1": "Olivine, (Mg,Fe)2SiO4 type",
    "ol": "Olivine",
    "plagioclase1": "Plagioclase feldspar",
    "plag": "Plagioclase feldspar",
    "alkali feldspar1": "Alkali feldspar",
    "quartz1": "Quartz",
    "qtz": "Quartz",
    "tridymite1": "Tridymite",
    "cristobalite1": "Cristobalite",
    "spinel1": "Spinel type oxide",
    "sp": "Spinel type oxide",
    "rhm oxide1": "Rhombohedral oxide",
    "rhm": "Rhombohedral oxide",
    "ilmenite1": "Ilmenite",
    "ilm": "Ilmenite",
    "garnet1": "Garnet",
    "melilite1": "Melilite",
    "nepheline1": "Nepheline",
    "leucite1": "Leucite",
    "kalsilite1": "Kalsilite",
    "wollastonite1": "Wollastonite",
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

    normalised_bulk = {
        oxide: 100.0 * value / total
        for oxide, value in bulk.items()
    }
    return normalised_bulk, total


def phase_name(raw_name):
    if pd.isna(raw_name):
        return "Not returned"

    text = str(raw_name)
    return PHASE_LABELS.get(text.lower(), text)


def print_table(title, rows):
    left_width = max(len(str(label)) for label, _ in rows)
    right_width = max(len(str(value)) for _, value in rows)

    print(f"\n{title}")
    print("=" * (left_width + right_width + 5))
    for label, value in rows:
        print(f"{label:<{left_width}} : {value:>{right_width}}")


def print_bulk(title, bulk):
    print(f"\n{title}")
    for oxide, value in bulk.items():
        print(f"{oxide:<12} : {value:10.5f} wt%")


def first_value(frame, column, default="Not returned"):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return default
    if column not in frame.columns:
        return default

    value = frame[column].iloc[0]
    return default if pd.isna(value) else value


def liquidus_summary(result):
    temperature = first_value(result, "T_Liq_C")
    raw_phase = first_value(result, "liquidus_phase")

    if isinstance(temperature, (int, float)):
        temperature_text = f"{float(temperature):.2f} degC"
    else:
        temperature_text = str(temperature)

    return temperature_text, phase_name(raw_phase)


def get_all_results(crystallisation_result):
    if isinstance(crystallisation_result, dict):
        for key in ("All", "all"):
            if key in crystallisation_result:
                return crystallisation_result[key]

    if isinstance(crystallisation_result, pd.DataFrame):
        return crystallisation_result

    raise TypeError(
        "Could not find an All pandas DataFrame in the crystallisation result."
    )


def find_temperature_column(frame):
    for column in ("T_C", "TC", "Temperature_C", "T"):
        if column in frame.columns:
            return column

    raise KeyError("No temperature column was found in the crystallisation output.")


def find_phase_mass_columns(frame):
    return [
        column
        for column in frame.columns
        if str(column).lower().startswith("mass_g_")
    ]


def phase_name_from_mass_column(column):
    name = str(column)
    prefix = "mass_g_"

    if name.lower().startswith(prefix):
        return name[len(prefix):]

    return name


def analyse_scan(all_results):
    scan = all_results.copy()
    temperature_column = find_temperature_column(scan)
    solid_columns = find_phase_mass_columns(scan)

    if not solid_columns:
        raise KeyError(
            "No phase mass columns beginning with mass_g_ were found. "
            "The total system mass column mass_g is deliberately excluded."
        )

    phase_masses = (
        scan[solid_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    scan["solid_mass_total_g"] = phase_masses.sum(axis=1)

    crystal_rows = scan.loc[
        scan["solid_mass_total_g"] > SOLID_MASS_TOLERANCE_G
    ].copy()

    if crystal_rows.empty:
        return (
            scan,
            "No crystals found in scan",
            "No crystalline phase",
            "0.000000e+00 g",
            solid_columns,
        )

    onset_index = crystal_rows[temperature_column].astype(float).idxmax()
    onset_row = scan.loc[onset_index]
    onset_phase_masses = phase_masses.loc[onset_index]
    dominant_column = onset_phase_masses.idxmax()
    dominant_phase = phase_name(phase_name_from_mass_column(dominant_column))

    return (
        scan,
        f"{float(onset_row[temperature_column]):.2f} degC",
        dominant_phase,
        f"{float(onset_row['solid_mass_total_g']):.6e} g",
        solid_columns,
    )


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    scan_composition = dict(DRY_ANALYSED_OXIDE_WT_PCT)
    scan_composition["H2O"] = SCAN_H2O_WT_PCT

    dry_bulk, dry_total = create_melts_bulk(DRY_ANALYSED_OXIDE_WT_PCT)
    scan_bulk, scan_total = create_melts_bulk(scan_composition)

    print_table(
        "MELTS LIQUIDUS VALIDATION SETTINGS",
        [
            ("MELTS model", MELTS_MODEL),
            ("Pressure", f"{PRESSURE_BAR:.4f} bar"),
            ("Fe3+ to total Fe", f"{FE3_FET_LIQ:.5f}"),
            ("Dry direct calculation H2O", "0.000 wt%"),
            ("Path calculation H2O", f"{SCAN_H2O_WT_PCT:.3f} wt%"),
            ("Scan end temperature", f"{SCAN_END_TEMPERATURE_C:.2f} degC"),
            ("Temperature increment", f"{TEMPERATURE_STEP_C:.2f} degC"),
            ("Dry bulk total after FeOt conversion", f"{dry_total:.5f} wt%"),
            ("Path bulk total after FeOt conversion", f"{scan_total:.5f} wt%"),
        ],
    )

    print("\nStep 1 of 3: Direct dry MELTS liquidus calculation")
    dry_liquidus = ptt.findLiq_multi(
        Model=MELTS_MODEL,
        bulk=dry_bulk,
        P_bar=PRESSURE_BAR,
        Fe3Fet_Liq=FE3_FET_LIQ,
    )
    if not isinstance(dry_liquidus, pd.DataFrame):
        raise TypeError("Dry findLiq_multi did not return a pandas DataFrame.")
    dry_liquidus.to_csv(
        OUTPUT_DIRECTORY / "direct_dry_liquidus.csv",
        index=False,
        encoding="utf_8",
    )

    print("\nStep 2 of 3: Direct near dry MELTS liquidus calculation")
    near_dry_liquidus = ptt.findLiq_multi(
        Model=MELTS_MODEL,
        bulk=scan_bulk,
        P_bar=PRESSURE_BAR,
        Fe3Fet_Liq=FE3_FET_LIQ,
    )
    if not isinstance(near_dry_liquidus, pd.DataFrame):
        raise TypeError("Near dry findLiq_multi did not return a pandas DataFrame.")
    near_dry_liquidus.to_csv(
        OUTPUT_DIRECTORY / "direct_near_dry_liquidus.csv",
        index=False,
        encoding="utf_8",
    )

    print("\nStep 3 of 3: Near dry equilibrium crystallisation scan")
    print("The 0.001 wt% H2O value is only used because this PetThermoTools path wrapper rejects zero H2O.")
    crystallisation = ptt.isobaric_crystallisation(
        Model=MELTS_MODEL,
        bulk=scan_bulk,
        find_liquidus=True,
        P_bar=PRESSURE_BAR,
        T_end_C=SCAN_END_TEMPERATURE_C,
        dt_C=TEMPERATURE_STEP_C,
        Fe3Fet_init=FE3_FET_LIQ,
        Frac_solid=False,
        Frac_fluid=False,
    )

    all_results = get_all_results(crystallisation)
    if not isinstance(all_results, pd.DataFrame):
        raise TypeError("The crystallisation All output is not a pandas DataFrame.")

    scan, onset_temperature, onset_phase, onset_solid_mass, solid_columns = analyse_scan(
        all_results
    )
    scan.to_csv(
        OUTPUT_DIRECTORY / "equilibrium_crystallisation_scan.csv",
        index=False,
        encoding="utf_8",
    )

    dry_temperature, dry_phase = liquidus_summary(dry_liquidus)
    near_dry_temperature, near_dry_phase = liquidus_summary(near_dry_liquidus)

    print_table(
        "LIQUIDUS CROSS CHECK",
        [
            ("Direct dry liquidus", dry_temperature),
            ("Direct dry primary phase", dry_phase),
            ("Direct near dry liquidus", near_dry_temperature),
            ("Direct near dry primary phase", near_dry_phase),
            ("Highest scan temperature with crystals", onset_temperature),
            ("Dominant phase at scan onset", onset_phase),
            ("Total solid mass at scan onset", onset_solid_mass),
        ],
    )

    print_bulk("NORMALISED DRY BULK SENT TO DIRECT MELTS", dry_bulk)
    print_bulk("NORMALISED NEAR DRY BULK SENT TO PATH MELTS", scan_bulk)

    print("\nSolid phase mass columns detected in the equilibrium scan:")
    for column in solid_columns:
        print(f"{column:<30} : {phase_name(phase_name_from_mass_column(column))}")

    print("\nFiles written to:")
    print(OUTPUT_DIRECTORY.resolve())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
