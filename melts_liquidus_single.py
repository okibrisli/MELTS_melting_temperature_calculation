from pathlib import Path
import multiprocessing
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
OUTPUT_FILE = Path("old/melts_liquidus_result.csv")

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

PHASE_LABELS = {
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
    if total <= 0:
        raise ValueError("The entered composition contains no positive oxide values.")
    return {oxide: 100.0 * value / total for oxide, value in bulk.items()}, total


def phase_name(raw_name):
    if pd.isna(raw_name):
        return "Not returned by MELTS"
    return PHASE_LABELS.get(str(raw_name).lower(), str(raw_name))


def get_value(result, column, default="Not returned"):
    if column not in result.columns:
        return default
    value = result[column].iloc[0]
    return default if pd.isna(value) else value


def print_two_column_table(title, rows):
    left_width = max(len(str(label)) for label, _ in rows)
    right_width = max(len(str(value)) for _, value in rows)
    print(f"\n{title}")
    print("=" * (left_width + right_width + 5))
    for label, value in rows:
        print(f"{label:<{left_width}} : {value:>{right_width}}")


def print_composition_table(title, composition):
    rows = [(oxide, f"{value:.4f} wt%") for oxide, value in composition.items()]
    print_two_column_table(title, rows)


def print_liquidus_summary(result, melts_bulk, feot_before_normalisation):
    liquidus_c = get_value(result, "T_Liq_C")
    liquidus_phase_raw = get_value(result, "liquidus_phase")
    fluid_saturated = get_value(result, "fluid_saturated")
    final_fe3fet = get_value(result, "Fe3Fet_Liq")

    if isinstance(liquidus_c, (float, int)):
        liquidus_display = f"{liquidus_c:.2f} degC"
    else:
        liquidus_display = str(liquidus_c)

    if isinstance(final_fe3fet, (float, int)):
        fe3fet_display = f"{final_fe3fet:.5f}"
    else:
        fe3fet_display = str(final_fe3fet)

    summary_rows = [
        ("MELTS model", MELTS_MODEL),
        ("Pressure", f"{PRESSURE_BAR:.4f} bar"),
        ("Liquidus temperature", liquidus_display),
        ("Primary liquidus phase", phase_name(liquidus_phase_raw)),
        ("MELTS phase identifier", str(liquidus_phase_raw)),
        ("Fluid saturated", str(fluid_saturated)),
        ("Input Fe3+ to total Fe", f"{FE3_FET_LIQ:.5f}"),
        ("Calculated Fe3+ to total Fe", fe3fet_display),
        ("FeOt before normalisation", f"{feot_before_normalisation:.4f} wt%"),
    ]
    print_two_column_table("MELTS LIQUIDUS SUMMARY", summary_rows)

    liquid_composition_columns = [
        "SiO2_Liq", "TiO2_Liq", "Al2O3_Liq", "Cr2O3_Liq", "FeOt_Liq",
        "MnO_Liq", "MgO_Liq", "CaO_Liq", "Na2O_Liq", "K2O_Liq",
        "P2O5_Liq", "H2O_Liq", "CO2_Liq",
    ]
    liquid_rows = []
    for column in liquid_composition_columns:
        value = get_value(result, column, None)
        if value is not None:
            liquid_rows.append((column.replace("_Liq", ""), f"{value:.4f} wt%"))

    if liquid_rows:
        print_two_column_table("LIQUID COMPOSITION AT LIQUIDUS", liquid_rows)

    print_composition_table("NORMALISED BULK COMPOSITION SENT TO MELTS", melts_bulk)


def main():
    print_composition_table("ANALYSED OXIDE COMPOSITION", ANALYSED_OXIDE_WT_PCT)

    melts_bulk, total_before_normalisation = create_melts_bulk(ANALYSED_OXIDE_WT_PCT)
    feot_before_normalisation = convert_fe_oxides_to_feot(ANALYSED_OXIDE_WT_PCT)

    print_two_column_table(
        "IRON CONVERSION",
        [
            ("Input Fe2O3", f"{ANALYSED_OXIDE_WT_PCT.get('Fe2O3', 0.0):.4f} wt%"),
            ("Input FeO", f"{ANALYSED_OXIDE_WT_PCT.get('FeO', 0.0):.4f} wt%"),
            ("Total FeOt as FeO equivalent", f"{feot_before_normalisation:.4f} wt%"),
            ("MELTS total before normalisation", f"{total_before_normalisation:.4f} wt%"),
        ],
    )

    print("\nStarting MELTS liquidus calculation.")
    print("alphaMELTS solver messages may appear before the summary table.")

    try:
        liquidus_result = ptt.findLiq_multi(
            Model=MELTS_MODEL,
            bulk=melts_bulk,
            P_bar=PRESSURE_BAR,
            Fe3Fet_Liq=FE3_FET_LIQ,
        )

        if not isinstance(liquidus_result, pd.DataFrame):
            raise TypeError("MELTS did not return a pandas DataFrame.")

        liquidus_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print_liquidus_summary(
            liquidus_result,
            melts_bulk,
            feot_before_normalisation,
        )
        print(f"\nFull unformatted MELTS output was saved to:\n{OUTPUT_FILE.resolve()}")

    except Exception as error:
        print_two_column_table(
            "MELTS LIQUIDUS CALCULATION FAILED",
            [("Error type", type(error).__name__), ("Error message", str(error))],
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
