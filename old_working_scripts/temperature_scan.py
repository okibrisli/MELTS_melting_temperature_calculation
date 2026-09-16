"""MELTS equilibrium temperature scan with a pycalphad style findings table."""

from pathlib import Path
import multiprocessing
import re
import numpy as np
import pandas as pd
import petthermotools as ptt


ANALYSED_OXIDE_WT_PCT = {
    "SiO2": 69.0,
    "TiO2": 0.0,
    "Al2O3": 5.0,
    "Fe2O3": 3.0,
    "FeO": 0.0,
    "MnO": 0.0,
    "MgO": 2.0,
    "CaO": 21.0,
    "Na2O": 0.0,
    "K2O": 0.0,
    "P2O5": 0.0,
    "H2O": 0.0,
    "CO2": 0.0,
}

PRESSURE_BAR = 1.0
MELTS_MODEL = "MELTSv1.0.2"
FE3_FET_LIQ = 0.15
T_MIN_C = 1000.0
T_MAX_C = 1400.0
TEMPERATURE_STEP_C = 5.0
FIRST_LIQUID_THRESHOLD = 1.0e-4
MILESTONES = (0.10, 0.50, 0.90, 0.99)
PRINT_FULL_SCAN_TO_TERMINAL = False

SCAN_FILE = Path("../old/melts_temperature_scan.csv")
FINDINGS_FILE = Path("melts_melting_findings.csv")
LIQUIDUS_FILE = Path("../old/melts_liquidus_result.csv")

FEO_MOLAR_MASS = 71.844
FE2O3_MOLAR_MASS = 159.688

PHASE_LABELS = {
    "clinopyroxene": "Clinopyroxene, Ca(Mg,Fe)Si2O6 type",
    "orthopyroxene": "Orthopyroxene, (Mg,Fe)SiO3 type",
    "olivine": "Olivine, (Mg,Fe)2SiO4 type",
    "plagioclase": "Plagioclase feldspar",
    "alkali feldspar": "Alkali feldspar",
    "quartz": "Quartz, SiO2",
    "tridymite": "Tridymite, SiO2",
    "cristobalite": "Cristobalite, SiO2",
    "spinel": "Spinel type oxide",
    "rhm oxide": "Rhombohedral oxide",
    "ilmenite": "Ilmenite",
    "garnet": "Garnet",
    "melilite": "Melilite",
    "nepheline": "Nepheline, NaAlSiO4 type",
    "leucite": "Leucite, KAlSi2O6",
    "kalsilite": "Kalsilite, KAlSiO4",
}


def to_feot(oxides):
    return oxides.get("FeO", 0.0) + oxides.get("Fe2O3", 0.0) * 2.0 * FEO_MOLAR_MASS / FE2O3_MOLAR_MASS


def make_bulk(oxides):
    bulk = {
        "SiO2_Liq": oxides.get("SiO2", 0.0),
        "TiO2_Liq": oxides.get("TiO2", 0.0),
        "Al2O3_Liq": oxides.get("Al2O3", 0.0),
        "Cr2O3_Liq": oxides.get("Cr2O3", 0.0),
        "FeOt_Liq": to_feot(oxides),
        "MnO_Liq": oxides.get("MnO", 0.0),
        "MgO_Liq": oxides.get("MgO", 0.0),
        "CaO_Liq": oxides.get("CaO", 0.0),
        "Na2O_Liq": oxides.get("Na2O", 0.0),
        "K2O_Liq": oxides.get("K2O", 0.0),
        "P2O5_Liq": oxides.get("P2O5", 0.0),
        "H2O_Liq": oxides.get("H2O", 0.0),
        "CO2_Liq": oxides.get("CO2", 0.0),
        "Fe3Fet_Liq": FE3_FET_LIQ,
    }
    total = sum(value for key, value in bulk.items() if key != "Fe3Fet_Liq")
    if total <= 0:
        raise ValueError("No positive oxide amount entered.")
    for key in bulk:
        if key != "Fe3Fet_Liq":
            bulk[key] = 100.0 * bulk[key] / total
    return bulk, total


def pretty_phase(raw_name):
    clean = re.sub(r"\d+$", "", str(raw_name)).replace("_", " ").lower()
    return PHASE_LABELS.get(clean, clean.title())


def solid_columns(dataframe):
    return [column for column in dataframe.columns if column.startswith("mass_g_") and column != "mass_g_Liq"]


def format_solids(row):
    values = []
    total = row["mass_g"]
    for column in solid_columns(row.to_frame().T):
        mass = row[column]
        if pd.notna(mass) and mass / total >= FIRST_LIQUID_THRESHOLD:
            values.append(f"{pretty_phase(column[7:])}: {mass / total:.5f}")
    return "; ".join(values) if values else "No solid phase above threshold"


def enrich_scan(dataframe):
    required = {"T_C", "mass_g", "mass_g_Liq"}
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError("Required MELTS output columns missing: " + ", ".join(sorted(missing)))
    result = dataframe.copy()
    result["liquid_fraction"] = result["mass_g_Liq"] / result["mass_g"]
    result["solid_fraction"] = 1.0 - result["liquid_fraction"]
    result["stable_solid_phases"] = result.apply(format_solids, axis=1)
    return result.sort_values("T_C").reset_index(drop=True)


def crossing(scan, threshold):
    valid = scan.dropna(subset=["liquid_fraction"])
    for index in range(1, len(valid)):
        low = valid.iloc[index - 1]
        high = valid.iloc[index]
        if low["liquid_fraction"] < threshold <= high["liquid_fraction"]:
            if high["liquid_fraction"] == low["liquid_fraction"]:
                temperature = high["T_C"]
            else:
                ratio = (threshold - low["liquid_fraction"]) / (high["liquid_fraction"] - low["liquid_fraction"])
                temperature = low["T_C"] + ratio * (high["T_C"] - low["T_C"])
            return float(temperature), high
    return None, None


def print_findings(findings):
    print("\nKEY FINDINGS")
    print("Finding                       Temperature degC   Liquid fraction   Solid fraction   Stable phases")
    print("=" * 170)
    for row in findings:
        temperature = "Not found" if row["temperature_degC"] is None else f"{row['temperature_degC']:.2f}"
        liquid = "" if row["liquid_fraction"] is None else f"{row['liquid_fraction']:.6f}"
        solid = "" if row["solid_fraction"] is None else f"{row['solid_fraction']:.6f}"
        print(f"{row['finding']:<29} {temperature:>18} {liquid:>17} {solid:>16}   {row['stable_phases']}")


def main():
    bulk, total_before_normalisation = make_bulk(ANALYSED_OXIDE_WT_PCT)
    print("\nInput oxide composition, wt%")
    for oxide, value in ANALYSED_OXIDE_WT_PCT.items():
        print(f"  {oxide:>6s} : {value:10.5f}")
    print(f"  Total  : {sum(ANALYSED_OXIDE_WT_PCT.values()):10.5f}")
    print(f"  Pressure: {PRESSURE_BAR:.5f} bar")
    print(f"  Fe3+ to total Fe: {FE3_FET_LIQ:.5f}")
    print(f"  FeOt before normalisation: {to_feot(ANALYSED_OXIDE_WT_PCT):.5f} wt%")
    print(f"  MELTS total before normalisation: {total_before_normalisation:.5f} wt%")

    temperatures = np.arange(T_MIN_C, T_MAX_C + 0.5 * TEMPERATURE_STEP_C, TEMPERATURE_STEP_C)
    print("\nCalculating MELTS equilibrium temperature scan")
    print(f"  Temperature range: {T_MIN_C:.1f} to {T_MAX_C:.1f} degC")
    print(f"  Temperature step: {TEMPERATURE_STEP_C:.1f} degC")

    raw_scan = ptt.phaseDiagram_calc(
        Model=MELTS_MODEL,
        bulk=bulk,
        P_bar=np.array([PRESSURE_BAR]),
        T_C=temperatures,
        i_max=15,
        refine=0,
    )
    scan = enrich_scan(raw_scan)
    scan.to_csv(SCAN_FILE, index=False, encoding="utf-8")

    if PRINT_FULL_SCAN_TO_TERMINAL:
        print("\nCoarse temperature scan")
        for _, row in scan.iterrows():
            print(f"  {row['T_C']:8.1f} degC   Liquid fraction = {row['liquid_fraction']:.8f}   {row['stable_solid_phases']}")

    findings = []
    for label, threshold in [("First liquid boundary", FIRST_LIQUID_THRESHOLD)] + [(f"T{int(value * 100):02d}, liquid fraction", value) for value in MILESTONES]:
        temperature, row = crossing(scan, threshold)
        if row is None:
            findings.append({"finding": label, "temperature_degC": None, "liquid_fraction": None, "solid_fraction": None, "stable_phases": "Not bracketed in selected temperature range"})
        else:
            findings.append({"finding": label, "temperature_degC": temperature, "liquid_fraction": threshold, "solid_fraction": 1.0 - threshold, "stable_phases": row["stable_solid_phases"]})

    print("\nCalculating MELTS true liquidus")
    liquidus = ptt.findLiq_multi(Model=MELTS_MODEL, bulk=bulk, P_bar=PRESSURE_BAR, Fe3Fet_Liq=FE3_FET_LIQ)
    liquidus.to_csv(LIQUIDUS_FILE, index=False, encoding="utf-8")
    liquidus_c = float(liquidus["T_Liq_C"].iloc[0])
    liquidus_phase = pretty_phase(liquidus["liquidus_phase"].iloc[0])
    findings.append({"finding": "MELTS true liquidus", "temperature_degC": liquidus_c, "liquid_fraction": 1.0, "solid_fraction": 0.0, "stable_phases": f"Primary liquidus phase: {liquidus_phase}"})

    findings_dataframe = pd.DataFrame(findings)
    findings_dataframe.to_csv(FINDINGS_FILE, index=False, encoding="utf-8")
    print_findings(findings)
    print("\nCSV files written")
    print(f"  {SCAN_FILE.resolve()}")
    print(f"  {FINDINGS_FILE.resolve()}")
    print(f"  {LIQUIDUS_FILE.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
