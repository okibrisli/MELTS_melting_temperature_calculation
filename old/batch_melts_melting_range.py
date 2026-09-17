"""
Batch MELTS melting-range calculator.

Reads a CSV of oxide compositions (wt%), runs each one through PetThermoTools
(alphaMELTS backend) to determine:
  - T_liquidus_C   : temperature at which the last solid disappears (full melt)
  - T_solidus_C    : temperature at which the first liquid appears on heating
                     (melting start), estimated from an equilibrium cooling
                     path run from the liquidus down to T_END_C
  - liquidus_phase : first phase to crystallise on cooling from full melt
  - melting_interval_C : T_liquidus_C - T_solidus_C

Any oxide column in the CSV that MELTS does not carry as a liquid component
(e.g. ZrO2, HfO2) is dropped before the calculation, the remaining oxides are
renormalised to 100 wt%, and the amount excluded is written into the report so
you can judge whether the approximation is still meaningful for that sample.

Each composition is wrapped in its own try/except so one bad/unstable sample
never aborts the whole batch; failures are logged with the error message.

Usage (PyCharm or terminal):
    python batch_melts_melting_range.py compositions.csv

Requires: petthermotools, pandas, numpy, alphaMELTS for Python already on path
(exactly the setup you already have working for melts_liquidus_single_final.py).
"""

from pathlib import Path
import multiprocessing
import sys
import traceback

import numpy as np
import pandas as pd
import petthermotools as ptt

# --------------------------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------------------------

INPUT_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("compositions.csv")
OUTPUT_DIR = Path("../melts_batch_output")

MELTS_MODEL = "MELTSv1.0.2"        # "pMELTS", "MELTSv1.0.2", "MELTSv1.1.0", "MELTSv1.2.0"
PRESSURE_BAR = 1.0                 # atmospheric, adjust if your kiln process runs otherwise
FE3_FET_LIQ = 1.0                  # only Fe2O3 is reported in the sheet -> assume fully oxidised Fe
T_END_C = 600.0                    # lower bound of the cooling scan (raise/lower if solidus not bracketed)
DT_C = 5.0                         # step size of the cooling path, degC (smaller = slower but finer)
LIQUID_FRACTION_ONSET = 0.01       # 1 wt% liquid = "melting has started" threshold
MAX_EXCLUDED_OXIDE_WT_PCT = 5.0    # warn if more than this wt% of the sample had to be dropped

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

# Oxide components MELTS/alphaMELTS actually solves for as liquid components.
# Anything in the CSV that is not in this set gets excluded and reported.
MELTS_SUPPORTED_OXIDES = {
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "Cr2O3", "FeO", "MnO", "MgO", "NiO",
    "CoO", "CaO", "Na2O", "K2O", "P2O5", "H2O", "CO2", "SO3", "F2O-1", "Cl2O-1",
}

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
    "rhm-oxide1": "Rhombohedral oxide, hematite ilmenite type",
    "ilmenite1": "Ilmenite type oxide",
    "garnet1": "Garnet solid solution",
    "melilite1": "Melilite solid solution",
    "nepheline1": "Nepheline, NaAlSiO4 type",
    "leucite1": "Leucite, KAlSi2O6",
    "kalsilite1": "Kalsilite, KAlSiO4",
}


def phase_name(raw_name):
    if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
        return "Not returned by MELTS"
    return PHASE_LABELS.get(str(raw_name).lower(), str(raw_name))


def convert_fe_oxides_to_feot(feo, fe2o3):
    return feo + fe2o3 * (2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3)


def clean_and_normalise(row, oxide_columns):
    """Split oxide columns into MELTS-supported vs excluded, renormalise the
    supported ones to 100 wt%. Returns (clean_dict, excluded_dict, excluded_total)."""
    supported, excluded = {}, {}
    for oxide in oxide_columns:
        value = float(row.get(oxide, 0.0) or 0.0)
        if value <= 0.0:
            continue
        if oxide in MELTS_SUPPORTED_OXIDES:
            supported[oxide] = value
        else:
            excluded[oxide] = value

    supported_total = sum(supported.values())
    excluded_total = sum(excluded.values())
    if supported_total <= 0:
        raise ValueError("No MELTS-supported oxides with a positive value in this row.")

    normalised = {ox: 100.0 * val / supported_total for ox, val in supported.items()}
    return normalised, excluded, excluded_total


def build_melts_bulk(normalised):
    feo = normalised.get("FeO", 0.0)
    fe2o3 = normalised.get("Fe2O3", 0.0)
    bulk = {
        "SiO2_Liq": normalised.get("SiO2", 0.0),
        "TiO2_Liq": normalised.get("TiO2", 0.0),
        "Al2O3_Liq": normalised.get("Al2O3", 0.0),
        "Cr2O3_Liq": normalised.get("Cr2O3", 0.0),
        "FeOt_Liq": convert_fe_oxides_to_feot(feo, fe2o3),
        "MnO_Liq": normalised.get("MnO", 0.0),
        "MgO_Liq": normalised.get("MgO", 0.0),
        "CaO_Liq": normalised.get("CaO", 0.0),
        "Na2O_Liq": normalised.get("Na2O", 0.0),
        "K2O_Liq": normalised.get("K2O", 0.0),
        "P2O5_Liq": normalised.get("P2O5", 0.0),
        "H2O_Liq": normalised.get("H2O", 0.0),
        "CO2_Liq": normalised.get("CO2", 0.0),
    }
    total = sum(bulk.values())
    if total <= 0:
        raise ValueError("Normalised composition sums to zero, cannot send to MELTS.")
    return {ox: 100.0 * val / total for ox, val in bulk.items()}


def get_first(df, column, default=None):
    if column not in df.columns or len(df) == 0:
        return default
    value = df[column].iloc[0]
    return default if pd.isna(value) else value


def find_solidus_from_path(xtal_result):
    """Given the dict returned by ptt.isobaric_crystallisation, walk the
    cooling path from cold to hot and return the temperature at which the
    liquid mass fraction first exceeds LIQUID_FRACTION_ONSET (i.e. melting
    start on heating)."""
    all_df = xtal_result.get("All")
    mass_df = xtal_result.get("mass_g")
    if all_df is None or mass_df is None or len(all_df) == 0:
        return None, "no path data returned"

    liquid_cols = [c for c in mass_df.columns if c.startswith("liquid") and not c.endswith("_cumsum")]
    if not liquid_cols:
        return None, "no liquid mass column in path output"
    liquid_mass = mass_df[liquid_cols[0]].fillna(0.0)

    total_mass = mass_df[[c for c in mass_df.columns if not c.endswith("_cumsum")]].sum(axis=1)
    total_mass = total_mass.replace(0.0, np.nan)
    liquid_fraction = (liquid_mass / total_mass).fillna(0.0)

    temperature = all_df["T_C"] if "T_C" in all_df.columns else all_df.get("Temp")
    if temperature is None:
        return None, "no temperature column in path output"

    order = np.argsort(temperature.values)
    temp_sorted = temperature.values[order]
    liq_sorted = liquid_fraction.values[order]

    above_onset = np.where(liq_sorted >= LIQUID_FRACTION_ONSET)[0]
    if len(above_onset) == 0:
        return None, f"liquid fraction never reached {LIQUID_FRACTION_ONSET:.0%} within scanned range"

    idx = above_onset[0]
    if idx == 0:
        return float(temp_sorted[0]), f"solidus below scanned range (T_END_C={T_END_C}), reported lower bound"

    t_low, t_high = temp_sorted[idx - 1], temp_sorted[idx]
    f_low, f_high = liq_sorted[idx - 1], liq_sorted[idx]
    if f_high == f_low:
        t_solidus = t_high
    else:
        frac = (LIQUID_FRACTION_ONSET - f_low) / (f_high - f_low)
        t_solidus = t_low + frac * (t_high - t_low)
    return float(t_solidus), "ok"


def process_composition(sample_id, row, oxide_columns):
    record = {
        "Sample_ID": sample_id,
        "T_liquidus_C": np.nan,
        "T_solidus_C": np.nan,
        "melting_interval_C": np.nan,
        "liquidus_phase": None,
        "fluid_saturated": None,
        "excluded_oxides": "",
        "excluded_oxides_wt_pct": 0.0,
        "status": "ok",
        "note": "",
    }
    try:
        normalised, excluded, excluded_total = clean_and_normalise(row, oxide_columns)
        record["excluded_oxides"] = ", ".join(f"{k}={v:.3g}wt%" for k, v in excluded.items())
        record["excluded_oxides_wt_pct"] = round(excluded_total, 3)
        if excluded_total > MAX_EXCLUDED_OXIDE_WT_PCT:
            record["note"] += (
                f"WARNING: {excluded_total:.2f} wt% of the raw composition is not "
                "supported by MELTS and was dropped before normalisation, treat "
                "results for this sample with caution. "
            )

        melts_bulk = build_melts_bulk(normalised)

        liq_result = ptt.findLiq_multi(
            Model=MELTS_MODEL,
            bulk=melts_bulk,
            P_bar=PRESSURE_BAR,
            Fe3Fet_Liq=FE3_FET_LIQ,
        )
        t_liquidus = get_first(liq_result, "T_Liq_C")
        liquidus_phase_raw = get_first(liq_result, "liquidus_phase")
        fluid_saturated = get_first(liq_result, "fluid_saturated")

        if t_liquidus is None:
            record["status"] = "failed"
            record["note"] += "MELTS could not find a liquidus for this composition. "
            return record

        record["T_liquidus_C"] = round(float(t_liquidus), 2)
        record["liquidus_phase"] = phase_name(liquidus_phase_raw)
        record["fluid_saturated"] = fluid_saturated

        xtal_result = ptt.isobaric_crystallisation(
            Model=MELTS_MODEL,
            bulk=melts_bulk,
            find_liquidus=True,
            P_bar=PRESSURE_BAR,
            Fe3Fet_Liq=FE3_FET_LIQ,
            T_end_C=T_END_C,
            dt_C=DT_C,
            Frac_solid=False,
            Frac_fluid=False,
        )

        # isobaric_crystallisation may return a dict keyed by a label (e.g. pressure)
        # when arrays are passed, or the path dict directly for a single condition.
        if "All" not in xtal_result:
            xtal_result = next(iter(xtal_result.values()))

        t_solidus, solidus_note = find_solidus_from_path(xtal_result)
        if t_solidus is not None:
            record["T_solidus_C"] = round(t_solidus, 2)
            record["melting_interval_C"] = round(record["T_liquidus_C"] - t_solidus, 2)
        if solidus_note != "ok":
            record["note"] += solidus_note + ". "

    except Exception as error:
        record["status"] = "failed"
        record["note"] += f"{type(error).__name__}: {error}"

    return record


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    if "Sample_ID" not in df.columns:
        df.insert(0, "Sample_ID", [f"S{i+1:03d}" for i in range(len(df))])

    oxide_columns = [c for c in df.columns if c != "Sample_ID"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []

    total = len(df)
    for i, row in df.iterrows():
        sample_id = str(row["Sample_ID"])
        print(f"[{i+1}/{total}] Processing {sample_id} ...")
        record = process_composition(sample_id, row, oxide_columns)
        results.append(record)
        status = record["status"]
        t_liq = record["T_liquidus_C"]
        t_sol = record["T_solidus_C"]
        print(f"    status={status}  T_liquidus_C={t_liq}  T_solidus_C={t_sol}")

    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / "melting_ranges_report.csv"
    results_df.to_csv(results_csv, index=False, encoding="utf-8")

    n_ok = (results_df["status"] == "ok").sum()
    n_failed = (results_df["status"] == "failed").sum()

    report_lines = [
        "MELTS BATCH MELTING RANGE REPORT",
        "=================================",
        f"Model: {MELTS_MODEL}   Pressure: {PRESSURE_BAR} bar   Fe3+/FeT: {FE3_FET_LIQ}",
        f"Cooling scan: liquidus -> {T_END_C} degC, step {DT_C} degC, "
        f"melting-start threshold = {LIQUID_FRACTION_ONSET:.0%} liquid",
        f"Total compositions: {total}   Succeeded: {n_ok}   Failed: {n_failed}",
        "",
    ]
    for _, r in results_df.iterrows():
        report_lines.append(
            f"{r['Sample_ID']:<8} | T_solidus={r['T_solidus_C']:>8} C | "
            f"T_liquidus={r['T_liquidus_C']:>8} C | interval={r['melting_interval_C']:>7} C | "
            f"liquidus phase={r['liquidus_phase']}"
        )
        if r["excluded_oxides"]:
            report_lines.append(f"           excluded oxides: {r['excluded_oxides']}")
        if r["note"]:
            report_lines.append(f"           note: {r['note']}")

    report_txt = OUTPUT_DIR / "melting_ranges_report.txt"
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed.")
    print(f"CSV report:  {results_csv.resolve()}")
    print(f"Text report: {report_txt.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
