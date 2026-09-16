"""
Batch MELTS melting-range calculator (v3).

Fix vs v2: findLiq_multi() does NOT accept a `timeout` kwarg in
PetThermoTools (only isobaric_crystallisation does). Passing it caused an
instant TypeError on every sample before MELTS was even called
(elapsed=0.0s, status=liquidus_failed). The timeout kwarg is now only
passed to isobaric_crystallisation, where it is actually supported and
where the original 5-minute-timeout problem happened.

Everything else from v2 is unchanged:
  - Trace H2O added per composition to damp MELTS phase add/drop
    oscillation (alkali-feldspar/whitlockite flip-flop seen in the first run).
  - Crystallisation sweep runs from the sample's own liquidus down to
    max(T_END_MIN_C, liquidus - MAX_UNDERCOOLING_C), not a fixed 600 degC
    for every sample.
  - Liquidus and solidus stages are tried/caught separately, so a solidus
    timeout no longer discards an already-successful liquidus value.
  - Unsupported oxides (e.g. ZrO2, HfO2) are dropped and renormalised, with
    the excluded amount reported per sample.

Usage (PyCharm terminal):
    python batch_melts_melting_range.py compositions.csv
"""

from pathlib import Path
import multiprocessing
import sys
import time

import numpy as np
import pandas as pd
import petthermotools as ptt

# --------------------------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------------------------

INPUT_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("compositions.csv")
OUTPUT_DIR = Path("../melts_batch_output")

MELTS_MODEL = "MELTSv1.0.2"
PRESSURE_BAR = 1.0
FE3_FET_LIQ = 1.0                  # only Fe2O3 given in the sheet -> fully oxidised Fe assumed

# --- crystallisation path scan settings ---
DT_C = 10.0
MAX_UNDERCOOLING_C = 400.0         # scan down to (T_liquidus - this), not a fixed floor for every sample
T_END_MIN_C = 500.0                # absolute floor, never scan colder than this
LIQUID_FRACTION_ONSET = 0.01       # 1 wt% liquid = "melting has started"

# --- solver stability / timeout settings ---
TRACE_H2O_WT_PCT = 0.2             # added to every composition to damp phase add/drop oscillation
ISOBARIC_TIMEOUT_S = 900           # findLiq_multi does NOT accept a timeout kwarg, only this call does

MAX_EXCLUDED_OXIDE_WT_PCT = 5.0

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

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
    "alkali-feldspar1": "Alkali feldspar, KAlSi3O8 to NaAlSi3O8",
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
    "whitlockite1": "Whitlockite, Ca3(PO4)2 type phosphate",
    "apatite1": "Apatite, Ca5(PO4)3(OH,F,Cl) type phosphate",
}


def phase_name(raw_name):
    if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
        return "Not returned by MELTS"
    return PHASE_LABELS.get(str(raw_name).lower(), str(raw_name))


def convert_fe_oxides_to_feot(feo, fe2o3):
    return feo + fe2o3 * (2.0 * MOLAR_MASS_FEO / MOLAR_MASS_FE2O3)


def clean_and_normalise(row, oxide_columns):
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

    if TRACE_H2O_WT_PCT > 0:
        scale = 100.0 / (100.0 + TRACE_H2O_WT_PCT)
        normalised = {ox: val * scale for ox, val in normalised.items()}
        normalised["H2O"] = normalised.get("H2O", 0.0) + TRACE_H2O_WT_PCT * scale

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
    all_df = xtal_result.get("All")
    mass_df = xtal_result.get("mass_g")
    if all_df is None or mass_df is None or len(all_df) == 0:
        return None, "no path data returned (path likely timed out before any steps completed)"

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
        return None, (
            f"liquid fraction never reached {LIQUID_FRACTION_ONSET:.0%} within the "
            f"scanned range -> solidus is below the scan window, raise MAX_UNDERCOOLING_C"
        )

    idx = above_onset[0]
    if idx == 0:
        return float(temp_sorted[0]), "solidus below scanned range, reported lower bound of the scan"

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
        "elapsed_s": np.nan,
    }
    start = time.time()
    try:
        normalised, excluded, excluded_total = clean_and_normalise(row, oxide_columns)
        record["excluded_oxides"] = ", ".join(f"{k}={v:.3g}wt%" for k, v in excluded.items())
        record["excluded_oxides_wt_pct"] = round(excluded_total, 3)
        if excluded_total > MAX_EXCLUDED_OXIDE_WT_PCT:
            record["note"] += (
                f"WARNING: {excluded_total:.2f} wt% of the raw composition is not "
                "supported by MELTS and was dropped before normalisation. "
            )

        melts_bulk = build_melts_bulk(normalised)

        # --- Stage 1: liquidus (full melt). No timeout kwarg here, findLiq_multi does not accept one. ---
        try:
            liq_result = ptt.findLiq_multi(
                Model=MELTS_MODEL,
                bulk=melts_bulk,
                P_bar=PRESSURE_BAR,
                Fe3Fet_Liq=FE3_FET_LIQ,
            )
        except Exception as error:
            record["status"] = "liquidus_failed"
            record["note"] += f"Liquidus step failed: {type(error).__name__}: {error}"
            return record

        t_liquidus = get_first(liq_result, "T_Liq_C")
        liquidus_phase_raw = get_first(liq_result, "liquidus_phase")
        fluid_saturated = get_first(liq_result, "fluid_saturated")

        if t_liquidus is None:
            record["status"] = "liquidus_failed"
            record["note"] += "MELTS could not find a liquidus for this composition. "
            return record

        record["T_liquidus_C"] = round(float(t_liquidus), 2)
        record["liquidus_phase"] = phase_name(liquidus_phase_raw)
        record["fluid_saturated"] = fluid_saturated

        # --- Stage 2: cooling path -> solidus (melting start). timeout is valid here. ---
        t_end_run = max(T_END_MIN_C, record["T_liquidus_C"] - MAX_UNDERCOOLING_C)
        try:
            xtal_result = ptt.isobaric_crystallisation(
                Model=MELTS_MODEL,
                bulk=melts_bulk,
                find_liquidus=True,
                P_bar=PRESSURE_BAR,
                Fe3Fet_Liq=FE3_FET_LIQ,
                T_end_C=t_end_run,
                dt_C=DT_C,
                Frac_solid=False,
                Frac_fluid=False,
                timeout=ISOBARIC_TIMEOUT_S,
            )
        except Exception as error:
            record["status"] = "solidus_incomplete"
            record["note"] += (
                f"Solidus sweep failed/timed out after {ISOBARIC_TIMEOUT_S}s: "
                f"{type(error).__name__}: {error}. Liquidus value above is still valid."
            )
            return record

        if "All" not in xtal_result:
            xtal_result = next(iter(xtal_result.values()), {})

        t_solidus, solidus_note = find_solidus_from_path(xtal_result)
        if t_solidus is not None:
            record["T_solidus_C"] = round(t_solidus, 2)
            record["melting_interval_C"] = round(record["T_liquidus_C"] - t_solidus, 2)
        if solidus_note != "ok":
            record["status"] = "solidus_incomplete"
            record["note"] += solidus_note + ". "

    except Exception as error:
        record["status"] = "failed"
        record["note"] += f"{type(error).__name__}: {error}"
    finally:
        record["elapsed_s"] = round(time.time() - start, 1)

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
        print(
            f"    status={record['status']}  T_liquidus_C={record['T_liquidus_C']}  "
            f"T_solidus_C={record['T_solidus_C']}  elapsed={record['elapsed_s']}s"
        )

    results_df = pd.DataFrame(results)
    results_csv = OUTPUT_DIR / "melting_ranges_report.csv"
    results_df.to_csv(results_csv, index=False, encoding="utf-8")

    counts = results_df["status"].value_counts().to_dict()

    report_lines = [
        "MELTS BATCH MELTING RANGE REPORT",
        "=================================",
        f"Model: {MELTS_MODEL}   Pressure: {PRESSURE_BAR} bar   Fe3+/FeT: {FE3_FET_LIQ}",
        f"Trace H2O added per sample: {TRACE_H2O_WT_PCT} wt% (stabilises solver, avoids phase flip-flop)",
        f"Cooling scan: liquidus -> max({T_END_MIN_C}, liquidus - {MAX_UNDERCOOLING_C}) degC, "
        f"step {DT_C} degC, melting-start threshold = {LIQUID_FRACTION_ONSET:.0%} liquid",
        f"Timeout: solidus sweep {ISOBARIC_TIMEOUT_S}s (findLiq_multi has no timeout kwarg)",
        f"Total compositions: {total}   Status counts: {counts}",
        "",
    ]
    for _, r in results_df.iterrows():
        report_lines.append(
            f"{r['Sample_ID']:<8} | status={r['status']:<18} | "
            f"T_solidus={r['T_solidus_C']:>8} C | T_liquidus={r['T_liquidus_C']:>8} C | "
            f"interval={r['melting_interval_C']:>7} C | liquidus phase={r['liquidus_phase']} | "
            f"{r['elapsed_s']}s"
        )
        if r["excluded_oxides"]:
            report_lines.append(f"           excluded oxides: {r['excluded_oxides']}")
        if r["note"]:
            report_lines.append(f"           note: {r['note']}")

    report_txt = OUTPUT_DIR / "melting_ranges_report.txt"
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nDone. Status counts: {counts}")
    print(f"CSV report:  {results_csv.resolve()}")
    print(f"Text report: {report_txt.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
