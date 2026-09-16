"""
MELTS phase diagram batch calculator driven by a CSV composition table.

Turnkey, standalone script. Reads a CSV of bulk compositions (one row per
composition point, oxide wt% in columns), runs a 1-atm temperature scan with
alphaMELTS/pyMELTScalc (petthermotools) for each row, and writes summary,
scan, settings-log and raw-vs-solver-composition CSV outputs.

Usage:
    Put your composition CSV next to this script (or set INPUT_CSV_PATH),
    then run:
        python melts_csv_batch_calc.py

IMPORTANT CONTEXT FOR REFRACTORY / KILN-FURNITURE USERS
--------------------------------------------------------
MELTS and pMELTS are petrology tools. Their liquid thermodynamic model was
regressed on natural igneous liquids (basalt to rhyolite bulk compositions,
roughly 45-77 wt% SiO2, with Al2O3 typically well under 25 wt%). They were
never fitted against high-alumina corundum refractories, spinel/magnesia
refractories, zirconia refractories, or SiC/Si3N4-bonded refractories. See
the long explanation block near the bottom of this file (function
`print_refractory_applicability_notes`) for what this means for materials
such as Rath K91S (corundum, ~91% Al2O3) or Saint-Gobain Refrax 20
(nitride-bonded SiC): use the numbers as a qualitative, order-of-magnitude
screening tool only, not as a validated liquidus prediction, and consider a
CALPHAD oxide database (FactSage FToxid/FTmisc, Thermo-Calc TCOX, or a
public SiO2-Al2O3-CaO-MgO-oxide assessment you script through pycalphad) as
the quantitative follow-up.
"""

from pathlib import Path
import csv
from itertools import count
import multiprocessing
import re
import shutil
import time

import numpy as np
import pandas as pd
import petthermotools as ptt


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
# Every setting below has a short "what it does" / "why it matters" note.
# These same notes are written out to melts_csv_batch_settings.csv on every
# run, next to the numeric value actually used, so you always have a record
# of exactly what produced a given summary/scan file.

INPUT_CSV_PATH = Path("compositions.csv")
# Path to your composition table. One row per composition point. Columns are
# oxide wt% (see SUPPORTED_OXIDES). Any other column is carried through to
# the outputs for bookkeeping but is NOT sent to MELTS and NOT included in
# the normalisation (see the normalisation explanation below).

MELTS_MODEL = "MELTSv1.0.2"
# Which parameterisation of the Ghiorso/Sack liquid model to use.
#   "MELTSv1.0.2"  = classic MELTS (Ghiorso & Sack, 1995; Asimow & Ghiorso,
#                    1998). Calibrated mainly on MORB/OIB and other mafic to
#                    intermediate natural magmas. This is the closest of the
#                    available options to "general purpose", but it is still
#                    a basalt-calibrated model.
#   "pMELTS"       = re-calibrated for peridotite/mantle bulk compositions,
#                    valid roughly 1000-2500 degC and 1-3 GPa. Not relevant
#                    to 1 atm refractory work.
#   "Rhyolite-MELTS" (1.0.2/1.1.0/1.2.0 via petthermotools Model= strings)
#                    = re-calibrated with better treatment of silicic,
#                    high-SiO2 hydrous liquids (granites/rhyolites) and
#                    quartz/feldspar/biotite/hornblende saturation.
# None of these were fit against corundum-, spinel-, magnesia-, zirconia- or
# SiC/Si3N4-based refractory chemistries, so changing this setting will not
# by itself make the tool "refractory-aware" -- see the applicability notes
# further down. Kept at classic MELTS here because it has the broadest solid
# phase list (feldspar, pyroxenes, olivine, spinel, garnet, oxides) which is
# useful for flagging which impurity/bond-phase minerals could plausibly
# crystallise from a silicate liquid derived from your refractory's minor
# oxide load.

PRESSURE_BAR = 1.0
# Total pressure of the calculation. 1 bar = ambient atmospheric pressure,
# correct for an open kiln/furnace chamber. Do not confuse with local
# mechanical contact stress at the pusher-plate/brick interface: MELTS
# pressure is a thermodynamic state variable (it shifts phase boundaries by
# tens of MPa to GPa), not a stress or abrasive load term. Abrasive wear
# itself is not a MELTS output at all (see applicability notes).

FE3_FET_LIQ = 0.80
# Fixed Fe3+/(Fe3+ + Fe2+) ratio imposed on the liquid. Because you did not
# ask MELTS to buffer to an oxygen fugacity curve (e.g. QFM, NNO), this
# single number silences the redox state of the whole scan: every
# temperature point gets the same fixed ratio regardless of what
# equilibrium fO2 that would imply. A value of 0.80 is a fairly OXIDISING
# choice (only 20% of total Fe held as Fe2+), plausible for iron
# contamination sitting in open air at your kiln's operating temperature,
# but it is a modelling assumption, not a measurement. If you know or can
# estimate the furnace atmosphere's pO2 (air-fired vs. reducing/inert kiln
# zones), it is more defensible to run an fO2-buffered calculation (e.g.
# buffer="QFM", buffer_offset=...) instead of a fixed Fe3Fet_Liq, because
# fixed-ratio results at your grain boundaries would otherwise be
# systematically biased if your kiln's actual pO2 differs from what 0.80
# implicitly assumes at each T. For most refractory oxide bulk chemistries
# total Fe is a minor/contamination component (a few tenths to a few wt%
# Fe2O3), so its practical effect on the bulk liquidus of a >85% Al2O3 or
# SiC-based material is small, but it still controls which Fe-bearing
# accessory phases (spinel, rhm-oxide) show up in the phase list.

T_MIN_C = 800.0
T_MAX_C = 1500.0
TEMPERATURE_STEP_C = 20.0
# The 1-atm temperature ladder MELTS is asked to equilibrate at, one
# Gibbs-energy minimisation per step, from T_MIN_C up to T_MAX_C. See
# "TEMPERATURE RANGE VALIDITY" below for where this range is/is not
# trustworthy. Coarser TEMPERATURE_STEP_C runs faster but the milestone
# temperatures (T01...T99, see below) are linearly interpolated between
# grid points, so a coarse step under-resolves a narrow melting interval.
# For refractory bond-phase melting (often a narrow eutectic-like interval)
# 25 degC steps are more defensible than 50 degC; halve the step if the
# milestone table below looks like it is jumping in big increments.

PHASE_DIAGRAM_CORES = 2
# Number of parallel alphaMELTS worker processes petthermotools spawns for
# the temperature scan of ONE composition. Increase towards your CPU core
# count to speed up a single composition's temperature ladder; it does not
# parallelise across different CSV rows (each row is still processed one at
# a time in the composition loop further down).

PATH_H2O_WT_PCT = 0.001
# A deliberately tiny nominal H2O added to every bulk composition before
# normalisation, independent of whatever H2O the CSV actually specifies.
# Reason: MELTS' Gibbs-energy solver is numerically much better behaved with
# a nonzero (even trace) H2O component present, because several of its
# internal derivative terms are ill-conditioned at exactly zero H2O. This
# is a numerical stabiliser, not a physical statement that your refractory
# contains 0.001 wt% water. If your material and process genuinely involve
# bound/adsorbed water (e.g. green/unfired castable before burn-out), put
# the real value in the CSV's H2O column and this constant will simply be
# used as an additional trace addition on top of it, since it is applied by
# overwriting the H2O key with this constant rather than adding to it (see
# `calculation_composition["H2O"] = PATH_H2O_WT_PCT` in `main`). Change this
# line to add-and-keep-CSV-value behaviour if you need to model real
# hydration/humidity effects.

FIRST_LIQUID_FRACTION = 1.0e-4
# Liquid mass fraction (0.01%) used to define "first liquid appears" for the
# first_liquid_boundary_degC column. This is an incipient-melting /
# sub-solidus-wetting indicator, i.e. the earliest temperature at which the
# solver reports any liquid at all, which is the number most relevant to
# creep, hot-strength loss and grain-boundary wetting/lubrication failure
# modes in refractories (liquid films at grain boundaries reduce strength
# and promote abrasive material loss long before the bulk is molten).

PRACTICAL_LIQUIDUS_FRACTION = 0.99
# Liquid mass fraction defining the "practical liquidus" (T99), i.e. the
# temperature above which the material is essentially fully molten
# (<1% residual solids). Distinguish this from the strict thermodynamic
# liquidus (100% liquid, FULL_LIQUID_TOLERANCE below); 99% is a pragmatic
# engineering cutoff because the last traces of a high-melting refractory
# phase can persist over a very long tail in T.

FULL_LIQUID_TOLERANCE = 1.0e-6
# How close to exactly 100% liquid (1 - this value) must be reached before
# a temperature is reported as "full_liquid_grid_boundary_degC". Kept very
# tight (six nines) deliberately; loosen it only if your scan never reaches
# full liquid within T_MAX_C and you want an approximate value anyway.

LIQUID_FRACTION_MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
# The liquid-fraction checkpoints reported as T01...T99 in the summary and
# scan CSVs. These describe the shape of the melting interval (narrow
# eutectic-like vs. broad mushy-zone-like), which matters for how abruptly
# a refractory loses load-bearing strength as it approaches its working
# temperature limit.

REPORTING_PHASE_FRACTION_MIN = 1.0e-4
# Solid/liquid phases below this mass fraction (0.01%) are treated as
# numerical noise and left out of the human-readable phase assemblage text
# columns, purely to keep those columns readable.

MAX_CALCULATIONS = 500
# Safety cap on the number of CSV rows processed in one run, to stop an
# accidentally huge input file from launching an unbounded number of
# alphaMELTS subprocesses.

OUTPUT_DIRECTORY = Path("melts_csv_batch_results")
TBL_OUTPUT_DIRECTORY = Path("old")
SUMMARY_FILE = OUTPUT_DIRECTORY / "melts_csv_batch_summary.csv"
SCAN_FILE = OUTPUT_DIRECTORY / "melts_csv_batch_scan.csv"
SETTINGS_LOG_FILE = OUTPUT_DIRECTORY / "melts_csv_batch_settings_log.csv"
COMPOSITION_AUDIT_FILE = OUTPUT_DIRECTORY / "melts_csv_batch_composition_audit.csv"
# SETTINGS_LOG_FILE: one row per setting used in this run, with its value and
#   the same explanation text as the comments above, so a run's output
#   folder is self-documenting even months later.
# COMPOSITION_AUDIT_FILE: one row per composition point comparing (a) the
#   raw oxide wt% you supplied in the CSV, (b) the calculation input after
#   the PATH_H2O_WT_PCT override, and (c) the normalised bulk composition
#   actually handed to the MELTS solver. See `build_composition_audit_record`
#   and the "NORMALISATION" explanation below for why (b) and (c) are
#   generally not identical to (a) even when your raw row already sums to
#   100 wt%.

MOLAR_MASS_FEO = 71.844
MOLAR_MASS_FE2O3 = 159.688

# Oxides that MELTS bulk composition (Liq endmember set) actually accepts.
# Any CSV column not in this list is reported but excluded from the MELTS
# bulk composition and from normalisation. Note for refractory work: ZrO2
# and Cr2O3 (common in zirconia-mullite and chrome-spinel refractories) are
# NOT in this list, i.e. MELTS' liquid model has no endmember for them.
# If your material relies on ZrO2 or Cr2O3 for its melting behaviour (e.g.
# Rath Z95S), the MELTS bulk composition silently omits that chemistry and
# the resulting liquidus/solidus is not representative of the real material.
SUPPORTED_OXIDES = (
    "SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO",
    "CaO", "Na2O", "K2O", "P2O5", "H2O", "CO2",
)

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

# Settings registry used to build the self-documenting settings log CSV.
# Add an entry here whenever you add a new tunable constant above.
SETTINGS_REGISTRY = [
    ("INPUT_CSV_PATH", lambda: str(INPUT_CSV_PATH),
     "Path to the input composition CSV (oxide wt% per row)."),
    ("MELTS_MODEL", lambda: MELTS_MODEL,
     "Liquid thermodynamic model/calibration used (MELTSv1.0.2 = classic "
     "Ghiorso and Sack MELTS, calibrated on natural mafic-intermediate "
     "magmas; not fit to refractory oxide systems)."),
    ("PRESSURE_BAR", lambda: PRESSURE_BAR,
     "Total pressure of the calculation in bar; 1 bar = open-atmosphere "
     "kiln condition. Not a mechanical/abrasive stress term."),
    ("FE3_FET_LIQ", lambda: FE3_FET_LIQ,
     "Fixed Fe3+/total-Fe ratio imposed on the liquid at every T step "
     "(0.80 = oxidising assumption); not fO2-buffered."),
    ("T_MIN_C", lambda: T_MIN_C, "Lowest temperature in the 1 atm scan, degC."),
    ("T_MAX_C", lambda: T_MAX_C, "Highest temperature in the 1 atm scan, degC."),
    ("TEMPERATURE_STEP_C", lambda: TEMPERATURE_STEP_C,
     "Grid spacing of the temperature scan, degC; milestones are linearly "
     "interpolated between grid points, so a coarser step blurs sharp "
     "melting intervals."),
    ("PHASE_DIAGRAM_CORES", lambda: PHASE_DIAGRAM_CORES,
     "Parallel alphaMELTS worker processes used per composition's "
     "temperature scan."),
    ("PATH_H2O_WT_PCT", lambda: PATH_H2O_WT_PCT,
     "Trace H2O (wt%) force-set on every bulk composition before "
     "normalisation, purely to keep the solver numerically well "
     "conditioned; overwrites (does not add to) any CSV H2O value."),
    ("FIRST_LIQUID_FRACTION", lambda: FIRST_LIQUID_FRACTION,
     "Liquid mass fraction defining 'first liquid appears' (incipient "
     "melting / grain-boundary wetting onset)."),
    ("PRACTICAL_LIQUIDUS_FRACTION", lambda: PRACTICAL_LIQUIDUS_FRACTION,
     "Liquid mass fraction defining the pragmatic 'practical liquidus' "
     "(T99, <1% residual solids)."),
    ("FULL_LIQUID_TOLERANCE", lambda: FULL_LIQUID_TOLERANCE,
     "How close to 100% liquid must be reached to report the strict "
     "thermodynamic liquidus."),
    ("LIQUID_FRACTION_MILESTONES", lambda: str(LIQUID_FRACTION_MILESTONES),
     "Liquid-fraction checkpoints (T01...T99) describing the shape of the "
     "melting interval."),
    ("REPORTING_PHASE_FRACTION_MIN", lambda: REPORTING_PHASE_FRACTION_MIN,
     "Minimum phase mass fraction shown in the human-readable phase "
     "assemblage text (numerical noise cutoff)."),
    ("MAX_CALCULATIONS", lambda: MAX_CALCULATIONS,
     "Safety cap on number of CSV rows processed per run."),
]


# ---------------------------------------------------------------------------
# COMPOSITION LOADING
# ---------------------------------------------------------------------------

def load_compositions_from_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Composition CSV not found: {path}")

    table = pd.read_csv(path)
    table.columns = [str(column).strip() for column in table.columns]

    unsupported_columns = [column for column in table.columns if column not in SUPPORTED_OXIDES]
    if unsupported_columns:
        offending = table[unsupported_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        nonzero = offending.columns[(offending.abs() > 0.0).any(axis=0)].tolist()
        if nonzero:
            print(
                "WARNING: columns "
                + ", ".join(nonzero)
                + " are not part of the MELTS liquid oxide set and contain "
                  "nonzero values. They will be reported in the output CSVs "
                  "but excluded from the MELTS bulk composition and from "
                  "normalisation."
            )
        else:
            print(
                "Note: columns "
                + ", ".join(unsupported_columns)
                + " are not part of the MELTS liquid oxide set; all values "
                  "are zero, so they are dropped from the calculation."
            )

    compositions = []
    extra_columns = {}
    for _, row in table.iterrows():
        composition = {}
        extras = {}
        for column in table.columns:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            value = 0.0 if pd.isna(value) else float(value)
            if column in SUPPORTED_OXIDES:
                composition[column] = value
            else:
                extras[column] = value
        for oxide in SUPPORTED_OXIDES:
            composition.setdefault(oxide, 0.0)
        compositions.append(composition)
        for name, value in extras.items():
            extra_columns.setdefault(name, []).append(value)

    return compositions, extra_columns


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
    """Build the MELTS Liq-endmember bulk dict and normalise it to sum to
    exactly 100 wt%.

    NORMALISATION -- why it happens even if your CSV row already sums to
    100 wt%:
      1. Unsupported oxides are dropped first. If your raw analysis
         includes ZrO2, Cr2O3, SO3, or any other oxide outside
         SUPPORTED_OXIDES, those wt% are removed from the total before this
         function ever sees the row, so the remaining supported oxides no
         longer sum to 100 even though your original row did.
      2. FeO and Fe2O3 are collapsed into a single FeOt_Liq value using the
         stoichiometric factor 2*M(FeO)/M(Fe2O3) (~0.8998). This factor is
         not 1, so replacing two numbers (FeO wt%, Fe2O3 wt%) with one
         (FeOt wt%) changes the arithmetic total by a small amount whenever
         Fe2O3 is nonzero, even if FeO+Fe2O3 was "correct" in the original
         analysis.
      3. PATH_H2O_WT_PCT is force-set on the calculation composition before
         this function runs (see `main`), adding mass that was not counted
         in your original 100%.
      4. Floating point rounding in the CSV itself (e.g. 99.98 or 100.03
         due to rounded lab data) is common and otherwise silently biases
         the phase fractions MELTS reports, because MELTS' internal mass
         balance and the mass-fraction quantities this script derives
         (liquid_fraction = mass_g_Liq / mass_g) are only meaningful
         relative to a bulk that sums to exactly 100.
    In short: normalising to exactly 100 wt% right before the solve is
    necessary to keep the reported phase mass fractions physically
    consistent, regardless of how close to 100 your raw CSV values already
    are. The COMPOSITION_AUDIT_FILE written by this script records the
    raw values, the pre-normalisation calculation input, and the final
    normalised bulk side by side so you can see exactly how large this
    correction was for each row.
    """
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
    normalised = {oxide: 100.0 * value / total for oxide, value in bulk.items()}
    return normalised, total


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
        destination = TBL_OUTPUT_DIRECTORY / f"csv_{composition_id:04d}_{source.name}"
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


def first_reported_liquid_temperature(milestones):
    """Walk the milestone fractions from lowest to highest (T01, T10, T25, ...)
    and return the temperature and label of the first one that was actually
    bracketed in the scan. This is the plain-language "temperature at which
    I first get liquid" the milestone table already computed, just picked
    out and put in its own column instead of making you scan the row."""
    for fraction in LIQUID_FRACTION_MILESTONES:
        label = f"T{int(round(fraction * 100)):02d}_degC"
        value = milestones.get(label)
        if value is not None:
            return value, label
    return None, None


def phase_text_near_temperature(scan, solid_columns, temperature):
    index = (scan["temperature_degC"] - temperature).abs().idxmin()
    return phase_text(scan.loc[index], solid_columns)


def make_summary(composition_id, composition, extras, scan, solid_columns):
    milestones = {
        f"T{int(round(fraction * 100)):02d}_degC": interpolate_crossing(scan, fraction)
        for fraction in LIQUID_FRACTION_MILESTONES
    }
    first_liquid = interpolate_crossing(scan, FIRST_LIQUID_FRACTION)
    first_reported_temperature, first_reported_label = first_reported_liquid_temperature(milestones)
    t99 = milestones["T99_degC"]
    full_liquid = interpolate_crossing(scan, 1.0 - FULL_LIQUID_TOLERANCE)

    return {
        "composition_id": composition_id,
        **{oxide: composition.get(oxide, 0.0) for oxide in SUPPORTED_OXIDES},
        **extras,
        "raw_oxide_total_wt_pct": sum(composition.values()) + sum(extras.values()),
        "first_liquid_boundary_degC": first_liquid,
        "scan_min_liquid_fraction": float(scan["liquid_fraction"].iloc[0]),
        **milestones,
        "first_liquid_temperature_degC": first_reported_temperature,
        "first_liquid_temperature_milestone": first_reported_label,
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


def make_scan_records(composition_id, composition, extras, scan, solid_columns):
    milestones = {
        f"T{int(round(fraction * 100)):02d}_degC": interpolate_crossing(scan, fraction)
        for fraction in LIQUID_FRACTION_MILESTONES
    }
    first_reported_temperature, first_reported_label = first_reported_liquid_temperature(milestones)

    records = []
    for _, row in scan.iterrows():
        records.append({
            "composition_id": composition_id,
            **{oxide: composition.get(oxide, 0.0) for oxide in SUPPORTED_OXIDES},
            **extras,
            "temperature_degC": float(row["temperature_degC"]),
            "pressure_bar": float(row["pressure_bar"]),
            "liquid_fraction": float(row["liquid_fraction"]),
            "solid_fraction": float(row["solid_fraction"]),
            "first_liquid_temperature_degC": first_reported_temperature,
            "first_liquid_temperature_milestone": first_reported_label,
            "stable_phases": phase_text(row, solid_columns),
        })
    return records


def build_composition_audit_record(composition_id, raw_composition, calc_composition, bulk, raw_total, extras):
    """One row comparing what you gave the script (raw_composition, straight
    from the CSV), what was actually fed to the normaliser (calc_composition,
    after the PATH_H2O_WT_PCT override), and what MELTS actually solved for
    (bulk, normalised to 100 wt% on the Liq_ endmember names). FeO/Fe2O3 are
    shown both as given and collapsed into the single FeOt equivalent MELTS
    uses, so you can see the small mass-balance shift that collapse causes.
    """
    record = {"composition_id": composition_id}
    for oxide in SUPPORTED_OXIDES:
        if oxide in ("FeO", "Fe2O3"):
            continue
        record[f"raw_{oxide}_wt_pct"] = raw_composition.get(oxide, 0.0)
        record[f"calc_input_{oxide}_wt_pct"] = calc_composition.get(oxide, 0.0)
        record[f"melts_solved_{oxide}_wt_pct"] = bulk.get(f"{oxide}_Liq", 0.0)

    record["raw_FeO_wt_pct"] = raw_composition.get("FeO", 0.0)
    record["raw_Fe2O3_wt_pct"] = raw_composition.get("Fe2O3", 0.0)
    record["raw_FeOt_equivalent_wt_pct"] = convert_fe_oxides_to_feot(raw_composition)
    record["calc_input_FeOt_equivalent_wt_pct"] = convert_fe_oxides_to_feot(calc_composition)
    record["melts_solved_FeOt_Liq_wt_pct"] = bulk.get("FeOt_Liq", 0.0)

    for name, value in extras.items():
        record[f"excluded_from_melts_{name}_wt_pct"] = value

    record["raw_oxide_sum_wt_pct"] = sum(raw_composition.values())
    record["calc_input_sum_before_normalisation_wt_pct"] = raw_total
    record["normalisation_factor_applied"] = (
        100.0 / raw_total if raw_total not in (0.0, None) else None
    )
    record["excluded_oxide_sum_wt_pct"] = sum(extras.values()) if extras else 0.0
    return record


def print_summary_table(summaries):
    columns = [
        "composition_id", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "first_liquid_boundary_degC", "scan_min_liquid_fraction",
        "T01_degC", "T10_degC", "T25_degC", "T50_degC", "T75_degC",
        "T90_degC", "T95_degC", "T99_degC",
        "first_liquid_temperature_degC", "practical_liquidus_degC",
        "full_liquid_grid_boundary_degC",
    ]
    headers = [
        "Point", "SiO2", "Al2O3", "Fe2O3", "MgO", "CaO",
        "First liquid", "Min liquid", "T01", "T10", "T25", "T50", "T75",
        "T90", "T95", "T99", "First liquid T", "Practical liquidus",
        "Full liquid",
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

    print("\nMELTS CSV BATCH SUMMARY")
    print(separator)
    print("|" + "|".join(f" {header:^{width}} " for header, width in zip(headers, widths)) + "|")
    print(separator)
    for row in rows:
        print("|" + "|".join(f" {value:>{width}} " for value, width in zip(row, widths)) + "|")
    print(separator)
    print("Composition values are raw input oxide wt%. Temperatures are degC.")
    print(
        "First liquid T is the temperature of the lowest liquid-fraction milestone "
        "(T01, T10, T25, ...) that was actually bracketed in the scan; the "
        "'first_liquid_temperature_milestone' column in the CSV names which one."
    )
    print("If first liquid is not found, use the reported T25, T50, T75, T90, T95, or T99 values.")


def print_temperature_range_notes():
    print("\nTEMPERATURE RANGE VALIDITY")
    print(
        f"This run scans T_MIN_C={T_MIN_C:.0f} degC to T_MAX_C={T_MAX_C:.0f} degC. "
        "Published guidance for the MELTS/pMELTS liquid model: nominal calibrated "
        "range is about 500-2000 degC and 0-2 GPa, but the practical lower limit "
        "where the solver behaves reliably is higher than that nominal floor. "
        "Caltech's alphaMELTS documentation states the standard MELTS liquid "
        "model is only intended for T >= ~773 K (~500 degC), yet also warns that "
        "at the low-temperature, near-subsolidus end, solid phases with cation "
        "ordering (spinel, orthopyroxene) frequently cause convergence problems "
        "or hangs. In practice this means results below roughly 800-900 degC "
        "should be treated as qualitative at best: the solver can converge to a "
        "spurious local minimum, oscillate between phase assemblages step to "
        "step, or simply fail silently for one row while succeeding for a "
        "neighbouring composition. For refractory bond-phase melting studies "
        "the temperatures of real interest (liquid films forming at grain "
        "boundaries in a >1300 degC service brick) are usually comfortably above "
        "this problem zone, so keeping T_MIN_C at 800-900 degC or higher (rather "
        "than 700 degC) will generally give more trustworthy low-T scan points, "
        "at the cost of not being able to see any spurious low-T liquid the "
        "model might otherwise report."
    )


def print_refractory_applicability_notes():
    print("\nAPPLICABILITY TO REFRACTORY / KILN-FURNITURE MATERIALS")
    print(
        "MELTS and pMELTS are geological tools: their liquid activity model was "
        "regressed against natural terrestrial magma compositions (basalt to "
        "rhyolite), essentially always below about 20-25 wt% Al2O3 and with "
        "SiO2 in the 45-77 wt% range. Two materials you mentioned sit outside "
        "that calibration space in different ways:\n"
        "  - Rath K91S is a corundum-based sliding-gate refractory, roughly "
        "90-91 wt% Al2O3 with the balance mostly SiO2/CaO/MgO impurity or "
        "bond-phase oxides. That bulk composition is an extreme high-Al2O3, "
        "low-alkali corner MELTS never saw during calibration; the classic "
        "MELTS solid list does include a corundum-type phase, so the tool will "
        "run and give a number, but the accuracy of the predicted liquidus for "
        "a >85 wt% Al2O3 bulk is unverified and should be treated as a rough, "
        "qualitative screen only, best used to compare relative shifts between "
        "your own compositions rather than as an absolute liquidus temperature.\n"
        "  - Saint-Gobain Refrax 20 is a nitride-bonded silicon carbide (SiC "
        "grains held together by a Si3N4 binder phase). SiC and Si3N4 are not "
        "part of the MELTS oxide/liquid component set at all: there is no way "
        "to represent Si-C or Si-N bonding in this framework. If you enter an "
        "oxide-only analysis for Refrax 20 (e.g. only its minor SiO2/Al2O3/"
        "Fe2O3 impurity or oxidation-scale oxides) MELTS will happily compute a "
        "liquidus for THAT ACCESSORY OXIDE PHASE, but that is not a melting "
        "point of the SiC/Si3N4 refractory itself; it is at best informative "
        "about the oxidation scale/bond glass that could form on its surface "
        "in service. Do not report a MELTS liquidus for Refrax 20 as if it "
        "described the bulk material.\n"
        "General recommendations for your abrasion/kiln-furniture failure "
        "work: (1) use MELTS results as a fast, free, order-of-magnitude "
        "screening tool for oxide impurity/bond-phase melting (e.g. estimating "
        "when a silicate glassy phase from contamination or slag ingress would "
        "start wetting grain boundaries and softening the microstructure, which "
        "is a genuine and relevant abrasion-adjacent failure mode); (2) do not "
        "rely on it for absolute liquidus/solidus values of pure corundum, "
        "spinel, magnesia, zirconia, or SiC/Si3N4 refractories; (3) for a "
        "quantitative answer on those systems, use a CALPHAD oxide database "
        "purpose-built for ceramics/refractories (FactSage FToxid/FTmisc, "
        "Thermo-Calc TCOX, or an assessed SiO2-Al2O3-CaO-MgO-(ZrO2, Cr2O3) "
        "database you can script with pycalphad, which you already use); "
        "(4) remember MELTS has no concept of abrasive/mechanical wear at all, "
        "it only tells you phase stability and liquid fraction as a function of "
        "T; wear rate itself needs a separate tribological model that takes the "
        "softened/liquid fraction MELTS predicts as one input among others "
        "(hardness loss, contact stress, sliding velocity, atmosphere)."
    )


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf_8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def write_settings_log():
    records = [
        {"setting": name, "value": getter(), "explanation": explanation}
        for name, getter, explanation in SETTINGS_REGISTRY
    ]
    write_csv(SETTINGS_LOG_FILE, records)


def failed_summary(composition_id, composition, extras, error):
    return {
        "composition_id": composition_id,
        **{oxide: composition.get(oxide, 0.0) for oxide in SUPPORTED_OXIDES},
        **extras,
        "raw_oxide_total_wt_pct": sum(composition.values()) + sum(extras.values()),
        "first_liquid_boundary_degC": None,
        "scan_min_liquid_fraction": None,
        **{f"T{int(round(fraction * 100)):02d}_degC": None for fraction in LIQUID_FRACTION_MILESTONES},
        "first_liquid_temperature_degC": None,
        "first_liquid_temperature_milestone": None,
        "practical_liquidus_degC": None,
        "full_liquid_grid_boundary_degC": None,
        "scan_max_liquid_fraction": None,
        "first_liquid_phase_assemblage": f"FAILED: {type(error).__name__}: {error}",
        "full_liquid_phase_assemblage": "Calculation failed",
    }


def main():
    compositions, extra_columns = load_compositions_from_csv(INPUT_CSV_PATH)
    temperatures = temperature_values()

    if len(compositions) > MAX_CALCULATIONS:
        raise ValueError(f"Input CSV contains {len(compositions)} rows, above MAX_CALCULATIONS.")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nMELTS CSV COMPOSITION BATCH")
    print(f"Input CSV: {INPUT_CSV_PATH}")
    print(f"MELTS model: {MELTS_MODEL}")
    print(f"Pressure: {PRESSURE_BAR:.5f} bar")
    print(f"Fe3+ to total Fe: {FE3_FET_LIQ:.5f}")
    print(f"Composition points: {len(compositions)}")
    print(f"Temperature range: {T_MIN_C:.2f} to {T_MAX_C:.2f} degC")
    print(f"Temperature step: {TEMPERATURE_STEP_C:.2f} degC")
    print(f"Temperature points per composition: {len(temperatures)}")

    summaries = []
    scan_records = []
    audit_records = []

    for composition_id, composition in zip(count(1), compositions):
        extras = {name: values[composition_id - 1] for name, values in extra_columns.items()}

        print("\n" + "=" * 76)
        print(f"COMPOSITION POINT {composition_id} OF {len(compositions)}")
        print("=" * 76)
        print(" ".join(f"{oxide}={value:.3f}" for oxide, value in composition.items() if value != 0.0))
        if extras:
            print("(excluded from MELTS bulk) " + " ".join(f"{name}={value:.3f}" for name, value in extras.items()))

        calculation_composition = dict(composition)
        calculation_composition["H2O"] = PATH_H2O_WT_PCT
        snapshot = tbl_snapshot()
        start_time_ns = time.time_ns()

        try:
            bulk, raw_total = create_melts_bulk(calculation_composition)
            audit_records.append(
                build_composition_audit_record(
                    composition_id, composition, calculation_composition, bulk, raw_total, extras
                )
            )
            result = ptt.phaseDiagram_calc(
                cores=PHASE_DIAGRAM_CORES,
                Model=MELTS_MODEL,
                bulk=bulk,
                T_C=temperatures,
                P_bar=np.array([PRESSURE_BAR], dtype=float),
                Fe3Fet_Liq=FE3_FET_LIQ,
            )
            scan, solid_columns = prepare_scan(result)
            summaries.append(make_summary(composition_id, composition, extras, scan, solid_columns))
            scan_records.extend(make_scan_records(composition_id, composition, extras, scan, solid_columns))
        except Exception as error:
            summaries.append(failed_summary(composition_id, composition, extras, error))
            print(f"Calculation failed: {type(error).__name__}: {error}")
        finally:
            move_new_tbl_files(snapshot, start_time_ns, composition_id)

    print_summary_table(summaries)
    print_temperature_range_notes()
    print_refractory_applicability_notes()

    write_csv(SUMMARY_FILE, summaries)
    write_csv(SCAN_FILE, scan_records)
    write_csv(COMPOSITION_AUDIT_FILE, audit_records)
    write_settings_log()

    print("\nCSV files written")
    print(f" {SUMMARY_FILE}")
    print(f" {SCAN_FILE}")
    print(f" {COMPOSITION_AUDIT_FILE}  (raw CSV oxides vs. normalised MELTS solver input)")
    print(f" {SETTINGS_LOG_FILE}  (every setting used in this run, with explanations)")
    print(f"New alphaMELTS tbl files, if produced, were moved to {TBL_OUTPUT_DIRECTORY}.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
