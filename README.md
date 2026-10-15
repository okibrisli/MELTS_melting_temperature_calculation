# MELTS Melting Temperature Calculation

Python scripts that use **alphaMELTS for Python** through
[`petthermotools`](https://github.com/gleesonm1/PetThermoTools) (pyMELTScalc) to
estimate the melting behaviour of oxide compositions: the first appearance of
liquid, the temperature at fixed liquid fractions (T01 to T99), the practical and
full liquidus, and the stable phase assemblage at each temperature.

You can run one composition, a list of compositions from a CSV file, or a grid
of compositions built from oxide ranges.

> **Applicability warning.** MELTS was calibrated on natural igneous liquids
> (basalt to rhyolite, about 45–77 wt% SiO2 and usually under 25 wt% Al2O3). It
> was never fitted to high-alumina, spinel, magnesia, zirconia or SiC/Si3N4
> refractories. Treat results for such materials as a qualitative screen for
> comparing compositions, not as validated liquidus temperatures. ZrO2, HfO2,
> Cr2O3 and other oxides outside the MELTS oxide set are **ignored** by the
> solver. For quantitative refractory work, use a CALPHAD oxide database
> (FactSage FToxid, Thermo-Calc TCOX, or pycalphad with an assessed database).

---

## Contents

- [Requirements](#requirements)
- [Scripts at a glance](#scripts-at-a-glance)
- [How a calculation works](#how-a-calculation-works)
- [Script reference](#script-reference)
  - [melt_PD_csv_comp_final.py — CSV batch phase-diagram scan (recommended)](#melt_pd_csv_comp_finalpy--csv-batch-phase-diagram-scan-recommended)
  - [melts_csv_batch_calc_with_findliq.py — CSV batch + direct liquidus](#melts_csv_batch_calc_with_findliqpy--csv-batch--direct-liquidus)
  - [melts_PD_final.py — single composition temperature scan](#melts_pd_finalpy--single-composition-temperature-scan)
  - [melts_PD_grid_final.py — composition grid temperature scan](#melts_pd_grid_finalpy--composition-grid-temperature-scan)
  - [2_1.py — CSV composition scan (older variant)](#2_1py--csv-composition-scan-older-variant)
  - [melts_liquidus_single_final.py — single composition direct liquidus](#melts_liquidus_single_finalpy--single-composition-direct-liquidus)
  - [run_melts_grid.py — composition grid of direct liquidus runs](#run_melts_gridpy--composition-grid-of-direct-liquidus-runs)
  - [Helper modules](#helper-modules)
- [Input CSV format](#input-csv-format)
- [Output files and columns](#output-files-and-columns)
- [Choosing settings](#choosing-settings)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

---

## Requirements

- Python 3.8 or newer
- `numpy`, `pandas`
- `petthermotools`, plus a working **alphaMELTS for Python** installation that
  `petthermotools` can find (follow the PetThermoTools installation guide)

```bash
pip install numpy pandas petthermotools
```

All scripts are configured by editing constants at the top of the file. There
are no command-line arguments. Run each script from the project root, because
paths are relative to the current working directory:

```bash
python melt_PD_csv_comp_final.py
```

---

## Scripts at a glance

| Script | Input | MELTS function | What you get |
|---|---|---|---|
| [melt_PD_csv_comp_final.py](melt_PD_csv_comp_final.py) | CSV, one composition per row | `phaseDiagram_calc` | Summary, full scan, composition audit and settings log per run |
| [melts_csv_batch_calc_with_findliq.py](melts_csv_batch_calc_with_findliq.py) | Same CSV | `findLiq_multi` + `phaseDiagram_calc` | Everything above, plus a direct liquidus and primary phase per row |
| [melts_PD_final.py](melts_PD_final.py) | One composition in the script | `phaseDiagram_calc` | Key findings table and full temperature scan |
| [melts_PD_grid_final.py](melts_PD_grid_final.py) | Oxide ranges in the script | `phaseDiagram_calc` | Summary and scan for every grid point |
| [2_1.py](2_1.py) | CSV | `phaseDiagram_calc` | Summary and scan (older, simpler CSV version) |
| [melts_liquidus_single_final.py](melts_liquidus_single_final.py) | One composition in the script | `findLiq_multi` | Liquidus temperature, primary phase, liquid composition |
| [run_melts_grid.py](run_melts_grid.py) | Oxide ranges in the script | `findLiq_multi` | One liquidus CSV per grid point and a printed summary |

**Two kinds of calculation:**

- **Temperature scan** (`phaseDiagram_calc`): equilibrates the composition at
  every temperature on a ladder at fixed pressure. Gives the liquid fraction
  versus temperature, so you see the whole melting interval.
- **Direct liquidus** (`findLiq_multi`): searches for the temperature at which
  the first solid crystallises on cooling from a fully molten state. Gives one
  liquidus temperature and the primary (first) phase, but no melting interval.

---

## How a calculation works

Every script follows the same steps:

1. **Read the composition** in oxide wt%. Supported oxides are
   `SiO2, TiO2, Al2O3, Fe2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, H2O, CO2`.
2. **Set trace water (scan scripts only).** H2O is **overwritten** with
   `PATH_H2O_WT_PCT` (0.001 wt% by default). This keeps the MELTS solver
   numerically stable; it replaces any H2O value you supplied rather than adding
   to it.
3. **Combine iron.** FeO and Fe2O3 are merged into one total-iron value,
   `FeOt = FeO + Fe2O3 × 2·M(FeO)/M(Fe2O3)` (≈ 0.8998 × Fe2O3). The
   Fe3+/Fe ratio is then set separately by `FE3_FET_LIQ`.
4. **Normalise** the supported oxides to exactly 100 wt%. Unsupported oxides
   (ZrO2, HfO2, Cr2O3, and so on) are dropped before normalising.
5. **Run MELTS** (temperature scan or direct liquidus).
6. **Compute results.** Liquid fraction = liquid mass / total mass at each
   temperature. Milestone temperatures are found by linear interpolation
   between neighbouring scan points.
7. **Write reports** to the console and to CSV files. Any `*_tbl.txt` files
   that alphaMELTS writes to the working folder are moved into `old/`.

If one composition fails, the batch scripts record the error in that row and
carry on with the next one.

---

## Script reference

### melt_PD_csv_comp_final.py — CSV batch phase-diagram scan (recommended)

The most complete script. It reads compositions from a CSV, runs a 1-atm
temperature scan for each row, and writes four self-documenting output files.
After the summary table it also prints notes on the valid temperature range and
on applicability to refractory materials.

#### Settings

| Setting | Default | Description |
|---|---|---|
| `INPUT_CSV_PATH` | `compositions_grid_sweep.csv` | Composition table to read. See [Input CSV format](#input-csv-format). |
| `MELTS_MODEL` | `"MELTSv1.0.2"` | Thermodynamic model. Options:<br>• `"MELTSv1.0.2"`: classic MELTS, calibrated on mafic to intermediate magmas, with the broadest list of solid phases<br>• `"pMELTS"`: peridotite/mantle, about 1–3 GPa, not suited to 1 atm work<br>• Rhyolite-MELTS strings accepted by petthermotools (for example `"MELTSv1.1.0"`, `"MELTSv1.2.0"`): better suited to silicic, hydrous liquids |
| `PRESSURE_BAR` | `1.0` | Total pressure in bar. 1 bar is an open kiln or furnace atmosphere. This is a thermodynamic state variable, not a mechanical contact stress. |
| `FE3_FET_LIQ` | `0.80` | Fixed Fe3+/(Fe3+ + Fe2+) ratio of the liquid, used at every temperature. 0.80 is fairly oxidising (air-fired). Lower it for reducing atmospheres. It mainly affects which Fe-bearing phases (spinel, rhombohedral oxide) appear. |
| `T_MIN_C` | `800.0` | Lowest scan temperature, °C. Below about 800–900 °C the solver may fail to converge or give unreliable results. |
| `T_MAX_C` | `1500.0` | Highest scan temperature, °C. Raise it if the full liquidus is "Not bracketed". |
| `TEMPERATURE_STEP_C` | `100.0` | Spacing of the temperature ladder, °C. `T_MAX_C − T_MIN_C` must be an exact multiple of it. A smaller step resolves narrow melting intervals better (25 °C is a good choice for refractory bond phases) but takes longer. |
| `PHASE_DIAGRAM_CORES` | `1` | Number of parallel alphaMELTS processes used for the temperature points of **one** composition. Rows are still processed one after another. |
| `PATH_H2O_WT_PCT` | `0.001` | Trace H2O in wt%. Replaces the CSV H2O value before normalising, to keep the solver stable. |
| `FIRST_LIQUID_FRACTION` | `1.0e-4` | Liquid fraction (0.01%) that defines "first liquid appears" (`first_liquid_boundary_degC`). This marks the onset of melting and grain-boundary wetting. |
| `PRACTICAL_LIQUIDUS_FRACTION` | `0.99` | Liquid fraction that defines the practical liquidus (less than 1% solid left). It is documented in the settings log. In practice the summary column `practical_liquidus_degC` is taken from the T99 milestone. |
| `FULL_LIQUID_TOLERANCE` | `1.0e-6` | The full liquidus is the temperature where liquid fraction reaches `1 − tolerance`. Loosen it if the scan never quite reaches 100% liquid. |
| `LIQUID_FRACTION_MILESTONES` | `(0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)` | Liquid fractions reported as `T01_degC` … `T99_degC`. Together they describe the shape of the melting interval. You can add or remove values; the column names follow the percentage. **Keep 0.99**, because the summary reads `T99_degC`. |
| `REPORTING_PHASE_FRACTION_MIN` | `1.0e-4` | Phases below this mass fraction are left out of the readable phase-assemblage text. This affects display only. |
| `MAX_CALCULATIONS` | `500` | Safety limit. The run stops if the CSV has more rows than this. |
| `OUTPUT_DIRECTORY` | `melts_csv_batch_results` | Folder for the output CSVs. It is created if missing. |
| `TBL_OUTPUT_DIRECTORY` | `old` | Folder that alphaMELTS `*_tbl.txt` files are moved into, renamed `csv_<id>_<name>`. |
| `SUMMARY_FILE`, `SCAN_FILE`, `SETTINGS_LOG_FILE`, `COMPOSITION_AUDIT_FILE` | inside `OUTPUT_DIRECTORY` | Output file names. See [Output files](#output-files-and-columns). |
| `SUPPORTED_OXIDES` | 13 MELTS oxides | Oxide columns sent to MELTS. Changing this does not add new chemistry to MELTS. |
| `PHASE_LABELS` | dict | Maps MELTS phase codes (such as `cpx`, `sp`) to readable names in the output. Add entries to rename more phases. |
| `SETTINGS_REGISTRY` | list | Settings written to the settings log. Add an entry here when you add a new setting. |

#### Outputs

- `melts_csv_batch_summary.csv`: one row per composition
- `melts_csv_batch_scan.csv`: one row per composition per temperature
- `melts_csv_batch_composition_audit.csv`: your raw CSV values, the values after
  the H2O override, and the normalised bulk actually solved, side by side,
  together with the normalisation factor and the total of excluded oxides
- `melts_csv_batch_settings_log.csv`: every setting used, its value, and an
  explanation, so each results folder records how it was produced

---

### melts_csv_batch_calc_with_findliq.py — CSV batch + direct liquidus

A wrapper around `melt_PD_csv_comp_final.py`. It uses that script's settings
(input CSV, model, pressure, temperatures, outputs) and **also** runs
`ptt.findLiq_multi` for each composition before the temperature scan. This gives
an independent liquidus temperature and the primary crystallising phase.

Change the shared settings in `melt_PD_csv_comp_final.py`. This file adds:

| Setting | Default | Description |
|---|---|---|
| `FIND_LIQUIDUS_ENABLED` | `True` | Run the direct liquidus search for each composition. If `False`, the extra columns read "Disabled". |
| `FIND_LIQUIDUS_RAW_FILE` | `<OUTPUT_DIRECTORY>/melts_csv_batch_findliq_raw.csv` | Unformatted `findLiq_multi` output for every composition, with a `composition_id` column added. |

Extra summary columns: `findliq_status` (Success / Failed / Disabled / Not run),
`findliq_liquidus_degC`, `findliq_primary_phase`, `findliq_phase_identifier`,
`findliq_fluid_saturated`, `findliq_error`.

The console table compares `findLiq` (direct liquidus) with `Full liquid` (the
liquidus estimated from the scan). If the two differ by much, the temperature
step is probably too coarse or `T_MAX_C` is too low.

---

### melts_PD_final.py — single composition temperature scan

Runs a temperature scan for one composition written directly in the script. It
prints every scan point and a **Key findings** table.

| Setting | Default | Description |
|---|---|---|
| `ANALYSED_OXIDE_WT_PCT` | SiO2 77, Al2O3 5, Fe2O3 1, MgO 2, CaO 15 | The composition in oxide wt%. It does not need to sum to 100. |
| `MELTS_MODEL`, `PRESSURE_BAR`, `FE3_FET_LIQ` | as above | See the [CSV batch settings](#settings). |
| `T_MIN_C` / `T_MAX_C` / `TEMPERATURE_STEP_C` | `600` / `1200` / `20` | Temperature ladder, °C. The range must be an exact multiple of the step. |
| `PHASE_DIAGRAM_CORES` | `1` | Parallel alphaMELTS processes. |
| `PATH_H2O_WT_PCT` | `0.001` | Trace H2O override. |
| `PRACTICAL_LIQUIDUS_FRACTION` | `0.99` | Liquid fraction for the "Practical liquidus" finding. |
| `LIQUID_FRACTION_MILESTONES` | 0.01 … 0.99 | Milestones reported as `Txx, liquid fraction`. |
| `REPORTING_PHASE_FRACTION_MIN` | `1.0e-4` | Display cutoff for phases. **In this script it is also the liquid fraction used for "First liquid boundary".** |
| `FULL_LIQUID_TOLERANCE` | `1.0e-6` | Defines the full liquid boundary. |
| `FINDINGS_OUTPUT_FILE` | `old_working_scripts/melts_melting_findings.csv` | Key findings CSV. |
| `SCAN_OUTPUT_FILE` | `old_working_scripts/melts_melting_scan.csv` | Full scan CSV. |
| `TBL_OUTPUT_DIRECTORY` | `old` | Destination for alphaMELTS `*_tbl.txt` files. |
| `PHASE_LABELS` | dict | Readable phase names. This list is longer and also covers feldspars, SiO2 polymorphs, melilite, nepheline, leucite and kalsilite. |

Key findings reported: first liquid boundary, T01–T99, practical liquidus, full
liquid grid boundary, and the maximum liquid fraction in the scan. Each finding
lists its liquid and solid fractions and the stable phases.

---

### melts_PD_grid_final.py — composition grid temperature scan

Builds every combination of oxide values from start/end/step ranges and runs a
temperature scan for each one.

| Setting | Default | Description |
|---|---|---|
| `COMPOSITION_RANGES` | see file | For each oxide, `(start, end, step)` in wt%. When start equals end, the oxide is held constant and the step is ignored. Otherwise `end − start` must be an exact multiple of `step`. The grid is the Cartesian product of all oxide value lists. Each generated point is normalised to 100%, so the values act as relative proportions. |
| `MAX_CALCULATIONS` | `30000` | The run stops if the grid has more points than this. **Check the grid size first**: each oxide range multiplies the total, and the default ranges already produce several thousand points. |
| `T_MIN_C` / `T_MAX_C` / `TEMPERATURE_STEP_C` | `800` / `2000` / `100` | Temperature ladder, °C. |
| `FIRST_LIQUID_FRACTION`, `PRACTICAL_LIQUIDUS_FRACTION`, `FULL_LIQUID_TOLERANCE`, `LIQUID_FRACTION_MILESTONES`, `REPORTING_PHASE_FRACTION_MIN`, `PATH_H2O_WT_PCT`, `MELTS_MODEL`, `PRESSURE_BAR`, `FE3_FET_LIQ`, `PHASE_DIAGRAM_CORES` | as in the CSV batch script | See [Settings](#settings). |
| `OUTPUT_DIRECTORY` | `melts_pd_grid_results` | Output folder. |
| `SUMMARY_FILE` / `SCAN_FILE` | `melts_pd_grid_summary.csv` / `melts_pd_grid_scan.csv` | Output files. |
| `TBL_OUTPUT_DIRECTORY` | `old` | `*_tbl.txt` files are renamed `grid_<id>_<name>`. |

---

### 2_1.py — CSV composition scan (older variant)

An earlier version of the CSV batch script, kept for reference. It reads
`INPUT_CSV_FILE` (default `compositions.csv`) and runs the same temperature scan,
writing to `melts_pd_grid_results/`. Note that it writes to **the same output
files as `melts_PD_grid_final.py`**, so running one overwrites the other's
results.

Differences from `melt_PD_csv_comp_final.py`:

- `COMPOSITION_RANGES` is **not** used to build a grid here. It only decides
  which oxide columns appear in the output CSVs.
- Unsupported oxides are listed on the console and ignored. There is no audit
  file and no settings log.
- Defaults: `T_MIN_C = 1000`, `T_MAX_C = 1400`, `TEMPERATURE_STEP_C = 100`,
  `MAX_CALCULATIONS = 500`.

The other settings mean the same as in the CSV batch script. Use
`melt_PD_csv_comp_final.py` for new work.

---

### melts_liquidus_single_final.py — single composition direct liquidus

Runs `ptt.findLiq_multi` for one composition and prints:

- the analysed composition and the iron conversion (Fe2O3 + FeO → FeOt, total
  before normalisation)
- the liquidus temperature, primary liquidus phase, MELTS phase identifier,
  fluid saturation, and the input and calculated Fe3+/Fe
- the liquid composition at the liquidus
- the normalised bulk composition sent to MELTS

| Setting | Default | Description |
|---|---|---|
| `ANALYSED_OXIDE_WT_PCT` | SiO2 80.94, Al2O3 10.64, Fe2O3 2.29, … | Composition in oxide wt%. Unlike the scan scripts, **H2O is used as given** here; there is no trace-water override. |
| `PRESSURE_BAR` | `1.0` | Pressure, bar. |
| `MELTS_MODEL` | `"MELTSv1.0.2"` | Thermodynamic model. |
| `FE3_FET_LIQ` | `0.80` | Fe3+/total Fe of the liquid. |
| `OUTPUT_FILE` | `old/melts_liquidus_result.csv` | Raw `findLiq_multi` output. |
| `PHASE_LABELS` | dict | Readable phase names. |

---

### run_melts_grid.py — composition grid of direct liquidus runs

Generates a composition grid and runs `melts_liquidus_single_final.py` once per
point, swapping in each composition and output file. It then prints a combined
table of liquidus temperatures and primary phases. This script requires
`old/run_melts_batch.py`, which provides the validation and summary helpers.

| Setting | Default | Description |
|---|---|---|
| `COMPOSITION_RANGES` | single point (SiO2 77, Al2O3 5, Fe2O3 1, MgO 2, CaO 15) | Per-oxide `(start, end, step)`, with the same rules as in `melts_PD_grid_final.py`. |
| `BATCH_OUTPUT_DIRECTORY` | `old/batch_melts_grid_results` | Receives `melts_liquidus_grid_<n>.csv`, one file per point. |
| `MAX_CALCULATIONS` | `500` | The run stops if the grid is larger than this. |

Model, pressure and Fe3+/Fe come from the settings in
`melts_liquidus_single_final.py`.

---

### Helper modules

- [composition_grid_generator.py](composition_grid_generator.py):
  `make_composition_grid(ranges)` yields every composition in the Cartesian
  product of the oxide ranges, and `composition_grid_size(ranges)` counts them.
  It uses exact decimal arithmetic, so steps such as 0.1 do not drift, and it
  supports descending ranges (negative step).
- [composition_generator.py](composition_generator.py):
  `linear_composition_path(start, end, point_count)` returns evenly spaced
  compositions along a straight line between two compositions. It is used by
  `old/run_melts_batch.py`.

---

## Input CSV format

One row per composition, one column per oxide (wt%). Column order does not
matter, and missing oxides count as 0.

```csv
Na2O,MgO,Al2O3,SiO2,P2O5,K2O,CaO,MnO,Fe2O3,ZrO2,HfO2
1.76,0.19,10.64,80.94,1.08,1.41,1.66,0.03,2.29,0,0
```

- Recognised columns: `SiO2, TiO2, Al2O3, Fe2O3, FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, H2O, CO2`
  (case-sensitive; surrounding spaces are trimmed).
- Any other column (for example `ZrO2`, `HfO2`, or a sample ID) is copied to the
  outputs but **not sent to MELTS**. A warning is printed if such a column has
  non-zero values. Non-numeric values become 0.
- Rows do not need to sum to 100; they are normalised.

Example files included:

| File | Rows | Contents |
|---|---|---|
| `compositions_single.csv` | 1 | One test composition |
| `compositions.csv` | 86 | Main composition set |
| `compositions_grid_sweep.csv` | 119 | Sweep around the base composition (default input of the CSV batch script) |
| `compositions_AI_gen.csv` | 122 | Generated high-Al2O3/SiO2 compositions |

---

## Output files and columns

### Summary CSV (one row per composition)

| Column | Meaning |
|---|---|
| `composition_id` | Row number, starting at 1 |
| oxide columns | Raw input wt%, before normalisation |
| extra columns | Unsupported or non-oxide CSV columns, copied unchanged (CSV scripts) |
| `raw_oxide_total_wt_pct` | Sum of the raw input values |
| `first_liquid_boundary_degC` | Interpolated temperature where liquid fraction reaches `FIRST_LIQUID_FRACTION` |
| `scan_min_liquid_fraction` | Liquid fraction at `T_MIN_C`. If it is above 0, melting began below the scan range. |
| `T01_degC` … `T99_degC` | Temperatures where the liquid fraction reaches each milestone |
| `first_liquid_temperature_degC` / `_milestone` | Lowest milestone that was actually found, and its name (CSV batch script only) |
| `practical_liquidus_degC` | Same value as T99 |
| `full_liquid_grid_boundary_degC` | Temperature where the liquid reaches 100% (within `FULL_LIQUID_TOLERANCE`) |
| `scan_max_liquid_fraction` | Highest liquid fraction reached. If it is below 1, raise `T_MAX_C`. |
| `first_liquid_phase_assemblage` | Phases at the scan point nearest the first liquid, or the error message if the run failed |
| `full_liquid_phase_assemblage` | Phases at the scan point nearest the full liquidus |

An empty value or "Not found" means the threshold was not crossed inside the
temperature range.

### Scan CSV (one row per composition per temperature)

`composition_id`, oxide (and extra) columns, `temperature_degC`, `pressure_bar`,
`liquid_fraction`, `solid_fraction`, and `stable_phases`. `stable_phases` is
readable text such as `Spinel type oxide: 0.01234; LIQUID [MELTS silicate liquid]: 0.98766`.

### alphaMELTS table files

alphaMELTS may write `*_tbl.txt` files to the working folder. The scripts detect
files created during each run and move them into `old/`.

---

## Choosing settings

- **Find the range first, then refine.** Run a coarse scan (100 °C steps) over a
  wide range. Then narrow `T_MIN_C`/`T_MAX_C` around the melting interval and
  use 10–25 °C steps.
- **"Not bracketed" or `scan_max_liquid_fraction` < 1**: raise `T_MAX_C`.
  **`scan_min_liquid_fraction` > 0**: lower `T_MIN_C`, but stay above about
  800 °C.
- **Range errors**: `T_MAX_C − T_MIN_C` must be an exact multiple of
  `TEMPERATURE_STEP_C`, and each varied oxide's `end − start` must be an exact
  multiple of its step.
- **Redox**: `FE3_FET_LIQ` of 0.8 assumes oxidising, air-fired conditions. Use
  lower values (for example 0.1–0.3) for reducing or inert atmospheres.
- **Speed**: raise `PHASE_DIAGRAM_CORES` towards your CPU core count. Run time
  grows with (number of compositions) × (number of temperature points).
- **Checking the liquidus**: use `melts_csv_batch_calc_with_findliq.py` to
  compare the scan-based full liquidus with the direct `findLiq` result.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Temperature range must be exactly divisible…` | Adjust `T_MIN_C`, `T_MAX_C` or `TEMPERATURE_STEP_C`. |
| `… above MAX_CALCULATIONS` | The CSV or grid is larger than the limit. Reduce it, or raise `MAX_CALCULATIONS` on purpose. |
| `Composition CSV not found` | Run from the project root or fix `INPUT_CSV_PATH`. |
| A row shows `FAILED: …` | MELTS did not converge for that composition. Try a different temperature range or step, or check that the composition is reasonable. |
| Hangs or odd results at low temperature | This is a known MELTS issue below about 800–900 °C. Raise `T_MIN_C`. |
| `ModuleNotFoundError: old.run_melts_batch` | `run_melts_grid.py` must be run from the project root, with `old/run_melts_batch.py` present. |
| Multiprocessing errors on Windows | Run the script directly (`python script.py`), not from an interactive session. The scripts already call `multiprocessing.freeze_support()`. |

---

## Repository layout

```
.
├── melt_PD_csv_comp_final.py            # CSV batch temperature scan (main tool)
├── melts_csv_batch_calc_with_findliq.py # CSV batch + direct liquidus
├── melts_PD_final.py                    # single-composition scan
├── melts_PD_grid_final.py               # composition-grid scan
├── 2_1.py                               # older CSV scan variant
├── melts_liquidus_single_final.py       # single-composition direct liquidus
├── run_melts_grid.py                    # grid of direct liquidus runs
├── composition_grid_generator.py        # grid helper
├── composition_generator.py             # linear path helper
├── compositions*.csv                    # example inputs
├── old/                                 # legacy scripts, tbl files, liquidus outputs
└── old_working_scripts/                 # earlier script versions; melts_PD_final.py output
```
