# Techno-Economic Optimisation of PEM and Alkaline Water Electrolyser Systems

**Master Thesis — Shubham Manchanda**  
*Fakultät Maschinenbau, M.Eng - Wasserstofftechnologie - und wirtschaft, TH Ingolstadt*

---

## What This Project Does

This codebase simulates **15 years of hourly operation** for two types of hydrogen electrolysers (PEM and Alkaline) powered by 132 MW of wind and solar energy. It calculates:

- How much hydrogen is produced each hour
- How the electrolyser degrades over time and when components need replacement
- The **Levelised Cost of Hydrogen (LCOH)** — the total cost per kg of H2
- The optimal electrolyser size and storage capacity for different renewable energy scenarios

### Key Results

| Metric | PEM (35 MW) | Alkaline (20 MW) |
|--------|-------------|------------------|
| **LCOH** | 9.06 EUR/kg H2 | 4.86 EUR/kg H2 |
| **Demand Met** | 95.2% | 96.0% |
| **Stack Replacements** | 1 (at ~60,000 h) | 1 (at ~80,000 h) |

---

## Quick Start (5 minutes)

### Prerequisites

- **Python 3.9 or higher** — check with `python3 --version`
- **pip** — comes with Python
- No MATLAB needed (data files are included)

### Step 1: Download the Code

**Option A — Git (recommended):**
```bash
git clone https://github.com/shubham0429/Techno_economic-Optimization-of-Electrolyser-performance-.git
cd Techno_economic-Optimization-of-Electrolyser-performance-
```

**Option B — Download ZIP:**  
Click the green **Code** button on GitHub, then **Download ZIP**, and extract it.

### Step 2: Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (Command Prompt)
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# Install dependencies (only 4 packages: numpy, scipy, pandas, matplotlib)
pip install -r requirements.txt
```

### Step 3: Run a Simulation

```bash
cd source_code
python run_pem_thesis_final.py
```

After about 6 minutes you will see results in `results/pem_thesis_final/`.

---

## What to Run (and in What Order)

All commands must be run from the `source_code/` folder:

```bash
cd source_code
```

### Essential Scripts (start here)

| # | Command | What It Does | Time |
|---|---------|--------------|------|
| 1 | `python run_pem_thesis_final.py` | Runs PEM simulation for 15 years, generates 8 figures + CSV data | ~6 min |
| 2 | `python run_alkaline_thesis_final.py` | Same for Alkaline technology | ~4 min |
| 3 | `python corrected_thesis_plots.py` | Generates comparison plots and optimisation heatmaps (uses pre-computed data) | ~30 sec |
| 4 | `python sensitivity_analysis_alkaline.py` | Monte Carlo uncertainty analysis + tornado diagrams | ~50 sec |

### Optional Scripts (only if needed)

| Command | What It Does | Time |
|---------|--------------|------|
| `python pem_optimization_v3.py` | Re-runs PEM grid search (600 configs x 6 RE scenarios). Pre-computed results already included. Only re-run if you change parameters | ~3-5 hrs |
| `python alkaline_optimization_v3.py` | Same grid search for Alkaline | ~3-5 hrs |

> **Warning:** The optimisation scripts take hours. The pre-computed CSVs in `results/data/` are ready to use.

---

## Where to Find Results

```
results/
|
+-- pem_thesis_final/              <-- PEM simulation output
|   +-- baseline_timeseries.csv        131,400 hourly data points (15 years)
|   +-- summary_results.csv            Key performance indicators
|   +-- fig1-fig8 (.png + .pdf)        Degradation, SEC, LCOH, Monte Carlo, etc.
|   +-- monte_carlo_*.csv              Uncertainty analysis data
|
+-- alkaline_thesis_final/         <-- Alkaline simulation output (same structure)
|
+-- thesis_final_plots/            <-- BEST plots for the thesis
|   +-- fig_lcoh_vs_re_comparison      LCOH vs renewable energy fraction
|   +-- fig_pem_performance_evolution  15-year PEM performance (3 panels)
|   +-- fig_alkaline_performance_evolution  15-year ALK performance
|   +-- pem_heatmap_RE20-RE100         Optimisation heatmaps (LCOH + demand)
|   +-- alkaline_heatmap_RE20-RE100
|   +-- fig_feasible_summary_table     Summary of feasible configurations
|
+-- plots/                         <-- All original thesis figures (90+ plots)
|
+-- data/                          <-- Pre-computed optimisation results
    +-- pem_grid_search_all_RE.csv       416 PEM configs
    +-- alkaline_grid_search_all_RE.csv  600 Alkaline configs
    +-- monte_carlo_results.csv
    +-- sensitivity_results.csv
```

---

## Project Structure

```
+-- README.md                    <-- You are here
+-- ABSTRACT.txt                 <-- 200-word thesis abstract
+-- FINAL_THESIS.docx            <-- Complete thesis document
+-- requirements.txt             <-- Python dependencies
|
+-- data/                        <-- Input data (do not modify)
|   +-- combined_wind_pv_DATA.mat    Hourly wind+solar power (8,760 hours)
|   +-- Company_2_hourly_gas_demand.csv  Hourly H2 demand profile
|
+-- source_code/                 <-- All Python code
    |
    |  SIMULATION ENGINES (core logic)
    +-- sim_concise.py               PEM electrolyser model (3,200 lines)
    +-- sim_alkaline.py              Alkaline electrolyser model (3,400 lines)
    +-- electrochemistry.py          Electrochemical equations (prototype)
    +-- data_loader.py               Data loading utilities
    |
    |  THESIS RUNNERS (run these)
    +-- run_pem_thesis_final.py      Chapter 4: PEM simulation + plots
    +-- run_alkaline_thesis_final.py Chapter 5: Alkaline simulation + plots
    |
    |  OPTIMISATION
    +-- pem_optimization_v3.py       PEM grid search (10 sizes x 10 storages x 6 RE)
    +-- alkaline_optimization_v3.py  Alkaline grid search
    |
    |  ANALYSIS AND PLOTTING
    +-- corrected_thesis_plots.py    ** Final corrected comparison figures **
    +-- pem_thesis_plots_complete.py PEM-specific figures
    +-- alkaline_thesis_plots.py     Alkaline-specific figures
    +-- thesis_gold_plots.py         Publication-style figures (see known issues)
    +-- pem_vs_alkaline_comparison.py
    +-- sensitivity_analysis_alkaline.py  Monte Carlo + tornado charts
    +-- stack_lifetime_sensitivity.py
    +-- optimization_heatmaps_with_demand.py
```

---

## How the Simulation Works

```
Input Data                    Simulation                        Economics
----------                    ----------                        ---------
Wind + Solar power    ->  Electrolyser model         ->  LCOH (EUR/kg H2)
(8,760 h/year)           - Polarisation curve            NPV, IRR
                         - Degradation over time          Payback period
H2 demand profile     ->  - Stack replacements       ->  Cost breakdown
(hourly kg)              - Storage buffer dynamics       (CAPEX, OPEX, energy)

                    Repeated for 15 years (131,400 hours)
```

Cell voltage model: V = E_rev(T,P) + n_activation + n_ohmic + n_concentration

Optimisation: Grid search over 600 combinations (10 electrolyser sizes x 10 storage
capacities x 6 renewable energy fractions), selecting the configuration with the
lowest LCOH that meets at least 95% of hydrogen demand.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'numpy'` | Run `pip install -r requirements.txt` (make sure your venv is activated) |
| `FileNotFoundError: .mat file not found` | Make sure you run scripts from inside `source_code/`, not the root folder |
| `python: command not found` | Use `python3` instead of `python` |
| Script takes too long | The optimisation scripts take 3-5 hours. This is normal. Use the pre-computed CSVs instead |
| Plots look different on Windows | Font rendering varies by OS. The data is identical |

---

## Known Issues

| File | Issue | What to Do |
|------|-------|------------|
| `thesis_gold_plots.py` | Contains hardcoded LCOH values (lines 675-680) that do not match the 95% demand filter | Use `corrected_thesis_plots.py` instead |
| `pem_vs_alkaline_comparison.py` | API mismatch with current `sim_alkaline.py` | Comparison data is in `results/data/comparison_summary.csv` |

---

## Technical Details

### Dependencies

Only 4 Python packages (installed automatically via requirements.txt):

| Package | Purpose |
|---------|---------|
| numpy | Numerical arrays and math |
| scipy | MATLAB .mat file loading, interpolation |
| pandas | DataFrames for timeseries data |
| matplotlib | All plots and figures |

### Reproducibility

- Random seed = 42: all stochastic operations are seeded for reproducibility
- 132 MW RE capacity: fixed in the input data file, not a tuneable parameter
- Tested on Python 3.9.6 (macOS). Compatible with Python 3.9+.

---

## Contact

**Shubham Manchanda**  
Master Thesis — Chair of Wind Energy Technology (WTW), TU Munich

## License

This code is provided for academic review and examination purposes.
Please contact the author before any reuse or redistribution.
