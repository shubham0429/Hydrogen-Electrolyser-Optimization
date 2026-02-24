#!/usr/bin/env python3
"""
Alkaline Electrolyser Grid Search Optimization — v3 MULTI-OBJECTIVE
====================================================================

Multi-objective optimization with THREE criteria:
  1. Minimise LCOH (€/kg H₂)
  2. Reliability ≥ 95% demand met (unmet demand < 5%)
  3. Maximise efficiency (minimise SEC kWh/kg)

Selection:  Filter → configs meeting ≥ 95% reliability
            Rank   → by LCOH ascending, then SEC ascending (tiebreaker)
            Pick   → first (lowest LCOH among reliable configs)

KEY FIX: Now loads real demand data and passes it to the ALK simulator,
enabling proper demand tracking and storage utilization.

Grid:     Electrolyser 10-50 MW (step 5)  ×  Storage 2,000-24,000 kg
RE Input: 132 MW combined wind + PV (actual hourly data)
RE Scenarios: 20%, 40%, 50%, 60%, 80%, 100%

Author:  Shubham Manchanda
Thesis:  Techno-Economic Optimization of Electrolyser Performance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.io
import sys, warnings, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sim_alkaline import (
    get_alkaline_config,
    simulate,
    compute_lcoh,
    load_demand_data as load_alk_demand,
)

# ── Plotting defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

OUTPUT = Path(__file__).parent.parent / "results" / "alkaline_optimization_v3"
OUTPUT.mkdir(parents=True, exist_ok=True)

SIM_YEARS = 15
LCOH_CUTOFF = 12.0         # €/kg
RELIABILITY_TARGET = 0.95  # 95% demand met → unmet < 5%

DEMAND_PATH = Path(__file__).parent.parent / "data" / "Company_2_hourly_gas_demand.csv"

# =============================================================================
# LOAD RE DATA
# =============================================================================
def load_re_power(simulation_years=SIM_YEARS):
    """Load actual wind+PV power profile."""
    mat_path = Path(__file__).parent.parent / "data" / "combined_wind_pv_DATA.mat"
    if not mat_path.exists():
        mat_path = Path("/Users/shubhammanchanda/Downloads/combined_wind_pv_data.mat")

    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    pv = np.array(mat["P_PV"]).flatten()
    wind = np.array(mat["P_wind_selected"]).flatten()
    base_W = pv + wind

    n_needed = simulation_years * 8760
    n_tiles = int(np.ceil(n_needed / len(base_W)))
    power_W = np.tile(base_W, n_tiles)[:n_needed]
    return power_W


# =============================================================================
# LOAD DEMAND DATA (for ALK)
# =============================================================================
def load_demand_for_alk(simulation_years=SIM_YEARS):
    """
    Load real demand data and compute mean hourly demand in kg H₂/h.
    This will be passed to the ALK config as demand_kg_h.
    """
    df = pd.read_csv(DEMAND_PATH)
    demand_kWh = pd.to_numeric(df['demand_kwh'], errors='coerce').values
    LHV_H2 = 33.33  # kWh/kg
    demand_kg_h = demand_kWh / LHV_H2

    mean_demand = np.mean(demand_kg_h)
    annual_demand_t = np.sum(demand_kg_h) / 1000
    print(f"  Demand data loaded: mean={mean_demand:.1f} kg/h, "
          f"annual={annual_demand_t:.0f} t/yr")
    return mean_demand, demand_kg_h


# =============================================================================
# GRID SEARCH — NO PRUNING (run all combos for multi-objective)
# =============================================================================
def run_grid(power_W_full, mean_demand_kg_h, sizes_MW, storages_kg,
             re_frac=1.0, sim_years=SIM_YEARS, label=""):
    """
    Exhaustive grid search — runs ALL size × storage combinations.
    No pruning, because larger storage can improve reliability even when
    LCOH increases.
    """
    power_W = power_W_full * re_frac
    rows = []
    run_count = 0

    for sz in sizes_MW:
        for st in storages_kg:
            run_count += 1
            try:
                cfg = get_alkaline_config(
                    P_nom_MW=sz,
                    simulation_years=sim_years,
                    storage_capacity_kg=st,
                    demand_kg_h=mean_demand_kg_h,     # ← KEY: enable demand tracking
                    enable_oxygen_credit=True,
                    enable_heat_recovery=True,
                    enable_replacement_performance_boost=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = simulate(power_W, cfg, verbose=False)
                eco = compute_lcoh(res, cfg)

                # Curtailment
                P_nom_W = sz * 1e6
                curtailed_Wh = np.sum(np.maximum(
                    power_W[:sim_years * 8760] - P_nom_W, 0))
                total_Wh = np.sum(power_W[:sim_years * 8760])
                curtailment = curtailed_Wh / total_Wh if total_Wh > 0 else 0

                lcoh = eco.lcoh_total

                # ── DEMAND TRACKING ──────────────────────────────────
                total_delivered = np.sum(res.h2_to_demand_kg)
                total_unmet = np.sum(res.unmet_demand_kg)
                total_demand = total_delivered + total_unmet
                unmet_frac = total_unmet / max(total_demand, 1e-9)
                demand_met_frac = 1.0 - unmet_frac

                rows.append({
                    're_fraction': re_frac,
                    'size_MW': sz,
                    'storage_kg': st,
                    'LCOH': lcoh,
                    'LCOH_credits': getattr(eco, 'lcoh_with_credits', lcoh),
                    'LCOH_capex': eco.lcoh_capex,
                    'LCOH_opex': eco.lcoh_opex_fixed,
                    'LCOH_elec': eco.lcoh_electricity,
                    'NPV_M': eco.npv / 1e6,
                    'IRR': eco.irr,
                    'payback': eco.payback_years,
                    'CAPEX_M': eco.total_capex / 1e6,
                    'H2_total_t': res.total_h2_production_kg / 1e3,
                    'H2_annual_t': res.total_h2_production_kg / 1e3 / sim_years,
                    'SEC': res.average_sec_kWh_kg,
                    'CF': res.capacity_factor_avg,
                    'op_hours': res.total_operating_hours,
                    'replacements': res.stack_replacements,
                    'curtailment': curtailment,
                    'cycles': res.total_cycles,
                    'demand_total_t': total_demand / 1e3,
                    'unmet_demand_t': total_unmet / 1e3,
                    'unmet_demand_pct': unmet_frac * 100,
                    'demand_met_pct': demand_met_frac * 100,
                })

                flag = "✓" if demand_met_frac >= RELIABILITY_TARGET else "✗"
                print(f"  [{label}] {sz:2.0f} MW, {st:5.0f} kg → "
                      f"LCOH €{lcoh:.2f}, CF {res.capacity_factor_avg:.1%}, "
                      f"Met {demand_met_frac:.1%} {flag}")

            except Exception as e:
                print(f"  [{label}] {sz:2.0f} MW, {st:5.0f} kg → ERROR: {e}")

    print(f"  [{label}] Done: {run_count} simulations")
    return pd.DataFrame(rows)


# =============================================================================
# MULTI-OBJECTIVE SELECTION
# =============================================================================
def select_optimal(df, reliability_target=RELIABILITY_TARGET):
    """
    Multi-objective optimal selection:
      1. Filter configs with demand_met_pct >= target (95%)
      2. Among feasible, sort by LCOH (asc), then SEC (asc)
      3. Return best row

    If NO config meets reliability, return the one with highest demand_met.
    """
    feasible = df[df['demand_met_pct'] >= reliability_target * 100]

    if len(feasible) > 0:
        best = feasible.sort_values(['LCOH', 'SEC']).iloc[0]
        return best, True
    else:
        best = df.sort_values('demand_met_pct', ascending=False).iloc[0]
        return best, False


# =============================================================================
# PLOTTING  —  HEATMAPS (2×3)
# =============================================================================
def plot_heatmaps(df, re_frac):
    """2×3 heatmap for a single RE fraction — includes demand met %."""
    sizes = sorted(df['size_MW'].unique())
    stores = sorted(df['storage_kg'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(f'Alkaline Electrolyser Multi-Objective Optimization — {re_frac*100:.0f}% RE\n'
                 f'(132 MW Wind+PV, {SIM_YEARS}-Year, Reliability Target ≥ {RELIABILITY_TARGET:.0%})',
                 fontsize=15, fontweight='bold', y=1.02)

    metrics = [
        ('LCOH',           'LCOH (€/kg)',          'YlGnBu_r', '€/kg',  '{:.2f}'),
        ('demand_met_pct', 'Demand Met (%)',        'RdYlGn',   '%',     '{:.1f}'),
        ('SEC',            'SEC (kWh/kg)',          'YlOrRd',   'kWh/kg','{:.1f}'),
        ('NPV_M',         'NPV (€ Million)',        'RdYlGn',   '€M',    '{:.1f}'),
        ('IRR',            'IRR (%)',               'RdYlGn',   '%',     '{:.1f}'),
        ('CF',             'Capacity Factor',        'YlGnBu',   '',      '{:.1%}'),
    ]

    for ax, (col, title, cmap, cbar_label, fmt) in zip(axes.flat, metrics):
        pivot = df.pivot_table(values=col, index='storage_kg', columns='size_MW')
        pivot = pivot.reindex(index=stores, columns=sizes)

        im = ax.pcolormesh(range(len(sizes)+1), range(len(stores)+1),
                           pivot.values, cmap=cmap, shading='flat')
        ax.set_xticks([i+0.5 for i in range(len(sizes))])
        ax.set_xticklabels([f'{s:.0f}' for s in sizes], fontsize=9)
        ax.set_yticks([i+0.5 for i in range(len(stores))])
        ax.set_yticklabels([f'{s:.0f}' for s in stores], fontsize=8)
        ax.set_xlabel('Electrolyser Size (MW)', fontweight='bold')
        ax.set_ylabel('Storage Size (kg)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontweight='bold')

        # Annotate
        for i, st in enumerate(stores):
            for j, sz in enumerate(sizes):
                val = pivot.loc[st, sz] if (st in pivot.index and sz in pivot.columns) else np.nan
                if np.isfinite(val):
                    txt = fmt.format(val)
                    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
                    mid = (vmin + vmax) / 2
                    color = 'white' if val > mid else 'black'
                    ax.text(j+0.5, i+0.5, txt, ha='center', va='center',
                            fontsize=5.5, color=color, fontweight='bold')

        # Star optimal
        best_row, feasible = select_optimal(df)
        if col == 'LCOH' and best_row['size_MW'] in sizes and best_row['storage_kg'] in stores:
            bj = sizes.index(best_row['size_MW'])
            bi = stores.index(best_row['storage_kg'])
            ax.plot(bj+0.5, bi+0.5, '*', ms=18, color='red',
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)

    plt.tight_layout()
    fname = f'alkaline_heatmaps_RE{re_frac*100:.0f}.png'
    plt.savefig(OUTPUT / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fname}")


# =============================================================================
# PLOTTING  —  RE SCENARIO COMPARISON
# =============================================================================
def plot_re_comparison(df_all):
    """4-panel: LCOH, Size+Storage, H2+Reliability, Efficiency vs RE%."""
    records = []
    for re_f in sorted(df_all['re_fraction'].unique()):
        sub = df_all[df_all['re_fraction'] == re_f]
        best, feasible = select_optimal(sub)
        rec = best.to_dict()
        rec['feasible'] = feasible
        records.append(rec)
    opt = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Alkaline Electrolyser — Multi-Objective Optimization Results\n'
                 f'(Reliability ≥ {RELIABILITY_TARGET:.0%}, then min LCOH, then min SEC)',
                 fontsize=14, fontweight='bold')

    # Panel 1: LCOH
    ax = axes[0, 0]
    ax.plot(opt['re_fraction']*100, opt['LCOH'], 'o-', color='#2ECC71', lw=2.5, ms=10)
    for _, row in opt.iterrows():
        marker = '✓' if row['feasible'] else '✗'
        ax.annotate(f"€{row['LCOH']:.2f}\n{row['size_MW']:.0f}MW/{row['storage_kg']:.0f}kg\n{marker}",
                    (row['re_fraction']*100, row['LCOH']),
                    textcoords='offset points', xytext=(0, 15),
                    ha='center', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Optimal LCOH (€/kg)', fontweight='bold')
    ax.set_title('Optimal LCOH vs RE Availability', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 2: Size & Storage
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.bar(opt['re_fraction']*100 - 2, opt['size_MW'], width=4, color='#27AE60',
           alpha=0.7, label='Size (MW)')
    ax2.bar(opt['re_fraction']*100 + 2, opt['storage_kg'], width=4, color='#8E44AD',
            alpha=0.7, label='Storage (kg)')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Electrolyser Size (MW)', fontweight='bold', color='#27AE60')
    ax2.set_ylabel('Storage Size (kg)', fontweight='bold', color='#8E44AD')
    ax.set_title('Optimal Size & Storage', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: H₂ & Reliability
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar(opt['re_fraction']*100, opt['H2_annual_t'], width=8, color='#2ECC71',
           alpha=0.7, label='H₂ Production (t/yr)')
    ax2.plot(opt['re_fraction']*100, opt['demand_met_pct'], 's-', color='#E74C3C',
             lw=2.5, ms=10, label='Demand Met (%)')
    ax2.axhline(RELIABILITY_TARGET*100, ls='--', color='red', alpha=0.5,
                label=f'Target ({RELIABILITY_TARGET:.0%})')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Annual H₂ Production (t)', fontweight='bold')
    ax2.set_ylabel('Demand Met (%)', fontweight='bold', color='#E74C3C')
    ax.set_title('H₂ Production & Reliability', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: CF & SEC
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(opt['re_fraction']*100, opt['CF']*100, 'o-', color='#F39C12', lw=2.5, ms=10,
            label='Capacity Factor (%)')
    ax2.plot(opt['re_fraction']*100, opt['SEC'], 's-', color='#555555', lw=2, ms=8,
             label='SEC (kWh/kg)')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Capacity Factor (%)', fontweight='bold', color='#F39C12')
    ax2.set_ylabel('SEC (kWh/kg)', fontweight='bold', color='#555555')
    ax.set_title('Efficiency Metrics', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'alkaline_re_scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved alkaline_re_scenario_comparison.png")


# =============================================================================
# PLOTTING  —  PARETO FRONT
# =============================================================================
def plot_pareto(df):
    """Pareto front: LCOH vs Demand Met %."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: LCOH vs Demand Met
    ax = axes[0]
    sc = ax.scatter(df['LCOH'], df['demand_met_pct'],
                    c=df['size_MW'], cmap='viridis',
                    s=30 + df['storage_kg']/50, alpha=0.7,
                    edgecolors='grey', linewidths=0.4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Electrolyser Size (MW)', fontweight='bold')

    ax.axhline(RELIABILITY_TARGET*100, ls='--', color='red', alpha=0.7,
               label=f'Reliability Target ({RELIABILITY_TARGET:.0%})')

    best, feasible = select_optimal(df)
    ax.plot(best['LCOH'], best['demand_met_pct'], '*', ms=20, color='red',
            markeredgecolor='black', markeredgewidth=1.5, zorder=10,
            label=f"Optimal: {best['size_MW']:.0f}MW/{best['storage_kg']:.0f}kg\n"
                  f"LCOH €{best['LCOH']:.2f}, Met {best['demand_met_pct']:.1f}%")

    ax.set_xlabel('LCOH (€/kg)', fontweight='bold')
    ax.set_ylabel('Demand Met (%)', fontweight='bold')
    ax.set_title('Alkaline — LCOH vs Reliability Trade-off\n'
                 '(Colour = Size, Bubble = Storage)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: LCOH vs NPV
    ax = axes[1]
    sc = ax.scatter(df['LCOH'], df['NPV_M'],
                    c=df['demand_met_pct'], cmap='RdYlGn',
                    s=30 + df['storage_kg']/50, alpha=0.7,
                    edgecolors='grey', linewidths=0.4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Demand Met (%)', fontweight='bold')

    ax.plot(best['LCOH'], best['NPV_M'], '*', ms=20, color='red',
            markeredgecolor='black', markeredgewidth=1.5, zorder=10,
            label=f"Optimal: €{best['LCOH']:.2f}/kg, NPV €{best['NPV_M']:.1f}M")
    ax.axhline(0, ls='--', color='grey', alpha=0.5, label='Break-even')

    ax.set_xlabel('LCOH (€/kg)', fontweight='bold')
    ax.set_ylabel('NPV (€ Million)', fontweight='bold')
    ax.set_title('Alkaline — LCOH vs NPV (colour = Reliability)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'alkaline_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved alkaline_pareto_front.png")


# =============================================================================
# PLOTTING  —  SIZE VS METRICS
# =============================================================================
def plot_size_metrics(df):
    """6-panel: metrics vs electrolyser size at optimal storage per size."""
    records = []
    for sz in sorted(df['size_MW'].unique()):
        sub = df[df['size_MW'] == sz]
        best_row, _ = select_optimal(sub)
        records.append(best_row.to_dict())
    df0 = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Alkaline Electrolyser — Size vs Performance (Multi-Objective Optimal Storage)\n'
                 f'(132 MW RE, Reliability ≥ {RELIABILITY_TARGET:.0%})',
                 fontsize=14, fontweight='bold', y=1.02)

    panels = [
        ('size_MW', 'LCOH',           'LCOH (€/kg)',          '#2ECC71', 'o-'),
        ('size_MW', 'NPV_M',          'NPV (€ Million)',      '#27AE60', 'o-'),
        ('size_MW', 'demand_met_pct', 'Demand Met (%)',        '#E74C3C', 'o-'),
        ('size_MW', 'SEC',            'SEC (kWh/kg)',          '#8E44AD', 'o-'),
        ('size_MW', 'CF',             'Capacity Factor',       '#F39C12', 'o-'),
        ('size_MW', 'storage_kg',     'Optimal Storage (kg)',  '#555555', 'o-'),
    ]

    for ax, (x, y, ylabel, color, style) in zip(axes.flat, panels):
        ax.plot(df0[x], df0[y], style, color=color, lw=2, ms=8)
        ax.set_xlabel('Electrolyser Size (MW)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(ylabel, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if y == 'CF':
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        if y == 'NPV_M':
            ax.axhline(0, ls='--', color='red', alpha=0.5)
        if y == 'demand_met_pct':
            ax.axhline(RELIABILITY_TARGET*100, ls='--', color='red', alpha=0.5,
                       label=f'Target ({RELIABILITY_TARGET:.0%})')
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'alkaline_size_vs_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved alkaline_size_vs_metrics.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("ALKALINE ELECTROLYSER — MULTI-OBJECTIVE GRID OPTIMIZATION v3")
    print(f"Criteria: (1) Reliability ≥ {RELIABILITY_TARGET:.0%}  "
          f"(2) Min LCOH  (3) Min SEC")
    print("Fixed 132 MW RE  ·  6 RE scenarios  ·  15-year simulation")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1] Loading data …")
    power_W = load_re_power(simulation_years=SIM_YEARS)
    print(f"    RE Profile: {len(power_W)} h, "
          f"max {power_W.max()/1e6:.1f} MW, "
          f"mean {power_W.mean()/1e6:.1f} MW, "
          f"CF {power_W.mean()/power_W.max():.1%}")

    mean_demand_kg_h, demand_kg_h_array = load_demand_for_alk(SIM_YEARS)
    annual_demand_t = np.sum(demand_kg_h_array[:8760]) / 1000
    print(f"    Demand: mean={mean_demand_kg_h:.1f} kg/h, "
          f"annual={annual_demand_t:.0f} t/yr")

    # ── Grid definition ──────────────────────────────────────────────────────
    # Actual demand: ~5,800 kg/day → 2-3 day buffer = 11,600-17,400 kg
    sizes_MW  = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    storages  = list(range(2000, 10001, 2000)) + list(range(12000, 25001, 3000))

    re_fractions = [0.20, 0.40, 0.50, 0.60, 0.80, 1.00]

    n_per_re = len(sizes_MW) * len(storages)
    print(f"\n    Grid: {len(sizes_MW)} sizes × {len(storages)} storages = "
          f"{n_per_re} configs per RE level")
    print(f"    RE scenarios: {re_fractions}")
    print(f"    Total simulations: {n_per_re * len(re_fractions)}")
    print(f"    Reliability target: ≥ {RELIABILITY_TARGET:.0%} demand met")

    # ── Run all RE scenarios ──────────────────────────────────────────────────
    all_frames = []
    for frac in re_fractions:
        print(f"\n{'─'*60}")
        print(f"[RE = {frac*100:.0f}%]  Running {n_per_re} simulations …")
        print(f"{'─'*60}")
        df_re = run_grid(power_W, mean_demand_kg_h, sizes_MW, storages,
                         re_frac=frac, label=f"RE{frac*100:.0f}")
        all_frames.append(df_re)

        # Save per-RE CSV
        csv_name = f"grid_search_RE{frac*100:.0f}.csv"
        df_re.to_csv(OUTPUT / csv_name, index=False)
        print(f"  ✓ {len(df_re)} rows → {csv_name}")

        if len(df_re) > 0:
            best, feasible = select_optimal(df_re)
            status = "FEASIBLE" if feasible else "INFEASIBLE (best reliability)"
            print(f"  ★ Optimal [{status}]: "
                  f"€{best['LCOH']:.2f}/kg — "
                  f"{best['size_MW']:.0f} MW, {best['storage_kg']:.0f} kg, "
                  f"CF={best['CF']:.1%}, "
                  f"Demand Met={best['demand_met_pct']:.1f}%, "
                  f"SEC={best['SEC']:.1f} kWh/kg")

    # ── Combine & save ────────────────────────────────────────────────────────
    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.to_csv(OUTPUT / 'alkaline_grid_search_all_RE.csv', index=False)
    print(f"\n✓ Combined: {len(df_all)} rows → alkaline_grid_search_all_RE.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[2] Generating plots …")

    for frac in re_fractions:
        sub = df_all[df_all['re_fraction'] == frac]
        if len(sub) > 0:
            plot_heatmaps(sub, frac)

    plot_re_comparison(df_all)

    df100 = df_all[df_all['re_fraction'] == 1.0]
    if len(df100) > 0:
        plot_pareto(df100)
        plot_size_metrics(df100)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MULTI-OBJECTIVE OPTIMAL CONFIGURATIONS BY RE AVAILABILITY")
    print(f"(Reliability ≥ {RELIABILITY_TARGET:.0%}, then min LCOH, then min SEC)")
    print(f"{'='*70}")
    print(f"{'RE%':>5} {'Size':>6} {'Stor':>6} {'LCOH':>8} {'CF':>7} "
          f"{'H2/yr':>8} {'Met%':>7} {'SEC':>7} {'NPV':>8} {'IRR':>6} {'Feas':>5}")
    print(f"{'':>5} {'(MW)':>6} {'(kg)':>6} {'(€/kg)':>8} {'(%)':>7} "
          f"{'(t)':>8} {'(%)':>7} {'kWh/kg':>7} {'(€M)':>8} {'(%)':>6} {'':>5}")
    print("-" * 80)

    for frac in re_fractions:
        sub = df_all[df_all['re_fraction'] == frac]
        if len(sub) == 0:
            continue
        best, feasible = select_optimal(sub)
        feas = "✓" if feasible else "✗"
        print(f"{frac*100:5.0f} {best['size_MW']:6.0f} {best['storage_kg']:6.0f} "
              f"{best['LCOH']:8.2f} {best['CF']*100:7.1f} "
              f"{best['H2_annual_t']:8.0f} {best['demand_met_pct']:7.1f} "
              f"{best['SEC']:7.1f} {best['NPV_M']:8.1f} {best['IRR']:6.1f} {feas:>5}")

    # Comparison table
    print(f"\n{'─'*80}")
    print("COMPARISON: Min-LCOH (unconstrained) vs Multi-Objective (reliability ≥ 95%)")
    print(f"{'─'*80}")
    print(f"{'RE%':>5} {'--- Unconstrained ---':>30} {'--- Multi-Objective ---':>40}")
    print(f"{'':>5} {'Size':>6} {'Stor':>6} {'LCOH':>8} {'Met%':>7} "
          f"{'Size':>8} {'Stor':>8} {'LCOH':>8} {'Met%':>7}")
    print("-" * 80)

    for frac in re_fractions:
        sub = df_all[df_all['re_fraction'] == frac]
        if len(sub) == 0:
            continue
        unc = sub.loc[sub['LCOH'].idxmin()]
        mo, _ = select_optimal(sub)
        print(f"{frac*100:5.0f} {unc['size_MW']:6.0f} {unc['storage_kg']:6.0f} "
              f"{unc['LCOH']:8.2f} {unc['demand_met_pct']:7.1f} "
              f"{mo['size_MW']:8.0f} {mo['storage_kg']:8.0f} "
              f"{mo['LCOH']:8.2f} {mo['demand_met_pct']:7.1f}")

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed/60:.1f} minutes")
    print(f"  Results: {OUTPUT}")


if __name__ == "__main__":
    main()
