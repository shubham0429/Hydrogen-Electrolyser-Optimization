#!/usr/bin/env python3
"""
PEM Electrolyser Grid Search Optimization — v3 MULTI-OBJECTIVE
===============================================================

Multi-objective optimization with THREE criteria:
  1. Minimise LCOH (€/kg H₂)
  2. Reliability ≥ 95% demand met (unmet demand < 5%)
  3. Maximise efficiency (minimise SEC kWh/kg)

Selection:  Filter → configs meeting ≥ 95% reliability
            Rank   → by LCOH ascending, then SEC ascending (tiebreaker)
            Pick   → first (lowest LCOH among reliable configs)

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
from sim_concise import (
    get_config,
    load_power_data,
    load_demand_data,
    synthesize_multiyear_data,
    simulate,
    compute_economics,
    RNG_SEED,
)

# ── Plotting defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.size': 11, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.linewidth': 0.8, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

OUTPUT = Path(__file__).parent.parent / "results" / "pem_optimization_v3"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
MAT_PATH = Path(__file__).parent.parent / "data" / "combined_wind_pv_DATA.mat"
DEMAND_PATH = Path(__file__).parent.parent / "data" / "Company_2_hourly_gas_demand.csv"

if not MAT_PATH.exists():
    MAT_PATH = Path("/Users/shubhammanchanda/Downloads/combined_wind_pv_data.mat")

SIM_YEARS = 15
LCOH_CUTOFF = 10.0       # €/kg — reject configs above this as infeasible
RELIABILITY_TARGET = 0.95  # 95% demand met → unmet < 5%


# =============================================================================
# SINGLE SIMULATION WRAPPER
# =============================================================================
def run_single(size_mw, storage_kg, power_1yr, demand_1yr, sim_years=SIM_YEARS):
    """Run one PEM simulation and return (df, econ, cfg)."""
    rng = np.random.default_rng(RNG_SEED)
    cfg = get_config(size_mw=size_mw, storage_kg=storage_kg)
    cfg['YEARS'] = sim_years

    power_multi, demand_multi = synthesize_multiyear_data(
        power_1yr, demand_1yr, sim_years, rng, deterministic=True)

    df = simulate(cfg, power_multi, demand_multi, rng)
    econ = compute_economics(cfg, df, h2_selling_price_eur_per_kg=12.0)
    return df, econ, cfg


# =============================================================================
# GRID SEARCH WITH SMART PRUNING
# =============================================================================
def run_grid(power_1yr, demand_1yr, sizes_MW, storages_kg, re_frac=1.0,
             sim_years=SIM_YEARS, label=""):
    """
    Grid search with SMART PRUNING for multi-objective optimization.

    Pruning rules (documented for thesis):
      1. Sizes tested small→large. If LCOH > LCOH_CUTOFF (€10/kg) at
         MINIMUM storage for a given size, that size AND all larger sizes
         are skipped — larger electrolysers at the same RE have lower CF
         and strictly worse economics.
      2. For each viable size, storages tested small→large. If LCOH
         exceeds LCOH_CUTOFF AND demand is already met (≥ 95%), stop
         adding storage — it only adds CAPEX with no reliability benefit.
      3. If LCOH > €15/kg at any storage, stop that size regardless —
         these configurations are techno-economically infeasible.

    Rejected configurations are noted in the log for thesis documentation.
    """
    power_scaled = power_1yr * re_frac
    rows = []
    run_count = 0
    skip_count = 0
    HARD_CUTOFF = 15.0  # €/kg — absolute reject regardless

    for sz in sizes_MW:
        size_viable = False   # True if ANY storage at this size has LCOH ≤ cutoff
        min_storage_lcoh = None  # LCOH at min storage (first iteration)

        for st_idx, st in enumerate(storages_kg):
            run_count += 1
            try:
                df, econ, cfg = run_single(sz, st, power_scaled, demand_1yr,
                                           sim_years=sim_years)
                lcoh = econ['LCOH_EUR_per_kg']

                # Capacity factor
                total_power_used = df['power_kW'].sum()
                if 'curtailed_power_kWh' in df.columns:
                    total_power_used -= df['curtailed_power_kWh'].sum()
                max_power = cfg['ELECTROLYSER_SIZE_KW'] * len(df)
                cf = total_power_used / max(max_power, 1e-9)

                # SEC
                sec_mask = df['SEC_total_kWh_per_kg'] > 0
                sec_avg = df.loc[sec_mask, 'SEC_total_kWh_per_kg'].mean() if sec_mask.any() else np.nan

                total_h2 = econ['total_H2_kg']
                curtailed_kWh = df['curtailed_power_kWh'].sum() if 'curtailed_power_kWh' in df.columns else 0
                total_input_kWh = power_scaled.sum() * sim_years
                curtailment = curtailed_kWh / max(total_input_kWh, 1e-9)

                # ── DEMAND TRACKING ──────────────────────────────────
                total_demand_kg = df['demand_H2_kg'].sum()
                total_unmet_kg = df['unmet_kg'].sum()
                unmet_frac = total_unmet_kg / max(total_demand_kg, 1e-9)
                demand_met_frac = 1.0 - unmet_frac

                rows.append({
                    're_fraction': re_frac,
                    'size_MW': sz,
                    'storage_kg': st,
                    'LCOH': lcoh,
                    'LCOH_credits': econ['LCOH_with_credits_EUR_per_kg'],
                    'LCOH_capex': econ['LCOH_capex_EUR_per_kg'],
                    'LCOH_opex': econ['LCOH_opex_fixed_EUR_per_kg'],
                    'LCOH_elec': econ['LCOH_electricity_EUR_per_kg'],
                    'NPV_M': econ.get('NPV_profit_EUR', 0) / 1e6,
                    'IRR': econ.get('IRR_percent', 0) or 0,
                    'payback': econ.get('payback_period_years', 99),
                    'CAPEX_M': econ['capex_total'] / 1e6,
                    'H2_total_t': total_h2 / 1e3,
                    'H2_annual_t': total_h2 / 1e3 / sim_years,
                    'SEC': sec_avg,
                    'CF': cf,
                    'op_hours': (df['H2_kg'] > 0).sum(),
                    'replacements': len(econ['stack_replacement_years']),
                    'curtailment': curtailment,
                    'demand_total_t': total_demand_kg / 1e3,
                    'unmet_demand_t': total_unmet_kg / 1e3,
                    'unmet_demand_pct': unmet_frac * 100,
                    'demand_met_pct': demand_met_frac * 100,
                })

                if lcoh <= LCOH_CUTOFF:
                    size_viable = True

                flag = "✓" if demand_met_frac >= RELIABILITY_TARGET else "✗"
                print(f"  [{label}] {sz:2.0f} MW, {st:5.0f} kg → "
                      f"LCOH €{lcoh:.2f}, CF {cf:.1%}, "
                      f"Met {demand_met_frac:.1%} {flag}")

                # Track min-storage LCOH
                if st_idx == 0:
                    min_storage_lcoh = lcoh

                # ── PRUNING RULE 3: Hard cutoff ──────────────────────
                if lcoh > HARD_CUTOFF:
                    remaining = len(storages_kg) - st_idx - 1
                    if remaining > 0:
                        skip_count += remaining
                        print(f"  [{label}] {sz:.0f} MW: LCOH €{lcoh:.2f} > €{HARD_CUTOFF:.0f} "
                              f"→ REJECT remaining {remaining} storages (infeasible)")
                    break

                # ── PRUNING RULE 2: LCOH above cutoff AND reliability met ─
                if lcoh > LCOH_CUTOFF and demand_met_frac >= RELIABILITY_TARGET:
                    remaining = len(storages_kg) - st_idx - 1
                    if remaining > 0:
                        skip_count += remaining
                        print(f"  [{label}] {sz:.0f} MW: LCOH €{lcoh:.2f} > €{LCOH_CUTOFF:.0f} "
                              f"& demand met {demand_met_frac:.1%} ≥ {RELIABILITY_TARGET:.0%} "
                              f"→ skip {remaining} larger storages")
                    break

            except Exception as e:
                print(f"  [{label}] {sz:2.0f} MW, {st:5.0f} kg → ERROR: {e}")

        # ── PRUNING RULE 1: If min-storage LCOH already above cutoff,
        #    all larger sizes will be worse (lower CF → higher LCOH) ──
        if min_storage_lcoh is not None and min_storage_lcoh > LCOH_CUTOFF and not size_viable:
            remaining_sizes = [s for s in sizes_MW if s > sz]
            if remaining_sizes:
                skip_count += len(remaining_sizes) * len(storages_kg)
                print(f"  [{label}] {sz:.0f} MW LCOH €{min_storage_lcoh:.2f} > "
                      f"€{LCOH_CUTOFF:.0f} at min storage → REJECT "
                      f"{len(remaining_sizes)} larger sizes "
                      f"({remaining_sizes[0]:.0f}-{remaining_sizes[-1]:.0f} MW) "
                      f"as infeasible")
                break

    print(f"  [{label}] Done: {run_count} run, {skip_count} skipped "
          f"({skip_count/(run_count+skip_count)*100:.0f}% pruned)")
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
        # Sort by LCOH first, then SEC (lower is better for both)
        best = feasible.sort_values(['LCOH', 'SEC']).iloc[0]
        return best, True
    else:
        # Fallback: best reliability
        best = df.sort_values('demand_met_pct', ascending=False).iloc[0]
        return best, False


# =============================================================================
# PLOTTING  —  HEATMAPS (2×3 with demand met %)
# =============================================================================
def plot_heatmaps(df, re_frac, tag=""):
    """2×3 heatmap for a single RE fraction — includes demand met %."""
    sizes = sorted(df['size_MW'].unique())
    stores = sorted(df['storage_kg'].unique())

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(f'PEM Electrolyser Multi-Objective Optimization — {re_frac*100:.0f}% RE\n'
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

        # Annotate cells
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

        # Star best (multi-objective optimal)
        best_row, feasible = select_optimal(df)
        if col == 'LCOH' and best_row['size_MW'] in sizes and best_row['storage_kg'] in stores:
            bj = sizes.index(best_row['size_MW'])
            bi = stores.index(best_row['storage_kg'])
            ax.plot(bj+0.5, bi+0.5, '*', ms=18, color='red',
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)

    plt.tight_layout()
    fname = f'pem_heatmaps_RE{re_frac*100:.0f}{tag}.png'
    plt.savefig(OUTPUT / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {fname}")


# =============================================================================
# PLOTTING  —  RE SCENARIO COMPARISON
# =============================================================================
def plot_re_comparison(df_all):
    """4-panel: LCOH, Size+Storage, H2+NPV, Reliability vs RE%."""
    # Best per RE (multi-objective)
    records = []
    for re_f in sorted(df_all['re_fraction'].unique()):
        sub = df_all[df_all['re_fraction'] == re_f]
        best, feasible = select_optimal(sub)
        rec = best.to_dict()
        rec['feasible'] = feasible
        records.append(rec)
    opt = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PEM Electrolyser — Multi-Objective Optimization Results\n'
                 f'(Reliability ≥ {RELIABILITY_TARGET:.0%}, then min LCOH, then min SEC)',
                 fontsize=14, fontweight='bold')

    # Panel 1: LCOH vs RE%
    ax = axes[0, 0]
    ax.plot(opt['re_fraction']*100, opt['LCOH'], 'o-', color='#2E86AB', lw=2.5, ms=10)
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

    # Panel 2: Optimal size & storage
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.bar(opt['re_fraction']*100 - 2, opt['size_MW'], width=4, color='#06A77D',
           alpha=0.7, label='Size (MW)')
    ax2.bar(opt['re_fraction']*100 + 2, opt['storage_kg'], width=4, color='#7B2D8E',
            alpha=0.7, label='Storage (kg)')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Electrolyser Size (MW)', fontweight='bold', color='#06A77D')
    ax2.set_ylabel('Storage Size (kg)', fontweight='bold', color='#7B2D8E')
    ax.set_title('Optimal Size & Storage', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: H₂ production & demand met
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar(opt['re_fraction']*100, opt['H2_annual_t'], width=8, color='#2E86AB',
           alpha=0.7, label='H₂ Production (t/yr)')
    ax2.plot(opt['re_fraction']*100, opt['demand_met_pct'], 's-', color='#C73E1D',
             lw=2.5, ms=10, label='Demand Met (%)')
    ax2.axhline(RELIABILITY_TARGET*100, ls='--', color='red', alpha=0.5,
                label=f'Target ({RELIABILITY_TARGET:.0%})')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Annual H₂ Production (t)', fontweight='bold')
    ax2.set_ylabel('Demand Met (%)', fontweight='bold', color='#C73E1D')
    ax.set_title('H₂ Production & Reliability', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: CF & SEC
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(opt['re_fraction']*100, opt['CF']*100, 'o-', color='#F18F01', lw=2.5, ms=10,
            label='Capacity Factor (%)')
    ax2.plot(opt['re_fraction']*100, opt['SEC'], 's-', color='#555555', lw=2, ms=8,
             label='SEC (kWh/kg)')
    ax.set_xlabel('RE Availability (%)', fontweight='bold')
    ax.set_ylabel('Capacity Factor (%)', fontweight='bold', color='#F18F01')
    ax2.set_ylabel('SEC (kWh/kg)', fontweight='bold', color='#555555')
    ax.set_title('Efficiency Metrics', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'pem_re_scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved pem_re_scenario_comparison.png")


# =============================================================================
# PLOTTING  —  PARETO FRONT (100% RE): LCOH vs Reliability
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

    # Mark optimal
    best, feasible = select_optimal(df)
    ax.plot(best['LCOH'], best['demand_met_pct'], '*', ms=20, color='red',
            markeredgecolor='black', markeredgewidth=1.5, zorder=10,
            label=f"Optimal: {best['size_MW']:.0f}MW/{best['storage_kg']:.0f}kg\n"
                  f"LCOH €{best['LCOH']:.2f}, Met {best['demand_met_pct']:.1f}%")

    ax.set_xlabel('LCOH (€/kg)', fontweight='bold')
    ax.set_ylabel('Demand Met (%)', fontweight='bold')
    ax.set_title('PEM — LCOH vs Reliability Trade-off\n'
                 '(Colour = Size, Bubble = Storage)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: LCOH vs NPV (coloured by demand met)
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
    ax.set_title('PEM — LCOH vs NPV (colour = Reliability)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'pem_pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved pem_pareto_front.png")


# =============================================================================
# PLOTTING  —  SIZE VS METRICS (100% RE, at optimal storage)
# =============================================================================
def plot_size_metrics(df):
    """6-panel: metrics vs electrolyser size at optimal storage per size."""
    # For each size, pick the storage that gives best LCOH among reliable configs
    records = []
    for sz in sorted(df['size_MW'].unique()):
        sub = df[df['size_MW'] == sz]
        best_row, _ = select_optimal(sub)
        records.append(best_row.to_dict())
    df0 = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'PEM Electrolyser — Size vs Performance (Multi-Objective Optimal Storage)\n'
                 f'(132 MW RE, Reliability ≥ {RELIABILITY_TARGET:.0%})',
                 fontsize=14, fontweight='bold', y=1.02)

    panels = [
        ('size_MW', 'LCOH',           'LCOH (€/kg)',          '#2E86AB', 'o-'),
        ('size_MW', 'NPV_M',          'NPV (€ Million)',      '#06A77D', 'o-'),
        ('size_MW', 'demand_met_pct', 'Demand Met (%)',        '#C73E1D', 'o-'),
        ('size_MW', 'SEC',            'SEC (kWh/kg)',          '#7B2D8E', 'o-'),
        ('size_MW', 'CF',             'Capacity Factor',       '#F18F01', 'o-'),
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
    plt.savefig(OUTPUT / 'pem_size_vs_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved pem_size_vs_metrics.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("PEM ELECTROLYSER — MULTI-OBJECTIVE GRID OPTIMIZATION v3")
    print(f"Criteria: (1) Reliability ≥ {RELIABILITY_TARGET:.0%}  "
          f"(2) Min LCOH  (3) Min SEC")
    print("Fixed 132 MW RE  ·  6 RE scenarios  ·  15-year simulation")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1] Loading data …")
    power_1yr = load_power_data(MAT_PATH)
    demand_1yr = load_demand_data(DEMAND_PATH, power_1yr.index)
    print(f"    Power: {len(power_1yr)} h, max {power_1yr.max():.0f} kW "
          f"({power_1yr.max()/1e3:.1f} MW), mean {power_1yr.mean():.0f} kW")
    print(f"    Demand: {len(demand_1yr)} h, "
          f"total {demand_1yr.sum():.0f} kWh/yr = "
          f"~{demand_1yr.sum()/33.33/1000:.0f} t H₂/yr")

    # ── Grid definition ──────────────────────────────────────────────────────
    # Actual demand: ~5,800 kg/day → 2-3 day buffer = 11,600-17,400 kg
    # Storage must go well above 10,000 kg to find 95% reliability configs
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
        df_re = run_grid(power_1yr, demand_1yr, sizes_MW, storages,
                         re_frac=frac, label=f"RE{frac*100:.0f}")
        all_frames.append(df_re)

        # Save per-RE CSV
        csv_name = f"grid_search_RE{frac*100:.0f}.csv"
        df_re.to_csv(OUTPUT / csv_name, index=False)
        print(f"  ✓ {len(df_re)} rows → {csv_name}")

        # Multi-objective best
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
    df_all.to_csv(OUTPUT / 'pem_grid_search_all_RE.csv', index=False)
    print(f"\n✓ Combined: {len(df_all)} rows → pem_grid_search_all_RE.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[2] Generating plots …")

    # Heatmaps for each RE level
    for frac in re_fractions:
        sub = df_all[df_all['re_fraction'] == frac]
        if len(sub) > 0:
            plot_heatmaps(sub, frac)

    # RE comparison
    plot_re_comparison(df_all)

    # Pareto & size metrics at 100% RE
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

    # Also show min-LCOH (unconstrained) for comparison
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
        # Unconstrained
        unc = sub.loc[sub['LCOH'].idxmin()]
        # Multi-objective
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
