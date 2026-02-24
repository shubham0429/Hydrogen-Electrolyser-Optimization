#!/usr/bin/env python3
"""
Optimization Heatmaps — LCOH + Demand Satisfaction with Optimal Star
===================================                    ax1.text(j+0.5, i+0.5, f'{val:.2f}', ha='center', va='center',=================================

Reads pre-computed grid search CSVs and generates clean 2-panel heatmaps:
  Left:  LCOH (€/kg)        — cost landscape
  Right: Demand Met (%)      — reliability landscape

Both panels show:
  ★ Red star  = optimal configuration (lowest LCOH among ≥95% demand met)
  ── White contour at 95% demand met (the feasibility boundary)

One figure per RE scenario, for both PEM and Alkaline.

Usage:
    cd source_code
    python optimization_heatmaps_with_demand.py

Author: Shubham Manchanda
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
PEM_CSV   = BASE / "results" / "data" / "pem_grid_search_all_RE.csv"
ALK_CSV   = BASE / "results" / "data" / "alkaline_grid_search_all_RE.csv"
OUTPUT    = BASE / "results" / "optimization_heatmaps"
OUTPUT.mkdir(parents=True, exist_ok=True)

RELIABILITY_TARGET = 95.0  # percent

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# OPTIMAL SELECTION (same logic as optimization scripts)
# =============================================================================
def select_optimal(df):
    """
    1. Filter: demand_met_pct >= 95%
    2. Rank: LCOH ascending, then SEC ascending
    3. Pick: first row
    Fallback: highest demand_met_pct if none meets 95%.
    """
    feasible = df[df['demand_met_pct'] >= RELIABILITY_TARGET]
    if len(feasible) > 0:
        best = feasible.sort_values(['LCOH', 'SEC']).iloc[0]
        return best, True
    else:
        best = df.sort_values('demand_met_pct', ascending=False).iloc[0]
        return best, False


# =============================================================================
# HEATMAP PLOTTING — 2 panels: LCOH + Demand Met
# =============================================================================
def plot_heatmap_pair(df_re, re_frac, tech_name, tech_color):
    """
    Generate a 2-panel heatmap for one RE scenario.
    
    Left:  LCOH (€/kg) with star on optimal
    Right: Demand Met (%) with star on optimal + 95% contour line
    """
    sizes = sorted(df_re['size_MW'].unique())
    stores = sorted(df_re['storage_kg'].unique())
    
    # Find optimal
    best, is_feasible = select_optimal(df_re)
    opt_size = best['size_MW']
    opt_store = best['storage_kg']
    opt_lcoh = best['LCOH']
    opt_demand = best['demand_met_pct']
    opt_sec = best['SEC']
    
    # Pivot tables
    lcoh_pivot = df_re.pivot_table(values='LCOH', index='storage_kg', columns='size_MW')
    lcoh_pivot = lcoh_pivot.reindex(index=stores, columns=sizes)
    
    demand_pivot = df_re.pivot_table(values='demand_met_pct', index='storage_kg', columns='size_MW')
    demand_pivot = demand_pivot.reindex(index=stores, columns=sizes)
    
    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    feasible_str = "FEASIBLE" if is_feasible else "NO CONFIG MEETS 95%"
    fig.suptitle(
        f'{tech_name} Optimization - {re_frac*100:.0f}% RE (132 MW Wind+PV)\n'
        f'Optimal: {opt_size:.0f} MW, {opt_store:.0f} kg  |  '
        f'LCOH: {opt_lcoh:.2f} EUR/kg  |  Demand Met: {opt_demand:.1f}%  |  SEC: {opt_sec:.1f} kWh/kg  [{feasible_str}]',
        fontsize=13, fontweight='bold', y=1.03
    )
    
    # ── LEFT PANEL: LCOH ──────────────────────────────────────────────────────
    im1 = ax1.pcolormesh(
        range(len(sizes)+1), range(len(stores)+1),
        lcoh_pivot.values, cmap='YlGnBu_r', shading='flat'
    )
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('LCOH (€/kg)', fontweight='bold')
    
    # Annotate every cell with LCOH value
    for i, st in enumerate(stores):
        for j, sz in enumerate(sizes):
            val = lcoh_pivot.loc[st, sz] if (st in lcoh_pivot.index and sz in lcoh_pivot.columns) else np.nan
            if np.isfinite(val):
                vmin, vmax = np.nanmin(lcoh_pivot.values), np.nanmax(lcoh_pivot.values)
                mid = (vmin + vmax) / 2
                color = 'white' if val > mid else 'black'
                ax1.text(j+0.5, i+0.5, f'€{val:.2f}', ha='center', va='center',
                         fontsize=7, color=color, fontweight='bold')
    
    ax1.set_title('LCOH (EUR/kg H2)', fontweight='bold', fontsize=13)
    
    # ── RIGHT PANEL: DEMAND MET ───────────────────────────────────────────────
    # Custom colormap: red below 95%, green above
    im2 = ax2.pcolormesh(
        range(len(sizes)+1), range(len(stores)+1),
        demand_pivot.values, cmap='RdYlGn', shading='flat',
        vmin=0, vmax=100
    )
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label('Demand Met (%)', fontweight='bold')
    
    # Annotate every cell with demand met value
    for i, st in enumerate(stores):
        for j, sz in enumerate(sizes):
            val = demand_pivot.loc[st, sz] if (st in demand_pivot.index and sz in demand_pivot.columns) else np.nan
            if np.isfinite(val):
                # Bold + different style for cells meeting 95%
                if val >= RELIABILITY_TARGET:
                    ax2.text(j+0.5, i+0.5, f'{val:.1f}%', ha='center', va='center',
                             fontsize=7, color='black', fontweight='bold')
                else:
                    ax2.text(j+0.5, i+0.5, f'{val:.1f}%', ha='center', va='center',
                             fontsize=6.5, color='white', fontweight='normal')
    
    # Draw 95% contour line on demand panel
    try:
        X, Y = np.meshgrid(
            [j+0.5 for j in range(len(sizes))],
            [i+0.5 for i in range(len(stores))]
        )
        ax2.contour(X, Y, demand_pivot.values, levels=[RELIABILITY_TARGET],
                    colors='white', linewidths=2.5, linestyles='--')
        # Label the contour
        ax2.text(0.02, 0.98, '--- 95% threshold', transform=ax2.transAxes,
                 fontsize=9, color='white', fontweight='bold',
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    except Exception:
        pass  # contour may fail if all values are on one side
    
    ax2.set_title('Demand Satisfaction (%)', fontweight='bold', fontsize=13)
    
    # ── STAR ON BOTH PANELS ───────────────────────────────────────────────────
    if opt_size in sizes and opt_store in stores:
        bj = sizes.index(opt_size)
        bi = stores.index(opt_store)
        for ax in [ax1, ax2]:
            ax.plot(bj+0.5, bi+0.5, '*', ms=22, color='red',
                    markeredgecolor='white', markeredgewidth=2.0, zorder=10)
    
    # ── AXES FORMATTING (both panels) ─────────────────────────────────────────
    for ax in [ax1, ax2]:
        ax.set_xticks([i+0.5 for i in range(len(sizes))])
        ax.set_xticklabels([f'{s:.0f}' for s in sizes])
        ax.set_yticks([i+0.5 for i in range(len(stores))])
        ax.set_yticklabels([f'{s/1000:.0f}k' for s in stores])
        ax.set_xlabel('Electrolyser Size (MW)', fontweight='bold')
        ax.set_ylabel('Storage Capacity (kg)', fontweight='bold')
    
    plt.tight_layout()
    
    tag = tech_name.lower().replace(' ', '_')
    fname = f'{tag}_RE{re_frac*100:.0f}_lcoh_demand.png'
    fig.savefig(OUTPUT / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}  —  Optimal: {opt_size:.0f} MW, {opt_store:.0f} kg, "
          f"LCOH €{opt_lcoh:.2f}, Demand {opt_demand:.1f}%")
    
    return {
        're_fraction': re_frac,
        'optimal_size_MW': opt_size,
        'optimal_storage_kg': opt_store,
        'LCOH_eur_kg': opt_lcoh,
        'demand_met_pct': opt_demand,
        'SEC_kWh_kg': opt_sec,
        'feasible': is_feasible,
    }


# =============================================================================
# SUMMARY TABLE PLOT
# =============================================================================
def plot_summary_table(records, tech_name):
    """Generate a clean summary table image showing optima per RE scenario."""
    df = pd.DataFrame(records)
    
    fig, ax = plt.subplots(figsize=(12, max(3, 1 + 0.5 * len(df))))
    ax.axis('off')
    ax.set_title(f'{tech_name} — Optimal Configuration per RE Scenario\n'
                 f'(Selection: min LCOH where Demand Met ≥ 95%)',
                 fontsize=13, fontweight='bold', pad=20)
    
    headers = ['RE %', 'Size (MW)', 'Storage (kg)', 'LCOH (EUR/kg)',
               'Demand Met (%)', 'SEC (kWh/kg)', 'Feasible']
    
    cell_data = []
    cell_colors = []
    for _, row in df.iterrows():
        cell_data.append([
            f"{row['re_fraction']*100:.0f}%",
            f"{row['optimal_size_MW']:.0f}",
            f"{row['optimal_storage_kg']:,.0f}",
            f"{row['LCOH_eur_kg']:.2f}",
            f"{row['demand_met_pct']:.1f}%",
            f"{row['SEC_kWh_kg']:.1f}",
            "Yes" if row['feasible'] else "No"
        ])
        bg = '#d4edda' if row['feasible'] else '#f8d7da'  # green/red tint
        cell_colors.append([bg]*7)
    
    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=['#e8e8e8']*7,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    
    # Bold headers
    for j in range(len(headers)):
        table[0, j].set_text_props(fontweight='bold')
    
    tag = tech_name.lower().replace(' ', '_')
    fname = f'{tag}_optimal_summary.png'
    fig.savefig(OUTPUT / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("OPTIMIZATION HEATMAPS — LCOH + DEMAND SATISFACTION")
    print("="*70)
    
    # ── Load pre-computed results ─────────────────────────────────────────────
    datasets = []
    
    if PEM_CSV.exists():
        df_pem = pd.read_csv(PEM_CSV)
        datasets.append(('PEM Electrolyser', df_pem, '#1f77b4'))
        print(f"\nLoaded PEM: {len(df_pem)} rows from {PEM_CSV.name}")
    else:
        print(f"\n⚠ PEM CSV not found: {PEM_CSV}")
    
    if ALK_CSV.exists():
        df_alk = pd.read_csv(ALK_CSV)
        datasets.append(('Alkaline Electrolyser', df_alk, '#2ca02c'))
        print(f"Loaded Alkaline: {len(df_alk)} rows from {ALK_CSV.name}")
    else:
        print(f"⚠ Alkaline CSV not found: {ALK_CSV}")
    
    if not datasets:
        print("\nNo data found. Run pem_optimization_v3.py and/or "
              "alkaline_optimization_v3.py first.")
        return
    
    # ── Generate heatmaps per technology × RE scenario ────────────────────────
    for tech_name, df, color in datasets:
        print(f"\n{'─'*50}")
        print(f"  {tech_name}")
        print(f"{'─'*50}")
        
        re_fracs = sorted(df['re_fraction'].unique())
        records = []
        
        for re_frac in re_fracs:
            df_re = df[df['re_fraction'] == re_frac].copy()
            rec = plot_heatmap_pair(df_re, re_frac, tech_name, color)
            records.append(rec)
        
        # Summary table
        plot_summary_table(records, tech_name)
        
        # Also save as CSV
        tag = tech_name.lower().replace(' ', '_')
        pd.DataFrame(records).to_csv(
            OUTPUT / f'{tag}_optimal_per_RE.csv', index=False)
        print(f"  ✓ {tag}_optimal_per_RE.csv")
    
    print(f"\n{'='*70}")
    print(f"ALL DONE — Output folder: {OUTPUT}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
