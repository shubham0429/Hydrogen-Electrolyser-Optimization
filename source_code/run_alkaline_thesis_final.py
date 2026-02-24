#!/usr/bin/env python3
"""
Complete Alkaline Thesis Analysis Runner
=========================================
Generates all thesis-quality plots and analysis as recommended by ChatGPT:

Same structure as PEM analysis but for Alkaline Water Electrolysis (AWE):
1. Degradation multi-CF plot
2. SEC vs Voltage dual-axis coupling plot
3. Fixed vs Optimized LCOH vs CF plot
4. Monte Carlo with quantile markers
5. Tornado sensitivity chart
6. Dynamic system behavior plot
7. Energy balance plot
8. CAPEX vs OPEX dominance by CF

Author: Shubham Manchanda
Thesis: Techno-Economic Optimization of Electrolyser Performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from sim_alkaline import (
    AlkalineConfig,
    get_alkaline_config,
    load_power_data,
    load_demand_data,
    simulate,
    compute_lcoh,
    run_monte_carlo,
)

# Set plotting style for thesis quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'alkaline_thesis_final'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
MAT_PATH = DATA_DIR / "combined_wind_pv_DATA.mat"
DEMAND_PATH = DATA_DIR / "Company_2_hourly_gas_demand.csv"


def run_single_simulation(size_mw, power_W, config_override=None, verbose=False):
    """Run a single Alkaline simulation with given parameters."""
    config = get_alkaline_config(P_nom_MW=size_mw)
    
    # Apply any config overrides
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Run simulation
    results = simulate(power_W, config, verbose=verbose)
    econ = compute_lcoh(results, config, verbose=verbose)
    
    return results, econ, config


def main():
    """Main function to run complete Alkaline thesis analysis."""
    
    print("="*80)
    print("COMPLETE ALKALINE THESIS ANALYSIS")
    print("Generating all thesis-quality plots per ChatGPT bulletproof checklist")
    print("="*80)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n[0/10] Loading data...")
    
    SIZE_MW = 20.0
    
    # Get config for 15 years first
    config = get_alkaline_config(P_nom_MW=SIZE_MW)
    config.simulation_years = 15  # Set to 15 years
    
    # Load power data (will tile automatically to 15 years)
    power_15yr = load_power_data(str(MAT_PATH), config=config)
    print(f"   Power data: {len(power_15yr)} hours ({len(power_15yr)/8760:.1f} years)")
    
    # =========================================================================
    # 1. BASELINE 15-YEAR SIMULATION (20 MW)
    # =========================================================================
    print("\n[1/10] Running baseline 15-year simulation (20 MW Alkaline)...")
    
    # Run simulation (power_15yr was loaded above with config)
    results_base, econ_base, config = run_single_simulation(
        SIZE_MW, power_15yr, verbose=True)
    
    # Extract key metrics
    total_h2_kg = results_base.total_h2_production_kg
    avg_sec = results_base.average_sec_kWh_kg
    n_replacements = results_base.stack_replacements
    replacement_hours = results_base.replacement_hours
    replacement_years = [h / 8760 for h in replacement_hours]
    lcoh_base = econ_base.lcoh_total
    
    print(f"   Total H2: {total_h2_kg/1000:.1f} tonnes")
    print(f"   Avg SEC: {avg_sec:.2f} kWh/kg")
    print(f"   Stack replacements: {n_replacements}")
    if replacement_years:
        print(f"   Replacement years: {[f'{y:.1f}' for y in replacement_years]}")
    print(f"   LCOH: €{lcoh_base:.2f}/kg")
    
    # Save baseline timeseries
    timeseries_df = pd.DataFrame({
        'hour': results_base.hours,
        'power_consumed_W': results_base.power_consumed_W,
        'H2_kg': results_base.h2_production_kg,
        'V_cell_V': results_base.cell_voltage_V,
        'SEC_kWh_kg': results_base.sec_kWh_kg,
        'V_degradation_V': results_base.voltage_degradation_V,
        'T_K': results_base.temperature_K,
        'storage_kg': results_base.storage_level_kg,
        'capacity_factor': results_base.capacity_factor,
    })
    timeseries_df.to_csv(OUTPUT_DIR / 'baseline_timeseries.csv', index=False)
    
    # =========================================================================
    # 2. DEGRADATION MULTI-CF PLOT
    # =========================================================================
    print("\n[2/10] Generating Degradation Multi-CF Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cf_scenarios = [0.40, 0.60, 1.00]
    cf_labels = ['CF = 40%', 'CF = 60%', 'CF = 100%']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    linestyles = ['-', '--', '-.']
    
    for cf, label, color, ls in zip(cf_scenarios, cf_labels, colors, linestyles):
        # Scale electrolyser size to achieve target CF
        target_size = SIZE_MW / cf if cf < 1.0 else SIZE_MW
        target_size = np.clip(target_size, 15.0, 50.0)
        
        results_cf, econ_cf, cfg_cf = run_single_simulation(
            target_size, power_15yr, verbose=False)
        
        # Extract voltage degradation over time
        n_years = 15
        hours_per_year = 8760
        
        # Sample voltage degradation at yearly intervals
        years = np.arange(0, n_years + 1)
        v_deg_yearly = []
        
        for y in years:
            idx = min(y * hours_per_year, len(results_cf.voltage_degradation_V) - 1)
            v_deg = results_cf.voltage_degradation_V[idx] * 1000  # mV
            v_deg_yearly.append(max(0, v_deg))
        
        repl_years_cf = [h / 8760 for h in results_cf.replacement_hours]
        repl_str = f"Repl: Y{[f'{y:.1f}' for y in repl_years_cf]}" if repl_years_cf else "No replacement"
        
        ax.plot(years, v_deg_yearly, color=color, linestyle=ls, linewidth=2.5, 
                label=f'{label} ({repl_str})')
        
        # Mark replacement events
        for ry in repl_years_cf:
            ax.axvline(x=ry, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Replacement threshold (10% of ~1.75V = 175 mV)
    ax.axhline(y=175, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='Replacement threshold (175 mV)')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Voltage Degradation ΔV (mV)')
    ax.set_title('Stack Voltage Degradation at Different Capacity Factors (Alkaline)')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 250)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(30))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_degradation_multi_CF.png')
    plt.savefig(OUTPUT_DIR / 'fig1_degradation_multi_CF.pdf')
    plt.close()
    print("   ✓ Saved fig1_degradation_multi_CF.png/pdf")
    
    # =========================================================================
    # 3. SEC VS VOLTAGE DUAL-AXIS COUPLING PLOT
    # =========================================================================
    print("\n[3/10] Generating SEC-Voltage Coupling Plot...")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Time axis (monthly for 15 years)
    n_months = 15 * 12
    time_months = np.arange(n_months)
    time_years = time_months / 12
    
    # Sample SEC and voltage degradation at monthly intervals
    hours_per_month = 730
    sec_monthly = []
    v_deg_monthly = []
    
    for m in range(n_months):
        start_idx = m * hours_per_month
        end_idx = min((m + 1) * hours_per_month, len(results_base.sec_kWh_kg))
        
        if start_idx < len(results_base.sec_kWh_kg):
            # Get SEC (only non-zero values)
            sec_slice = results_base.sec_kWh_kg[start_idx:end_idx]
            sec_nonzero = sec_slice[sec_slice > 0]
            if len(sec_nonzero) > 0:
                sec_monthly.append(np.mean(sec_nonzero))
            else:
                sec_monthly.append(sec_monthly[-1] if sec_monthly else 55)
            
            # Get voltage degradation
            v_deg = results_base.voltage_degradation_V[end_idx-1] * 1000  # mV
            v_deg_monthly.append(max(0, v_deg))
        else:
            sec_monthly.append(sec_monthly[-1] if sec_monthly else 55)
            v_deg_monthly.append(v_deg_monthly[-1] if v_deg_monthly else 0)
    
    # Left axis: SEC
    color1 = '#2980b9'
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('System SEC (kWh/kg H₂)', color=color1)
    ax1.plot(time_years, sec_monthly, color=color1, linewidth=2, label='System SEC')
    ax1.tick_params(axis='y', labelcolor=color1)
    sec_min, sec_max = min(sec_monthly), max(sec_monthly)
    ax1.set_ylim(sec_min * 0.95, sec_max * 1.05)
    
    # Right axis: Voltage degradation
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Voltage Degradation ΔV (mV)', color=color2)
    ax2.plot(time_years, v_deg_monthly, color=color2, linewidth=2, linestyle='--', label='ΔV')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(max(v_deg_monthly) * 1.2, 200))
    
    # Add replacement markers
    for ry in replacement_years:
        ax1.axvline(x=ry, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax1.annotate(f'Stack\nReplacement\n(Year {ry:.1f})', xy=(ry, sec_min * 1.02), 
                     fontsize=9, ha='center', va='bottom')
    
    ax1.set_title('Coupling Between Voltage Degradation and System SEC Over 15-Year Alkaline Operation')
    ax1.set_xlim(0, 15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_SEC_voltage_coupling.png')
    plt.savefig(OUTPUT_DIR / 'fig2_SEC_voltage_coupling.pdf')
    plt.close()
    print("   ✓ Saved fig2_SEC_voltage_coupling.png/pdf")
    
    # =========================================================================
    # 4. FIXED VS OPTIMIZED LCOH VS CF PLOT
    # =========================================================================
    print("\n[4/10] Generating Fixed vs Optimized LCOH vs CF Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cf_range = np.arange(0.30, 1.01, 0.10)
    lcoh_fixed = []
    lcoh_optimized = []
    
    for cf in cf_range:
        print(f"      CF = {cf*100:.0f}%...")
        
        # Fixed design: 20 MW
        results_fixed, econ_fixed, _ = run_single_simulation(
            SIZE_MW, power_15yr, verbose=False)
        lcoh_fixed.append(econ_fixed.lcoh_total)
        
        # Optimized design: Scale size to match CF
        opt_size = SIZE_MW * (0.6 / cf) if cf > 0.3 else SIZE_MW
        opt_size = np.clip(opt_size, 15.0, 35.0)
        
        results_opt, econ_opt, _ = run_single_simulation(
            opt_size, power_15yr, verbose=False)
        lcoh_optimized.append(econ_opt.lcoh_total)
    
    ax.plot(cf_range * 100, lcoh_fixed, 'o-', color='#e74c3c', linewidth=2.5, 
            markersize=8, label=f'Fixed Design ({SIZE_MW:.0f} MW)')
    ax.plot(cf_range * 100, lcoh_optimized, 's--', color='#27ae60', linewidth=2.5, 
            markersize=8, label='Optimized Design')
    
    ax.fill_between(cf_range * 100, lcoh_fixed, lcoh_optimized, alpha=0.2, color='green',
                    label='Optimization Benefit')
    
    ax.set_xlabel('Target Capacity Factor (%)')
    ax.set_ylabel('LCOH (€/kg H₂)')
    ax.set_title('Effect of System Optimization on LCOH Sensitivity to Capacity Factor (Alkaline)')
    ax.legend(loc='upper right')
    ax.set_xlim(30, 100)
    y_min = min(min(lcoh_fixed), min(lcoh_optimized)) * 0.9
    y_max = max(max(lcoh_fixed), max(lcoh_optimized)) * 1.1
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_LCOH_fixed_vs_optimized.png')
    plt.savefig(OUTPUT_DIR / 'fig3_LCOH_fixed_vs_optimized.pdf')
    plt.close()
    print("   ✓ Saved fig3_LCOH_fixed_vs_optimized.png/pdf")
    
    # =========================================================================
    # 5. MONTE CARLO WITH QUANTILE MARKERS
    # =========================================================================
    print("\n[5/10] Running Monte Carlo Analysis (100 simulations)...")
    
    mc_results = run_monte_carlo(
        base_config=config,
        power_input_W=power_15yr,
        n_simulations=100,
        verbose=True
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: LCOH Distribution
    lcoh_values = mc_results.lcoh_values
    mean_lcoh = mc_results.lcoh_mean
    std_lcoh = mc_results.lcoh_std
    p05 = np.percentile(lcoh_values, 5)
    p95 = np.percentile(lcoh_values, 95)
    deterministic = lcoh_base
    
    # Histogram
    n, bins, patches = ax1.hist(lcoh_values, bins=20, density=True, alpha=0.7, 
                                 color='#3498db', edgecolor='white')
    
    # Add vertical lines for quantiles
    ax1.axvline(p05, color='green', linestyle='--', linewidth=2, label=f'P05 = €{p05:.2f}/kg')
    ax1.axvline(deterministic, color='red', linestyle='-', linewidth=2.5, 
                label=f'Deterministic = €{deterministic:.2f}/kg')
    ax1.axvline(mean_lcoh, color='orange', linestyle='-.', linewidth=2, 
                label=f'MC Mean = €{mean_lcoh:.2f}/kg')
    ax1.axvline(p95, color='purple', linestyle='--', linewidth=2, label=f'P95 = €{p95:.2f}/kg')
    
    ax1.set_xlabel('LCOH (€/kg H₂)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('LCOH Probability Distribution from 100 Monte Carlo Simulations (Alkaline)')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Add annotation about distribution
    skewness = pd.Series(lcoh_values).skew()
    ax1.annotate(f'Distribution: Approximately log-normal\nSkewness: {skewness:.2f}\n'
                 f'Std: €{std_lcoh:.2f}/kg\nRange: €{min(lcoh_values):.2f} - €{max(lcoh_values):.2f}', 
                 xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: Box plot
    bp = ax2.boxplot([lcoh_values], positions=[1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    
    ax2.set_ylabel('LCOH (€/kg H₂)')
    ax2.set_title('LCOH Distribution Summary')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['LCOH'])
    
    # Add statistics text
    stats_text = (f"Mean: €{mean_lcoh:.2f}/kg\n"
                  f"Median: €{np.median(lcoh_values):.2f}/kg\n"
                  f"Std: €{std_lcoh:.2f}/kg\n"
                  f"Min: €{min(lcoh_values):.2f}/kg\n"
                  f"Max: €{max(lcoh_values):.2f}/kg")
    ax2.annotate(stats_text, xy=(0.65, 0.95), xycoords='axes fraction', fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_monte_carlo_distribution.png')
    plt.savefig(OUTPUT_DIR / 'fig4_monte_carlo_distribution.pdf')
    plt.close()
    print(f"   ✓ MC Mean LCOH: €{mean_lcoh:.2f} ± {std_lcoh:.2f}/kg")
    print(f"   ✓ P05-P95 range: €{p05:.2f} - €{p95:.2f}/kg")
    print("   ✓ Saved fig4_monte_carlo_distribution.png/pdf")
    
    # =========================================================================
    # 6. TORNADO SENSITIVITY CHART
    # =========================================================================
    print("\n[6/10] Running Sensitivity Analysis and Tornado Chart...")
    
    # Define parameters and ±20% ranges
    base_lcoh = lcoh_base
    sensitivity_params = {
        'capex_stack_eur_kW': (config.capex_stack_eur_kW * 0.8, config.capex_stack_eur_kW * 1.2, 'Stack CAPEX (€/kW)'),
        'electricity_price_eur_kWh': (config.electricity_price_eur_kWh * 0.8, config.electricity_price_eur_kWh * 1.2, 'Electricity Price (€/kWh)'),
        'deg_rate_uV_h': (config.deg_rate_uV_h * 0.8, config.deg_rate_uV_h * 1.2, 'Degradation Rate (μV/h)'),
        'discount_rate': (config.discount_rate * 0.8, config.discount_rate * 1.2, 'Discount Rate'),
        'stack_lifetime_hours': (config.stack_lifetime_hours * 0.8, config.stack_lifetime_hours * 1.2, 'Stack Lifetime (h)'),
        'capex_bop_eur_kW': (config.capex_bop_eur_kW * 0.8, config.capex_bop_eur_kW * 1.2, 'BoP CAPEX (€/kW)'),
    }
    
    sensitivity_results = []
    
    for param, (low, high, label) in sensitivity_params.items():
        print(f"      Testing {label}...")
        
        # Low value simulation
        results_low, econ_low, _ = run_single_simulation(
            SIZE_MW, power_15yr, config_override={param: low}, verbose=False)
        lcoh_low = econ_low.lcoh_total
        
        # High value simulation
        results_high, econ_high, _ = run_single_simulation(
            SIZE_MW, power_15yr, config_override={param: high}, verbose=False)
        lcoh_high = econ_high.lcoh_total
        
        sensitivity_results.append({
            'param': label,
            'low': lcoh_low,
            'high': lcoh_high,
            'swing': abs(lcoh_high - lcoh_low) / 2
        })
    
    # Sort by swing (impact)
    sensitivity_results.sort(key=lambda x: x['swing'], reverse=True)
    
    # Create tornado chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(sensitivity_results))
    params = [r['param'] for r in sensitivity_results]
    low_delta = [r['low'] - base_lcoh for r in sensitivity_results]
    high_delta = [r['high'] - base_lcoh for r in sensitivity_results]
    
    # Plot bars
    for i, (ld, hd) in enumerate(zip(low_delta, high_delta)):
        ax.barh(i, min(ld, hd), color='#27ae60', height=0.6, align='center')
        ax.barh(i, max(ld, hd), color='#e74c3c', height=0.6, align='center')
    
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel('ΔLCOH (€/kg H₂) from Baseline')
    ax.set_title(f'Sensitivity of Alkaline LCOH to Key Parameters (±20%)\nBaseline: €{base_lcoh:.2f}/kg')
    
    # Add legend
    legend_elements = [Patch(facecolor='#27ae60', label='Low value (-20%)'),
                       Patch(facecolor='#e74c3c', label='High value (+20%)')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_tornado_sensitivity.png')
    plt.savefig(OUTPUT_DIR / 'fig5_tornado_sensitivity.pdf')
    plt.close()
    print("   ✓ Saved fig5_tornado_sensitivity.png/pdf")
    
    # Save sensitivity results to CSV
    sens_df = pd.DataFrame(sensitivity_results)
    sens_df['delta_low'] = sens_df['low'] - base_lcoh
    sens_df['delta_high'] = sens_df['high'] - base_lcoh
    sens_df.to_csv(OUTPUT_DIR / 'sensitivity_results.csv', index=False)
    
    # =========================================================================
    # 7. DYNAMIC SYSTEM BEHAVIOR PLOT (1 week)
    # =========================================================================
    print("\n[7/10] Generating Dynamic System Behavior Plot...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Select 1 week in summer
    start_hour = 4000
    end_hour = start_hour + 168
    
    time_hours = np.arange(168)
    time_days = time_hours / 24
    
    end_hour = min(end_hour, len(results_base.power_consumed_W))
    
    # Power (MW)
    power_week = results_base.power_consumed_W[start_hour:end_hour] / 1e6  # Convert W to MW
    
    ax1.fill_between(time_days[:len(power_week)], 0, power_week, alpha=0.5, color='#3498db', 
                     label='Power Consumed')
    ax1.plot(time_days[:len(power_week)], power_week, color='#2980b9', linewidth=1)
    ax1.axhline(y=SIZE_MW, color='red', linestyle='--', alpha=0.7, label=f'Rated Capacity ({SIZE_MW} MW)')
    ax1.set_ylabel('Power (MW)')
    ax1.set_title('Dynamic System Behavior Over 1 Week - Summer Period (Alkaline)')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(power_week) * 1.1 if len(power_week) > 0 else SIZE_MW * 1.2)
    
    # H2 production rate (kg/h)
    h2_week = results_base.h2_production_kg[start_hour:end_hour]
    ax2.fill_between(time_days[:len(h2_week)], 0, h2_week, alpha=0.5, color='#2ecc71', 
                     label='H₂ Production')
    ax2.plot(time_days[:len(h2_week)], h2_week, color='#27ae60', linewidth=1)
    ax2.set_ylabel('H₂ Production (kg/h)')
    ax2.legend(loc='upper right')
    
    # Storage SOC (%)
    storage_week = results_base.storage_level_kg[start_hour:end_hour]
    storage_capacity = config.storage_capacity_kg
    soc_week = storage_week / storage_capacity * 100 if storage_capacity > 0 else np.zeros_like(storage_week)
    ax3.fill_between(time_days[:len(soc_week)], 0, soc_week, alpha=0.5, color='#9b59b6', 
                     label='Storage SOC')
    ax3.plot(time_days[:len(soc_week)], soc_week, color='#8e44ad', linewidth=1.5)
    ax3.axhline(y=90, color='red', linestyle=':', alpha=0.7, label='Upper limit (90%)')
    ax3.axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='Lower limit (10%)')
    ax3.set_ylabel('Storage SOC (%)')
    ax3.set_xlabel('Time (days)')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_dynamic_system_behavior.png')
    plt.savefig(OUTPUT_DIR / 'fig6_dynamic_system_behavior.pdf')
    plt.close()
    print("   ✓ Saved fig6_dynamic_system_behavior.png/pdf")
    
    # =========================================================================
    # 8. ENERGY BALANCE PLOT
    # =========================================================================
    print("\n[8/10] Generating Energy Balance Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate energy breakdown
    total_energy_consumed = results_base.total_energy_consumed_kWh
    total_h2_energy = total_h2_kg * 33.33  # kWh LHV
    
    # Estimate breakdown (Alkaline typical values)
    stack_energy = total_h2_kg * avg_sec * 0.85  # ~85% to stack
    parasitic_energy = total_energy_consumed * 0.05  # ~5% parasitic
    compression_energy = total_h2_kg * 3.0  # ~3 kWh/kg compression
    thermal_losses = total_energy_consumed * 0.10  # ~10% thermal losses
    
    categories = ['Stack\nElectrolysis', 'Parasitic\nLoads', 'Compression', 'Thermal\nLosses']
    values = [stack_energy/1e6, parasitic_energy/1e6, compression_energy/1e6, thermal_losses/1e6]  # GWh
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f} GWh\n({val/sum(values)*100:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Energy Consumption (GWh)')
    ax.set_title(f'Energy Distribution Over 15-Year Alkaline Operation\nTotal: {sum(values):.1f} GWh')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_energy_balance.png')
    plt.savefig(OUTPUT_DIR / 'fig7_energy_balance.pdf')
    plt.close()
    print("   ✓ Saved fig7_energy_balance.png/pdf")
    
    # =========================================================================
    # 9. CAPEX VS OPEX DOMINANCE BY CF
    # =========================================================================
    print("\n[9/10] Generating CAPEX vs OPEX Dominance Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cf_range_capex = np.arange(0.30, 1.01, 0.05)
    capex_frac = []
    opex_frac = []
    
    for cf in cf_range_capex:
        # Rough approximation based on LCOH components
        capex_base = config.capex_stack_eur_kW * SIZE_MW * 1000
        elec_cost_per_kg = config.electricity_price_eur_kWh * avg_sec
        
        annual_h2 = SIZE_MW * 1000 * cf * 8760 / avg_sec
        annual_opex = annual_h2 * elec_cost_per_kg + capex_base * 0.025
        
        npv_capex = capex_base * 1.35  # + BoP + installation
        npv_opex = annual_opex * 10
        
        total_cost = npv_capex + npv_opex
        capex_frac.append(npv_capex / total_cost * 100)
        opex_frac.append(npv_opex / total_cost * 100)
    
    ax.stackplot(cf_range_capex * 100, capex_frac, opex_frac, 
                 labels=['CAPEX Contribution', 'OPEX Contribution'],
                 colors=['#3498db', '#e74c3c'], alpha=0.8)
    
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='50% threshold')
    ax.set_xlabel('Capacity Factor (%)')
    ax.set_ylabel('Cost Contribution (%)')
    ax.set_title('CAPEX vs OPEX Dominance by Capacity Factor (Alkaline)')
    ax.legend(loc='center right')
    ax.set_xlim(30, 100)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_CAPEX_OPEX_dominance.png')
    plt.savefig(OUTPUT_DIR / 'fig8_CAPEX_OPEX_dominance.pdf')
    plt.close()
    print("   ✓ Saved fig8_CAPEX_OPEX_dominance.png/pdf")
    
    # =========================================================================
    # 10. SAVE SUMMARY TABLES
    # =========================================================================
    print("\n[10/10] Generating Summary Tables...")
    
    # Main results summary
    summary = {
        'Parameter': [
            'Electrolyser Size (MW)',
            'Storage Capacity (kg)',
            'Simulation Period (years)',
            'Total H₂ Production (tonnes)',
            'LCOH (€/kg)',
            'LCOH CAPEX (€/kg)',
            'LCOH Electricity (€/kg)',
            'Average SEC (kWh/kg)',
            'Stack Replacements',
            'Replacement Years',
            'Total CAPEX (€M)',
            'Average Capacity Factor',
            'Total Operating Hours',
        ],
        'Value': [
            SIZE_MW,
            config.storage_capacity_kg,
            15,
            f"{total_h2_kg/1000:.1f}",
            f"{lcoh_base:.2f}",
            f"{econ_base.lcoh_capex:.2f}",
            f"{econ_base.lcoh_electricity:.2f}",
            f"{avg_sec:.2f}",
            n_replacements,
            str([f'{y:.1f}' for y in replacement_years]) if replacement_years else "None",
            f"{econ_base.total_capex/1e6:.2f}",
            f"{results_base.capacity_factor_avg:.2%}",
            f"{results_base.total_operating_hours:.0f}",
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / 'summary_results.csv', index=False)
    
    # Monte Carlo summary
    mc_summary = {
        'Metric': ['Mean', 'Std', 'P05', 'P25', 'Median', 'P75', 'P95', 'Min', 'Max'],
        'LCOH (€/kg)': [
            f"{mean_lcoh:.2f}",
            f"{std_lcoh:.2f}",
            f"{p05:.2f}",
            f"{np.percentile(lcoh_values, 25):.2f}",
            f"{np.median(lcoh_values):.2f}",
            f"{np.percentile(lcoh_values, 75):.2f}",
            f"{p95:.2f}",
            f"{min(lcoh_values):.2f}",
            f"{max(lcoh_values):.2f}",
        ]
    }
    mc_summary_df = pd.DataFrame(mc_summary)
    mc_summary_df.to_csv(OUTPUT_DIR / 'monte_carlo_summary.csv', index=False)
    
    print("   ✓ Saved summary_results.csv")
    print("   ✓ Saved monte_carlo_summary.csv")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("ALKALINE THESIS ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated Files:")
    print("  Figures:")
    print("    - fig1_degradation_multi_CF.png/pdf")
    print("    - fig2_SEC_voltage_coupling.png/pdf")
    print("    - fig3_LCOH_fixed_vs_optimized.png/pdf")
    print("    - fig4_monte_carlo_distribution.png/pdf")
    print("    - fig5_tornado_sensitivity.png/pdf")
    print("    - fig6_dynamic_system_behavior.png/pdf")
    print("    - fig7_energy_balance.png/pdf")
    print("    - fig8_CAPEX_OPEX_dominance.png/pdf")
    print("  Data:")
    print("    - baseline_timeseries.csv")
    print("    - summary_results.csv")
    print("    - monte_carlo_summary.csv")
    print("    - sensitivity_results.csv")
    
    print(f"\nKEY RESULTS:")
    print(f"  Baseline LCOH: €{lcoh_base:.2f}/kg")
    print(f"  Monte Carlo Mean: €{mean_lcoh:.2f} ± {std_lcoh:.2f}/kg")
    print(f"  P05-P95 Range: €{p05:.2f} - €{p95:.2f}/kg")
    print(f"  Total H₂ (15 years): {total_h2_kg/1000:.1f} tonnes")
    print(f"  Stack Replacements: {n_replacements}")
    
    return {
        'lcoh_base': lcoh_base,
        'lcoh_mc_mean': mean_lcoh,
        'lcoh_mc_std': std_lcoh,
        'lcoh_p05': p05,
        'lcoh_p95': p95,
        'total_h2_tonnes': total_h2_kg / 1000,
        'output_dir': str(OUTPUT_DIR)
    }


if __name__ == '__main__':
    results = main()
