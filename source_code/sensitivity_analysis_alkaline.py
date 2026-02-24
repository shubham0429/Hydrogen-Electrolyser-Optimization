"""
Sensitivity Analysis for Alkaline Electrolyser LCOH
=====================================================

Creates tornado diagrams and sensitivity analysis showing the impact 
of each uncertain parameter on LCOH for Alkaline electrolysers.

Key differences from PEM sensitivity:
- Different parameter ranges (lower CAPEX, different efficiency)
- Different degradation model (linear + cycling)
- Different dominant parameters

Usage:
    python sensitivity_analysis_alkaline.py
    python sensitivity_analysis_alkaline.py --size 20 --output results/sensitivity
    
Author: Thesis Project
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import from sim_alkaline
from sim_alkaline import (
    get_alkaline_config,
    simulate,
    compute_lcoh,
    SimulationResults,
    EconomicResults,
    AlkalineConfig
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    
    # System parameters
    size_MW: float = 20.0
    simulation_years: int = 15  # Use 15 years for full lifecycle
    
    # Base economic parameters
    base_electricity_price: float = 0.07
    base_h2_price: float = 6.0
    
    # Variation range (±% from base)
    variation_pct: float = 20.0  # ±20% for most parameters
    
    # Output settings
    output_folder: str = 'results/alkaline_sensitivity'
    
    # Random seed
    seed: int = 42


# Parameter definitions with base values and ranges
# Format: {param_name: (display_name, unit, base_value, low_value, high_value)}
PARAMETER_DEFINITIONS = {
    # Economic parameters
    'electricity_price_eur_kWh': (
        'Electricity Price', '€/kWh', 0.07, 0.03, 0.10
    ),
    'capex_stack_eur_kW': (
        'Stack CAPEX', '€/kW', 550, 440, 660
    ),
    'capex_bop_eur_kW': (
        'BoP CAPEX', '€/kW', 400, 320, 480
    ),
    'h2_selling_price_eur_kg': (
        'H2 Selling Price', '€/kg', 6.0, 4.0, 10.0
    ),
    'discount_rate': (
        'Discount Rate', '%', 0.08, 0.05, 0.12
    ),
    'project_lifetime_years': (
        'Project Lifetime', 'years', 15, 15, 25
    ),
    
    # Technical parameters
    'stack_lifetime_hours': (
        'Stack Lifetime', 'h', 90000, 70000, 110000
    ),
    'deg_rate_uV_h': (
        'Degradation Rate', 'μV/h', 0.8, 0.5, 1.5
    ),
    'cycling_penalty_hours': (
        'Cycling Penalty', 'h', 2.5, 1.0, 7.0
    ),
    
    # Efficiency parameters  
    'eta_rectifier': (
        'Rectifier Efficiency', '-', 0.97, 0.94, 0.99
    ),
    'P_bop_fraction': (
        'BoP Parasitic Load', '%', 0.08, 0.05, 0.12
    ),
    
    # Operating parameters
    'T_op_C': (
        'Operating Temperature', '°C', 70, 60, 80
    ),
    'min_load_fraction': (
        'Minimum Load', '%', 0.10, 0.05, 0.20
    ),
}


# =============================================================================
# POWER PROFILE GENERATION
# =============================================================================

def generate_power_profile(
    P_nom_MW: float,
    n_hours: int,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate synthetic variable renewable power profile.
    
    Features:
    - Deterministic (same seed always gives same profile)
    - 10% variability added via deterministic sinusoidal patterns
    - Realistic daily and seasonal patterns
    - ~60% average capacity factor (typical for wind+solar mix)
    
    Parameters
    ----------
    P_nom_MW : float
        Nominal power capacity [MW]
    n_hours : int
        Number of hours to simulate
    seed : int
        Random seed for reproducibility (default 42)
        
    Returns
    -------
    np.ndarray
        Power profile in Watts
    """
    # Always use fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    
    P_nom_W = P_nom_MW * 1e6
    t = np.arange(n_hours)
    
    # Daily solar-like pattern (peak at noon)
    daily_solar = 0.5 * (1 + np.sin(2 * np.pi * t / 24 - np.pi/2))
    
    # Daily wind pattern (slightly anti-correlated with solar)
    daily_wind = 0.6 + 0.3 * np.sin(2 * np.pi * t / 24 + np.pi/3)
    
    # Combined daily pattern (weighted mix)
    daily = 0.5 * daily_solar + 0.5 * daily_wind
    
    # Seasonal variation (lower in winter for solar, higher for wind)
    seasonal = 0.7 + 0.3 * np.sin(2 * np.pi * t / 8760 - np.pi/2)
    
    # 10% DETERMINISTIC variability using multiple sinusoidal patterns
    # (no randomness - same every time)
    variability = (
        0.03 * np.sin(2 * np.pi * t / 6)           # 6-hour pattern
        + 0.03 * np.sin(2 * np.pi * t / 12 + 1.5)  # 12-hour pattern
        + 0.02 * np.sin(2 * np.pi * t / 48 + 0.7)  # 2-day pattern
        + 0.02 * np.sin(2 * np.pi * t / 168 + 2.3) # Weekly pattern
    )
    
    # Base capacity factor ~60%
    power_fraction = 0.25 + 0.45 * daily * seasonal + variability
    
    # Add some deterministic "weather events" (low production periods)
    # Every ~200 hours, have a 10-20 hour low period
    for i in range(0, n_hours, 200):
        event_start = i + (i * 7) % 50  # Deterministic offset
        event_duration = 10 + (i * 3) % 15  # 10-25 hours
        if event_start + event_duration < n_hours:
            power_fraction[event_start:event_start + event_duration] *= 0.3
    
    # Clip to valid range
    power_fraction = np.clip(power_fraction, 0.0, 1.0)
    
    return power_fraction * P_nom_W


# Parameters that require re-running simulation (physical parameters)
SIMULATION_PARAMS = {
    'deg_rate_uV_h',
    'cycling_penalty_hours',
    'stack_lifetime_hours',
    'eta_rectifier',
    'P_bop_fraction',
    'T_op_C',
    'min_load_fraction',
}

# Parameters that only affect economics (no re-simulation needed)
ECONOMIC_ONLY_PARAMS = {
    'electricity_price_eur_kWh',
    'capex_stack_eur_kW',
    'capex_bop_eur_kW',
    'h2_selling_price_eur_kg',
    'discount_rate',
    'project_lifetime_years',
}


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_one_way_sensitivity(
    config: SensitivityConfig,
    power_profile: np.ndarray,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run one-way sensitivity analysis for each parameter.
    
    For PHYSICAL parameters (degradation, efficiency, etc.): Re-runs simulation
    For ECONOMIC parameters (prices, CAPEX, etc.): Only recalculates economics
    
    Returns
    -------
    Dict[str, Dict]
        {param_name: {'low': lcoh_low, 'base': lcoh_base, 'high': lcoh_high, ...}}
    """
    results = {}
    
    # Get base configuration
    base_config = get_alkaline_config(
        P_nom_MW=config.size_MW,
        simulation_years=config.simulation_years,
        electricity_price_eur_kWh=config.base_electricity_price,
        h2_selling_price_eur_kg=config.base_h2_price
    )
    
    # Run base simulation
    base_sim = simulate(power_profile, base_config, verbose=False)
    base_econ = compute_lcoh(base_sim, base_config, verbose=False)
    base_lcoh = base_econ.lcoh_total
    base_npv = base_econ.npv
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ONE-WAY SENSITIVITY ANALYSIS - {config.size_MW:.0f} MW Alkaline")
        print(f"{'='*70}")
        print(f"  Base LCOH: {base_lcoh:.3f} EUR/kg")
        print(f"  Base NPV: €{base_npv/1e6:.1f}M")
        print(f"\n{'Parameter':<25} | {'Low':<10} | {'Base':<10} | {'High':<10} | {'Swing':<10}")
        print(f"{'-'*25} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    for param_name, (display_name, unit, base_val, low_val, high_val) in PARAMETER_DEFINITIONS.items():
        try:
            # Determine if we need to re-run simulation
            needs_simulation = param_name in SIMULATION_PARAMS
            
            # Run with low value
            config_low = get_alkaline_config(
                P_nom_MW=config.size_MW,
                simulation_years=config.simulation_years,
                **{param_name: low_val}
            )
            # Copy stack design from base
            config_low.n_stacks = base_config.n_stacks
            config_low.n_cells = base_config.n_cells
            config_low.cell_area_cm2 = base_config.cell_area_cm2
            
            # Re-run simulation if physical parameter changed
            if needs_simulation:
                sim_low = simulate(power_profile, config_low, verbose=False)
                econ_low = compute_lcoh(sim_low, config_low, verbose=False)
            else:
                econ_low = compute_lcoh(base_sim, config_low, verbose=False)
            lcoh_low = econ_low.lcoh_total
            npv_low = econ_low.npv
            
            # Run with high value
            config_high = get_alkaline_config(
                P_nom_MW=config.size_MW,
                simulation_years=config.simulation_years,
                **{param_name: high_val}
            )
            config_high.n_stacks = base_config.n_stacks
            config_high.n_cells = base_config.n_cells
            config_high.cell_area_cm2 = base_config.cell_area_cm2
            
            # Re-run simulation if physical parameter changed
            if needs_simulation:
                sim_high = simulate(power_profile, config_high, verbose=False)
                econ_high = compute_lcoh(sim_high, config_high, verbose=False)
            else:
                econ_high = compute_lcoh(base_sim, config_high, verbose=False)
            lcoh_high = econ_high.lcoh_total
            npv_high = econ_high.npv
            
            # Calculate swing (impact range)
            swing = abs(lcoh_high - lcoh_low)
            
            results[param_name] = {
                'display_name': display_name,
                'unit': unit,
                'base_value': base_val,
                'low_value': low_val,
                'high_value': high_val,
                'lcoh_low': lcoh_low,
                'lcoh_base': base_lcoh,
                'lcoh_high': lcoh_high,
                'lcoh_swing': swing,
                'npv_low': npv_low,
                'npv_base': base_npv,
                'npv_high': npv_high,
            }
            
            if verbose:
                print(f"  {display_name:<23} | {lcoh_low:<10.3f} | {base_lcoh:<10.3f} | "
                      f"{lcoh_high:<10.3f} | ±{swing/2:<9.3f}")
                
        except Exception as e:
            if verbose:
                print(f"  {display_name:<23} | ERROR: {e}")
            results[param_name] = {'error': str(e)}
    
    return results, base_lcoh, base_npv


def compute_sensitivity_ranking(results: Dict) -> pd.DataFrame:
    """
    Compute sensitivity ranking based on LCOH swing.
    
    Returns DataFrame sorted by impact (swing).
    """
    rows = []
    for param_name, data in results.items():
        if 'error' not in data:
            rows.append({
                'parameter': param_name,
                'display_name': data['display_name'],
                'unit': data['unit'],
                'base_value': data['base_value'],
                'low_value': data['low_value'],
                'high_value': data['high_value'],
                'lcoh_low': data['lcoh_low'],
                'lcoh_base': data['lcoh_base'],
                'lcoh_high': data['lcoh_high'],
                'swing': data['lcoh_swing'],
                'swing_pct': data['lcoh_swing'] / data['lcoh_base'] * 100,
                'npv_swing': abs(data['npv_high'] - data['npv_low']),
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('swing', ascending=False)
    
    return df


def run_spider_analysis(
    config: SensitivityConfig,
    power_profile: np.ndarray,
    n_points: int = 11,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Run spider plot analysis - vary each parameter from low to high.
    
    Returns arrays of LCOH values for each parameter across the range.
    """
    results = {}
    
    base_config = get_alkaline_config(
        P_nom_MW=config.size_MW,
        simulation_years=config.simulation_years,
    )
    
    # Run base simulation once
    base_sim = simulate(power_profile, base_config, verbose=False)
    
    if verbose:
        print(f"\n[Spider Analysis - {n_points} points per parameter]")
    
    # Parameters that must be integers
    integer_params = {'project_lifetime_years', 'stack_lifetime_hours'}
    
    for param_name, (display_name, unit, base_val, low_val, high_val) in PARAMETER_DEFINITIONS.items():
        try:
            # Determine if we need to re-run simulation
            needs_simulation = param_name in SIMULATION_PARAMS
            
            values = np.linspace(low_val, high_val, n_points)
            
            # Cast to int for parameters that require integers
            if param_name in integer_params:
                values = values.astype(int)
            
            lcohs = np.zeros(n_points)
            
            for i, val in enumerate(values):
                # Convert numpy types to Python native for config
                param_val = int(val) if param_name in integer_params else float(val)
                
                config_i = get_alkaline_config(
                    P_nom_MW=config.size_MW,
                    simulation_years=config.simulation_years,
                    **{param_name: param_val}
                )
                config_i.n_stacks = base_config.n_stacks
                config_i.n_cells = base_config.n_cells
                config_i.cell_area_cm2 = base_config.cell_area_cm2
                
                # Re-run simulation if physical parameter
                if needs_simulation:
                    sim_i = simulate(power_profile, config_i, verbose=False)
                    econ_i = compute_lcoh(sim_i, config_i, verbose=False)
                else:
                    econ_i = compute_lcoh(base_sim, config_i, verbose=False)
                lcohs[i] = econ_i.lcoh_total
            
            results[param_name] = {
                'display_name': display_name,
                'unit': unit,
                'values': values,
                'lcohs': lcohs,
                'base_value': base_val,
            }
            
            if verbose:
                print(f"    {display_name}: {lcohs.min():.3f} - {lcohs.max():.3f} EUR/kg")
                
        except Exception as e:
            if verbose:
                print(f"    {display_name}: ERROR - {e}")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_tornado_diagram(
    sensitivity_df: pd.DataFrame,
    base_lcoh: float,
    size_MW: float,
    output_folder: str,
    top_n: int = 10
) -> None:
    """Create tornado diagram showing parameter sensitivity."""
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get top N parameters by swing
    top_params = sensitivity_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_params))
    
    # Calculate bar positions relative to base LCOH
    left_bars = top_params['lcoh_low'].values - base_lcoh
    right_bars = top_params['lcoh_high'].values - base_lcoh
    
    # Create labels
    labels = [f"{row['display_name']}\n({row['unit']})" 
              for _, row in top_params.iterrows()]
    
    # Plot bars
    # Left side (low value effect)
    colors_left = ['#2ca02c' if lb < 0 else '#d62728' for lb in left_bars]
    bars_left = ax.barh(y_pos, left_bars, align='center', color=colors_left, 
                        edgecolor='black', alpha=0.8, height=0.6)
    
    # Right side (high value effect)
    colors_right = ['#d62728' if rb > 0 else '#2ca02c' for rb in right_bars]
    bars_right = ax.barh(y_pos, right_bars, align='center', color=colors_right,
                         edgecolor='black', alpha=0.8, height=0.6)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Change in LCOH (EUR/kg)', fontsize=12)
    ax.set_title(f'Tornado Diagram: LCOH Sensitivity Analysis\n'
                 f'{size_MW:.0f} MW Alkaline Electrolyser (Base LCOH = {base_lcoh:.2f} EUR/kg)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value annotations
    for i, (left, right) in enumerate(zip(left_bars, right_bars)):
        if left < 0:
            ax.text(left - 0.02, i, f'{left:.2f}', va='center', ha='right', fontsize=8)
        else:
            ax.text(left + 0.02, i, f'+{left:.2f}', va='center', ha='left', fontsize=8)
        
        if right > 0:
            ax.text(right + 0.02, i, f'+{right:.2f}', va='center', ha='left', fontsize=8)
        else:
            ax.text(right - 0.02, i, f'{right:.2f}', va='center', ha='right', fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='Increases LCOH'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Decreases LCOH')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / f'tornado_alkaline_{size_MW:.0f}MW.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / f'tornado_alkaline_{size_MW:.0f}MW.png'}")
    plt.close()


def create_spider_plot(
    spider_results: Dict,
    base_lcoh: float,
    size_MW: float,
    output_folder: str,
    top_n: int = 6
) -> None:
    """Create spider plot showing how LCOH varies with each parameter."""
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get top N parameters by range
    ranges = {k: v['lcohs'].max() - v['lcohs'].min() 
              for k, v in spider_results.items() if 'lcohs' in v}
    top_params = sorted(ranges.keys(), key=lambda x: ranges[x], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_params)))
    
    for i, param in enumerate(top_params):
        data = spider_results[param]
        
        # Normalize x-axis to % of base value
        x_pct = (data['values'] - data['base_value']) / data['base_value'] * 100
        
        ax.plot(x_pct, data['lcohs'], 'o-', linewidth=2, markersize=4,
                color=colors[i], label=data['display_name'])
    
    ax.axhline(y=base_lcoh, color='black', linestyle='--', linewidth=1, 
               label=f'Base LCOH ({base_lcoh:.2f})')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Parameter Variation from Base (%)', fontsize=12)
    ax.set_ylabel('LCOH (EUR/kg)', fontsize=12)
    ax.set_title(f'Spider Plot: LCOH Sensitivity\n{size_MW:.0f} MW Alkaline Electrolyser',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'spider_alkaline_{size_MW:.0f}MW.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / f'spider_alkaline_{size_MW:.0f}MW.png'}")
    plt.close()


def create_waterfall_chart(
    sensitivity_df: pd.DataFrame,
    base_lcoh: float,
    size_MW: float,
    output_folder: str,
    scenario: str = 'worst'  # 'worst' or 'best'
) -> None:
    """Create waterfall chart showing cumulative LCOH impact."""
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select which direction for each parameter
    if scenario == 'worst':
        # Take the value that increases LCOH the most
        impacts = []
        for _, row in sensitivity_df.iterrows():
            impact = max(row['lcoh_low'] - row['lcoh_base'], 
                        row['lcoh_high'] - row['lcoh_base'])
            impacts.append({
                'name': row['display_name'],
                'impact': impact
            })
        title_suffix = "Worst Case"
    else:
        # Take the value that decreases LCOH the most
        impacts = []
        for _, row in sensitivity_df.iterrows():
            impact = min(row['lcoh_low'] - row['lcoh_base'],
                        row['lcoh_high'] - row['lcoh_base'])
            impacts.append({
                'name': row['display_name'],
                'impact': impact
            })
        title_suffix = "Best Case"
    
    # Sort by absolute impact
    impacts = sorted(impacts, key=lambda x: abs(x['impact']), reverse=True)[:8]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Build waterfall
    cumulative = base_lcoh
    bars_data = [('Base LCOH', base_lcoh, 0, base_lcoh)]
    
    for item in impacts:
        start = cumulative
        cumulative += item['impact']
        bars_data.append((item['name'], item['impact'], start, cumulative))
    
    bars_data.append(('Final LCOH', cumulative, 0, cumulative))
    
    # Plot
    x_pos = np.arange(len(bars_data))
    
    for i, (name, val, start, end) in enumerate(bars_data):
        if i == 0:  # Base
            ax.bar(i, val, bottom=0, color='#3498db', edgecolor='black')
        elif i == len(bars_data) - 1:  # Final
            ax.bar(i, end, bottom=0, color='#2ecc71' if end < base_lcoh else '#e74c3c', 
                   edgecolor='black')
        else:  # Increments
            color = '#e74c3c' if val > 0 else '#2ecc71'
            ax.bar(i, abs(val), bottom=min(start, end), color=color, edgecolor='black')
    
    # Add connecting lines
    for i in range(len(bars_data) - 1):
        _, _, _, end1 = bars_data[i]
        ax.hlines(end1, i + 0.4, i + 0.6, color='black', linewidth=1)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b[0] for b in bars_data], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('LCOH (EUR/kg)', fontsize=12)
    ax.set_title(f'Waterfall Chart: LCOH {title_suffix}\n{size_MW:.0f} MW Alkaline Electrolyser',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (name, val, start, end) in enumerate(bars_data):
        if i == 0 or i == len(bars_data) - 1:
            ax.text(i, end + 0.05, f'{end:.2f}', ha='center', fontsize=9, fontweight='bold')
        else:
            mid = (start + end) / 2
            sign = '+' if val > 0 else ''
            ax.text(i, mid, f'{sign}{val:.2f}', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / f'waterfall_alkaline_{size_MW:.0f}MW_{scenario}.png', 
                dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / f'waterfall_alkaline_{size_MW:.0f}MW_{scenario}.png'}")
    plt.close()


def create_sensitivity_summary(
    sensitivity_df: pd.DataFrame,
    base_lcoh: float,
    base_npv: float,
    size_MW: float,
    output_folder: str
) -> None:
    """Create formatted summary table."""
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary
    summary = sensitivity_df.copy()
    summary['swing_pct'] = summary['swing'] / base_lcoh * 100
    summary['rank'] = range(1, len(summary) + 1)
    
    # Select columns for output
    output_cols = ['rank', 'display_name', 'unit', 'low_value', 'base_value', 
                   'high_value', 'lcoh_low', 'lcoh_base', 'lcoh_high', 
                   'swing', 'swing_pct']
    
    summary = summary[output_cols]
    summary.columns = ['Rank', 'Parameter', 'Unit', 'Low Value', 'Base Value',
                       'High Value', 'LCOH Low', 'LCOH Base', 'LCOH High',
                       'Swing (€/kg)', 'Swing (%)']
    
    summary.to_csv(output_path / f'sensitivity_summary_{size_MW:.0f}MW.csv', index=False)
    print(f"  Saved: {output_path / f'sensitivity_summary_{size_MW:.0f}MW.csv'}")
    
    # Print formatted table
    print(f"\n{'='*80}")
    print(f"SENSITIVITY RANKING - {size_MW:.0f} MW Alkaline")
    print(f"Base LCOH: {base_lcoh:.3f} EUR/kg | Base NPV: €{base_npv/1e6:.1f}M")
    print(f"{'='*80}")
    print(f"{'Rank':<5} | {'Parameter':<25} | {'Swing':<12} | {'Swing %':<10}")
    print(f"{'-'*5} | {'-'*25} | {'-'*12} | {'-'*10}")
    
    for _, row in summary.head(10).iterrows():
        print(f"{row['Rank']:<5} | {row['Parameter']:<25} | "
              f"±{row['Swing (€/kg)']/2:<10.3f} | ±{row['Swing (%)']/2:<8.1f}%")
    
    print(f"\n📊 KEY INSIGHT: Top 3 drivers account for "
          f"{summary.head(3)['Swing (€/kg)'].sum() / summary['Swing (€/kg)'].sum() * 100:.0f}% "
          f"of total LCOH variation")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for sensitivity analysis."""
    
    parser = argparse.ArgumentParser(
        description='Alkaline Electrolyser Sensitivity Analysis'
    )
    parser.add_argument('--size', type=float, default=20.0,
                        help='Electrolyser size [MW]')
    parser.add_argument('--years', type=int, default=1,
                        help='Simulation years (1 for fast, 5 for accurate)')
    parser.add_argument('--output', type=str, default='results/alkaline_sensitivity',
                        help='Output folder')
    parser.add_argument('--spider-points', type=int, default=11,
                        help='Number of points for spider analysis')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ALKALINE ELECTROLYSER SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Create configuration
    config = SensitivityConfig(
        size_MW=args.size,
        simulation_years=args.years,
        output_folder=args.output,
        seed=args.seed
    )
    
    # Generate power profile
    n_hours = config.simulation_years * 8760
    power_profile = generate_power_profile(
        config.size_MW, n_hours, config.seed
    )
    
    # Run one-way sensitivity
    print("\n[1/4] Running one-way sensitivity analysis...")
    results, base_lcoh, base_npv = run_one_way_sensitivity(
        config, power_profile, verbose=True
    )
    
    # Compute ranking
    sensitivity_df = compute_sensitivity_ranking(results)
    
    # Run spider analysis
    print("\n[2/4] Running spider analysis...")
    spider_results = run_spider_analysis(
        config, power_profile, n_points=args.spider_points, verbose=True
    )
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    
    create_tornado_diagram(sensitivity_df, base_lcoh, config.size_MW, args.output)
    create_spider_plot(spider_results, base_lcoh, config.size_MW, args.output)
    create_waterfall_chart(sensitivity_df, base_lcoh, config.size_MW, args.output, 'worst')
    create_waterfall_chart(sensitivity_df, base_lcoh, config.size_MW, args.output, 'best')
    
    # Create summary
    print("\n[4/4] Creating summary...")
    create_sensitivity_summary(sensitivity_df, base_lcoh, base_npv, 
                               config.size_MW, args.output)
    
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"  Output folder: {args.output}")
    
    return sensitivity_df, spider_results


if __name__ == "__main__":
    sensitivity_df, spider_results = main()
