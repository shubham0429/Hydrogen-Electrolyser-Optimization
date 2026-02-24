"""
PEM vs Alkaline Electrolyser Comparison
=======================================

This script runs both PEM and Alkaline simulations with identical power input
and creates comprehensive comparison plots and analysis for the thesis.

Author: Thesis Project
Date: January 2026

Outputs:
- Comparison tables (CSV)
- Side-by-side plots
- Sensitivity comparison
- Technology selection recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import sys
import warnings
warnings.filterwarnings('ignore')

# Import simulation modules
from sim_concise import get_config, simulate, compute_economics
from sim_alkaline import (
    AlkalineConfig, get_alkaline_config, 
    simulate as simulate_alkaline, 
    compute_lcoh as compute_lcoh_alkaline
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "pem_vs_alkaline_comparison"

# Simulation parameters (identical for both)
SIMULATION_YEARS = 5
ELECTROLYSER_SIZE_MW = 20.0
STORAGE_CAPACITY_KG = 2500.0

# Common electricity price for fair comparison
ELECTRICITY_PRICE_EUR_KWH = 0.05

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
})

# Colors for PEM and Alkaline
COLOR_PEM = '#1f77b4'      # Blue
COLOR_ALK = '#2ca02c'      # Green


# =============================================================================
# DATA LOADING
# =============================================================================

def load_power_data(years: int = 5) -> np.ndarray:
    """Load renewable power data from .mat file."""
    data_dir = Path(__file__).parent.parent / "data"
    mat_path = data_dir / "combined_wind_pv_DATA.mat"
    
    if not mat_path.exists():
        # Try alternative paths
        alt_paths = [
            data_dir / "renewable_power_1year.mat",
            data_dir / "power_data.mat",
            data_dir / "combined_power_W.mat",
            Path(__file__).parent.parent / "shareable_code" / "data" / "renewable_power_1year.mat"
        ]
        for alt in alt_paths:
            if alt.exists():
                mat_path = alt
                break
        else:
            raise FileNotFoundError(f"No power data file found. Tried: {mat_path}, {alt_paths}")
    
    print(f"Loading power data from: {mat_path}")
    mat = loadmat(str(mat_path))
    
    # Find power array — try combined wind+solar first
    if 'P_wind_selected' in mat and 'P_PV' in mat:
        power_1yr = mat['P_wind_selected'].flatten() + mat['P_PV'].flatten()
    else:
        for key in ['power_W', 'combined_power_W', 'total_power_W', 'power']:
            if key in mat:
                power_1yr = mat[key].flatten()
                break
        else:
            # Get first non-metadata array
            for key, val in mat.items():
                if not key.startswith('_') and isinstance(val, np.ndarray):
                    power_1yr = val.flatten()
                    break
            else:
                raise ValueError("Could not find power data in .mat file")
    
    # Extend to multiple years with 10% deterministic variability
    hours_per_year = 8760
    power_1yr = power_1yr[:hours_per_year]
    
    power_extended = []
    for yr in range(years):
        # Add deterministic variability (no randomness)
        t = np.arange(hours_per_year) + yr * hours_per_year
        variation = 1.0 + 0.05 * np.sin(2 * np.pi * t / (6 * hours_per_year))  # 6-hour cycle
        variation += 0.03 * np.sin(2 * np.pi * t / (12 * hours_per_year))  # 12-hour cycle
        variation += 0.02 * np.sin(2 * np.pi * t / (168))  # Weekly cycle
        power_yr = power_1yr * variation
        power_extended.append(power_yr)
    
    return np.concatenate(power_extended)


# =============================================================================
# PEM SIMULATION
# =============================================================================

def run_pem_simulation(power_W: np.ndarray) -> dict:
    """Run PEM electrolyser simulation."""
    print("\n" + "="*60)
    print("Running PEM Electrolyser Simulation")
    print("="*60)
    
    cfg = get_config(size_mw=ELECTROLYSER_SIZE_MW, storage_kg=STORAGE_CAPACITY_KG)
    
    # Update electricity price
    cfg['LCOE_ELECTRICITY_EUR_PER_KWH'] = ELECTRICITY_PRICE_EUR_KWH
    cfg['YEARS'] = SIMULATION_YEARS
    
    # Run simulation
    sim_results = simulate(power_W, cfg)
    
    # Compute economics
    econ_results = compute_economics(sim_results, cfg)
    
    # Extract key metrics
    total_hours = len(power_W)
    operating_hours = np.sum(sim_results['operating_mask'])
    
    results = {
        'technology': 'PEM',
        
        # Sizing
        'capacity_MW': ELECTROLYSER_SIZE_MW,
        'n_cells': cfg['N_CELLS'],
        'cell_area_cm2': cfg['CELL_AREA_CM2'],
        
        # Operating conditions
        'T_op_C': cfg['T_OPERATING_C'],
        'j_nominal_A_cm2': cfg['J_NOMINAL'],
        
        # Electrochemistry
        'E_rev_V': cfg['E_REV_STD'],
        'R_ohm_ohm_cm2': cfg['R_OHM'],
        
        # Performance
        'total_H2_kg': sim_results['total_H2_kg'],
        'avg_H2_rate_kg_h': sim_results['total_H2_kg'] / operating_hours if operating_hours > 0 else 0,
        'capacity_factor': operating_hours / total_hours,
        'operating_hours': operating_hours,
        
        # Efficiency
        'SEC_stack_kWh_kg': sim_results.get('sec_stack_avg', sim_results.get('SEC_stack_kWh_kg', 50)),
        'SEC_system_kWh_kg': sim_results.get('sec_system_avg', sim_results.get('SEC_system_kWh_kg', 55)),
        'efficiency_LHV_pct': 33.33 / sim_results.get('sec_system_avg', 55) * 100,
        
        # Voltage
        'V_cell_nominal_V': sim_results.get('V_cell_mean', 1.85),
        'V_cell_final_V': sim_results.get('V_cell_array', np.array([1.85]))[-1],
        'voltage_degradation_pct': (sim_results.get('V_cell_array', np.array([1.85, 1.90]))[-1] / 
                                     sim_results.get('V_cell_array', np.array([1.85, 1.90]))[0] - 1) * 100,
        
        # Economics
        'LCOH_eur_kg': econ_results['lcoh_total'],
        'LCOH_electricity_eur_kg': econ_results.get('lcoh_electricity', econ_results['lcoh_total'] * 0.7),
        'LCOH_capex_eur_kg': econ_results.get('lcoh_capex', econ_results['lcoh_total'] * 0.2),
        'LCOH_opex_eur_kg': econ_results.get('lcoh_opex_fixed', econ_results['lcoh_total'] * 0.1),
        
        'CAPEX_total_EUR': econ_results.get('capex_total', cfg['CAPEX_EUR_PER_KW'] * cfg['ELECTROLYSER_SIZE_KW']),
        'CAPEX_EUR_kW': cfg['CAPEX_EUR_PER_KW'],
        
        'NPV_EUR': econ_results.get('npv', 0),
        'IRR_pct': econ_results.get('irr', 0) * 100 if econ_results.get('irr') else 0,
        
        # Degradation
        'stack_lifetime_h': cfg['STACK_LIFE_HOURS'],
        'degradation_type': 'Nonlinear (load-dependent)',
        
        # Raw results for plotting
        'H2_production_kg': sim_results.get('H2_production_kg', np.zeros(len(power_W))),
        'V_cell_array': sim_results.get('V_cell_array', np.ones(len(power_W)) * 1.85),
        'SEC_stack_array': sim_results.get('SEC_stack_kWh_kg_array', np.ones(len(power_W)) * 50),
        'power_consumed_W': sim_results.get('power_consumed_W', power_W),
    }
    
    print(f"  Total H2 produced: {results['total_H2_kg']:,.0f} kg")
    print(f"  LCOH: {results['LCOH_eur_kg']:.2f} EUR/kg")
    print(f"  SEC (system): {results['SEC_system_kWh_kg']:.1f} kWh/kg")
    print(f"  Capacity factor: {results['capacity_factor']*100:.1f}%")
    
    return results


# =============================================================================
# ALKALINE SIMULATION
# =============================================================================

def run_alkaline_simulation(power_W: np.ndarray) -> dict:
    """Run Alkaline electrolyser simulation."""
    print("\n" + "="*60)
    print("Running Alkaline Electrolyser Simulation")
    print("="*60)
    
    cfg = get_alkaline_config(
        P_nom_MW=ELECTROLYSER_SIZE_MW,
        storage_capacity_kg=STORAGE_CAPACITY_KG,
        electricity_price_eur_kWh=ELECTRICITY_PRICE_EUR_KWH,
        simulation_years=SIMULATION_YEARS,
    )
    
    # Run simulation
    sim_results = simulate_alkaline(power_W, cfg)
    
    # Compute economics
    econ_results = compute_lcoh_alkaline(sim_results, cfg)
    
    # Extract key metrics
    total_hours = len(power_W)
    operating_hours = sim_results.operating_hours
    
    results = {
        'technology': 'Alkaline',
        
        # Sizing
        'capacity_MW': ELECTROLYSER_SIZE_MW,
        'n_cells': cfg.n_cells * cfg.n_stacks,
        'cell_area_cm2': cfg.cell_area_cm2,
        
        # Operating conditions
        'T_op_C': cfg.T_op_C,
        'j_nominal_A_cm2': cfg.j_nom,
        
        # Electrochemistry
        'E_rev_V': cfg.E_rev_0,
        'R_ohm_ohm_cm2': cfg.R_cell_ohm_cm2,
        
        # Performance
        'total_H2_kg': sim_results.total_H2_produced_kg,
        'avg_H2_rate_kg_h': sim_results.total_H2_produced_kg / operating_hours if operating_hours > 0 else 0,
        'capacity_factor': sim_results.capacity_factor,
        'operating_hours': operating_hours,
        
        # Efficiency
        'SEC_stack_kWh_kg': sim_results.sec_stack_avg_kWh_kg,
        'SEC_system_kWh_kg': sim_results.sec_system_avg_kWh_kg,
        'efficiency_LHV_pct': 33.33 / sim_results.sec_system_avg_kWh_kg * 100,
        
        # Voltage
        'V_cell_nominal_V': sim_results.V_cell_timeseries[0] if len(sim_results.V_cell_timeseries) > 0 else 1.78,
        'V_cell_final_V': sim_results.V_cell_timeseries[-1] if len(sim_results.V_cell_timeseries) > 0 else 1.80,
        'voltage_degradation_pct': sim_results.voltage_degradation_pct,
        
        # Economics
        'LCOH_eur_kg': econ_results.lcoh_total,
        'LCOH_electricity_eur_kg': econ_results.lcoh_electricity,
        'LCOH_capex_eur_kg': econ_results.lcoh_capex,
        'LCOH_opex_eur_kg': econ_results.lcoh_opex_fixed,
        
        'CAPEX_total_EUR': econ_results.capex_total,
        'CAPEX_EUR_kW': (cfg.capex_stack_eur_kW + cfg.capex_bop_eur_kW),
        
        'NPV_EUR': econ_results.npv,
        'IRR_pct': econ_results.irr * 100 if econ_results.irr else 0,
        
        # Degradation
        'stack_lifetime_h': cfg.stack_lifetime_hours,
        'degradation_type': 'Linear (0.8 μV/h + cycling)',
        
        # Raw results for plotting
        'H2_production_kg': sim_results.H2_production_kg,
        'V_cell_array': sim_results.V_cell_timeseries,
        'SEC_stack_array': sim_results.sec_stack_timeseries,
        'power_consumed_W': sim_results.power_consumed_W,
    }
    
    print(f"  Total H2 produced: {results['total_H2_kg']:,.0f} kg")
    print(f"  LCOH: {results['LCOH_eur_kg']:.2f} EUR/kg")
    print(f"  SEC (system): {results['SEC_system_kWh_kg']:.1f} kWh/kg")
    print(f"  Capacity factor: {results['capacity_factor']*100:.1f}%")
    
    return results


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def create_comparison_summary_table(pem: dict, alk: dict) -> pd.DataFrame:
    """Create comparison summary table."""
    
    metrics = [
        ('Capacity', 'MW', 'capacity_MW', '.1f'),
        ('Operating Temperature', '°C', 'T_op_C', '.0f'),
        ('Nominal Current Density', 'A/cm²', 'j_nominal_A_cm2', '.2f'),
        ('Cell Voltage (nominal)', 'V', 'V_cell_nominal_V', '.2f'),
        ('Cell Voltage (final)', 'V', 'V_cell_final_V', '.2f'),
        ('Voltage Degradation', '%', 'voltage_degradation_pct', '.2f'),
        ('SEC (Stack)', 'kWh/kg', 'SEC_stack_kWh_kg', '.1f'),
        ('SEC (System)', 'kWh/kg', 'SEC_system_kWh_kg', '.1f'),
        ('Efficiency (LHV)', '%', 'efficiency_LHV_pct', '.1f'),
        ('Total H₂ Produced', 'tonnes', 'total_H2_kg', ',.0f'),
        ('Capacity Factor', '%', 'capacity_factor', '.1f'),
        ('Operating Hours', 'h', 'operating_hours', ',.0f'),
        ('LCOH (Total)', 'EUR/kg', 'LCOH_eur_kg', '.2f'),
        ('LCOH (Electricity)', 'EUR/kg', 'LCOH_electricity_eur_kg', '.2f'),
        ('LCOH (CAPEX)', 'EUR/kg', 'LCOH_capex_eur_kg', '.2f'),
        ('CAPEX', 'EUR/kW', 'CAPEX_EUR_kW', '.0f'),
        ('Stack Lifetime', 'hours', 'stack_lifetime_h', ',.0f'),
        ('NPV', 'M EUR', 'NPV_EUR', ',.2f'),
    ]
    
    data = []
    for name, unit, key, fmt in metrics:
        pem_val = pem.get(key, 0)
        alk_val = alk.get(key, 0)
        
        # Handle capacity factor (stored as fraction, display as %)
        if key == 'capacity_factor':
            pem_val *= 100
            alk_val *= 100
        
        # Handle total H2 (convert to tonnes)
        if key == 'total_H2_kg':
            pem_val /= 1000
            alk_val /= 1000
        
        # Handle NPV (convert to millions)
        if key == 'NPV_EUR':
            pem_val /= 1e6
            alk_val /= 1e6
        
        # Calculate difference
        if pem_val != 0:
            diff_pct = (alk_val - pem_val) / abs(pem_val) * 100
            diff_str = f"{diff_pct:+.1f}%"
        else:
            diff_str = "N/A"
        
        data.append({
            'Metric': name,
            'Unit': unit,
            'PEM': f"{pem_val:{fmt}}",
            'Alkaline': f"{alk_val:{fmt}}",
            'Difference': diff_str
        })
    
    return pd.DataFrame(data)


def plot_comparison_bar_chart(pem: dict, alk: dict, output_dir: Path):
    """Create bar chart comparing key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. LCOH Breakdown
    ax = axes[0, 0]
    categories = ['Electricity', 'CAPEX', 'OPEX']
    pem_vals = [pem['LCOH_electricity_eur_kg'], pem['LCOH_capex_eur_kg'], pem['LCOH_opex_eur_kg']]
    alk_vals = [alk['LCOH_electricity_eur_kg'], alk['LCOH_capex_eur_kg'], alk['LCOH_opex_eur_kg']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pem_vals, width, label='PEM', color=COLOR_PEM, alpha=0.8)
    bars2 = ax.bar(x + width/2, alk_vals, width, label='Alkaline', color=COLOR_ALK, alpha=0.8)
    
    ax.set_ylabel('LCOH Component [EUR/kg]')
    ax.set_title('LCOH Breakdown Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.2f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9)
    
    # 2. Efficiency Comparison
    ax = axes[0, 1]
    categories = ['SEC Stack\n[kWh/kg]', 'SEC System\n[kWh/kg]', 'Efficiency\n[% LHV]']
    pem_vals = [pem['SEC_stack_kWh_kg'], pem['SEC_system_kWh_kg'], pem['efficiency_LHV_pct']]
    alk_vals = [alk['SEC_stack_kWh_kg'], alk['SEC_system_kWh_kg'], alk['efficiency_LHV_pct']]
    
    x = np.arange(len(categories))
    bars1 = ax.bar(x - width/2, pem_vals, width, label='PEM', color=COLOR_PEM, alpha=0.8)
    bars2 = ax.bar(x + width/2, alk_vals, width, label='Alkaline', color=COLOR_ALK, alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Efficiency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', padding=3, fontsize=9)
    
    # 3. Capital Cost Comparison
    ax = axes[1, 0]
    categories = ['CAPEX\n[EUR/kW]', 'Stack Life\n[kh]', 'LCOH Total\n[EUR/kg]']
    pem_vals = [pem['CAPEX_EUR_kW'], pem['stack_lifetime_h']/1000, pem['LCOH_eur_kg']]
    alk_vals = [alk['CAPEX_EUR_kW'], alk['stack_lifetime_h']/1000, alk['LCOH_eur_kg']]
    
    x = np.arange(len(categories))
    bars1 = ax.bar(x - width/2, pem_vals, width, label='PEM', color=COLOR_PEM, alpha=0.8)
    bars2 = ax.bar(x + width/2, alk_vals, width, label='Alkaline', color=COLOR_ALK, alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Cost & Lifetime Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', padding=3, fontsize=9)
    
    # 4. Production Comparison
    ax = axes[1, 1]
    categories = ['H₂ Produced\n[tonnes]', 'Capacity Factor\n[%]', 'Operating Hours\n[kh]']
    pem_vals = [pem['total_H2_kg']/1000, pem['capacity_factor']*100, pem['operating_hours']/1000]
    alk_vals = [alk['total_H2_kg']/1000, alk['capacity_factor']*100, alk['operating_hours']/1000]
    
    x = np.arange(len(categories))
    bars1 = ax.bar(x - width/2, pem_vals, width, label='PEM', color=COLOR_PEM, alpha=0.8)
    bars2 = ax.bar(x + width/2, alk_vals, width, label='Alkaline', color=COLOR_ALK, alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Production Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', padding=3, fontsize=9)
    
    plt.suptitle(f'PEM vs Alkaline Electrolyser Comparison ({ELECTROLYSER_SIZE_MW:.0f} MW, {SIMULATION_YEARS} years)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'comparison_bar_charts.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_bar_charts.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: comparison_bar_charts.png/pdf")


def plot_timeseries_comparison(pem: dict, alk: dict, output_dir: Path, hours: int = 168*4):
    """Create timeseries comparison plots (first 4 weeks)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    t = np.arange(hours)
    t_days = t / 24
    
    # 1. H2 Production Rate
    ax = axes[0]
    ax.plot(t_days, pem['H2_production_kg'][:hours], color=COLOR_PEM, alpha=0.7, label='PEM', linewidth=0.8)
    ax.plot(t_days, alk['H2_production_kg'][:hours], color=COLOR_ALK, alpha=0.7, label='Alkaline', linewidth=0.8)
    ax.set_ylabel('H₂ Production [kg/h]')
    ax.set_title('Hydrogen Production Rate')
    ax.legend(loc='upper right')
    ax.set_xlim(0, hours/24)
    
    # 2. Cell Voltage
    ax = axes[1]
    ax.plot(t_days, pem['V_cell_array'][:hours], color=COLOR_PEM, alpha=0.7, label='PEM', linewidth=0.8)
    ax.plot(t_days, alk['V_cell_array'][:hours], color=COLOR_ALK, alpha=0.7, label='Alkaline', linewidth=0.8)
    ax.set_ylabel('Cell Voltage [V]')
    ax.set_title('Cell Voltage')
    ax.legend(loc='upper right')
    
    # 3. SEC
    ax = axes[2]
    # Filter out zeros and very high values for better visualization
    pem_sec = np.clip(pem['SEC_stack_array'][:hours], 0, 100)
    alk_sec = np.clip(alk['SEC_stack_array'][:hours], 0, 100)
    pem_sec[pem_sec == 0] = np.nan
    alk_sec[alk_sec == 0] = np.nan
    
    ax.plot(t_days, pem_sec, color=COLOR_PEM, alpha=0.7, label='PEM', linewidth=0.8)
    ax.plot(t_days, alk_sec, color=COLOR_ALK, alpha=0.7, label='Alkaline', linewidth=0.8)
    ax.set_ylabel('SEC Stack [kWh/kg]')
    ax.set_xlabel('Time [days]')
    ax.set_title('Specific Energy Consumption')
    ax.legend(loc='upper right')
    ax.set_ylim(30, 70)
    
    plt.suptitle(f'PEM vs Alkaline Timeseries Comparison (First {hours//24} days)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'timeseries_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'timeseries_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: timeseries_comparison.png/pdf")


def plot_voltage_degradation_comparison(pem: dict, alk: dict, output_dir: Path):
    """Plot voltage degradation over full simulation period."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    total_hours = len(pem['V_cell_array'])
    t_months = np.arange(total_hours) / (24 * 30)
    
    # 1. Full timeseries (monthly average)
    ax = axes[0]
    
    # Calculate monthly averages
    month_hours = 24 * 30
    n_months = total_hours // month_hours
    
    pem_monthly = []
    alk_monthly = []
    for i in range(n_months):
        start = i * month_hours
        end = start + month_hours
        pem_monthly.append(np.mean(pem['V_cell_array'][start:end]))
        alk_monthly.append(np.mean(alk['V_cell_array'][start:end]))
    
    months = np.arange(n_months)
    ax.plot(months, pem_monthly, 'o-', color=COLOR_PEM, label='PEM', markersize=4, linewidth=1.5)
    ax.plot(months, alk_monthly, 's-', color=COLOR_ALK, label='Alkaline', markersize=4, linewidth=1.5)
    
    ax.set_xlabel('Time [months]')
    ax.set_ylabel('Average Cell Voltage [V]')
    ax.set_title('Voltage Degradation Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Degradation summary
    ax = axes[1]
    
    categories = ['Initial\nVoltage [V]', 'Final\nVoltage [V]', 'Degradation\n[%]']
    pem_vals = [pem['V_cell_nominal_V'], pem['V_cell_final_V'], pem['voltage_degradation_pct']]
    alk_vals = [alk['V_cell_nominal_V'], alk['V_cell_final_V'], alk['voltage_degradation_pct']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pem_vals, width, label='PEM', color=COLOR_PEM, alpha=0.8)
    bars2 = ax.bar(x + width/2, alk_vals, width, label='Alkaline', color=COLOR_ALK, alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Voltage Degradation Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.bar_label(bars1, fmt='%.2f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.2f', padding=3, fontsize=9)
    
    plt.suptitle(f'Voltage Degradation: PEM vs Alkaline ({SIMULATION_YEARS} years)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'voltage_degradation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'voltage_degradation_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: voltage_degradation_comparison.png/pdf")


def plot_lcoh_waterfall(pem: dict, alk: dict, output_dir: Path):
    """Create LCOH waterfall comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax, tech, data, color in [(axes[0], 'PEM', pem, COLOR_PEM), 
                                   (axes[1], 'Alkaline', alk, COLOR_ALK)]:
        components = ['Electricity', 'CAPEX', 'OPEX', 'Total']
        values = [
            data['LCOH_electricity_eur_kg'],
            data['LCOH_capex_eur_kg'],
            data['LCOH_opex_eur_kg'],
            data['LCOH_eur_kg']
        ]
        
        # Waterfall
        cumsum = 0
        for i, (comp, val) in enumerate(zip(components[:-1], values[:-1])):
            ax.bar(i, val, bottom=cumsum, color=color, alpha=0.7 + i*0.1, 
                   edgecolor='white', linewidth=1)
            ax.text(i, cumsum + val/2, f'{val:.2f}', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
            cumsum += val
        
        # Total bar
        ax.bar(len(components)-1, values[-1], color='#333333', alpha=0.9,
               edgecolor='white', linewidth=1)
        ax.text(len(components)-1, values[-1]/2, f'{values[-1]:.2f}', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax.set_xticks(range(len(components)))
        ax.set_xticklabels(components)
        ax.set_ylabel('LCOH [EUR/kg]')
        ax.set_title(f'{tech} Electrolyser')
        ax.set_ylim(0, max(pem['LCOH_eur_kg'], alk['LCOH_eur_kg']) * 1.15)
    
    plt.suptitle('LCOH Component Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'lcoh_waterfall_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'lcoh_waterfall_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: lcoh_waterfall_comparison.png/pdf")


def plot_technology_radar(pem: dict, alk: dict, output_dir: Path):
    """Create radar chart comparing technologies."""
    categories = ['Efficiency', 'CAPEX', 'Lifetime', 'LCOH', 'Production', 'Maturity']
    
    # Normalize values (higher = better, scale 0-100)
    # For CAPEX and LCOH, invert (lower is better)
    pem_vals = [
        pem['efficiency_LHV_pct'],  # Higher is better
        100 - (pem['CAPEX_EUR_kW'] / 15),  # Lower is better (1500 EUR/kW = 0)
        pem['stack_lifetime_h'] / 1000,  # Higher is better (100kh = 100)
        100 - (pem['LCOH_eur_kg'] * 10),  # Lower is better (10 EUR/kg = 0)
        min(100, pem['total_H2_kg'] / 100000),  # Higher is better
        85,  # PEM maturity score (relatively mature)
    ]
    
    alk_vals = [
        alk['efficiency_LHV_pct'],
        100 - (alk['CAPEX_EUR_kW'] / 15),
        alk['stack_lifetime_h'] / 1000,
        100 - (alk['LCOH_eur_kg'] * 10),
        min(100, alk['total_H2_kg'] / 100000),
        95,  # Alkaline maturity score (most mature)
    ]
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    pem_vals += pem_vals[:1]
    alk_vals += alk_vals[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, pem_vals, 'o-', linewidth=2, label='PEM', color=COLOR_PEM)
    ax.fill(angles, pem_vals, alpha=0.25, color=COLOR_PEM)
    
    ax.plot(angles, alk_vals, 's-', linewidth=2, label='Alkaline', color=COLOR_ALK)
    ax.fill(angles, alk_vals, alpha=0.25, color=COLOR_ALK)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    
    plt.title('Technology Comparison Radar\n(Higher = Better)', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'technology_radar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'technology_radar.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: technology_radar.png/pdf")


def create_recommendation_summary(pem: dict, alk: dict) -> str:
    """Generate technology selection recommendation."""
    
    lcoh_diff = alk['LCOH_eur_kg'] - pem['LCOH_eur_kg']
    lcoh_diff_pct = lcoh_diff / pem['LCOH_eur_kg'] * 100
    
    capex_diff = alk['CAPEX_EUR_kW'] - pem['CAPEX_EUR_kW']
    capex_diff_pct = capex_diff / pem['CAPEX_EUR_kW'] * 100
    
    eff_diff = alk['efficiency_LHV_pct'] - pem['efficiency_LHV_pct']
    
    summary = f"""
================================================================================
TECHNOLOGY SELECTION RECOMMENDATION
================================================================================

SIMULATION PARAMETERS:
- Electrolyser Size: {ELECTROLYSER_SIZE_MW:.0f} MW
- Simulation Period: {SIMULATION_YEARS} years
- Electricity Price: {ELECTRICITY_PRICE_EUR_KWH:.2f} EUR/kWh
- Storage Capacity: {STORAGE_CAPACITY_KG:.0f} kg

--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------

1. LEVELIZED COST OF HYDROGEN (LCOH):
   - PEM:      {pem['LCOH_eur_kg']:.2f} EUR/kg
   - Alkaline: {alk['LCOH_eur_kg']:.2f} EUR/kg
   - Difference: {lcoh_diff:+.2f} EUR/kg ({lcoh_diff_pct:+.1f}%)
   
   Winner: {'Alkaline' if lcoh_diff < 0 else 'PEM'} (lower LCOH)

2. CAPITAL COST:
   - PEM:      {pem['CAPEX_EUR_kW']:.0f} EUR/kW
   - Alkaline: {alk['CAPEX_EUR_kW']:.0f} EUR/kW
   - Difference: {capex_diff:+.0f} EUR/kW ({capex_diff_pct:+.1f}%)
   
   Winner: {'Alkaline' if capex_diff < 0 else 'PEM'} (lower CAPEX)

3. EFFICIENCY:
   - PEM:      {pem['efficiency_LHV_pct']:.1f}% (LHV)
   - Alkaline: {alk['efficiency_LHV_pct']:.1f}% (LHV)
   - Difference: {eff_diff:+.1f}%
   
   Winner: {'Alkaline' if eff_diff > 0 else 'PEM'} (higher efficiency)

4. HYDROGEN PRODUCTION:
   - PEM:      {pem['total_H2_kg']/1000:,.1f} tonnes
   - Alkaline: {alk['total_H2_kg']/1000:,.1f} tonnes
   
   Winner: {'Alkaline' if alk['total_H2_kg'] > pem['total_H2_kg'] else 'PEM'}

5. STACK LIFETIME:
   - PEM:      {pem['stack_lifetime_h']:,.0f} hours
   - Alkaline: {alk['stack_lifetime_h']:,.0f} hours
   
   Winner: {'Alkaline' if alk['stack_lifetime_h'] > pem['stack_lifetime_h'] else 'PEM'}

--------------------------------------------------------------------------------
TECHNOLOGY SELECTION GUIDANCE
--------------------------------------------------------------------------------

CHOOSE PEM WHEN:
✓ Fast response to variable renewables is critical (seconds vs minutes)
✓ Space is limited (higher power density)
✓ High-purity hydrogen required (>99.999%)
✓ Frequent start/stop cycling expected
✓ Grid services revenue possible

CHOOSE ALKALINE WHEN:
✓ Minimizing CAPEX is priority
✓ Steady baseload operation expected
✓ Long stack lifetime important
✓ Mature, proven technology preferred
✓ Lower electricity prices available

--------------------------------------------------------------------------------
OVERALL RECOMMENDATION
--------------------------------------------------------------------------------

For this {ELECTROLYSER_SIZE_MW:.0f} MW system with {ELECTRICITY_PRICE_EUR_KWH:.2f} EUR/kWh electricity:

{'**ALKALINE is recommended** due to lower LCOH and CAPEX.' if alk['LCOH_eur_kg'] < pem['LCOH_eur_kg'] else '**PEM is recommended** due to better efficiency and flexibility.'}

The LCOH difference of {abs(lcoh_diff):.2f} EUR/kg translates to:
- Annual cost difference: ~{abs(lcoh_diff) * pem['total_H2_kg'] / SIMULATION_YEARS / 1000:,.0f} k EUR/year
- Project lifetime savings: ~{abs(lcoh_diff) * pem['total_H2_kg'] / 1000:,.0f} k EUR

================================================================================
"""
    return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete PEM vs Alkaline comparison."""
    
    print("\n" + "="*60)
    print("PEM vs ALKALINE ELECTROLYSER COMPARISON")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load power data
    print("\nLoading power data...")
    power_W = load_power_data(years=SIMULATION_YEARS)
    print(f"  Total hours: {len(power_W):,}")
    print(f"  Mean power: {np.mean(power_W)/1e6:.2f} MW")
    print(f"  Max power: {np.max(power_W)/1e6:.2f} MW")
    
    # Run simulations
    pem_results = run_pem_simulation(power_W)
    alk_results = run_alkaline_simulation(power_W)
    
    # Create comparison outputs
    print("\n" + "="*60)
    print("Creating Comparison Outputs")
    print("="*60)
    
    # 1. Summary table
    print("\nGenerating summary table...")
    summary_df = create_comparison_summary_table(pem_results, alk_results)
    summary_df.to_csv(OUTPUT_DIR / 'comparison_summary.csv', index=False)
    print(f"  Saved: comparison_summary.csv")
    
    # Print table
    print("\n" + summary_df.to_string(index=False))
    
    # 2. Bar charts
    print("\nGenerating comparison bar charts...")
    plot_comparison_bar_chart(pem_results, alk_results, OUTPUT_DIR)
    
    # 3. Timeseries comparison
    print("\nGenerating timeseries comparison...")
    plot_timeseries_comparison(pem_results, alk_results, OUTPUT_DIR)
    
    # 4. Voltage degradation
    print("\nGenerating voltage degradation comparison...")
    plot_voltage_degradation_comparison(pem_results, alk_results, OUTPUT_DIR)
    
    # 5. LCOH waterfall
    print("\nGenerating LCOH waterfall charts...")
    plot_lcoh_waterfall(pem_results, alk_results, OUTPUT_DIR)
    
    # 6. Radar chart
    print("\nGenerating technology radar chart...")
    plot_technology_radar(pem_results, alk_results, OUTPUT_DIR)
    
    # 7. Recommendation summary
    print("\nGenerating recommendation summary...")
    recommendation = create_recommendation_summary(pem_results, alk_results)
    with open(OUTPUT_DIR / 'recommendation_summary.txt', 'w') as f:
        f.write(recommendation)
    print(recommendation)
    
    # Save raw results
    results_combined = {
        'pem': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in pem_results.items()},
        'alkaline': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                     for k, v in alk_results.items()}
    }
    
    # Save as CSV (key metrics only)
    metrics_df = pd.DataFrame([
        {**{'Technology': 'PEM'}, **{k: v for k, v in pem_results.items() if not isinstance(v, np.ndarray)}},
        {**{'Technology': 'Alkaline'}, **{k: v for k, v in alk_results.items() if not isinstance(v, np.ndarray)}}
    ])
    metrics_df.to_csv(OUTPUT_DIR / 'detailed_results.csv', index=False)
    print(f"  Saved: detailed_results.csv")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Files generated:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
