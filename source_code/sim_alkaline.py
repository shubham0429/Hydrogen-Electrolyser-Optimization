"""
Alkaline Electrolyser Simulation Module
======================================
Complete techno-economic simulation for Alkaline Water Electrolysis (AWE).

This module is SEPARATE from PEM simulation (sim_concise.py) with:
- Alkaline-specific electrochemistry (lower current density, higher voltage)
- Linear degradation model + cycling penalty (based on industrial data)
- Different economic assumptions (lower CAPEX, different efficiency)

Key References:
- Nel Hydrogen: 1.5 μV/h linear degradation (continuous operation)
- NREL 2020: 128 μV/h with cycling (2-3x faster aging)
- ThyssenKrupp: 80,000h stack lifetime at baseload

Author: Thesis Project
Date: January 2026
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union
import warnings

# Type alias for numpy array or scalar
ArrayLike = Union[float, np.ndarray]


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
FARADAY = 96485.33  # C/mol - Faraday constant
R_GAS = 8.314       # J/(mol·K) - Universal gas constant
HHV_H2 = 39.41      # kWh/kg - Higher heating value of hydrogen
LHV_H2 = 33.33      # kWh/kg - Lower heating value of hydrogen
M_H2 = 2.016e-3     # kg/mol - Molar mass of hydrogen


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class AlkalineConfig:
    """
    Configuration for Alkaline electrolyser simulation.
    
    All parameters based on literature and industrial data for modern
    Alkaline Water Electrolysis (AWE) systems.
    
    Key differences from PEM:
    - Lower current density (0.2-0.4 A/cm² vs 1.5-2.0 A/cm²)
    - Higher cell voltage at nominal load
    - Linear degradation (not exponential)
    - Severe cycling penalty
    - Lower CAPEX, similar OPEX
    """
    
    # === SYSTEM SIZING ===
    P_nom_MW: float = 20.0          # Nominal electrolyser capacity [MW]
    n_stacks: int = 4               # Number of parallel stacks
    
    # === STACK DESIGN ===
    # Modern Advanced Alkaline (2024-2025): Higher current density, pressurized operation
    # References: IRENA (2024), Nel A-Series, ThyssenKrupp 20MW, Sunfire
    # 
    # P_stack = n_cells × V_cell × j_nom × A_cell
    # At j=0.50 A/cm², V≈1.75V: 5 MW = n_cells × 1.75 × 0.50 × A
    # Modern large-scale: 150-200 cells, 30,000-50,000 cm² per cell
    n_cells: int = 180              # Cells per stack (modern compact design)
    cell_area_cm2: float = 32000.0  # Active cell area [cm²] = 3.2 m² (modern industrial)
    # This gives: 180 × 1.75 × 0.50 × 32000 = 5.04 MW per stack ≈ 5 MW ✓
    
    # === OPERATING CONDITIONS ===
    # Modern pressurized Alkaline operates at higher T and P
    T_op_C: float = 80.0            # Operating temperature [°C] (70-90°C modern)
    p_op_bar: float = 30.0          # Operating pressure [bar] (30 bar - pressurized Alkaline)
    KOH_concentration: float = 0.30  # KOH electrolyte concentration (30% typical)
    
    # === ELECTROCHEMISTRY PARAMETERS ===
    # Reversible voltage (Nernst equation base)
    E_rev_0: float = 1.229          # Standard reversible voltage at 25°C [V]
    
    # Activation overpotential (Tafel equation)
    # For modern advanced Ni-Mo cathodes and Ni-Fe anodes
    # Reference: Schalenbach et al. (2018), Zeng & Zhang (2010)
    alpha_a: float = 0.35           # Anode transfer coefficient (improved catalysts)
    alpha_c: float = 0.55           # Cathode transfer coefficient (Ni-Mo cathodes)
    j0_a: float = 5e-4              # Anode exchange current density [A/cm²] (improved)
    j0_c: float = 5e-3              # Cathode exchange current density [A/cm²] (Ni-Mo)
    
    # Empirical coefficients (calibrated to MODERN Alkaline - Nel A-Series, ThyssenKrupp)
    # V = E_rev + r*j + s*log(t*j + 1)
    # Calibrated for: V ≈ 1.75V at j=0.50 A/cm², T=80°C, SEC ≈ 48-50 kWh/kg (stack)
    # Reference: David et al. (2019), Olivier et al. (2017)
    r1: float = 0.00004             # Ohmic parameter 1 [Ω·m²] - reduced for Zirfon
    r2: float = -4.0e-8             # Ohmic parameter 2 [Ω·m²/K]
    s1: float = 0.10                # Overvoltage parameter [V] - lower for better catalysts
    s2: float = -2.0e-4             # Overvoltage parameter [V/K]
    s3: float = 2.5e-7              # Overvoltage parameter [V/K²]
    t1: float = 0.12                # Log term parameter [m²/A]
    t2: float = 35.0                # Log term parameter [m²·K/A]
    t3: float = -700.0              # Log term parameter [m²·K²/A]
    
    use_empirical_model: bool = True  # Use empirical model instead of Butler-Volmer
    
    # Ohmic resistance - Modern Zirfon-type diaphragms have lower resistance
    # Reference: Brauns & Turek (2020), Zirfon Perl UTP 500
    R_cell_ohm_cm2: float = 0.18    # Cell resistance [Ω·cm²] (vs 0.25 for legacy)
    
    # Current density limits - MODERN values significantly higher
    # Reference: IRENA (2024), Nel A-500, ThyssenKrupp 20MW specs
    j_nom: float = 0.50             # Nominal current density [A/cm²] (vs 0.30 legacy)
    j_min: float = 0.05             # Minimum current density (10% of nominal)
    j_max: float = 0.80             # Maximum current density [A/cm²] (vs 0.45 legacy)
    
    # === FARADAY EFFICIENCY ===
    # Modern Zirfon diaphragms have much better gas separation
    # Reference: Schalenbach et al. (2016), AGFA Zirfon specs
    f1: float = 120.0               # Faraday efficiency parameter 1 [mA²/cm⁴] (improved)
    f2: float = 0.985               # Faraday efficiency parameter 2 [-] (higher for Zirfon)
    
    # === DEGRADATION PARAMETERS ===
    # MODERN Advanced Alkaline (2024-2025): Major improvements over legacy
    # References: 
    # [1] IRENA (2024) - "Green Hydrogen Cost Reduction"
    # [2] Nel ASA (2023) - A-Series warranty and lifetime data
    # [3] Smolinka et al. (2022) - "Electrolyser Technology Review"
    # [4] Brauns & Turek (2020) - "Alkaline Water Electrolysis Review"
    #
    # Modern improvements:
    # - Advanced Ni-Mo/Ni-Fe electrodes with nanostructured surfaces
    # - Zirfon Perl diaphragms (vs polysulfone) - better stability
    # - Improved cell compression and sealing
    # - Active thermal management systems
    # - Advanced power electronics for smoother operation
    #
    deg_rate_uV_h: float = 0.8        # Linear degradation rate [μV/h] - CONSERVATIVE (vs 0.5 optimistic)
    cycling_penalty_hours: float = 2.5  # Hours of equivalent aging per cycle - CONSERVATIVE
    stack_lifetime_hours: float = 80000  # Design lifetime [h] - CONSERVATIVE (vs 90k-100k optimistic)
    voltage_increase_limit: float = 0.08  # 8% voltage increase triggers replacement (more conservative)
    
    # === THERMAL PARAMETERS ===
    T_ambient_mean_C: float = 15.0  # Mean ambient temperature [°C]
    T_ambient_amp_C: float = 10.0   # Seasonal temperature amplitude [°C]
    
    # === BALANCE OF PLANT ===
    # Modern systems have improved power electronics and reduced parasitic loads
    eta_rectifier: float = 0.96     # Rectifier efficiency (conservative for large-scale)
    eta_transformer: float = 0.99   # Transformer efficiency
    P_bop_fraction: float = 0.06    # BoP parasitic load (% of stack power) - reduced
    
    # === MINIMUM LOAD AND RAMP RATE ===
    # MODERN Advanced Alkaline has improved turndown ratio but still worse than PEM
    # Reference: Nel A-Series (2024) specs, ThyssenKrupp scalum
    # PEM can go to 10%, Alkaline typically 20-30% minimum
    min_load_fraction: float = 0.30  # Minimum stable load (30% for MODERN - vs 20-25% legacy, vs 10% PEM)
    max_ramp_rate_per_h: float = 0.50  # Maximum ramp rate [fraction/hour] (50%/h MODERN)
    # Reference: Nel (2024) - Modern Alkaline can ramp 10-20%/sec when warm
    # We use 50%/h as conservative for hourly simulation with cold starts
    
    # === PARTIAL LOAD EFFICIENCY ===
    # Alkaline has worse part-load behavior than PEM due to:
    # - Higher parasitic losses relative to stack power at low loads
    # - Increased gas crossover at low current densities
    # - Less efficient thermal management at low heat generation
    # Reference: Brauns & Turek (2020), IRENA (2024)
    partial_load_threshold: float = 0.40  # Below 40% load, efficiency drops significantly
    partial_load_eff_min: float = 0.75    # Minimum efficiency factor at min load (75% - parasitic loads dominate)
    
    # === ECONOMIC PARAMETERS ===
    # CAPEX - German 2024-2025 CONSERVATIVE values
    # Reference: IRENA (2024), BloombergNEF (2024), German electrolyser projects
    capex_stack_eur_kW: float = 550.0    # Stack CAPEX [EUR/kW] (conservative 2024-2025)
    capex_bop_eur_kW: float = 400.0      # Balance of Plant CAPEX [EUR/kW]
    capex_installation_eur_kW: float = 120.0  # Installation, commissioning
    capex_engineering_eur_kW: float = 80.0    # Engineering, project management
    # Total: ~1150 EUR/kW (conservative 2024-2025)
    # vs PEM ~1950 EUR/kW (Alkaline cheaper but not by as much as optimistic estimates)
    
    # System boundary items (harmonized with PEM for fair comparison)
    # Alkaline needs slightly less water treatment (no ultra-pure DI water needed)
    # Reference: IRENA (2024), IEA (2024)
    capex_water_treatment_eur_kW: float = 30.0   # Water treatment [EUR/kW] (simpler than PEM)
    capex_site_preparation_eur_kW: float = 40.0   # Site preparation, civil works [EUR/kW]
    capex_grid_connection_eur_kW: float = 35.0    # Grid connection [EUR/kW]
    
    # Storage and Compression CAPEX (aligned with PEM for fair comparison)
    # Reference: IEA (2024), IRENA (2024)
    storage_capex_eur_kg: float = 800.0      # Type IV tanks @ 350 bar [EUR/kg] (conservative)
    compressor_capex_eur_kg: float = 55.0    # Compressor CAPEX [EUR/kg capacity]
    compressor_maint_fraction: float = 0.04  # Compressor maintenance [fraction of CAPEX/year]
    
    # Curtailment penalty (opportunity cost of wasted renewable energy)
    curtailment_penalty_eur_per_MWh: float = 25.0  # €25/MWh (50% of electricity cost)
    enable_curtailment_penalty: bool = False       # Disabled — standard LCOH excludes opportunity costs
    
    # === COMPONENT REPLACEMENT SCHEDULES (Alkaline-specific) ===
    # Alkaline has different component lifetimes than PEM
    # Reference: Nel Hydrogen (2024), ThyssenKrupp (2024), Academic literature
    #
    # Component replacement intervals (years at 60% CF, ~5000 h/year):
    # - Electrolyte (KOH): Every 3-5 years (contamination, dilution)
    # - Electrodes/Catalyst: Every 10-15 years (performance degradation)
    # - Diaphragm/Separator: Every 8-12 years (crossover increase)
    # - Full stack: Every 15-20 years (cumulative degradation)
    # - Mechanical seals/gaskets: Every 5-7 years
    electrolyte_replacement_years: float = 4.0    # KOH replacement interval [years]
    catalyst_replacement_years: float = 12.0       # Electrode/catalyst replacement [years]
    diaphragm_replacement_years: float = 10.0      # Diaphragm/separator replacement [years]
    mechanical_replacement_years: float = 6.0      # Seals, gaskets, etc. [years]
    
    # Component replacement costs as fraction of stack CAPEX
    electrolyte_replacement_frac: float = 0.06    # Electrolyte: 6% of stack CAPEX (conservative)
    catalyst_replacement_frac: float = 0.28       # Electrodes: 28% of stack CAPEX
    diaphragm_replacement_frac: float = 0.18      # Diaphragm: 18% of stack CAPEX  
    mechanical_replacement_frac: float = 0.10     # Mechanical: 10% of stack CAPEX
    stack_replacement_cost_fraction: float = 0.55  # Full stack: 55% of initial stack cost
    
    # Learning rate for cost reduction
    learning_rate: float = 0.03  # 3% annual cost reduction (conservative for mature tech)
    
    # OPEX - German 2024-2025 CONSERVATIVE
    opex_fixed_fraction: float = 0.04    # Fixed O&M [4% of CAPEX/year] (conservative)
    opex_fixed_eur_kW_yr: float = 25.0   # Fixed O&M [EUR/kW/year] (alternative)
    land_lease_eur_kW_yr: float = 5.0    # Annual land lease [EUR/kW/year] (harmonized with PEM)
    electricity_price_eur_kWh: float = 0.07  # Electricity price [EUR/kWh]
    water_price_eur_m3: float = 2.50     # Water price [EUR/m³] (conservative)
    water_consumption_L_kg: float = 10.0  # Water consumption [L/kg H2]
    
    # Compression to 350 bar (same as PEM for fair comparison)
    compression_energy_kWh_kg: float = 3.5  # Compression to 350 bar [kWh/kg]
    
    # === BY-PRODUCT REVENUES ===
    # Oxygen credit - industrial O2 market value (harmonized with PEM)
    # Reference: Bertuccioli et al. (2014), IRENA (2020), Air Liquide market data
    enable_oxygen_credit: bool = True            # Enable oxygen revenue
    oxygen_selling_price_eur_tonne: float = 50.0   # €50/tonne O2 (conservative, harmonized with PEM)
    oxygen_to_h2_mass_ratio: float = 8.0         # 8 kg O2 per kg H2 (stoichiometric)
    oxygen_purity: float = 0.995                 # 99.5% purity from electrolyser
    
    # Waste heat recovery - low-grade heat for district heating/industrial use
    # Reference: Ursúa et al. (2012), Buttler & Spliethoff (2018)
    enable_heat_recovery: bool = True             # Enable heat recovery revenue
    heat_recovery_efficiency: float = 0.45        # 45% of electrical input as recoverable heat (conservative)
    heat_temperature_C: float = 70.0              # Heat output temperature (60-80°C range)
    heat_selling_price_eur_MWh: float = 15.0      # €15/MWh thermal (conservative, harmonized with PEM)
    
    # === COMPONENT REPLACEMENT PERFORMANCE IMPROVEMENTS ===
    # Fresh components restore performance closer to new condition
    # Reference: Haug et al. (2017), Brauns & Turek (2020)
    enable_replacement_performance_boost: bool = True  # Model performance improvement after replacement
    electrolyte_replacement_efficiency_gain: float = 0.04  # 4% efficiency gain (removes contamination)
    catalyst_replacement_efficiency_gain: float = 0.05    # 5% efficiency gain (restores kinetics)
    diaphragm_replacement_efficiency_gain: float = 0.03   # 3% efficiency gain (reduces crossover)
    mechanical_replacement_efficiency_gain: float = 0.01  # 1% efficiency gain (sealing improvement)
    
    # Financial parameters - CONSERVATIVE
    discount_rate: float = 0.08      # Discount rate (8%)
    inflation_rate: float = 0.025    # 2.5% annual inflation
    contingency_fraction: float = 0.10  # 10% project contingency
    project_lifetime_years: int = 15  # Project lifetime [years] - aligned with PEM
    
    # Revenue (for profitability analysis)
    h2_selling_price_eur_kg: float = 6.0  # H2 selling price [EUR/kg]
    
    # === HYDROGEN STORAGE (optional) ===
    storage_capacity_kg: float = 0.0     # Storage capacity [kg] (0 = no storage)
    storage_initial_fraction: float = 0.5  # Initial fill level
    storage_efficiency: float = 0.98      # Round-trip efficiency
    
    # === DEMAND PROFILE ===
    demand_kg_h: float = 0.0             # Constant demand [kg/h] (0 = no demand constraint)
    
    # === SIMULATION PARAMETERS ===
    dt_hours: float = 1.0            # Timestep [hours]
    simulation_years: int = 15       # Simulation duration [years] - aligned with PEM
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Calculate derived parameters
        self.P_nom_W = self.P_nom_MW * 1e6
        self.T_op_K = self.T_op_C + 273.15
        
        # Validate ranges
        if not 0.10 <= self.j_nom <= 0.50:
            warnings.warn(f"Unusual nominal current density: {self.j_nom} A/cm²")
        if not 500 <= self.capex_stack_eur_kW + self.capex_bop_eur_kW <= 1000:
            warnings.warn(f"CAPEX outside typical Alkaline range: {self.capex_stack_eur_kW + self.capex_bop_eur_kW} EUR/kW")


def get_alkaline_config(**kwargs) -> AlkalineConfig:
    """
    Factory function to create Alkaline configuration.
    
    Parameters
    ----------
    **kwargs : dict
        Override any default configuration parameter.
        
    Returns
    -------
    AlkalineConfig
        Configuration object with all parameters.
        
    Example
    -------
    >>> config = get_alkaline_config(P_nom_MW=10.0, electricity_price_eur_kWh=0.04)
    """
    return AlkalineConfig(**kwargs)


# =============================================================================
# ELECTROCHEMISTRY FUNCTIONS
# =============================================================================

def compute_reversible_voltage(T_K: float, p_bar: float = 1.0) -> float:
    """
    Compute temperature-dependent reversible cell voltage (Nernst equation).
    
    E_rev(T) = 1.229 - 0.9e-3 * (T - 298.15) + (RT/2F) * ln(p_H2 * sqrt(p_O2))
    
    Simplified for Alkaline with atmospheric products.
    
    Parameters
    ----------
    T_K : float
        Operating temperature [K]
    p_bar : float
        Operating pressure [bar]
        
    Returns
    -------
    float
        Reversible voltage [V]
        
    Reference
    ---------
    Ulleberg (2003), Modeling of advanced alkaline electrolyzers
    """
    E_rev = 1.229 - 0.9e-3 * (T_K - 298.15)
    
    # Pressure correction (Nernst)
    if p_bar > 1.0:
        E_rev += (R_GAS * T_K) / (2 * FARADAY) * np.log(p_bar)
    
    return E_rev


def compute_activation_overpotential(
    j: ArrayLike,
    T_K: float,
    config: AlkalineConfig
) -> ArrayLike:
    """
    Compute activation overpotential using Tafel equation.
    
    η_act = (RT/αF) * asinh(j / 2j0)
    
    For Alkaline: Both anode (OER) and cathode (HER) contribute.
    
    Parameters
    ----------
    j : float or array
        Current density [A/cm²]
    T_K : float
        Temperature [K]
    config : AlkalineConfig
        Configuration with exchange current densities
        
    Returns
    -------
    float or array
        Activation overpotential [V]
    """
    # Anode activation (oxygen evolution - rate limiting)
    eta_a = (R_GAS * T_K) / (config.alpha_a * FARADAY) * np.arcsinh(j / (2 * config.j0_a))
    
    # Cathode activation (hydrogen evolution)
    eta_c = (R_GAS * T_K) / (config.alpha_c * FARADAY) * np.arcsinh(j / (2 * config.j0_c))
    
    return eta_a + eta_c


def compute_ohmic_overpotential(
    j: ArrayLike,
    T_K: float,
    config: AlkalineConfig
) -> ArrayLike:
    """
    Compute ohmic overpotential.
    
    η_ohm = j * R_cell
    
    For Alkaline, R_cell includes:
    - Electrolyte (KOH) resistance
    - Diaphragm/separator resistance
    - Electrode and contact resistance
    
    Temperature dependence: R decreases ~2%/°C
    
    Parameters
    ----------
    j : float or array
        Current density [A/cm²]
    T_K : float
        Temperature [K]
    config : AlkalineConfig
        Configuration with cell resistance
        
    Returns
    -------
    float or array
        Ohmic overpotential [V]
    """
    # Temperature correction (resistance decreases with temperature)
    T_ref = 343.15  # 70°C reference
    R_cell = config.R_cell_ohm_cm2 * (1 - 0.02 * (T_K - T_ref))
    R_cell = np.maximum(R_cell, 0.1 * config.R_cell_ohm_cm2)  # Floor at 10% of base
    
    return j * R_cell


def compute_cell_voltage(
    j: ArrayLike,
    T_K: float,
    config: AlkalineConfig,
    degradation_V: float = 0.0
) -> ArrayLike:
    """
    Compute total cell voltage at given current density.
    
    Two models available:
    1. Empirical (Ulleberg 2003) - calibrated to real Alkaline data
    2. Butler-Volmer (first-principles) - for sensitivity studies
    
    Parameters
    ----------
    j : float or array
        Current density [A/cm²]
    T_K : float
        Temperature [K]
    config : AlkalineConfig
        Configuration object
    degradation_V : float
        Cumulative degradation voltage increase [V]
        
    Returns
    -------
    float or array
        Cell voltage [V]
        
    Notes
    -----
    Typical Alkaline cell voltage: 1.8-2.0V at 0.3 A/cm²
    """
    if config.use_empirical_model:
        # Ulleberg empirical model (validated against Nel, Hydrogenics data)
        # V = E_rev + r*j + s*log(t*j + 1)
        
        E_rev = compute_reversible_voltage(T_K, config.p_op_bar)
        
        # Ohmic resistance term (Ω·m²)
        r = config.r1 + config.r2 * T_K
        
        # Overvoltage coefficient
        s = config.s1 + config.s2 * T_K + config.s3 * T_K**2
        
        # Log term coefficient (convert j from A/cm² to A/m²)
        j_m2 = j * 1e4  # A/cm² to A/m²
        t = config.t1 + config.t2 / T_K + config.t3 / T_K**2
        
        # Prevent log of negative numbers
        log_arg = np.maximum(t * j_m2 + 1, 1e-6)
        
        V_cell = E_rev + r * j_m2 + s * np.log(log_arg) + degradation_V
        
    else:
        # Butler-Volmer model
        E_rev = compute_reversible_voltage(T_K, config.p_op_bar)
        eta_act = compute_activation_overpotential(j, T_K, config)
        eta_ohm = compute_ohmic_overpotential(j, T_K, config)
        V_cell = E_rev + eta_act + eta_ohm + degradation_V
    
    # Ensure voltage is reasonable
    V_cell = np.clip(V_cell, 1.4, 3.0)
    
    return V_cell


def compute_faraday_efficiency(
    j: ArrayLike,
    T_K: float,
    config: AlkalineConfig
) -> ArrayLike:
    """
    Compute Faraday (current) efficiency using Ulleberg model.
    
    η_F = (j² / (f1 + j²)) * f2
    
    Accounts for parasitic currents and gas crossover.
    Lower at low current densities.
    
    Parameters
    ----------
    j : float or array
        Current density [A/cm²]
    T_K : float
        Temperature [K] (minor effect)
    config : AlkalineConfig
        Configuration with f1, f2 parameters
        
    Returns
    -------
    float or array
        Faraday efficiency [0-1]
        
    Reference
    ---------
    Ulleberg (2003), DOI: 10.1016/S0360-3199(02)00033-2
    """
    # Convert to mA/cm² for Ulleberg model
    j_mA = j * 1000
    
    # Ulleberg model
    eta_F = (j_mA**2 / (config.f1 + j_mA**2)) * config.f2
    
    # Temperature correction (slight improvement at higher T)
    T_factor = 1 + 0.001 * (T_K - 343.15)
    eta_F = eta_F * np.clip(T_factor, 0.95, 1.02)
    
    return np.clip(eta_F, 0.0, 1.0)


def compute_hydrogen_production_rate(
    I_total_A: ArrayLike,
    eta_F: ArrayLike
) -> ArrayLike:
    """
    Compute hydrogen production rate from Faraday's law.
    
    ṁ_H2 = (η_F * I * M_H2) / (2 * F)
    
    Parameters
    ----------
    I_total_A : float or array
        Total stack current [A]
    eta_F : float or array
        Faraday efficiency [0-1]
        
    Returns
    -------
    float or array
        Hydrogen production rate [kg/s]
    """
    # Faraday's law: n_H2 = I / (2F) [mol/s]
    n_H2_mol_s = I_total_A / (2 * FARADAY)
    
    # Mass flow rate
    m_H2_kg_s = n_H2_mol_s * M_H2 * eta_F
    
    return m_H2_kg_s


def compute_stack_efficiency(
    V_cell: ArrayLike,
    eta_F: ArrayLike,
    basis: str = 'LHV'
) -> ArrayLike:
    """
    Compute stack efficiency (HHV or LHV basis).
    
    η_stack = (V_tn / V_cell) * η_F
    
    where V_tn is thermoneutral voltage (1.48V for HHV, 1.25V for LHV)
    
    Parameters
    ----------
    V_cell : float or array
        Cell voltage [V]
    eta_F : float or array
        Faraday efficiency [0-1]
    basis : str
        'HHV' or 'LHV' for efficiency calculation
        
    Returns
    -------
    float or array
        Stack efficiency [0-1]
    """
    V_tn = 1.481 if basis == 'HHV' else 1.253
    
    eta_stack = (V_tn / V_cell) * eta_F
    
    return np.clip(eta_stack, 0.0, 1.0)


def compute_system_efficiency(
    eta_stack: ArrayLike,
    config: AlkalineConfig
) -> ArrayLike:
    """
    Compute overall system efficiency including BoP losses.
    
    η_sys = η_stack * η_rectifier * η_transformer * (1 - P_bop_fraction)
    
    Parameters
    ----------
    eta_stack : float or array
        Stack efficiency [0-1]
    config : AlkalineConfig
        Configuration with BoP parameters
        
    Returns
    -------
    float or array
        System efficiency [0-1]
    """
    eta_sys = (eta_stack 
               * config.eta_rectifier 
               * config.eta_transformer 
               * (1 - config.P_bop_fraction))
    
    return eta_sys


def compute_partial_load_efficiency_factor(
    load_fraction: float,
    config: AlkalineConfig
) -> float:
    """
    Compute efficiency penalty factor for partial load operation.
    
    At low loads, efficiency drops due to:
    1. Higher relative parasitic losses (pumps, controls run at same power)
    2. Increased gas crossover at low current densities
    3. Less optimal thermal management
    
    For Alkaline: Efficiency drops below ~40% load (worse than PEM's ~30%)
    
    Parameters
    ----------
    load_fraction : float
        Current load as fraction of nominal [0-1]
    config : AlkalineConfig
        Configuration with partial load parameters
        
    Returns
    -------
    float
        Efficiency factor [0-1] to multiply with base efficiency
        
    References
    ----------
    [1] Ursúa et al. (2012) - "Hydrogen Production from Water Electrolysis: 
        Current Status and Future Trends", Proc. IEEE, 100(2), 410-426.
    [2] Brauns & Turek (2020) - "Alkaline Water Electrolysis Powered by 
        Renewable Energy: A Review", Processes, 8(2), 248.
    [3] Buttler & Spliethoff (2018) - "Current Status of Water Electrolysis 
        for Energy Storage", Renewable and Sustainable Energy Reviews, 82, 2440-2454.
    
    Notes
    -----
    Alkaline electrolysers have worse part-load behavior than PEM due to:
    - Higher internal resistance at low currents
    - Gas purity issues (increased crossover) below ~20-25% load
    - Thermal management difficulties at low heat generation
    """
    threshold = config.partial_load_threshold
    eff_min = config.partial_load_eff_min
    
    if load_fraction >= threshold:
        # Above threshold: no penalty
        return 1.0
    elif load_fraction <= config.min_load_fraction:
        # At or below minimum load: maximum penalty
        return eff_min
    else:
        # Linear interpolation between min_load and threshold
        # From eff_min at min_load to 1.0 at threshold
        slope = (1.0 - eff_min) / (threshold - config.min_load_fraction)
        return eff_min + slope * (load_fraction - config.min_load_fraction)


def compute_specific_energy_consumption(
    eta_sys: ArrayLike,
    basis: str = 'LHV'
) -> ArrayLike:
    """
    Compute specific energy consumption [kWh/kg H2].
    
    SEC = HV_H2 / η_sys
    
    Parameters
    ----------
    eta_sys : float or array
        System efficiency [0-1]
    basis : str
        'HHV' or 'LHV'
        
    Returns
    -------
    float or array
        Specific energy consumption [kWh/kg H2]
        
    Notes
    -----
    Typical Alkaline SEC: 50-55 kWh/kg (LHV basis)
    vs PEM: 50-55 kWh/kg (similar, but PEM better at part load)
    """
    HV = HHV_H2 if basis == 'HHV' else LHV_H2
    
    SEC = HV / np.maximum(eta_sys, 0.01)
    
    return SEC


# =============================================================================
# POWER-CURRENT CONVERSION
# =============================================================================

def power_to_current_density(
    P_stack_W: ArrayLike,
    config: AlkalineConfig,
    T_K: float,
    degradation_V: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 50
) -> ArrayLike:
    """
    Convert stack power to current density using Newton-Raphson.
    
    Solves: P = n_cells * V_cell(j) * j * A_cell
    
    Parameters
    ----------
    P_stack_W : float or array
        Stack power input [W]
    config : AlkalineConfig
        Configuration object
    T_K : float
        Temperature [K]
    degradation_V : float
        Voltage degradation [V]
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float or array
        Current density [A/cm²]
    """
    # Initial guess from linear approximation
    V_approx = 2.0  # Approximate cell voltage
    P_per_cell = P_stack_W / config.n_cells
    j = P_per_cell / (V_approx * config.cell_area_cm2)
    j = np.clip(j, config.j_min, config.j_max)
    
    # Newton-Raphson iteration
    for _ in range(max_iter):
        V_cell = compute_cell_voltage(j, T_K, config, degradation_V)
        P_calc = config.n_cells * V_cell * j * config.cell_area_cm2
        
        error = P_calc - P_stack_W
        
        if np.all(np.abs(error) < tol * P_stack_W):
            break
            
        # Numerical derivative
        dj = 1e-6
        V_cell_dj = compute_cell_voltage(j + dj, T_K, config, degradation_V)
        P_calc_dj = config.n_cells * V_cell_dj * (j + dj) * config.cell_area_cm2
        
        dP_dj = (P_calc_dj - P_calc) / dj
        dP_dj = np.where(np.abs(dP_dj) < 1e-10, 1e-10, dP_dj)
        
        j = j - error / dP_dj
        j = np.clip(j, config.j_min, config.j_max)
    
    return j


# =============================================================================
# TEMPERATURE MODEL
# =============================================================================

def compute_operating_temperature(
    hour_of_year: ArrayLike,
    config: AlkalineConfig
) -> ArrayLike:
    """
    Compute operating temperature with seasonal variation.
    
    Simple sinusoidal model for ambient temperature affecting
    cooling requirements and thus stack temperature.
    
    Parameters
    ----------
    hour_of_year : int or array
        Hour of year [0-8759]
    config : AlkalineConfig
        Configuration with temperature parameters
        
    Returns
    -------
    float or array
        Operating temperature [K]
        
    Notes
    -----
    Limitation: This is a simplified model. Real systems have
    active thermal management that maintains more constant temperature.
    """
    hours_per_year = 8760
    
    # Seasonal variation
    T_ambient = (config.T_ambient_mean_C + 
                 config.T_ambient_amp_C * np.sin(2 * np.pi * hour_of_year / hours_per_year - np.pi/2))
    
    # Stack temperature rises above ambient
    # Cooling system maintains ~70°C but has some ambient dependence
    delta_T = config.T_op_C - config.T_ambient_mean_C
    T_stack = T_ambient + delta_T
    
    # Clamp to reasonable operating range
    T_stack = np.clip(T_stack, 60, 80)  # Alkaline: 60-80°C
    
    return T_stack + 273.15  # Convert to Kelvin


# =============================================================================
# DEGRADATION MODEL
# =============================================================================
# Modern Alkaline Electrolyzers (2023-2025) - Significantly Improved
#
# Key improvements over legacy systems:
# 1. Advanced Ni-based electrodes: Better microstructure, higher stability
# 2. Zirfon/AGFA diaphragms: Replace asbestos, better ionic conductivity
# 3. Improved thermal management: Faster warm-up, reduced thermal stress
# 4. Better gaskets/seals: Reduced degradation from pressure cycling
#
# Degradation characteristics:
# - Still LINEAR (unlike PEM which is more exponential)
# - Reduced cycling penalty (2-3h vs 7-10h for legacy)
# - Longer baseload lifetime (90,000-100,000h vs 60,000-80,000h)
#
# References:
# - IRENA (2023): Green Hydrogen Cost Reduction - Scaling up Electrolysers
# - Nel Hydrogen ASeries specifications (2024)
# - ThyssenKrupp nucera technical documentation (2024)
# - Trinke et al. (2024): Int J Hydrogen Energy - Alkaline cycling performance
# =============================================================================

@dataclass
class DegradationState:
    """
    Track degradation state over simulation.
    
    Alkaline degradation consists of:
    1. Time-based (linear): ~1.5 μV/h continuous operation
    2. Cycling penalty: Each start = 5-10h equivalent aging
    3. Load cycling: Rapid load changes cause additional stress
    """
    cumulative_hours: float = 0.0       # Total operating hours
    cumulative_cycles: int = 0          # Number of start/stop cycles
    cumulative_load_changes: int = 0    # Number of significant load changes
    voltage_degradation_V: float = 0.0  # Cumulative voltage increase [V]
    capacity_factor: float = 1.0        # Remaining capacity [0-1]
    stack_replacements: int = 0         # Number of stack replacements
    replacement_hours: list = field(default_factory=list)  # Hours when replaced
    
    # Component replacement performance boosts
    efficiency_multiplier: float = 1.0   # Cumulative efficiency improvement from replacements
    last_electrolyte_replacement_year: int = 0
    last_catalyst_replacement_year: int = 0
    last_diaphragm_replacement_year: int = 0
    last_mechanical_replacement_year: int = 0
    
    def __post_init__(self):
        if not isinstance(self.replacement_hours, list):
            self.replacement_hours = []


def compute_degradation_rate(
    is_operating: bool,
    was_operating: bool,
    load_fraction: float,
    prev_load_fraction: float,
    config: AlkalineConfig
) -> tuple:
    """
    Compute degradation for current timestep.
    
    Degradation sources:
    1. Time-based: Linear voltage increase during operation
    2. Start/stop: Penalty per cycle (thermal/mechanical stress)
    3. Load transients: Additional penalty for large load changes
    
    Parameters
    ----------
    is_operating : bool
        Currently operating (above min load)
    was_operating : bool
        Was operating in previous timestep
    load_fraction : float
        Current load as fraction of nominal [0-1]
    prev_load_fraction : float
        Previous load fraction
    config : AlkalineConfig
        Configuration object
        
    Returns
    -------
    tuple
        (delta_V_degradation, is_cycle, is_load_change)
        
    References
    ----------
    - IRENA (2023): Modern Alkaline achieves 0.5-1.0 μV/h
    - Nel Hydrogen: 0.8 μV/h baseline for A-Series
    - ThyssenKrupp nucera: 90,000h+ lifetime at baseload
    - Cycling penalty reduced to 2-3h equivalent per start (from 7-10h legacy)
    """
    delta_V = 0.0
    is_cycle = False
    is_load_change = False
    
    if is_operating:
        # 1. Time-based linear degradation [V/h]
        # Convert μV/h to V/h
        delta_V += config.deg_rate_uV_h * 1e-6 * config.dt_hours
        
        # 2. Cycling penalty (start-up)
        if not was_operating:
            # Each start = cycling_penalty_hours of equivalent aging
            cycle_penalty_V = config.cycling_penalty_hours * config.deg_rate_uV_h * 1e-6
            delta_V += cycle_penalty_V
            is_cycle = True
        
        # 3. Load transient penalty
        # Significant load change = >30% change in load fraction
        load_change = abs(load_fraction - prev_load_fraction)
        if load_change > 0.30 and was_operating:
            # Smaller penalty than full cycle, but still significant
            transient_penalty_V = 0.5 * config.deg_rate_uV_h * 1e-6  # 0.5h equivalent
            delta_V += transient_penalty_V
            is_load_change = True
    
    return delta_V, is_cycle, is_load_change


def compute_equivalent_operating_hours(
    operating_hours: float,
    n_cycles: int,
    n_load_changes: int,
    config: AlkalineConfig
) -> float:
    """
    Compute equivalent operating hours including cycling penalty.
    
    Equivalent hours = actual hours + (cycles × penalty_hours) + (load_changes × 0.5)
    
    This is used for stack lifetime estimation.
    
    Parameters
    ----------
    operating_hours : float
        Actual operating hours
    n_cycles : int
        Number of start/stop cycles
    n_load_changes : int
        Number of significant load changes
    config : AlkalineConfig
        Configuration object
        
    Returns
    -------
    float
        Equivalent operating hours
    """
    equiv_hours = (operating_hours 
                   + n_cycles * config.cycling_penalty_hours
                   + n_load_changes * 0.5)  # Load changes = 0.5h each
    
    return equiv_hours


def check_stack_replacement(
    degradation_state: DegradationState,
    V_cell_nominal: float,
    config: AlkalineConfig
) -> tuple:
    """
    Check if stack replacement is needed based on voltage degradation.
    
    Replacement criteria:
    1. Voltage increase > voltage_increase_limit (default 10%)
    2. Equivalent hours > stack_lifetime_hours
    
    Parameters
    ----------
    degradation_state : DegradationState
        Current degradation state
    V_cell_nominal : float
        Initial (BoL) cell voltage at nominal conditions [V]
    config : AlkalineConfig
        Configuration object
        
    Returns
    -------
    tuple
        (needs_replacement, reason)
    """
    # Check voltage increase
    voltage_increase_frac = degradation_state.voltage_degradation_V / V_cell_nominal
    if voltage_increase_frac >= config.voltage_increase_limit:
        return True, f"Voltage increased by {voltage_increase_frac:.1%} (limit: {config.voltage_increase_limit:.0%})"
    
    # Check equivalent hours
    equiv_hours = compute_equivalent_operating_hours(
        degradation_state.cumulative_hours,
        degradation_state.cumulative_cycles,
        degradation_state.cumulative_load_changes,
        config
    )
    if equiv_hours >= config.stack_lifetime_hours:
        return True, f"Equivalent hours {equiv_hours:.0f}h exceeded lifetime {config.stack_lifetime_hours:.0f}h"
    
    return False, ""


def reset_degradation_state(
    degradation_state: DegradationState,
    current_hour: int
) -> DegradationState:
    """
    Reset degradation state after stack replacement.
    
    Parameters
    ----------
    degradation_state : DegradationState
        Current state
    current_hour : int
        Simulation hour when replacement occurs
        
    Returns
    -------
    DegradationState
        New state after replacement
    """
    new_state = DegradationState(
        cumulative_hours=0.0,
        cumulative_cycles=0,
        cumulative_load_changes=0,
        voltage_degradation_V=0.0,
        capacity_factor=1.0,
        stack_replacements=degradation_state.stack_replacements + 1,
        replacement_hours=degradation_state.replacement_hours + [current_hour],
        # Preserve efficiency multiplier from component replacements
        efficiency_multiplier=degradation_state.efficiency_multiplier,
        last_electrolyte_replacement_year=degradation_state.last_electrolyte_replacement_year,
        last_catalyst_replacement_year=degradation_state.last_catalyst_replacement_year,
        last_diaphragm_replacement_year=degradation_state.last_diaphragm_replacement_year,
        last_mechanical_replacement_year=degradation_state.last_mechanical_replacement_year
    )
    return new_state


def estimate_remaining_lifetime(
    degradation_state: DegradationState,
    avg_cycles_per_year: float,
    avg_operating_hours_per_year: float,
    config: AlkalineConfig
) -> float:
    """
    Estimate remaining stack lifetime in years.
    
    Parameters
    ----------
    degradation_state : DegradationState
        Current degradation state
    avg_cycles_per_year : float
        Average start/stop cycles per year
    avg_operating_hours_per_year : float
        Average operating hours per year
    config : AlkalineConfig
        Configuration object
        
    Returns
    -------
    float
        Estimated remaining lifetime [years]
    """
    # Current equivalent hours
    current_equiv_hours = compute_equivalent_operating_hours(
        degradation_state.cumulative_hours,
        degradation_state.cumulative_cycles,
        degradation_state.cumulative_load_changes,
        config
    )
    
    # Remaining hours
    remaining_hours = config.stack_lifetime_hours - current_equiv_hours
    
    if remaining_hours <= 0:
        return 0.0
    
    # Annual equivalent hours consumption
    annual_equiv_hours = (avg_operating_hours_per_year 
                          + avg_cycles_per_year * config.cycling_penalty_hours)
    
    if annual_equiv_hours <= 0:
        return float('inf')
    
    return remaining_hours / annual_equiv_hours


def compute_capacity_fade(
    voltage_degradation_V: float,
    V_cell_nominal: float
) -> float:
    """
    Compute capacity fade based on voltage degradation.
    
    As voltage increases, the electrolyser can produce less H2
    for the same power input (or needs more power for same H2).
    
    Approximation: capacity_factor ≈ V_nominal / V_degraded
    
    Parameters
    ----------
    voltage_degradation_V : float
        Cumulative voltage increase [V]
    V_cell_nominal : float
        Initial cell voltage [V]
        
    Returns
    -------
    float
        Capacity factor [0-1]
    """
    V_degraded = V_cell_nominal + voltage_degradation_V
    capacity_factor = V_cell_nominal / V_degraded
    
    return np.clip(capacity_factor, 0.5, 1.0)  # Floor at 50%


# =============================================================================
# PART 3: MAIN SIMULATION FUNCTION
# =============================================================================

@dataclass
class SimulationResults:
    """
    Container for all simulation outputs.
    
    Stores hourly timeseries and aggregated metrics for the full simulation.
    """
    # Timeseries arrays (hourly, length = n_hours)
    hours: np.ndarray                    # Hour index [0, 1, 2, ...]
    power_available_W: np.ndarray        # Available renewable power [W]
    power_consumed_W: np.ndarray         # Actual power consumed [W]
    h2_production_kg: np.ndarray         # H2 produced per hour [kg]
    current_density_A_cm2: np.ndarray    # Operating current density [A/cm²]
    cell_voltage_V: np.ndarray           # Cell voltage [V]
    faraday_efficiency: np.ndarray       # Faraday efficiency [-]
    system_efficiency: np.ndarray        # System efficiency [-]
    sec_kWh_kg: np.ndarray               # Specific energy consumption [kWh/kg]
    temperature_K: np.ndarray            # Operating temperature [K]
    
    # Degradation tracking
    voltage_degradation_V: np.ndarray    # Cumulative voltage degradation [V]
    capacity_factor: np.ndarray          # Capacity factor [-]
    is_operating: np.ndarray             # Operating flag (bool)
    n_cycles_cumulative: np.ndarray      # Cumulative start/stop cycles
    
    # Storage (if enabled)
    storage_level_kg: np.ndarray         # Storage level [kg]
    h2_to_demand_kg: np.ndarray          # H2 delivered to demand [kg]
    unmet_demand_kg: np.ndarray          # Unmet demand [kg]
    
    # Stack replacement events
    stack_replacements: int = 0
    replacement_hours: list = field(default_factory=list)
    
    # Summary metrics (computed after simulation)
    total_h2_production_kg: float = 0.0
    total_energy_consumed_kWh: float = 0.0
    average_sec_kWh_kg: float = 0.0
    capacity_factor_avg: float = 0.0
    total_cycles: int = 0
    total_operating_hours: float = 0.0
    availability: float = 0.0
    
    def __post_init__(self):
        if not isinstance(self.replacement_hours, list):
            self.replacement_hours = []


def add_deterministic_variability(
    data: np.ndarray,
    variability_pct: float = 10.0
) -> np.ndarray:
    """
    Add deterministic variability to a timeseries (no randomness).
    
    Uses multi-frequency sinusoidal patterns to create realistic
    variability that is identical on every run.
    
    Parameters
    ----------
    data : np.ndarray
        Original timeseries data
    variability_pct : float
        Percentage variability to add (e.g., 10 = ±10%)
        
    Returns
    -------
    np.ndarray
        Data with added deterministic variability
    """
    n = len(data)
    t = np.arange(n)
    
    # Deterministic variability using multiple sinusoidal patterns
    # These patterns sum to ~10% max variation
    variability = (
        0.03 * np.sin(2 * np.pi * t / 6)           # 6-hour pattern
        + 0.03 * np.sin(2 * np.pi * t / 12 + 1.5)  # 12-hour pattern  
        + 0.02 * np.sin(2 * np.pi * t / 48 + 0.7)  # 2-day pattern
        + 0.02 * np.sin(2 * np.pi * t / 168 + 2.3) # Weekly pattern
    )
    
    # Scale to desired variability percentage
    scale_factor = variability_pct / 10.0  # Base patterns give ~10%
    variability = variability * scale_factor
    
    # Apply variability (multiplicative)
    data_varied = data * (1 + variability)
    
    # Ensure non-negative
    return np.maximum(data_varied, 0.0)


def load_power_data(
    data_path: Optional[str] = None,
    config: Optional[AlkalineConfig] = None,
    add_variability: bool = True,
    variability_pct: float = 10.0
) -> np.ndarray:
    """
    Load renewable power timeseries from real .mat file.
    
    Uses actual wind + solar data from combined_wind_pv_DATA.mat.
    Adds deterministic variability (default 10%) for realistic variation
    without randomness (reproducible across runs).
    
    Parameters
    ----------
    data_path : str, optional
        Path to .mat file. If None, uses default combined_wind_pv_DATA.mat
    config : AlkalineConfig, optional
        Configuration for simulation years (determines array length)
    add_variability : bool
        Whether to add deterministic variability (default True)
    variability_pct : float
        Percentage variability to add (default 10%)
        
    Returns
    -------
    np.ndarray
        Power timeseries in Watts [W], shape (n_hours,)
    """
    if data_path is None:
        # Default path - use real wind+solar data
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / "combined_wind_pv_DATA.mat"
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Power data not found: {data_path}")
    
    # Load .mat file
    mat_data = loadmat(str(data_path))
    
    # Load wind and solar separately and combine
    if 'P_wind_selected' in mat_data and 'P_PV' in mat_data:
        P_wind = mat_data['P_wind_selected'].flatten()
        P_PV = mat_data['P_PV'].flatten()
        power = P_wind + P_PV  # Combined wind + solar
        print(f"  Loaded real power data: {len(power)} hours")
        print(f"    Wind: mean={np.mean(P_wind)/1e6:.2f} MW, max={np.max(P_wind)/1e6:.2f} MW")
        print(f"    Solar: mean={np.mean(P_PV)/1e6:.2f} MW, max={np.max(P_PV)/1e6:.2f} MW")
        print(f"    Combined: mean={np.mean(power)/1e6:.2f} MW, max={np.max(power)/1e6:.2f} MW")
    else:
        # Try other key names
        for key in ['combined_power_W', 'P_total_W', 'power_W', 'power']:
            if key in mat_data:
                power = mat_data[key].flatten()
                break
        else:
            # Get first non-metadata key
            data_keys = [k for k in mat_data.keys() if not k.startswith('_')]
            if data_keys:
                power = mat_data[data_keys[0]].flatten()
            else:
                raise KeyError(f"No power data found in {data_path}")
    
    # Ensure correct length for multi-year simulation
    if config is not None:
        n_hours_needed = int(config.simulation_years * 8760)
        if len(power) < n_hours_needed:
            # Tile the data to reach required length
            n_tiles = int(np.ceil(n_hours_needed / len(power)))
            power = np.tile(power, n_tiles)[:n_hours_needed]
            print(f"    Tiled to {n_hours_needed} hours ({config.simulation_years} years)")
        else:
            power = power[:n_hours_needed]
    
    # Add deterministic variability (no randomness)
    if add_variability and variability_pct > 0:
        power = add_deterministic_variability(power, variability_pct)
        print(f"    Added {variability_pct}% deterministic variability")
    
    return power.astype(np.float64)


def load_demand_data(
    data_path: Optional[str] = None,
    config: Optional[AlkalineConfig] = None,
    add_variability: bool = True,
    variability_pct: float = 5.0
) -> np.ndarray:
    """
    Load hydrogen demand timeseries from CSV file.
    
    Uses actual demand data from Company_2_hourly_gas_demand.csv.
    Converts from kWh to kg H2 using LHV (33.33 kWh/kg).
    Adds deterministic variability (default 5%) for realistic variation.
    
    Parameters
    ----------
    data_path : str, optional
        Path to CSV file. If None, uses default Company_2_hourly_gas_demand.csv
    config : AlkalineConfig, optional
        Configuration for simulation years (determines array length)
    add_variability : bool
        Whether to add deterministic variability (default True)
    variability_pct : float
        Percentage variability to add (default 5%)
        
    Returns
    -------
    np.ndarray
        Demand timeseries in kg H2/h, shape (n_hours,)
    """
    import pandas as pd
    
    if data_path is None:
        # Default path - use real demand data
        base_dir = Path(__file__).parent.parent
        data_path = base_dir / "data" / "Company_2_hourly_gas_demand.csv"
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Demand data not found: {data_path}")
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Get demand column (in kWh)
    if 'demand_kwh' in df.columns:
        demand_kWh = df['demand_kwh'].values
    elif 'demand_kWh' in df.columns:
        demand_kWh = df['demand_kWh'].values
    else:
        # Try to find demand-like column
        demand_cols = [c for c in df.columns if 'demand' in c.lower()]
        if demand_cols:
            demand_kWh = df[demand_cols[0]].values
        else:
            demand_kWh = df.iloc[:, 2].values  # Assume third column
    
    # Convert kWh to kg H2 using LHV (33.33 kWh/kg)
    LHV_H2 = 33.33  # kWh/kg
    demand_kg_h = demand_kWh / LHV_H2
    
    print(f"  Loaded real demand data: {len(demand_kg_h)} hours")
    print(f"    Mean demand: {np.mean(demand_kg_h):.2f} kg/h")
    print(f"    Max demand: {np.max(demand_kg_h):.2f} kg/h")
    print(f"    Total annual: {np.sum(demand_kg_h)/1000:.1f} tonnes")
    
    # Ensure correct length for multi-year simulation
    if config is not None:
        n_hours_needed = int(config.simulation_years * 8760)
        if len(demand_kg_h) < n_hours_needed:
            # Tile the data to reach required length
            n_tiles = int(np.ceil(n_hours_needed / len(demand_kg_h)))
            demand_kg_h = np.tile(demand_kg_h, n_tiles)[:n_hours_needed]
            print(f"    Tiled to {n_hours_needed} hours ({config.simulation_years} years)")
        else:
            demand_kg_h = demand_kg_h[:n_hours_needed]
    
    # Add deterministic variability (no randomness)
    if add_variability and variability_pct > 0:
        demand_kg_h = add_deterministic_variability(demand_kg_h, variability_pct)
        print(f"    Added {variability_pct}% deterministic variability")
    
    return demand_kg_h.astype(np.float64)


def simulate(
    power_input_W: np.ndarray,
    config: Optional[AlkalineConfig] = None,
    verbose: bool = True
) -> SimulationResults:
    """
    Run hourly simulation of Alkaline electrolyser over multi-year period.
    
    This is the main simulation function integrating:
    - Electrochemistry (voltage, efficiency, H2 production)
    - Degradation (linear + cycling penalty)
    - Stack replacement when degradation limit reached
    - Optional hydrogen storage and demand tracking
    
    Parameters
    ----------
    power_input_W : np.ndarray
        Hourly available power timeseries [W], shape (n_hours,)
    config : AlkalineConfig, optional
        Configuration object. If None, uses defaults.
    verbose : bool
        Print progress updates
        
    Returns
    -------
    SimulationResults
        Container with all hourly and summary results
        
    Example
    -------
    >>> config = get_alkaline_config(P_nom_MW=20.0)
    >>> power = np.random.uniform(5e6, 25e6, 8760*5)  # 5 years
    >>> results = simulate(power, config)
    >>> print(f"Total H2: {results.total_h2_production_kg/1000:.1f} tonnes")
    """
    if config is None:
        config = get_alkaline_config()
    
    n_hours = len(power_input_W)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Alkaline Electrolyser Simulation")
        print(f"{'='*60}")
        print(f"  System size: {config.P_nom_MW:.1f} MW")
        print(f"  Duration: {n_hours/8760:.1f} years ({n_hours:,} hours)")
        print(f"  Degradation rate: {config.deg_rate_uV_h} μV/h")
        print(f"  Stack lifetime: {config.stack_lifetime_hours:,}h")
    
    # Initialize output arrays
    power_consumed_W = np.zeros(n_hours)
    h2_production_kg = np.zeros(n_hours)
    current_density = np.zeros(n_hours)
    cell_voltage = np.zeros(n_hours)
    faraday_eff = np.zeros(n_hours)
    system_eff = np.zeros(n_hours)
    sec_array = np.zeros(n_hours)
    temperature_K = np.zeros(n_hours)
    voltage_deg = np.zeros(n_hours)
    cap_factor = np.ones(n_hours)
    is_operating = np.zeros(n_hours, dtype=bool)
    n_cycles_cum = np.zeros(n_hours, dtype=int)
    
    # Storage arrays
    storage_level = np.zeros(n_hours)
    h2_to_demand = np.zeros(n_hours)
    unmet_demand = np.zeros(n_hours)
    
    # Initialize degradation state
    deg_state = DegradationState()
    
    # Get nominal voltage for degradation reference
    V_cell_nominal = compute_cell_voltage(config.j_nom, config.T_op_K, config)
    
    # Calculate power limits
    P_min_W = config.P_nom_W * config.min_load_fraction
    P_max_W = config.P_nom_W
    
    # Initialize storage
    if config.storage_capacity_kg > 0:
        storage_level[0] = config.storage_capacity_kg * config.storage_initial_fraction
    
    # Previous state for cycling detection
    was_operating = False
    prev_load_fraction = 0.0
    
    # Progress reporting
    report_interval = n_hours // 10 if n_hours > 1000 else max(n_hours // 5, 1)
    
    # Ramp rate constraint
    ramp_rate_W = config.max_ramp_rate_per_h * config.P_nom_W  # W per hour
    prev_power_W = 0.0  # Track previous hour's power for ramping
    
    # Main simulation loop
    for h in range(n_hours):
        # Get hour of year for temperature
        hour_of_year = h % 8760
        T_K = compute_operating_temperature(hour_of_year, config)
        temperature_K[h] = T_K
        
        # Get available power
        P_available = power_input_W[h]
        
        # Apply ramp rate constraint (only when operating)
        # Note: Ramp rate applies to power changes while running
        # Startup/shutdown can be faster but we model as limited by ramp rate from 0
        if was_operating and P_available >= P_min_W:
            # Already operating - apply ramp constraint
            P_ramped = np.clip(P_available, 
                               prev_power_W - ramp_rate_W, 
                               prev_power_W + ramp_rate_W)
        elif P_available >= P_min_W:
            # Starting up - can start directly at minimum load or higher (up to ramp limit from 0)
            # Allow faster startup: can go directly to min_load or ramp from 0
            P_ramped = min(P_available, max(P_min_W, prev_power_W + ramp_rate_W))
        else:
            # Not enough power to operate
            P_ramped = 0.0
        
        # Determine if operating
        if P_ramped >= P_min_W:
            # Operating - clamp to valid range
            P_actual = min(P_ramped, P_max_W)
            is_operating[h] = True
            load_fraction = P_actual / config.P_nom_W
            
            # Calculate current density (with degradation offset)
            j = power_to_current_density(
                P_actual / config.n_stacks,  # Power per stack
                config,
                T_K,
                degradation_V=deg_state.voltage_degradation_V
            )
            
            # Electrochemistry
            V_cell = compute_cell_voltage(j, T_K, config, deg_state.voltage_degradation_V)
            eta_F = compute_faraday_efficiency(j, T_K, config)
            eta_stack = compute_stack_efficiency(V_cell, eta_F, basis='LHV')
            
            # Apply partial load efficiency penalty to stack efficiency
            pl_eff_factor = compute_partial_load_efficiency_factor(load_fraction, config)
            eta_stack_adjusted = eta_stack * pl_eff_factor
            
            # System efficiency includes BoP losses
            eta_sys = compute_system_efficiency(eta_stack_adjusted, config)
            
            # Apply component replacement performance boost (if enabled)
            if getattr(config, 'enable_replacement_performance_boost', False):
                eta_sys_boosted = eta_sys * deg_state.efficiency_multiplier
            else:
                eta_sys_boosted = eta_sys
            
            # H2 production (using system efficiency with partial load and replacement boost)
            I_total = j * config.cell_area_cm2 * config.n_stacks  # Total current
            m_H2_kg_s = compute_hydrogen_production_rate(I_total * config.n_cells, eta_F)
            
            # Apply efficiency boost to production (higher efficiency = more H2 from same power)
            # SEC = HHV/eta, so production = Power * eta / HHV
            if getattr(config, 'enable_replacement_performance_boost', False):
                production_boost_factor = deg_state.efficiency_multiplier
                m_H2_kg_h = m_H2_kg_s * 3600 * production_boost_factor
            else:
                m_H2_kg_h = m_H2_kg_s * 3600
            
            # Store results (use boosted efficiency for SEC calculation)
            power_consumed_W[h] = P_actual
            h2_production_kg[h] = m_H2_kg_h
            current_density[h] = j
            cell_voltage[h] = V_cell
            faraday_eff[h] = eta_F
            system_eff[h] = eta_sys_boosted  # Store boosted efficiency
            sec_array[h] = compute_specific_energy_consumption(eta_sys_boosted, basis='LHV')
            
            # Update degradation
            delta_V, is_cycle, _ = compute_degradation_rate(
                True, was_operating, load_fraction, prev_load_fraction, config
            )
            deg_state.voltage_degradation_V += delta_V
            deg_state.cumulative_hours += config.dt_hours
            if is_cycle:
                deg_state.cumulative_cycles += 1
            
            prev_load_fraction = load_fraction
            prev_power_W = P_actual  # Update for next iteration
            
        else:
            # Not operating - below minimum load
            is_operating[h] = False
            power_consumed_W[h] = 0.0
            h2_production_kg[h] = 0.0
            current_density[h] = 0.0
            cell_voltage[h] = 0.0
            faraday_eff[h] = 0.0
            system_eff[h] = 0.0
            sec_array[h] = 0.0
            prev_load_fraction = 0.0
            prev_power_W = 0.0  # Reset power tracking when off
        
        # Update cumulative state
        voltage_deg[h] = deg_state.voltage_degradation_V
        cap_factor[h] = compute_capacity_fade(deg_state.voltage_degradation_V, V_cell_nominal)
        n_cycles_cum[h] = deg_state.cumulative_cycles
        
        # Check for stack replacement
        needs_replacement, reason = check_stack_replacement(deg_state, V_cell_nominal, config)
        if needs_replacement:
            if verbose and deg_state.stack_replacements == 0:
                print(f"  Stack replacement at hour {h:,}: {reason}")
            deg_state = reset_degradation_state(deg_state, h)
        
        # Handle storage and demand (if enabled)
        if config.storage_capacity_kg > 0 or config.demand_kg_h > 0:
            h2_available = h2_production_kg[h]
            
            if h > 0:
                storage_level[h] = storage_level[h-1]
            
            # Add production to storage
            if config.storage_capacity_kg > 0:
                storage_space = config.storage_capacity_kg - storage_level[h]
                h2_to_storage = min(h2_available, storage_space)
                storage_level[h] += h2_to_storage * config.storage_efficiency
                h2_available -= h2_to_storage
            
            # Meet demand
            if config.demand_kg_h > 0:
                demand = config.demand_kg_h
                
                # First use direct production
                from_production = min(h2_available, demand)
                h2_to_demand[h] += from_production
                demand -= from_production
                
                # Then use storage
                if demand > 0 and config.storage_capacity_kg > 0:
                    from_storage = min(storage_level[h], demand)
                    storage_level[h] -= from_storage
                    h2_to_demand[h] += from_storage
                    demand -= from_storage
                
                unmet_demand[h] = demand
        
        was_operating = is_operating[h]
        
        # Component replacement performance tracking (check yearly)
        if getattr(config, 'enable_replacement_performance_boost', False):
            current_year = int(h / 8760) + 1  # Year 1, 2, 3, ...
            
            # Check electrolyte replacement
            electrolyte_years = getattr(config, 'electrolyte_replacement_years', 4.0)
            if electrolyte_years > 0 and current_year % int(electrolyte_years) == 0:
                if deg_state.last_electrolyte_replacement_year != current_year:
                    gain = getattr(config, 'electrolyte_replacement_efficiency_gain', 0.04)
                    deg_state.efficiency_multiplier *= (1 + gain)
                    deg_state.last_electrolyte_replacement_year = current_year
                    if verbose and h % 8760 < 24:  # Print once near start of year
                        print(f"  Year {current_year}: Electrolyte replacement → "
                              f"+{gain*100:.1f}% efficiency (total: {(deg_state.efficiency_multiplier-1)*100:.1f}%)")
            
            # Check catalyst replacement
            catalyst_years = getattr(config, 'catalyst_replacement_years', 12.0)
            if catalyst_years > 0 and current_year % int(catalyst_years) == 0:
                if deg_state.last_catalyst_replacement_year != current_year:
                    gain = getattr(config, 'catalyst_replacement_efficiency_gain', 0.05)
                    deg_state.efficiency_multiplier *= (1 + gain)
                    deg_state.last_catalyst_replacement_year = current_year
                    if verbose and h % 8760 < 24:
                        print(f"  Year {current_year}: Catalyst replacement → "
                              f"+{gain*100:.1f}% efficiency (total: {(deg_state.efficiency_multiplier-1)*100:.1f}%)")
            
            # Check diaphragm replacement
            diaphragm_years = getattr(config, 'diaphragm_replacement_years', 10.0)
            if diaphragm_years > 0 and current_year % int(diaphragm_years) == 0:
                if deg_state.last_diaphragm_replacement_year != current_year:
                    gain = getattr(config, 'diaphragm_replacement_efficiency_gain', 0.03)
                    deg_state.efficiency_multiplier *= (1 + gain)
                    deg_state.last_diaphragm_replacement_year = current_year
                    if verbose and h % 8760 < 24:
                        print(f"  Year {current_year}: Diaphragm replacement → "
                              f"+{gain*100:.1f}% efficiency (total: {(deg_state.efficiency_multiplier-1)*100:.1f}%)")
            
            # Check mechanical replacement
            mechanical_years = getattr(config, 'mechanical_replacement_years', 6.0)
            if mechanical_years > 0 and current_year % int(mechanical_years) == 0:
                if deg_state.last_mechanical_replacement_year != current_year:
                    gain = getattr(config, 'mechanical_replacement_efficiency_gain', 0.01)
                    deg_state.efficiency_multiplier *= (1 + gain)
                    deg_state.last_mechanical_replacement_year = current_year
                    if verbose and h % 8760 < 24:
                        print(f"  Year {current_year}: Mechanical replacement → "
                              f"+{gain*100:.1f}% efficiency (total: {(deg_state.efficiency_multiplier-1)*100:.1f}%)")
        
        # Progress report
        if verbose and h > 0 and h % report_interval == 0:
            progress = h / n_hours * 100
            print(f"  Progress: {progress:.0f}% ({h:,}h) - "
                  f"H2: {np.sum(h2_production_kg[:h])/1000:.1f}t, "
                  f"Cycles: {deg_state.cumulative_cycles}, "
                  f"V_deg: {deg_state.voltage_degradation_V*1000:.1f}mV")
    
    # Create results object
    results = SimulationResults(
        hours=np.arange(n_hours),
        power_available_W=power_input_W.copy(),
        power_consumed_W=power_consumed_W,
        h2_production_kg=h2_production_kg,
        current_density_A_cm2=current_density,
        cell_voltage_V=cell_voltage,
        faraday_efficiency=faraday_eff,
        system_efficiency=system_eff,
        sec_kWh_kg=sec_array,
        temperature_K=temperature_K,
        voltage_degradation_V=voltage_deg,
        capacity_factor=cap_factor,
        is_operating=is_operating,
        n_cycles_cumulative=n_cycles_cum,
        storage_level_kg=storage_level,
        h2_to_demand_kg=h2_to_demand,
        unmet_demand_kg=unmet_demand,
        stack_replacements=deg_state.stack_replacements,
        replacement_hours=deg_state.replacement_hours
    )
    
    # Compute summary metrics
    results.total_h2_production_kg = np.sum(h2_production_kg)
    results.total_energy_consumed_kWh = np.sum(power_consumed_W) / 1000  # W to kWh
    results.total_operating_hours = np.sum(is_operating) * config.dt_hours
    results.total_cycles = deg_state.cumulative_cycles
    
    # Average SEC (only when operating)
    operating_mask = is_operating & (sec_array > 0)
    if np.any(operating_mask):
        results.average_sec_kWh_kg = np.mean(sec_array[operating_mask])
    
    # Capacity factor
    results.capacity_factor_avg = results.total_operating_hours / n_hours
    
    # Availability (fraction of time able to operate when power available)
    power_sufficient = power_input_W >= P_min_W
    results.availability = np.sum(is_operating) / max(np.sum(power_sufficient), 1)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulation Complete")
        print(f"{'='*60}")
        print(f"  Total H2 production: {results.total_h2_production_kg/1000:.2f} tonnes")
        print(f"  Total energy consumed: {results.total_energy_consumed_kWh/1e6:.2f} GWh")
        print(f"  Average SEC: {results.average_sec_kWh_kg:.1f} kWh/kg")
        print(f"  Operating hours: {results.total_operating_hours:,.0f}h")
        print(f"  Capacity factor: {results.capacity_factor_avg:.1%}")
        print(f"  Start/stop cycles: {results.total_cycles}")
        print(f"  Stack replacements: {results.stack_replacements}")
        print(f"  Final voltage degradation: {voltage_deg[-1]*1000:.1f} mV")
    
    return results


def quick_simulate(
    P_nom_MW: float = 20.0,
    simulation_years: int = 5,
    power_profile: str = 'variable',
    verbose: bool = True
) -> SimulationResults:
    """
    Quick simulation with synthetic power profile for testing.
    
    Parameters
    ----------
    P_nom_MW : float
        Nominal electrolyser size [MW]
    simulation_years : int
        Simulation duration [years]
    power_profile : str
        Type of power profile:
        - 'constant': Constant at 80% of nominal
        - 'variable': Variable renewable (30-100% range)
        - 'intermittent': Highly intermittent (0-100%, 50% capacity factor)
        - 'real': Use real wind+solar data from combined_wind_pv_DATA.mat
    verbose : bool
        Print progress
        
    Returns
    -------
    SimulationResults
        Simulation results
    """
    config = get_alkaline_config(
        P_nom_MW=P_nom_MW,
        simulation_years=simulation_years
    )
    
    n_hours = simulation_years * 8760
    P_nom_W = P_nom_MW * 1e6
    
    if power_profile == 'constant':
        power = np.ones(n_hours) * 0.8 * P_nom_W
        
    elif power_profile == 'variable':
        # Simulate variable renewable: sine wave + noise
        t = np.arange(n_hours)
        daily_pattern = 0.5 * (1 + np.sin(2 * np.pi * t / 24 - np.pi/2))  # Peak at noon
        seasonal_pattern = 0.8 + 0.2 * np.sin(2 * np.pi * t / 8760 - np.pi/2)  # Summer peak
        noise = 0.1 * np.random.randn(n_hours)
        
        power_fraction = 0.3 + 0.5 * daily_pattern * seasonal_pattern + noise
        power_fraction = np.clip(power_fraction, 0.0, 1.0)
        power = power_fraction * P_nom_W
        
    elif power_profile == 'intermittent':
        # Highly intermittent: 50% of time operating
        power = np.random.uniform(0.0, 1.0, n_hours)
        power = np.where(power > 0.5, power * P_nom_W, 0.0)
        
    elif power_profile == 'real':
        # Use real wind+solar data with 10% variability
        power = load_power_data(config=config, add_variability=True, variability_pct=10.0)
        
    else:
        raise ValueError(f"Unknown power profile: {power_profile}")
    
    return simulate(power, config, verbose=verbose)


def run_with_real_data(
    P_nom_MW: float = 20.0,
    simulation_years: int = 5,
    power_variability_pct: float = 10.0,
    demand_variability_pct: float = 5.0,
    electricity_price: float = 0.07,
    h2_price: float = 6.0,
    include_demand: bool = True,
    storage_capacity_kg: float = 5000.0,
    verbose: bool = True,
    save_results: bool = True,
    output_folder: str = 'results/alkaline_real_data'
) -> Tuple[SimulationResults, 'EconomicResults']:
    """
    Run full simulation using real wind+solar power and demand data.
    
    Uses:
    - Power: combined_wind_pv_DATA.mat (wind + solar, 1 year tiled)
    - Demand: Company_2_hourly_gas_demand.csv (converted kWh → kg H2)
    
    Adds deterministic variability (no randomness between runs):
    - Power: 10% variability (default)
    - Demand: 5% variability (default)
    
    Compression to 350 bar is included in LCOH (3.5 kWh/kg).
    
    Parameters
    ----------
    P_nom_MW : float
        Nominal electrolyser capacity [MW]
    simulation_years : int
        Simulation duration [years]
    power_variability_pct : float
        Percentage variability to add to power data (default 10%)
    demand_variability_pct : float
        Percentage variability to add to demand data (default 5%)
    electricity_price : float
        Electricity price [EUR/kWh] (default 0.07, same as PEM)
    h2_price : float
        Hydrogen selling price [EUR/kg]
    include_demand : bool
        Whether to include demand tracking (default True)
    storage_capacity_kg : float
        Hydrogen storage capacity [kg] (default 5000)
    verbose : bool
        Print progress messages
    save_results : bool
        Whether to save results to CSV files
    output_folder : str
        Folder for saving results
        
    Returns
    -------
    Tuple[SimulationResults, EconomicResults]
        Simulation and economic results
    """
    print("=" * 70)
    print("ALKALINE ELECTROLYSER SIMULATION - REAL DATA")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Electrolyser size: {P_nom_MW} MW")
    print(f"  Simulation period: {simulation_years} years")
    print(f"  Power variability: ±{power_variability_pct}%")
    print(f"  Demand variability: ±{demand_variability_pct}%")
    print(f"  Storage capacity: {storage_capacity_kg} kg")
    print(f"  Electricity price: €{electricity_price}/kWh")
    print(f"  H2 selling price: €{h2_price}/kg")
    
    # Create configuration
    config = get_alkaline_config(
        P_nom_MW=P_nom_MW,
        simulation_years=simulation_years,
        electricity_price_eur_kWh=electricity_price,
        h2_selling_price_eur_kg=h2_price,
        storage_capacity_kg=storage_capacity_kg if include_demand else 0.0
    )
    
    # Load real power data with variability
    print(f"\n[Loading Power Data]")
    power_W = load_power_data(
        config=config, 
        add_variability=True, 
        variability_pct=power_variability_pct
    )
    
    # Load real demand data with variability (if enabled)
    if include_demand:
        print(f"\n[Loading Demand Data]")
        demand_kg_h = load_demand_data(
            config=config,
            add_variability=True,
            variability_pct=demand_variability_pct
        )
        # Set average demand in config
        config.demand_kg_h = float(np.mean(demand_kg_h))
        print(f"    Using time-varying demand (avg: {config.demand_kg_h:.2f} kg/h)")
    
    # Run simulation
    print(f"\n[Running Simulation]")
    results = simulate(power_W, config, verbose=verbose)
    
    # Run economics
    print(f"\n[Computing Economics]")
    econ = compute_lcoh(results, config, verbose=verbose)
    
    # Summary
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nProduction:")
    print(f"  Total H2 produced: {results.total_h2_production_kg/1000:.1f} tonnes")
    print(f"  Operating hours: {results.total_operating_hours:,.0f} h")
    print(f"  Capacity factor: {results.capacity_factor_avg*100:.1f}%")
    print(f"  Average SEC: {results.average_sec_kWh_kg:.2f} kWh/kg")
    
    # Calculate derived metrics from arrays
    final_voltage_degradation = results.voltage_degradation_V[-1] if len(results.voltage_degradation_V) > 0 else 0.0
    total_cycles = results.n_cycles_cumulative[-1] if len(results.n_cycles_cumulative) > 0 else 0
    
    print(f"\nDegradation:")
    print(f"  Final voltage increase: {final_voltage_degradation*1000:.1f} mV")
    print(f"  Stack replacements: {results.stack_replacements}")
    print(f"  Number of cycles: {total_cycles}")
    
    print(f"\nEconomics:")
    print(f"  LCOH: €{econ.lcoh_total:.2f}/kg")
    print(f"    - CAPEX: €{econ.lcoh_capex:.2f}/kg ({econ.lcoh_capex/econ.lcoh_total*100:.0f}%)")
    print(f"    - Electricity: €{econ.lcoh_electricity:.2f}/kg ({econ.lcoh_electricity/econ.lcoh_total*100:.0f}%)")
    print(f"    - Compression (350 bar): €{econ.lcoh_compression:.2f}/kg ({econ.lcoh_compression/econ.lcoh_total*100:.0f}%)")
    print(f"    - O&M: €{econ.lcoh_opex_fixed:.2f}/kg ({econ.lcoh_opex_fixed/econ.lcoh_total*100:.0f}%)")
    print(f"  NPV: €{econ.npv/1e6:.2f}M")
    print(f"  IRR: {econ.irr:.1f}%")
    print(f"  Payback: {econ.payback_years:.1f} years")
    
    # Save results
    if save_results:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Summary CSV
        import pandas as pd
        summary_data = {
            'Parameter': [
                'Electrolyser Size [MW]',
                'Simulation Years',
                'Total H2 Production [tonnes]',
                'Operating Hours [h]',
                'Capacity Factor [%]',
                'Average SEC [kWh/kg]',
                'Compression Energy [kWh/kg]',
                'Final Voltage Degradation [mV]',
                'Stack Replacements',
                'Number of Cycles',
                'Electricity Price [EUR/kWh]',
                'LCOH [EUR/kg]',
                'LCOH CAPEX [EUR/kg]',
                'LCOH Electricity [EUR/kg]',
                'LCOH Compression [EUR/kg]',
                'LCOH O&M [EUR/kg]',
                'NPV [EUR M]',
                'IRR [%]',
                'Payback [years]',
            ],
            'Value': [
                P_nom_MW,
                simulation_years,
                results.total_h2_production_kg / 1000,
                results.total_operating_hours,
                results.capacity_factor_avg * 100,
                results.average_sec_kWh_kg,
                config.compression_energy_kWh_kg,
                final_voltage_degradation * 1000,
                results.stack_replacements,
                total_cycles,
                electricity_price,
                econ.lcoh_total,
                econ.lcoh_capex,
                econ.lcoh_electricity,
                econ.lcoh_compression,
                econ.lcoh_opex_fixed,
                econ.npv / 1e6,
                econ.irr,
                econ.payback_years,
            ]
        }
        pd.DataFrame(summary_data).to_csv(output_path / 'summary_alkaline.csv', index=False)
        
        # Timeseries CSV (sampled for size)
        n_hours = len(results.h2_production_kg)
        # Save full 5-year timeseries
        ts_data = {
            'hour': np.arange(n_hours),
            'power_available_W': results.power_available_W,
            'power_consumed_W': results.power_consumed_W,
            'h2_production_kg': results.h2_production_kg,
            'cell_voltage_V': results.cell_voltage_V,
            'current_density_A_cm2': results.current_density_A_cm2,
            'sec_kWh_kg': results.sec_kWh_kg,
            'faraday_efficiency_pct': results.faraday_efficiency * 100,
            'is_operating': results.is_operating.astype(int),
            'voltage_degradation_V': results.voltage_degradation_V,
        }
        if hasattr(results, 'storage_level_kg'):
            ts_data['storage_level_kg'] = results.storage_level_kg
        
        pd.DataFrame(ts_data).to_csv(output_path / f'timeseries_full_{simulation_years}years_alkaline.csv', index=False)
        
        print(f"\n[Results saved to {output_path}/]")
    
    return results, econ


# =============================================================================
# PART 4: ECONOMICS AND PROFITABILITY
# =============================================================================

@dataclass
class EconomicResults:
    """
    Container for economic analysis results.
    
    Includes LCOH breakdown, profitability metrics, and cash flow analysis.
    """
    # LCOH Components [EUR/kg]
    lcoh_total: float = 0.0
    lcoh_capex: float = 0.0
    lcoh_opex_fixed: float = 0.0
    lcoh_electricity: float = 0.0
    lcoh_compression: float = 0.0        # Compression to 350 bar
    lcoh_water: float = 0.0
    lcoh_stack_replacement: float = 0.0
    lcoh_curtailment: float = 0.0        # Curtailment penalty
    
    # Profitability Metrics
    npv: float = 0.0                     # Net Present Value [EUR]
    irr: float = 0.0                     # Internal Rate of Return [%]
    payback_years: float = 0.0           # Simple payback period [years]
    discounted_payback_years: float = 0.0  # Discounted payback [years]
    profitability_index: float = 0.0     # PI = NPV / Initial Investment + 1
    
    # Annual Cash Flows
    annual_revenue: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_costs: np.ndarray = field(default_factory=lambda: np.array([]))
    annual_cash_flow: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_cash_flow: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Total Lifetime Values
    total_capex: float = 0.0
    total_revenue: float = 0.0
    total_costs: float = 0.0
    total_h2_production: float = 0.0
    
    # Risk/Sensitivity
    lcoh_sensitivity: Dict[str, float] = field(default_factory=dict)


def compute_capex(config: AlkalineConfig) -> Dict[str, float]:
    """
    Compute total CAPEX breakdown.
    
    Parameters
    ----------
    config : AlkalineConfig
        Configuration with economic parameters
        
    Returns
    -------
    Dict[str, float]
        CAPEX breakdown by component [EUR]
        
    References
    ----------
    [1] IRENA (2023) - "Green Hydrogen Cost Reduction: Scaling up Electrolysers"
    [2] BloombergNEF (2023) - "Hydrogen Economy Outlook"
    """
    P_kW = config.P_nom_MW * 1000
    
    # Electrolyser CAPEX (with detailed breakdown)
    capex_stack = config.capex_stack_eur_kW * P_kW
    capex_bop = config.capex_bop_eur_kW * P_kW
    capex_installation = getattr(config, 'capex_installation_eur_kW', 120.0) * P_kW
    capex_engineering = getattr(config, 'capex_engineering_eur_kW', 80.0) * P_kW
    capex_electrolyser = capex_stack + capex_bop + capex_installation + capex_engineering
    
    # System boundary items (harmonized with PEM for fair comparison)
    capex_water_treatment = getattr(config, 'capex_water_treatment_eur_kW', 30.0) * P_kW
    capex_site_preparation = getattr(config, 'capex_site_preparation_eur_kW', 40.0) * P_kW
    capex_grid_connection = getattr(config, 'capex_grid_connection_eur_kW', 35.0) * P_kW
    capex_system_boundary = capex_water_treatment + capex_site_preparation + capex_grid_connection
    
    # Storage CAPEX (Type IV tanks @ 350 bar)
    capex_storage = config.storage_capacity_kg * config.storage_capex_eur_kg
    
    # Compressor CAPEX (sized for storage capacity)
    capex_compressor = config.storage_capacity_kg * config.compressor_capex_eur_kg
    
    # Subtotal before contingency
    capex_subtotal = capex_electrolyser + capex_system_boundary + capex_storage + capex_compressor
    
    # Contingency (10% default)
    contingency_frac = getattr(config, 'contingency_fraction', 0.10)
    capex_contingency = capex_subtotal * contingency_frac
    
    capex = {
        'stack': capex_stack,
        'bop': capex_bop,
        'installation': capex_installation,
        'engineering': capex_engineering,
        'electrolyser': capex_electrolyser,
        'water_treatment': capex_water_treatment,
        'site_preparation': capex_site_preparation,
        'grid_connection': capex_grid_connection,
        'system_boundary': capex_system_boundary,
        'storage': capex_storage,
        'compressor': capex_compressor,
        'contingency': capex_contingency,
        'total': capex_subtotal + capex_contingency
    }
    
    return capex


def compute_annual_opex(
    config: AlkalineConfig,
    annual_energy_kWh: float,
    annual_h2_kg: float,
    stack_replacements_this_year: int = 0
) -> Dict[str, float]:
    """
    Compute annual OPEX breakdown.
    
    Parameters
    ----------
    config : AlkalineConfig
        Configuration object
    annual_energy_kWh : float
        Annual electricity consumption [kWh]
    annual_h2_kg : float
        Annual H2 production [kg]
    stack_replacements_this_year : int
        Number of stack replacements in this year
        
    Returns
    -------
    Dict[str, float]
        OPEX breakdown by category [EUR/year]
    """
    P_kW = config.P_nom_MW * 1000
    
    # Fixed O&M for electrolyser
    fixed_om = config.opex_fixed_eur_kW_yr * P_kW
    
    # Land lease (harmonized with PEM)
    land_lease = getattr(config, 'land_lease_eur_kW_yr', 5.0) * P_kW
    
    # Electricity cost (electrolysis only)
    electricity = annual_energy_kWh * config.electricity_price_eur_kWh
    
    # Compression cost (3.5 kWh/kg to 350 bar)
    compression_energy = annual_h2_kg * config.compression_energy_kWh_kg
    compression = compression_energy * config.electricity_price_eur_kWh
    
    # Compressor maintenance
    compressor_capex = config.storage_capacity_kg * config.compressor_capex_eur_kg
    compressor_maint = compressor_capex * config.compressor_maint_fraction
    
    # Water cost
    water_m3 = annual_h2_kg * config.water_consumption_L_kg / 1000
    water = water_m3 * config.water_price_eur_m3
    
    # Stack replacement cost
    stack_cost = 0.0
    if stack_replacements_this_year > 0:
        stack_cost = (stack_replacements_this_year 
                      * config.capex_stack_eur_kW * P_kW 
                      * config.stack_replacement_cost_fraction)
    
    opex = {
        'fixed_om': fixed_om,
        'land_lease': land_lease,
        'electricity': electricity,
        'compression': compression,
        'compressor_maint': compressor_maint,
        'water': water,
        'stack_replacement': stack_cost,
        'total': fixed_om + land_lease + electricity + compression + compressor_maint + water + stack_cost
    }
    
    return opex


def compute_lcoh(
    results: SimulationResults,
    config: AlkalineConfig,
    verbose: bool = False
) -> EconomicResults:
    """
    Compute Levelized Cost of Hydrogen (LCOH) and profitability metrics.
    
    LCOH = (CAPEX_annualized + OPEX) / Annual_H2_production
    
    Uses CRF (Capital Recovery Factor) for CAPEX annualization.
    
    Parameters
    ----------
    results : SimulationResults
        Simulation results with H2 production and energy consumption
    config : AlkalineConfig
        Configuration with economic parameters
    verbose : bool
        Print detailed breakdown
        
    Returns
    -------
    EconomicResults
        Complete economic analysis
        
    Notes
    -----
    LCOH typical ranges for Alkaline (2024):
    - Best case (cheap electricity): 2-3 EUR/kg
    - Average: 4-6 EUR/kg
    - High cost electricity: 8-12 EUR/kg
    """
    econ = EconomicResults()
    
    # Simulation duration
    n_hours = len(results.hours)
    n_years = n_hours / 8760
    project_years = config.project_lifetime_years
    
    # Scale simulation results to annual values
    annual_h2_kg = results.total_h2_production_kg / n_years
    annual_energy_kWh = results.total_energy_consumed_kWh / n_years
    annual_cycles = results.total_cycles / n_years
    
    # Calculate curtailed energy (available - consumed)
    total_available_kWh = np.sum(results.power_available_W) / 1000.0  # W·h to kWh
    total_consumed_kWh = results.total_energy_consumed_kWh
    total_curtailed_kWh = max(0, total_available_kWh - total_consumed_kWh)
    annual_curtailed_kWh = total_curtailed_kWh / n_years
    
    # CAPEX
    capex = compute_capex(config)
    econ.total_capex = capex['total']
    
    # Capital Recovery Factor
    r = config.discount_rate
    n = project_years
    if r > 0:
        crf = (r * (1 + r)**n) / ((1 + r)**n - 1)
    else:
        crf = 1 / n
    
    # Annualized CAPEX
    annualized_capex = capex['total'] * crf
    
    # Stack replacements over project lifetime
    # Estimate based on simulation cycling rate
    avg_hours_per_year = results.total_operating_hours / n_years
    equiv_hours_per_year = avg_hours_per_year + annual_cycles * config.cycling_penalty_hours
    years_per_stack = config.stack_lifetime_hours / equiv_hours_per_year
    n_stack_replacements = max(0, int(project_years / years_per_stack) - 1)  # First stack is CAPEX
    
    # Stack CAPEX base for replacement calculations
    P_kW = config.P_nom_MW * 1000
    stack_capex_base = config.capex_stack_eur_kW * P_kW
    learning_rate = getattr(config, 'learning_rate', 0.05)
    
    # =========================================================================
    # COMPONENT-LEVEL REPLACEMENT COSTS (Alkaline-specific)
    # =========================================================================
    # Alkaline electrolysers have different component lifetimes:
    # - Electrolyte (KOH): replaced every 3-5 years
    # - Electrodes/Catalyst: replaced every 10-15 years  
    # - Diaphragm/Separator: replaced every 8-12 years
    # - Mechanical (seals, gaskets): replaced every 5-7 years
    # - Full stack: replaced when cumulative degradation exceeds threshold
    
    component_costs = {
        'electrolyte': [],
        'catalyst': [],
        'diaphragm': [],
        'mechanical': [],
        'stack': []
    }
    
    # Get component replacement parameters (with defaults for backward compatibility)
    electrolyte_years = getattr(config, 'electrolyte_replacement_years', 4.0)
    catalyst_years = getattr(config, 'catalyst_replacement_years', 12.0)
    diaphragm_years = getattr(config, 'diaphragm_replacement_years', 10.0)
    mechanical_years = getattr(config, 'mechanical_replacement_years', 6.0)
    
    electrolyte_frac = getattr(config, 'electrolyte_replacement_frac', 0.05)
    catalyst_frac = getattr(config, 'catalyst_replacement_frac', 0.25)
    diaphragm_frac = getattr(config, 'diaphragm_replacement_frac', 0.15)
    mechanical_frac = getattr(config, 'mechanical_replacement_frac', 0.08)
    
    total_replacement_cost = 0.0
    
    for year in range(1, project_years + 1):
        # Apply learning rate: cost decreases over time
        learning_factor = (1 - learning_rate) ** (year - 1)
        
        # Electrolyte replacement
        if electrolyte_years > 0 and year % int(electrolyte_years) == 0:
            cost = stack_capex_base * electrolyte_frac * learning_factor
            component_costs['electrolyte'].append((year, cost))
            total_replacement_cost += cost / (1 + r) ** year
        
        # Catalyst/Electrode replacement
        if catalyst_years > 0 and year % int(catalyst_years) == 0:
            cost = stack_capex_base * catalyst_frac * learning_factor
            component_costs['catalyst'].append((year, cost))
            total_replacement_cost += cost / (1 + r) ** year
        
        # Diaphragm replacement
        if diaphragm_years > 0 and year % int(diaphragm_years) == 0:
            cost = stack_capex_base * diaphragm_frac * learning_factor
            component_costs['diaphragm'].append((year, cost))
            total_replacement_cost += cost / (1 + r) ** year
        
        # Mechanical (seals, gaskets) replacement
        if mechanical_years > 0 and year % int(mechanical_years) == 0:
            cost = stack_capex_base * mechanical_frac * learning_factor
            component_costs['mechanical'].append((year, cost))
            total_replacement_cost += cost / (1 + r) ** year
    
    # Full stack replacement (if needed based on degradation)
    for year in range(1, project_years + 1):
        if years_per_stack > 0 and year % int(years_per_stack) == 0 and year < project_years:
            learning_factor = (1 - learning_rate) ** (year - 1)
            cost = stack_capex_base * config.stack_replacement_cost_fraction * learning_factor
            component_costs['stack'].append((year, cost))
            total_replacement_cost += cost / (1 + r) ** year
    
    # Annualized replacement cost
    annual_replacement_cost = total_replacement_cost * crf
    
    # Annual OPEX (excluding replacements - handled separately)
    opex = compute_annual_opex(config, annual_energy_kWh, annual_h2_kg, 0)
    annual_opex = opex['total']
    
    # Inflation adjustment for OPEX (harmonized with PEM model)
    # OPEX escalates with inflation over project lifetime
    # Effective multiplier = sum((1+inf)^y / (1+r)^y) / sum(1/(1+r)^y)
    inflation_rate = getattr(config, 'inflation_rate', 0.025)
    if inflation_rate > 0:
        npv_opex_inflated = sum((1+inflation_rate)**y / (1+r)**y for y in range(1, project_years+1))
        npv_opex_nominal = sum(1/(1+r)**y for y in range(1, project_years+1))
        inflation_multiplier = npv_opex_inflated / npv_opex_nominal
        annual_opex *= inflation_multiplier
    
    # Curtailment penalty (opportunity cost of wasted renewable energy)
    enable_curtailment = getattr(config, 'enable_curtailment_penalty', True)
    curtailment_penalty_eur_per_MWh = getattr(config, 'curtailment_penalty_eur_per_MWh', 25.0)
    if enable_curtailment:
        annual_curtailment_cost = (annual_curtailed_kWh / 1000.0) * curtailment_penalty_eur_per_MWh
    else:
        annual_curtailment_cost = 0.0
    
    # =========================================================================
    # BY-PRODUCT REVENUES (Oxygen + Waste Heat)
    # =========================================================================
    # Oxygen credit - industrial O2 market value
    # Stoichiometry: 2H2O → 2H2 + O2, so 8 kg O2 per kg H2
    # Market: €50-150/tonne for industrial O2 (conservative: €100/tonne)
    enable_oxygen = getattr(config, 'enable_oxygen_credit', True)
    if enable_oxygen:
        oxygen_ratio = getattr(config, 'oxygen_to_h2_mass_ratio', 8.0)  # kg O2 per kg H2
        o2_price_eur_tonne = getattr(config, 'oxygen_selling_price_eur_tonne', 100.0)
        o2_purity = getattr(config, 'oxygen_purity', 0.995)
        annual_o2_kg = annual_h2_kg * oxygen_ratio * o2_purity
        annual_o2_revenue = (annual_o2_kg / 1000.0) * o2_price_eur_tonne
    else:
        annual_o2_revenue = 0.0
    
    # Waste heat recovery - low-grade heat for district heating
    # Alkaline electrolysers produce 50-60% of electrical input as waste heat (60-80°C)
    # Value: €15-25/MWh thermal (conservative: €20/MWh)
    enable_heat = getattr(config, 'enable_heat_recovery', True)
    if enable_heat:
        heat_recovery_eff = getattr(config, 'heat_recovery_efficiency', 0.55)  # 55%
        heat_price_eur_MWh = getattr(config, 'heat_selling_price_eur_MWh', 20.0)
        annual_heat_MWh = annual_energy_kWh * heat_recovery_eff / 1000.0  # kWh to MWh
        annual_heat_revenue = annual_heat_MWh * heat_price_eur_MWh
    else:
        annual_heat_revenue = 0.0
    
    # Total by-product revenue
    annual_byproduct_revenue = annual_o2_revenue + annual_heat_revenue
    
    # LCOH Components (with by-product credits reducing LCOH)
    if annual_h2_kg > 0:
        econ.lcoh_capex = annualized_capex / annual_h2_kg
        econ.lcoh_opex_fixed = (opex['fixed_om'] + opex.get('land_lease', 0)) / annual_h2_kg
        econ.lcoh_electricity = opex['electricity'] / annual_h2_kg
        econ.lcoh_compression = opex['compression'] / annual_h2_kg
        econ.lcoh_water = opex['water'] / annual_h2_kg
        econ.lcoh_stack_replacement = annual_replacement_cost / annual_h2_kg
        econ.lcoh_curtailment = annual_curtailment_cost / annual_h2_kg
        
        # By-product credits (negative LCOH components)
        econ.lcoh_oxygen_credit = -annual_o2_revenue / annual_h2_kg if enable_oxygen else 0.0
        econ.lcoh_heat_credit = -annual_heat_revenue / annual_h2_kg if enable_heat else 0.0
        
        econ.lcoh_total = (econ.lcoh_capex + econ.lcoh_opex_fixed + 
                          econ.lcoh_electricity + econ.lcoh_compression +
                          econ.lcoh_water + econ.lcoh_stack_replacement +
                          econ.lcoh_curtailment +
                          econ.lcoh_oxygen_credit + econ.lcoh_heat_credit)
    
    # Store by-product details
    econ.annual_o2_revenue = annual_o2_revenue
    econ.annual_heat_revenue = annual_heat_revenue
    econ.annual_byproduct_revenue = annual_byproduct_revenue
    
    # Store component replacement details
    econ.component_costs = component_costs
    econ.learning_rate = learning_rate
    
    # =========================================================================
    # PROFITABILITY ANALYSIS
    # =========================================================================
    
    # Annual revenue
    annual_revenue = annual_h2_kg * config.h2_selling_price_eur_kg
    
    # Cash flow arrays
    years = np.arange(project_years + 1)  # Year 0 to project_years
    cash_flows = np.zeros(project_years + 1)
    
    # Year 0: Initial investment
    cash_flows[0] = -capex['total']
    
    # Years 1 to n: Operating cash flows
    for year in range(1, project_years + 1):
        revenue = annual_revenue
        costs = annual_opex
        
        # Add stack replacement cost in relevant years
        if years_per_stack > 0 and year % int(years_per_stack) == 0 and year < project_years:
            costs += (config.capex_stack_eur_kW * P_kW 
                     * config.stack_replacement_cost_fraction)
        
        cash_flows[year] = revenue - costs
    
    econ.annual_cash_flow = cash_flows
    econ.cumulative_cash_flow = np.cumsum(cash_flows)
    econ.annual_revenue = np.full(project_years, annual_revenue)
    econ.annual_costs = np.full(project_years, annual_opex)
    econ.total_revenue = annual_revenue * project_years
    econ.total_costs = annual_opex * project_years
    econ.total_h2_production = annual_h2_kg * project_years
    
    # NPV
    discount_factors = (1 + r) ** (-years)
    econ.npv = np.sum(cash_flows * discount_factors)
    
    # Profitability Index
    econ.profitability_index = 1 + econ.npv / capex['total']
    
    # IRR (using numpy's financial functions approximation)
    try:
        # Simple IRR calculation via Newton-Raphson
        irr = _compute_irr(cash_flows)
        econ.irr = irr * 100  # Convert to percentage
    except:
        econ.irr = 0.0
    
    # Simple Payback
    cumulative = np.cumsum(cash_flows)
    payback_idx = np.where(cumulative >= 0)[0]
    if len(payback_idx) > 0:
        econ.payback_years = payback_idx[0]
    else:
        econ.payback_years = float('inf')
    
    # Discounted Payback
    discounted_cf = cash_flows * discount_factors
    cumulative_discounted = np.cumsum(discounted_cf)
    disc_payback_idx = np.where(cumulative_discounted >= 0)[0]
    if len(disc_payback_idx) > 0:
        econ.discounted_payback_years = disc_payback_idx[0]
    else:
        econ.discounted_payback_years = float('inf')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Economic Analysis - Alkaline Electrolyser")
        print(f"{'='*60}")
        print(f"\n[Project Parameters]")
        print(f"  System size: {config.P_nom_MW:.1f} MW")
        print(f"  Project lifetime: {project_years} years")
        print(f"  Discount rate: {r:.1%}")
        print(f"  H2 selling price: {config.h2_selling_price_eur_kg:.2f} EUR/kg")
        print(f"  Electricity price: {config.electricity_price_eur_kWh:.3f} EUR/kWh")
        
        print(f"\n[CAPEX Breakdown]")
        print(f"  Stack CAPEX: €{capex['stack']/1e6:.2f}M")
        print(f"  BoP CAPEX: €{capex['bop']/1e6:.2f}M")
        print(f"  Electrolyser Total: €{capex['electrolyser']/1e6:.2f}M")
        print(f"  System Boundary (water/site/grid): €{capex.get('system_boundary', 0)/1e6:.2f}M")
        print(f"  Storage (350 bar): €{capex['storage']/1e6:.2f}M")
        print(f"  Compressor: €{capex['compressor']/1e6:.2f}M")
        print(f"  Total CAPEX: €{capex['total']/1e6:.2f}M")
        print(f"  Specific CAPEX (electrolyser): {(capex['electrolyser']/(config.P_nom_MW*1000)):.0f} EUR/kW")
        
        print(f"\n[Annual Operations]")
        print(f"  H2 production: {annual_h2_kg/1000:.1f} tonnes/year")
        print(f"  Energy consumption: {annual_energy_kWh/1e6:.1f} GWh/year")
        print(f"  Average SEC: {results.average_sec_kWh_kg:.1f} kWh/kg")
        print(f"  Capacity factor: {results.capacity_factor_avg:.1%}")
        
        print(f"\n[LCOH Breakdown]")
        print(f"  CAPEX contribution: {econ.lcoh_capex:.2f} EUR/kg")
        print(f"  Fixed O&M: {econ.lcoh_opex_fixed:.2f} EUR/kg")
        print(f"  Electricity: {econ.lcoh_electricity:.2f} EUR/kg")
        print(f"  Compression (350 bar): {econ.lcoh_compression:.2f} EUR/kg")
        print(f"  Water: {econ.lcoh_water:.2f} EUR/kg")
        print(f"  Stack replacement: {econ.lcoh_stack_replacement:.2f} EUR/kg")
        print(f"  ─────────────────────────────")
        print(f"  TOTAL LCOH: {econ.lcoh_total:.2f} EUR/kg")
        
        print(f"\n[Profitability Metrics]")
        print(f"  NPV: €{econ.npv/1e6:.2f}M")
        print(f"  IRR: {econ.irr:.1f}%")
        print(f"  Simple Payback: {econ.payback_years:.1f} years")
        print(f"  Discounted Payback: {econ.discounted_payback_years:.1f} years")
        print(f"  Profitability Index: {econ.profitability_index:.2f}")
        
        profitable = econ.npv > 0
        print(f"\n  {'✅ PROJECT IS PROFITABLE' if profitable else '❌ PROJECT NOT PROFITABLE'}")
    
    return econ


def _compute_irr(cash_flows: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Compute Internal Rate of Return using Newton-Raphson method.
    
    Parameters
    ----------
    cash_flows : np.ndarray
        Cash flows starting from year 0
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float
        IRR as decimal (e.g., 0.15 for 15%)
    """
    # Initial guess
    irr = 0.10
    
    for _ in range(max_iter):
        npv = 0.0
        npv_deriv = 0.0
        
        for t, cf in enumerate(cash_flows):
            discount = (1 + irr) ** t
            npv += cf / discount
            if t > 0:
                npv_deriv -= t * cf / ((1 + irr) ** (t + 1))
        
        if abs(npv_deriv) < 1e-10:
            break
            
        irr_new = irr - npv / npv_deriv
        
        if abs(irr_new - irr) < tol:
            return irr_new
            
        irr = irr_new
        
        # Keep IRR in reasonable range
        irr = max(-0.99, min(irr, 1.0))
    
    return irr


def compute_lcoh_sensitivity(
    results: SimulationResults,
    config: AlkalineConfig,
    parameter: str,
    values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute LCOH sensitivity to a single parameter.
    
    Parameters
    ----------
    results : SimulationResults
        Base simulation results
    config : AlkalineConfig
        Base configuration
    parameter : str
        Parameter to vary (e.g., 'electricity_price_eur_kWh')
    values : np.ndarray
        Array of values to test
        
    Returns
    -------
    Dict[str, np.ndarray]
        {'parameter_values': values, 'lcoh': lcoh_array}
    """
    lcoh_values = np.zeros(len(values))
    
    for i, val in enumerate(values):
        # Create modified config
        config_mod = get_alkaline_config(**{parameter: val})
        # Copy other relevant attributes
        for attr in ['P_nom_MW', 'simulation_years']:
            setattr(config_mod, attr, getattr(config, attr))
        
        econ = compute_lcoh(results, config_mod, verbose=False)
        lcoh_values[i] = econ.lcoh_total
    
    return {
        'parameter_values': values,
        'lcoh': lcoh_values,
        'parameter_name': parameter
    }


def compare_scenarios(
    scenarios: Dict[str, AlkalineConfig],
    power_input_W: np.ndarray,
    verbose: bool = True
) -> Dict[str, Tuple[SimulationResults, EconomicResults]]:
    """
    Compare multiple scenarios (e.g., different sizes, electricity prices).
    
    Parameters
    ----------
    scenarios : Dict[str, AlkalineConfig]
        Dictionary of scenario name -> configuration
    power_input_W : np.ndarray
        Power timeseries (will be scaled per scenario)
    verbose : bool
        Print comparison table
        
    Returns
    -------
    Dict[str, Tuple[SimulationResults, EconomicResults]]
        Results for each scenario
    """
    all_results = {}
    
    for name, config in scenarios.items():
        # Scale power to match system size
        power_scaled = power_input_W * (config.P_nom_MW / 20.0)  # Assume 20 MW base
        
        sim_results = simulate(power_scaled, config, verbose=False)
        econ_results = compute_lcoh(sim_results, config, verbose=False)
        
        all_results[name] = (sim_results, econ_results)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Scenario Comparison")
        print(f"{'='*80}")
        print(f"{'Scenario':<20} | {'Size':<8} | {'H2 [t/y]':<10} | {'LCOH':<10} | {'NPV':<12} | {'IRR':<8}")
        print(f"{'-'*20} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*8}")
        
        for name, (sim, econ) in all_results.items():
            config = scenarios[name]
            h2_annual = sim.total_h2_production_kg / (len(sim.hours) / 8760) / 1000
            print(f"{name:<20} | {config.P_nom_MW:<8.1f} | {h2_annual:<10.1f} | "
                  f"€{econ.lcoh_total:<9.2f} | €{econ.npv/1e6:<11.1f}M | {econ.irr:<7.1f}%")
    
    return all_results


# =============================================================================
# PART 5: MONTE CARLO UNCERTAINTY ANALYSIS
# =============================================================================

@dataclass
class MonteCarloResults:
    """
    Results from Monte Carlo uncertainty analysis.
    """
    n_simulations: int = 0
    
    # Output distributions
    lcoh_values: np.ndarray = field(default_factory=lambda: np.array([]))
    npv_values: np.ndarray = field(default_factory=lambda: np.array([]))
    irr_values: np.ndarray = field(default_factory=lambda: np.array([]))
    h2_production_values: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Statistics
    lcoh_mean: float = 0.0
    lcoh_std: float = 0.0
    lcoh_p10: float = 0.0
    lcoh_p50: float = 0.0
    lcoh_p90: float = 0.0
    
    npv_mean: float = 0.0
    npv_std: float = 0.0
    npv_p10: float = 0.0
    npv_p90: float = 0.0
    probability_profitable: float = 0.0
    
    # Input parameters used
    parameter_samples: Dict[str, np.ndarray] = field(default_factory=dict)


def run_monte_carlo(
    base_config: AlkalineConfig,
    power_input_W: np.ndarray,
    n_simulations: int = 100,
    parameter_ranges: Optional[Dict[str, Tuple[float, float, str]]] = None,
    seed: Optional[int] = None,
    verbose: bool = True
) -> MonteCarloResults:
    """
    Run Monte Carlo uncertainty analysis on Alkaline electrolyser economics.
    
    Samples uncertain parameters from distributions and computes
    LCOH and profitability metrics for each sample.
    
    Parameters
    ----------
    base_config : AlkalineConfig
        Base configuration (mean values)
    power_input_W : np.ndarray
        Power timeseries
    n_simulations : int
        Number of Monte Carlo iterations
    parameter_ranges : Dict[str, Tuple[float, float, str]], optional
        Parameter uncertainty ranges: {name: (low, high, distribution)}
        Distribution can be 'uniform', 'normal', 'triangular'
        If None, uses default ranges.
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Print progress
        
    Returns
    -------
    MonteCarloResults
        Distribution of outputs
        
    Example
    -------
    >>> results = run_monte_carlo(config, power, n_simulations=500)
    >>> print(f"LCOH: {results.lcoh_mean:.2f} ± {results.lcoh_std:.2f} EUR/kg")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default parameter ranges (based on literature uncertainty)
    if parameter_ranges is None:
        parameter_ranges = {
            # Economic parameters
            'electricity_price_eur_kWh': (0.03, 0.10, 'uniform'),
            'capex_stack_eur_kW': (440, 660, 'triangular'),  # ±20% around 550
            'capex_bop_eur_kW': (320, 480, 'triangular'),    # ±20% around 400
            'h2_selling_price_eur_kg': (4.0, 8.0, 'uniform'),
            
            # Technical parameters
            'deg_rate_uV_h': (0.5, 1.2, 'triangular'),  # Modern range
            'stack_lifetime_hours': (80000, 100000, 'uniform'),
            
            # Operating parameters
            'P_bop_fraction': (0.06, 0.10, 'uniform'),
            'eta_rectifier': (0.95, 0.98, 'uniform'),
        }
    
    # Initialize results arrays
    lcoh_values = np.zeros(n_simulations)
    npv_values = np.zeros(n_simulations)
    irr_values = np.zeros(n_simulations)
    h2_values = np.zeros(n_simulations)
    
    # Store sampled parameters
    param_samples = {name: np.zeros(n_simulations) for name in parameter_ranges}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Monte Carlo Uncertainty Analysis")
        print(f"{'='*60}")
        print(f"  Simulations: {n_simulations}")
        print(f"  Uncertain parameters: {len(parameter_ranges)}")
        for name, (low, high, dist) in parameter_ranges.items():
            print(f"    - {name}: [{low:.3f}, {high:.3f}] ({dist})")
    
    # Run base simulation once (power timeseries doesn't change)
    base_results = simulate(power_input_W, base_config, verbose=False)
    
    # Monte Carlo loop
    report_interval = max(n_simulations // 10, 1)
    
    for i in range(n_simulations):
        # Sample parameters
        sampled_params = {}
        for name, (low, high, dist) in parameter_ranges.items():
            if dist == 'uniform':
                val = np.random.uniform(low, high)
            elif dist == 'normal':
                mean = (low + high) / 2
                std = (high - low) / 4  # 95% within range
                val = np.clip(np.random.normal(mean, std), low, high)
            elif dist == 'triangular':
                mode = (low + high) / 2
                val = np.random.triangular(low, mode, high)
            else:
                val = np.random.uniform(low, high)
            
            sampled_params[name] = val
            param_samples[name][i] = val
        
        # Create config with sampled parameters
        config_i = get_alkaline_config(**sampled_params)
        # Copy fixed parameters from base config
        config_i.P_nom_MW = base_config.P_nom_MW
        config_i.simulation_years = base_config.simulation_years
        config_i.n_stacks = base_config.n_stacks
        config_i.n_cells = base_config.n_cells
        config_i.cell_area_cm2 = base_config.cell_area_cm2
        
        # Compute economics with sampled parameters
        econ_i = compute_lcoh(base_results, config_i, verbose=False)
        
        lcoh_values[i] = econ_i.lcoh_total
        npv_values[i] = econ_i.npv
        irr_values[i] = econ_i.irr
        h2_values[i] = base_results.total_h2_production_kg
        
        if verbose and (i + 1) % report_interval == 0:
            print(f"  Progress: {(i+1)/n_simulations*100:.0f}%")
    
    # Compute statistics
    mc_results = MonteCarloResults(
        n_simulations=n_simulations,
        lcoh_values=lcoh_values,
        npv_values=npv_values,
        irr_values=irr_values,
        h2_production_values=h2_values,
        parameter_samples=param_samples,
        
        lcoh_mean=np.mean(lcoh_values),
        lcoh_std=np.std(lcoh_values),
        lcoh_p10=np.percentile(lcoh_values, 10),
        lcoh_p50=np.percentile(lcoh_values, 50),
        lcoh_p90=np.percentile(lcoh_values, 90),
        
        npv_mean=np.mean(npv_values),
        npv_std=np.std(npv_values),
        npv_p10=np.percentile(npv_values, 10),
        npv_p90=np.percentile(npv_values, 90),
        probability_profitable=np.mean(npv_values > 0) * 100
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Monte Carlo Results")
        print(f"{'='*60}")
        print(f"\n[LCOH Distribution]")
        print(f"  Mean: {mc_results.lcoh_mean:.2f} EUR/kg")
        print(f"  Std Dev: {mc_results.lcoh_std:.2f} EUR/kg")
        print(f"  P10: {mc_results.lcoh_p10:.2f} EUR/kg (optimistic)")
        print(f"  P50: {mc_results.lcoh_p50:.2f} EUR/kg (median)")
        print(f"  P90: {mc_results.lcoh_p90:.2f} EUR/kg (conservative)")
        
        print(f"\n[NPV Distribution]")
        print(f"  Mean: €{mc_results.npv_mean/1e6:.1f}M")
        print(f"  Std Dev: €{mc_results.npv_std/1e6:.1f}M")
        print(f"  P10: €{mc_results.npv_p10/1e6:.1f}M")
        print(f"  P90: €{mc_results.npv_p90/1e6:.1f}M")
        print(f"  Probability of profit: {mc_results.probability_profitable:.1f}%")
    
    return mc_results


def compute_sensitivity_tornado(
    base_config: AlkalineConfig,
    base_results: SimulationResults,
    parameters: Optional[Dict[str, Tuple[float, float]]] = None,
    verbose: bool = True
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute tornado diagram data for LCOH sensitivity.
    
    Parameters
    ----------
    base_config : AlkalineConfig
        Base configuration
    base_results : SimulationResults
        Base simulation results
    parameters : Dict[str, Tuple[float, float]], optional
        Parameters to vary: {name: (low, high)}
    verbose : bool
        Print results
        
    Returns
    -------
    Dict[str, Tuple[float, float, float]]
        {parameter: (lcoh_low, lcoh_base, lcoh_high)}
    """
    if parameters is None:
        parameters = {
            'electricity_price_eur_kWh': (0.03, 0.10),
            'capex_stack_eur_kW': (440, 660),
            'capex_bop_eur_kW': (320, 480),
            'h2_selling_price_eur_kg': (4.0, 8.0),
            'discount_rate': (0.05, 0.12),
            'stack_lifetime_hours': (70000, 100000),
        }
    
    # Base LCOH
    base_econ = compute_lcoh(base_results, base_config, verbose=False)
    base_lcoh = base_econ.lcoh_total
    
    sensitivities = {}
    
    for param, (low, high) in parameters.items():
        # Low value
        config_low = get_alkaline_config(**{param: low})
        config_low.P_nom_MW = base_config.P_nom_MW
        config_low.n_stacks = base_config.n_stacks
        config_low.n_cells = base_config.n_cells
        config_low.cell_area_cm2 = base_config.cell_area_cm2
        econ_low = compute_lcoh(base_results, config_low, verbose=False)
        
        # High value
        config_high = get_alkaline_config(**{param: high})
        config_high.P_nom_MW = base_config.P_nom_MW
        config_high.n_stacks = base_config.n_stacks
        config_high.n_cells = base_config.n_cells
        config_high.cell_area_cm2 = base_config.cell_area_cm2
        econ_high = compute_lcoh(base_results, config_high, verbose=False)
        
        sensitivities[param] = (econ_low.lcoh_total, base_lcoh, econ_high.lcoh_total)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Tornado Diagram - LCOH Sensitivity")
        print(f"{'='*60}")
        print(f"  Base LCOH: {base_lcoh:.2f} EUR/kg")
        print(f"\n  {'Parameter':<30} | {'Low':<10} | {'High':<10} | {'Swing':<10}")
        print(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10}")
        
        # Sort by swing (impact)
        sorted_params = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1][2] - x[1][0]),
            reverse=True
        )
        
        for param, (lcoh_low, _, lcoh_high) in sorted_params:
            swing = abs(lcoh_high - lcoh_low)
            print(f"  {param:<30} | {lcoh_low:<10.2f} | {lcoh_high:<10.2f} | ±{swing/2:<9.2f}")
    
    return sensitivities


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for Alkaline electrolyser simulation.
    
    Runs complete simulation with economics and optional Monte Carlo.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Alkaline Electrolyser Techno-Economic Simulation'
    )
    parser.add_argument('--size', type=float, default=20.0,
                        help='Electrolyser size [MW]')
    parser.add_argument('--years', type=int, default=5,
                        help='Simulation duration [years]')
    parser.add_argument('--elec-price', type=float, default=0.05,
                        help='Electricity price [EUR/kWh]')
    parser.add_argument('--h2-price', type=float, default=6.0,
                        help='H2 selling price [EUR/kg]')
    parser.add_argument('--monte-carlo', type=int, default=0,
                        help='Number of Monte Carlo simulations (0=skip)')
    parser.add_argument('--power-profile', type=str, default='variable',
                        choices=['constant', 'variable', 'intermittent'],
                        help='Power input profile type')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ALKALINE ELECTROLYSER TECHNO-ECONOMIC SIMULATION")
    print("="*70)
    
    # Create configuration
    config = get_alkaline_config(
        P_nom_MW=args.size,
        simulation_years=args.years,
        electricity_price_eur_kWh=args.elec_price,
        h2_selling_price_eur_kg=args.h2_price
    )
    
    # Run simulation
    results = quick_simulate(
        P_nom_MW=args.size,
        simulation_years=args.years,
        power_profile=args.power_profile,
        verbose=True
    )
    
    # Economic analysis
    econ = compute_lcoh(results, config, verbose=True)
    
    # Monte Carlo (if requested)
    if args.monte_carlo > 0:
        # Generate power profile for MC
        n_hours = args.years * 8760
        t = np.arange(n_hours)
        daily = 0.5 * (1 + np.sin(2 * np.pi * t / 24 - np.pi/2))
        seasonal = 0.8 + 0.2 * np.sin(2 * np.pi * t / 8760 - np.pi/2)
        noise = 0.1 * np.random.randn(n_hours)
        power = np.clip(0.3 + 0.5 * daily * seasonal + noise, 0, 1) * config.P_nom_W
        
        mc_results = run_monte_carlo(
            base_config=config,
            power_input_W=power,
            n_simulations=args.monte_carlo,
            verbose=True
        )
    
    # Save results
    if args.output:
        import pandas as pd
        
        # Create summary dataframe
        summary = {
            'Parameter': [
                'System Size [MW]', 'Simulation Years', 'Electricity Price [EUR/kWh]',
                'H2 Price [EUR/kg]', 'Total H2 [tonnes]', 'Total Energy [GWh]',
                'Avg SEC [kWh/kg]', 'Capacity Factor', 'LCOH [EUR/kg]',
                'NPV [EUR M]', 'IRR [%]', 'Payback [years]'
            ],
            'Value': [
                args.size, args.years, args.elec_price, args.h2_price,
                results.total_h2_production_kg / 1000,
                results.total_energy_consumed_kWh / 1e6,
                results.average_sec_kWh_kg, results.capacity_factor_avg,
                econ.lcoh_total, econ.npv / 1e6, econ.irr, econ.payback_years
            ]
        }
        pd.DataFrame(summary).to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    print("\n✓ Simulation complete!")
    
    return results, econ


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Alkaline Electrolyser - Part 1: Config + Electrochemistry")
    print("=" * 60)
    
    # Create default configuration
    config = get_alkaline_config(P_nom_MW=20.0)
    
    print(f"\n[Configuration]")
    print(f"  Nominal power: {config.P_nom_MW} MW")
    print(f"  Number of stacks: {config.n_stacks}")
    print(f"  Cells per stack: {config.n_cells}")
    print(f"  Cell area: {config.cell_area_cm2} cm²")
    print(f"  Operating temperature: {config.T_op_C}°C")
    print(f"  Operating pressure: {config.p_op_bar} bar")
    print(f"  CAPEX: {config.capex_stack_eur_kW + config.capex_bop_eur_kW} EUR/kW")
    
    # Test electrochemistry at nominal conditions
    T_K = config.T_op_K
    j = config.j_nom  # 0.30 A/cm²
    
    print(f"\n[Electrochemistry at j={j} A/cm², T={config.T_op_C}°C]")
    
    E_rev = compute_reversible_voltage(T_K, config.p_op_bar)
    print(f"  Reversible voltage: {E_rev:.3f} V")
    
    eta_act = compute_activation_overpotential(j, T_K, config)
    print(f"  Activation overpotential: {eta_act:.3f} V")
    
    eta_ohm = compute_ohmic_overpotential(j, T_K, config)
    print(f"  Ohmic overpotential: {eta_ohm:.3f} V")
    
    V_cell = compute_cell_voltage(j, T_K, config)
    print(f"  Total cell voltage: {V_cell:.3f} V")
    
    eta_F = compute_faraday_efficiency(j, T_K, config)
    print(f"  Faraday efficiency: {eta_F:.1%}")
    
    eta_stack = compute_stack_efficiency(V_cell, eta_F, basis='LHV')
    print(f"  Stack efficiency (LHV): {eta_stack:.1%}")
    
    eta_sys = compute_system_efficiency(eta_stack, config)
    print(f"  System efficiency (LHV): {eta_sys:.1%}")
    
    SEC = compute_specific_energy_consumption(eta_sys, basis='LHV')
    print(f"  Specific energy consumption: {SEC:.1f} kWh/kg H2")
    
    # Test over range of current densities
    print(f"\n[Voltage-Current Curve]")
    print(f"  {'j [A/cm²]':>10} | {'V_cell [V]':>10} | {'η_F [%]':>8} | {'η_sys [%]':>10}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*8} | {'-'*10}")
    
    for j_test in [0.05, 0.10, 0.20, 0.30, 0.40]:
        V = compute_cell_voltage(j_test, T_K, config)
        eta_F_test = compute_faraday_efficiency(j_test, T_K, config)
        eta_stack_test = compute_stack_efficiency(V, eta_F_test, basis='LHV')
        eta_sys_test = compute_system_efficiency(eta_stack_test, config)
        print(f"  {j_test:>10.2f} | {V:>10.3f} | {eta_F_test*100:>7.1f}% | {eta_sys_test*100:>9.1f}%")
    
    # =========================================================================
    # PART 2: TEST DEGRADATION MODEL
    # =========================================================================
    print("\n" + "=" * 60)
    print("Alkaline Electrolyser - Part 2: Degradation Model")
    print("=" * 60)
    
    # Get nominal voltage for degradation calculations
    V_cell_nominal = compute_cell_voltage(config.j_nom, T_K, config)
    print(f"\n[Degradation Parameters]")
    print(f"  Linear degradation rate: {config.deg_rate_uV_h} μV/h")
    print(f"  Cycling penalty: {config.cycling_penalty_hours}h equivalent per start")
    print(f"  Stack lifetime: {config.stack_lifetime_hours:,}h")
    print(f"  Voltage increase limit: {config.voltage_increase_limit:.0%}")
    
    # Scenario 1: Continuous operation (baseload)
    print(f"\n[Scenario 1: Baseload Operation - 8000h/year, 2 cycles/year]")
    baseload_hours_per_year = 8000
    baseload_cycles_per_year = 2  # Very few shutdowns
    
    # After 5 years
    years = 5
    total_hours = baseload_hours_per_year * years
    total_cycles = baseload_cycles_per_year * years
    
    V_deg_baseload = total_hours * config.deg_rate_uV_h * 1e-6
    V_deg_baseload += total_cycles * config.cycling_penalty_hours * config.deg_rate_uV_h * 1e-6
    
    equiv_hours_baseload = compute_equivalent_operating_hours(
        total_hours, total_cycles, 0, config
    )
    capacity_baseload = compute_capacity_fade(V_deg_baseload, V_cell_nominal)
    
    print(f"  After {years} years:")
    print(f"    Operating hours: {total_hours:,}h")
    print(f"    Cycles: {total_cycles}")
    print(f"    Equivalent hours: {equiv_hours_baseload:,.0f}h")
    print(f"    Voltage degradation: {V_deg_baseload*1000:.1f} mV ({V_deg_baseload/V_cell_nominal*100:.1f}%)")
    print(f"    Capacity factor: {capacity_baseload:.1%}")
    print(f"    Stack replacement needed: {'Yes' if equiv_hours_baseload > config.stack_lifetime_hours else 'No'}")
    
    # Scenario 2: Cycling operation (renewable-coupled)
    print(f"\n[Scenario 2: Cycling Operation - 4000h/year, 365 cycles/year]")
    cycling_hours_per_year = 4000
    cycling_cycles_per_year = 365  # Daily start/stop
    
    total_hours_cyc = cycling_hours_per_year * years
    total_cycles_cyc = cycling_cycles_per_year * years
    
    V_deg_cycling = total_hours_cyc * config.deg_rate_uV_h * 1e-6
    V_deg_cycling += total_cycles_cyc * config.cycling_penalty_hours * config.deg_rate_uV_h * 1e-6
    
    equiv_hours_cycling = compute_equivalent_operating_hours(
        total_hours_cyc, total_cycles_cyc, 0, config
    )
    capacity_cycling = compute_capacity_fade(V_deg_cycling, V_cell_nominal)
    
    print(f"  After {years} years:")
    print(f"    Operating hours: {total_hours_cyc:,}h")
    print(f"    Cycles: {total_cycles_cyc:,}")
    print(f"    Equivalent hours: {equiv_hours_cycling:,.0f}h")
    print(f"    Voltage degradation: {V_deg_cycling*1000:.1f} mV ({V_deg_cycling/V_cell_nominal*100:.1f}%)")
    print(f"    Capacity factor: {capacity_cycling:.1%}")
    print(f"    Stack replacement needed: {'Yes' if equiv_hours_cycling > config.stack_lifetime_hours else 'No'}")
    
    # Comparison
    print(f"\n[Comparison]")
    aging_rate_baseload = equiv_hours_baseload / total_hours  # Equiv hours per actual hour
    aging_rate_cycling = equiv_hours_cycling / total_hours_cyc
    print(f"  Equivalent hours per actual operating hour:")
    print(f"    Baseload: {aging_rate_baseload:.3f}x (minimal cycling penalty)")
    print(f"    Cycling:  {aging_rate_cycling:.3f}x ({(aging_rate_cycling-1)*100:.0f}% penalty)")
    print(f"  → Modern Alkaline handles cycling much better than legacy systems!")
    print(f"  → Legacy systems had ~1.5-2.0x aging rate under cycling")
    
    # Stack replacement estimation
    print(f"\n[Stack Replacement Estimation]")
    lifetime_baseload = config.stack_lifetime_hours / (baseload_hours_per_year + baseload_cycles_per_year * config.cycling_penalty_hours)
    lifetime_cycling = config.stack_lifetime_hours / (cycling_hours_per_year + cycling_cycles_per_year * config.cycling_penalty_hours)
    
    print(f"  Baseload: Stack lasts ~{lifetime_baseload:.1f} years")
    print(f"  Cycling:  Stack lasts ~{lifetime_cycling:.1f} years")
    print(f"  Modern vs Legacy improvement: ~50% longer lifetime")
    
    print("\n✓ Part 2 complete - Degradation model with modern (2023+) parameters!")
    
    # =========================================================================
    # PART 3: TEST MAIN SIMULATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("Alkaline Electrolyser - Part 3: Main Simulation")
    print("=" * 60)
    
    # Quick simulation with synthetic variable power (1 year for fast test)
    print("\n[Running 1-year simulation with variable renewable power...]")
    results = quick_simulate(
        P_nom_MW=20.0,
        simulation_years=1,
        power_profile='variable',
        verbose=True
    )
    
    # Detailed results
    print(f"\n[Detailed Results]")
    print(f"  Hourly SEC range: {np.min(results.sec_kWh_kg[results.is_operating]):.1f} - "
          f"{np.max(results.sec_kWh_kg[results.is_operating]):.1f} kWh/kg")
    print(f"  Cell voltage range: {np.min(results.cell_voltage_V[results.is_operating]):.3f} - "
          f"{np.max(results.cell_voltage_V[results.is_operating]):.3f} V")
    print(f"  Current density range: {np.min(results.current_density_A_cm2[results.is_operating]):.3f} - "
          f"{np.max(results.current_density_A_cm2[results.is_operating]):.3f} A/cm²")
    
    # Monthly breakdown
    print(f"\n[Monthly H2 Production]")
    hours_per_month = 730  # Approximate
    for month in range(12):
        start_h = month * hours_per_month
        end_h = min((month + 1) * hours_per_month, len(results.h2_production_kg))
        monthly_h2 = np.sum(results.h2_production_kg[start_h:end_h])
        monthly_op_hours = np.sum(results.is_operating[start_h:end_h])
        print(f"  Month {month+1:2d}: {monthly_h2/1000:.1f} tonnes, {monthly_op_hours}h operating")
    
    print("\n✓ Part 3 complete - Main simulation function works!")
    
    # =========================================================================
    # PART 4: TEST ECONOMICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("Alkaline Electrolyser - Part 4: Economics & Profitability")
    print("=" * 60)
    
    # Run economic analysis on the simulation results
    econ = compute_lcoh(results, config, verbose=True)
    
    # Sensitivity to electricity price
    print(f"\n[LCOH Sensitivity to Electricity Price]")
    elec_prices = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"  Elec Price [EUR/kWh] | LCOH [EUR/kg]")
    print(f"  {'-'*21}|{'-'*15}")
    for price in elec_prices:
        config_test = get_alkaline_config(P_nom_MW=20.0, electricity_price_eur_kWh=price)
        econ_test = compute_lcoh(results, config_test, verbose=False)
        print(f"  {price:>21.2f}| {econ_test.lcoh_total:>14.2f}")
    
    # Sensitivity to H2 selling price
    print(f"\n[Profitability vs H2 Selling Price]")
    h2_prices = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    print(f"  H2 Price [EUR/kg] | NPV [EUR M] | IRR [%] | Profitable?")
    print(f"  {'-'*18}|{'-'*13}|{'-'*9}|{'-'*13}")
    for price in h2_prices:
        config_test = get_alkaline_config(P_nom_MW=20.0, h2_selling_price_eur_kg=price)
        econ_test = compute_lcoh(results, config_test, verbose=False)
        profitable = '✅ Yes' if econ_test.npv > 0 else '❌ No'
        print(f"  {price:>18.1f}| {econ_test.npv/1e6:>11.1f} | {econ_test.irr:>7.1f} | {profitable}")
    
    print("\n✓ Part 4 complete - Economics and profitability analysis works!")
    
    # =========================================================================
    # PART 5: TEST MONTE CARLO
    # =========================================================================
    print("\n" + "=" * 60)
    print("Alkaline Electrolyser - Part 5: Monte Carlo Analysis")
    print("=" * 60)
    
    # Generate power profile for MC
    n_hours = 8760  # 1 year
    t = np.arange(n_hours)
    daily = 0.5 * (1 + np.sin(2 * np.pi * t / 24 - np.pi/2))
    seasonal = 0.8 + 0.2 * np.sin(2 * np.pi * t / 8760 - np.pi/2)
    noise = 0.1 * np.random.randn(n_hours)
    power = np.clip(0.3 + 0.5 * daily * seasonal + noise, 0, 1) * config.P_nom_W
    
    # Run Monte Carlo with reduced iterations for quick test
    mc_results = run_monte_carlo(
        base_config=config,
        power_input_W=power,
        n_simulations=50,  # Quick test
        seed=42,
        verbose=True
    )
    
    # Tornado diagram
    print("\n[Tornado Diagram]")
    base_results_for_tornado = simulate(power, config, verbose=False)
    tornado = compute_sensitivity_tornado(config, base_results_for_tornado, verbose=True)
    
    print("\n✓ Part 5 complete - Monte Carlo and sensitivity analysis works!")
    
    # =========================================================================
    # PART 6: RUN WITH REAL DATA (5-YEAR SIMULATION)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Alkaline Electrolyser - Part 6: Real Data Simulation (5 Years)")
    print("=" * 70)
    
    # Run 5-year simulation with real wind+solar data and real demand
    # Power: 10% deterministic variability
    # Demand: 5% deterministic variability
    results_real, econ_real = run_with_real_data(
        P_nom_MW=20.0,
        simulation_years=5,
        power_variability_pct=10.0,
        demand_variability_pct=5.0,
        electricity_price=0.05,
        h2_price=6.0,
        include_demand=False,  # Set to True if you want demand tracking
        storage_capacity_kg=5000.0,
        verbose=True,
        save_results=True,
        output_folder='results/alkaline_real_data'
    )
    
    print("\n✓ Part 6 complete - 5-year simulation with real data saved!")
    
    print("\n" + "=" * 70)
    print("ALL PARTS COMPLETE - Alkaline Electrolyser Module Ready!")
    print("=" * 70)


