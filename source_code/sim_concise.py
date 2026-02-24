"""
PEM Electrolyser Techno-Economic Simulation - ENHANCED PHYSICS VERSION
========================================================================

Author: Shubham Manchanda
Thesis: Techno-Economic Optimization of Electrolyser Performance

================================================================================
VERSION: 2.1 - Updated CAPEX & Expanded Optimization Ranges
================================================================================

PHYSICS ENHANCEMENTS (vs. simplified models):
1. Dual Tafel slopes for OER (anode) and HER (cathode)
2. Concentration overpotential at high current densities
3. Pressure-corrected reversible voltage (Nernst equation)
4. Variable Faradaic efficiency (crossover losses at low j)
5. Dynamic thermal management with cold start penalties
6. Temperature-dependent membrane conductivity

ECONOMICS UPDATES (2024-2025 realistic values):
- CAPEX: €1,950/kW total (detailed breakdown below)
  * Stack: €950/kW (MEA, bipolar plates)
  * BoP: €550/kW (rectifier, pumps, controls)
  * Installation: €200/kW
  * Engineering: €150/kW
  * Water Treatment: €50/kW
  * Site Prep: €50/kW
- Storage: €800/kg (Type IV composite @ 350 bar)
- Compressor: €2,800 per kg/h capacity
- Grid Connection: €45/kW (even for off-grid systems)
- Contingency: 10% of CAPEX
- OPEX: 4% fixed + 2.5% variable + water + land lease
- Learning rate: 3% annual for replacement stacks
- Credits: Heat recovery (€0.03/kWh), O2 byproduct (€0.20/kg H2)

================================================================================
HOW TO RUN THIS SIMULATION
================================================================================

1. STANDARD SIMULATION:
   $ cd src
   $ python sim_concise.py
   
   - Enter electrolyser size when prompted (e.g., 10, 15, 20 MW)
   - Enter 'n' for standard simulation
   - Results saved to: results/{size}MW_sim/

2. MONTE CARLO UNCERTAINTY ANALYSIS:
   $ python sim_concise.py
   
   - Enter electrolyser size (e.g., 20)
   - Enter 'y' for Monte Carlo
   - Enter number of simulations (e.g., 100)
   - Results include LCOH distribution and sensitivity analysis

3. KEY PARAMETERS (in get_config()):
   - CAPEX_EUR_PER_KW: Total system CAPEX (default: €1,950)
   - LCOE_ELECTRICITY_EUR_PER_KWH: Electricity price (default: €0.070)
   - DISCOUNT_RATE: WACC (default: 0.08 = 8%)
   - STACK_LIFETIME_HOURS: 60,000-80,000 hours (creates LCOH range)
   - R_OHM: Cell resistance (default: 0.18 Ω·cm²)
   - USE_DUAL_TAFEL: Enable advanced polarization model (default: True)

================================================================================
OPTIMIZATION RANGES
================================================================================
- Electrolyser Size: 5 - 50 MW (10 steps)
- Storage Capacity: 300 - 10,000 kg @ 350 bar (11 steps)
- RE Fraction: 20%, 40%, 60%, 80%, 100%
- Stack Lifetime: 60,000 - 80,000 hours (LCOH range driver)

================================================================================
DATA SOURCES
================================================================================
- Power: Real wind+solar data from .mat file (8,760 hours)
- Demand: Real industrial H2 demand from CSV
- Variability: ±8% power, ±5% demand per year
- Modes: Random (default) or Deterministic (reproducible base case)

================================================================================
EXPECTED OUTPUTS (20 MW system, 15-year lifetime)
================================================================================
- LCOH: ~6.5-9.5 EUR/kg (depending on CF, RE fraction, lifetime)
- SEC Stack: ~48-52 kWh/kg H2 (increases with degradation)
- SEC System: ~54-60 kWh/kg H2 (stack + 1.5 parasitic + 3.5 compression + cooling)
- Total H2: ~25-35 million kg over 15 years
- Stack replacements: 1-2 (depending on capacity factor)

================================================================================
MATHEMATICAL MODELS (ENHANCED)
================================================================================

1. POLARIZATION CURVE (Full physics):
   V = E_rev(T,P) + η_act,anode + η_act,cathode + η_ohm + η_conc
   
   Where:
   - E_rev(T,P) = 1.229 - 0.00085(T-25) + 0.0295·log₁₀(P/P₀)
   - η_act,anode = B_a·ln(j/j₀_a)  [OER: B_a=0.06V, j₀=10⁻⁷ A/cm²]
   - η_act,cathode = B_c·ln(j/j₀_c) [HER: B_c=0.03V, j₀=10⁻³ A/cm²]
   - η_ohm = R_ohm(T)·j  [R=0.22 Ω·cm² @ 80°C]
   - η_conc = C·(j/j_lim)^m / (1-j/j_lim) [mass transport limitation]
   
   Sources: Carmo et al. (2013), Ursua et al. (2012), Trinke et al. (2018)

2. FARADAIC EFFICIENCY (Variable):
   η_F(j) = η_F_max × (1 - j_crossover/j)
   
   At low j: crossover losses dominate → η_F decreases
   At high j: η_F → 99%
   Source: Schalenbach et al. (2016)

3. DEGRADATION MODEL (Literature-validated):
   ΔV_deg = ∫ r_base × f_load × f_temp dt + n_cycles × ΔV_cycle
   
   - r_base = 2.5 μV/h (Frensch 2019: 2-5 μV/h load-following operation)
   - f_load = (j/j_nom)^1.5 (load dependence)
   - f_temp = exp(E_a/R × (1/T_ref - 1/T)) [Arrhenius, E_a=50 kJ/mol]
   - ΔV_cycle = 15 μV per start/stop (Weiß 2019)

4. ECONOMICS (NPV method):
   LCOH = [CAPEX + NPV(OPEX) + NPV(replacements) - NPV(credits)] / NPV(H₂)
   
   CAPEX includes: Stack, BoP, installation, engineering, storage, 
                   compressor, water treatment, site prep, contingency
   OPEX includes: Electricity, fixed maintenance, water, land lease
   Credits: Heat recovery, O2 byproduct (optional)

================================================================================
LITERATURE REFERENCES
================================================================================
- Carmo et al. (2013) Int. J. Hydrogen Energy - PEM fundamentals
- Ursua et al. (2012) Proc. IEEE - Comprehensive electrolyser review
- Trinke et al. (2018) J. Electrochem. Soc. - Mass transport
- Schalenbach et al. (2016) J. Phys. Chem. C - Gas crossover
- Frensch et al. (2019) Int. J. Hydrogen Energy - Degradation
- IRENA (2024) Green Hydrogen Cost Reduction - Economics
- IEA (2024) Global Hydrogen Review - Market data

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io
import os
from scipy.optimize import curve_fit
from numpy.polynomial import polynomial as P
from scipy import stats

# Set reproducible random seed
RNG_SEED = 42


# =============================================================================
# CONFIGURATION - EDIT PARAMETERS HERE
# =============================================================================

def get_config(size_mw=20.0, storage_kg=2500.0):
    """Return configuration dictionary for simulation."""
    return {
        # Simulation parameters
        'YEARS': 15,  # Extended to 15 years per thesis requirement
        'HOURS_PER_YEAR': 8760,
        
        # Electrolyser sizing
        'ELECTROLYSER_SIZE_MW': size_mw,
        'ELECTROLYSER_SIZE_KW': size_mw * 1000.0,
        
        # Power conversion
        'RECTIFIER_EFF': 0.93,           # AC/DC rectifier efficiency (conservative: 0.93-0.97)
        'PARASITIC_KWH_PER_KG': 1.5,     # Parasitic loads: pumps, controls (kWh/kg) - reduced, cooling now explicit
        'COMP_ENERGY_KWH_PER_KG': 3.5,   # Compression to 350 bar (literature: 2.5-4 kWh/kg)
        'COMPRESSOR_EFF': 0.75,          # Multi-stage compressor efficiency
        'COMPRESSOR_CAPEX_EUR_PER_KG': 45.0,
        'COMPRESSOR_MAINT_FRAC': 0.04,
        
        # Storage
        'STORAGE_CAPACITY_KG': storage_kg,
        'STORAGE_EFF': 0.98,  # Roundtrip efficiency (applied on discharge)
        
        # =========================================================================
        # OPERATING CONSTRAINTS (PEM advantages over Alkaline)
        # =========================================================================
        # PEM is known for excellent dynamic response:
        # - Minimum load: 5-10% (vs 15-25% for Alkaline)
        # - Ramp rate: >100%/s possible (vs 10-20%/min for Alkaline)
        # References: IRENA (2024), Nel Hydrogen, ITM Power specs
        # =========================================================================
        'MIN_LOAD_FRAC': 0.15,           # 15% minimum load (conservative PEM, real-world 10-15%)
        'MAX_RAMP_FRAC_PER_H': 1.0,      # 100%/hour (PEM can ramp full range in <1 min, conservative hourly)
        'STACK_LIFE_HOURS': 60000.0,     # Conservative PEM lifetime (60k proven, 80k is future target)
        
        # =========================================================================
        # THERMAL MANAGEMENT PARAMETERS
        # Sources: Olivier et al. (2017), Trinke et al. (2018), Scheepers et al. (2022)
        # =========================================================================
        # 
        # THERMAL MODEL APPROACH:
        # - Full model tracks T_stack dynamically with energy balance
        # - For TEA (hourly timestep): thermal time constant τ << 1 hour
        # - Result: T_stack stabilizes to T_OPERATING_C within first timestep
        # - Effectively "constant temperature" for economic calculations
        # - Cold start penalty captures startup transients economically
        #
        # =========================================================================
        
        # Operating temperatures
        'T_AMBIENT_C': 20.0,          # Ambient temperature (°C)
        'T_OPERATING_C': 80.0,        # Target operating temperature (°C) - steady-state
        'T_MIN_OPERATING_C': 50.0,    # Minimum temp for efficient operation (°C)
        'T_MAX_OPERATING_C': 90.0,    # Maximum safe operating temp (°C)
        'T_REF_C': 25.0,              # Reference temperature for E_rev (°C)
        'R_OHM_REF_TEMP_C': 80.0,     # Reference temp for R_ohm (°C)
        'R_OHM_TEMP_COEFF': -0.005,   # R_ohm decreases ~0.5%/°C (negative = decreases with T)
        
        # Thermal properties
        'STACK_THERMAL_MASS_KJ_PER_K': 500.0,  # Thermal mass per MW (kJ/K/MW) - steel + water
        'COOLANT_FLOW_KG_PER_S_PER_MW': 2.0,   # Coolant flow rate (kg/s per MW)
        'COOLANT_CP_KJ_PER_KG_K': 4.18,        # Water specific heat (kJ/kg·K)
        'COOLANT_INLET_TEMP_C': 40.0,          # Coolant inlet temperature (°C)
        'HEAT_TRANSFER_COEFF': 0.95,           # Heat exchanger effectiveness
        
        # Cold start parameters
        'COLD_START_TIME_HOURS': 0.5,          # Time to reach operating temp from cold (h)
        'COLD_START_EFFICIENCY_PENALTY': 0.20, # Efficiency reduction during warmup (20% - conservative)
        'HOT_STANDBY_POWER_FRAC': 0.02,        # Power for hot standby (2% of rated)
        'STANDBY_TEMP_C': 60.0,                # Temperature maintained in standby (°C)
        
        # Heat recovery (optional economic benefit)
        'HEAT_RECOVERY_ENABLED': True,         # Enable heat recovery calculation
        'HEAT_RECOVERY_EFFICIENCY': 0.50,      # Heat recovery system efficiency (conservative: 40-60%)
        'HEAT_VALUE_EUR_PER_KWH': 0.03,        # Value of recovered heat (EUR/kWh_th)
        
        # Thermoneutral voltage for heat calculation
        'V_THERMONEUTRAL': 1.48,               # Thermoneutral voltage (V) at which reaction is adiabatic
        
        # =========================================================================
        # PARTIAL LOAD EFFICIENCY PARAMETERS (PEM advantage over Alkaline)
        # =========================================================================
        # PEM has excellent part-load performance due to:
        # - Lower internal resistance at low currents (thin membrane)
        # - Better gas separation (solid polymer vs liquid electrolyte)
        # - Faster startup and shutdown (less thermal mass)
        # 
        # References: Buttler & Spliethoff (2018), IRENA (2024)
        # PEM can maintain good efficiency down to 5-10% load
        # =========================================================================
        'PARTIAL_LOAD_THRESHOLD': 0.25,  # Below 25%, efficiency drops (conservative PEM)
        'PARTIAL_LOAD_EFF_MIN': 0.85,    # Minimum efficiency factor at min load (conservative: 0.85)
        
        # =========================================================================
        # ENHANCED ELECTROCHEMISTRY PARAMETERS (REALISTIC PHYSICS)
        # Sources: Carmo et al. (2013), Buttler & Spliethoff (2018), Babic (2017),
        #          Ursua et al. (2012), Trinke et al. (2018), Schalenbach et al. (2016)
        # =========================================================================
        'N_CELLS': 300,
        'CELL_AREA_CM2': 3000.0,
        
        # Reversible voltage (thermodynamic)
        'E_REV_STD': 1.229,                  # E° at STP (V) - thermodynamic value
        'E_REV_TEMP_COEFF': -0.00085,        # dE/dT (V/K) - Nernst equation
        'E_REV_PRESSURE_COEFF': 0.0295,      # Pressure correction (V per decade of pressure ratio)
        'OPERATING_PRESSURE_BAR': 30.0,      # Cell operating pressure (bar) - realistic PEM
        
        # =========================================================================
        # DUAL TAFEL SLOPES for HER and OER (Calibrated to literature)
        # Target: V ≈ 1.77V at j=1.5 A/cm², 80°C, 30 bar (modern PEM)
        # 
        # At 80°C: E_rev ≈ 1.182V (temperature correction)
        # Pressure: +0.044V (30 bar)
        # E_rev(80°C, 30bar) ≈ 1.226V
        # 
        # Required overpotential at j=1.5: 1.77 - 1.226 = 0.544V
        #   - η_act (total): ~0.33V
        #   - η_ohm: ~0.21V (at R=0.14 Ω·cm², Nafion 212)
        #   - η_conc: ~0V (negligible at 1.5 A/cm²)
        # =========================================================================
        'B_TAFEL_ANODE': 0.040,              # OER Tafel slope (V) - IrO2/RuO2
        'B_TAFEL_CATHODE': 0.020,            # HER Tafel slope (V) - Pt/C
        'J0_ANODE': 2.0e-3,                  # OER exchange current density (A/cm²) - effective
        'J0_CATHODE': 5.0e-2,                # HER exchange current density (A/cm²) - Pt is fast
        
        # Legacy single Tafel (used when USE_DUAL_TAFEL=False)
        'B_TAFEL': 0.050,                    # Effective combined (V)
        'J0': 0.01,                          # Effective j0 (A/cm²)
        'USE_DUAL_TAFEL': True,              # Enable dual electrode model
        
        # =========================================================================
        # OHMIC RESISTANCE (Membrane + Contact)
        # Modern PEM uses thin membranes (Nafion 212, ~50μm) with lower resistance
        # At j=1.5 A/cm², R=0.14 Ω·cm² gives η_ohm = 0.21V
        # References: Babic (2017), Trinke (2018), IRENA (2024)
        # =========================================================================
        'R_OHM': 0.14,                       # Total ohmic (Ω·cm²) - Nafion 212
        'R_MEMBRANE': 0.08,                  # Membrane contribution (Nafion 212, ~50μm)
        'R_CONTACT': 0.06,                   # Contact/plate resistance
        
        # =========================================================================
        # CONCENTRATION OVERPOTENTIAL (Mass Transport Limitation)
        # Only significant above j > 2 A/cm² for well-designed cells
        # =========================================================================
        'ENABLE_CONCENTRATION_OVERPOTENTIAL': True,
        'J_LIMITING': 4.0,                   # Limiting current density (A/cm²)
        'C_CONC': 0.03,                      # Concentration coefficient (V) - small effect
        'M_CONC': 2.0,                       # Concentration exponent
        
        # =========================================================================
        # DEGRADATION PARAMETERS - COMPONENT-LEVEL (Literature-based)
        # =========================================================================
        # 
        # REFERENCES:
        # - Frensch et al. (2019) Int. J. Hydrogen Energy 44: 29338-29349
        #   "Impact of intermittent operation on lifetime of PEM electrolyzers"
        #   → 2-5 μV/h for load-following operation
        # 
        # - Rakousky et al. (2017) J. Power Sources 342: 38-47
        #   "Influence of cycling on degradation in PEM electrolysis"
        #   → Cycling adds 10-20 μV per start/stop
        # 
        # - Chandesris et al. (2015) Int. J. Hydrogen Energy 40: 1353-1366
        #   "Membrane degradation in PEM electrolysis: A modeling approach"
        #   → Membrane: 0.8-1.5 μV/h (chemical degradation)
        #   → E_a ≈ 50 kJ/mol for membrane degradation
        # 
        # - Feng et al. (2017) J. Power Sources 366: 33-55
        #   "A comprehensive review of PEM electrolysis degradation"
        #   → Catalyst (anode): 0.5-1.0 μV/h (Ir dissolution, oxide formation)
        #   → Catalyst (cathode): 0.2-0.5 μV/h (Pt coarsening, ECSA loss)
        # 
        # - Weiß et al. (2019) J. Electrochem. Soc. 166: F487-F497
        #   "Start/stop degradation in PEM electrolysis"
        #   → 15-25 μV per start/stop cycle
        # 
        # - IRENA (2024) Green Hydrogen Cost Reduction: Scaling up Electrolysers
        #   → Target: 80,000 h stack lifetime for modern PEM
        # 
        # DEGRADATION MODEL:
        # Total degradation = Membrane + Catalyst (anode) + Catalyst (cathode) + Cycling
        # 
        # =========================================================================
        
        # Component-level degradation rates (V/h = μV/h × 1e-6)
        'R_DEG_MEMBRANE': 1.0e-6,    # Membrane degradation: 1.0 μV/h (Chandesris 2015)
        'R_DEG_CATALYST_ANODE': 0.6e-6,  # Anode catalyst (Ir): 0.6 μV/h (Feng 2017)
        'R_DEG_CATALYST_CATHODE': 0.4e-6, # Cathode catalyst (Pt): 0.4 μV/h (Feng 2017)
        
        # Combined base rate (sum of components) = 2.0 μV/h
        'R_DEG_BASE': 2.0e-6,       # Total: 1.0 + 0.6 + 0.4 = 2.0 μV/h (Frensch 2019)
        
        'DEG_LOAD_EXPONENT': 1.3,   # Load dependence exponent (higher load = faster degradation)
        'DELTA_V_CYCLE': 18e-6,     # Voltage penalty per start/stop cycle (V = 18 μV)
                                    # (Weiß 2019: 15-25 μV per cycle, use mid-range)
        
        'DEG_E_ACT': 50000.0,       # Activation energy for degradation (J/mol)
                                    # (Chandesris 2015: ~50 kJ/mol for membrane)
        'T_DEG_REF': 343.15,        # Reference temperature for degradation (K = 70°C)
        
        'V_THRESHOLD_INCREASE': 0.15,  # Cumulative voltage degradation threshold (V = 150 mV)
                                        # ~9% efficiency loss, industry standard for replacement
        
        # Hours-based lifetime limit (dual criteria)
        'STACK_LIFETIME_HOURS': 60000,  # 60,000 h - conservative proven PEM lifetime
        'USE_HOURS_BASED_REPLACEMENT': True,  # Enable hours-based check
        
        # =========================================================================
        # VARIABLE FARADAIC EFFICIENCY MODEL (Schalenbach et al. 2016)
        # Faradaic efficiency decreases at low current density due to:
        # - Gas crossover through membrane (H2 permeation to anode)
        # - Recombination losses
        # - Side reactions
        # η_F(j) = η_F_max × (1 - j_crossover/j)
        # =========================================================================
        'FARADAY': 96485.0,
        'M_H2': 0.002016,
        'ETA_F_MAX': 0.99,                   # Maximum Faradaic efficiency at high j
        'ETA_F_MIN': 0.85,                   # Minimum Faradaic efficiency at very low j
        'J_CROSSOVER': 0.01,                 # Equivalent crossover current density (A/cm²)
        'ENABLE_VARIABLE_FARADAIC_EFF': True,
        'ETA_F': 0.97,                       # Nominal Faradaic efficiency (for backward compatibility)
        'J_NOMINAL': 1.5,                    # Nominal current density (A/cm²)
        
        # =========================================================================
        # REALISTIC 2024-2025 ECONOMICS (IRENA 2024, IEA 2024, BloombergNEF)
        # CONSERVATIVE values for commercial PEM systems (not optimistic)
        # =========================================================================
        
        # =========================================================================
        # CAPEX BREAKDOWN - DETAILED (2024-2025 Conservative Estimates)
        # Total Target: €1,950/kW (mid-range for commercial PEM systems)
        # Sources: IRENA (2024), IEA (2024), BloombergNEF, industry quotes
        # =========================================================================
        # 
        # Component                  | €/kW   | % of Total | Notes
        # --------------------------|--------|------------|---------------------------
        # Stack                      | 950    | 48.7%      | MEA, bipolar plates, endplates
        # Balance of Plant (BoP)     | 550    | 28.2%      | Power electronics, pumps, controls
        # Installation & Commission  | 200    | 10.3%      | Labor, testing, startup
        # Engineering & Management   | 150    | 7.7%       | Design, procurement, PM
        # Water Treatment System     | 50     | 2.6%       | RO + deionization (separate)
        # Site Preparation           | 50     | 2.6%       | Civil works, foundations
        # --------------------------|--------|------------|---------------------------
        # SUBTOTAL                   | 1950   | 100%       | Before contingency
        # Contingency (10%)          | 195    |            | Risk buffer
        # --------------------------|--------|------------|---------------------------
        # TOTAL with Contingency     | 2145   |            | All-in installed cost
        # 
        # Note: Grid connection (€45/kW), compressor, and storage calculated separately
        # Note: System boundary items (water treatment, site prep, grid) are included
        #       in BOTH PEM and Alkaline models for fair comparison
        # =========================================================================
        
        'STACK_CAPEX_EUR_PER_KW': 950.0,     # Stack only (MEA, plates) - €900-1100 range
        'BOP_CAPEX_EUR_PER_KW': 550.0,       # BoP: rectifier, pumps, heat exchanger, controls
        'INSTALLATION_CAPEX_EUR_PER_KW': 200.0,  # Installation, commissioning, testing
        'ENGINEERING_CAPEX_EUR_PER_KW': 150.0,   # Engineering, project management, permitting
        'CAPEX_EUR_PER_KW': 1950.0,          # Total CAPEX = 950+550+200+150+50+50 (2024 realistic)
        'BOP_CAPEX_FRAC': 0.0,               # Already included in total (not using fraction)
        
        # OPEX - realistic maintenance burden (harmonized with Alkaline)
        'FIXED_OPEX_FRAC': 0.03,             # 3% of CAPEX/year (IRENA 2024 mid-range: 2-4%)
        'VAR_OPEX_FRAC': 0.0,                # Absorbed into fixed OPEX for harmonization
        'STORAGE_OPEX_FRAC': 0.03,           # 3% for pressurized storage (harmonized)
        
        # Storage - REALISTIC Type IV composite tanks @ 350 bar
        'STORAGE_CAPEX_EUR_PER_KG': 800.0,   # Type IV tanks (conservative: €700-900)
        
        # Compressor - multi-stage to 350 bar (harmonized with Alkaline model)
        # Sized based on storage capacity for fair comparison
        # Reference: IEA (2024), IRENA (2024) — €50-80/kg capacity
        'COMPRESSOR_CAPEX_EUR_PER_KG_STORED': 55.0,  # €/kg storage capacity (harmonized)
        'COMPRESSOR_CAPEX_EUR_PER_KG_H2_PER_H': 0.0,  # Disabled — using per-kg-stored model
        
        # Financial parameters - typical project finance
        'DISCOUNT_RATE': 0.08,               # 8% WACC (typical green H2 project)
        'INFLATION_RATE': 0.025,             # 2.5% annual inflation
        
        # Electricity cost - German renewable PPA (2024-2025)
        'LCOE_ELECTRICITY_EUR_PER_KWH': 0.070,  # Renewable PPA (conservative: €0.065-0.080)
        'GRID_CONNECTION_EUR_PER_KW': 40.0,    # Grid connection for backup/export
        
        # =========================================================================
        # CURTAILMENT PENALTY (Opportunity cost of wasted renewable energy)
        # =========================================================================
        'CURTAILMENT_PENALTY_EUR_PER_MWH': 25.0,  # €25/MWh (50% of electricity cost)
        'ENABLE_CURTAILMENT_PENALTY': False,      # DISABLED: standard LCOH excludes opportunity costs
        
        # H2 reference
        'H2_LHV_KWH_PER_KG': 33.33,          # Lower heating value
        'H2_HHV_KWH_PER_KG': 39.41,          # Higher heating value (for comparison)
        
        # Stack lifetime and replacement - CONSERVATIVE
        'LEARNING_RATE': 0.03,               # 3% annual cost reduction (conservative for mature tech)
        
        # =========================================================================
        # COMPONENT-LEVEL REPLACEMENT COSTS (Literature-based)
        # =========================================================================
        # PEM stack components degrade at different rates. When replacement is
        # triggered (by voltage threshold or hours), all components are replaced
        # together as a "stack refurbishment" for practical/economic reasons.
        #
        # REFERENCES:
        # - Ayers et al. (2019), ECS Trans. 92(8): 15-28 - PEM component costs
        # - IRENA (2020), Green Hydrogen Cost Reduction - Stack cost breakdown
        # - Feng et al. (2017), J. Power Sources 366: 33-55 - Degradation mechanisms
        # - Babic et al. (2017), J. Electrochem. Soc. 164: F387-F399 - Catalyst
        # - Chandesris et al. (2015), Int. J. Hydrogen Energy 40: 1353-1366 - Membrane
        # 
        # STACK COMPONENT BREAKDOWN (% of stack cost):
        # -----------------------------------------------
        # | Component          | % Stack | Degradation Mode           |
        # |--------------------|---------|----------------------------|
        # | MEA (Membrane+CL)  | 45-55%  | Chemical + mechanical      |
        # | Bipolar Plates     | 25-30%  | Corrosion, passivation     |
        # | PTL/GDL            | 10-15%  | Compaction, oxidation      |
        # | Endplates/Seals    | 5-10%   | Creep, seal degradation    |
        # -----------------------------------------------
        # 
        # REPLACEMENT STRATEGY:
        # Full stack refurbishment replaces MEA, PTL, seals (60-70% of stack cost)
        # Bipolar plates often reused if not corroded (save 25-30%)
        # =========================================================================
        
        'REPLACEMENT_FRAC': 0.55,            # Stack refurbishment = 55% of original stack cost
                                              # (MEA: 50% + PTL: 12% + seals: 5% = 67%, minus 
                                              # reuse savings = ~55%, Ayers 2019, IRENA 2020)
        
        # Component-specific degradation tracking (for detailed analysis)
        'MEA_REPLACEMENT_FRAC': 0.50,        # MEA is ~50% of stack cost
        'PTL_REPLACEMENT_FRAC': 0.12,        # PTL/GDL is ~12% of stack cost
        'SEAL_REPLACEMENT_FRAC': 0.05,       # Seals/gaskets ~5% of stack cost
        'BIPOLAR_PLATE_REUSE': True,         # Bipolar plates reused (saves ~30%)
        
        # Profitability analysis (optional)
        'H2_SELLING_PRICE_EUR_PER_KG': None,  # Set to enable profitability metrics (e.g., 12.0)
        
        # =========================================================================
        # BYPRODUCT CREDITS (Harmonized with Alkaline model for fair comparison)
        # =========================================================================
        
        # Oxygen credit: 8 kg O2 per 1 kg H2 (stoichiometry)
        # Market: industrial O2 €30-80/tonne → conservative €50/tonne
        # Net credit: 8 × 0.995 × €50/1000 = €0.40/kg H2
        # Reference: Bertuccioli et al. (2014), IRENA (2020), Air Liquide
        'OXYGEN_CREDIT_ENABLED': True,        # ENABLED: base case includes O2 revenue
        'OXYGEN_CREDIT_EUR_PER_KG_H2': 0.40,  # 8 × 0.995 × €50/t (harmonized with Alkaline)
        
        # Heat credit: low-grade waste heat (60-80°C water)
        # Suitable for district heating, preheating, greenhouses
        # Realistic recovery: 70% of waste heat at €0.03/kWh
        'HEAT_CREDIT_EUR_PER_KWH': 0.03,      # District heating rate
        
        # =========================================================================
        # WATER TREATMENT AND SITE COSTS
        # =========================================================================
        'WATER_CONSUMPTION_L_PER_KG_H2': 10.0,  # Deionized water (stoich: 9L + losses)
        'WATER_COST_EUR_PER_M3': 2.50,          # Industrial deionized water
        'WATER_TREATMENT_CAPEX_EUR_PER_KW': 30.0,  # RO + deionization system (harmonized with ALK)
        
        # =========================================================================
        # LAND AND SITE COSTS (Project-level costs)
        # =========================================================================
        'SITE_PREPARATION_EUR_PER_KW': 30.0,    # Civil works, foundations (harmonized with ALK)
        'LAND_LEASE_EUR_PER_KW_PER_YEAR': 5.0,  # Annual land lease
        'CONTINGENCY_FRAC': 0.10,               # 10% project contingency
        
        # =========================================================================
        # GRID CONNECTION (Even for off-grid - backup/export capability)
        # =========================================================================
        'GRID_CONNECTION_CAPEX_EUR_PER_KW': 35.0,  # Grid infrastructure (harmonized with ALK)
        'GRID_CONNECTION_ENABLED': True,           # Include in CAPEX even if off-grid
    }


# =============================================================================
# UNCERTAINTY PARAMETERS (REALISTIC RANGES)
# =============================================================================

# Define uncertainty ranges for key parameters (mean, std or min/max)
# Format: (distribution_type, param1, param2)
# 'normal': (mean, std), 'uniform': (min, max), 'triangular': (min, mode, max)
# Aligned with IRENA 2024, IEA 2024, BloombergNEF data

UNCERTAINTY_PARAMS = {
    # =========================================================================
    # ELECTROCHEMISTRY UNCERTAINTIES
    # =========================================================================
    'E_REV': ('normal', 1.229, 0.015),          # ±1.5% uncertainty (thermodynamic)
    'B_TAFEL_ANODE': ('triangular', 0.050, 0.060, 0.080),  # OER catalyst variability
    'B_TAFEL_CATHODE': ('triangular', 0.025, 0.030, 0.040),  # HER catalyst variability
    'R_OHM': ('triangular', 0.12, 0.14, 0.20),  # Optimistic (Nafion 212), typical, pessimistic (aged)
    'J_LIMITING': ('triangular', 2.5, 3.0, 4.0),  # Mass transport limit varies with design
    
    # Faradaic efficiency uncertainty
    'ETA_F_MAX': ('triangular', 0.97, 0.99, 0.995),  # Best case to typical
    'J_CROSSOVER': ('triangular', 0.005, 0.01, 0.02),  # Membrane quality variation
    
    # =========================================================================
    # ECONOMIC UNCERTAINTIES (2024-2025 realistic ranges)
    # =========================================================================
    
    # CAPEX uncertainties - Updated to match €1,950/kW base
    'STACK_CAPEX_EUR_PER_KW': ('triangular', 800, 950, 1200),     # Stack: €800-1200/kW range
    'BOP_CAPEX_EUR_PER_KW': ('triangular', 450, 550, 700),        # BoP: €450-700/kW range
    'CAPEX_EUR_PER_KW': ('triangular', 1700, 1950, 2300),         # Total: €1,700-2,300/kW (thesis range)
    
    # Storage uncertainties (expanded range: 300-10,000 kg)
    'STORAGE_CAPEX_EUR_PER_KG': ('triangular', 650, 800, 1100),   # Type IV @ 350 bar
    
    # Operating cost uncertainties
    'LCOE_ELECTRICITY_EUR_PER_KWH': ('triangular', 0.045, 0.070, 0.10),  # Renewable PPA range (location-dependent)
    'DISCOUNT_RATE': ('triangular', 0.06, 0.08, 0.12),            # Project risk-dependent WACC
    
    # =========================================================================
    # STACK LIFETIME RANGE (Key driver for LCOH uncertainty)
    # Range: 60,000 - 80,000 hours (modern PEM, IRENA 2024 targets)
    # This creates LCOH range in results based on lifetime variation
    # =========================================================================
    'STACK_LIFE_HOURS': ('triangular', 60000, 70000, 80000),      # 60k-80k hours (thesis requirement)
    'STACK_LIFETIME_HOURS': ('triangular', 60000, 70000, 80000),  # Alias for consistency
    
    # Efficiency uncertainties
    'COMPRESSOR_EFF': ('triangular', 0.68, 0.75, 0.82),           # Multi-stage compressor range
    'RECTIFIER_EFF': ('triangular', 0.93, 0.95, 0.97),            # Power electronics quality
    
    # =========================================================================
    # DEGRADATION UNCERTAINTIES (Field data limited) - CONSERVATIVE
    # =========================================================================
    'R_DEG_BASE': ('triangular', 2.0e-6, 3.0e-6, 5.5e-6),  # μV/h degradation rate
    'DELTA_V_CYCLE': ('triangular', 15e-6, 20e-6, 35e-6),  # Cycling damage per start/stop
    'V_THRESHOLD_INCREASE': ('triangular', 0.08, 0.12, 0.20),  # EOL voltage threshold
    
    # =========================================================================
    # RENEWABLE POWER & DEMAND UNCERTAINTIES
    # For deterministic mode, set USE_DETERMINISTIC_DATA=True in config
    # =========================================================================
    'POWER_VARIATION': ('normal', 1.0, 0.08),    # ±8% annual renewable variation
    'DEMAND_VARIATION': ('normal', 1.0, 0.05),   # ±5% annual demand variation
    
    # Weather/resource uncertainty (inter-annual variability)
    'CAPACITY_FACTOR_VARIATION': ('normal', 1.0, 0.10),  # ±10% CF variation year-to-year
}

# =============================================================================
# STACK LIFETIME SCENARIOS (For LCOH Range Calculation)
# =============================================================================
STACK_LIFETIME_SCENARIOS = {
    'conservative': 60000,  # 60,000 hours - more replacements, higher LCOH
    'baseline': 70000,      # 70,000 hours - typical modern PEM
    'optimistic': 80000,    # 80,000 hours - best case, fewer replacements, lower LCOH
}


def sample_uncertain_param(param_name, rng):
    """
    Sample a value from the uncertainty distribution for a parameter.
    
    Parameters
    ----------
    param_name : str
        Name of the parameter
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    float
        Sampled parameter value
    """
    if param_name not in UNCERTAINTY_PARAMS:
        return None
    
    dist_type, *params = UNCERTAINTY_PARAMS[param_name]
    
    if dist_type == 'normal':
        mean, std = params
        return rng.normal(mean, std)
    elif dist_type == 'uniform':
        low, high = params
        return rng.uniform(low, high)
    elif dist_type == 'triangular':
        left, mode, right = params
        return rng.triangular(left, mode, right)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def get_config_with_uncertainty(size_mw=20.0, storage_kg=2500.0, rng=None):
    """
    Return configuration with sampled uncertain parameters.
    
    Parameters
    ----------
    size_mw : float
        Electrolyser size in MW
    storage_kg : float
        Storage capacity in kg
    rng : np.random.Generator, optional
        Random number generator. If None, returns deterministic config.
        
    Returns
    -------
    dict
        Configuration dictionary with sampled parameters
    """
    cfg = get_config(size_mw, storage_kg)
    
    if rng is None:
        return cfg
    
    # Sample uncertain parameters
    for param_name in UNCERTAINTY_PARAMS:
        if param_name in cfg:
            cfg[param_name] = sample_uncertain_param(param_name, rng)
    
    # Update derived parameters
    cfg['ELECTROLYSER_SIZE_KW'] = cfg['ELECTROLYSER_SIZE_MW'] * 1000.0
    
    return cfg


# =============================================================================
# ENHANCED ELECTROCHEMISTRY (Realistic Physics)
# Sources: Carmo et al. (2013), Ursua et al. (2012), Trinke et al. (2018),
#          Schalenbach et al. (2016), Buttler & Spliethoff (2018)
# =============================================================================

def get_temperature_adjusted_params(cfg):
    """
    Calculate temperature-adjusted electrochemistry parameters.
    
    E_rev decreases with temperature (thermodynamics):
        E_rev(T) = E_rev_std + dE/dT * (T - T_ref)
        where dE/dT ≈ -0.85 mV/K for water electrolysis
        
    R_ohm decreases with temperature (membrane conductivity improves):
        R_ohm(T) = R_ohm_ref * (1 + coeff * (T - T_ref))
        
    Reference: Carmo et al. (2013), Buttler & Spliethoff (2018)
    
    Returns
    -------
    tuple
        (E_rev_T, R_ohm_T) - temperature-adjusted values
    """
    T_op = cfg['T_OPERATING_C']
    T_ref = cfg['T_REF_C']
    
    # Temperature-adjusted reversible voltage (decreases with T)
    # E_rev(T) = 1.229 - 0.00085 * (T - 25)
    E_rev_T = cfg['E_REV_STD'] + cfg['E_REV_TEMP_COEFF'] * (T_op - T_ref)
    
    # Temperature-adjusted ohmic resistance (decreases with T)
    # Membrane conductivity increases with temperature
    T_ref_R = cfg['R_OHM_REF_TEMP_C']
    R_ohm_T = cfg['R_OHM'] * (1.0 + cfg['R_OHM_TEMP_COEFF'] * (T_op - T_ref_R))
    R_ohm_T = max(R_ohm_T, 0.05)  # Minimum physical limit
    
    return E_rev_T, R_ohm_T


def partial_load_efficiency_factor(load_fraction, cfg):
    """
    Calculate efficiency correction factor for partial load operation.
    
    At low loads (<30%), parasitic losses (pumps, controls, heat loss)
    become a larger fraction of total power, reducing effective efficiency.
    
    Model: Linear interpolation below threshold
        Above threshold: factor = 1.0
        Below threshold: factor decreases linearly to min value
        
    Reference: Buttler & Spliethoff (2018), IRENA (2020)
    
    Parameters
    ----------
    load_fraction : float
        Current load as fraction of nominal (0 to 1+)
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    float
        Efficiency factor (0.7 to 1.0)
    """
    threshold = cfg['PARTIAL_LOAD_THRESHOLD']
    eff_min = cfg['PARTIAL_LOAD_EFF_MIN']
    
    if load_fraction >= threshold:
        return 1.0
    elif load_fraction <= 0:
        return eff_min
    else:
        # Linear interpolation from eff_min at 0% to 1.0 at threshold
        return eff_min + (1.0 - eff_min) * (load_fraction / threshold)


def cell_voltage_bol(j, cfg):
    """
    ENHANCED Beginning-of-life cell voltage with comprehensive physics.
    
    Full polarization model including:
    1. Reversible voltage (temperature and pressure corrected)
    2. Activation overpotential (dual Tafel for anode + cathode)
    3. Ohmic overpotential (membrane + contact resistance)
    4. Concentration overpotential (mass transport limitation)
    
    V = E_rev(T,P) + η_act_anode + η_act_cathode + η_ohm + η_conc
    
    Sources: Carmo et al. (2013), Ursua et al. (2012), Trinke et al. (2018)
    
    Parameters
    ----------
    j : float
        Current density (A/cm²)
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    float
        Cell voltage (V)
    """
    if j <= 0:
        return 0.0
    
    # ==========================================================================
    # 1. REVERSIBLE VOLTAGE (Temperature + Pressure corrected)
    # ==========================================================================
    E_rev_T, R_ohm_T = get_temperature_adjusted_params(cfg)
    
    # Pressure correction (Nernst equation): increases E_rev at elevated pressure
    # ΔE_pressure = (RT/4F) × ln(P_O2 × P_H2²) ≈ 0.0295V per decade of pressure
    P_op = cfg.get('OPERATING_PRESSURE_BAR', 1.0)
    P_ref = 1.0  # Reference pressure (bar)
    if P_op > P_ref:
        pressure_coeff = cfg.get('E_REV_PRESSURE_COEFF', 0.0295)
        E_pressure_correction = pressure_coeff * np.log10(P_op / P_ref)
    else:
        E_pressure_correction = 0.0
    
    E_rev = E_rev_T + E_pressure_correction
    
    # ==========================================================================
    # 2. ACTIVATION OVERPOTENTIAL (Dual Tafel model for HER + OER)
    # ==========================================================================
    use_dual_tafel = cfg.get('USE_DUAL_TAFEL', False)
    
    if use_dual_tafel:
        # Anode (OER): η_act,a = B_a × ln(j / j0_a) for j >> j0
        # OER is the rate-limiting reaction (slower kinetics)
        B_anode = cfg.get('B_TAFEL_ANODE', 0.060)
        j0_anode = cfg.get('J0_ANODE', 1.0e-7)
        eta_act_anode = B_anode * np.log(j / j0_anode) if j > j0_anode else 0.0
        eta_act_anode = max(eta_act_anode, 0.0)
        
        # Cathode (HER): η_act,c = B_c × ln(j / j0_c)
        # HER is faster with Pt catalyst
        B_cathode = cfg.get('B_TAFEL_CATHODE', 0.030)
        j0_cathode = cfg.get('J0_CATHODE', 1.0e-3)
        eta_act_cathode = B_cathode * np.log(j / j0_cathode) if j > j0_cathode else 0.0
        eta_act_cathode = max(eta_act_cathode, 0.0)
        
        eta_act_total = eta_act_anode + eta_act_cathode
    else:
        # Single effective Tafel (backward compatibility)
        eta_act_total = cfg['B_TAFEL'] * np.log(1.0 + j / cfg['J0'])
    
    # ==========================================================================
    # 3. OHMIC OVERPOTENTIAL
    # ==========================================================================
    eta_ohm = R_ohm_T * j
    
    # ==========================================================================
    # 4. CONCENTRATION OVERPOTENTIAL (Mass transport limitation)
    # ==========================================================================
    enable_conc = cfg.get('ENABLE_CONCENTRATION_OVERPOTENTIAL', False)
    
    if enable_conc and j > 0.1:  # Only significant at moderate to high j
        j_lim = cfg.get('J_LIMITING', 3.0)
        C_conc = cfg.get('C_CONC', 0.10)
        m_conc = cfg.get('M_CONC', 3.0)
        
        # Concentration overpotential increases exponentially as j → j_lim
        # η_conc = C × (j / j_lim)^m / (1 - j/j_lim)
        if j < 0.95 * j_lim:
            ratio = j / j_lim
            eta_conc = C_conc * (ratio ** m_conc) / (1.0 - ratio)
            eta_conc = min(eta_conc, 0.5)  # Cap at reasonable value
        else:
            eta_conc = 0.5  # Limiting case
    else:
        eta_conc = 0.0
    
    # ==========================================================================
    # TOTAL CELL VOLTAGE
    # ==========================================================================
    V_cell = E_rev + eta_act_total + eta_ohm + eta_conc
    
    # Physical bounds (literature: 1.4V minimum practical, 2.5V maximum safe)
    V_cell = np.clip(V_cell, 1.4, 2.5)
    
    return V_cell


def compute_faradaic_efficiency(j, cfg):
    """
    Calculate variable Faradaic efficiency as function of current density.
    
    At low current density, Faradaic efficiency decreases due to:
    - H2 crossover through membrane (permeation to anode → recombination)
    - Parasitic side reactions
    - Gas mixing losses
    
    Model (Schalenbach et al. 2016):
        η_F(j) = η_F_max × (1 - j_crossover / j)
        
    At high j: η_F → η_F_max (crossover current negligible)
    At low j: η_F drops significantly
    
    Parameters
    ----------
    j : float
        Current density (A/cm²)
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    float
        Faradaic efficiency (0 to 1)
    """
    if not cfg.get('ENABLE_VARIABLE_FARADAIC_EFF', False):
        return cfg.get('ETA_F', 0.97)
    
    if j <= 0:
        return 0.0
    
    eta_f_max = cfg.get('ETA_F_MAX', 0.99)
    eta_f_min = cfg.get('ETA_F_MIN', 0.85)
    j_crossover = cfg.get('J_CROSSOVER', 0.01)
    
    # Crossover model: η_F = η_max × (1 - j_cross/j)
    if j > j_crossover:
        eta_f = eta_f_max * (1.0 - j_crossover / j)
    else:
        eta_f = eta_f_min
    
    # Bound to realistic range
    return np.clip(eta_f, eta_f_min, eta_f_max)


def compute_stack_count(cfg):
    """
    Compute number of stacks needed for target electrolyser size.
    
    For a typical commercial PEM stack:
    - Cell area: 500-3000 cm²
    - Current density: 1-2 A/cm²
    - Cell voltage: 1.7-2.0 V
    - Cells per stack: 50-400
    - Stack power: 0.5-5 MW
    
    Returns
    -------
    tuple
        (n_stacks, p_stack_nom_kW)
    """
    v_nom = cell_voltage_bol(cfg['J_NOMINAL'], cfg)
    
    # Current per cell: I = j × A (no unit conversion needed, both in A and cm²→A/cm²)
    i_cell_nom = cfg['J_NOMINAL'] * cfg['CELL_AREA_CM2']  # A (e.g., 1.5 A/cm² × 3000 cm² = 4500 A)
    
    # Power per stack: P = I × V × n_cells
    p_stack_nom_kW = i_cell_nom * v_nom * cfg['N_CELLS'] / 1000.0  # kW
    
    # Number of stacks needed
    n_stacks = cfg['ELECTROLYSER_SIZE_KW'] / p_stack_nom_kW
    
    return n_stacks, p_stack_nom_kW


# =============================================================================
# THERMAL MANAGEMENT MODEL
# Sources: Olivier et al. (2017), Trinke et al. (2018), Scheepers et al. (2022)
# =============================================================================

def compute_heat_generation(v_cell, i_total, cfg):
    """
    Calculate heat generation from electrochemical overpotential losses.
    
    Heat is generated when V_cell > V_thermoneutral (1.48V).
    Below thermoneutral, the reaction is endothermic and absorbs heat.
    
    Q_gen = I × (V_cell - V_thermo) [kW]
    
    Parameters
    ----------
    v_cell : float
        Cell voltage (V)
    i_total : float
        Total current (A)
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    float
        Heat generation rate (kW), positive = exothermic
    """
    if v_cell <= 0 or i_total <= 0:
        return 0.0
    
    V_thermo = cfg['V_THERMONEUTRAL']  # 1.48 V
    
    # Heat generated per cell, then multiply by number of cells
    q_gen_kW = i_total * (v_cell - V_thermo) / 1000.0
    
    return max(q_gen_kW, 0.0)  # Only positive heat generation


def compute_cooling_power(T_stack, cfg, size_mw, Q_gen=None):
    """
    Calculate cooling power with proportional control to maintain target temperature.
    
    Uses a realistic cooling control strategy:
    - No cooling below target temperature (allow warmup)
    - Proportional cooling above target (prevent overcooling)
    - Maximum cooling only when approaching T_MAX
    
    Parameters
    ----------
    T_stack : float
        Current stack temperature (°C)
    cfg : dict
        Configuration dictionary
    size_mw : float
        Electrolyser size (MW)
    Q_gen : float, optional
        Current heat generation (kW) - used for proportional control
        
    Returns
    -------
    tuple
        (Q_cool_kW, cooling_energy_kWh) - cooling power and electrical energy for cooling
    """
    m_dot = cfg['COOLANT_FLOW_KG_PER_S_PER_MW'] * size_mw  # kg/s
    Cp = cfg['COOLANT_CP_KJ_PER_KG_K']  # kJ/(kg·K)
    T_in = cfg['COOLANT_INLET_TEMP_C']
    effectiveness = cfg['HEAT_TRANSFER_COEFF']
    T_target = cfg['T_OPERATING_C']  # 80°C
    T_max = cfg['T_MAX_OPERATING_C']  # 90°C
    
    # Maximum heat that can be removed at current temperature
    Q_cool_max = m_dot * Cp * (T_stack - T_in) * effectiveness  # kW
    Q_cool_max = max(Q_cool_max, 0.0)
    
    # PROPORTIONAL COOLING CONTROL:
    # - Below target temperature: minimal/no cooling to allow warmup
    # - At target temperature: match Q_gen to maintain steady state
    # - Above target: increase cooling to bring temperature down
    # - Near T_max: maximum cooling
    
    if T_stack <= T_target - 5.0:
        # Well below target: no active cooling (allow natural warmup)
        # Only passive heat loss considered
        Q_cool = 0.0
    elif T_stack <= T_target:
        # Approaching target: start gentle cooling
        # Proportional: 0% at T_target-5, ~50% at T_target
        warmup_fraction = (T_stack - (T_target - 5.0)) / 5.0  # 0 to 1
        if Q_gen is not None and Q_gen > 0:
            # Match a fraction of heat generation
            Q_cool = Q_gen * warmup_fraction * 0.5
        else:
            Q_cool = Q_cool_max * warmup_fraction * 0.2
    elif T_stack <= T_max:
        # Above target: proportional cooling to stabilize at target
        # At T_target: Q_cool = Q_gen (steady state)
        # At T_max: Q_cool = Q_cool_max (maximum)
        overheat_fraction = (T_stack - T_target) / (T_max - T_target)  # 0 to 1
        if Q_gen is not None and Q_gen > 0:
            # Between matching Q_gen and Q_cool_max
            Q_cool = Q_gen + overheat_fraction * (Q_cool_max - Q_gen)
        else:
            Q_cool = Q_cool_max * (0.5 + 0.5 * overheat_fraction)
    else:
        # At or above T_max: maximum cooling
        Q_cool = Q_cool_max
    
    # Ensure Q_cool doesn't exceed maximum capacity
    Q_cool = min(Q_cool, Q_cool_max)
    Q_cool = max(Q_cool, 0.0)
    
    # Cooling pump electrical consumption (~3-5% of cooling capacity)
    cooling_pump_fraction = 0.04
    cooling_elec_kW = Q_cool * cooling_pump_fraction
    
    return Q_cool, cooling_elec_kW


def update_stack_temperature(T_stack_prev, Q_gen, Q_cool, cfg, size_mw, dt_hours=1.0):
    """
    Update stack temperature based on thermal balance.
    
    dT/dt = (Q_gen - Q_cool) / (m × Cp)
    
    Parameters
    ----------
    T_stack_prev : float
        Previous stack temperature (°C)
    Q_gen : float
        Heat generation (kW)
    Q_cool : float
        Cooling power (kW)
    cfg : dict
        Configuration dictionary
    size_mw : float
        Electrolyser size (MW)
    dt_hours : float
        Time step (hours)
        
    Returns
    -------
    float
        New stack temperature (°C)
    """
    # Thermal mass (kJ/K)
    thermal_mass = cfg['STACK_THERMAL_MASS_KJ_PER_K'] * size_mw
    
    # Net heat (kW = kJ/s)
    Q_net = Q_gen - Q_cool
    
    # Temperature change: dT = Q_net × dt / thermal_mass
    # Convert hours to seconds for consistency
    dt_seconds = dt_hours * 3600.0
    dT = (Q_net * dt_seconds) / thermal_mass
    
    # Update temperature with limits
    T_new = T_stack_prev + dT
    T_new = np.clip(T_new, cfg['T_AMBIENT_C'], cfg['T_MAX_OPERATING_C'])
    
    return T_new


def compute_cold_start_penalty(T_stack, cfg, hours_since_shutdown):
    """
    Calculate efficiency penalty during cold start.
    
    When stack is below operating temperature, efficiency is reduced.
    Penalty decreases linearly as stack warms up.
    
    Parameters
    ----------
    T_stack : float
        Current stack temperature (°C)
    cfg : dict
        Configuration dictionary
    hours_since_shutdown : float
        Hours since last operation (for warmup tracking)
        
    Returns
    -------
    float
        Efficiency multiplier (0.85 to 1.0)
    """
    T_min = cfg['T_MIN_OPERATING_C']  # 50°C
    T_target = cfg['T_OPERATING_C']   # 80°C
    max_penalty = cfg['COLD_START_EFFICIENCY_PENALTY']  # 0.15
    
    if T_stack >= T_target:
        return 1.0  # No penalty at operating temperature
    elif T_stack <= T_min:
        return 1.0 - max_penalty  # Maximum penalty
    else:
        # Linear interpolation
        warmup_fraction = (T_stack - T_min) / (T_target - T_min)
        return 1.0 - max_penalty * (1.0 - warmup_fraction)


def compute_heat_recovery(Q_gen, cfg):
    """
    Calculate recoverable heat and its economic value.
    
    Waste heat can be used for district heating, process heat, etc.
    
    Parameters
    ----------
    Q_gen : float
        Heat generation (kW)
    cfg : dict
        Configuration dictionary
        
    Returns
    -------
    tuple
        (Q_recovered_kW, heat_revenue_EUR_per_hour)
    """
    if not cfg.get('HEAT_RECOVERY_ENABLED', False) or Q_gen <= 0:
        return 0.0, 0.0
    
    eta_recovery = cfg['HEAT_RECOVERY_EFFICIENCY']  # 0.70
    heat_value = cfg['HEAT_VALUE_EUR_PER_KWH']      # 0.03 EUR/kWh
    
    Q_recovered = Q_gen * eta_recovery
    revenue = Q_recovered * heat_value
    
    return Q_recovered, revenue


# =============================================================================
# SIMULATION
# =============================================================================

def simulate(cfg, power_1yr, demand_1yr, rng=None):
    """
    Run multi-year PEM electrolyser simulation with thermal management.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    power_1yr : pd.Series
        1-year power availability (kW)

    demand_1yr : pd.Series
        1-year H2 demand (kWh)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Hourly simulation results
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    
    n_stacks, _ = compute_stack_count(cfg)
    n_hours = cfg['YEARS'] * cfg['HOURS_PER_YEAR']
    
    # Tile 1-year data to 5 years
    power_5yr = np.tile(power_1yr.values, cfg['YEARS'])[:n_hours]
    demand_5yr = np.tile(demand_1yr.values, cfg['YEARS'])[:n_hours]
    demand_5yr_kg = demand_5yr / cfg['H2_LHV_KWH_PER_KG']
    
    index = pd.date_range(start="2020-01-01", periods=n_hours, freq="h")
    
    # Initialize arrays and state variables
    storage_kg = 0.0
    delta_V_deg = 0.0  # Accumulated voltage degradation (V) - replaces 'damage'
    n_cycles = 0       # Number of start/stop cycles
    stack_replacement_years = []
    stack_replacement_timestamps = []  # Store exact timestamps for plotting
    minor_maintenance_hours = []
    next_minor_maint = rng.integers(2000, 4000)
    op_hours = 0.0
    prev_frac = 0.0
    prev_power = 0.0
    
    # Output arrays
    curtailed_arr = np.zeros(n_hours)
    storage_arr = np.zeros(n_hours)
    unmet_arr = np.zeros(n_hours)
    h2_prod_arr = np.zeros(n_hours)
    vcell_arr = np.zeros(n_hours)
    sec_stack_arr = np.zeros(n_hours)
    sec_bop_arr = np.zeros(n_hours)
    sec_total_arr = np.zeros(n_hours)
    stack_eff_arr = np.zeros(n_hours)
    system_eff_arr = np.zeros(n_hours)
    
    # Thermal management arrays (NEW)
    T_stack_arr = np.zeros(n_hours)
    Q_gen_arr = np.zeros(n_hours)
    Q_cool_arr = np.zeros(n_hours)
    Q_recovered_arr = np.zeros(n_hours)
    cooling_power_arr = np.zeros(n_hours)
    heat_revenue_arr = np.zeros(n_hours)
    
    # Gas constant for Arrhenius calculation
    R_gas = 8.314  # J/(mol·K)
    
    # Voltage threshold for stack replacement
    V_BOL_nom = cell_voltage_bol(cfg['J_NOMINAL'], cfg)
    V_threshold = V_BOL_nom + cfg['V_THRESHOLD_INCREASE']
    
    # Thermal state initialization (NEW - dynamic temperature)
    T_stack = cfg['T_AMBIENT_C']  # Start at ambient (cold start)
    T_target = cfg['T_OPERATING_C']
    T_mean = cfg['T_OPERATING_C']  # For voltage temp coefficient reference
    V_temp_coeff = -0.0015  # V/°C (reversible temperature effect on voltage)
    hours_since_shutdown = 0  # Track for cold start penalty
    size_mw = cfg['ELECTROLYSER_SIZE_MW']
    
    # Operating constraints
    max_load = cfg['ELECTROLYSER_SIZE_KW']
    min_load = cfg['MIN_LOAD_FRAC'] * max_load
    ramp_rate = cfg['MAX_RAMP_FRAC_PER_H'] * max_load
    
    for i in range(n_hours):
        available_power = power_5yr[i]
        demand_kg = demand_5yr_kg[i]
        
        # Ramping constraint
        ramped_power = np.clip(available_power, prev_power - ramp_rate, prev_power + ramp_rate)
        
        # Minimum load constraint
        if ramped_power < min_load:
            used_power = 0.0
            frac = 0.0
            curtailed_power = available_power
        else:
            used_power = min(ramped_power, max_load)
            frac = used_power / max_load
            curtailed_power = available_power - used_power
        
        prev_power = used_power
        curtailed_arr[i] = curtailed_power
        
        # Track operating hours and shutdown tracking for cold start
        if frac > 0.0:
            op_hours += 1
            hours_since_shutdown = 0
        else:
            hours_since_shutdown += 1
        
        # Minor maintenance scheduling
        if op_hours >= next_minor_maint:
            minor_maintenance_hours.append(index[i])
            op_hours = 0.0
            next_minor_maint = rng.integers(2000, 4000)
        
        # =====================================================================
        # DYNAMIC THERMAL MANAGEMENT MODEL (NEW)
        # Sources: Olivier et al. (2017), Trinke et al. (2018), Scheepers et al. (2022)
        # =====================================================================
        
        # During operation: calculate heat generation and update temperature
        if frac > 0.0:
            # Estimate current and voltage for heat calculation
            j_est = frac * cfg['J_NOMINAL']
            v_cell_est = cell_voltage_bol(j_est, cfg) + delta_V_deg
            i_cell_est = j_est * cfg['CELL_AREA_CM2']  # A (j × A)
            i_total_est = i_cell_est * cfg['N_CELLS'] * n_stacks
            
            # Heat generation from overpotential losses
            Q_gen = compute_heat_generation(v_cell_est, i_total_est, cfg)
            
            # Cooling power and electrical consumption (pass Q_gen for proportional control)
            Q_cool, cooling_elec = compute_cooling_power(T_stack, cfg, size_mw, Q_gen=Q_gen)
            
            # Update stack temperature (thermal balance)
            T_stack = update_stack_temperature(T_stack, Q_gen, Q_cool, cfg, size_mw, dt_hours=1.0)
            
            # Heat recovery calculation
            Q_recovered, heat_revenue = compute_heat_recovery(Q_gen, cfg)
            
            # Cold start efficiency penalty
            cold_start_factor = compute_cold_start_penalty(T_stack, cfg, hours_since_shutdown)
        else:
            # Standby: stack cools down toward ambient (or maintains hot standby)
            Q_gen = 0.0
            cooling_elec = 0.0
            
            # Hot standby: maintain temperature with small heat input
            if cfg.get('HOT_STANDBY_POWER_FRAC', 0) > 0:
                # Maintain standby temperature
                T_standby = cfg.get('STANDBY_TEMP_C', 60.0)
                T_stack = max(T_stack - 0.5, T_standby)  # Slow cooling to standby temp
            else:
                # Cool down toward ambient
                T_stack = T_stack - 0.5 * (T_stack - cfg['T_AMBIENT_C']) / 10.0
            
            Q_cool = 0.0
            Q_recovered = 0.0
            heat_revenue = 0.0
            cold_start_factor = 1.0
        
        # Store thermal values
        T_stack_arr[i] = T_stack
        Q_gen_arr[i] = Q_gen
        Q_cool_arr[i] = Q_cool
        Q_recovered_arr[i] = Q_recovered
        cooling_power_arr[i] = cooling_elec if frac > 0 else 0.0
        heat_revenue_arr[i] = heat_revenue
        
        # Use dynamic temperature for calculations
        T = T_stack
        T_K = T + 273.15  # Convert to Kelvin
        
        # =====================================================================
        # DEGRADATION MODEL (VALIDATED - Literature-based)
        # Sources: Frensch 2019, Rakousky 2017, Weiß 2019, Chandesris 2015
        # =====================================================================
        
        # 1. Cycling damage: discrete penalty per start/stop cycle
        #    Weiß 2019: ~15 μV per cycle for hot standby operation
        if prev_frac == 0.0 and frac > 0.0:
            n_cycles += 1
            delta_V_deg += cfg['DELTA_V_CYCLE']
        
        # 2. Time-based irreversible degradation (LINEAR, not sigmoidal)
        #    Only accumulates during operation
        if frac > 0.0:
            # Load factor: j^1.5 relationship (Frensch 2019)
            f_load = frac ** cfg['DEG_LOAD_EXPONENT']
            
            # Temperature factor: Arrhenius-derived
            # f_temp = exp(E_a/R × (1/T_ref - 1/T))
            # E_a ≈ 50 kJ/mol for membrane degradation (Chandesris 2015)
            f_temp = np.exp(cfg['DEG_E_ACT'] / R_gas * (1.0/cfg['T_DEG_REF'] - 1.0/T_K))
            
            # Hourly irreversible degradation
            # r_deg = r_base × f_load × f_temp
            delta_V_deg += cfg['R_DEG_BASE'] * f_load * f_temp
        
        prev_frac = frac
        
        # Production calculations
        if frac > 0.0:
            j = frac * cfg['J_NOMINAL']
            
            # Degraded cell voltage (LINEAR degradation model)
            # V_cell = V_BOL(j,T) + ΔV_degradation + ΔV_temp_reversible
            v_cell = (cell_voltage_bol(j, cfg) 
                      + delta_V_deg 
                      + V_temp_coeff * (T - T_mean))
            v_cell = np.clip(v_cell, 1.5, 2.5)  # Physical bounds
            
            # Current and H2 production
            # Current per cell: I = j × A (A/cm² × cm² = A)
            i_cell = j * cfg['CELL_AREA_CM2']  # A
            i_total = i_cell * cfg['N_CELLS'] * n_stacks
            
            # Apply partial load efficiency factor AND cold start penalty
            # At low loads, parasitic losses reduce effective H2 production
            pl_eff = partial_load_efficiency_factor(frac, cfg)
            thermal_eff = cold_start_factor  # From thermal management model
            combined_eff = pl_eff * thermal_eff
            
            # VARIABLE FARADAIC EFFICIENCY (new physics)
            # At low j, H2 crossover through membrane reduces efficiency
            eta_f = compute_faradaic_efficiency(j, cfg)
            
            m_dot = (i_total / (2.0 * cfg['FARADAY'])) * cfg['M_H2'] * eta_f * combined_eff
            h2_kg = m_dot * 3600.0
            
            # Power calculations (include cooling energy)
            p_dc_kW = i_total * v_cell / 1000.0
            p_ac_elec_kW = p_dc_kW / cfg['RECTIFIER_EFF']
            p_cooling_kW = cooling_power_arr[i]  # Cooling pump energy
            
            # =================================================================
            # STACK REPLACEMENT CHECK (Dual criteria - aligned with Alkaline)
            # =================================================================
            # Modern PEM replacement criteria (IRENA 2024, DOE targets):
            # 1. Voltage degradation exceeds threshold (~9% = 150 mV)
            # 2. Operating hours exceed lifetime (80,000 h)
            # 
            # This dual approach ensures:
            # - Aggressive cycling doesn't lead to excessive voltage degradation
            # - Well-operated stacks reach their hour-based lifetime
            # =================================================================
            needs_replacement = False
            
            # Criterion 1: Voltage-based (degradation exceeds threshold)
            if delta_V_deg > cfg['V_THRESHOLD_INCREASE']:
                needs_replacement = True
            
            # Criterion 2: Hours-based (exceeds 80,000 h operating hours)
            if cfg.get('USE_HOURS_BASED_REPLACEMENT', True):
                if op_hours >= cfg.get('STACK_LIFETIME_HOURS', 80000):
                    needs_replacement = True
            
            if needs_replacement:
                stack_replacement_years.append(index[i].year)
                stack_replacement_timestamps.append(index[i])  # Store exact timestamp
                delta_V_deg = 0.0  # Reset degradation after stack replacement
                n_cycles = 0       # Reset cycle count
                op_hours = 0.0     # Reset operating hours
                minor_maintenance_hours.append(index[i])
        else:
            v_cell = 0.0
            h2_kg = 0.0
            p_ac_elec_kW = 0.0
            p_cooling_kW = 0.0
        
        # BoP energy (now includes explicit cooling)
        # Note: COMP_ENERGY_KWH_PER_KG (3.5 kWh/kg) already includes real compressor losses
        # per Sdanghi et al. (2019). Do NOT divide by efficiency again.
        p_parasitic_kW = cfg['PARASITIC_KWH_PER_KG'] * h2_kg  # Pumps, controls (reduced from 2.0 to 1.5)
        p_comp_kW = cfg['COMP_ENERGY_KWH_PER_KG'] * h2_kg
        p_total_kW = p_ac_elec_kW + p_parasitic_kW + p_comp_kW + p_cooling_kW
        
        # Specific Energy Consumption (SEC)
        if h2_kg > 1e-9:
            sec_stack = p_ac_elec_kW / h2_kg
            sec_bop = (p_parasitic_kW + p_comp_kW + p_cooling_kW) / h2_kg
            sec_total = p_total_kW / h2_kg
            stack_eff = cfg['H2_LHV_KWH_PER_KG'] / sec_stack
            system_eff = cfg['H2_LHV_KWH_PER_KG'] / sec_total
        else:
            sec_stack = sec_bop = sec_total = 0.0
            stack_eff = system_eff = 0.0
        
        sec_stack_arr[i] = sec_stack
        sec_bop_arr[i] = sec_bop
        sec_total_arr[i] = sec_total
        stack_eff_arr[i] = stack_eff
        system_eff_arr[i] = system_eff
        
        # Storage logic (efficiency applied on discharge only)
        storage_kg += h2_kg
        if storage_kg > cfg['STORAGE_CAPACITY_KG']:
            surplus = storage_kg - cfg['STORAGE_CAPACITY_KG']
            storage_kg = cfg['STORAGE_CAPACITY_KG']
            curtailed_arr[i] += surplus * cfg['H2_LHV_KWH_PER_KG']  # Convert surplus H2 to kWh
        
        # Delivery with roundtrip efficiency loss
        delivered = min(storage_kg * cfg['STORAGE_EFF'], demand_kg)
        storage_kg -= delivered / cfg['STORAGE_EFF']  # Account for efficiency
        storage_kg = max(storage_kg, 0.0)
        unmet = demand_kg - delivered
        
        storage_arr[i] = storage_kg
        unmet_arr[i] = unmet
        h2_prod_arr[i] = h2_kg
        vcell_arr[i] = v_cell
    
    # Build output DataFrame (including thermal management data)
    df = pd.DataFrame({
        "power_kW": power_5yr,
        "demand_kWh": demand_5yr,
        "demand_H2_kg": demand_5yr_kg,
        "H2_kg": h2_prod_arr,
        "storage_kg": storage_arr,
        "unmet_kg": unmet_arr,
        "V_cell_V": vcell_arr,
        "curtailed_power_kWh": curtailed_arr,
        "SEC_stack_kWh_per_kg": sec_stack_arr,
        "SEC_BoP_kWh_per_kg": sec_bop_arr,
        "SEC_total_kWh_per_kg": sec_total_arr,
        "stack_efficiency": stack_eff_arr,
        "system_efficiency": system_eff_arr,
        # Thermal management columns (NEW)
        "T_stack_C": T_stack_arr,
        "Q_heat_gen_kW": Q_gen_arr,
        "Q_cooling_kW": Q_cool_arr,
        "Q_recovered_kW": Q_recovered_arr,
        "cooling_power_kW": cooling_power_arr,
        "heat_revenue_EUR": heat_revenue_arr,
    }, index=index)
    
    df.attrs['stack_replacement_years'] = stack_replacement_years
    df.attrs['stack_replacement_timestamps'] = stack_replacement_timestamps  # Exact timestamps
    df.attrs['minor_maintenance_hours'] = minor_maintenance_hours
    df.attrs['total_heat_recovered_kWh'] = Q_recovered_arr.sum()  # Total heat recovery
    df.attrs['total_heat_revenue_EUR'] = heat_revenue_arr.sum()   # Total heat revenue
    
    return df


# =============================================================================
# ECONOMICS
# =============================================================================

def compute_economics(cfg, df, h2_selling_price_eur_per_kg=None):
    """
    ENHANCED LCOH and economic metrics using NPV methodology.
    
    Comprehensive cost breakdown including:
    - CAPEX: Stack, BoP, installation, engineering, water treatment, contingency
    - OPEX: Fixed (maintenance), variable (electricity, water, consumables)
    - Replacement costs with learning curve
    - Optional credits: Heat recovery, oxygen byproduct
    
    LCOH = NPV(all costs) / NPV(H2 production)
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    df : pd.DataFrame
        Simulation results
    h2_selling_price_eur_per_kg : float, optional
        H2 selling price in EUR/kg. If None, profitability metrics are not calculated.
        
    Returns
    -------
    dict
        Comprehensive economic metrics
    """
    years = cfg['YEARS']
    size_kw = cfg['ELECTROLYSER_SIZE_KW']
    
    # ==========================================================================
    # COMPREHENSIVE CAPEX BREAKDOWN
    # ==========================================================================
    
    # Core electrolyser (use detailed breakdown if available)
    capex_stack = size_kw * cfg.get('STACK_CAPEX_EUR_PER_KW', 750.0)
    capex_bop = size_kw * cfg.get('BOP_CAPEX_EUR_PER_KW', 550.0)
    capex_installation = size_kw * cfg.get('INSTALLATION_CAPEX_EUR_PER_KW', 150.0)
    capex_engineering = size_kw * cfg.get('ENGINEERING_CAPEX_EUR_PER_KW', 100.0)
    
    # If using simplified CAPEX (backward compatibility)
    if cfg.get('BOP_CAPEX_FRAC', 0) > 0:
        capex_elect = size_kw * cfg['CAPEX_EUR_PER_KW']
        capex_bop = capex_elect * cfg['BOP_CAPEX_FRAC']
        capex_stack = capex_elect - capex_bop
        capex_installation = 0
        capex_engineering = 0
    
    capex_electrolyser_total = capex_stack + capex_bop + capex_installation + capex_engineering
    
    # Storage system
    storage_kg = cfg['STORAGE_CAPACITY_KG']
    capex_storage = storage_kg * cfg['STORAGE_CAPEX_EUR_PER_KG']
    
    # Compressor (harmonized: sized by storage capacity, same as Alkaline)
    # Reference: IEA (2024), IRENA (2024) — €50-80/kg capacity
    capex_compressor_per_kg = cfg.get('COMPRESSOR_CAPEX_EUR_PER_KG_STORED', 55.0)
    capex_compressor_per_rate = cfg.get('COMPRESSOR_CAPEX_EUR_PER_KG_H2_PER_H', 0.0)
    
    if capex_compressor_per_rate > 0:
        # Legacy throughput-based model (backward compatibility)
        max_h2_rate_kg_h = df['H2_kg'].max() if 'H2_kg' in df.columns else storage_kg / 24
        capex_compressor = max(max_h2_rate_kg_h, 50) * capex_compressor_per_rate
    else:
        # Harmonized: storage-capacity-based model (matches Alkaline)
        capex_compressor = storage_kg * capex_compressor_per_kg  # Minimum 50 kg/h
    
    # Water treatment system (often overlooked)
    capex_water_treatment = size_kw * cfg.get('WATER_TREATMENT_CAPEX_EUR_PER_KW', 25.0)
    
    # Site preparation and civil works
    capex_site_prep = size_kw * cfg.get('SITE_PREPARATION_EUR_PER_KW', 20.0)
    
    # Grid connection (if applicable)
    capex_grid = size_kw * cfg.get('GRID_CONNECTION_EUR_PER_KW', 35.0)
    
    # Subtotal before contingency
    capex_subtotal = (capex_electrolyser_total + capex_storage + capex_compressor + 
                      capex_water_treatment + capex_site_prep + capex_grid)
    
    # Contingency (10% default)
    contingency_frac = cfg.get('CONTINGENCY_FRAC', 0.10)
    capex_contingency = capex_subtotal * contingency_frac
    
    # TOTAL CAPEX
    capex_total = capex_subtotal + capex_contingency
    
    # Retrieve maintenance events
    replacement_years = df.attrs.get('stack_replacement_years', [])
    replacement_timestamps = df.attrs.get('stack_replacement_timestamps', [])
    minor_maintenance_hours = df.attrs.get('minor_maintenance_hours', [])
    
    # Maintenance cost fractions (realistic values)
    MAINT_FRAC_MAJOR = 0.025   # 2.5% of CAPEX for major overhaul
    MAINT_FRAC_MINOR = 0.003   # 0.3% for minor maintenance
    DOWNTIME_HOURS_MAJOR = 72  # 3 days for stack replacement
    DOWNTIME_HOURS_MINOR = 12  # Half day for minor maintenance
    
    # Calculate annual costs and H2 production
    total_H2 = df["H2_kg"].sum()
    total_P_kWh = df["power_kW"].sum()
    annual_elec_kWh = total_P_kWh / years
    hours_per_year = cfg['HOURS_PER_YEAR']
    
    # Annual H2 production adjusted for downtime
    annual_H2 = []
    unique_years = df.index.year.unique()
    for y in unique_years:
        mask = df.index.year == y
        h2_y = df.loc[mask, "H2_kg"].sum()
        n_major = replacement_years.count(y)
        n_minor = sum([dt.year == y for dt in minor_maintenance_hours])
        downtime_y = n_major * DOWNTIME_HOURS_MAJOR + n_minor * DOWNTIME_HOURS_MINOR
        h2_y *= (hours_per_year - downtime_y) / hours_per_year
        annual_H2.append(h2_y)
    
    # ==========================================================================
    # ANNUAL OPERATING COSTS with learning curve
    # ==========================================================================
    cash_costs = []
    discount_factors = []
    electricity_costs = []
    water_costs = []
    curtailment_costs = []  # NEW: Track curtailment penalty
    
    # Calculate total curtailed energy for penalty
    total_curtailed_kWh = df['curtailed_power_kWh'].sum() if 'curtailed_power_kWh' in df.columns else 0.0
    annual_curtailed_kWh = total_curtailed_kWh / years
    curtailment_penalty_eur_per_mwh = cfg.get('CURTAILMENT_PENALTY_EUR_PER_MWH', 25.0)
    enable_curtailment_penalty = cfg.get('ENABLE_CURTAILMENT_PENALTY', True)
    
    # Apply learning rate to replacement costs
    learning_rate = cfg.get('LEARNING_RATE', 0.03)
    replacement_costs = {}
    for yr in replacement_years:
        year_idx = yr - unique_years[0]  # Years since project start
        # Stack replacement cost with learning curve applied
        # Cost = Base × REPLACEMENT_FRAC × (1 - learning_rate)^year
        replacement_cost = cfg['REPLACEMENT_FRAC'] * capex_stack * ((1 - learning_rate) ** year_idx)
        replacement_costs[yr] = replacement_cost
    
    # Inflation rate for OPEX escalation
    inflation_rate = cfg.get('INFLATION_RATE', 0.025)
    
    for y in range(1, years + 1):
        year_val = unique_years[y-1] if y <= len(unique_years) else unique_years[-1]
        
        # Inflation factor for OPEX escalation
        inflation_factor = (1 + inflation_rate) ** (y - 1)
        
        # Fixed OPEX (maintenance, insurance, personnel)
        fixed_opex = cfg['FIXED_OPEX_FRAC'] * capex_electrolyser_total * inflation_factor
        storage_opex = cfg['STORAGE_OPEX_FRAC'] * capex_storage * inflation_factor
        
        # Electricity cost (main OPEX driver)
        # Use actual H2 production to estimate electricity consumption
        h2_this_year = annual_H2[y-1] if y-1 < len(annual_H2) else annual_H2[-1]
        avg_sec = df['SEC_total_kWh_per_kg'].mean() if 'SEC_total_kWh_per_kg' in df.columns else 55.0
        elec_this_year_kWh = h2_this_year * avg_sec
        elec_cost = elec_this_year_kWh * cfg['LCOE_ELECTRICITY_EUR_PER_KWH'] * inflation_factor
        electricity_costs.append(elec_cost)
        
        # Water cost (often overlooked)
        water_L = h2_this_year * cfg.get('WATER_CONSUMPTION_L_PER_KG_H2', 10.0)
        water_cost = (water_L / 1000.0) * cfg.get('WATER_COST_EUR_PER_M3', 2.50) * inflation_factor
        water_costs.append(water_cost)
        
        # Curtailment penalty (opportunity cost of wasted renewable energy)
        if enable_curtailment_penalty:
            curtail_cost = (annual_curtailed_kWh / 1000.0) * curtailment_penalty_eur_per_mwh * inflation_factor
        else:
            curtail_cost = 0.0
        curtailment_costs.append(curtail_cost)
        
        # Land lease (annual)
        land_lease = size_kw * cfg.get('LAND_LEASE_EUR_PER_KW_PER_YEAR', 5.0) * inflation_factor
        
        # Variable OPEX (consumables, chemicals, etc.)
        var_opex = elec_cost * cfg['VAR_OPEX_FRAC']
        
        # Replacement costs (with learning curve)
        replacement = replacement_costs.get(year_val, 0.0)
        
        # Maintenance costs
        n_major_this_year = replacement_years.count(year_val)
        n_minor_this_year = sum([dt.year == year_val for dt in minor_maintenance_hours])
        maint_major = MAINT_FRAC_MAJOR * capex_total * n_major_this_year
        maint_minor = MAINT_FRAC_MINOR * capex_total * n_minor_this_year
        maint_compressor = cfg.get('COMPRESSOR_MAINT_FRAC', 0.04) * capex_compressor
        
        # Total annual cost (INCLUDING CURTAILMENT PENALTY)
        costs_y = (fixed_opex + storage_opex + elec_cost + water_cost + curtail_cost + land_lease +
                   var_opex + replacement + maint_major + maint_minor + maint_compressor)
        cash_costs.append(costs_y)
        discount_factors.append(1.0 / (1.0 + cfg['DISCOUNT_RATE']) ** y)
    
    # NPV calculations
    npv_costs = capex_total + np.sum(np.array(cash_costs) * np.array(discount_factors))
    npv_curtailment = np.sum(np.array(curtailment_costs) * np.array(discount_factors))
    discounted_H2 = np.sum([annual_H2[y-1] / (1.0 + cfg['DISCOUNT_RATE']) ** y 
                            for y in range(1, min(years + 1, len(annual_H2) + 1))])
    
    # LCOH (base, no credits)
    LCOH = npv_costs / max(discounted_H2, 1e-9)
    
    # ==========================================================================
    # BYPRODUCT CREDITS (reduce effective LCOH)
    # ==========================================================================
    
    # Heat recovery credit
    total_heat_recovered_kWh = df.attrs.get('total_heat_recovered_kWh', 0.0)
    heat_credit_per_kwh = cfg.get('HEAT_CREDIT_EUR_PER_KWH', cfg.get('HEAT_VALUE_EUR_PER_KWH', 0.025))
    total_heat_revenue_EUR = total_heat_recovered_kWh * heat_credit_per_kwh
    
    # NPV of heat recovery revenue
    annual_heat_revenue = total_heat_revenue_EUR / years
    npv_heat_revenue = np.sum([annual_heat_revenue / (1.0 + cfg['DISCOUNT_RATE']) ** y 
                               for y in range(1, years + 1)])
    
    # Oxygen credit (optional)
    # 8 kg O2 produced per 1 kg H2 (stoichiometry: 2H2O → 2H2 + O2)
    oxygen_credit_enabled = cfg.get('OXYGEN_CREDIT_ENABLED', False)
    oxygen_credit_per_kg_h2 = cfg.get('OXYGEN_CREDIT_EUR_PER_KG_H2', 0.20)
    
    if oxygen_credit_enabled:
        total_oxygen_revenue_EUR = total_H2 * oxygen_credit_per_kg_h2
        annual_oxygen_revenue = total_oxygen_revenue_EUR / years
        npv_oxygen_revenue = np.sum([annual_oxygen_revenue / (1.0 + cfg['DISCOUNT_RATE']) ** y 
                                     for y in range(1, years + 1)])
    else:
        total_oxygen_revenue_EUR = 0.0
        npv_oxygen_revenue = 0.0
    
    # Adjusted LCOH with all credits
    npv_costs_adjusted = npv_costs - npv_heat_revenue - npv_oxygen_revenue
    LCOH_with_credits = npv_costs_adjusted / max(discounted_H2, 1e-9)
    
    # ==========================================================================
    # LCOH BREAKDOWN (for analysis)
    # ==========================================================================
    total_elec_cost = sum(electricity_costs)
    total_water_cost = sum(water_costs)
    
    # Calculate LCOH components (€/kg H2)
    lcoh_capex = (capex_total / discounted_H2) if discounted_H2 > 0 else 0
    lcoh_electricity = (sum([e*d for e,d in zip(electricity_costs, discount_factors)]) / discounted_H2) if discounted_H2 > 0 else 0
    lcoh_opex_fixed = (sum([cfg['FIXED_OPEX_FRAC']*capex_electrolyser_total*d for d in discount_factors]) / discounted_H2) if discounted_H2 > 0 else 0
    lcoh_replacement = (sum([replacement_costs.get(unique_years[y-1] if y<=len(unique_years) else unique_years[-1], 0)*d 
                            for y,d in enumerate(discount_factors, 1)]) / discounted_H2) if discounted_H2 > 0 else 0
    
    # Maintenance totals
    maint_cost_major = capex_total * MAINT_FRAC_MAJOR * len(replacement_years)
    maint_cost_minor = capex_total * MAINT_FRAC_MINOR * len(minor_maintenance_hours)
    maint_cost_compressor = capex_compressor * cfg.get('COMPRESSOR_MAINT_FRAC', 0.04) * years
    maint_cost_total = maint_cost_major + maint_cost_minor + maint_cost_compressor
    
    # Downtime
    downtime_major = len(replacement_years) * DOWNTIME_HOURS_MAJOR
    downtime_minor = len(minor_maintenance_hours) * DOWNTIME_HOURS_MINOR
    downtime_total = downtime_major + downtime_minor
    
    # Base economic metrics (always calculated)
    base_metrics = {
        "total_H2_kg": total_H2,
        "NPV_cost_EUR": npv_costs,
        "NPV_cost_with_credits_EUR": npv_costs_adjusted,
        "LCOH_EUR_per_kg": LCOH,
        "LCOH_with_credits_EUR_per_kg": LCOH_with_credits,
        # CAPEX breakdown
        "capex_total": capex_total,
        "capex_stack": capex_stack,
        "capex_bop": capex_bop,
        "capex_storage": capex_storage,
        "capex_compressor": capex_compressor,
        "capex_installation": capex_installation,
        "capex_engineering": capex_engineering,
        "capex_water_treatment": capex_water_treatment,
        "capex_site_prep": capex_site_prep,
        "capex_grid_connection": capex_grid,
        "capex_contingency": capex_contingency,
        # LCOH breakdown
        "LCOH_capex_EUR_per_kg": lcoh_capex,
        "LCOH_electricity_EUR_per_kg": lcoh_electricity,
        "LCOH_opex_fixed_EUR_per_kg": lcoh_opex_fixed,
        "LCOH_replacement_EUR_per_kg": lcoh_replacement,
        "LCOH_curtailment_EUR_per_kg": npv_curtailment / max(discounted_H2, 1e-9),
        # OPEX totals
        "total_electricity_cost_EUR": total_elec_cost,
        "total_water_cost_EUR": total_water_cost,
        "total_curtailment_cost_EUR": sum(curtailment_costs),
        "total_curtailed_energy_MWh": total_curtailed_kWh / 1000.0,
        "NPV_curtailment_cost_EUR": npv_curtailment,
        # Maintenance
        "stack_replacement_years": replacement_years,
        "stack_replacement_timestamps": replacement_timestamps,
        "minor_maintenance_hours": minor_maintenance_hours,
        "maintenance_cost_major": maint_cost_major,
        "maintenance_cost_minor": maint_cost_minor,
        "maint_cost_total": maint_cost_total,
        "downtime_hours_total": downtime_total,
        # Credits
        "total_heat_recovered_kWh": total_heat_recovered_kWh,
        "total_heat_revenue_EUR": total_heat_revenue_EUR,
        "NPV_heat_revenue_EUR": npv_heat_revenue,
        "oxygen_credit_enabled": oxygen_credit_enabled,
        "total_oxygen_revenue_EUR": total_oxygen_revenue_EUR,
        "NPV_oxygen_revenue_EUR": npv_oxygen_revenue,
        # Learning rate
        "learning_rate": learning_rate,
        "replacement_costs_with_learning": replacement_costs,
    }
    
    # Profitability analysis (only if selling price provided)
    if h2_selling_price_eur_per_kg is not None:
        profitability_metrics = compute_profitability(
            cfg, annual_H2, capex_total, cash_costs, 
            discount_factors, h2_selling_price_eur_per_kg
        )
        base_metrics.update(profitability_metrics)
    
    return base_metrics


def compute_profitability(cfg, annual_H2, capex_total, cash_costs, discount_factors, h2_selling_price_eur_per_kg):
    """
    Compute profitability metrics given H2 selling price.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    annual_H2 : list
        Annual H2 production in kg for each year
    capex_total : float
        Total capital expenditure in EUR
    cash_costs : list
        Annual cash costs (OPEX) for each year
    discount_factors : list
        Discount factors for each year
    h2_selling_price_eur_per_kg : float
        H2 selling price in EUR/kg
        
    Returns
    -------
    dict
        Profitability metrics: NPV_profit, IRR, payback_period, ROI
    """
    years = len(annual_H2)
    discount_rate = cfg['DISCOUNT_RATE']
    
    # Calculate annual revenues
    annual_revenues = [h2_prod * h2_selling_price_eur_per_kg for h2_prod in annual_H2]
    
    # Calculate annual profits (revenue - costs)
    annual_profits = [rev - cost for rev, cost in zip(annual_revenues, cash_costs)]
    
    # NPV of profit = NPV(revenues) - NPV(costs)
    npv_revenue = np.sum(np.array(annual_revenues) * np.array(discount_factors))
    npv_costs_operating = np.sum(np.array(cash_costs) * np.array(discount_factors))
    npv_profit = npv_revenue - capex_total - npv_costs_operating
    
    # Calculate IRR (Internal Rate of Return)
    # Cash flows: year 0 = -CAPEX, years 1-N = annual_profits
    cash_flows = [-capex_total] + annual_profits
    irr = calculate_irr(cash_flows)
    
    # Calculate payback period (simple, undiscounted)
    cumulative_cash = -capex_total
    payback_period_years = None
    for year, profit in enumerate(annual_profits, start=1):
        cumulative_cash += profit
        if cumulative_cash >= 0:
            payback_period_years = year
            break
    
    # If payback not achieved, set to None or total years + 1
    if payback_period_years is None:
        payback_period_years = years + 1  # Beyond project lifetime
    
    # ROI (Return on Investment) = Total profit / Initial investment
    total_undiscounted_profit = sum(annual_profits)
    roi_percent = (total_undiscounted_profit / capex_total) * 100 if capex_total > 0 else 0.0
    
    return {
        "h2_selling_price_EUR_per_kg": h2_selling_price_eur_per_kg,
        "NPV_revenue_EUR": npv_revenue,
        "NPV_profit_EUR": npv_profit,
        "IRR_percent": irr * 100 if irr is not None else None,
        "payback_period_years": payback_period_years,
        "ROI_percent": roi_percent,
        "total_revenue_EUR": sum(annual_revenues),
        "total_profit_EUR": total_undiscounted_profit,
    }


def calculate_irr(cash_flows, max_iterations=100, tolerance=1e-6):
    """
    Calculate Internal Rate of Return using Newton-Raphson method.
    
    Parameters
    ----------
    cash_flows : list
        List of cash flows, starting with initial investment (negative)
    max_iterations : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance
        
    Returns
    -------
    float or None
        IRR as decimal (e.g., 0.10 for 10%), or None if not converged
    """
    # Initial guess
    irr_guess = 0.1
    
    for iteration in range(max_iterations):
        # Calculate NPV and its derivative at current guess
        npv = 0.0
        npv_derivative = 0.0
        
        for t, cf in enumerate(cash_flows):
            npv += cf / ((1 + irr_guess) ** t)
            if t > 0:
                npv_derivative -= t * cf / ((1 + irr_guess) ** (t + 1))
        
        # Check convergence
        if abs(npv) < tolerance:
            return irr_guess
        
        # Newton-Raphson update
        if abs(npv_derivative) < 1e-10:
            # Derivative too small, try different starting point
            return None
        
        irr_guess = irr_guess - npv / npv_derivative
        
        # Prevent unrealistic values
        if irr_guess < -0.99 or irr_guess > 10.0:
            return None
    
    # Did not converge
    return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_power_data(mat_path):
    """Load power data from .mat file."""
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)
    pv = np.array(mat["P_PV"]).flatten()
    wind = np.array(mat["P_wind_selected"]).flatten()
    total_kW = (pv + wind) / 1000.0
    power_1yr = pd.Series(
        total_kW, 
        index=pd.date_range(start="2020-01-01", periods=len(total_kW), freq="h"),
        name="power_kW"
    )
    return power_1yr


def load_demand_data(csv_path, power_index):
    """Load demand data from CSV and align with power index."""
    df_demand = pd.read_csv(csv_path)
    
    # Find time and value columns
    time_cols = [c for c in df_demand.columns if "start" in c.lower()]
    col_time = time_cols[0]
    value_cols = [c for c in df_demand.columns if ("demand" in c.lower()) or ("value" in c.lower())]
    col_value = value_cols[0]
    
    ts = pd.to_datetime(df_demand[col_time])
    
    # Parse demand values
    if pd.api.types.is_numeric_dtype(df_demand[col_value]):
        demand_kWh = df_demand[col_value].astype(float)
    else:
        raw = df_demand[col_value].astype(str)
        raw = raw.str.replace(".", "", regex=False)
        raw = raw.str.replace(",", ".", regex=False)
        demand_kWh = raw.astype(float)
    
    demand_1yr = pd.Series(demand_kWh.values, index=ts, name="demand_kWh")
    demand_1yr = demand_1yr[~demand_1yr.index.duplicated(keep='first')]
    
    # ── Align demand year to power index year ──────────────────────────────
    # Demand CSV may be from a different year (e.g. 2023) than the power data
    # (e.g. 2020).  Replace the year so reindex(method="nearest") works
    # correctly instead of collapsing everything to a single boundary value.
    demand_year = demand_1yr.index[0].year
    power_year  = power_index[0].year
    if demand_year != power_year:
        demand_1yr.index = demand_1yr.index.map(
            lambda t: t.replace(year=power_year))
        # Remove any duplicates introduced by the year shift (leap → non-leap)
        demand_1yr = demand_1yr[~demand_1yr.index.duplicated(keep='first')]
    
    demand_1yr = demand_1yr.reindex(power_index, method="nearest")
    
    return demand_1yr


def synthesize_multiyear_data(power_1yr, demand_1yr, years, rng, deterministic=False):
    """
    Create multi-year data with controlled year-to-year variation.
    
    Parameters
    ----------
    power_1yr : pd.Series
        1-year power data
    demand_1yr : pd.Series
        1-year demand data
    years : int
        Number of years
    rng : np.random.Generator
        Random number generator
    deterministic : bool, optional
        If True, use fixed ±8% and ±5% variation without randomness.
        Year 1: base, Year 2: +4%, Year 3: -4%, etc. (deterministic pattern)
        Default is False (random variation each year).
        
    Returns
    -------
    tuple
        (power_5yr, demand_5yr) as pd.Series
    
    Notes
    -----
    Variability values based on European studies:
    - Power (Wind+Solar): ±8% per year (CV ≈ 4.6%)
      Sources: Jurasz et al. (2020), Staffell & Pfenninger (2018)
    - Demand: ±5% per year (CV ≈ 2.9%)
      Sources: IEA (2021), industrial process stability literature
    
    When deterministic=True:
    - Uses a fixed sinusoidal pattern to simulate inter-annual variability
    - Power varies: 0%, +5.6%, +8%, +5.6%, 0%, -5.6%, -8%, -5.6%, ... (8-year cycle)
    - Demand varies: 0%, +3.5%, +5%, +3.5%, 0%, -3.5%, -5%, -3.5%, ... (8-year cycle)
    - This gives realistic range without randomness for reproducible base case
    """
    n_hours = years * 8760
    
    power_5yr = []
    demand_5yr = []
    
    for y in range(years):
        if deterministic:
            # Deterministic mode: sinusoidal pattern for reproducible results
            # 8-year cycle with ±8% power and ±5% demand variation
            phase = 2 * np.pi * y / 8  # 8-year cycle
            power_var = 1.0 + 0.08 * np.sin(phase)   # ±8% sinusoidal
            demand_var = 1.0 + 0.05 * np.sin(phase)  # ±5% sinusoidal
        else:
            # Random mode: ±8% and ±5% uniform random variation per year
            # Reference: Jurasz et al. (2020), Staffell & Pfenninger (2018)
            power_var = 1.0 + rng.uniform(-0.08, 0.08)
            # Reference: IEA World Energy Outlook, industrial H2 demand stability
            demand_var = 1.0 + rng.uniform(-0.05, 0.05)
        
        power_5yr.append(power_1yr.values * power_var)
        demand_5yr.append(demand_1yr.values * demand_var)
    
    power_5yr = np.concatenate(power_5yr)[:n_hours]
    demand_5yr = np.concatenate(demand_5yr)[:n_hours]
    
    power_index = pd.date_range(start="2020-01-01", periods=n_hours, freq="h")
    power_5yr = pd.Series(power_5yr, index=power_index, name="power_kW")
    demand_5yr = pd.Series(demand_5yr, index=power_index, name="demand_kWh")
    
    return power_5yr, demand_5yr


# =============================================================================
# PLOTTING - IMPROVED FOR CLARITY
# =============================================================================

# Set global matplotlib parameters for better-looking plots
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.titlepad': 15,
    'axes.labelpad': 10,
})


def plot_timeseries(df, col, ylabel, title, filepath, years=10):
    """Plot a timeseries with proper formatting.
    For SEC columns, shows daily rolling average for cleaner visualization.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # For SEC columns, show only the daily rolling average (no raw data)
    if 'SEC' in col:
        data = df[col].replace(0, np.nan)  # Replace 0 with NaN
        # Daily rolling average (only over operating hours)
        daily_avg = data.resample('D').mean()
        ax.plot(daily_avg.index, daily_avg.values, 'b-', linewidth=0.8, alpha=0.5, label='Daily average')
        # Monthly rolling mean for trend
        monthly_trend = daily_avg.rolling(window=30, min_periods=1).mean()
        ax.plot(daily_avg.index, monthly_trend, 'r-', label='Monthly trend', linewidth=2)
        ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
        # Set y-axis to reasonable range around the data
        valid_data = daily_avg.dropna()
        if len(valid_data) > 0:
            y_min = valid_data.min() * 0.95
            y_max = valid_data.max() * 1.05
            ax.set_ylim(y_min, y_max)
    else:
        ax.plot(df.index, df[col], label=col, linewidth=0.5)
    
    ax.set_xlabel("Time", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis for better readability with longer time series
    fig.autofmt_xdate(rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_weekly(df, col, ylabel, title, filepath, years=10):
    """Plot weekly-averaged timeseries.
    For SEC columns, only averages over operating hours (non-zero values).
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # For SEC, only average non-zero values (operating hours)
    if 'SEC' in col:
        weekly = df[col].replace(0, np.nan).resample('W').mean()
        ax.plot(weekly.index, weekly.values, 'b-', linewidth=1.5, alpha=0.7, label='Weekly average')
        # Add monthly trend line
        monthly_trend = weekly.rolling(window=4, min_periods=1).mean()
        ax.plot(weekly.index, monthly_trend, 'r-', linewidth=2, label='Monthly trend')
        valid_data = weekly.dropna()
        if len(valid_data) > 0:
            y_min = valid_data.min() * 0.95
            y_max = valid_data.max() * 1.05
            ax.set_ylim(y_min, y_max)
    else:
        weekly = df[col].resample('W').mean()
        ax.plot(weekly.index, weekly.values, 'b-', linewidth=1.5, label=col)
    
    ax.set_xlabel("Time", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title + " (Weekly Average)", fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    
    # Format x-axis for better readability
    fig.autofmt_xdate(rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_with_curve_fitting(df, econ, sim_folder, size_mw, years=10):
    """
    Generate high-resolution plots with polynomial curve fitting for SEC and voltage.
    Uses monthly data for higher resolution and fits curves per stack lifetime segment.
    Places replacement markers at the PEAK (last point before drop).
    """
    # Use monthly averaging for higher resolution (instead of 6-month)
    df_monthly = df.resample('ME').mean()
    times = df_monthly.index
    
    # Get replacement timestamps
    replacement_timestamps = econ.get('stack_replacement_timestamps', [])
    replacement_times = [pd.Timestamp(ts) for ts in replacement_timestamps] if replacement_timestamps else []
    
    # Add simulation start and end to create segments
    segment_boundaries = [df.index[0]] + replacement_times + [df.index[-1]]
    
    # Colors for different stack segments
    segment_colors = plt.cm.tab10(np.linspace(0, 1, len(segment_boundaries)-1))
    
    # Plot configurations
    plot_configs = [
        ('SEC_stack_kWh_per_kg', 'Stack SEC', 'SEC Stack', 'upper left', 'SEC_stack_fitted.png'),
        ('SEC_total_kWh_per_kg', 'System SEC', 'System SEC', 'upper left', 'SEC_total_fitted.png'),
        ('V_cell_V', 'Cell Voltage', 'V_cell', 'upper left', 'Vcell_fitted.png'),
    ]
    
    for col, ylabel, label_prefix, legend_loc, filename in plot_configs:
        if col not in df_monthly.columns:
            continue
            
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Process each stack segment separately
        for seg_idx in range(len(segment_boundaries) - 1):
            seg_start = segment_boundaries[seg_idx]
            seg_end = segment_boundaries[seg_idx + 1]
            
            # Get data for this segment
            mask = (df_monthly.index >= seg_start) & (df_monthly.index < seg_end)
            seg_data = df_monthly.loc[mask, col].dropna()
            
            if len(seg_data) < 3:
                continue
            
            # Convert timestamps to numeric for fitting
            x_numeric = np.arange(len(seg_data))
            y_data = seg_data.values
            
            # Remove zeros/invalids for fitting
            valid_mask = y_data > 0
            if valid_mask.sum() < 3:
                continue
            x_fit = x_numeric[valid_mask]
            y_fit = y_data[valid_mask]
            
            # Fit polynomial (degree 2 for gradual degradation curve)
            try:
                # Use robust polynomial fitting
                coeffs = np.polyfit(x_fit, y_fit, deg=2)
                poly_func = np.poly1d(coeffs)
                y_fitted = poly_func(x_fit)
                
                # Check fit quality (R²)
                ss_res = np.sum((y_fit - y_fitted) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Plot data points
                x_times = seg_data.index[valid_mask]
                color = segment_colors[seg_idx]
                ax.scatter(x_times, y_fit, alpha=0.5, s=20, color=color, 
                          label=f'Stack {seg_idx + 1} Data' if seg_idx == 0 else f'Stack {seg_idx + 1}')
                
                # Plot fitted curve
                ax.plot(x_times, y_fitted, '-', linewidth=2.5, color=color,
                       label=f'Stack {seg_idx + 1} Fit (R²={r_squared:.3f})')
                
                # Find and mark the PEAK (max value before replacement)
                if seg_idx < len(replacement_times):
                    peak_idx = np.argmax(y_fit)
                    peak_time = x_times[peak_idx]
                    peak_val = y_fit[peak_idx]
                    ax.axvline(peak_time, color='red', linestyle='--', linewidth=2, alpha=0.8,
                              label='Stack Replacement' if seg_idx == 0 else '')
                    ax.scatter([peak_time], [peak_val], color='red', s=100, marker='v', zorder=5,
                              label='Peak before replacement' if seg_idx == 0 else '')
                    
            except Exception as e:
                print(f"Warning: Could not fit curve for {col} segment {seg_idx}: {e}")
                # Fallback: just plot the data
                ax.scatter(seg_data.index, seg_data.values, alpha=0.5, s=20, 
                          color=segment_colors[seg_idx], label=f'Stack {seg_idx + 1}')
        
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel(f'{ylabel} ({"kWh/kg" if "SEC" in col else "V"})', fontweight='bold')
        ax.set_title(f'{label_prefix} Evolution with Polynomial Fit ({size_mw} MW, {years} Years)', 
                    fontweight='bold', pad=15)
        ax.legend(loc=legend_loc, framealpha=0.9, edgecolor='gray', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.autofmt_xdate(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'{sim_folder}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  📈 Generated curve-fitted plots in {sim_folder}")


def plot_6month_metrics(df, econ, sim_folder, size_mw, years=10):
    """Generate 6-month interval analysis plots with improved clarity."""
    df_6mo = df.resample('6ME').mean()
    times_6mo = df_6mo.index
    
    # Use exact timestamps for replacement markers (not Jan 1st of year)
    replacement_timestamps = econ.get('stack_replacement_timestamps', [])
    if replacement_timestamps:
        replacement_times = [pd.Timestamp(ts) for ts in replacement_timestamps]
    else:
        # Fallback to year-based if timestamps not available
        replacement_years = econ['stack_replacement_years']
        replacement_times = [pd.Timestamp(f"{y}-07-01") for y in set(replacement_years)]
    
    # LCOH timeseries
    lcoh_6mo = []
    for t in times_6mo:
        mask = (df.index <= t)
        h2_cum = df.loc[mask, 'H2_kg'].sum()
        cost_cum = econ['NPV_cost_EUR'] * (mask.sum() / len(df))
        lcoh_6mo.append(cost_cum / max(h2_cum, 1e-9))
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times_6mo, lcoh_6mo, 'b-o', markersize=6, linewidth=2, label='LCOH (EUR/kg)')
    for i, rt in enumerate(replacement_times):
        ax.axvline(rt, color='red', linestyle='--', linewidth=2, 
                   label='Stack Replacement' if i == 0 else '', alpha=0.8)
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('LCOH (EUR/kg)', fontweight='bold')
    ax.set_title(f'LCOH Evolution over {years} Years ({size_mw} MW PEM Electrolyser)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/LCOH_variation_years.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SEC timeseries
    for col, label, color in [('SEC_stack_kWh_per_kg', 'Stack SEC', 'blue'), 
                               ('SEC_total_kWh_per_kg', 'System SEC', 'green')]:
        sec_6mo = df[col].resample('6ME').mean()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(sec_6mo.index, sec_6mo.values, f'{color[0]}-o', markersize=6, linewidth=2, label=f'{label} (kWh/kg)')
        for i, rt in enumerate(replacement_times):
            ax.axvline(rt, color='red', linestyle='--', linewidth=2,
                       label='Stack Replacement' if i == 0 else '', alpha=0.8)
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel(f'{label} (kWh/kg)', fontweight='bold')
        ax.set_title(f'{label} Evolution over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
        ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.autofmt_xdate(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'{sim_folder}/{col.replace("kWh_per_kg", "timeseries")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # H2 production
    h2_6mo = df['H2_kg'].resample('6ME').sum()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(h2_6mo.index, h2_6mo.values, 'g-o', markersize=6, linewidth=2, label='H2 Production (kg/6-month)')
    for i, rt in enumerate(replacement_times):
        ax.axvline(rt, color='red', linestyle='--', linewidth=2,
                   label='Stack Replacement' if i == 0 else '', alpha=0.8)
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('H2 Produced (kg/6-month)', fontweight='bold')
    ax.set_title(f'Hydrogen Production over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/H2_production_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots(df, econ, cfg, sim_folder, size_mw):
    """Generate all output plots with improved formatting."""
    years = cfg['YEARS']
    
    # Weekly plots
    plot_weekly(df, "SEC_stack_kWh_per_kg", "SEC stack (kWh/kg)", 
                f"SEC Stack over {years} Years", f"{sim_folder}/SEC_stack_timeseries_weekly.png", years)
    plot_weekly(df, "SEC_total_kWh_per_kg", "System SEC (kWh/kg)", 
                f"System SEC over {years} Years", f"{sim_folder}/SEC_total_weekly.png", years)
    plot_weekly(df, "V_cell_V", "Cell Voltage (V)", 
                f"Cell Voltage over {years} Years", f"{sim_folder}/Vcell_timeseries_weekly.png", years)
    plot_weekly(df, "H2_kg", "Hydrogen Production (kg/h)", 
                f"Hydrogen Production over {years} Years", f"{sim_folder}/H2_production_timeseries_weekly.png", years)
    plot_weekly(df, "storage_kg", "Storage State of Charge (kg)", 
                f"Storage SOC over {years} Years", f"{sim_folder}/storage_SOC_timeseries_weekly.png", years)
    
    # Full timeseries plots
    plot_timeseries(df, "SEC_stack_kWh_per_kg", "SEC stack (kWh/kg)", 
                    f"SEC Stack over {years} Years ({size_mw} MW)", f"{sim_folder}/SEC_stack_timeseries.png", years)
    plot_timeseries(df, "SEC_total_kWh_per_kg", "System SEC (kWh/kg)", 
                    f"System SEC over {years} Years ({size_mw} MW)", f"{sim_folder}/SEC_total_timeseries.png", years)
    plot_timeseries(df, "V_cell_V", "Cell Voltage (V)", 
                    f"Cell Voltage over {years} Years ({size_mw} MW)", f"{sim_folder}/Vcell_timeseries.png", years)
    plot_timeseries(df, "storage_kg", "Storage SOC (kg)", 
                    f"Storage SOC over {years} Years ({size_mw} MW)", f"{sim_folder}/storage_SOC_timeseries.png", years)
    
    # 6-month metrics
    plot_6month_metrics(df, econ, sim_folder, size_mw, years)
    
    # High-resolution curve-fitted plots for SEC and voltage
    plot_with_curve_fitting(df, econ, sim_folder, size_mw, years)
    
    # Production vs demand (first year, weekly average)
    one_year_mask = (df.index >= df.index[0]) & (df.index < df.index[0] + pd.Timedelta(days=365))
    df_one_year = df.loc[one_year_mask]
    df_weekly = df_one_year.resample('W').mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_weekly.index, df_weekly['H2_kg'], 'b-', linewidth=2, label='Production (kg/h)')
    ax.plot(df_weekly.index, df_weekly['demand_H2_kg'], 'r--', linewidth=2, label='Demand (kg/h)')
    ax.plot(df_weekly.index, df_weekly['storage_kg'], 'g-', linewidth=1.5, alpha=0.7, label='Storage (kg)')
    ax.set_xlabel('Week', fontweight='bold')
    ax.set_ylabel('Hydrogen (kg)', fontweight='bold')
    ax.set_title(f'Production vs Demand vs Storage (First Year, {size_mw} MW)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/production_vs_demand.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save one-year CSV
    df_one_year[['H2_kg', 'demand_H2_kg', 'storage_kg']].to_csv(
        f'{sim_folder}/production_vs_demand_vs_storage_1year.csv')
    
    # =========================================================================
    # ADDITIONAL PLOTS FOR COMPREHENSIVE ANALYSIS
    # =========================================================================
    
    # 1. Degradation voltage increase over time
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_vcell = df['V_cell_V'].replace(0, np.nan).resample('ME').mean()
    ax.plot(monthly_vcell.index, monthly_vcell.values, 'b-o', markersize=4, linewidth=2, label='Monthly Avg Cell Voltage')
    
    # Add linear trend line
    valid_mask = ~np.isnan(monthly_vcell.values)
    if valid_mask.sum() > 1:
        x_numeric = np.arange(len(monthly_vcell))[valid_mask]
        y_values = monthly_vcell.values[valid_mask]
        z = np.polyfit(x_numeric, y_values, 1)
        p = np.poly1d(z)
        ax.plot(monthly_vcell.index, p(np.arange(len(monthly_vcell))), 'r--', linewidth=2, 
                label=f'Trend: +{z[0]*12*1000:.1f} mV/year')
    
    # Mark replacement times using exact timestamps
    replacement_timestamps = econ.get('stack_replacement_timestamps', [])
    if replacement_timestamps:
        replacement_times = [pd.Timestamp(ts) for ts in replacement_timestamps]
    else:
        replacement_years = econ['stack_replacement_years']
        replacement_times = [pd.Timestamp(f"{y}-07-01") for y in set(replacement_years)]
    for i, rt in enumerate(replacement_times):
        ax.axvline(rt, color='green', linestyle='--', linewidth=2,
                   label='Stack Replacement' if i == 0 else '', alpha=0.8)
    
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Cell Voltage (V)', fontweight='bold')
    ax.set_title(f'Cell Voltage Degradation over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/voltage_degradation_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Efficiency evolution over time
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate efficiency: η = (LHV_H2 * H2_kg) / (kWh_consumed)
    # η = HHV/SEC_stack ≈ 39.4 / SEC_stack (or 33.33/SEC for LHV)
    efficiency = 33.33 / df['SEC_stack_kWh_per_kg'].replace(0, np.nan)
    monthly_eff = efficiency.resample('ME').mean() * 100  # Convert to percentage
    
    ax.plot(monthly_eff.index, monthly_eff.values, 'g-o', markersize=4, linewidth=2, label='Monthly Avg Efficiency')
    
    # Add trend line
    valid_mask = ~np.isnan(monthly_eff.values)
    if valid_mask.sum() > 1:
        x_numeric = np.arange(len(monthly_eff))[valid_mask]
        y_values = monthly_eff.values[valid_mask]
        z = np.polyfit(x_numeric, y_values, 1)
        p = np.poly1d(z)
        ax.plot(monthly_eff.index, p(np.arange(len(monthly_eff))), 'r--', linewidth=2,
                label=f'Trend: {z[0]*12:.2f} %/year')
    
    # Mark replacement times
    for i, rt in enumerate(replacement_times):
        ax.axvline(rt, color='blue', linestyle='--', linewidth=2,
                   label='Stack Replacement' if i == 0 else '', alpha=0.8)
    
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Efficiency (%)', fontweight='bold')
    ax.set_title(f'Electrolyser Efficiency Evolution over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/efficiency_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Annual summary bar chart
    annual_h2 = df['H2_kg'].resample('YE').sum() / 1000  # Convert to tonnes
    annual_energy = df['power_kW'].resample('YE').sum()  # kWh
    annual_sec_stack = df['SEC_stack_kWh_per_kg'].replace(0, np.nan).resample('YE').mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # H2 Production
    years_labels = [f'Year {i+1}' for i in range(len(annual_h2))]
    bars1 = axes[0].bar(years_labels, annual_h2.values, color='green', edgecolor='darkgreen', alpha=0.8)
    axes[0].set_ylabel('H2 Production (tonnes)', fontweight='bold')
    axes[0].set_title('Annual H2 Production', fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
    # Add value labels on bars
    for bar, val in zip(bars1, annual_h2.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                     f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # SEC Stack
    bars2 = axes[1].bar(years_labels, annual_sec_stack.values, color='blue', edgecolor='darkblue', alpha=0.8)
    axes[1].set_ylabel('SEC Stack (kWh/kg)', fontweight='bold')
    axes[1].set_title('Annual Average SEC Stack', fontweight='bold', pad=10)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    for bar, val in zip(bars2, annual_sec_stack.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Mark replacement years in bar charts
    replacement_years_list = econ.get('stack_replacement_years', [])
    for i, yr in enumerate(replacement_years_list):
        year_idx = yr - df.index[0].year
        if 0 <= year_idx < len(years_labels):
            axes[0].patches[year_idx].set_edgecolor('red')
            axes[0].patches[year_idx].set_linewidth(3)
            axes[1].patches[year_idx].set_edgecolor('red')
            axes[1].patches[year_idx].set_linewidth(3)
    
    # Cumulative energy
    cum_energy = df['power_kW'].cumsum().resample('YE').last() / 1e6  # MWh → GWh conversion: /1e3 then /1e3
    axes[2].bar(years_labels, cum_energy.values/1000, color='orange', edgecolor='darkorange', alpha=0.8)  # Convert to GWh
    axes[2].set_ylabel('Cumulative Energy (GWh)', fontweight='bold')
    axes[2].set_title('Cumulative Energy Consumption', fontweight='bold', pad=10)
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/annual_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Operating load distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    load_frac = df['power_kW'] / cfg['ELECTROLYSER_SIZE_KW']
    # Only non-zero loads
    operating_loads = load_frac[load_frac > 0]
    
    ax.hist(operating_loads * 100, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(10, color='red', linestyle='--', linewidth=2, label='Min Load (10%)')
    ax.axvline(operating_loads.mean() * 100, color='green', linestyle='-', linewidth=2, 
               label=f'Mean Load ({operating_loads.mean()*100:.1f}%)')
    ax.axvline(operating_loads.median() * 100, color='orange', linestyle='-', linewidth=2,
               label=f'Median Load ({operating_loads.median()*100:.1f}%)')
    
    ax.set_xlabel('Load Factor (%)', fontweight='bold')
    ax.set_ylabel('Frequency (hours)', fontweight='bold')
    ax.set_title(f'Operating Load Distribution over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{sim_folder}/load_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # THERMAL MANAGEMENT PLOTS (NEW)
    # =========================================================================
    
    # 5. Stack Temperature Evolution
    if 'T_stack_C' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Weekly average temperature
        T_weekly = df['T_stack_C'].resample('W').mean()
        ax.plot(T_weekly.index, T_weekly.values, 'r-', linewidth=1.5, alpha=0.7, label='Weekly Avg Temperature')
        
        # Monthly trend
        T_monthly = df['T_stack_C'].resample('ME').mean()
        ax.plot(T_monthly.index, T_monthly.values, 'b-o', markersize=4, linewidth=2, label='Monthly Avg')
        
        # Reference lines
        ax.axhline(cfg['T_OPERATING_C'], color='green', linestyle='--', linewidth=2, 
                   label=f"Target Temp ({cfg['T_OPERATING_C']}°C)", alpha=0.8)
        ax.axhline(cfg['T_MIN_OPERATING_C'], color='orange', linestyle=':', linewidth=2,
                   label=f"Min Operating ({cfg['T_MIN_OPERATING_C']}°C)", alpha=0.8)
        
        ax.set_xlabel('Time', fontweight='bold')
        ax.set_ylabel('Stack Temperature (°C)', fontweight='bold')
        ax.set_title(f'Stack Temperature Evolution over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([cfg['T_AMBIENT_C'] - 5, cfg['T_MAX_OPERATING_C'] + 5])
        fig.autofmt_xdate(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'{sim_folder}/thermal_temperature_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Heat Generation and Recovery
    if 'Q_heat_gen_kW' in df.columns and 'Q_recovered_kW' in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Monthly totals
        Q_gen_monthly = df['Q_heat_gen_kW'].resample('ME').sum() / 1000  # MWh
        Q_recovered_monthly = df['Q_recovered_kW'].resample('ME').sum() / 1000  # MWh
        Q_cooling_monthly = df['Q_cooling_kW'].resample('ME').sum() / 1000  # MWh
        
        # Plot 1: Heat flows
        axes[0].bar(Q_gen_monthly.index, Q_gen_monthly.values, width=25, alpha=0.7, 
                   color='red', label='Heat Generated')
        axes[0].bar(Q_recovered_monthly.index, Q_recovered_monthly.values, width=25, alpha=0.7,
                   color='green', label='Heat Recovered')
        axes[0].set_ylabel('Heat Energy (MWh/month)', fontweight='bold')
        axes[0].set_title(f'Heat Generation and Recovery over {years} Years ({size_mw} MW)', fontweight='bold', pad=15)
        axes[0].legend(loc='upper right', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Cumulative heat recovery and revenue
        cum_recovered = df['Q_recovered_kW'].cumsum() / 1e6  # GWh
        cum_revenue = df['heat_revenue_EUR'].cumsum() / 1e6  # Million EUR
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(cum_recovered.index, cum_recovered.values, 'g-', linewidth=2, 
                        label='Cumulative Heat Recovered')
        line2 = ax2_twin.plot(cum_revenue.index, cum_revenue.values, 'b--', linewidth=2,
                             label='Cumulative Revenue')
        
        ax2.set_xlabel('Time', fontweight='bold')
        ax2.set_ylabel('Cumulative Heat Recovered (GWh)', fontweight='bold', color='green')
        ax2_twin.set_ylabel('Cumulative Revenue (Million EUR)', fontweight='bold', color='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        fig.autofmt_xdate(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(f'{sim_folder}/thermal_heat_recovery.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Thermal-Electrical Energy Sankey-style breakdown (bar chart)
    if 'Q_heat_gen_kW' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate energy totals
        total_elec_in = df['power_kW'].sum() / 1e6  # GWh
        total_H2_energy = df['H2_kg'].sum() * cfg['H2_LHV_KWH_PER_KG'] / 1e6  # GWh
        total_heat_gen = df['Q_heat_gen_kW'].sum() / 1e6  # GWh
        total_heat_recovered = df['Q_recovered_kW'].sum() / 1e6  # GWh
        total_cooling_elec = df['cooling_power_kW'].sum() / 1e6  # GWh
        
        categories = ['Electricity\nInput', 'H₂ Energy\n(LHV)', 'Waste Heat\nGenerated', 
                     'Heat\nRecovered', 'Cooling\nElectricity']
        values = [total_elec_in, total_H2_energy, total_heat_gen, total_heat_recovered, total_cooling_elec]
        colors = ['steelblue', 'green', 'red', 'orange', 'purple']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Energy (GWh)', fontweight='bold')
        ax.set_title(f'Energy Balance Summary ({years} Years, {size_mw} MW)', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add efficiency annotation
        elec_eff = (total_H2_energy / total_elec_in) * 100
        heat_recovery_rate = (total_heat_recovered / total_heat_gen) * 100 if total_heat_gen > 0 else 0
        ax.annotate(f'Electrical Efficiency: {elec_eff:.1f}%\nHeat Recovery: {heat_recovery_rate:.1f}%', 
                   xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top',
                   fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{sim_folder}/thermal_energy_balance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  🌡️ Generated thermal management plots")


# =============================================================================
# MONTE CARLO UNCERTAINTY ANALYSIS
# =============================================================================

def run_monte_carlo(size_mw, storage_kg, power_1yr, demand_1yr, n_simulations=100, 
                    output_folder=None, verbose=True):
    """
    Run Monte Carlo simulation to quantify uncertainty in LCOH and other outputs.
    
    Parameters
    ----------
    size_mw : float
        Electrolyser size in MW
    storage_kg : float
        Storage capacity in kg
    power_1yr : pd.Series
        1-year power data
    demand_1yr : pd.Series
        1-year demand data
    n_simulations : int
        Number of Monte Carlo iterations
    output_folder : str, optional
        Folder to save results
    verbose : bool
        Print progress updates
        
    Returns
    -------
    dict
        Monte Carlo results with statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"MONTE CARLO UNCERTAINTY ANALYSIS")
        print(f"{'='*60}")
        print(f"Simulations: {n_simulations}")
        print(f"Electrolyser: {size_mw} MW, Storage: {storage_kg} kg")
    
    # Store results
    results = {
        'LCOH': [],
        'total_H2_kg': [],
        'NPV_cost_EUR': [],
        'SEC_stack_mean': [],
        'SEC_total_mean': [],
        'stack_replacements': [],
        'unmet_demand_frac': [],
        'capacity_factor': [],
    }
    
    # Store sampled parameters for sensitivity analysis
    sampled_params = {param: [] for param in UNCERTAINTY_PARAMS}
    
    # Run simulations
    for i in range(n_simulations):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Simulation {i+1}/{n_simulations}...")
        
        # Create RNG with unique seed for this iteration
        rng = np.random.default_rng(RNG_SEED + i)
        
        # Get config with uncertainty
        cfg = get_config_with_uncertainty(size_mw, storage_kg, rng)
        
        # Store sampled parameters - handle all cases
        for param in UNCERTAINTY_PARAMS:
            if param in cfg:
                sampled_params[param].append(cfg[param])
            elif param == 'POWER_VARIATION':
                val = sample_uncertain_param(param, rng)
                sampled_params[param].append(val)
            elif param == 'DEMAND_VARIATION':
                val = sample_uncertain_param(param, rng)
                sampled_params[param].append(val)
            else:
                # Parameter not in config, sample it directly
                val = sample_uncertain_param(param, rng)
                sampled_params[param].append(val)
        
        # Apply power and demand variation
        power_var = sampled_params['POWER_VARIATION'][-1] if 'POWER_VARIATION' in sampled_params else 1.0
        demand_var = sampled_params['DEMAND_VARIATION'][-1] if 'DEMAND_VARIATION' in sampled_params else 1.0
        
        power_1yr_var = power_1yr * power_var
        demand_1yr_var = demand_1yr * demand_var
        
        # Synthesize multi-year data
        power_5yr, demand_5yr = synthesize_multiyear_data(
            power_1yr_var, demand_1yr_var, cfg['YEARS'], rng)
        
        # Run simulation
        try:
            df = simulate(cfg, power_5yr, demand_5yr, rng)
            econ = compute_economics(cfg, df, cfg.get('H2_SELLING_PRICE_EUR_PER_KG'))
            
            # Store results
            results['LCOH'].append(econ['LCOH_EUR_per_kg'])
            results['total_H2_kg'].append(econ['total_H2_kg'])
            results['NPV_cost_EUR'].append(econ['NPV_cost_EUR'])
            # Only average SEC over operating hours (non-zero)
            results['SEC_stack_mean'].append(df.loc[df['SEC_stack_kWh_per_kg'] > 0, 'SEC_stack_kWh_per_kg'].mean())
            results['SEC_total_mean'].append(df.loc[df['SEC_total_kWh_per_kg'] > 0, 'SEC_total_kWh_per_kg'].mean())
            results['stack_replacements'].append(len(econ['stack_replacement_years']))
            
            # Calculate unmet demand fraction
            total_demand = df['demand_H2_kg'].sum()
            total_unmet = df['unmet_kg'].sum()
            results['unmet_demand_frac'].append(total_unmet / max(total_demand, 1e-9))
            
            # Calculate capacity factor
            total_power_used = df['power_kW'].sum() - df['curtailed_power_kWh'].sum()
            max_power = cfg['ELECTROLYSER_SIZE_KW'] * len(df)
            results['capacity_factor'].append(total_power_used / max(max_power, 1e-9))
            
        except Exception as e:
            if verbose:
                print(f"    Warning: Simulation {i+1} failed - {e}")
            continue
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    for key in sampled_params:
        sampled_params[key] = np.array(sampled_params[key])
    
    # Calculate statistics
    stats_dict = {}
    for key, values in results.items():
        if len(values) > 0:
            stats_dict[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'p5': np.percentile(values, 5),
                'p25': np.percentile(values, 25),
                'p75': np.percentile(values, 75),
                'p95': np.percentile(values, 95),
                'min': np.min(values),
                'max': np.max(values),
            }
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("MONTE CARLO RESULTS")
        print(f"{'='*60}")
        print(f"Successful simulations: {len(results['LCOH'])}/{n_simulations}")
        print(f"\nLCOH (EUR/kg):")
        print(f"  Mean ± Std:  {stats_dict['LCOH']['mean']:.2f} ± {stats_dict['LCOH']['std']:.2f}")
        print(f"  Median:      {stats_dict['LCOH']['median']:.2f}")
        print(f"  90% CI:      [{stats_dict['LCOH']['p5']:.2f}, {stats_dict['LCOH']['p95']:.2f}]")
        print(f"  Range:       [{stats_dict['LCOH']['min']:.2f}, {stats_dict['LCOH']['max']:.2f}]")
        
        print(f"\nTotal H2 (kg, 5 years):")
        print(f"  Mean ± Std:  {stats_dict['total_H2_kg']['mean']:.0f} ± {stats_dict['total_H2_kg']['std']:.0f}")
        print(f"  90% CI:      [{stats_dict['total_H2_kg']['p5']:.0f}, {stats_dict['total_H2_kg']['p95']:.0f}]")
        
        print(f"\nSEC Stack (kWh/kg):")
        print(f"  Mean ± Std:  {stats_dict['SEC_stack_mean']['mean']:.1f} ± {stats_dict['SEC_stack_mean']['std']:.1f}")
        
        print(f"\nUnmet Demand Fraction:")
        print(f"  Mean ± Std:  {stats_dict['unmet_demand_frac']['mean']:.3f} ± {stats_dict['unmet_demand_frac']['std']:.3f}")
    
    # Save results if output folder specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save raw results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{output_folder}/monte_carlo_results.csv", index=False)
        
        # Save statistics
        stats_df = pd.DataFrame(stats_dict).T
        stats_df.to_csv(f"{output_folder}/monte_carlo_statistics.csv")
        
        # Save sampled parameters
        params_df = pd.DataFrame(sampled_params)
        params_df.to_csv(f"{output_folder}/monte_carlo_sampled_params.csv", index=False)
        
        # Generate plots
        plot_monte_carlo_results(results, sampled_params, stats_dict, output_folder, size_mw)
        
        if verbose:
            print(f"\n✓ Monte Carlo results saved to {output_folder}/")
    
    return {
        'results': results,
        'statistics': stats_dict,
        'sampled_params': sampled_params,
    }


def plot_monte_carlo_results(results, sampled_params, stats_dict, output_folder, size_mw):
    """Generate plots for Monte Carlo analysis."""
    
    # 1. LCOH histogram with distribution fit
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(results['LCOH'], bins=30, density=True, alpha=0.7, 
                                color='steelblue', edgecolor='white')
    
    # Fit normal distribution
    mu, std = stats_dict['LCOH']['mean'], stats_dict['LCOH']['std']
    x = np.linspace(results['LCOH'].min(), results['LCOH'].max(), 100)
    pdf = stats.norm.pdf(x, mu, std)
    ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
    
    # Add percentile lines
    ax.axvline(stats_dict['LCOH']['p5'], color='green', linestyle='--', 
               label=f"5th percentile: {stats_dict['LCOH']['p5']:.2f}")
    ax.axvline(stats_dict['LCOH']['p95'], color='green', linestyle='--',
               label=f"95th percentile: {stats_dict['LCOH']['p95']:.2f}")
    ax.axvline(mu, color='red', linestyle='-', linewidth=2,
               label=f"Mean: {mu:.2f}")
    
    ax.set_xlabel('LCOH (EUR/kg)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'LCOH Distribution from Monte Carlo ({size_mw} MW)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/mc_lcoh_distribution.png", dpi=300)
    plt.close()
    
    # 2. Box plots for key outputs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # LCOH
    axes[0, 0].boxplot(results['LCOH'], vert=True)
    axes[0, 0].set_ylabel('LCOH (EUR/kg)')
    axes[0, 0].set_title('LCOH Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # SEC
    axes[0, 1].boxplot([results['SEC_stack_mean'], results['SEC_total_mean']], 
                       tick_labels=['Stack', 'System'])
    axes[0, 1].set_ylabel('SEC (kWh/kg)')
    axes[0, 1].set_title('SEC Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # H2 Production
    axes[1, 0].boxplot(results['total_H2_kg'] / 1e6)
    axes[1, 0].set_ylabel('Total H2 (Million kg)')
    axes[1, 0].set_title('H2 Production Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Unmet demand
    axes[1, 1].boxplot(results['unmet_demand_frac'] * 100)
    axes[1, 1].set_ylabel('Unmet Demand (%)')
    axes[1, 1].set_title('Unmet Demand Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Monte Carlo Results Summary ({size_mw} MW)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/mc_boxplots.png", dpi=300)
    plt.close()
    
    # 3. Tornado chart (sensitivity analysis)
    # Calculate correlation between inputs and LCOH
    correlations = {}
    for param, values in sampled_params.items():
        if len(values) == len(results['LCOH']) and np.std(values) > 0:
            corr = np.corrcoef(values, results['LCOH'])[0, 1]
            if not np.isnan(corr):
                correlations[param] = corr
    
    if correlations:
        # Sort by absolute correlation
        sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        params = [p[0] for p in sorted_params[:10]]  # Top 10
        corrs = [p[1] for p in sorted_params[:10]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if c > 0 else 'green' for c in corrs]
        y_pos = np.arange(len(params))
        ax.barh(y_pos, corrs, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel('Correlation with LCOH')
        ax.set_title('Sensitivity Analysis: Parameter Impact on LCOH')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (param, corr) in enumerate(zip(params, corrs)):
            ax.annotate(f'{corr:.2f}', xy=(corr, i), 
                       xytext=(5 if corr > 0 else -5, 0),
                       textcoords='offset points',
                       ha='left' if corr > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_folder}/mc_tornado_chart.png", dpi=300)
        plt.close()
    
    # 4. Scatter plot: LCOH vs key parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scatter_params = ['CAPEX_EUR_PER_KW', 'LCOE_ELECTRICITY_EUR_PER_KWH', 
                      'R_OHM', 'DISCOUNT_RATE']
    scatter_labels = ['CAPEX (EUR/kW)', 'Electricity Price (EUR/kWh)',
                      'Ohmic Resistance (Ω·cm²)', 'Discount Rate']
    
    for ax, param, label in zip(axes.flat, scatter_params, scatter_labels):
        if param in sampled_params and len(sampled_params[param]) == len(results['LCOH']):
            ax.scatter(sampled_params[param], results['LCOH'], alpha=0.5, s=20)
            ax.set_xlabel(label)
            ax.set_ylabel('LCOH (EUR/kg)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(sampled_params[param], results['LCOH'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sampled_params[param].min(), sampled_params[param].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
    
    plt.suptitle(f'LCOH Sensitivity to Key Parameters ({size_mw} MW)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/mc_scatter_sensitivity.png", dpi=300)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for simulation."""
    # Initialize RNG for reproducibility
    rng = np.random.default_rng(RNG_SEED)
    
    # Get user input for electrolyser size
    try:
        size_mw = float(input("Enter electrolyser size in MW (e.g., 10, 15, 20): "))
    except (ValueError, EOFError):
        print("Invalid input. Using default size 20 MW.")
        size_mw = 20.0
    
    # Scale storage proportionally (125 kg/MW ratio = 7.4 hours buffer)
    STORAGE_RATIO_KG_PER_MW = 125.0
    storage_kg = size_mw * STORAGE_RATIO_KG_PER_MW
    
    # Ask about Monte Carlo analysis
    try:
        run_mc = input("Run Monte Carlo uncertainty analysis? (y/n) [n]: ").strip().lower()
    except EOFError:
        run_mc = 'n'
    
    # Setup output folder
    sim_folder = f"results/{int(size_mw)}MW_sim"
    os.makedirs(sim_folder, exist_ok=True)
    
    # File paths - use project data folder
    script_dir = Path(__file__).parent.parent
    mat_path = script_dir / "data" / "combined_wind_pv_DATA.mat"
    demand_csv_path = script_dir / "data" / "Company_2_hourly_gas_demand.csv"
    
    print(f"\n{'='*60}")
    print(f"PEM Electrolyser Simulation - {size_mw} MW")
    print(f"{'='*60}")
    print(f"Storage capacity: {storage_kg:.0f} kg ({storage_kg/size_mw:.0f} kg/MW)")
    
    try:
        # Load data
        print("Loading power data...")
        power_1yr = load_power_data(mat_path)
        
        print("Loading demand data...")
        demand_1yr = load_demand_data(demand_csv_path, power_1yr.index)
        
        # Create configuration
        cfg = get_config(size_mw=size_mw, storage_kg=storage_kg)
        
        # Synthesize multi-year data
        print(f"Synthesizing {cfg['YEARS']}-year data...")
        power_5yr, demand_5yr = synthesize_multiyear_data(
            power_1yr, demand_1yr, cfg['YEARS'], rng)
        
        # Run Monte Carlo if requested
        if run_mc == 'y':
            try:
                n_sim = int(input("Number of Monte Carlo simulations [100]: ") or "100")
            except (ValueError, EOFError):
                n_sim = 100
            
            print(f"\nRunning Monte Carlo analysis with {n_sim} simulations...")
            mc_output = run_monte_carlo(
                size_mw=size_mw,
                storage_kg=storage_kg,
                power_1yr=power_1yr,
                demand_1yr=demand_1yr,
                n_simulations=n_sim,
                output_folder=sim_folder,
                verbose=True
            )
            
            print(f"\n✓ Monte Carlo analysis complete. Results saved in {sim_folder}/")
            return  # Exit after Monte Carlo
        
        # Run standard simulation
        print("Running simulation...")
        df = simulate(cfg, power_5yr, demand_5yr, rng)
        
        # Compute economics
        print("Computing economics...")
        econ = compute_economics(cfg, df, cfg.get('H2_SELLING_PRICE_EUR_PER_KG'))
        
        # Print results
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Electrolyser size: {size_mw} MW")
        print(f"Simulation period: {cfg['YEARS']} years")
        print(f"LCOH: {econ['LCOH_EUR_per_kg']:.2f} EUR/kg")
        print(f"Total H2 produced: {econ['total_H2_kg']:.0f} kg ({cfg['YEARS']} years)")
        print(f"NPV cost: {econ['NPV_cost_EUR']:.0f} EUR")
        
        # Print profitability metrics if calculated
        if 'NPV_profit_EUR' in econ:
            print(f"\n{'='*60}")
            print("PROFITABILITY ANALYSIS")
            print(f"{'='*60}")
            print(f"H2 Selling Price: {econ['h2_selling_price_EUR_per_kg']:.2f} EUR/kg")
            print(f"Total Revenue: {econ['total_revenue_EUR']:,.0f} EUR")
            print(f"Total Profit (undiscounted): {econ['total_profit_EUR']:,.0f} EUR")
            print(f"NPV of Profit: {econ['NPV_profit_EUR']:,.0f} EUR")
            if econ['IRR_percent'] is not None:
                print(f"IRR: {econ['IRR_percent']:.2f}%")
            else:
                print(f"IRR: Not calculable (negative cash flows)")
            print(f"Payback Period: {econ['payback_period_years']:.1f} years")
            print(f"ROI: {econ['ROI_percent']:.1f}%")
        print(f"NPV cost: {econ['NPV_cost_EUR']:.0f} EUR")
        print(f"Downtime: {econ['downtime_hours_total']} hours")
        print(f"Stack replacements: {len(econ['stack_replacement_years'])}")
        print(f"Minor maintenance events: {len(econ['minor_maintenance_hours'])}")
        
        # Thermal management results (NEW)
        print(f"\n{'='*60}")
        print("THERMAL MANAGEMENT")
        print(f"{'='*60}")
        print(f"Heat recovered: {econ['total_heat_recovered_kWh']/1e6:.2f} GWh")
        print(f"Heat revenue: {econ['total_heat_revenue_EUR']:,.0f} EUR")
        print(f"LCOH with heat credit: {econ['LCOH_with_heat_credit_EUR_per_kg']:.2f} EUR/kg")
        print(f"LCOH reduction from heat: {(econ['LCOH_EUR_per_kg'] - econ['LCOH_with_heat_credit_EUR_per_kg']):.3f} EUR/kg")
        
        # Save outputs
        print(f"\nSaving results to {sim_folder}/...")
        
        # Full timeseries CSV
        df.to_csv(f"{sim_folder}/timeseries_full_{cfg['YEARS']}years_PEM.csv")
        
        # Summary CSV (updated with thermal data)
        summary_dict = {
            'size_mw': size_mw,
            'storage_kg': storage_kg,
            'years': cfg['YEARS'],
            'LCOH_EUR_per_kg': econ['LCOH_EUR_per_kg'],
            'LCOH_with_heat_credit_EUR_per_kg': econ['LCOH_with_heat_credit_EUR_per_kg'],
            'total_H2_kg': econ['total_H2_kg'],
            'NPV_cost_EUR': econ['NPV_cost_EUR'],
            'capex_total': econ['capex_total'],
            'maint_cost_total': econ['maint_cost_total'],
            'downtime_hours': econ['downtime_hours_total'],
            'stack_replacements': len(econ['stack_replacement_years']),
            # Only average SEC over operating hours (non-zero values)
            'SEC_stack_mean': df.loc[df['SEC_stack_kWh_per_kg'] > 0, 'SEC_stack_kWh_per_kg'].mean(),
            'SEC_total_mean': df.loc[df['SEC_total_kWh_per_kg'] > 0, 'SEC_total_kWh_per_kg'].mean(),
            # Thermal metrics (NEW)
            'heat_recovered_GWh': econ['total_heat_recovered_kWh'] / 1e6,
            'heat_revenue_EUR': econ['total_heat_revenue_EUR'],
            'avg_stack_temp_C': df['T_stack_C'].mean(),
        }
        pd.DataFrame([summary_dict]).to_csv(f"{sim_folder}/summary_PEM.csv", index=False)
        
        # 6-month metrics
        df_metrics = []
        period = 4380  # 6 months in hours
        n_hours = len(df)
        for i in range(0, n_hours, period):
            df_slice = df.iloc[i:i+period]
            if len(df_slice) == 0:
                continue
            # Only average SEC and V_cell over operating hours
            sec_stack_op = df_slice.loc[df_slice['SEC_stack_kWh_per_kg'] > 0, 'SEC_stack_kWh_per_kg']
            sec_total_op = df_slice.loc[df_slice['SEC_total_kWh_per_kg'] > 0, 'SEC_total_kWh_per_kg']
            vcell_op = df_slice.loc[df_slice['V_cell_V'] > 0, 'V_cell_V']
            df_metrics.append({
                "period_start": df_slice.index[0],
                "period_end": df_slice.index[-1],
                "H2_kg": df_slice["H2_kg"].sum(),
                "sec_stack": sec_stack_op.mean() if len(sec_stack_op) > 0 else 0,
                "sec_total": sec_total_op.mean() if len(sec_total_op) > 0 else 0,
                "vcell_mean": vcell_op.mean() if len(vcell_op) > 0 else 0,
            })
        pd.DataFrame(df_metrics).to_csv(f"{sim_folder}/metrics_6month.csv", index=False)
        
        # Generate plots
        print("Generating plots...")
        generate_all_plots(df, econ, cfg, sim_folder, size_mw)
        
        print(f"\n✓ Simulation complete. All results saved in {sim_folder}/")
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found - {e}")
        print("Please update the file paths in the script.")
        return
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return


def run_uncertainty_analysis_standalone(size_mw: float = 20.0, n_simulations: int = 100):
    """
    Standalone function to run Monte Carlo uncertainty analysis.
    Can be called programmatically without interactive prompts.
    
    Parameters:
    -----------
    size_mw : float
        Electrolyser size in MW
    n_simulations : int
        Number of Monte Carlo simulations
        
    Returns:
    --------
    dict : Monte Carlo output dictionary with results, stats, and sampled_params
    """
    storage_kg = 2500.0
    
    # File paths - use relative paths from project root
    script_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    mat_path = script_dir / "data" / "combined_wind_pv_DATA.mat"
    demand_csv_path = script_dir / "data" / "Company_2_hourly_gas_demand.csv"
    
    # Load data
    print("Loading data...")
    power_1yr = load_power_data(mat_path)
    demand_1yr = load_demand_data(demand_csv_path, power_1yr.index)
    
    # Output folder
    output_folder = f"results/{int(size_mw)}MW_sim"
    os.makedirs(output_folder, exist_ok=True)
    
    # Run Monte Carlo
    print(f"Running {n_simulations} Monte Carlo simulations...")
    output = run_monte_carlo(
        size_mw=size_mw,
        storage_kg=storage_kg,
        power_1yr=power_1yr,
        demand_1yr=demand_1yr,
        n_simulations=n_simulations,
        output_folder=output_folder,
        verbose=True
    )
    
    return output


if __name__ == "__main__":
    main()
