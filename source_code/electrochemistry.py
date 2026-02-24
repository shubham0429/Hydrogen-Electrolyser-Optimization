from __future__ import annotations
from dataclasses import dataclass
import math

FARADAY = 96485.0          # C/mol
M_H2 = 0.002016            # kg/mol
LHV_H2_kWh_per_kg = 33.33  # kWh/kg

# ---------------------------------------------------------------------
# PARAMETER DATA CLASSES
# ---------------------------------------------------------------------


@dataclass
class StackParams:
    # geometry
    cell_area_cm2: float = 3000.0
    n_cells: int = 270

    # electrochemistry (fresh stack)
    E_rev_V: float = 1.23
    j0_A_per_cm2: float = 0.01
    B_V: float = 0.05          # Tafel slope parameter
    R_ohm_Ohm_cm2: float = 0.2

    # degradation
    kv_V_per_damage: float = 0.2  # voltage rise between BOL and D=1

    # faradaic efficiency
    faradaic_eff: float = 0.98


@dataclass
class BoPParams:
    # rectifier & auxiliaries
    rectifier_eff: float = 0.97   # AC->DC
    aux_fraction_of_elec: float = 0.05  # auxiliary loads as fraction of DC electrolyser power

    # compression to 350 bar (electric energy)
    compression_SEC_kWh_per_kg: float = 2.5  # kWh/kg H2 for compression to 350 bar


@dataclass
class EconomicParams:
    capex_stack_per_kW: float = 1000.0     # €/kW
    capex_bop_per_kW: float = 300.0       # €/kW (including rectifier, auxiliaries)
    capex_compression_per_kgph: float = 800.0  # €/ (kg H2 / h) capacity
    fixed_opex_frac_per_year: float = 0.03     # fraction of CAPEX per year
    project_lifetime_years: int = 15
    discount_rate: float = 0.08


# ---------------------------------------------------------------------
# ELECTROCHEMISTRY HELPERS
# ---------------------------------------------------------------------


def cell_voltage_fresh(j_A_per_cm2: float, p: StackParams) -> float:
    """Baseline polarization curve for fresh stack (no degradation)."""
    if j_A_per_cm2 <= 0:
        return p.E_rev_V
    eta_act = p.B_V * math.log(1.0 + j_A_per_cm2 / p.j0_A_per_cm2)
    eta_ohm = p.R_ohm_Ohm_cm2 * j_A_per_cm2
    return p.E_rev_V + eta_act + eta_ohm


def cell_voltage_degraded(j_A_per_cm2: float, damage: float, p: StackParams) -> float:
    """Voltage including degradation (damage in [0,1])."""
    v0 = cell_voltage_fresh(j_A_per_cm2, p)
    return v0 + p.kv_V_per_damage * max(0.0, min(1.0, damage))


def stack_power_and_h2(
    j_A_per_cm2: float,
    damage: float,
    sp: StackParams,
    bp: BoPParams,
) -> dict:
    """
    Compute power & hydrogen flow for given current density and damage.

    Returns a dict with:
      I_stack_A, V_cell_V, V_stack_V,
      P_DC_elec_W, P_AC_elec_W,
      P_aux_W, P_comp_W, P_AC_total_W,
      H2_kg_per_h,
      SEC_electrolyser_kWh_per_kg,
      SEC_total_kWh_per_kg,
      j_A_per_cm2, damage
    """
    # geometry & current
    A_cell_cm2 = sp.cell_area_cm2
    I_cell_A = j_A_per_cm2 * A_cell_cm2          # cell current
    I_stack_A = I_cell_A                         # series stack: same current

    V_cell_V = cell_voltage_degraded(j_A_per_cm2, damage, sp)
    V_stack_V = V_cell_V * sp.n_cells

    P_DC_elec_W = I_stack_A * V_stack_V
    P_AC_elec_W = P_DC_elec_W / bp.rectifier_eff

    # hydrogen production (NOTE: multiply by n_cells!)
    m_dot_kg_per_s = (
        sp.n_cells * I_stack_A / (2.0 * FARADAY) * M_H2 * sp.faradaic_eff
    )
    H2_kg_per_h = m_dot_kg_per_s * 3600.0

    # Auxiliaries (cooling, pumps, control) as fraction of electrolyser DC power
    P_aux_W = bp.aux_fraction_of_elec * P_DC_elec_W

    # Compression to 350 bar
    # SEC_comp in kWh/kg, so power is SEC_comp * H2_kg_per_h (kW) => x1000 for W
    P_comp_W = bp.compression_SEC_kWh_per_kg * H2_kg_per_h * 1000.0

    # Total AC power
    P_AC_total_W = P_AC_elec_W + P_aux_W + P_comp_W

    # Specific energy consumption
    if H2_kg_per_h > 0:
        SEC_electrolyser_kWh_per_kg = P_AC_elec_W / 1000.0 / H2_kg_per_h
        SEC_total_kWh_per_kg = P_AC_total_W / 1000.0 / H2_kg_per_h
    else:
        SEC_electrolyser_kWh_per_kg = float("inf")
        SEC_total_kWh_per_kg = float("inf")

    return {
        "I_stack_A": I_stack_A,
        "V_cell_V": V_cell_V,
        "V_stack_V": V_stack_V,
        "P_DC_elec_W": P_DC_elec_W,
        "P_AC_elec_W": P_AC_elec_W,
        "P_aux_W": P_aux_W,
        "P_comp_W": P_comp_W,
        "P_AC_total_W": P_AC_total_W,
        "H2_kg_per_h": H2_kg_per_h,
        "SEC_electrolyser_kWh_per_kg": SEC_electrolyser_kWh_per_kg,
        "SEC_total_kWh_per_kg": SEC_total_kWh_per_kg,
        "damage": damage,
        "j_A_per_cm2": j_A_per_cm2,
    }


# ---------------------------------------------------------------------
# ECONOMICS
# ---------------------------------------------------------------------


def capital_costs_at_nominal(
    j_nom_A_per_cm2: float,
    damage_nom: float,
    sp: StackParams,
    bp: BoPParams,
    ep: EconomicParams,
) -> dict:
    """
    Compute CAPEX and simple annualised CAPEX based on nominal operating point.
    We use nominal DC power and nominal H2 production to size equipment.
    """
    perf = stack_power_and_h2(j_nom_A_per_cm2, damage_nom, sp, bp)

    P_DC_nom_kW = perf["P_DC_elec_W"] / 1000.0
    H2_nom_kgph = perf["H2_kg_per_h"]

    # Electrolyser (stack + BoP) CAPEX
    capex_stack = ep.capex_stack_per_kW * P_DC_nom_kW
    capex_bop = ep.capex_bop_per_kW * P_DC_nom_kW

    # Compression sized by H2 flow (kg/h)
    capex_comp = ep.capex_compression_per_kgph * H2_nom_kgph

    capex_total = capex_stack + capex_bop + capex_comp

    # Annualization (Capital Recovery Factor)
    r = ep.discount_rate
    n = ep.project_lifetime_years
    crf = (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    annualized_capex = capex_total * crf
    fixed_opex = ep.fixed_opex_frac_per_year * capex_total
    annual_fixed_cost = annualized_capex + fixed_opex

    return {
        "P_DC_nom_kW": P_DC_nom_kW,
        "H2_nom_kgph": H2_nom_kgph,
        "capex_stack": capex_stack,
        "capex_bop": capex_bop,
        "capex_comp": capex_comp,
        "capex_total": capex_total,
        "annualized_capex": annualized_capex,
        "fixed_opex": fixed_opex,
        "annual_fixed_cost": annual_fixed_cost,
        "CRF": crf,
    }


# ---------------------------------------------------------------------
# SIMPLE TEST / DEMO WHEN RUN AS SCRIPT
# ---------------------------------------------------------------------


def print_performance(label: str, perf: dict) -> None:
    print(f"\n{label}:")
    for k in [
        "I_stack_A",
        "V_cell_V",
        "V_stack_V",
        "P_DC_elec_W",
        "P_AC_elec_W",
        "P_aux_W",
        "P_comp_W",
        "P_AC_total_W",
        "H2_kg_per_h",
        "SEC_electrolyser_kWh_per_kg",
        "SEC_total_kWh_per_kg",
        "damage",
        "j_A_per_cm2",
    ]:
        print(f"{k:35s} = {perf[k]}")


if __name__ == "__main__":
    sp = StackParams()
    bp = BoPParams()
    ep = EconomicParams()

    j_test = 1.5  # A/cm2

    perf_BOL = stack_power_and_h2(j_test, damage=0.0, sp=sp, bp=bp)
    perf_EoL = stack_power_and_h2(j_test, damage=0.8, sp=sp, bp=bp)

    print_performance("BOL at j=1.5 A/cm² (with BoP)", perf_BOL)
    print_performance("Near EOL at j=1.5 A/cm² (with BoP)", perf_EoL)

    econ_nom = capital_costs_at_nominal(j_nom_A_per_cm2=j_test, damage_nom=0.0,
                                        sp=sp, bp=bp, ep=ep)

    print("\nEconomics at nominal BOL operation:")
    for k, v in econ_nom.items():
        print(f"{k:25s} = {v}")
