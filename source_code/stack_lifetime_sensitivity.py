import sys
import numpy as np
import scipy.io
import warnings

sys.path.insert(0, '/Users/shubhammanchanda/Thesis_project-main/src')
warnings.filterwarnings('ignore')

mat = scipy.io.loadmat(
    '/Users/shubhammanchanda/Thesis_project-main/data/combined_wind_pv_DATA.mat',
    squeeze_me=True)
pv = np.array(mat['P_PV']).flatten()
wind = np.array(mat['P_wind_selected']).flatten()
power_W = pv + wind

RE_FRACS = [0.20, 0.40, 0.50, 0.60, 0.80, 1.00]
SIZE_MW = 20
STORAGE_KG = 3000
SIM_YEARS = 15
RNG_SEED = 42
PEM_LIFETIMES = [60000, 70000, 80000]
ALK_LIFETIMES = [80000, 90000, 100000]

from sim_concise import (get_config, simulate as pem_simulate,
                         compute_economics, load_power_data,
                         load_demand_data, synthesize_multiyear_data)

DATA_DIR = '/Users/shubhammanchanda/Thesis_project-main/data'
power_1yr = load_power_data(DATA_DIR + '/combined_wind_pv_DATA.mat')
demand_1yr = load_demand_data(DATA_DIR + '/Company_2_hourly_gas_demand.csv', power_1yr.index)

print('=' * 100)
print('PEM STACK LIFETIME SENSITIVITY')
print('=' * 100)
print(f"{'RE%':>5}  | {'60kh':>10} {'70kh':>10} {'80kh':>10} | {'60kh CR':>10} {'70kh CR':>10} {'80kh CR':>10} | Repl")
print('-' * 100)

pem_results = {}
for re_frac in RE_FRACS:
    pem_results[re_frac] = {}
    for lt in PEM_LIFETIMES:
        rng = np.random.default_rng(RNG_SEED)
        cfg = get_config(size_mw=SIZE_MW, storage_kg=STORAGE_KG)
        cfg['YEARS'] = SIM_YEARS
        cfg['STACK_LIFETIME_HOURS'] = lt
        cfg['STACK_LIFE_HOURS'] = float(lt)
        power_scaled = power_1yr * re_frac
        power_multi, demand_multi = synthesize_multiyear_data(
            power_scaled, demand_1yr, SIM_YEARS, rng, deterministic=True)
        df = pem_simulate(cfg, power_multi, demand_multi, rng)
        eco = compute_economics(cfg, df, h2_selling_price_eur_per_kg=12.0)
        lcoh = eco['LCOH_EUR_per_kg']
        lcoh_cr = eco['LCOH_with_credits_EUR_per_kg']
        n_repl = len(eco['stack_replacement_years'])
        pem_results[re_frac][lt] = {'lcoh': lcoh, 'lcoh_cr': lcoh_cr, 'repl': n_repl}
    v = [pem_results[re_frac][l]['lcoh'] for l in PEM_LIFETIMES]
    c = [pem_results[re_frac][l]['lcoh_cr'] for l in PEM_LIFETIMES]
    r = [pem_results[re_frac][l]['repl'] for l in PEM_LIFETIMES]
    print(f"{re_frac*100:5.0f}%  | {v[0]:10.2f} {v[1]:10.2f} {v[2]:10.2f} | {c[0]:10.2f} {c[1]:10.2f} {c[2]:10.2f} | {r[0]},{r[1]},{r[2]}")
    sys.stdout.flush()

from sim_alkaline import get_alkaline_config, simulate as alk_simulate, compute_lcoh

print()
print('=' * 100)
print('ALKALINE STACK LIFETIME SENSITIVITY')
print('=' * 100)
print(f"{'RE%':>5}  | {'80kh':>10} {'90kh':>10} {'100kh':>10} | Repl")
print('-' * 80)

alk_results = {}
for re_frac in RE_FRACS:
    alk_results[re_frac] = {}
    for lt in ALK_LIFETIMES:
        pw = power_W * re_frac
        pw_multi = np.tile(pw, SIM_YEARS)[:SIM_YEARS * 8760]
        acfg = get_alkaline_config(P_nom_MW=SIZE_MW, simulation_years=SIM_YEARS, storage_capacity_kg=STORAGE_KG)
        acfg.stack_lifetime_hours = lt
        res = alk_simulate(pw_multi, acfg, verbose=False)
        eco_a = compute_lcoh(res, acfg)
        lcoh_a = eco_a.lcoh_total
        n_repl_a = 0
        for attr in ['n_replacements', 'num_replacements', 'total_replacements', 'n_stack_replacements']:
            val = getattr(eco_a, attr, None)
            if val is None:
                val = getattr(res, attr, None)
            if val is not None and val > 0:
                n_repl_a = int(val)
                break
        alk_results[re_frac][lt] = {'lcoh': lcoh_a, 'repl': n_repl_a}
    va = [alk_results[re_frac][l]['lcoh'] for l in ALK_LIFETIMES]
    ra = [alk_results[re_frac][l]['repl'] for l in ALK_LIFETIMES]
    print(f"{re_frac*100:5.0f}%  | {va[0]:10.2f} {va[1]:10.2f} {va[2]:10.2f} | {ra[0]},{ra[1]},{ra[2]}")
    sys.stdout.flush()

print()
print('=' * 100)
print('COMBINED SUMMARY')
print('=' * 100)
print(f"{'RE%':>5} | {'PEM LCOH':>16} | {'PEM w/Credits':>16} | {'ALK LCOH':>16} | {'Best Gap':>9} | {'Worst Gap':>10}")
print('-' * 90)

for re_frac in RE_FRACS:
    p_vals = [pem_results[re_frac][l]['lcoh'] for l in PEM_LIFETIMES]
    p_crs = [pem_results[re_frac][l]['lcoh_cr'] for l in PEM_LIFETIMES]
    a_vals = [alk_results[re_frac][l]['lcoh'] for l in ALK_LIFETIMES]
    best_gap = (min(p_crs) / max(a_vals) - 1) * 100
    worst_gap = (max(p_vals) / min(a_vals) - 1) * 100
    print(f"{re_frac*100:5.0f}% | {min(p_vals):6.2f}-{max(p_vals):6.2f}   | {min(p_crs):6.2f}-{max(p_crs):6.2f}   | {min(a_vals):6.2f}-{max(a_vals):6.2f}   | {best_gap:+7.0f}% | {worst_gap:+8.0f}%")

print()
print('DONE')
