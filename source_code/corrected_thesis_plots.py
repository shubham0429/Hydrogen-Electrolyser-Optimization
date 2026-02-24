#!/usr/bin/env python3
"""
Corrected Thesis Plots — v3 (all audit fixes applied)
======================================================

Fixes applied:
  1. RE comparison: infeasible band ends at 79 (not 75); text uses transAxes
  2. PEM performance: annotation positions use axes fraction, not premature ylim
  3. ALK performance: event detection from data + schedule lookup; labels shown;
     twin-axis right spine explicitly enabled
  4. Heatmaps: uniform 10x10 grid for all (PEM pruned cells shown gray "N/A");
     LCOH vmax capped at 95th percentile to prevent colormap washout
  5. Summary table: unchanged (was already correct)

Output: results/thesis_final_plots/

Usage:
    cd source_code
    python corrected_thesis_plots.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
PEM_CSV  = BASE / "results" / "data" / "pem_grid_search_all_RE.csv"
ALK_CSV  = BASE / "results" / "data" / "alkaline_grid_search_all_RE.csv"
PEM_TS   = BASE / "results" / "pem_thesis_final" / "baseline_timeseries.csv"
ALK_TS   = BASE / "results" / "alkaline_thesis_final" / "baseline_timeseries.csv"
OUTPUT   = BASE / "results" / "thesis_final_plots"
OUTPUT.mkdir(parents=True, exist_ok=True)

RELIABILITY_TARGET = 95.0

# ── Palette ───────────────────────────────────────────────────────────────────
C_PEM       = '#1565C0'
C_ALK       = '#2E7D32'
C_PEM_LIGHT = '#BBDEFB'
C_ALK_LIGHT = '#C8E6C9'

# ── Light-only colormaps ──────────────────────────────────────────────────────
_lcoh_cmap = LinearSegmentedColormap.from_list(
    'lcoh_light', ['#FFFFFF', '#FFF8E1', '#FFE0B2', '#FFCC80', '#FFB74D'], N=256)

_demand_cmap = LinearSegmentedColormap.from_list(
    'demand_light', ['#FFCDD2', '#FFE0E0', '#FFFFFF', '#E8F5E9', '#C8E6C9'], N=256)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    print(f"  -> {name}.png + .pdf")


def select_optimal(df):
    feasible = df[df['demand_met_pct'] >= RELIABILITY_TARGET]
    if len(feasible) > 0:
        return feasible.sort_values(['LCOH', 'SEC']).iloc[0], True
    else:
        return df.sort_values('demand_met_pct', ascending=False).iloc[0], False


# =============================================================================
# PLOT 1: RE FRACTION vs LCOH
# =============================================================================
def plot_re_fraction_lcoh():
    print("\n[1/5] RE Fraction vs LCOH...")
    pem = pd.read_csv(PEM_CSV)
    alk = pd.read_csv(ALK_CSV)
    re_fracs = sorted(pem['re_fraction'].unique())

    rows = {'PEM': [], 'Alkaline': []}
    for re in re_fracs:
        for label, full in [('PEM', pem), ('Alkaline', alk)]:
            best, feas = select_optimal(full[full['re_fraction'] == re])
            rows[label].append(dict(
                re=re*100, lcoh=best['LCOH'], size=best['size_MW'],
                storage=best['storage_kg'], demand=best['demand_met_pct'],
                feasible=feas))

    pem_df = pd.DataFrame(rows['PEM'])
    alk_df = pd.DataFrame(rows['Alkaline'])

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(pem_df['re'], pem_df['lcoh'], 'o-', color=C_PEM, lw=2.5, ms=9, zorder=4)
    ax.plot(alk_df['re'], alk_df['lcoh'], 's-', color=C_ALK, lw=2.5, ms=9, zorder=4)

    for df_t, clr, mk, lab in [
        (pem_df, C_PEM, 'o', 'PEM'), (alk_df, C_ALK, 's', 'Alkaline')]:
        feas   = df_t[df_t['feasible']]
        infeas = df_t[~df_t['feasible']]
        if len(feas):
            ax.scatter(feas['re'], feas['lcoh'], s=130, marker=mk, color=clr,
                       edgecolors='white', linewidths=2, zorder=6,
                       label=f'{lab} -- feasible (>=95% demand)')
        if len(infeas):
            ax.scatter(infeas['re'], infeas['lcoh'], s=130, marker=mk,
                       facecolors='none', edgecolors=clr, linewidths=2.5,
                       zorder=6, label=f'{lab} -- infeasible (<95%)')

    for df_t, clr in [(pem_df, C_PEM), (alk_df, C_ALK)]:
        feas = df_t[df_t['feasible']]
        if len(feas):
            opt = feas.loc[feas['lcoh'].idxmin()]
            ax.scatter(opt['re'], opt['lcoh'], s=350, marker='*',
                       color='#FFD600', edgecolors=clr, linewidths=1.5, zorder=10)
            ax.annotate(
                f"{opt['lcoh']:.2f} EUR/kg\n{opt['size']:.0f} MW, "
                f"{opt['storage']:,.0f} kg\nDemand: {opt['demand']:.1f}%",
                xy=(opt['re'], opt['lcoh']),
                xytext=(opt['re']-22, opt['lcoh']+(1.2 if clr == C_PEM else -1.8)),
                fontsize=10, color=clr, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=clr, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=clr, alpha=0.95))

    p100 = pem_df[pem_df['re'] == 100].iloc[0]
    a100 = alk_df[alk_df['re'] == 100].iloc[0]
    gap  = p100['lcoh'] - a100['lcoh']
    ax.annotate(f"Cost gap: {gap:.2f} EUR/kg ({gap/p100['lcoh']*100:.0f}%)",
                xy=(100, (p100['lcoh']+a100['lcoh'])/2),
                xytext=(68, (p100['lcoh']+a100['lcoh'])/2 + 0.6),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='gray'))

    # FIX #5: band to 79, text via transAxes
    ax.axvspan(15, 79, alpha=0.04, color='red')
    ax.text(0.30, 0.95,
            'No config meets 95% demand\nbelow 80% RE availability',
            transform=ax.transAxes,
            fontsize=9, ha='center', va='top', style='italic', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#E57373', alpha=0.8))

    ax.set_xlabel('Renewable Energy Fraction (%)', fontweight='bold')
    ax.set_ylabel('Optimal LCOH (EUR/kg H$_2$)', fontweight='bold')
    ax.set_xlim(15, 105)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.25, ls='--')
    ax.set_title('LCOH vs Renewable Energy Availability -- Constrained Optimization',
                 fontweight='bold', pad=12)
    fig.tight_layout()
    save(fig, 'fig_lcoh_vs_re_comparison')
    plt.close(fig)

    print("    RE%   PEM        ALK")
    for _, p in pem_df.iterrows():
        a = alk_df[alk_df['re'] == p['re']].iloc[0]
        pf = 'YES' if p['feasible'] else 'no '
        af = 'YES' if a['feasible'] else 'no '
        print(f"    {p['re']:3.0f}%  {p['lcoh']:6.2f} {pf}   {a['lcoh']:6.2f} {af}")


# =============================================================================
# PLOT 2a: PEM 15-year performance
# =============================================================================
def plot_pem_performance():
    print("\n[2/5] PEM Performance Evolution...")
    df = pd.read_csv(PEM_TS)
    hours = np.arange(len(df))
    df['year'] = hours / 8760.0

    df['month_idx'] = (hours // 730).astype(int)
    op = df[df['H2_kg'] > 0].copy()
    monthly = op.groupby('month_idx').agg(
        SEC_total=('SEC_total_kWh_per_kg', 'mean'),
        SEC_stack=('SEC_stack_kWh_per_kg', 'mean'),
        V_cell=('V_cell_V', 'mean'),
        H2=('H2_kg', 'sum'),
    ).reset_index()
    monthly['year'] = monthly['month_idx'] * 730 / 8760
    monthly['cum_h2_t'] = monthly['H2'].cumsum() / 1000

    yearly_v = op.groupby((op['year']).astype(int))['V_cell_V'].mean()
    repl_years = [yr for yr in range(1, len(yearly_v))
                  if yearly_v.iloc[yr] < yearly_v.iloc[yr-1] - 0.02]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.38)

    # (a) SEC
    ax = axes[0]
    ax.plot(monthly['year'], monthly['SEC_total'], color=C_PEM, lw=2,
            label='SEC system (stack + BoP + compression)')
    ax.plot(monthly['year'], monthly['SEC_stack'], '--', color=C_PEM, lw=1.2,
            alpha=0.6, label='SEC stack only')
    # FIX #2: use axes-fraction for y position
    for yr in repl_years:
        ax.axvline(yr, color='#E53935', ls='--', lw=1.3, alpha=0.7)
        ax.annotate('Stack\nreplaced',
                    xy=(yr, 0), xycoords=('data', 'axes fraction'),
                    xytext=(yr+0.3, 0.15), textcoords=('data', 'axes fraction'),
                    fontsize=8, color='#E53935', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E53935', lw=1))
    ax.set_ylabel('SEC (kWh / kg H$_2$)', fontweight='bold')
    ax.set_title('(a)  Specific Energy Consumption', fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.25, ls='--')
    ax.text(0.98, 0.95, 'SEC rises with degradation,\ndrops at stack replacement',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

    # (b) Cell voltage
    ax = axes[1]
    ax.plot(monthly['year'], monthly['V_cell'], color='#E65100', lw=2)
    for yr in repl_years:
        ax.axvline(yr, color='#E53935', ls='--', lw=1.3, alpha=0.7)
    ax.set_ylabel('Cell Voltage (V)', fontweight='bold')
    ax.set_title('(b)  Cell Voltage Degradation', fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--')

    # (c) Cumulative H2
    ax = axes[2]
    ax.fill_between(monthly['year'], 0, monthly['cum_h2_t'], color=C_PEM_LIGHT, alpha=0.6)
    ax.plot(monthly['year'], monthly['cum_h2_t'], color=C_PEM, lw=2)
    tot = monthly['cum_h2_t'].iloc[-1]
    ax.text(0.98, 0.08, f'Total: {tot:,.0f} t over 15 yr  ({tot/15:,.0f} t/yr avg)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', fc='white', ec=C_PEM, alpha=0.95))
    ax.set_ylabel('Cumulative H$_2$ (tonnes)', fontweight='bold')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_title('(c)  Cumulative Hydrogen Production', fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--')

    for a in axes:
        a.set_xlim(0, 15)
    fig.suptitle('PEM Electrolyser -- 15-Year Performance Evolution',
                 fontsize=15, fontweight='bold', y=1.01)
    save(fig, 'fig_pem_performance_evolution')
    plt.close(fig)


# =============================================================================
# PLOT 2b: ALKALINE 15-year performance  (FIX #3 + #4 — event detection,
#          labels on event lines, twin-axis spine)
# =============================================================================
def plot_alkaline_performance():
    print("\n[3/5] Alkaline Performance Evolution...")
    if not ALK_TS.exists():
        print("  SKIP -- alkaline baseline_timeseries.csv not found")
        return

    df = pd.read_csv(ALK_TS)
    df['year'] = df['hour'] / 8760.0
    df['month_idx'] = (df['hour'] // 730).astype(int)
    op = df[df['H2_kg'] > 0].copy()
    monthly = op.groupby('month_idx').agg(
        SEC=('SEC_kWh_kg', 'mean'),
        V_cell=('V_cell_V', 'mean'),
        V_deg=('V_degradation_V', 'mean'),
        H2=('H2_kg', 'sum'),
    ).reset_index()
    monthly['year'] = monthly['month_idx'] * 730 / 8760
    monthly['cum_h2_t'] = monthly['H2'].cumsum() / 1000

    # Yearly summary
    op['yr_int'] = (op['hour'] // 8760).astype(int)
    yearly = op.groupby('yr_int').agg(
        SEC=('SEC_kWh_kg', 'mean'),
        V_cell=('V_cell_V', 'mean'),
        V_deg=('V_degradation_V', 'mean'),
    ).reset_index()

    # FIX #3a: Detect events from actual data, label from known schedule
    replacement_schedule = {
        4:  'Electrolyte',
        6:  'Mechanical',
        8:  'Electrolyte',
        10: 'Diaphragm',
        12: 'Electrolyte +\nCatalyst + Mech.',
    }
    events = []
    for i in range(1, len(yearly)):
        sec_drop = yearly.iloc[i-1]['SEC'] - yearly.iloc[i]['SEC']
        vdeg_drop = yearly.iloc[i-1]['V_deg'] - yearly.iloc[i]['V_deg']
        yr = yearly.iloc[i]['yr_int']
        if vdeg_drop > 0.005:
            events.append((yr, 'Stack +\ncomponents', '#D50000'))
        elif sec_drop > 0.8:
            label = replacement_schedule.get(yr, 'Component')
            events.append((yr, label, '#E65100'))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.38)

    # (a) SEC
    ax = axes[0]
    ax.plot(monthly['year'], monthly['SEC'], color=C_ALK, lw=2.2,
            label='SEC system (kWh/kg)')

    # FIX #3b: vertical lines WITH text labels
    for idx, (yr, label, clr) in enumerate(events):
        ax.axvline(yr, color=clr, ls='--', lw=1.3, alpha=0.7)
        # Alternate label y position to avoid overlap
        y_frac = 0.92 if idx % 2 == 0 else 0.78
        ax.annotate(label, xy=(yr, y_frac), xycoords=('data', 'axes fraction'),
                    fontsize=8, color=clr, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=clr, alpha=0.9))

    sec_y0 = yearly.iloc[0]['SEC']
    sec_y14 = yearly.iloc[-1]['SEC']
    ax.annotate(f'Year 0: {sec_y0:.1f} kWh/kg',
                xy=(0.5, sec_y0), xytext=(2.5, sec_y0+1.5),
                fontsize=10, color=C_ALK, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ALK),
                bbox=dict(boxstyle='round', fc='white', ec=C_ALK, alpha=0.9))
    ax.annotate(f'Year 14: {sec_y14:.1f} kWh/kg',
                xy=(14, sec_y14), xytext=(11, sec_y14-2.5),
                fontsize=10, color=C_ALK, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_ALK),
                bbox=dict(boxstyle='round', fc='white', ec=C_ALK, alpha=0.9))

    ax.set_ylabel('SEC (kWh / kg H$_2$)', fontweight='bold')
    ax.set_title('(a)  Specific Energy Consumption', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.25, ls='--')
    ax.text(0.02, 0.05,
            'SEC decreases due to component replacement\n'
            'efficiency boosts (electrolyte +4% every 4 yr,\n'
            'catalyst +5% at yr 12, diaphragm +3% at yr 10)',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', fc='#E8F5E9', ec=C_ALK, alpha=0.85))

    # (b) Cell voltage + degradation (FIX #4: enable right spine)
    ax = axes[1]
    ax.plot(monthly['year'], monthly['V_cell'], color='#E65100', lw=2,
            label='Cell voltage')
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)   # override global rcParams
    ax2.spines['top'].set_visible(False)
    ax2.plot(monthly['year'], monthly['V_deg']*1000, color='#7B1FA2', lw=1.5,
             ls='--', alpha=0.7, label='Cumulative degradation')
    ax2.set_ylabel('Voltage Degradation (mV)', fontweight='bold', color='#7B1FA2')
    ax2.tick_params(axis='y', labelcolor='#7B1FA2')

    for yr, label, clr in events:
        ax.axvline(yr, color=clr, ls='--', lw=1.3, alpha=0.6)

    ax.set_ylabel('Cell Voltage (V)', fontweight='bold')
    ax.set_title('(b)  Cell Voltage & Degradation', fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.25, ls='--')

    # (c) Cumulative H2
    ax = axes[2]
    ax.fill_between(monthly['year'], 0, monthly['cum_h2_t'], color=C_ALK_LIGHT, alpha=0.6)
    ax.plot(monthly['year'], monthly['cum_h2_t'], color=C_ALK, lw=2)
    tot = monthly['cum_h2_t'].iloc[-1]
    ax.text(0.98, 0.08,
            f'Total: {tot:,.0f} t over 15 yr  ({tot/15:,.0f} t/yr avg)',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', fc='white', ec=C_ALK, alpha=0.95))
    ax.set_ylabel('Cumulative H$_2$ (tonnes)', fontweight='bold')
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_title('(c)  Cumulative Hydrogen Production', fontweight='bold')
    ax.grid(True, alpha=0.25, ls='--')

    for a in [axes[0], axes[1], axes[2]]:
        a.set_xlim(0, 15)
    fig.suptitle('Alkaline Electrolyser -- 15-Year Performance Evolution',
                 fontsize=15, fontweight='bold', y=1.01)
    save(fig, 'fig_alkaline_performance_evolution')
    plt.close(fig)


# =============================================================================
# PLOT 3: HEATMAPS  (FIX #4/#6 — uniform grid, vmax cap, gray N/A cells)
# =============================================================================
# Full uniform grid — same for EVERY heatmap
ALL_SIZES  = list(range(5, 55, 5))    # 5,10,...,50 MW
ALL_STORES = [2000, 4000, 6000, 8000, 10000, 12000, 15000, 18000, 21000, 24000]


def plot_heatmaps():
    print("\n[4/5] Optimization Heatmaps (uniform 10x10 grid)...")

    datasets = []
    if PEM_CSV.exists():
        datasets.append(('PEM', pd.read_csv(PEM_CSV)))
    if ALK_CSV.exists():
        datasets.append(('Alkaline', pd.read_csv(ALK_CSV)))

    ns = len(ALL_SIZES)
    nt = len(ALL_STORES)
    records = []

    for tech, df_full in datasets:
        for re_frac in sorted(df_full['re_fraction'].unique()):
            df = df_full[df_full['re_fraction'] == re_frac].copy()
            best, is_feasible = select_optimal(df)

            # Pivot onto the FULL uniform grid — missing combos become NaN
            lcoh_piv   = df.pivot_table('LCOH', 'storage_kg', 'size_MW').reindex(
                             index=ALL_STORES, columns=ALL_SIZES)
            demand_piv = df.pivot_table('demand_met_pct', 'storage_kg', 'size_MW').reindex(
                             index=ALL_STORES, columns=ALL_SIZES)

            fig, (ax_l, ax_d) = plt.subplots(1, 2, figsize=(22, 10))

            # ── LEFT: LCOH ────────────────────────────────────────────────
            # Cap vmax at 95th percentile to prevent outlier washout
            lcoh_flat = lcoh_piv.values[np.isfinite(lcoh_piv.values)]
            vmin_l = np.min(lcoh_flat)
            vmax_l = float(np.percentile(lcoh_flat, 95)) if len(lcoh_flat) > 5 else np.max(lcoh_flat)
            if vmax_l <= vmin_l:
                vmax_l = vmin_l + 1.0

            im1 = ax_l.imshow(lcoh_piv.values, cmap=_lcoh_cmap, aspect='auto',
                              origin='lower', vmin=vmin_l, vmax=vmax_l)
            cb1 = fig.colorbar(im1, ax=ax_l, shrink=0.82, pad=0.02)
            cb1.set_label('LCOH (EUR / kg H$_2$)', fontsize=13, fontweight='bold')
            cb1.ax.tick_params(labelsize=11)

            for i in range(nt):
                for j in range(ns):
                    v = lcoh_piv.iloc[i, j]
                    if np.isfinite(v):
                        txt = f'{v:.1f}' if v <= vmax_l * 1.5 else f'>{vmax_l:.0f}'
                        ax_l.text(j, i, txt, ha='center', va='center',
                                  fontsize=11, color='black', fontweight='bold')
                    else:
                        ax_l.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       facecolor='#E0E0E0', edgecolor='#BDBDBD', zorder=2))
                        ax_l.text(j, i, 'N/A', ha='center', va='center',
                                  fontsize=9, color='#757575', style='italic', zorder=3)

            ax_l.set_title('LCOH  (EUR / kg H$_2$)', fontsize=15, fontweight='bold', pad=12)

            # ── RIGHT: DEMAND MET ─────────────────────────────────────────
            im2 = ax_d.imshow(demand_piv.values, cmap=_demand_cmap, aspect='auto',
                              origin='lower', vmin=0, vmax=100)
            cb2 = fig.colorbar(im2, ax=ax_d, shrink=0.82, pad=0.02)
            cb2.set_label('Demand Satisfied (%)', fontsize=13, fontweight='bold')
            cb2.ax.tick_params(labelsize=11)

            for i in range(nt):
                for j in range(ns):
                    v = demand_piv.iloc[i, j]
                    if np.isfinite(v):
                        wt = 'bold' if v >= RELIABILITY_TARGET else 'normal'
                        ax_d.text(j, i, f'{v:.0f}%', ha='center', va='center',
                                  fontsize=11, color='black', fontweight=wt)
                    else:
                        ax_d.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       facecolor='#E0E0E0', edgecolor='#BDBDBD', zorder=2))
                        ax_d.text(j, i, 'N/A', ha='center', va='center',
                                  fontsize=9, color='#757575', style='italic', zorder=3)

            # 95% contour
            try:
                X, Y = np.meshgrid(np.arange(ns), np.arange(nt))
                d_masked = np.where(np.isfinite(demand_piv.values), demand_piv.values, np.nan)
                cs = ax_d.contour(X, Y, d_masked, levels=[RELIABILITY_TARGET],
                                  colors='black', linewidths=2.5, linestyles='--')
                ax_d.clabel(cs, fmt='95%%', fontsize=12, inline=True, inline_spacing=8)
            except Exception:
                pass

            ax_d.set_title('Demand Satisfied  (%)', fontsize=15, fontweight='bold', pad=12)

            # ── STAR on optimal ───────────────────────────────────────────
            if best['size_MW'] in ALL_SIZES and best['storage_kg'] in ALL_STORES:
                bj = ALL_SIZES.index(int(best['size_MW']))
                bi = ALL_STORES.index(int(best['storage_kg']))
                for ax in [ax_l, ax_d]:
                    ax.plot(bj, bi, '*', ms=32, color='#D50000',
                            markeredgecolor='white', markeredgewidth=2.5, zorder=10)

            # ── Axes (same labels on every heatmap) ───────────────────────
            for ax in [ax_l, ax_d]:
                ax.set_xticks(range(ns))
                ax.set_xticklabels([str(s) for s in ALL_SIZES], fontsize=12)
                ax.set_yticks(range(nt))
                ax.set_yticklabels([f'{s//1000}k' for s in ALL_STORES], fontsize=12)
                ax.set_xlabel('Electrolyser Size (MW)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Storage Capacity (kg)', fontsize=14, fontweight='bold')
                for edge in np.arange(-0.5, ns, 1):
                    ax.axvline(edge, color='#BDBDBD', lw=0.5)
                for edge in np.arange(-0.5, nt, 1):
                    ax.axhline(edge, color='#BDBDBD', lw=0.5)

            # ── Suptitle ──────────────────────────────────────────────────
            feas_tag = 'FEASIBLE' if is_feasible else 'INFEASIBLE (best effort shown)'
            n_pruned = int(np.isnan(lcoh_piv.values).sum())
            prune_note = f'  ({n_pruned} configs pruned)' if n_pruned > 0 else ''
            fig.suptitle(
                f'{tech} Electrolyser  |  RE = {re_frac*100:.0f}%  |  '
                f'132 MW Wind+PV  |  15 Years{prune_note}\n'
                f'Optimal: {best["size_MW"]:.0f} MW,  '
                f'{best["storage_kg"]:,.0f} kg storage,  '
                f'LCOH = {best["LCOH"]:.2f} EUR/kg,  '
                f'Demand = {best["demand_met_pct"]:.1f}%  '
                f'[{feas_tag}]',
                fontsize=14, fontweight='bold', y=1.02)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            save(fig, f'{tech.lower()}_heatmap_RE{re_frac*100:.0f}')
            plt.close(fig)

            records.append(dict(
                Technology=tech, RE_pct=re_frac*100,
                Size_MW=best['size_MW'], Storage_kg=best['storage_kg'],
                LCOH=best['LCOH'], Demand_pct=best['demand_met_pct'],
                SEC=best['SEC'], Feasible=is_feasible))

    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / 'optimization_summary.csv', index=False)
    print(f"  -> optimization_summary.csv")
    print(f"\n  {'Tech':<10} {'RE':>4} {'MW':>5} {'Store':>7} {'LCOH':>7} {'Dem%':>6} {'Feas':>5}")
    print("  " + "-"*48)
    for _, r in summary.iterrows():
        f = 'YES' if r['Feasible'] else ' no'
        print(f"  {r['Technology']:<10} {r['RE_pct']:3.0f}% {r['Size_MW']:4.0f} "
              f"{r['Storage_kg']:6.0f} {r['LCOH']:6.2f} {r['Demand_pct']:5.1f}% {f:>4}")


# =============================================================================
# PLOT 4: SUMMARY TABLE
# =============================================================================
def plot_summary_table():
    print("\n[5/5] Feasible Summary Table...")
    summary = pd.read_csv(OUTPUT / 'optimization_summary.csv')
    feasible = summary[summary['Feasible'] == True]
    if len(feasible) == 0:
        print("  No feasible configurations -- skipping.")
        return

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis('off')
    headers = ['Technology', 'RE (%)', 'Size (MW)', 'Storage (kg)',
               'LCOH (EUR/kg)', 'Demand (%)', 'SEC (kWh/kg)']
    rows, row_colors = [], []
    for _, r in feasible.iterrows():
        rows.append([
            r['Technology'], f"{r['RE_pct']:.0f}", f"{r['Size_MW']:.0f}",
            f"{r['Storage_kg']:,.0f}", f"{r['LCOH']:.2f}",
            f"{r['Demand_pct']:.1f}", f"{r['SEC']:.1f}"])
        bg = C_PEM_LIGHT if r['Technology'] == 'PEM' else C_ALK_LIGHT
        row_colors.append([bg]*7)

    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellColours=row_colors, colColours=['#EEEEEE']*7,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.0, 2.2)
    for j in range(len(headers)):
        tbl[0, j].set_text_props(fontweight='bold', fontsize=12)
    ax.set_title('Feasible Optimal Configurations  (Demand >= 95%)\n'
                 'Selection: minimum LCOH among all configs meeting the reliability target',
                 fontsize=14, fontweight='bold', pad=25)
    save(fig, 'fig_feasible_summary_table')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  CORRECTED THESIS PLOTS  v3 (all audit fixes)")
    print("=" * 65)

    plot_re_fraction_lcoh()
    plot_pem_performance()
    plot_alkaline_performance()
    plot_heatmaps()
    plot_summary_table()

    print(f"\n{'='*65}")
    print(f"  DONE -- all output in: {OUTPUT}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
