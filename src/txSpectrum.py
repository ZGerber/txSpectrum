import argparse
import logging
import lmfit

import numpy as np
import awkward as ak
from matplotlib import pyplot as plt

import FC

# Constants
A0 = 7853950218.047947
OMEGA0 = 4.13
A_GEOM = A0 * OMEGA0
ONTIME_SECONDS = 3714.2 * 3600

BIN_EDGES = np.array([18.5, 18.6, 18.7, 18.8, 18.9, 19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 20.5])
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTHS = np.diff(BIN_EDGES)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(dt_file, mc_file, ta_file):
    logger.info("Loading data...")
    dt = ak.from_parquet(dt_file)
    mc = ak.from_parquet(mc_file)
    ta = ak.from_parquet(ta_file)
    return dt, mc, ta


def get_event_counts_per_bin(data, bin_edges):
    counts = np.zeros(len(bin_edges) - 1)
    for E in data.LogEnergy:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= E < bin_edges[i + 1]:
                counts[i] += 1
                break
    logger.info(f"Counts per bin: {counts}")
    return counts


def apply_cuts(data, use_border, use_geom, which_cuts="zane"):
    cut_sets = {
        "matt": [5, 50, 50, 0, 55, 145],
        "zane": [3.5, 25, 20, 400, 55, 145],
    }
    logger.info(f"Applying cuts: {cut_sets[which_cuts]}, use_border={use_border}, use_geom={use_geom}")
    if which_cuts not in cut_sets:
        raise ValueError(f"Unknown cuts: {which_cuts}")
    c = cut_sets[which_cuts]
    cuts = ((data.CoreProximity <= c[0]) &
            (data.ProfileFitQuality <= c[1]) &
            (data.GeometryFitQuality <= c[2]))
    if use_border:
        cuts = cuts & (data.BorderDistance >= c[3])
    if use_geom:
        cuts = cuts & (data.Zenith <= c[4]) & (data.Psi <= c[5])
    return data[cuts]


def feldman_cousins_flux_errors(counts, exposure, t, conf=0.6827, use_correction=True):
    lower, upper = [], []
    for n in counts:
        ci = FC.FC_poisson(int(n), b=0.0, t=t, conf=conf, useCorrection=use_correction)
        lower.append(ci[0])
        upper.append(ci[1])
    lower = np.array(lower) / exposure
    upper = np.array(upper) / exposure
    mid = counts / exposure
    return mid - lower, upper - mid


def reweight_true_energy(true_energy):
    """Reweight events from E^-2 to match Telescope Array three-breakpoint spectrum."""
    E0, E1, E2, E3 = 10 ** 18.0, 10 ** 18.7, 10 ** 19.15, 10 ** 19.83
    g1, g2, g3, g4 = (-3.28 + 2), (-2.62 + 2), (-2.83 + 2), (-4.61 + 2)

    energy = ak.to_numpy(true_energy)
    w = np.ones_like(energy)

    w[energy <= E1] = (energy[energy <= E1] / E0) ** g1
    mask2 = (energy > E1) & (energy <= E2)
    w[mask2] = (E1 / E0) ** g1 * (energy[mask2] / E1) ** g2
    mask3 = (energy > E2) & (energy <= E3)
    w[mask3] = (E1 / E0) ** g1 * (E2 / E1) ** g2 * (energy[mask3] / E2) ** g3
    mask4 = (energy > E3)
    w[mask4] = (E1 / E0) ** g1 * (E2 / E1) ** g2 * (E3 / E2) ** g3 * (energy[mask4] / E3) ** g4

    return ak.from_numpy(w)


def prepare_data(dt, mc, cuts, use_border, use_geom):
    logger.info("Applying cuts...")
    dt_cut = apply_cuts(dt, use_border=use_border, use_geom=use_geom, which_cuts=cuts)
    mc_cut = apply_cuts(mc, use_border=use_border, use_geom=use_geom, which_cuts=cuts)

    logger.info("Counting events...")
    data_counts = get_event_counts_per_bin(dt_cut, BIN_EDGES)
    mc_counts = get_event_counts_per_bin(mc_cut, BIN_EDGES)

    logger.info("Reweighting MC energies...")
    weights = reweight_true_energy(mc_cut.Energy_true)

    return dt_cut, mc_cut, data_counts, mc_counts, weights


def compute_flux(data_counts, mc, weights):
    mc_thrown_counts, _ = np.histogram(mc.LogEnergy_true, bins=BIN_EDGES, weights=weights)
    mc_reco_counts = get_event_counts_per_bin(mc, BIN_EDGES)

    efficiency = mc_reco_counts / mc_thrown_counts
    aperture_shape = efficiency * A_GEOM
    dE = np.log(10) * (10 ** BIN_CENTERS) * BIN_WIDTHS
    exposure_shape = aperture_shape * ONTIME_SECONDS * dE
    flux_shape = data_counts / exposure_shape

    flux_err_lower, flux_err_upper = feldman_cousins_flux_errors(data_counts, exposure_shape, ONTIME_SECONDS,
                                                                 use_correction=False)

    return flux_shape, flux_err_lower, flux_err_upper, exposure_shape


def normalize_flux(flux_shape, flux_err_lower, flux_err_upper):
    logger.info("Normalizing to TA 19.25 flux point...")

    ta_flux_1925 = 4.112E-34
    dt_flux_1925 = flux_shape[BIN_CENTERS == 19.25][0]

    norm_factor = ta_flux_1925 / dt_flux_1925
    flux_norm = flux_shape * norm_factor
    flux_err_lower_norm = flux_err_lower * norm_factor
    flux_err_upper_norm = flux_err_upper * norm_factor

    return flux_norm, flux_err_lower_norm, flux_err_upper_norm, norm_factor


def plot_flux(bin_centers, bin_widths, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta):
    ta_E = 10 ** ta.logE_bin_centers
    plt.errorbar(bin_centers, flux_norm, yerr=[flux_err_lower_norm, flux_err_upper_norm],
                 xerr=bin_widths / 2, label='TAx4 Hybrid 4 Years', color='black', marker='s', linestyle='None')
    plt.errorbar(ta.logE_bin_centers, ta.flux, yerr=[ta.flux - ta.flux_lower_sigma, ta.flux_upper_sigma - ta.flux],
                 label='TA 16 Years', color='red', marker='o', linestyle='None')
    plt.yscale('log')
    plt.ylabel("$J(E)$ $[eV^{-1} m^{-2} sr^{-1} s^{-1}]$", fontsize=14)
    plt.xlabel("$\\log_{10} E/eV $", fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.tick_params(axis='x', which='both', direction='in', top=True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_e3_flux(bin_centers, bin_widths, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta, fit_result=None, fit_range=None):
    e3 = (10 ** bin_centers) ** 3
    e3j = e3 * flux_norm
    e3j_err_lower = e3 * flux_err_lower_norm
    e3j_err_upper = e3 * flux_err_upper_norm
    ta_E = 10 ** ta.logE_bin_centers
    ta_e3 = ta_E ** 3

    plt.errorbar(bin_centers, e3j, yerr=[e3j_err_lower, e3j_err_upper],
                 xerr=bin_widths / 2, label='TAx4 Hybrid 4 Years', color='black', marker='s', linestyle='None')
    plt.errorbar(ta.logE_bin_centers, ta.flux * ta_e3,
                 yerr=[ta_e3 * (ta.flux - ta.flux_lower_sigma), ta_e3 * (ta.flux_upper_sigma - ta.flux)],
                 label='TA 16 Years', color='red', marker='o', linestyle='None')

    if fit_result is not None:
        if fit_range is None:
            xfit = np.linspace(min(bin_centers), max(bin_centers), 500)
        else:
            xfit = np.linspace(fit_range[0], fit_range[1], 500)

        yfit = fit_result.eval(x=xfit)
        ymin = np.min(e3j[e3j > 0]) / 10  # slightly below lowest data point
        yfit = np.clip(yfit, a_min=ymin, a_max=None)
        plt.plot(xfit, yfit, linestyle=':', color='blue', label='BPL Fit')

    plt.yscale('log')
    plt.ylabel("$E^3 \\times J(E)$ $[eV^{2} m^{-2} sr^{-1} s^{-1}]$", fontsize=14)
    plt.xlabel("$\\log_{10} E/eV $", fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.tick_params(axis='x', which='both', direction='in', top=True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_spectrum(bin_centers, bin_widths, data_counts, exposure_shape, norm_factor):
    logger.info("Saving spectrum to my_spectrum.txt...")
    exposure = exposure_shape * norm_factor
    rows = np.column_stack(
        [bin_centers, np.full_like(bin_centers, bin_widths), data_counts, np.full_like(bin_centers, exposure)])
    np.savetxt("my_spectrum.txt", rows,
               fmt=["%7.2f", "%10.2f", "%15.5e", "%15.5e"],
               header="log10en log10en_bsize       nevents        exposure",
               comments="")


def compare_border_effect(dt, mc, ta, cuts, use_geom):
    logger.info("Generating comparison: border cut vs no border cut")

    # --- With border cuts ---
    dt_cut, mc_cut, data_counts_cut, mc_counts_cut, weights_cut = prepare_data(dt, mc, cuts, use_border=True, use_geom=use_geom)
    flux_shape_cut, flux_err_lower_cut, flux_err_upper_cut, exposure_shape_cut = compute_flux(data_counts_cut, mc_cut, weights_cut)
    flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut, norm_factor_cut = normalize_flux(flux_shape_cut, flux_err_lower_cut, flux_err_upper_cut)

    # --- Without border cuts ---
    dt_noborder, mc_noborder, data_counts_noborder, mc_counts_noborder, weights_noborder = prepare_data(dt, mc, cuts, use_border=False, use_geom=use_geom)
    flux_shape_noborder, flux_err_lower_noborder, flux_err_upper_noborder, exposure_shape_noborder = compute_flux(data_counts_noborder, mc_noborder, weights_noborder)
    flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder, norm_factor_noborder = normalize_flux(flux_shape_noborder, flux_err_lower_noborder, flux_err_upper_noborder)

    # --- Plot ---
    plot_border_comparison(BIN_CENTERS, BIN_WIDTHS,
                           flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut,
                           flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder,
                           ta)

    plot_border_comparison_e3(BIN_CENTERS, BIN_WIDTHS,
                              flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut,
                              flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder,
                              ta)


def plot_border_comparison(bin_centers, bin_widths,
                           flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut,
                           flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder,
                           ta):
    ta_E = 10 ** ta.logE_bin_centers

    plt.errorbar(bin_centers, flux_norm_cut,
                 yerr=[flux_err_lower_norm_cut, flux_err_upper_norm_cut],
                 xerr=bin_widths / 2, label='TAx4 Hybrid (Border Cut)', color='black', marker='s', linestyle='None')

    plt.errorbar(bin_centers, flux_norm_noborder,
                 yerr=[flux_err_lower_norm_noborder, flux_err_upper_norm_noborder],
                 xerr=bin_widths / 2, label='TAx4 Hybrid (No Border Cut)', color='blue', marker='^', linestyle='None')

    plt.errorbar(ta.logE_bin_centers, ta.flux,
                 yerr=[ta.flux - ta.flux_lower_sigma, ta.flux_upper_sigma - ta.flux],
                 label='TA 16 Years', color='red', marker='o', linestyle='None')

    plt.yscale('log')
    plt.ylabel("$J(E)$ $[eV^{-1} m^{-2} sr^{-1} s^{-1}]$", fontsize=14)
    plt.xlabel("$\\log_{10} E/eV $", fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.tick_params(axis='x', which='both', direction='in', top=True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_border_comparison_e3(bin_centers, bin_widths,
                               flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut,
                               flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder,
                               ta):
    e3 = (10 ** bin_centers) ** 3
    e3j_cut = e3 * flux_norm_cut
    e3j_err_lower_cut = e3 * flux_err_lower_norm_cut
    e3j_err_upper_cut = e3 * flux_err_upper_norm_cut

    e3j_noborder = e3 * flux_norm_noborder
    e3j_err_lower_noborder = e3 * flux_err_lower_norm_noborder
    e3j_err_upper_noborder = e3 * flux_err_upper_norm_noborder

    ta_E = 10 ** ta.logE_bin_centers
    ta_e3 = ta_E ** 3

    plt.errorbar(bin_centers, e3j_cut,
                 yerr=[e3j_err_lower_cut, e3j_err_upper_cut],
                 xerr=bin_widths / 2, label='TAx4 Hybrid (Border Cut)', color='black', marker='s', linestyle='None')

    plt.errorbar(bin_centers, e3j_noborder,
                 yerr=[e3j_err_lower_noborder, e3j_err_upper_noborder],
                 xerr=bin_widths / 2, label='TAx4 Hybrid (No Border Cut)', color='blue', marker='^', linestyle='None')

    plt.errorbar(ta.logE_bin_centers, ta.flux * ta_e3,
                 yerr=[ta_e3 * (ta.flux - ta.flux_lower_sigma), ta_e3 * (ta.flux_upper_sigma - ta.flux)],
                 label='TA 16 Years', color='red', marker='o', linestyle='None')

    plt.yscale('log')
    plt.ylabel("$E^3 \\times J(E)$ $[eV^{2} m^{-2} sr^{-1} s^{-1}]$", fontsize=14)
    plt.xlabel("$\\log_{10} E/eV $", fontsize=14)
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--", alpha=0.5, linewidth=0.5)
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', direction='in', top=True)
    plt.legend()
    plt.show()


def broken_powerlaw_1break(x, A, m1, m2, logE_break):
    out = np.zeros_like(x)
    mask1 = (x < logE_break)
    mask2 = (x >= logE_break)

    out[mask1] = A * 10 ** (m1 * (x[mask1] - logE_break))
    A2 = A
    out[mask2] = A2 * 10 ** (m2 * (x[mask2] - logE_break))

    return out


def broken_powerlaw_2break(x, A1, m1, m2, m3, logE1, logE2):
    out = np.zeros_like(x)
    mask1 = (x < logE1)
    mask2 = (x >= logE1) & (x < logE2)
    mask3 = (x >= logE2)

    out[mask1] = A1 * 10 ** (m1 * (x[mask1] - logE1))
    A2 = A1  # ensure continuity at logE1
    out[mask2] = A2 * 10 ** (m2 * (x[mask2] - logE1))  # shift relative to logE1
    A3 = A2 * 10 ** (m2 * (logE2 - logE1))  # propagate to second break
    out[mask3] = A3 * 10 ** (m3 * (x[mask3] - logE2))  # shift relative to logE2

    return out


def broken_powerlaw_3break(x, A1, m1, m2, m3, m4, logE1, logE2, logE3):
    out = np.zeros_like(x)
    mask1 = (x < logE1)
    mask2 = (x >= logE1) & (x < logE2)
    mask3 = (x >= logE2) & (x < logE3)
    mask4 = (x >= logE3)

    out[mask1] = A1 * 10 ** (m1 * (x[mask1] - logE1))
    A2 = A1 * 10 ** (m1 * (logE2 - logE1))
    out[mask2] = A2 * 10 ** (m2 * (x[mask2] - logE2))
    A3 = A2 * 10 ** (m2 * (logE3 - logE2))
    out[mask3] = A3 * 10 ** (m3 * (x[mask3] - logE3))
    A4 = A3 * 10 ** (m3 * (x[mask3][-1] - logE3))  # careful with continuity
    out[mask4] = A4 * 10 ** (m4 * (x[mask4] - logE3))

    return out


def fit_broken_powerlaw(bin_centers, e3j, e3j_err, n_breaks=1):
    logger.info(f"Fitting TAx4 E³J(E) spectrum with {n_breaks} break(s)...")

    if n_breaks == 1:
        model = lmfit.Model(broken_powerlaw_1break)
        params = model.make_params(
            A=1e24,
            m1=-2.7,
            m2=-4.6,
            logE_break=19.83)
        params['A'].set(min=1e20, max=1e30)
        params['m1'].set(min=-10, max=+10)
        params['m2'].set(min=-10, max=+10)
        params['logE_break'].set(min=19.5, max=20.)

    elif n_breaks == 2:
        model = lmfit.Model(broken_powerlaw_2break)
        params = model.make_params(
            A1=1e24,
            m1=-2.6,
            m2=-2.83,
            m3=-4.6,
            logE1=19.15,
            logE2=19.83)
        params['A1'].set(min=1e20, max=1e30)
        params['m1'].set(min=-10, max=+10)
        params['m2'].set(min=-10, max=+10)
        params['m3'].set(min=-10, max=+10)
        params['logE1'].set(min=18.8, max=19.5)
        params['logE2'].set(min=19.5, max=20.0)

    else:
        raise ValueError("Only n_breaks = 1 or 2 are supported!")

    sigma = np.asarray(e3j_err)
    result = model.fit(e3j, params, x=bin_centers, weights=1/sigma)
    print(result.fit_report())
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute cosmic-ray spectrum from TAx4 hybrid data.")
    parser.add_argument("dt_file", help="Path to data parquet file")
    parser.add_argument("mc_file", help="Path to MC parquet file")
    parser.add_argument("ta_file", help="Path to TA reference spectrum parquet file")
    parser.add_argument("--cuts", choices=["zane", "matt"], default="zane", help="Which cuts to apply (default: zane)")
    parser.add_argument("--no-border", dest="use_border", action="store_false",
                        help="Disable border distance cuts (default: cuts applied)")
    parser.add_argument("--n-breaks", type=int, choices=[1, 2], default=1,
                        help="Number of breakpoints in broken power-law fit (default: 1)")
    parser.add_argument("--use-geom", action="store_true", help="Apply geometry cuts (zenith, psi)")
    parser.add_argument("--save-spectrum", action="store_true", help="Save output spectrum to my_spectrum.txt")
    parser.set_defaults(use_border=True)
    args = parser.parse_args()

    dt, mc, ta = load_data(args.dt_file, args.mc_file, args.ta_file)
    dt_cut, mc_cut, data_counts, mc_counts, weights = prepare_data(dt, mc, args.cuts, args.use_border, args.use_geom)
    flux_shape, flux_err_lower, flux_err_upper, exposure_shape = compute_flux(data_counts, mc_cut, weights)
    flux_norm, flux_err_lower_norm, flux_err_upper_norm, norm_factor = normalize_flux(flux_shape, flux_err_lower, flux_err_upper)

    e3 = (10 ** BIN_CENTERS) ** 3
    e3j = e3 * flux_norm
    e3j_err_lower = e3 * flux_err_lower_norm
    e3j_err_upper = e3 * flux_err_upper_norm
    e3j_err = (e3j_err_lower + e3j_err_upper) / 2

    fit_mask = BIN_CENTERS >= 18.7
    fit_bin_centers = BIN_CENTERS[fit_mask]
    fit_e3j = e3j[fit_mask]
    fit_e3j_err = e3j_err[fit_mask]
    fit_result = fit_broken_powerlaw(fit_bin_centers, fit_e3j, fit_e3j_err, n_breaks=args.n_breaks)

    plot_flux(BIN_CENTERS, BIN_WIDTHS, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta)
    # plot_e3_flux(BIN_CENTERS, BIN_WIDTHS, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta,
    #              fit_result=fit_result)
    plot_e3_flux(BIN_CENTERS, BIN_WIDTHS, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta,
                 fit_result=fit_result, fit_range=(min(fit_bin_centers), max(fit_bin_centers)))

    # compare_border_effect(dt, mc, ta, args.cuts, args.use_geom)

    if args.save_spectrum:
        save_spectrum(BIN_CENTERS, BIN_WIDTHS, data_counts, exposure_shape, norm_factor)


if __name__ == "__main__":
    main()