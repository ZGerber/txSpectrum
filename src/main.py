import argparse
import logging
import models

import data
import flux
import plots
import fit
from config.constants import BIN_CENTERS, BIN_WIDTHS

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    parser.add_argument("--compare-border", action="store_true", help="Compare fluxes with/without border cut")
    parser.set_defaults(use_border=True)
    args = parser.parse_args()

    # --- Load ---
    dt, mc, ta = data.load_data(args.dt_file, args.mc_file, args.ta_file)

    # synthetic_logE, bin_edges, bin_centers = models.synthetic_energy_distribution()
    # weights_full = flux.reweight_true_energy(10**synthetic_logE)

    # --- Prepare ---
    dt_cut, mc_cut, data_counts, mc_counts, weights, weights_full = data.prepare_data(
        dt, mc, cuts=args.cuts, use_border=args.use_border, use_geom=args.use_geom)

    # --- Flux Calculation ---
    flux_shape, flux_err_lower, flux_err_upper, exposure_shape = flux.compute_flux(
        data_counts, mc_cut, mc, weights, weights_full, logE_thrown_override=None)
    flux_norm, flux_err_lower_norm, flux_err_upper_norm, norm_factor = flux.normalize_flux(
        flux_shape, flux_err_lower, flux_err_upper)

    # --- Prepare for Fitting ---
    e3 = (10 ** BIN_CENTERS) ** 3
    e3j = e3 * flux_norm
    e3j_err_lower = e3 * flux_err_lower_norm
    e3j_err_upper = e3 * flux_err_upper_norm
    e3j_err = (e3j_err_lower + e3j_err_upper) / 2

    fit_mask = BIN_CENTERS >= 18.7
    fit_bin_centers = BIN_CENTERS[fit_mask]
    fit_e3j = e3j[fit_mask]
    fit_e3j_err = e3j_err[fit_mask]

    fit_result = fit.fit_broken_powerlaw(
        fit_bin_centers, fit_e3j, fit_e3j_err, n_breaks=args.n_breaks)

    # plots.plot_synthetic_distribution(synthetic_logE, bin_edges, bin_centers)

    plots.plot_flux(
        BIN_CENTERS, BIN_WIDTHS, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta)

    plots.plot_e3_flux(
        BIN_CENTERS, BIN_WIDTHS, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta,
        fit_result=fit_result, fit_range=(min(fit_bin_centers), max(fit_bin_centers)))

    # plots.plot_exposure(
    #     BIN_CENTERS, BIN_WIDTHS, exposure_shape, norm_factor)

    if args.save_spectrum:
        flux.save_spectrum(BIN_CENTERS, BIN_WIDTHS, data_counts, exposure_shape, norm_factor)

    if args.compare_border:
        plots.compare_border_effect(dt, mc, ta, cuts=args.cuts, use_geom=args.use_geom)


if __name__ == "__main__":
    main()
