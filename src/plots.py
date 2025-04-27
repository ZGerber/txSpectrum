import matplotlib.pyplot as plt
import numpy as np
import flux
import data
from config.constants import BIN_CENTERS, BIN_WIDTHS

import logging
logger = logging.getLogger(__name__)


def plot_flux(bin_centers, bin_widths, flux_norm, flux_err_lower_norm, flux_err_upper_norm, ta):
    ta_E = 10 ** ta.logE_bin_centers
    plt.errorbar(bin_centers, flux_norm, yerr=[flux_err_lower_norm, flux_err_upper_norm],
                 xerr=bin_widths / 2, label='TAx4 Hybrid 4 Years', color='black', marker='s', linestyle='None')
    plt.errorbar(ta.logE_bin_centers, ta.flux, yerr=[ta.flux - ta.flux_lower_sigma, ta.flux_upper_sigma - ta.flux],
                 label='TA 16 Years', color='red', marker='o', linestyle='None')
    plt.yscale('log')
    plt.ylabel("$J(E)$ $[eV^{-1} m^{-2} sr^{-1} s^{-1}]$")
    plt.xlabel("$\\log_{10} E/eV$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
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
        ymin = np.min(e3j[e3j > 0]) / 10
        yfit = np.clip(yfit, a_min=ymin, a_max=None)
        plt.plot(xfit, yfit, linestyle=':', color='blue', label='BPL Fit')

    plt.yscale('log')
    plt.ylabel("$E^3 \\times J(E)$ $[eV^{2} m^{-2} sr^{-1} s^{-1}]$")
    plt.xlabel("$\\log_{10} E/eV$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    plt.ylabel("$J(E)$ $[eV^{-1} m^{-2} sr^{-1} s^{-1}]$")
    plt.xlabel("$\\log_{10} E/eV$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
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
    plt.ylabel("$E^3 \\times J(E)$ $[eV^{2} m^{-2} sr^{-1} s^{-1}]$")
    plt.xlabel("$\\log_{10} E/eV$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_border_effect(dt, mc, ta, cuts, use_geom):
    logger.info("Generating comparison: border cut vs no border cut")

    # --- With border cuts ---
    dt_cut, mc_cut, data_counts_cut, mc_counts_cut, weights_cut = data.prepare_data(dt, mc, cuts, use_border=True, use_geom=use_geom)
    flux_shape_cut, flux_err_lower_cut, flux_err_upper_cut, exposure_shape_cut = flux.compute_flux(data_counts_cut, mc_cut, weights_cut)
    flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut, norm_factor_cut = flux.normalize_flux(flux_shape_cut, flux_err_lower_cut, flux_err_upper_cut)

    # --- Without border cuts ---
    dt_noborder, mc_noborder, data_counts_noborder, mc_counts_noborder, weights_noborder = data.prepare_data(dt, mc, cuts, use_border=False, use_geom=use_geom)
    flux_shape_noborder, flux_err_lower_noborder, flux_err_upper_noborder, exposure_shape_noborder = flux.compute_flux(data_counts_noborder, mc_noborder, weights_noborder)
    flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder, norm_factor_noborder = flux.normalize_flux(flux_shape_noborder, flux_err_lower_noborder, flux_err_upper_noborder)

    # --- Plot comparisons ---
    plot_border_comparison(BIN_CENTERS, BIN_WIDTHS,
                           flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut,
                           flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder,
                           ta)

    plot_border_comparison_e3(BIN_CENTERS, BIN_WIDTHS,
                              flux_norm_cut, flux_err_lower_norm_cut, flux_err_upper_norm_cut,
                              flux_norm_noborder, flux_err_lower_norm_noborder, flux_err_upper_norm_noborder,
                              ta)
