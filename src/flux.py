import awkward as ak
import numpy as np
import FC
import logging
from config.constants import A_GEOM, ONTIME_SECONDS, BIN_EDGES, BIN_CENTERS, BIN_WIDTHS

logger = logging.getLogger(__name__)


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


def compute_flux(data_counts, mc, weights):
    mc_thrown_counts, _ = np.histogram(mc.LogEnergy_true, bins=BIN_EDGES, weights=weights)
    mc_reco_counts = get_event_counts_per_bin(mc, BIN_EDGES)

    efficiency = mc_reco_counts / mc_thrown_counts
    aperture_shape = efficiency * A_GEOM
    dE = np.log(10) * (10 ** BIN_CENTERS) * BIN_WIDTHS
    exposure_shape = aperture_shape * ONTIME_SECONDS * dE
    flux_shape = data_counts / exposure_shape

    flux_err_lower, flux_err_upper = feldman_cousins_flux_errors(data_counts, exposure_shape, ONTIME_SECONDS, use_correction=False)

    return flux_shape, flux_err_lower, flux_err_upper, exposure_shape


def normalize_flux(flux_shape, flux_err_lower, flux_err_upper):
    """Normalize flux to match TA spectrum at 19.25."""
    logger.info("Normalizing to TA 19.25 flux point...")
    ta_flux_1925 = 4.112E-34
    dt_flux_1925 = flux_shape[BIN_CENTERS == 19.25][0]

    norm_factor = ta_flux_1925 / dt_flux_1925
    flux_norm = flux_shape * norm_factor
    flux_err_lower_norm = flux_err_lower * norm_factor
    flux_err_upper_norm = flux_err_upper * norm_factor

    return flux_norm, flux_err_lower_norm, flux_err_upper_norm, norm_factor


def save_spectrum(bin_centers, bin_widths, data_counts, exposure_shape, norm_factor):
    """Save spectrum to my_spectrum.txt."""
    logger.info("Saving spectrum to my_spectrum.txt...")
    exposure = exposure_shape * norm_factor
    rows = np.column_stack([
        bin_centers, np.full_like(bin_centers, bin_widths),
        data_counts, np.full_like(bin_centers, exposure)])
    np.savetxt("my_spectrum.txt", rows,
               fmt=["%7.2f", "%10.2f", "%15.5e", "%15.5e"],
               header="log10en log10en_bsize       nevents        exposure",
               comments="")
