import awkward as ak
import logging

from config.constants import BIN_EDGES
from flux import reweight_true_energy, get_event_counts_per_bin, apply_cuts

logger = logging.getLogger(__name__)


def load_data(dt_file, mc_file, ta_file):
    """Load data, MC, and TA reference spectrum from parquet files."""
    logger.info("Loading data...")
    dt = ak.from_parquet(dt_file)
    mc = ak.from_parquet(mc_file)
    ta = ak.from_parquet(ta_file)
    return dt, mc, ta


def prepare_data(dt, mc, cuts, use_border, use_geom):
    """Apply cuts, bin events, and reweight MC."""
    logger.info("Applying cuts...")
    dt_cut = apply_cuts(dt, use_border=use_border, use_geom=use_geom, which_cuts=cuts)
    mc_cut = apply_cuts(mc, use_border=use_border, use_geom=use_geom, which_cuts=cuts)

    logger.info("Counting events...")
    data_counts = get_event_counts_per_bin(dt_cut, BIN_EDGES)
    mc_counts = get_event_counts_per_bin(mc_cut, BIN_EDGES)

    logger.info("Reweighting MC energies...")
    weights = reweight_true_energy(mc_cut.Energy_true)
    weights_full = reweight_true_energy(mc.Energy_true)

    return dt_cut, mc_cut, data_counts, mc_counts, weights, weights_full
