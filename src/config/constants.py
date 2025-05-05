import numpy as np

BIN_EDGES = np.array([18.5, 18.6, 18.7, 18.8, 18.9, 19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 20., 20.2])
# BIN_EDGES = np.array([18.5, 18.6, 18.7, 18.8, 18.9, 19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 20.2])
# BIN_EDGES = np.array([18.5, 18.6, 18.7, 18.8, 18.9, 19.0, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 20.5])

BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTHS = np.diff(BIN_EDGES)

A0 = 7853950218.047947
OMEGA0 = 4.13
A_GEOM = A0 * OMEGA0
ONTIME_SECONDS = 3714.2 * 3600