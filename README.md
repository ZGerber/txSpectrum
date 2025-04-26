# TAx4 Hybrid Spectrum Fitting

This repository contains a script to compute and fit the cosmic-ray energy spectrum using TAx4 hybrid data.

## Usage

```bash
python txSpectrum.py data.parquet mc.parquet ta_reference.parquet [options]
```

### Options
- `--cuts {zane,matt}`: Choose quality cuts (default: zane)
- `--no-border`: Disable border distance cuts
- `--use-geom`: Apply geometry cuts (zenith, psi)
- `--n-breaks {1,2}`: Number of breakpoints in the fit (default: 1)
- `--save-spectrum`: Save the output spectrum to `my_spectrum.txt`

## Requirements
- Python 3.8+
- `numpy`
- `awkward`
- `matplotlib`
- `lmfit`
- `FC.py` (Feldman-Cousins module) https://github.com/usnistgov/FCpy

---

Developed for Telescope Array (TAx4) hybrid cosmic-ray analysis.
