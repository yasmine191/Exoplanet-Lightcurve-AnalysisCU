# Exoplanet Light Curve Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A simple, end-to-end Python pipeline built by physics and astronomy students to process, model, and analyze Hot Jupiter transit light curves using observation data from space missions.

## What This Project Does

This tool automates the process of fetching raw light curves, searching for transits, fitting quadratic limb-darkening models, and estimating physical planet parameters:

* **Archive Querying:** Interactive search against the NASA Exoplanet Archive TAP service for short-period, near-circular Hot Jupiters ($e \le 0.05$).
* **Data Processing:** Automated download via `lightkurve` (supporting TESS, Kepler, and K2), applying bitmask quality filters, outlier removal, and Savitzky-Golay trend flattening.
* **Period Search (BLS):** Box Least Squares analysis to estimate the orbital period, mid-transit time ($t_0$), depth, and adopted theoretical transit duration.
* **Bayesian Transit Modeling:** Fits quadratic limb-darkening transit models using `PyTransit` and MCMC ensemble sampling (`emcee`) to derive 68% credible intervals for $R_p/R_*$, $a/R_*$, inclination, and impact parameter $b$.
* **Diagnostics & Visualization:** Generates BLS periodograms, phase-folded transits, best-fit models overlaid with residuals, and prayer-bead red-noise quality checks.
* **Archive Comparison:** Compares custom fitted parameters directly against published NASA Exoplanet Archive values.

## Dependencies

Ensure you are using Python version 3.10 or higher.

Ensure you have the following packages installed:

```bash
pip install numpy scipy pandas matplotlib astropy lightkurve pytransit emcee requests
```
## How to run

Run the main execution script directly from your terminal:

```bash
python main.py
```
