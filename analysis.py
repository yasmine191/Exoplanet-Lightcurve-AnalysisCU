# ============================================================
# ---------------lightcurve_analysis.py------------------------------------
#
# BLS period search, transit duration (theoretical = adopted,
# BLS-refined = comparison only), scaled semi-major axis (a/R*),
# and packaging the adopted solution used downstream by
# pre_model.py / model.py.
#
# astropy BLS docs: https://docs.astropy.org/en/stable/timeseries/bls.html
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G

from astropy.timeseries import BoxLeastSquares

from query import fetch_archive_params


# ------------------------------------------------------------
#  BLS period search
# ------------------------------------------------------------

def build_bls(lc):
    """
    Builds a BoxLeastSquares object using the light curve's real per-point
    flux uncertainty (dy), instead of leaving it unweighted.
    """
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    # guard against any residual NaNs/zeros in the error array
    flux_err = np.where(np.isfinite(flux_err) & (flux_err > 0),
                         flux_err, np.nanmedian(flux_err))
    return BoxLeastSquares(time, flux, dy=flux_err)


def estimate_period(lc, min_period=0.5, max_period=None, max_duration_hours=6):
    """
    Estimates orbital period, t0, and a coarse duration using BLS.
    Two-stage search: a coarse grid over the full period range, then a
    fine grid (+/- 1%) around the coarse peak. The duration returned here
    is only a byproduct of the period search — NOT the adopted duration
    (see expected_duration / get_robust_duration below for that).
    """
    time_span = lc.time[-1].value - lc.time[0].value
    if max_period is None:
        max_period = min(20, time_span / 2)

    bls = build_bls(lc)

    durations = np.linspace(0.5 / 24.0, max_duration_hours / 24.0, 20)
    trial_periods = np.linspace(min_period, max_period, 20000)

    # coarse search
    results = bls.power(trial_periods, durations)
    best_period = results.period[np.argmax(results.power)]

    # fine grid search (+/- 1% around coarse peak)
    fine_grid = np.linspace(best_period * 0.99, best_period * 1.01, 20000)
    results_fine = bls.power(fine_grid, durations)
    best_idx = np.argmax(results_fine.power)

    best_period = results_fine.period[best_idx]
    best_t0 = results_fine.transit_time[best_idx]
    best_dur = results_fine.duration[best_idx]  # in days

    print(f"\n--- BLS Period Search ---")
    print(f"Estimated period ~= {best_period:.6f} days")
    print(f"Estimated t0 ~= {best_t0:.6f} BTJD")
    print(f"Estimated duration ~= {24 * best_dur:.2f} hours (coarse, from period search only)")

    return best_period, best_t0, best_dur, bls, results_fine


# ------------------------------------------------------------
# Scaled semi-major axis 
# ------------------------------------------------------------

def scaled_semi_major(period_days, planet_name, stellar_mass=None, stellar_radius=None):
    """
    Computes the unitless scaled semi-major axis (a / R_s) from host star
    properties, via Kepler's third law.
    If stellar_mass / stellar_radius are provided, uses those directly — no
    second archive query. Otherwise falls back to the archive via
    fetch_archive_params() (in query.py) — the one shared query function used
    everywhere else in the pipeline, so this doesn't run its own separate
    TAP call.
    """
    if stellar_mass is None or stellar_radius is None:
        params = fetch_archive_params(planet_name, ["st_mass", "st_rad"])
        stellar_mass = params["st_mass"]
        stellar_radius = params["st_rad"]
        if stellar_mass is None or stellar_radius is None:
            raise RuntimeError(
                f"Could not retrieve stellar mass/radius for '{planet_name}' "
                "from the NASA Exoplanet Archive (pscomppars/ps)."
            )

    Ms = (stellar_mass * u.M_sun).to(u.kg)
    Rs = (stellar_radius * u.R_sun).to(u.m)
    P = (period_days * u.day).to(u.s)

    a_meters = ((G * Ms * P**2) / (4 * np.pi**2)) ** (1 / 3)
    a_Rs = (a_meters / Rs).decompose().value

    print(f"\n--- {planet_name} ---")
    print(f"Scaled semi-major axis (a/R_s) = {a_Rs:.4f}, using Kepler's law")
    return a_Rs


# ------------------------------------------------------------
# Transit duration: theoretical (ADOPTED) vs. BLS-refined (comparison only)
# ------------------------------------------------------------

def expected_duration(a_Rs, period_days):
    """
    Computes the theoretical central transit duration (b=0), in hours,
    for a nearly circular orbit.

    This is the duration that gets ADOPTED as the solution duration going
    forward — it only depends on a/R* and P, so it doesn't degrade on a
    noisy/gappy light curve the way a BLS duration search can. The
    BLS-refined duration (refine_duration, below) is computed purely so
    the user can compare the two, never to override this value.
    """
    p_hours = period_days * 24.0
    duration_hr = p_hours / (np.pi * a_Rs)
    print(f"Theoretical central duration (ADOPTED) = {duration_hr:.2f} hours")
    return duration_hr


def refine_duration(bls, best_period, duration_min_hr, duration_max_hr=None, n_grid=200):
    """
    Searches a duration grid (centered on duration_min_hr, normally the
    theoretical duration) at fixed best_period, and returns the
    BLS-preferred duration/t0/depth.

    COMPARISON ONLY — this is shown to the user alongside the adopted
    theoretical duration so they can sanity-check the two against each
    other. It is never used to override the adopted duration, since a
    BLS duration search is unreliable on a noisy/low-S/N light curve.

    Returns duration in DAYS, t0, and depth (fraction) — matching the
    original BLS results units, so callers multiply by 24 for hours.
    """
    if duration_max_hr is None:
        duration_max_hr = duration_min_hr + 4.8

    duration_grid_days = np.linspace(duration_min_hr / 24.0, duration_max_hr / 24.0, n_grid)
    periods_fixed = np.full_like(duration_grid_days, best_period)

    results = bls.power(periods_fixed, duration_grid_days)
    best_idx = np.argmax(results.power)

    refined_dur = results.duration[best_idx]        # days
    refined_t0 = results.transit_time[best_idx]
    refined_depth = results.depth[best_idx]          # fraction

    print(f"BLS-refined duration (comparison only) = {refined_dur * 24.0:.2f} hours")
    print(f"BLS-refined depth (comparison only)    = {refined_depth * 100.0:.4f}%")
    return refined_dur, refined_t0, refined_depth


def get_robust_duration(lc, bls, best_period, best_t0, best_dur,
                         planet_name, stellar_mass=None, stellar_radius=None,
                         snr_threshold=8.0):
    """
    Adopts the THEORETICAL duration (expected_duration) as the solution
    duration, and shows the BLS-refined duration purely as a comparison
    for the user — it is never adopted.

    The BLS transit depth S/N (depth / depth_err, from bls.compute_stats —
    the same depth-based S/N family as the snr_depth_fit computed later,
    post-fit, in step 10.b of model.py) is used here as a data-quality
    flag: below snr_threshold the light curve is too noisy for a BLS
    duration search to mean much, so the comparison number is skipped
    rather than shown and silently trusted.
    """
    stats = bls.compute_stats(best_period, best_dur, best_t0)
    depth_val, depth_err = stats["depth"]
    snr = float(depth_val / depth_err)
    print(f"BLS transit depth S/N = {snr:.2f}")

    a_Rs = scaled_semi_major(best_period, planet_name, stellar_mass, stellar_radius)
    expected_dur_hr = expected_duration(a_Rs, best_period)  # <-- adopted

    if snr >= snr_threshold:
        refine_duration(bls, best_period, duration_min_hr=expected_dur_hr)
    else:
        print(f"BLS-refined duration comparison skipped "
              f"(S/N={snr:.1f} < {snr_threshold} — BLS duration search unreliable here).")

    # depth/t0 at the ADOPTED (theoretical) duration — this is what gets used downstream
    fixed = bls.power(np.array([best_period]), np.array([expected_dur_hr / 24.0]))
    final_t0 = fixed.transit_time[0]
    final_depth = fixed.depth[0]
    final_dur_hr = expected_dur_hr
    source = "theoretical (adopted)"

    print(f"Adopted duration = {final_dur_hr:.2f} hr  [{source}]")
    return final_dur_hr, final_t0, final_depth, snr, source, a_Rs


# ------------------------------------------------------------
# Step 3: package the adopted solution
# ------------------------------------------------------------

def save_transit_solution(planet_name, best_period, final_t0, final_dur_hr,
                           a_Rs, final_depth, snr, source):
    """
    Packages the adopted transit solution (period, mid-transit, duration,
    a/R_s) into one dict for later use — folding, transit modeling, etc.
    """
    solution = {
        "planet_name": planet_name,
        "period_days": best_period,
        "t0": final_t0,
        "duration_hr": final_dur_hr,
        "duration_days": final_dur_hr / 24.0,
        "a_Rs": a_Rs,
        "depth": final_depth,
        "snr": snr,
        "duration_source": source,
    }
    print(f"\n--- Saved solution for {planet_name} ---")
    for k, v in solution.items():
        print(f"{k}: {v}")
    return solution


# ------------------------------------------------------------
# Step 4: transit window + data-quality check on it
# ------------------------------------------------------------

def get_transit_window(lc, solution, window_factor=1.5):
    """
    Folds the light curve on the adopted period/t0 and trims to
    +/- (window_factor/2) * duration around mid-transit, so both the
    in-transit points and an out-of-transit baseline are kept.
    """
    period = solution["period_days"]
    t0 = solution["t0"]
    duration_days = solution["duration_days"]

    folded = lc.fold(period=period, epoch_time=t0)

    half_window = 0.5 * window_factor * duration_days
    mask = np.abs(folded.time.value) <= half_window
    windowed = folded[mask]

    print(f"Transit window: +/-{half_window * 24:.2f} hr around mid-transit, "
          f"{mask.sum()} points kept out of {len(folded)}")
    return windowed


def check_transit_minimum(transit_window, solution, min_points_in_transit=5, core_fraction=0.5):
    """
    Verifies the folded/windowed light curve actually has data points
    covering the transit minimum (not just ingress/egress wings).
    A missing minimum makes the fold unusable for modeling.

    Raises
    ------
    ValueError
        If the transit core has too few points — modeling cannot proceed.
    """
    duration_days = solution["duration_days"]
    half_core = 0.5 * core_fraction * duration_days

    phase = transit_window.time.value
    core_mask = np.abs(phase) <= half_core
    n_core = int(core_mask.sum())

    if n_core < min_points_in_transit:
        raise ValueError(
            f"\n*** CRITICAL: transit minimum is missing for {solution['planet_name']} ***\n"
            f"Only {n_core} point(s) found within +/-{half_core * 24:.2f} hr of mid-transit "
            f"(need at least {min_points_in_transit}).\n"
            f"The actual transit dip isn't covered by data, so modeling cannot proceed.\n"
            f"Please choose a different planet or a different sector/quarter/campaign."
        )

    print(f"Transit minimum check passed: {n_core} points within "
          f"+/-{half_core * 24:.2f} hr of mid-transit.")
    return True


# ------------------------------------------------------------
# Step 5: plots
# ------------------------------------------------------------

def plot_bls_periodogram(results_fine, best_period, planet_name):
    plt.figure(figsize=(8, 4))
    plt.plot(results_fine.period, results_fine.power)
    plt.axvline(best_period, color='red', ls='--', label='chosen peak')
    plt.xscale("log")
    plt.xlabel("Period [days]")
    plt.ylabel("BLS Power")
    plt.title(f"BLS Periodogram — {planet_name}")
    plt.legend()
    plt.show()


def plot_folded_transit(transit_window, solution):
    plt.figure(figsize=(6, 4))
    transit_window.scatter(marker='.', s=10, color='black', alpha=0.6)
    plt.axvline(0.0, color='red', ls='--', label='t0')
    plt.axvline(-solution["duration_days"] / 2, color='blue', ls=':', label='transit edges')
    plt.axvline(solution["duration_days"] / 2, color='blue', ls=':')
    plt.xlabel("Phase (days)")
    plt.ylabel("Normalized Flux")
    plt.title(f"Folded transit — {solution['planet_name']}")
    plt.legend()
    plt.show()