import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
from astropy.timeseries import BoxLeastSquares
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


def build_bls(lc):
    """
    builds a BoxLeastSquares object using the light curve's real per-point
    flux uncertainty (dy), instead of leaving it unweighted.
    """
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    # guard against any residual NaNs/zeros in the error array
    flux_err = np.where(np.isfinite(flux_err) & (flux_err > 0),
                         flux_err, np.nanmedian(flux_err))
    return BoxLeastSquares(time, flux, dy=flux_err)

#======================================================================================================

def estimate_period(lc, min_period=0.5, max_period=None, max_duration_hours=6):
    """
    estimates orbital period, t0, and coarse duration using BLS.
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
    # the duration won't be good anyways
    fine_grid = np.linspace(best_period * 0.99, best_period * 1.01, 20000)
    results_fine = bls.power(fine_grid, durations)
    best_idx = np.argmax(results_fine.power)

    best_period = results_fine.period[best_idx]
    best_t0 = results_fine.transit_time[best_idx]
    best_dur = results_fine.duration[best_idx]  # in days

    print(f"\n--- BLS Period Search ---")
    print(f"Estimated period ≈ {best_period:.6f} days")
    print(f"Estimated t0 ≈ {best_t0:.6f} BTJD")
    print(f"Estimated duration ≈ {24 * best_dur:.2f} hours")

    return best_period, best_t0, best_dur, bls, results_fine

#======================================================================================================

def scaled_semi_major(period_days, planet_name, stellar_mass=None, stellar_radius=None,
                       max_retries=4, initial_delay=3):
    """
    computes unitless scaled semi-major axis (a / R_s) from host star properties.

    if stellar_mass / stellar_radius are provided (e.g. already pulled once by
    get_user_input() during planet selection), uses those directly — no second
    archive query. otherwise falls back to querying the archive itself.
    """
    if stellar_mass is None or stellar_radius is None:
        import time as _time
        res = None
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                res = NasaExoplanetArchive.query_object(planet_name, table="pscomppars")
                break
            except Exception as err:
                print(f"Archive query attempt {attempt}/{max_retries} failed: {err}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"NASA Exoplanet Archive query for '{planet_name}' failed after "
                        f"{max_retries} attempts. Try again in a few minutes."
                    ) from err
                print(f"Retrying in {delay:.0f}s...")
                _time.sleep(delay)
                delay *= 2
        stellar_mass = float(res['st_mass'][0].value)
        stellar_radius = float(res['st_rad'][0].value)

    Ms = (stellar_mass * u.M_sun).to(u.kg)
    Rs = (stellar_radius * u.R_sun).to(u.m)
    P = (period_days * u.day).to(u.s)

    a_meters = (((G * Ms * P**2) / (4 * np.pi**2)) ** (1 / 3))
    a_Rs = (a_meters / Rs).decompose().value

    print(f"\n--- {planet_name} Physics ---")
    print(f"Scaled semi-major axis (a/R_s) = {a_Rs:.4f}")
    return a_Rs

#======================================================================================================

def expected_duration(a_Rs, period_days):
    """
    computes theoretical central transit duration (b=0) in hours for early circular orbits only from some paper.
    """
    p_hours = period_days * 24.0
    duration_hr = p_hours / (np.pi * a_Rs)
    print(f"Expected theoretical central duration = {duration_hr:.2f} hours")
    return duration_hr

#======================================================================================================

def refine_duration(bls, best_period, duration_min_hr=0.5, duration_max_hr=None, n_grid=200):
    """
    refines transit duration at fixed best_period, searching a grid centered
    on duration_min_hr (typically the analytic expected duration) out to
    duration_min_hr + 4.8 hr by default, or an explicit duration_max_hr.

    returns duration in DAYS, t0, and depth (fraction) — matching the
    original BLS results units, so callers multiply by 24 for hours.
    won't do good work also
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

    print(f"\n--- Refinement Results ---")
    print(f"Refined duration = {refined_dur * 24.0:.2f} hours")
    print(f"Refined depth = {refined_depth * 100.0:.4f}%")
    return refined_dur, refined_t0, refined_depth

#======================================================================================================

def get_robust_duration(lc, bls, best_period, best_t0, best_dur,
                         planet_name, stellar_mass=None, stellar_radius=None,
                         snr_threshold=8.0):
    """
    generalized duration estimate for hot Jupiters on near-circular orbits.
    Uses BLS transit S/N (depth / depth_err, from the light curve itself —
    not a catalogue value) as an objective stand-in for "how scattered/noisy
    is this light curve", instead of eyeballing it.
    """
    stats = bls.compute_stats(best_period, best_dur, best_t0)
    depth_val, depth_err = stats["depth"]
    snr = float(depth_val / depth_err)
    print(f"BLS transit depth S/N = {snr:.2f}")

    a_Rs = scaled_semi_major(best_period, planet_name, stellar_mass, stellar_radius)
    expected_dur_hr = expected_duration(a_Rs, best_period)

    if snr >= snr_threshold:
        refined_dur, refined_t0, refined_depth = refine_duration(
            bls, best_period, duration_min_hr=expected_dur_hr
        )
        final_dur_hr = refined_dur * 24.0
        source = "BLS-refined"
    else:
        final_dur_hr = expected_dur_hr
        fixed = bls.power(np.array([best_period]),
                           np.array([expected_dur_hr / 24.0]))
        refined_t0 = fixed.transit_time[0]
        refined_depth = fixed.depth[0]
        source = f"analytic (S/N={snr:.1f} < {snr_threshold}, BLS duration search unreliable)"

    print(f"Adopted duration = {final_dur_hr:.2f} hr  [{source}]")
    return final_dur_hr, refined_t0, refined_depth, snr, source, a_Rs

#======================================================================================================

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

#======================================================================================================

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

    print(f"Transit window: ±{half_window * 24:.2f} hr around mid-transit, "
          f"{mask.sum()} points kept out of {len(folded)}")
    return windowed

#======================================================================================================

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

#======================================================================================================

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