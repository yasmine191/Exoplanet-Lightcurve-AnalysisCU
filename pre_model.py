# ============================================================
# ---------------pre_model.py-----------------------------------
#
# Calculations that happen AFTER the BLS/duration/a_Rs solution
# (analysis.py) but BEFORE the transit model is actually fit
# (model.py):
#   1. an independent transit-depth check from the folded window,
#      as a sanity cross-check against the BLS depth,
#   2. planet radius from that depth,
#   3. inclination bounds used to seed/bound the model fit.
#
# Stellar mass/radius are NOT re-queried here -- they're fetched
# once via query.fetch_archive_params() (module 1) during planet
# selection and carried through in `solution` / passed in directly.
# Scaled semi-major axis (a/R*) is likewise not recomputed here --
# it's analysis.scaled_semi_major(), already stored as solution['a_Rs'].
#
# References
# ----------
# Seager & Mallen-Ornelas (2003), ApJ 585, 1038.
#   "A Unique Solution of Planet and Star Parameters from an
#   Extrasolar Planet Transit Light Curve" -- depth ~= (Rp/R*)^2
#   (their Eq. 3), and the transit/impact-parameter geometry used
#   in calculate_inclination_limits below (their Eq. 7-8).
# Winn (2010), "Transits and Occultations", arXiv:1001.2010.
#   Standard reference for transit depth, duration, and impact
#   parameter b = (a/R*) cos(i) for a circular orbit (Sec. 3).
# ============================================================

import numpy as np


# ------------------------------------------------------------
# Independent transit-depth check (cross-check vs. BLS depth)
# ------------------------------------------------------------

def measure_transit_depth(phase, flux, period, duration, t0):
    """
    Measures transit depth directly from a folded light curve by
    averaging flux in vs. out of the transit window, as an INDEPENDENT
    cross-check against the BLS depth already carried in
    solution['depth'] (from analysis.py). If the two disagree
    substantially, that's worth a second look before committing to a
    full model fit -- see prepare_model_inputs() below, which does
    exactly that comparison.

    Parameters
    ----------
    phase : array
        Phase of the folded light curve (days), t0-centered so transit
        minimum is at phase = 0.
    flux : array
        Normalized flux values.
    period : float
        Orbital period (days). Kept for interface consistency even
        though it isn't used directly once phase is already folded.
    duration : float
        Transit duration (days) -- solution['duration_days'], i.e. the
        ADOPTED theoretical duration from analysis.py.
    t0 : float
        Mid-transit time. Kept for interface consistency; phase is
        assumed already centered on t0.

    Returns
    -------
    tuple
        (baseline_flux, in_transit_flux, depth)
    """
    duration_phase = duration / period
    transit_center = 0.0

    in_transit_mask = np.abs(phase - transit_center) < duration_phase / 2
    baseline_mask = ~in_transit_mask

    baseline_flux = np.mean(flux[baseline_mask])
    in_transit_flux = np.mean(flux[in_transit_mask])
    depth = baseline_flux - in_transit_flux

    return baseline_flux, in_transit_flux, depth


# ------------------------------------------------------------
# Planet radius
# ------------------------------------------------------------

def calculate_planet_radius(depth, st_rad):
    """
    Rp/R* = sqrt(depth), the simple geometric relation for an opaque
    disk transiting a uniform stellar disk (Seager & Mallen-Ornelas
    2003, Eq. 3; ignores limb darkening, which the full model fit in
    model.py accounts for properly). Converts Rp/R* to physical units
    using the star's known radius.

    Parameters
    ----------
    depth : float
        Transit depth (fraction, e.g. 0.01 = 1%).
    st_rad : float
        Stellar radius in solar radii (R_sun).

    Returns
    -------
    dict
        {'rp_rstar': float, 'rp_rsun': float, 'rp_rjup': float, 'rp_rearth': float}
    """
    R_sun_to_Rjup = 9.73
    R_sun_to_Rearth = 109.2

    rp_rstar = np.sqrt(depth)

    rp_rsun = rp_rstar * st_rad
    rp_rjup = rp_rsun * R_sun_to_Rjup
    rp_rearth = rp_rsun * R_sun_to_Rearth

    return {
        'rp_rstar': rp_rstar,
        'rp_rsun': rp_rsun,
        'rp_rjup': rp_rjup,
        'rp_rearth': rp_rearth
    }


# ------------------------------------------------------------
# Inclination bounds (for the model fit)
# ------------------------------------------------------------

def calculate_inclination_limits(a_rstar, depth):
    """
    Bounds on orbital inclination consistent with a transit being
    observed at all, plus an initial guess -- used by model.py to
    bound/seed the least-squares and MCMC fits rather than leaving
    inclination totally unconstrained.

    Geometry (circular orbit, Winn 2010 Sec. 3): a transit requires
    impact parameter b = (a/R*) cos(i) <~ 1 + Rp/R* (a grazing transit
    at worst). Rp/R* is allowed to vary +/-20% around sqrt(depth) here
    to give the bound some slack rather than pinning it to one exact
    depth measurement. Solving for i gives the minimum inclination
    below which no transit would be seen; i = 90 deg (edge-on, b=0)
    is always the maximum.

    Parameters
    ----------
    a_rstar : float
        Scaled semi-major axis (a/R*) -- solution['a_Rs'] from
        analysis.scaled_semi_major().
    depth : float
        Transit depth (fraction) -- solution['depth'] (BLS-adopted).

    Returns
    -------
    dict
        {'min_i_deg', 'max_i_deg', 'ini_i_deg',
         'min_i_rad', 'max_i_rad', 'ini_i_rad',
         'min_size', 'max_size', 'ini_size'}
        where '..._size' is Rp/R* at the -20%/+20%/midpoint depth used
        for the bound (not the adopted Rp/R* from calculate_planet_radius).
    """
    max_size = np.sqrt(depth + depth * 0.2)
    min_size = np.sqrt(depth - depth * 0.2)
    ini_size = (min_size + max_size) / 2

    max_i_deg = 90.0

    b_max = 1 + max_size
    min_i_rad = np.arccos((1 / a_rstar) * b_max)
    min_i_deg = np.rad2deg(min_i_rad)

    ini_i_deg = (min_i_deg + max_i_deg) / 2

    return {
        'min_size': min_size,
        'max_size': max_size,
        'ini_size': ini_size,
        'min_i_deg': min_i_deg,
        'max_i_deg': max_i_deg,
        'ini_i_deg': ini_i_deg,
        'min_i_rad': min_i_rad,
        'max_i_rad': np.deg2rad(max_i_deg),
        'ini_i_rad': np.deg2rad(ini_i_deg)
    }


# ------------------------------------------------------------
# run all pre-model steps, package for model.py
# ------------------------------------------------------------

def prepare_model_inputs(transit_window, solution, stellar_radius, depth_tolerance=0.2):
    """
    Runs the pre-model calculations against the adopted BLS solution
    from analysis.py and packages everything model.py needs to build
    and bound the transit model fit, in one call.

    Steps
    -----
    1. measure_transit_depth() on the folded/windowed light curve, as
       an independent check against solution['depth'] (BLS-adopted) --
       flags (doesn't fail) if the two disagree by more than
       depth_tolerance.
    2. calculate_planet_radius() from the adopted BLS depth.
    3. calculate_inclination_limits() from solution['a_Rs'] and the
       adopted BLS depth.

    Parameters
    ----------
    transit_window : LightCurve
        Output of analysis.get_transit_window().
    solution : dict
        Output of analysis.save_transit_solution() -- needs
        'period_days', 'duration_days', 't0', 'depth', 'a_Rs'.
    stellar_radius : float
        Stellar radius in solar radii (R_sun) -- carried through from
        query.py, not re-queried here.
    depth_tolerance : float
        Fractional disagreement between measured and BLS depth above
        which a warning is printed (default 0.2 = 20%).

    Returns
    -------
    dict
        Combined depth-check, radius, and inclination-bound results,
        ready to hand to model.py.
    """
    phase = transit_window.time.value
    flux = transit_window.flux.value

    baseline_flux, in_transit_flux, depth_measured = measure_transit_depth(
        phase, flux,
        solution["period_days"], solution["duration_days"], solution["t0"]
    )

    depth_bls = solution["depth"]
    rel_diff = abs(depth_measured - depth_bls) / depth_bls

    print(f"\n--- Pre-model checks for {solution['planet_name']} ---")
    print(f"Measured depth (folded window) = {depth_measured * 100:.4f}%")
    print(f"BLS depth (adopted)            = {depth_bls * 100:.4f}%")
    if rel_diff > depth_tolerance:
        print(f"WARNING: depths disagree by {rel_diff * 100:.1f}% "
              f"(> {depth_tolerance * 100:.0f}% tolerance) -- worth double-checking "
              f"the fold/window before modeling.")
    else:
        print(f"Depths agree within {rel_diff * 100:.1f}% -- consistent, proceeding.")

    radius_info = calculate_planet_radius(depth_bls, stellar_radius)
    print(f"Rp/R* = {radius_info['rp_rstar']:.4f}")
    print(f"Rp = {radius_info['rp_rjup']:.3f} R_Jup = {radius_info['rp_rearth']:.2f} R_Earth")

    inclination_info = calculate_inclination_limits(solution["a_Rs"], depth_bls)
    print(f"Inclination bounds: [{inclination_info['min_i_deg']:.2f}, "
          f"{inclination_info['max_i_deg']:.2f}] deg, "
          f"initial guess {inclination_info['ini_i_deg']:.2f} deg")

    return {
        "depth_measured": depth_measured,
        "depth_bls": depth_bls,
        "baseline_flux": baseline_flux,
        "in_transit_flux": in_transit_flux,
        **radius_info,
        **inclination_info,
    }