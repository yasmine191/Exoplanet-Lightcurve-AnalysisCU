# ============================================================
# ---------------model.py-----------------------------------
#
# Transit model + fit. Uncertainties are reported from the MCMC
# (Bayesian) posterior throughout, NOT from the least-squares
# covariance -- least-squares is used only to get a fast best-fit
# starting point for the MCMC walkers.
#
# Only TWO transit S/N numbers are computed here: boxcar and
# trapezoidal (Kipping 2023), evaluated on the SAME T14 window so
# they're directly comparable. 
#
# References
# ----------
# Kipping (2013), MNRAS 435, 2152 -- (q1, q2) <-> (u1, u2) limb-darkening
#   reparameterization used in kipping_q_to_u / kipping_u_to_q, chosen so
#   uniform sampling on [0,1]^2 always lands on a physical (u1, u2).
# Coulombe et al. (2024), Nature -- notes the Kipping (2013) physical
#   triangle can exclude limb-brightening solutions and bias Rp/R* low in
#   weak-limb-darkening regimes (referenced in is_physical_ld below).
# Parviainen (2015), MNRAS 450, 3233 -- PyTransit, used here for the
#   quadratic-limb-darkening transit model itself (build_transit_model).
# Foreman-Mackey et al. (2013), PASP 125, 306 -- emcee, the affine-
#   invariant MCMC ensemble sampler used in run_mcmc_fit.
# Southworth (2008), MNRAS 386, 1644 -- prayer-bead (residual-shift)
#   red-noise error estimation (prayer_bead_errors) and the white/red
#   "beta factor" diagnostic (diagnose_fit_quality).
# Winn (2010), "Transits and Occultations", arXiv:1001.2010 -- T14/T23
#   duration relations (compute_transit_durations, Eqs. 14-15) and the
#   boxcar transit S/N definition (compute_transit_snr_boxcar).
# Kipping (2023) -- trapezoidal-refined transit S/N
#   (compute_transit_snr_trapezoidal), weighting the flat-bottom (T23)
#   portion of the transit differently from the ingress/egress wings
#   instead of treating the whole T14 window as one boxcar.
# Müller et al. (2013), A&A 560, A112; Southworth (2008); Gazak et al.
#   (2011), Adv. Astron. 2011 -- derived quantities (depth, b, physical
#   u1/u2, ...) should be propagated through the full MCMC posterior,
#   not via linear/Gaussian error propagation on a point estimate
#   (summarize_posterior_physical).
# ============================================================

import numpy as np
from pytransit import QuadraticModel
from scipy.optimize import least_squares


# ------------------------------------------------------------
# Limb darkening: Kipping (2013) q1,q2 <-> physical u1,u2
# ------------------------------------------------------------

def kipping_q_to_u(q1, q2):
    """
    Convert Kipping (2013) sampling parameters (q1, q2) to physical
    quadratic limb-darkening coefficients (u1, u2):
        u1 = 2*sqrt(q1)*q2
        u2 = sqrt(q1)*(1 - 2*q2)
    Used inside the transit model during fitting/MCMC -- q1,q2 sampled
    uniformly on [0,1]^2 always maps to a physically valid (u1,u2).
    """
    sqrt_q1 = np.sqrt(q1)
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return u1, u2


def kipping_u_to_q(u1, u2):
    """
    Inverse of kipping_q_to_u: convert physical (u1, u2) to Kipping
    (q1, q2). Used to convert MCMC posterior samples back to physical
    coefficients for reporting, or to convert a theoretical (u1, u2)
    guess (e.g. from a Claret table) into a q1,q2 starting point.
        q1 = (u1 + u2)^2
        q2 = 0.5 * u1 / (u1 + u2)
    """
    usum = u1 + u2
    q1 = usum ** 2
    q2 = 0.5 * u1 / usum
    return q1, q2


def is_physical_ld(u1, u2):
    """
    Checks the three Kipping (2013) physical conditions on the quadratic
    limb-darkening law directly in u1,u2 space:
      - positivity everywhere:        u1 + 2*u2 > 0
      - monotonic center-to-limb:     u1 > 0
      - finite intensity at the limb: u1 + u2 < 1
    Sampling via q1,q2 in [0,1]^2 satisfies these by construction --
    this is a sanity check, not a filter.

    Note (Coulombe et al. 2024): enforcing this triangle excludes
    limb-brightening solutions and can bias Rp/R* slightly small in
    *weak*-limb-darkening regimes (e.g. JWST infrared). TESS's bandpass
    (~600-1000 nm) is not in that weak-LD regime, so this is a documented
    limitation rather than a concern for this fit.
    """
    return (u1 > 0) and (u1 + 2.0 * u2 > 0) and (u1 + u2 < 1.0)


# ------------------------------------------------------------
# Transit model (PyTransit, Parviainen 2015)
# ------------------------------------------------------------

def build_transit_model(time_array, period_days):
    """
    Builds a quadratic-limb-darkening transit model (PyTransit's
    QuadraticModel) evaluated at time_array, fixed at period_days.

    Returns
    -------
    model_given_theta : callable
        theta = [k, dt, c0, a, inc, q1, q2] -> model flux array.
        k=Rp/R*, dt=mid-transit offset, c0=flux offset, a=a/R*,
        inc=inclination (radians), q1/q2=Kipping LD sampling params.
    tm : pytransit.QuadraticModel
        The underlying model object, in case direct access is needed.
    """
    tm = QuadraticModel()
    tm.set_data(time_array)

    def model_given_theta(theta):
        k, dt, c0, a, inc, q1, q2 = theta
        u1, u2 = kipping_q_to_u(q1, q2)
        model_flux = tm.evaluate(
            k=float(k),
            ldc=[u1, u2],
            t0=float(dt),
            p=float(period_days),
            a=float(a),
            i=float(inc)
        )
        return model_flux + float(c0)

    return model_given_theta, tm


# ------------------------------------------------------------
# Least-squares fit -- STARTING POINT ONLY, not final uncertainties
# ------------------------------------------------------------

def fit_transit_model(model_given_theta, flux, ferr, theta0, bounds):
    """
    Least-squares transit fit with SVD-based (linearized) parameter
    uncertainties. Used ONLY to get a fast best-fit as the MCMC starting
    point (run_mcmc_fit below) -- these SVD errors treat residuals as
    white/uncorrelated and are NOT what gets reported as the final
    parameter uncertainties anywhere downstream. See run_mcmc_fit and
    summarize_posterior_physical for the numbers that are actually
    reported.

    Parameters
    ----------
    model_given_theta : callable
        theta -> model flux array (from build_transit_model).
    flux : array
        Observed (normalized) flux.
    ferr : array or None
        Per-point flux uncertainty. If None, residuals are unweighted.
    theta0 : array-like, length 7
        Initial guess [k, dt, c0, a, inc, q1, q2].
    bounds : tuple(list, list)
        (lower_bounds, upper_bounds), each length 7.

    Returns
    -------
    dict
        {'theta': best-fit parameters (starting point for MCMC),
         'perr': white-noise 1-sigma errors (SVD covariance) -- for
             diagnostics (e.g. the beta factor) only,
         'model_best': best-fit model flux,
         'residuals': flux - model_best,
         'chi2_red': reduced chi-square (None if ferr is None),
         'result': the raw scipy OptimizeResult}
    """
    def residuals(theta):
        mod = model_given_theta(theta)
        return (flux - mod) / ferr if ferr is not None else flux - mod

    result = least_squares(residuals, theta0, bounds=bounds)
    best_theta = result.x

    if result.jac is not None:
        _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov = np.dot(VT.T / s**2, VT)
        perr = np.sqrt(np.diag(cov))
    else:
        perr = np.full_like(best_theta, np.nan)

    model_best = model_given_theta(best_theta)
    residuals_best = flux - model_best

    chi2_red = None
    if ferr is not None:
        chi2 = np.sum(((flux - model_best) / ferr) ** 2)
        dof = len(flux) - len(best_theta)
        chi2_red = chi2 / dof

    return {
        'theta': best_theta,
        'perr': perr,
        'model_best': model_best,
        'residuals': residuals_best,
        'chi2_red': chi2_red,
        'result': result
    }


# ------------------------------------------------------------
# MCMC posterior -- PRIMARY (Bayesian) source of uncertainties
# ------------------------------------------------------------

def run_mcmc_fit(model_given_theta, flux, ferr, best_theta, bounds,
                  nwalkers=32, nsteps=5000, burn_in=1000, thin=15):
    """
    Samples the parameter posterior with emcee (Foreman-Mackey et al.
    2013). This is the PRIMARY source of every parameter uncertainty
    reported by this pipeline: it gives marginalized, possibly-asymmetric
    68% credible intervals that correctly capture parameter degeneracies
    (e.g. a/R* vs inclination) -- something the linearized SVD covariance
    in fit_transit_model cannot do.

    Parameters
    ----------
    model_given_theta : callable
    flux, ferr : array
    best_theta : array-like, length 7
        Starting point -- fit_transit_model(...)['theta'].
    bounds : tuple(list, list)
        (lower_bounds, upper_bounds) -- used as a flat (uniform) prior.
    nwalkers, nsteps, burn_in, thin : int
        emcee sampling controls. Increase nsteps / check convergence
        (e.g. sampler.get_autocorr_time()) for a real analysis.

    Returns
    -------
    dict
        {'flat_samples': ndarray (n_samples, 7) posterior draws,
         'median': array, 'err_lo': array, 'err_hi': array,
         'sampler': the emcee.EnsembleSampler, in case you want
         trace/corner plots}
    """
    import emcee

    lb, ub = bounds
    ndim = len(best_theta)

    def log_prior(theta):
        if all(lo <= v <= hi for v, lo, hi in zip(theta, lb, ub)):
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        mod = model_given_theta(theta)
        sigma2 = ferr**2 if ferr is not None else np.var(flux - mod)
        return -0.5 * np.sum((flux - mod)**2 / sigma2 + np.log(2 * np.pi * sigma2))

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    best_theta = np.asarray(best_theta, dtype=float)
    pos = best_theta + 1e-5 * np.maximum(np.abs(best_theta), 1e-3) * np.random.randn(nwalkers, ndim)
    pos = np.clip(pos, lb, ub)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    median = np.percentile(flat_samples, 50, axis=0)
    err_lo = median - np.percentile(flat_samples, 16, axis=0)
    err_hi = np.percentile(flat_samples, 84, axis=0) - median

    return {
        'flat_samples': flat_samples,
        'median': median,
        'err_lo': err_lo,
        'err_hi': err_hi,
        'sampler': sampler
    }


# ------------------------------------------------------------
# Prayer-bead red-noise check
# ------------------------------------------------------------

def prayer_bead_errors(model_given_theta, flux, ferr, best_theta, bounds, residuals_best):
    """
    Residual-permutation ("prayer-bead", Southworth 2008) uncertainty
    estimate. Cyclically SHIFTS the best-fit residuals and re-fits at
    each shift, so any time-correlated ("red") noise in the light curve
    (systematics, stellar variability) shows up as extra scatter in the
    re-fit parameters -- something the white-noise SVD covariance in
    fit_transit_model can't see. Feeds the beta factor in
    diagnose_fit_quality below.

    Runs in O(N) refits, so call this on the windowed/folded transit data
    (e.g. transit_window), not a full multi-sector light curve.

    Parameters
    ----------
    model_given_theta : callable
    flux, ferr : array
    best_theta : array-like, length 7
    bounds : tuple(list, list)
    residuals_best : array
        Residuals (flux - model_best) from the best fit.

    Returns
    -------
    dict
        {'perr_red': red-noise 1-sigma errors, length 7,
         'shifted_fits': ndarray (n_points-1, 7) of every re-fit}
    """
    model_best = model_given_theta(best_theta)
    n_points = len(flux)
    shifted_fits = []

    for shift in range(1, n_points):
        flux_shifted = model_best + np.roll(residuals_best, shift)

        def residuals(theta, flux_shifted=flux_shifted):
            mod = model_given_theta(theta)
            return (flux_shifted - mod) / ferr if ferr is not None else flux_shifted - mod

        res_pb = least_squares(residuals, best_theta, bounds=bounds)
        shifted_fits.append(res_pb.x)

    shifted_fits = np.array(shifted_fits)
    perr_red = np.std(shifted_fits, axis=0)

    return {'perr_red': perr_red, 'shifted_fits': shifted_fits}


# ------------------------------------------------------------
# Transit durations from geometry (needed for trapezoidal S/N)
# ------------------------------------------------------------

def compute_transit_durations(period_days, a_Rs, inc_rad, k, b):
    """
    Analytic T14 (1st-to-4th contact) and T23 (2nd-to-3rd contact,
    flat-bottom) durations from transit geometry (Winn 2010, Eqs. 14-15).
    Vectorized: works for scalars or full posterior arrays.

        T14 = (P/pi) * arcsin[ sqrt((1+k)^2 - b^2) / (a_Rs * sin(i)) ]
        T23 = (P/pi) * arcsin[ sqrt((1-k)^2 - b^2) / (a_Rs * sin(i)) ]

    T23 -> 0 for near-grazing transits where (1-k)^2 <= b^2 (no flat
    bottom).
    """
    sin_i = np.sin(inc_rad)

    inside14 = np.maximum((1.0 + k) ** 2 - b ** 2, 0.0)
    arg14 = np.clip(np.sqrt(inside14) / (a_Rs * sin_i), -1.0, 1.0)
    T14 = (period_days / np.pi) * np.arcsin(arg14)

    inside23 = np.maximum((1.0 - k) ** 2 - b ** 2, 0.0)
    arg23 = np.clip(np.sqrt(inside23) / (a_Rs * sin_i), -1.0, 1.0)
    T23 = (period_days / np.pi) * np.arcsin(arg23)

    return T14, T23


# ------------------------------------------------------------
# Transit S/N -- boxcar AND trapezoidal ONLY, same T14 window
# ------------------------------------------------------------

def compute_transit_snr_boxcar(depth, sigma_phot, N14):
    """
    Boxcar transit S/N (Winn 2010): treats the entire T14 window as one
    flat box of in-transit points.

        SNR_boxcar = (depth / sigma_phot) * sqrt(N14)

    N14 : number of in-transit points within the T14 window (float here
    since it's typically T14/cadence rather than an integer count).
    """
    return (depth / sigma_phot) * np.sqrt(N14)


def compute_transit_snr_trapezoidal(depth, sigma_phot, N14, N23):
    """
    Trapezoidal-refined transit S/N (Kipping 2023): down-weights the
    sloped ingress/egress relative to the flat-bottom (T23) portion,
    instead of treating the whole T14 window as one boxcar.

        SNR_trapezoidal = (depth / sigma_phot) * sqrt((N14 + 2*N23) / 3)
    """
    return (depth / sigma_phot) * np.sqrt((N14 + 2.0 * N23) / 3.0)


def compute_transit_snr_comparison(period_days, times, fit_result, mcmc_result):
    """
    The S/N step used by this pipeline: boxcar vs. trapezoidal (Kipping
    2023), evaluated on the SAME T14 window (from the best-fit geometry)
    so the two are directly comparable -- an earlier version of this
    pipeline computed sigma_phot/N14 slightly differently for each,
    which isn't an apples-to-apples comparison. Both S/N values are
    propagated through the full MCMC posterior (depth, a/R*, inc, k all
    vary per-sample), not evaluated once at a point estimate.

    An earlier combined "detection S/N + depth-fit S/N" function
    (mixing a Winn-2010-style detection S/N with a separate depth/
    sigma_depth ratio from the fit covariance) is intentionally not
    used here -- boxcar and trapezoidal are the two S/N numbers this
    pipeline reports.

    Parameters
    ----------
    period_days : float
    times : array
        Time values of the windowed/folded light curve used in the fit.
    fit_result : dict
        Output of fit_transit_model -- used only for the best-fit
        geometry that defines the shared T14 window, and its residuals
        for sigma_phot.
    mcmc_result : dict
        Output of run_mcmc_fit -- flat_samples provide the posterior
        propagation for k, a/R*, inc (and therefore depth, b, T14, T23).

    Returns
    -------
    dict
        {'snr_boxcar': (median, lo68, hi68),
         'snr_trapezoidal': (median, lo68, hi68),
         'T14_median_days': float, 'T23_median_days': float}
    """
    cadence_days = np.median(np.diff(np.sort(times)))

    samples = mcmc_result['flat_samples']
    k_s, a_s, inc_rad_s = samples[:, 0], samples[:, 3], samples[:, 4]
    b_s = a_s * np.cos(inc_rad_s)
    depth_s = k_s ** 2

    T14_s, T23_s = compute_transit_durations(period_days, a_s, inc_rad_s, k_s, b_s)

    # Shared T14 window, from the best-fit (not posterior-sampled) geometry,
    # used identically for both boxcar and trapezoidal sigma_phot/masking.
    k_fit, dt_fit, c0_fit, a_fit, inc_fit = fit_result['theta'][:5]
    b_fit = a_fit * np.cos(inc_fit)
    T14_best, T23_best = compute_transit_durations(period_days, a_fit, inc_fit, k_fit, b_fit)

    t0_fit = dt_fit  # times are already relative to t0 in the windowed fit
    phase = ((times - t0_fit + period_days / 2) % period_days) - period_days / 2
    in_transit_mask_full = np.abs(phase) < T14_best / 2
    sigma_phot = np.std(fit_result['residuals'][~in_transit_mask_full])

    N14_s = T14_s / cadence_days
    N23_s = T23_s / cadence_days

    snr_box_s = compute_transit_snr_boxcar(depth_s, sigma_phot, N14_s)
    snr_trap_s = compute_transit_snr_trapezoidal(depth_s, sigma_phot, N14_s, N23_s)

    def summarize(arr):
        med, lo, hi = np.percentile(arr, [50, 16, 84])
        return (med, med - lo, hi - med)

    box = summarize(snr_box_s)
    trap = summarize(snr_trap_s)

    print("--- Transit S/N: boxcar vs. trapezoidal (Kipping 2023), shared T14 window ---")
    print(f"SNR_boxcar       = {box[0]:.2f}  +{box[2]:.2f} / -{box[1]:.2f}")
    print(f"SNR_trapezoidal  = {trap[0]:.2f}  +{trap[2]:.2f} / -{trap[1]:.2f}")

    return {
        'snr_boxcar': box,
        'snr_trapezoidal': trap,
        'T14_median_days': float(np.median(T14_s)),
        'T23_median_days': float(np.median(T23_s)),
    }


# ------------------------------------------------------------
# Fit quality diagnostic
# ------------------------------------------------------------

def diagnose_fit_quality(fit_result, pb_result, times,
                          chi2_warn_threshold=1.5, beta_warn_threshold=1.5,
                          n_bins=10, depth_frac_threshold=0.2):
    """
    Flags whether residuals look like white noise or hide structure
    (unremoved trend, spot-crossing, contamination, ephemeris mismatch).

    Uses the "beta factor" (Southworth 2008): the ratio of prayer-bead
    (red-noise) to white-noise (SVD) uncertainty on Rp/R*. beta >> 1
    means real time-correlated noise is present that the white-noise
    error underestimates.

    In-transit points are identified directly from the best-fit model
    (model flux more than depth_frac_threshold of the full depth below
    baseline), not from an externally-passed mask/duration -- avoids a
    unit mismatch silently producing an empty or wrong selection.
    """
    chi2_red = fit_result['chi2_red']
    beta_k = pb_result['perr_red'][0] / fit_result['perr'][0]

    model_best = fit_result['model_best']
    residuals = fit_result['residuals']

    baseline = np.median(model_best)
    depth_est = baseline - np.min(model_best)
    if depth_est <= 0:
        raise ValueError("Best-fit model shows no dip below baseline -- "
                          "check that the fit actually converged (fit_result['theta']).")

    in_transit_mask = model_best < (baseline - depth_frac_threshold * depth_est)

    t_in = times[in_transit_mask]
    r_in = residuals[in_transit_mask]

    if t_in.size < n_bins:
        print(f"Only {t_in.size} in-transit points found -- too few to bin reliably.")
        print("Reporting global chi^2/beta only, skipping binned structure check.")
        flagged_bins = []
    else:
        bin_edges = np.linspace(t_in.min(), t_in.max(), n_bins + 1)
        flagged_bins = []
        for i in range(n_bins):
            sel = (t_in >= bin_edges[i]) & (t_in < bin_edges[i + 1])
            if sel.sum() < 3:
                continue
            bin_mean = np.mean(r_in[sel])
            bin_sem = np.std(r_in[sel]) / np.sqrt(sel.sum())
            n_sigma = bin_mean / bin_sem if bin_sem > 0 else 0
            if abs(n_sigma) > 3:
                flagged_bins.append((bin_edges[i], bin_edges[i + 1], n_sigma))

    print(f"Reduced chi^2               = {chi2_red:.3f}")
    print(f"Beta factor (red/white)     = {beta_k:.2f}")
    print(f"In-transit points used      = {t_in.size}")
    print(f"Flagged residual bins (|mean| > 3 SEM): {len(flagged_bins)} / {n_bins}")

    if chi2_red <= chi2_warn_threshold and not flagged_bins:
        verdict = "GOOD"
        print("\nFit quality looks good -- residuals consistent with noise.")

    elif flagged_bins or beta_k > beta_warn_threshold:
        verdict = "SYSTEMATIC"
        print("\nResiduals show STRUCTURE, not just scatter.")
        print("Inspect the light curve for an unremoved trend, a spot-crossing,")
        print("contamination, or an ephemeris mismatch.")
        if flagged_bins:
            print("Flagged time ranges (days from mid-transit):")
            for t0, t1, ns in flagged_bins:
                print(f"  [{t0:+.4f}, {t1:+.4f}]  ({ns:+.1f} sigma)")

    else:
        verdict = "UNDERESTIMATED_ERRORS"
        print("\nchi^2_red > 1 but residuals look white (beta ~ 1).")
        print("Points to underestimated ferr, not a broken model.")
        print("Use the prayer-bead (red-noise) uncertainties, not the white-noise ones.")

    return {'chi2_red': chi2_red, 'beta_k': beta_k,
            'flagged_bins': flagged_bins, 'verdict': verdict,
            'n_in_transit': int(t_in.size)}


# ------------------------------------------------------------
# Final physical parameters -- full posterior propagation (Bayesian)
# ------------------------------------------------------------
# Per Müller et al. (2013): derived quantities should be propagated
# through posterior samples, not via linear/Gaussian error propagation
# on the median. Per Southworth (2008) / Gazak et al. (2011): report
# 68% credible intervals from the MCMC posterior as the primary
# uncertainty, not covariance-matrix (least-squares) errors, which are
# unreliable under parameter correlation.

def summarize_posterior_physical(mcmc_result):
    """
    Converts the full MCMC posterior (still in fit units: theta =
    [k, dt, c0, a, inc_rad, q1, q2]) into physical, reportable
    quantities, with every derived value's uncertainty coming from
    percentiles of the posterior itself -- not linear propagation of
    point-estimate errors. This is the PRIMARY source of the reported
    parameter values and uncertainties for this pipeline.
    """
    samples = mcmc_result['flat_samples']  # (n_samples, 7)

    k_s = samples[:, 0]
    a_s = samples[:, 3]
    inc_rad_s = samples[:, 4]
    q1_s, q2_s = samples[:, 5], samples[:, 6]

    depth_s = k_s ** 2
    inc_deg_s = np.rad2deg(inc_rad_s)
    u1_s, u2_s = kipping_q_to_u(q1_s, q2_s)
    b_s = a_s * np.cos(inc_rad_s)   # impact parameter, since we're here

    def report(name, arr, fmt="{:.5f}"):
        med, lo16, hi84 = np.percentile(arr, [50, 16, 84])
        print(f"{name:14s} = {fmt.format(med)}  "
              f"+{fmt.format(hi84 - med)} / -{fmt.format(med - lo16)}")
        return med, med - lo16, hi84 - med

    print("--- Final parameter uncertainties (68% credible intervals, full posterior) ---")
    results = {
        'Rp/R*':   report("Rp/R*", k_s),
        'a/R*':    report("a/R*", a_s),
        'inc_deg': report("inc (deg)", inc_deg_s, "{:.3f}"),
        'depth':   report("depth", depth_s),
        'u1':      report("u1", u1_s),
        'u2':      report("u2", u2_s),
        'b':       report("impact param b", b_s),
    }
    return results


# ------------------------------------------------------------
# Driver: run the whole model stage
# ------------------------------------------------------------

def run_transit_model(transit_window, solution, pre_model_result):
    """
    Runs the full model stage end to end:
      1. build the transit model,
      2. least-squares fit  -- MCMC starting point only,
      3. MCMC posterior     -- PRIMARY uncertainties (Bayesian),
      4. prayer-bead red-noise check,
      5. fit-quality diagnostic,
      6. boxcar + trapezoidal S/N comparison (only these two),
      7. final physical parameter summary from the full posterior.

    Parameters
    ----------
    transit_window : LightCurve
        Output of analysis.get_transit_window().
    solution : dict
        Output of analysis.save_transit_solution().
    pre_model_result : dict
        Output of pre_model.prepare_model_inputs() -- provides the
        inclination/size bounds used to seed and bound the fit.

    Returns
    -------
    dict
        Everything comparison.py needs: fit_result, mcmc_result,
        pb_result, diagnostics, snr_comparison, posterior_summary.
    """
    times = np.asarray(transit_window.time.value)
    flux = np.asarray(transit_window.flux.value)
    ferr = np.asarray(transit_window.flux_err.value) if transit_window.flux_err is not None else None

    period = solution["period_days"]

    theta0 = [pre_model_result['ini_size'], 0.0, 0.0, solution['a_Rs'],
              pre_model_result['ini_i_rad'], 0.5, 0.5]
    lb = [pre_model_result['min_size'], -0.01, -0.05, 1.0,
          pre_model_result['min_i_rad'], 0.0, 0.0]
    ub = [pre_model_result['max_size'], 0.01, 0.05, solution['a_Rs'] * 2,
          pre_model_result['max_i_rad'], 1.0, 1.0]
    bounds = (lb, ub)

    model_given_theta, tm = build_transit_model(times, period)

    print("Starting least-squares fit (MCMC starting point only)...")
    fit_result = fit_transit_model(model_given_theta, flux, ferr, theta0, bounds)
    print(f"Least-squares best fit found, reduced chi^2 = {fit_result['chi2_red']:.3f}")

    print("\nRunning MCMC (primary, Bayesian uncertainties)...")
    mcmc_result = run_mcmc_fit(model_given_theta, flux, ferr, fit_result['theta'], bounds)

    print("\nRunning prayer-bead red-noise check...")
    pb_result = prayer_bead_errors(model_given_theta, flux, ferr, fit_result['theta'],
                                    bounds, fit_result['residuals'])

    print("\nDiagnosing fit quality...")
    diagnostics = diagnose_fit_quality(fit_result, pb_result, times)

    print("\nComputing transit S/N (boxcar vs. trapezoidal)...")
    snr_comparison = compute_transit_snr_comparison(period, times, fit_result, mcmc_result)

    print()
    posterior_summary = summarize_posterior_physical(mcmc_result)

    return {
        'fit_result': fit_result,
        'mcmc_result': mcmc_result,
        'pb_result': pb_result,
        'diagnostics': diagnostics,
        'snr_comparison': snr_comparison,
        'posterior_summary': posterior_summary,
        'times': times,
        'flux': flux,
        'ferr': ferr,
    }
import matplotlib.pyplot as plt

def plot_transit_model(model_result, planet_name=""):
    """
    Plots the phased light curve with the best-fit transit model in the top panel
    and fit residuals in the bottom panel.
    """
    times = model_result['times']
    flux = model_result['flux']
    ferr = model_result['ferr']
    model_best = model_result['fit_result']['model_best']
    residuals = model_result['fit_result']['residuals']

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # Sort array by phase/time so the model plots as a smooth line
    sort_idx = np.argsort(times)

    # --- Top Panel: Lightcurve + Model ---
    if ferr is not None:
        ax1.errorbar(times, flux, yerr=ferr, fmt='.', color='black', alpha=0.3, label='Data', zorder=1)
    else:
        ax1.plot(times, flux, '.', color='black', alpha=0.3, label='Data', zorder=1)

    ax1.plot(times[sort_idx], model_best[sort_idx], color='red', lw=2, label='Best-fit Model', zorder=2)
    ax1.set_ylabel("Normalized Flux")
    ax1.set_title(f"Transit Model & Residuals — {planet_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Bottom Panel: Residuals ---
    ax2.scatter(times, residuals, color='purple', s=8, alpha=0.5)
    ax2.axhline(0.0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel("Phase / Time relative to mid-transit (days)")
    ax2.set_ylabel("Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()