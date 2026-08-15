# ============================================================
# ---------------comparison.py---------------------------------
#
# Compares this pipeline's fitted transit parameters against the
# published values on the NASA Exoplanet Archive for the same planet.
# Limb darkening (u1, u2) is intentionally excluded -- not tabulated
# in the archive for any target, current-mission or otherwise.
#
# Uses query.fetch_archive_params() -- the one shared archive-query
# function (also used by query.py's Hot-Jupiter table and by
# analysis.scaled_semi_major's stellar-parameter fallback) -- instead
# of a separate raw-request implementation, which is what this file
# used to contain as a third, slightly different copy of the same
# retry/query logic.
# ============================================================

from query import fetch_archive_params

ARCHIVE_COLUMNS = [
    "pl_orbper", "pl_orbpererr1",
    "pl_ratdor", "pl_ratdorerr1",
    "pl_orbincl", "pl_orbinclerr1",
    "pl_imppar", "pl_impparerr1",
    "pl_trandep", "pl_trandeperr1",
    "pl_trandur", "pl_trandurerr1",
    "pl_ratror", "pl_ratrorerr1",
]


def query_published_transit_params(planet_name):
    """
    Fetches published transit/orbital parameters for `planet_name` via
    query.fetch_archive_params() (pscomppars, falling back to ps
    column-by-column) and renames them to friendly keys.

    Returns
    -------
    dict
        {'period_days','period_err','a_Rs','a_Rs_err','inc_deg','inc_err',
         'b','b_err','depth_pct','depth_err','duration_hr','duration_err',
         'rprs','rprs_err'} -- floats, or None if that value isn't
        published for this planet in either archive table.
    """
    raw = fetch_archive_params(planet_name, ARCHIVE_COLUMNS)

    return {
        'period_days': raw['pl_orbper'],   'period_err':   raw['pl_orbpererr1'],
        'a_Rs':        raw['pl_ratdor'],   'a_Rs_err':     raw['pl_ratdorerr1'],
        'inc_deg':     raw['pl_orbincl'],  'inc_err':      raw['pl_orbinclerr1'],
        'b':           raw['pl_imppar'],   'b_err':        raw['pl_impparerr1'],
        'depth_pct':   raw['pl_trandep'],  'depth_err':    raw['pl_trandeperr1'],
        'duration_hr': raw['pl_trandur'],  'duration_err': raw['pl_trandurerr1'],
        'rprs':        raw['pl_ratror'],   'rprs_err':     raw['pl_ratrorerr1'],
    }


def offer_archive_comparison(planet_name, solution, model_result):
    """
    Interactively asks the user whether to compare our fitted transit
    parameters against the published NASA Exoplanet Archive values, and
    prints a side-by-side table if so.

    Parameters
    ----------
    planet_name : str
    solution : dict
        Output of analysis.save_transit_solution() -- used for the
        adopted period (fixed during the fit, not a posterior parameter).
    model_result : dict
        Output of model.run_transit_model(). Uses:
          - posterior_summary[...] : each a (median, err_lo, err_hi)
            tuple, the full-posterior values (Müller et al. 2013;
            Southworth 2008) -- NOT the least-squares point estimate.
          - snr_comparison['T14_median_days'] : duration comparison.

    Returns
    -------
    dict or None
        The published archive values, or None if the user declined or
        no archive record was found.
    """
    resp = input(f"Compare fitted parameters for {planet_name} to the "
                 f"NASA Exoplanet Archive? [y/n]: ").strip().lower()
    if resp not in ('y', 'yes'):
        print("Skipping archive comparison.")
        return None

    print("Querying NASA Exoplanet Archive...")
    archive = query_published_transit_params(planet_name)
    if all(v is None for v in archive.values()):
        print(f"No archive record found for '{planet_name}'.")
        return None

    posterior = model_result['posterior_summary']
    rprs_med, rprs_lo, rprs_hi = posterior['Rp/R*']
    a_Rs_med, a_Rs_lo, a_Rs_hi = posterior['a/R*']
    inc_med, inc_lo, inc_hi = posterior['inc_deg']
    depth_med, _, _ = posterior['depth']
    b_med, _, _ = posterior['b']
    T14_hr = model_result['snr_comparison']['T14_median_days'] * 24.0

    rprs_err = (rprs_lo + rprs_hi) / 2
    a_Rs_err = (a_Rs_lo + a_Rs_hi) / 2
    inc_err = (inc_lo + inc_hi) / 2

    rows = [
        ("Period (days)",      solution['period_days'], None,
                                archive['period_days'], archive['period_err']),
        ("Rp/R*",              rprs_med, rprs_err,
                                archive['rprs'], archive['rprs_err']),
        ("a/R*",               a_Rs_med, a_Rs_err,
                                archive['a_Rs'], archive['a_Rs_err']),
        ("Inclination (deg)",  inc_med, inc_err,
                                archive['inc_deg'], archive['inc_err']),
        ("Impact parameter b", b_med, None,
                                archive['b'], archive['b_err']),
        ("Depth (%)",          depth_med * 100, None,
                                archive['depth_pct'], archive['depth_err']),
        ("Duration (hr)",      T14_hr, None,
                                archive['duration_hr'], archive['duration_err']),
    ]

    print(f"\n{'Parameter':<20}{'Our fit':<20}{'NASA Archive':<20}{'Diff':<10}")
    print("-" * 70)
    for name, our_val, our_err, arc_val, arc_err in rows:
        our_str = "N/A" if our_val is None else (
            f"{our_val:.4f} ± {our_err:.4f}" if our_err else f"{our_val:.4f}")
        arc_str = "N/A" if arc_val is None else (
            f"{arc_val:.4f} ± {arc_err:.4f}" if arc_err else f"{arc_val:.4f}")
        diff_str = f"{our_val - arc_val:+.4f}" if (our_val is not None and arc_val is not None) else "—"
        print(f"{name:<20}{our_str:<20}{arc_str:<20}{diff_str:<10}")

    return archive