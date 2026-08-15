# ============================================================
# ---------------download.py----------------------------------
#
# Interactive light-curve download: lets the user pick which
# sector/quarter/campaign to use, which quality bitmask to
# apply, downloads that single observation, and runs a data-
# quality gate before handing it off to analysis.py.
#
# lightkurve docs: https://docs.lightkurve.org
# ============================================================

import re
import numpy as np
import pandas as pd
import lightkurve as lk

from query import DEFAULT_AUTHORS


# ------------------------------------------------------------
# Data-quality gate — shared by download_selected() below.
# ------------------------------------------------------------

def check_lightcurve_quality(lc, max_gap_fraction=0.3, min_points=500, max_single_gap_days=2.0):
    """
    Quick data-quality gate run right after download, before any BLS/model
    fitting starts. 
    Returns True if the light curve is usable, False (with
    a printed reason) if not. 

    Checks
    ------
    1. Enough total points (min_points).
    2. Enough of the expected time span is actually covered by data
       (coverage_fraction, based on median cadence) — catches heavily
       gapped light curves.
    3. No single gap longer than max_single_gap_days — catches a major
       data downlink/sector-boundary gap that a coverage-fraction check
       alone might miss.
    """
    time = lc.time.value
    n_points = len(time)
    if n_points < min_points:
        print(f"Too few data points ({n_points} < {min_points}) — try a different target.")
        return False

    dt = np.diff(time)
    median_cadence = np.median(dt)
    time_span = time[-1] - time[0]
    expected_points = time_span / median_cadence
    coverage_fraction = n_points / expected_points
    if coverage_fraction < (1 - max_gap_fraction):
        print(f"Light curve has too many gaps (only {coverage_fraction*100:.1f}% "
              f"of expected coverage) — try a different target.")
        return False

    longest_gap = dt.max()
    if longest_gap > max_single_gap_days:
        print(f"Longest single gap is {longest_gap:.2f} days — likely a major "
              f"data downlink/sector gap. Try a different target or a different sector.")
        return False

    print(f"Light curve OK: {n_points} points, {coverage_fraction*100:.1f}% coverage, "
          f"longest gap {longest_gap:.2f} d")
    return True


# ------------------------------------------------------------
# Interactive sector/quarter/campaign + quality-bitmask choice
# ------------------------------------------------------------

def select_observation(planet_name, mission, author=None):
    """
    Searches for all available observations of a planet for a given
    mission, displays them, and lets the user pick one row.

    Returns
    -------
    search_result : lightkurve.SearchResult
        The full search result (download_selected() downloads the chosen row from this).
    idx : int
        Index of the row the user picked.
    """
    author = author or DEFAULT_AUTHORS[mission]
    print(f"Searching for {planet_name} in {mission}...")

    search_result = lk.search_lightcurve(planet_name, mission=mission, author=author)
    if len(search_result) == 0:
        raise ValueError(f"No {mission} data found for {planet_name} with author {author}.")

    label_map = {"TESS": "Sector", "Kepler": "Quarter", "K2": "Campaign"}
    label = label_map.get(mission, "Observation")

    table = search_result.table.to_pandas()

    def extract_number(mission_str):
        """Pulls the sector/quarter/campaign number out of the 'mission' column string."""
        match = re.search(r"(\d+)", str(mission_str))
        return int(match.group(1)) if match else None

    table[label] = table["mission"].apply(extract_number)

    display_cols = [c for c in [label, "author", "exptime", "year"] if c in table.columns]
    print(f"\nAvailable observations for {planet_name} ({mission}):")
    print(table[display_cols].to_string())

    while True:
        choice = input(f"\nEnter the row index of the {label.lower()} you want to download: ").strip()
        if choice.isdigit() and int(choice) in table.index:
            idx = int(choice)
            print(f"You selected row {idx}: {label} {table.loc[idx, label]}")
            return search_result, idx
        else:
            print("Invalid index. Please try again.")


def get_quality_choice():
    """
    Lets the user choose the quality bitmask used when downloading the
    light curve. Stricter bitmasks remove more flagged cadences.
    (See lightkurve docs for what each bitmask level filters.)

    Returns
    -------
    str : one of "none", "default", "hard", "hardest".
    """
    options = {
        "1": ("none", "Keep all cadences, no quality filtering."),
        "2": ("default", "Remove cadences with the most common quality issues."),
        "3": ("hard", "Remove cadences flagged for most known instrumental issues."),
        "4": ("hardest", "Remove every cadence with any quality flag set (strictest & recommended!!)."),
    }

    print("\nAvailable quality bitmask options:")
    for key, (name, desc) in options.items():
        print(f"{key}. {name} - {desc}")

    while True:
        choice = input("Enter option number (1-4), default is hardest: ").strip()
        if choice == "":
            return "hardest"
        elif choice in options:
            return options[choice][0]
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def download_selected(search_result, idx, quality_bitmask="hardest", flux_column="pdcsap_flux"):
    """
    Downloads the single chosen observation and cleans it: normalize,
    remove NaNs, remove 5-sigma outliers, and flatten (removes long-
    timescale stellar/instrumental trends via a Savitzky-Golay filter
    while preserving the much shorter transit signal).

    flatten() parameters
    ---------------------
    window_length=401 : filter window in cadences — wide enough to not
        distort a transit (which spans far fewer points than this) while
        still tracking slower trends.
    break_tolerance=5 : gaps > 5x the median cadence are treated as
        separate segments rather than flattened across (avoids the
        filter being warped by a real data gap).

    Returns
    -------
    lc_clean : lightkurve.LightCurve
    """
    print(f"Downloading row {idx} with quality_bitmask='{quality_bitmask}'...")

    lc = search_result[idx].download(quality_bitmask=quality_bitmask, flux_column=flux_column)

    lc_clean = lc.normalize().remove_nans().remove_outliers(sigma=5).flatten(
        window_length=401,
        break_tolerance=5
    )

    print(f"Download complete. {len(lc_clean)} data points.")

    if not check_lightcurve_quality(lc_clean):
        raise ValueError("Light curve failed quality check — choose a different target.")

    return lc_clean