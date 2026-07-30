import numpy as np
import lightkurve as lk
DEFAULT_AUTHORS = {"TESS": "SPOC", "Kepler": "Kepler", "K2": "K2"}


def check_lightcurve_quality(lc, max_gap_fraction=0.3, min_points=500, max_single_gap_days=2.0):
    """
    Quick data-quality gate before running BLS.
    Returns True if the light curve is usable, False (with a printed reason) if not.
    """
    time = lc.time.value
    n_points = len(time)
    if n_points < min_points:
        print(f"Too few data points ({n_points} < {min_points}) — try a different target.")
        return False
    # Expected cadence: median spacing between consecutive points
    dt = np.diff(time)
    median_cadence = np.median(dt)
    # Total time span vs. how much is actually covered by data
    time_span = time[-1] - time[0]
    expected_points = time_span / median_cadence
    coverage_fraction = n_points / expected_points
    if coverage_fraction < (1 - max_gap_fraction):
        print(f"Light curve has too many gaps (only {coverage_fraction*100:.1f}% "
              f"of expected coverage) — try a different target.")
        return False
    # Longest single gap, in days
    longest_gap = dt.max()
    if longest_gap > max_single_gap_days:
        print(f"Longest single gap is {longest_gap:.2f} days — likely a major "
              f"data downlink/sector gap. Try a different target or a different sector.")
        return False
    print(f"Light curve OK: {n_points} points, {coverage_fraction*100:.1f}% coverage, "
          f"longest gap {longest_gap:.2f} d")
    return True


def lc_download(planet_name, mission, author=None):
    """
    Downloads and cleans a light curve for a given planet.

    Parameters
    ----------
    planet_name : str
        Name of the planet (written exactly like tje catalogue)
    mission : str
        Mission to search (TESS, Kepler, K2)
        
    Returns
    -------
    lc_clean : LightCurve
        Cleaned, normalized, flattened light curve
    """
    author = author or DEFAULT_AUTHORS[mission]
    print(f"Searching for {planet_name} in {mission}...")
    
    search_result = lk.search_lightcurve(
        planet_name, 
        mission=mission, 
        author=author
    )
    
#   check if anything was found
    if len(search_result) == 0:
        raise ValueError(f"No {mission} data found for {planet_name} with author {author}.")
    
    print(f"Found {len(search_result)} sector(s) — downloading all...")
    
#   download all sectors and stitch it later
    lc_collection = search_result.download_all(
        quality_bitmask="hardest", 
        flux_column="pdcsap_flux"
    )
    
#   stitch if multiple sectors
    if isinstance(lc_collection, lk.LightCurveCollection):
        lc = lc_collection.stitch()
    else:
        lc = lc_collection
    
#   clean and normalize and flatten
    lc_clean = lc.normalize().remove_nans().remove_outliers(sigma=5).flatten(
    window_length=401,
    break_tolerance=5   # gaps > 5x the median cadence spacing are treated as breaks
    )
    
    print(f"Download complete. {len(lc_clean)} data points.")

#   instant quality gate — bail out here if the light curve isn't usable
    if not check_lightcurve_quality(lc_clean):
        raise ValueError("Light curve failed quality check — choose a different target.")

    return lc_clean

#==================================================================================================================

import re
import pandas as pd

def select_observation(planet_name, mission, author=None):
    """
    Search for all available observations of a planet for a given mission,
    display the available sector/quarter/campaign options, and let the
    user pick one.

    Parameters
    ----------
    planet_name : str
    mission : str
        "TESS", "Kepler", or "K2"

    Returns
    -------
    search_result : lightkurve.SearchResult
        The full search result (needed to download the chosen row).
    idx : int
        Index of the row the user picked in search_result.
    """
    author = author or DEFAULT_AUTHORS[mission]
    print(f"Searching for {planet_name} in {mission}...")

    search_result = lk.search_lightcurve(
        planet_name,
        mission=mission,
        author=author
    )

    if len(search_result) == 0:
        raise ValueError(f"No {mission} data found for {planet_name} with author {author}.")

    # Label for the sector/quarter/campaign number, depending on mission
    label_map = {"TESS": "Sector", "Kepler": "Quarter", "K2": "Campaign"}
    label = label_map.get(mission, "Observation")

    table = search_result.table.to_pandas()

    # Pull out the sector/quarter/campaign number from the "mission" column
    def extract_number(mission_str):
        """
        Let the user choose the sector/quarter/campaign number
        
        """
        match = re.search(r"(\d+)", str(mission_str))
        return int(match.group(1)) if match else None

    table[label] = table["mission"].apply(extract_number)

    display_cols = [label, "author", "exptime", "year"]
    display_cols = [c for c in display_cols if c in table.columns]

    print(f"\nAvailable observations for {planet_name} ({mission}):")
    print(table[display_cols].to_string())

    while True:
        choice = table_choice = input(
            f"\nEnter the row index of the {label.lower()} you want to download: "
        ).strip()
        if choice.isdigit() and int(choice) in table.index:
            idx = int(choice)
            print(f"You selected row {idx}: {label} {table.loc[idx, label]}")
            return search_result, idx
        else:
            print("Invalid index. Please try again.")

#==================================================================================================================

def get_quality_choice():
    """
    Let the user choose the quality bitmask used when downloading the
    light curve. Stricter bitmasks remove more flagged cadences.

    Returns
    -------
    quality_bitmask : str
        One of "none", "default", "hard", "hardest".
    """
    options = {
        "1": ("none", "Keep all cadences, no quality filtering."),
        "2": ("default", "Remove cadences with the most common quality issues."),
        "3": ("hard", "Remove cadences flagged for most known instrumental issues."),
        "4": ("hardest", "Remove every cadence with any quality flag set (strictest) and the most recommended."),
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

#==================================================================================================================

def download_selected(search_result, idx, quality_bitmask="hardest", flux_column="pdcsap_flux"):
    """
    Download a single chosen observation (sector/quarter/campaign) and
    clean it (normalize, remove NaNs/outliers, flatten).

    Parameters
    ----------
    search_result : lightkurve.SearchResult
        Result returned by select_observation().
    idx : int
        Row index chosen by the user.
    quality_bitmask : str
        Quality bitmask to apply during download.
    flux_column : str
        Flux column to use (default "pdcsap_flux").

    Returns
    -------
    lc_clean : LightCurve
        Cleaned, normalized, flattened light curve for the chosen observation.
    """
    print(f"Downloading row {idx} with quality_bitmask='{quality_bitmask}'...")

    lc = search_result[idx].download(
        quality_bitmask=quality_bitmask,
        flux_column=flux_column
    )

    lc_clean = lc.normalize().remove_nans().remove_outliers(sigma=5).flatten(
    window_length=401,
    break_tolerance=5   # gaps > 5x the median cadence spacing are treated as breaks
    )

    print(f"Download complete. {len(lc_clean)} data points.")

#   instant quality gate — bail out here if the light curve isn't usable
    if not check_lightcurve_quality(lc_clean):
        raise ValueError("Light curve failed quality check — choose a different target.")

    return lc_clean