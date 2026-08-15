# ============================================================
# -----------------query.py-----------------------------------
#
# This file handles all direct communication with the NASA Exoplanet
# Archive's TAP service, and the interactive planet/mission
# selection at the start of the pipeline.
#
# Archive reference: NASA Exoplanet Archive TAP service,
# https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usage.html
# Table schema (ps / pscomppars column definitions):
# https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
# ============================================================

import time
import requests
import pandas as pd
from io import StringIO
import lightkurve as lk

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Default pipeline/author to use per mission when downloading light curves
# (lightkurve needs this to disambiguate multiple reduction pipelines for
# the same mission). Used here for the mission-availability check, and
# again in download.py for the actual download.
DEFAULT_AUTHORS = {"TESS": "SPOC", "Kepler": "Kepler", "K2": "K2"}


def _run_tap_query(query, expect_prefix, max_retries=4, initial_delay=3):
    """
    Runs one ADQL query against the NASA Exoplanet Archive TAP service
    and returns the result as a pandas DataFrame, with exponential-backoff
    retries. This replaces three separate copies of this retry loop that
    existed across the original notebook (query_hot_jupiters,
    get_stellar_parameters, query_nasa_archive_transit_params).

    Parameters
    ----------
    query : str
        A complete ADQL SELECT statement.
    expect_prefix : str
        The expected first column name in a successful CSV response.
        A failed TAP query can still return HTTP 200 with an error message
        or VOTable body instead of CSV — checking the prefix catches that
        rather than silently parsing garbage as data.
    max_retries : int
    initial_delay : float
        Seconds before the first retry; doubles on each subsequent attempt.

    Returns
    -------
    pandas.DataFrame or None
        None only if every retry attempt failed (network/service issue) —
        callers decide how to handle that (raise, fall back, warn).
    """
    params = {"query": query.strip(), "format": "csv"}
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(TAP_URL, params=params, timeout=30)
            resp.raise_for_status()
            if not resp.text.lstrip().startswith(expect_prefix):
                raise ValueError(f"Unexpected response body: {resp.text[:200]}")
            return pd.read_csv(StringIO(resp.text))
        except Exception as err:
            print(f"TAP query attempt {attempt}/{max_retries} failed: {err}")
            if attempt == max_retries:
                print("NASA Exoplanet Archive query failed after all retries. "
                      "The TAP service may be temporarily down — see "
                      "https://exoplanetarchive.ipac.caltech.edu/ for status.")
                return None
            print(f"Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2


def _first_non_null(series):
    """Returns the first non-null value in a pandas Series, or None."""
    s = series.dropna()
    return float(s.iloc[0]) if not s.empty else None


def fetch_archive_params(planet_name, columns, table="pscomppars", fallback_table="ps"):
    """
    THE single function for "get me parameter columns for one named planet
    from the archive" — used throughout the pipeline (stellar mass/radius
    during planet selection, a/R* fallback in pre_model.py, published
    transit parameters in comparison.py) instead of each caller writing
    its own query.

    Tries `table` (default: pscomppars, the archive's one-row-per-planet
    curated "best value" table) first. For any column still null, falls
    back to `fallback_table` (default: ps, which has one row per published
    solution/paper — different papers populate different columns, so this
    takes the first non-null value found across all of that planet's rows).
    This two-table fallback is what fixed the earlier bug where Rp/R* and
    inclination showed as N/A despite being visible on the archive website:
    that data existed in a `ps` row that wasn't row 0.

    Parameters
    ----------
    planet_name : str
    columns : list of str
        Archive column names to fetch, e.g. ["st_mass", "st_rad"].
    table, fallback_table : str

    Returns
    -------
    dict
        {column_name: float or None}. None means the value truly isn't
        published in either table for this planet — not a query failure.
    """
    col_str = ", ".join(columns)

    df_primary = _run_tap_query(
        f"SELECT {col_str} FROM {table} WHERE pl_name = '{planet_name}'",
        expect_prefix=columns[0]
    )
    df_fallback = _run_tap_query(
        f"SELECT {col_str} FROM {fallback_table} WHERE pl_name = '{planet_name}'",
        expect_prefix=columns[0]
    )

    result = {}
    for col in columns:
        val = _first_non_null(df_primary[col]) if (df_primary is not None and col in df_primary) else None
        if val is None and df_fallback is not None and col in df_fallback:
            val = _first_non_null(df_fallback[col])
        result[col] = val

    return result


# ------------------------------------------------------------
# Hot Jupiter candidate table
# ------------------------------------------------------------

def query_hot_jupiters(max_retries=4, initial_delay=3):
    """
    Queries the archive for confirmed Hot Jupiters with near-zero
    eccentricity (e <= 0.05), transiting, short-period. This is a bulk
    multi-planet query (not per-planet), so it uses _run_tap_query
    directly rather than fetch_archive_params.

    Filter choices:
      - 0.25 < M_p < 13 M_Jup: standard hot-Jupiter mass range
      - e <= 0.05: near-circular orbit (simplifies duration/geometry
        assumptions used later in pre_model.py)
      - tran_flag = 1: must be a known transiting planet
      - P <= 10 days: "hot" — short period

    Returns
    -------
    pandas.DataFrame
        Columns: pl_name, hostname, pl_massj, pl_radj, pl_orbper,
        pl_orbeccen, tran_flag, disc_facility, disc_year, st_mass, st_rad.
        Rows missing any of the essential columns are dropped.
    """
    query = """
    SELECT pl_name, hostname, pl_massj, pl_radj,
           pl_orbper, pl_orbeccen, tran_flag,
           disc_facility, disc_year, st_mass, st_rad
    FROM ps
    WHERE pl_massj > 0.25 AND pl_massj < 13
      AND pl_orbeccen <= 0.05
      AND tran_flag = 1
      AND pl_orbper <= 10
    """
    print("Querying NASA Exoplanet Archive for Hot Jupiters...")
    df = _run_tap_query(query, expect_prefix="pl_name",
                         max_retries=max_retries, initial_delay=initial_delay)
    if df is None:
        raise RuntimeError("Could not retrieve the Hot Jupiter candidate table.")

    df = df.dropna(subset=["pl_name", "pl_orbper", "pl_massj", "pl_radj", "st_mass", "st_rad"])
    print(f"Found {len(df)} Hot Jupiters with e <= 0.05")
    return df


# ------------------------------------------------------------
# Interactive planet + mission selection
# ------------------------------------------------------------

def choose_planet(df):
    """
    Displays the candidate table and lets the user pick one planet by name.

    Returns
    -------
    planet_name : str
    stellar_mass : float   (st_mass, from the candidate table — no extra query)
    stellar_radius : float (st_rad, from the candidate table — no extra query)
    """
    print("\nAvailable Hot Jupiters:")
    print(df[["pl_name", "hostname", "pl_massj", "pl_orbper", "pl_orbeccen"]].to_string(index=False))

    while True:
        choice = input("\nEnter the planet name exactly as shown above: ").strip()
        if choice in df["pl_name"].values:
            print(f"You selected: {choice}")
            row = df.loc[df["pl_name"] == choice].iloc[0]
            return choice, row["st_mass"], row["st_rad"]
        else:
            print("Planet not found. Please try again and make sure the name is exactly the same!")


def get_available_missions(planet_name):
    """
    Single unfiltered lightkurve search to see which missions actually
    have data for this planet, so the user is only ever offered missions
    that exist — avoids a later "no data found" error after several prompts.

    Returns
    -------
    list of str : subset of ["TESS", "Kepler", "K2"] with at least one observation.
    """
    print(f"Checking data availability for {planet_name}...")
    search_result = lk.search_lightcurve(planet_name)

    if len(search_result) == 0:
        return []

    mission_strings = search_result.table.to_pandas()["mission"].astype(str)
    return [key for key in ["TESS", "Kepler", "K2"] if mission_strings.str.startswith(key).any()]


def get_user_input(df):
    """
    Top-level interactive entry point for this file: gets the user's
    planet choice, then mission choice (only from missions with real data).

    Returns
    -------
    planet_name : str
    mission : str
    stellar_mass : float
    stellar_radius : float
    """
    planet_name, stellar_mass, stellar_radius = choose_planet(df)
    available_missions = get_available_missions(planet_name)

    if not available_missions:
        print(f"\nNo TESS, Kepler, or K2 light curve data found for {planet_name}. "
              "Please choose a different planet.")
        return get_user_input(df)

    if len(available_missions) == 1:
        mission = available_missions[0]
        print(f"\n{planet_name} has data only from {mission} — using {mission} automatically.")
        return planet_name, mission, stellar_mass, stellar_radius

    print(f"\nAvailable missions for {planet_name}:")
    menu = {}
    for i, m in enumerate(available_missions, start=1):
        print(f"{i}. {m}")
        menu[str(i)] = m

    while True:
        choice = input(f"Enter mission number (1-{len(available_missions)}), "
                        f"default is {available_missions[0]}: ").strip()
        if choice == "":
            mission = available_missions[0]
            break
        elif choice in menu:
            mission = menu[choice]
            break
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(available_missions)}.")

    print(f"\nSelected: {planet_name} from {mission}")
    return planet_name, mission, stellar_mass, stellar_radius