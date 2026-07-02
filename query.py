import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import time
import requests
from io import StringIO

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def query_hot_jupiters(max_retries=4, initial_delay=3):
    """
    Query the NASA Exoplanet Archive for confirmed Hot Jupiters
    with near-zero eccentricity (e <= 0.05), via direct TAP/CSV request
    (bypasses astroquery's TAP schema-validation call, which is what
    was actually throwing the VOTable parse error).
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
    """.strip()

    params = {
        "query": query,
        "format": "csv",
    }

    print("Querying NASA Exoplanet Archive (direct TAP/CSV)...")
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(TAP_URL, params=params, timeout=30)
            resp.raise_for_status()
            # A failed TAP query often still returns HTTP 200 but with
            # an error message/VOTable body instead of CSV — catch that.
            if not resp.text.lstrip().startswith("pl_name"):
                raise ValueError(f"Unexpected response body: {resp.text[:200]}")
            df = pd.read_csv(StringIO(resp.text))
            break
        except Exception as err:
            last_error = err
            is_last_attempt = attempt == max_retries
            print(f"Attempt {attempt}/{max_retries} failed: {err}")
            if is_last_attempt:
                raise RuntimeError(
                    "NASA Exoplanet Archive query failed after "
                    f"{max_retries} attempts. The archive's TAP service may "
                    "be temporarily down (502/503 errors are common under load). "
                    "Try again in a few minutes, or check "
                    "https://exoplanetarchive.ipac.caltech.edu/ for status."
                ) from last_error
            print(f"Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= 2

    df = df.dropna(subset=["pl_name", "pl_orbper", "pl_massj", "pl_radj", "st_mass", "st_rad"])
    print(f"Found {len(df)} Hot Jupiters with e <= 0.05")
    return df

#==================================================================================================================

def choose_planet(df):
    """
    Display the planet list and let the user choose one.

    Returns
    -------
    planet_name : str
        The name of the chosen planet.
    """
    print("\nAvailable Hot Jupiters:")
    print(df[["pl_name", "hostname", "pl_massj", 
              "pl_orbper", "pl_orbeccen"]].to_string(index=False))

    while True:
        choice = input("\nEnter the planet name exactly as shown above: ").strip()
        if choice in df["pl_name"].values:
            print(f"You selected: {choice}")
            return choice
        else:
            print("Planet not found. Please try again and make sure the name is exactly the same!.")


DEFAULT_AUTHORS = {"TESS": "SPOC", "Kepler": "Kepler", "K2": "K2"}

def get_available_missions(planet_name):
    """
    Single lightkurve search (no mission filter) to find which missions
    actually have data for this planet, so the user is only ever offered
    missions that exist — avoids a later 'No data found' error after
    they've already answered several prompts.

    Returns
    -------
    available : list of str
        Missions with at least one observation, e.g. ["TESS", "K2"].
    """
    print(f"Checking data availability for {planet_name}...")
    search_result = lk.search_lightcurve(planet_name)  # no author filter here

    if len(search_result) == 0:
        return []

    mission_strings = search_result.table.to_pandas()["mission"].astype(str)

    available = [
        key for key in ["TESS", "Kepler", "K2"]
        if mission_strings.str.startswith(key).any()
    ]
    return available

#==================================================================================================================

def get_user_input(df):
    """
    Gets planet name and mission choice from the user for downloading the data.
    Only offers missions that actually have data for the chosen planet.

    Returns
    -------
    planet_name : str
    mission : str
    """
    planet_name = choose_planet(df)

    available_missions = get_available_missions(planet_name)

    if not available_missions:
        print(f"\nNo TESS, Kepler, or K2 light curve data found for {planet_name}. "
              "Please choose a different planet.")
        return get_user_input(df)

    if len(available_missions) == 1:
        mission = available_missions[0]
        print(f"\n{planet_name} has data only from {mission} — using {mission} automatically.")
        return planet_name, mission

    print(f"\nAvailable missions for {planet_name}:")
    menu = {}
    for i, m in enumerate(available_missions, start=1):
        print(f"{i}. {m}")
        menu[str(i)] = m

    while True:
        choice = input(
            f"Enter mission number (1-{len(available_missions)}), default is {available_missions[0]}: "
        ).strip()
        if choice == "":
            mission = available_missions[0]
            break
        elif choice in menu:
            mission = menu[choice]
            break
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(available_missions)}.")

    print(f"\nSelected: {planet_name} from {mission}")
    return planet_name, mission