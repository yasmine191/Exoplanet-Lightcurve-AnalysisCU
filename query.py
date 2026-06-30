from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd


def query_hot_jupiters():
    """
    Query the NASA Exoplanet Archive for confirmed Hot Jupiters
    with near-zero eccentricity (e <= 0.05) that can be approximated to zero.

    Returns
    -------
    df : pandas DataFrame
        Table of filtered Hot Jupiter planets.
    """
    print("Querying NASA Exoplanet Archive...")

    result = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="pl_name, hostname, pl_massj, pl_radj, \
                pl_orbper, pl_orbeccen, tran_flag, \
                disc_facility, disc_year, st_mass, st_rad",
        where="pl_massj > 0.25 AND pl_massj < 13 \
               AND pl_orbeccen <= 0.05 \
               AND tran_flag = 1 \
               AND pl_orbper <= 10"
    )

    df = result.to_pandas()

    # Drop rows with missing key values
    df = df.dropna(subset=["pl_name", "pl_orbper", "pl_massj", 
                            "pl_radj", "st_mass", "st_rad"])

    print(f"Found {len(df)} Hot Jupiters with e <= 0.05")
    return df


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
            print("Planet not found. Please try again.")


def get_user_input(df):
    """
    Gets planet name and mission choice from the user.

    Returns
    -------
    planet_name : str
    mission : str
    """
    # Planet choice
    planet_name = choose_planet(df)

    # Mission choice
    print("\nAvailable missions:")
    print("1. TESS")
    print("2. Kepler")
    print("3. K2")

    missions = {"1": "TESS", "2": "Kepler", "3": "K2"}

    while True:
        choice = input("Enter mission number (1/2/3), default is TESS: ").strip()
        if choice == "":
            mission = "TESS"
            break
        elif choice in missions:
            mission = missions[choice]
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print(f"\nSelected: {planet_name} from {mission}")
    return planet_name, mission