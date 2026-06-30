import lightkurve as lk

def lc_download(planet_name, mission, author="SPOC"):
    """
    Downloads and cleans a light curve for a given planet.

    Parameters
    ----------
    planet_name : str
        Name of the planet (written exactly like tje catalogue)
    mission : str
        Mission to search (TESS, Kepler, K2)
    author : str
        Pipeline author (default SPOC for PDCSAP flux)

    Returns
    -------
    lc_clean : LightCurve
        Cleaned, normalized, flattened light curve
    """
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
    lc_clean = lc.normalize().remove_nans().remove_outliers(sigma=5).flatten(window_length=401)
    
    print(f"Download complete. {len(lc_clean)} data points.")
    return lc_clean