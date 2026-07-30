
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
from io import StringIO
from astropy.constants import G, M_sun, R_sun
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. GET STELLAR PARAMETERS
# ============================================================

def get_stellar_parameters(planet_name, TIC_id=None):
    """
    Get stellar mass and radius from NASA Exoplanet Archive.
    Uses requests instead of astroquery (faster and more reliable).

    Parameters
    ----------
    planet_name : str
        Name of the planet
    TIC_id : str, optional
        TIC ID if known 

    Returns
    -------
    dict
        {'st_mass': float, 'st_rad': float, 'hostname': str}
    """
    TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    # If TIC_id is provided, use it; otherwise use planet name
    if TIC_id is not None:
        query = f"""
        SELECT pl_name, hostname, st_mass, st_rad
        FROM ps
        WHERE hostname LIKE '%{TIC_id}%' OR pl_name LIKE '%{TIC_id}%'
        """
    else:
        query = f"""
        SELECT pl_name, hostname, st_mass, st_rad
        FROM ps
        WHERE pl_name = '{planet_name}'
        """

    params = {"query": query.strip(), "format": "csv"}

    try:
        resp = requests.get(TAP_URL, params=params, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        # Clean data - remove NaNs
        df_clean = df.dropna(subset=["st_mass", "st_rad"])

        if df_clean.empty:
            print(f" No stellar parameters found for {planet_name}")
            return None

        # Take mean of all valid values
        st_mass = df_clean['st_mass'].mean()
        st_rad = df_clean['st_rad'].mean()

        print(f"Stellar Mass: {st_mass:.3f} M☉")
        print(f"Stellar Radius: {st_rad:.3f} R☉")

        return {
            'st_mass': st_mass,
            'st_rad': st_rad,
            'hostname': df_clean.iloc[0].get('hostname', None)
        }

    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================
# 2. MEASURE TRANSIT DEPTH
# ============================================================

def measure_transit_depth(phase, flux, period, duration, t0):
    """
    Measure transit depth from folded light curve.

    Parameters
    ----------
    phase : array
        Phase of the folded light curve (days)
    flux : array
        Normalized flux values
    period : float
        Orbital period (days)
    duration : float
        Transit duration (days)
    t0 : float
        Mid-transit time (days)

    Returns
    -------
    tuple
        (baseline_flux, in_transit_flux, depth)
    """
    # Convert duration in days to phase fraction
    duration_phase = duration / period

    # Center is zero after folding
    transit_center = 0.0

    # Mask for in-transit points
    in_transit_mask = np.abs(phase - transit_center) < duration_phase / 2

    # Mask for baseline (outside transit)
    baseline_mask = ~in_transit_mask

    # Calculate flux levels
    baseline_flux = np.mean(flux[baseline_mask])
    in_transit_flux = np.mean(flux[in_transit_mask])
    depth = baseline_flux - in_transit_flux

    return baseline_flux, in_transit_flux, depth


# ============================================================
# 3. CALCULATE PLANET RADIUS
# ============================================================

def calculate_planet_radius(depth, st_rad):
    """
    Calculate planet radius from transit depth and stellar radius.

    Parameters
    ----------
    depth : float
        Transit depth (fraction, e.g., 0.01 = 1%)
    st_rad : float
        Stellar radius in solar radii (R☉)

    Returns
    -------
    dict
        {'rp_rstar': float, 'rp_rsun': float, 'rp_rjup': float, 'rp_rearth': float}
    """
    # Constants
    R_sun_to_Rjup = 9.73
    R_sun_to_Rearth = 109.2

    # Rp/Rstar from depth
    rp_rstar = np.sqrt(depth)

    # Rp in various units
    rp_rsun = rp_rstar * st_rad
    rp_rjup = rp_rsun * R_sun_to_Rjup
    rp_rearth = rp_rsun * R_sun_to_Rearth

    return {
        'rp_rstar': rp_rstar,
        'rp_rsun': rp_rsun,
        'rp_rjup': rp_rjup,
        'rp_rearth': rp_rearth
    }


# ============================================================
# 4. CALCULATE SCALED SEMI-MAJOR AXIS (a/Rstar)
# ============================================================

def calculate_a_over_rstar(period_days, M_star_solar, R_star_solar):
    """
    Compute semi-major axis / stellar radius (a/R*) using Kepler's third law.

    Parameters
    ----------
    period_days : float
        Orbital period in days
    M_star_solar : float
        Stellar mass in solar masses (M☉)
    R_star_solar : float
        Stellar radius in solar radii (R☉)

    Returns
    -------
    float
        a/R* (dimensionless)
    """
    # Convert to SI units
    P = period_days * u.day
    M_star = M_star_solar * u.M_sun
    R_star = R_star_solar * u.R_sun

    # Kepler's third law: a^3 = G * M * P^2 / (4 * pi^2)
    a = ((G * M_star * P**2) / (4.0 * np.pi**2))**(1/3)

    # Return dimensionless a/R*
    return (a / R_star).decompose().value


# ============================================================
# 5. CALCULATE INCLINATION LIMITS
# ============================================================

def calculate_inclination_limits(a_rstar, depth):
    """
    Calculate inclination limits for the planet.

    Parameters
    ----------
    a_rstar : float
        Scaled semi-major axis (a/Rstar)
    depth : float
        Transit depth (fraction)

    Returns
    -------
    dict
        {'min_i_deg': float, 'max_i_deg': float, 'ini_i_deg': float,
         'min_i_rad': float, 'max_i_rad': float, 'ini_i_rad': float,
         'max_size': float, 'min_size': float, 'ini_size': float}
    """
    # Size limits (±20% of depth)
    max_size = np.sqrt(depth + depth * 0.2)
    min_size = np.sqrt(depth - depth * 0.2)
    ini_size = (min_size + max_size) / 2

    # Maximum inclination is 90 degrees
    max_i_deg = 90.0

    # Minimum inclination (transit condition)
    b_max = 1 + max_size
    min_i_rad = np.arccos((1 / a_rstar) * b_max)
    min_i_deg = np.rad2deg(min_i_rad)

    # Initial inclination (average of min and max)
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