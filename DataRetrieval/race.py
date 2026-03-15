import pandas as pd

from Utils.common import get_session_with_retry, load_session_with_retry, verify_session_data, logger

def get_race_results_fastf1(season, round_no, session=None):
    """Extract race results. If session is provided (already loaded), it is used directly."""
    try:
        if session is None:
            session = get_session_with_retry(season, round_no, 'R')
            session = load_session_with_retry(session)
        verify_session_data(session, ['FullName', 'Abbreviation', 'Position', 'GridPosition', 'TeamName'], context="Race")
    except Exception as e:
        logger.warning(f"Could not load race session for round {round_no}: {e}")
        return pd.DataFrame()

    results = session.results
    if results is None or results.empty:
        logger.warning("No race results found.")
        return pd.DataFrame()

    results = results.copy()
    results['driver_name'] = results['FullName']
    results['driver_id'] = results['Abbreviation'].str.lower()  # Normalize
    results['position'] = results['Position']
    results['grid'] = results['GridPosition']
    results['status'] = results['Status']
    results['constructor_name'] = results['TeamName']
    
    # Fastest lap info
    flaps = session.laps.pick_fastest()
    if flaps is not None and hasattr(flaps, 'Driver') and pd.notna(flaps.Driver):
        results['fastest_lap'] = results['Abbreviation'] == flaps.Driver
        results['fastest_lap_time'] = flaps.LapTime.total_seconds() if pd.notna(flaps.LapTime) else None
    else:
        results['fastest_lap'] = False
        results['fastest_lap_time'] = None

    return results[['driver_id', 'driver_name', 'constructor_name', 'grid', 'position', 'status', 'fastest_lap', 'fastest_lap_time']]