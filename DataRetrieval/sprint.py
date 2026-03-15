import pandas as pd

from Utils.common import get_session_with_retry, load_session_with_retry, verify_session_data, logger

def get_sprint_results_fastf1(season, round_no):
    try:
        session = get_session_with_retry(season, round_no, 'S')  # Sprint session
        session = load_session_with_retry(session)
        verify_session_data(session, ['Abbreviation', 'Position', 'GridPosition', 'Status'], context="Sprint")
    except Exception as e:
        logger.info(f"No sprint session for {round_no}")
        return pd.DataFrame()

    results = session.results
    if results is None or results.empty:
        logger.info(f"No sprint results for {round_no}")
        return pd.DataFrame()

    results = results.copy()
    results['driver_id'] = results['Abbreviation'].str.lower()

    # Add fastest lap flag
    flaps = session.laps.pick_fastest()
    if flaps is not None and hasattr(flaps, 'Driver') and pd.notna(flaps.Driver):
        results['sprint_fastest_lap'] = results['Abbreviation'] == flaps.Driver
    else:
        results['sprint_fastest_lap'] = False

    df = results[['driver_id', 'Position', 'GridPosition', 'Status', 'sprint_fastest_lap']]
    df.columns = ['driver_id', 'sprint_pos', 'sprint_grid', 'sprint_status', 'sprint_fastest_lap']
    return df