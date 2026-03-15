import pandas as pd

from Utils.common import get_session_with_retry, load_session_with_retry, verify_session_data, logger

def get_qualifying_results_fastf1(season, round_no):
    try:
        session = get_session_with_retry(season, round_no, 'Q')
        session = load_session_with_retry(session)
        verify_session_data(session, ['Abbreviation', 'Position'], context="Qualifying")
    except Exception as e:
        logger.warning(f"Could not load qualifying session for round {round_no}: {e}")
        return pd.DataFrame()

    results = session.results
    if results is None or results.empty:
        logger.warning("No qualifying results found.")
        return pd.DataFrame()

    df = results[['Abbreviation', 'Position', 'Status']].copy()
    df.columns = ['driver_id', 'qualifying_pos', 'qualifying_status']
    df['driver_id'] = df['driver_id'].str.lower()
    return df