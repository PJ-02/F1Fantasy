"""
Fetches practice session (FP1 / FP2 / FP3) lap pace data for each driver.

Returns relative pace — gap to the session's fastest lap, expressed as a
percentage. Lower is better; 0.0 = fastest driver in that session.

Only 'quick laps' (clean, representative laps) are used, so pit-in/out and
outlier laps are excluded automatically via FastF1's pick_quicklaps().

Sprint weekends have no FP2 or FP3 — callers should handle an empty
DataFrame gracefully when those sessions don't exist.
"""

import pandas as pd
from Utils.common import get_session_with_retry, load_session_with_retry, logger


def get_practice_pace(season: int, round_no: int, session_type: str) -> pd.DataFrame:
    """
    Fetch pace data from a single practice session.

    Parameters
    ----------
    season       : e.g. 2024
    round_no     : e.g. 3
    session_type : 'FP1', 'FP2', or 'FP3'

    Returns
    -------
    DataFrame with columns:
        driver_id            — lowercased 3-letter abbreviation
        {prefix}_gap_pct     — % gap to session best (0 = fastest; e.g. 'fp2_gap_pct')
        {prefix}_laps        — number of quick laps completed
    Empty DataFrame on failure (session doesn't exist, no data, API error).
    """
    prefix = session_type.lower()   # 'fp1', 'fp2', 'fp3'

    try:
        session = get_session_with_retry(season, round_no, session_type)
        session = load_session_with_retry(session, telemetry=False, laps=True)
    except Exception as e:
        logger.warning(f"Could not load {session_type} for {season} R{round_no}: {e}")
        return pd.DataFrame()

    try:
        laps = session.laps.pick_quicklaps()
        if laps is None or laps.empty:
            logger.warning(f"{session_type} has no quick laps for {season} R{round_no}")
            return pd.DataFrame()

        # Best lap time per driver in seconds
        best_s = laps.groupby('Driver')['LapTime'].min().dt.total_seconds()
        if best_s.empty or best_s.isna().all():
            return pd.DataFrame()

        session_best = best_s.min()
        gap_pct      = (best_s - session_best) / session_best * 100
        lap_count    = laps.groupby('Driver').size().reindex(best_s.index).fillna(0).astype(int)

        df = pd.DataFrame({
            'driver_id':        [d.lower() for d in best_s.index],
            f'{prefix}_gap_pct': gap_pct.values,
            f'{prefix}_laps':    lap_count.values,
        })
        logger.info(f"{session_type} {season} R{round_no}: {len(df)} drivers, "
                    f"best gap range 0–{gap_pct.max():.2f}%")
        return df

    except Exception as e:
        logger.warning(f"Practice pace computation failed ({session_type} {season} R{round_no}): {e}")
        return pd.DataFrame()
