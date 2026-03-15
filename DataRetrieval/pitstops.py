import pandas as pd

from Utils.common import logger

def get_team_fastest_pitstops(session):
    """
    Extract fastest pit lane time per constructor from an already-loaded race session.

    NOTE: FastF1 exposes pit LANE traversal time (entry to exit, typically 18–35s),
    not the stationary stop time (2–4s) used in the official F1 Fantasy scoring thresholds.
    As a result, pitstop bonus points in constructorFantasyPoints.py will always be 0
    with this data source. The time values are still stored in the pickle and are useful
    as relative ML features (team ranking by pit speed).
    """
    try:
        # PitInTime and PitOutTime are on separate consecutive rows per driver:
        # the pit-in lap has PitInTime set; the following lap has PitOutTime set.
        # Pair them by matching each PitInTime row with the next row's PitOutTime for the same driver.
        laps = session.laps[['Driver', 'Team', 'LapNumber', 'PitInTime', 'PitOutTime']].copy()

        pit_in  = laps[laps['PitInTime'].notna()][['Driver', 'Team', 'LapNumber', 'PitInTime']].copy()
        pit_out = laps[laps['PitOutTime'].notna()][['Driver', 'LapNumber', 'PitOutTime']].copy()

        # Match by driver: each pit_in lap N pairs with the pit_out whose LapNumber > N (nearest)
        stops = []
        for _, row in pit_in.iterrows():
            drv = row['Driver']
            lap_n = row['LapNumber']
            out_rows = pit_out[(pit_out['Driver'] == drv) & (pit_out['LapNumber'] > lap_n)]
            if out_rows.empty:
                continue
            out_row = out_rows.iloc[0]
            duration = (out_row['PitOutTime'] - row['PitInTime']).total_seconds()
            if 1.0 < duration < 60.0:  # sanity filter: between 1s and 60s
                stops.append({'Driver': drv, 'Team': row['Team'], 'PitStopDuration': duration})

        if not stops:
            logger.warning("No valid pitstop data found in session.")
            return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

        pitstops = pd.DataFrame(stops)
        team_fastest = pitstops.groupby('Team')['PitStopDuration'].min().reset_index()
        return team_fastest.rename(columns={'Team': 'constructor_name', 'PitStopDuration': 'fastest_pitstop_time'})
    except Exception as e:
        logger.warning(f"Failed to extract pitstop data: {e}")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])