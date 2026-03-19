import pandas as pd

from Utils.common import logger
from DataRetrieval.pitstops_openf1 import get_openf1_pitstops


def get_team_fastest_pitstops(session, season: int = None, round_no: int = None):
    """
    Return fastest pit stop time per constructor.

    Strategy (in order):
    1. OpenF1 API — provides true stationary stop_duration (~2–4s).
       Available from Austin 2024 (2024 R19) onwards.  When OpenF1 data is
       usable, pitstop bonus points in constructorFantasyPoints.py are scored
       correctly against the 2s/2.2s/2.5s/3s thresholds.
    2. FastF1 fallback — returns pit LANE traversal time (~18–35s).
       With this data pitstop bonuses will always be 0, but the relative
       team ranking is still useful as an ML feature.
    """
    # Try OpenF1 first when season/round are provided
    if season is not None and round_no is not None:
        openf1_df = get_openf1_pitstops(season, round_no)
        if not openf1_df.empty:
            logger.info(f"Using OpenF1 stationary pit times for {season} R{round_no}")
            return openf1_df
        logger.debug(f"OpenF1 unavailable for {season} R{round_no} — falling back to FastF1 traversal times")

    # FastF1 fallback
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