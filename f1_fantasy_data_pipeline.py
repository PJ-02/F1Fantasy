import pandas as pd
import fastf1
import logging
import os
import time

from race_calendars import races_2019, races_2020, races_2021, races_2022, races_2023, races_2024, test_races

fastf1.Cache.enable_cache('cache/f1_cache')  # this creates cache for speed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


SEASON_RACE_MAP = {
    # 2019: races_2019,
    # 2020: races_2020,
    # 2021: races_2021,
    # 2022: races_2022,
    # 2023: races_2023,
    # 2024: races_2024,
    2024:test_races,
}


def get_race_results_fastf1(season, round_no):
    try:
        session = fastf1.get_session(season, round_no, 'R')
        session.load()
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
    results['fastest_lap'] = results['Abbreviation'] == flaps.Driver
    results['fastest_lap_time'] = flaps.LapTime.total_seconds()

    return results[['driver_id', 'driver_name', 'constructor_name', 'grid', 'position', 'status', 'fastest_lap', 'fastest_lap_time']]


def preprocess_driver_data(df):
    df['DNF'] = df['status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)
    df['positions_gained'] = df['grid'] - df['position']
    df['positions_gained'] = df['positions_gained'].fillna(0)

    # Sprint fantasy features
    df['sprint_positions_gained'] = df['sprint_grid'] - df['sprint_pos']
    df['sprint_positions_gained'] = df['sprint_positions_gained'].fillna(0)
    df['sprint_dnf'] = df['sprint_status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)

    return df


def get_qualifying_results_fastf1(season, round_no):
    try:
        session = fastf1.get_session(season, round_no, 'Q')
        session.load()
    except Exception as e:
        logger.warning(f"Could not load qualifying session for round {round_no}: {e}")
        return pd.DataFrame()

    results = session.results
    if results is None or results.empty:
        logger.warning("No qualifying results found.")
        return pd.DataFrame()

    df = results[['Abbreviation', 'Position']].copy()
    df.columns = ['driver_id', 'qualifying_pos']
    df['driver_id'] = df['driver_id'].str.lower()
    return df


def get_sprint_results_fastf1(season, round_no):
    try:
        session = fastf1.get_session(season, round_no, 'S')  # Sprint session
        session.load()
    except Exception as e:
        logger.info(f"No sprint session for {round_no}")
        return pd.DataFrame()

    results = session.results
    if results is None or results.empty:
        logger.info(f"No sprint results for {round_no}")
        return pd.DataFrame()

    results = results.copy()
    results['driver_id'] = results['Abbreviation'].str.lower()
    df = results[['driver_id', 'Position', 'GridPosition', 'Status']]
    df.columns = ['driver_id', 'sprint_pos', 'sprint_grid', 'sprint_status']
    return df


def get_team_fastest_pitstops(year, round_no): 
    session = fastf1.get_session(year, round_no, 'R')
    session.load()

    pitstops = session.laps[['Driver', 'PitInTime', 'PitOutTime']].dropna()

    # Convert to timestamps and check validity
    pitstops['PitInTime'] = pd.to_datetime(pitstops['PitInTime'], errors='coerce')
    pitstops['PitOutTime'] = pd.to_datetime(pitstops['PitOutTime'], errors='coerce')

    # Filter out invalid rows
    mask = pitstops['PitInTime'].notna() & pitstops['PitOutTime'].notna()
    pitstops = pitstops[mask]

    pitstops['PitStopDuration'] = (pitstops['PitOutTime'] - pitstops['PitInTime']).dt.total_seconds()

    driver_teams = session.laps[['Driver', 'Team']].drop_duplicates()
    pitstops = pitstops.merge(driver_teams, on='Driver', how='left')

    team_fastest = pitstops.groupby('Team')['PitStopDuration'].min().reset_index()
    return team_fastest.rename(columns={'Team': 'constructor_name', 'PitStopDuration': 'fastest_pitstop_time'})


def calculate_fantasy_points(df):
    # --- Qualifying Points ---
    quali_points_map = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
    df['qualifying_points'] = df['qualifying_pos'].map(quali_points_map).fillna(0)

    # Penalty for no time or DQ (mock: if quali_pos is NaN)
    df.loc[df['qualifying_pos'].isna(), 'qualifying_points'] = -5

    # --- Sprint Points ---
    sprint_points_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
    df['sprint_points'] = df['sprint_pos'].map(sprint_points_map).fillna(0)

    df['sprint_gain_points'] = df['sprint_positions_gained']  # 1 pt per position gained/lost
    df['sprint_dnf_penalty'] = df['sprint_dnf'] * -20

    # --- Race Points ---
    race_points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df['race_points'] = df['position'].map(race_points_map).fillna(0)

    df['race_gain_points'] = df['positions_gained']  # 1 pt per position gained/lost
    df['fastest_lap_bonus'] = df['fastest_lap'].apply(lambda x: 10 if x else 0)

    # Mocking DOTD for now (you can set one manually or with real data later)
    df['dotd_bonus'] = 0
    df.loc[df['driver_name'] == "Max Verstappen", 'dotd_bonus'] = 10

    df['race_dnf_penalty'] = df['DNF'] * -20

    # --- Total Fantasy Score ---
    df['fantasy_points_total'] = (
        df['qualifying_points'] +
        df['sprint_points'] +
        df['sprint_gain_points'] +
        df['sprint_dnf_penalty'] +
        df['race_points'] +
        df['race_gain_points'] +
        df['fastest_lap_bonus'] +
        df['dotd_bonus'] +
        df['race_dnf_penalty']
    )

    return df


def calculate_constructor_points(driver_df, pitstop_df):
    # Group by constructor
    grouped = driver_df.groupby('constructor_name')

    constructor_rows = []

    for constructor, group in grouped:
        entry = {'constructor_name': constructor}

        # Combine driver-level fantasy points
        entry['driver_points_total'] = group['fantasy_points_total'].sum()

        # Qualifying Bonuses
        q2_count = group['qualifying_pos'].apply(lambda x: x <= 15 if pd.notna(x) else False).sum()
        q3_count = group['qualifying_pos'].apply(lambda x: x <= 10 if pd.notna(x) else False).sum()

        if q2_count == 0:
            qual_bonus = -1
        elif q2_count == 1:
            qual_bonus = 1
        else:
            qual_bonus = 3

        if q3_count == 1:
            qual_bonus += 5
        elif q3_count == 2:
            qual_bonus += 10

        entry['qualifying_bonus'] = qual_bonus

        # Pitstop time from FastF1
        pitstop_points = 0
        if pitstop_df is not None and constructor in pitstop_df['constructor_name'].values:
            p_time = pitstop_df.loc[pitstop_df['constructor_name'] == constructor, 'fastest_pitstop_time'].values[0]
            entry['fastest_pitstop_time'] = p_time

            if p_time < 2.0:
                pitstop_points = 20
            elif 2.0 <= p_time < 2.2:
                pitstop_points = 10
            elif 2.2 <= p_time < 2.5:
                pitstop_points = 5
            elif 2.5 <= p_time < 3.0:
                pitstop_points = 2
            else:
                pitstop_points = 0
        else:
            entry['fastest_pitstop_time'] = None
        entry['pitstop_points'] = pitstop_points

        # Constructor penalties (e.g., DQs)
        dq_count = group['status'].apply(lambda x: 'DSQ' in x).sum()
        entry['dq_penalty'] = dq_count * -10

        # Final Constructor Fantasy Points
        entry['fastest_pitstop_bonus'] = 0
        entry['world_record_bonus'] = 0

        entry['constructor_fantasy_points'] = (
            entry['driver_points_total'] +
            entry['qualifying_bonus'] +
            entry['pitstop_points'] +
            entry['fastest_pitstop_bonus'] +
            entry['world_record_bonus'] +
            entry['dq_penalty']
        )

        constructor_rows.append(entry)

    return pd.DataFrame(constructor_rows)


def process_race(season, round_no):
    try:
        race_df = get_race_results_fastf1(season, round_no)
        if race_df.empty:
            logger.warning(f"Skipping Round {round_no}: No race data found.")
            return None, None

        quali_df = get_qualifying_results_fastf1(season, round_no)
        sprint_df = get_sprint_results_fastf1(season, round_no)
        merged_df = race_df.merge(quali_df, on="driver_id", how="left")

        if not sprint_df.empty and 'driver_id' in sprint_df.columns:
            merged_df = merged_df.merge(sprint_df, on="driver_id", how="left")
        else:
            merged_df['sprint_pos'] = None
            merged_df['sprint_status'] = ""
            merged_df['sprint_grid'] = None

        clean_data = preprocess_driver_data(merged_df)
        final_data = calculate_fantasy_points(clean_data)

        # Pitstop data
        pitstop_df = get_team_fastest_pitstops(season, round_no)
        constructor_data = calculate_constructor_points(final_data, pitstop_df)

        # Add metadata
        final_data['season'] = season
        final_data['round'] = round_no
        constructor_data['season'] = season
        constructor_data['round'] = round_no

        # Saving data to cache
        os.makedirs("cache/pickles", exist_ok=True)
        final_data.to_pickle(f"cache/pickles/driver_{season}_r{round_no}.pkl")
        constructor_data.to_pickle(f"cache/pickles/constructor_{season}_r{round_no}.pkl")

        logger.info(f"Processed Round {round_no}")
        return final_data, constructor_data

    except Exception as e:
        logger.warning(f"Error in Round {round_no}: {e}")
        return None, None


def run_full_season(season, races):
    all_driver_data = []
    all_constructor_data = []

    for round_no in races:
        driver_df, constructor_df = process_race(season, round_no)
        if driver_df is not None:
            all_driver_data.append(driver_df)
        if constructor_df is not None:
            all_constructor_data.append(constructor_df)
        time.sleep(20)  # delay of 20 seconds between race API calls

    if all_driver_data:
        full_driver_df = pd.concat(all_driver_data, ignore_index=True)
        full_driver_df.to_excel(f"{season}_fantasy_driver_data.xlsx", index=False)
        logger.info(f"Saved full season driver data: {season}_fantasy_driver_data.xlsx")

    if all_constructor_data:
        full_constructor_df = pd.concat(all_constructor_data, ignore_index=True)
        full_constructor_df.to_excel(f"GeneratedSpreadsheets/{season}_fantasy_constructor_data.xlsx", index=False)
        logger.info(f"Saved full season constructor data: {season}_fantasy_constructor_data.xlsx")


def combine_all_seasons(output_folder=".", file_prefix="fantasy_driver_data"):
    all_dfs = []
    for season in SEASON_RACE_MAP:
        file = f"{output_folder}/{season}_{file_prefix}.xlsx"
        if os.path.exists(file):
            df = pd.read_excel(file)
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        output_file = f"{output_folder}/all_seasons_{file_prefix}.xlsx"
        combined.to_excel(output_file, index=False)
        logger.info(f" Combined file saved: {output_file}")


def main():
    for season, races in SEASON_RACE_MAP.items():
        logger.info(f"Starting season {season}...")
        run_full_season(season, races)
    
    combine_all_seasons("GeneratedSpreadsheets",file_prefix="fantasy_driver_data")
    combine_all_seasons("GeneratedSpreadsheets",file_prefix="fantasy_constructor_data")



# Run the pipeline
main()