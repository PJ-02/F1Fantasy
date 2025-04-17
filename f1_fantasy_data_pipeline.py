import fastf1.req
import pandas as pd
import fastf1
import logging
import os
import time
import functools
import random
from tqdm import tqdm

from race_calendars import races_2019, races_2020, races_2021, races_2022, races_2023, races_2024, races_2025

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
    2019: races_2019,
    2020: races_2020,
    2021: races_2021,
    2022: races_2022,
    2023: races_2023,
    2024: races_2024,
    2025: races_2025
}



def retry_on_rate_limit(max_retries = 5, base_delay = 60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except fastf1.req.RateLimitExceededError as e:
                    delay = base_delay * (2 ** attempt) + random.randint(1,10)
                    logger.warning(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1})")
                    time.sleep(delay)
            logger.error(f"Max retries exceeded for {func.__name__}")
            return None
        return wrapper
    return decorator



@retry_on_rate_limit()
def get_session_with_retry(season, round_no, session_type):
    return fastf1.get_session(season, round_no, session_type)

@retry_on_rate_limit()
def load_session_with_retry(session, telemetry=True, laps=True):
    session.load(telemetry=telemetry, laps=laps)
    return session




def is_race_cached(season, round_no):
    driver_path = f"cache/pickles/driver_{season}_r{round_no}.pkl"
    constructor_path = f"cache/pickles/constructor_{season}_r{round_no}.pkl"
    return os.path.exists(driver_path) and os.path.exists(constructor_path)


def get_race_results_fastf1(season, round_no):
    try:
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
    results['fastest_lap'] = results['Abbreviation'] == flaps.Driver
    results['fastest_lap_time'] = flaps.LapTime.total_seconds()

    return results[['driver_id', 'driver_name', 'constructor_name', 'grid', 'position', 'status', 'fastest_lap', 'fastest_lap_time']]



def preprocess_driver_data(df):
    df['DNF'] = df['status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)
    df['positions_gained'] = df['grid'] - df['position']
    df['positions_gained'] = df['positions_gained'].fillna(0)

    # Sprint fantasy features
    df['sprint_positions_gained'] = df['sprint_grid'] - df['sprint_pos']
    df['sprint_positions_gained'] = df['sprint_positions_gained'].fillna(0).astype(int)
    df['sprint_dnf'] = df['sprint_status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)
    df['sprint_disqualified'] = df['sprint_status'].apply(lambda x: 1 if 'DSQ' in str(x).upper() else 0)
    df['qualifying_disqualified'] = df['qualifying_pos'].apply(lambda x: 1 if str(x).upper() == 'DSQ' else 0)


    return df


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

    df = results[['Abbreviation', 'Position']].copy()
    df.columns = ['driver_id', 'qualifying_pos']
    df['driver_id'] = df['driver_id'].str.lower()
    return df



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
    results['sprint_fastest_lap'] = results['Abbreviation'] == flaps.Driver

    df = results[['driver_id', 'Position', 'GridPosition', 'Status', 'sprint_fastest_lap']]
    df.columns = ['driver_id', 'sprint_pos', 'sprint_grid', 'sprint_status', 'sprint_fastest_lap']
    return df



def get_team_fastest_pitstops(year, round_no): 
    session = get_session_with_retry(year, round_no, 'R')
    session = load_session_with_retry(session)

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



def get_driver_telemetry(session):
    telemetry_data = []
    pos_data = session.laps[['Driver', 'LapNumber', 'Position']]

    for drv in session.drivers:
        try:
            driver_laps = pos_data[pos_data['Driver'] == drv]
            driver_laps = driver_laps.sort_values('LapNumber')

            overtakes = 0
            prev_pos = None
            for _, row in driver_laps.iterrows():
                current_pos = row['Position']
                if prev_pos and current_pos < prev_pos:
                    overtakes += 1
                prev_pos = current_pos
            
            fastest_lap = session.laps.pick_drivers(drv).pick_fastest()
            if fastest_lap is None or fastest_lap.empty:
                continue
            car_data = fastest_lap.get_car_data().add_distance()
            telemetry_data.append({
                'driver_id':drv.lower(),
                'top_speed': car_data['Speed'].max(),
                'avg_throttle': car_data['Throttle'].mean(),
                'brake_usage': car_data['Brake'].mean(),
                'overtakes': overtakes
            })

        except Exception as e:
            logger.warning(f"Failed to load telemetry for driver {drv}: {e}")
            continue
    return pd.DataFrame(telemetry_data)



def calculate_fantasy_points(df):
    # --- Qualifying Points ---
    quali_points_map = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
    df['qualifying_points'] = df['qualifying_pos'].map(quali_points_map).fillna(0)

    # Penalty for no time or DQ (mock: if quali_pos is NaN)
    df['qualifying_points'] = df['qualifying_pos'].map(quali_points_map).fillna(0)

    # Apply -5 penalty for No Time (Q1) but NOT for DSQ
    df.loc[(df['qualifying_pos'].isna()) & (df['qualifying_disqualified'] == 0), 'qualifying_points'] = -5

    # Apply DSQ penalty
    df.loc[df['qualifying_disqualified'] == 1, 'qualifying_points'] = -5


    # --- Sprint Points ---
    sprint_points_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
    df['sprint_points'] = df['sprint_pos'].map(sprint_points_map).fillna(0)

    # Positions Gained / Lost
    df['sprint_gain_points'] = df['sprint_positions_gained'].fillna(0)

    # Fastest lap in sprint (if column exists, else default False)
    if 'sprint_fastest_lap' in df.columns:
        df['sprint_fastest_lap_bonus'] = df['sprint_fastest_lap'].apply(lambda x: 5 if x else 0)
    else:
        df['sprint_fastest_lap_bonus'] = 0

    # Sprint Overtakes
    df['sprint_overtakes'] = df.get('sprint_overtakes', 0)
    df['sprint_overtake_points'] = df['sprint_overtakes'].fillna(0) * 1

    # Penalty for DNF/Not Classified
    df['sprint_dnf_penalty'] = df['sprint_status'].apply(
        lambda x: -20 if str(x).lower() in ['retired', 'accident', 'not classified', 'nc', 'dnf'] else 0
    )
    # Penalty for Disqualified in Sprint
    df['sprint_dsq_penalty'] = df['sprint_disqualified'] * -20


    # Total sprint score
    df['sprint_total'] = (
        df['sprint_points'] +
        df['sprint_gain_points'] +
        df['sprint_overtake_points'] +
        df['sprint_fastest_lap_bonus'] +
        df['sprint_dnf_penalty'] +
        df['sprint_dsq_penalty']
    )


    # --- Race Points ---
    race_points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df['race_points'] = df['position'].map(race_points_map).fillna(0)

    df['overtakes'] = df.get('overtakes', 0)

    # Positions Gained / Lost
    df['race_gain_points'] = df['positions_gained'].fillna(0)

    # Fastest Lap
    df['race_fastest_lap_bonus'] = df['fastest_lap'].apply(lambda x: 10 if x else 0)

    # Driver of the Day (for now, still mock it)
    df['dotd_bonus'] = 0
    df.loc[df['driver_name'] == "Max Verstappen", 'dotd_bonus'] = 10

    # Overtakes
    df['overtakes'] = df.get('overtakes', 0)
    df['race_overtake_points'] = df['overtakes'] * 1

    # DNF / NC / DSQ
    df['race_dnf_penalty'] = df['status'].apply(
        lambda x: -20 if str(x).lower() in ['retired', 'accident', 'not classified', 'nc', 'dnf'] else 0
    )


    # --- Total Fantasy Score ---
    df['fantasy_points_total'] = (
    df['qualifying_points'] +
    df['sprint_total'] +
    df['race_points'] +
    df['race_gain_points'] +
    df['race_overtake_points'] +
    df['race_fastest_lap_bonus'] +
    df['dotd_bonus'] +
    df['race_dnf_penalty']
    )


    return df



def calculate_constructor_points(driver_df, pitstop_df):
    # Detect fastest pitstop of the race
    fastest_pitstop = pitstop_df['fastest_pitstop_time'].min()
    if pitstop_df.empty or 'fastest_pitstop_time' not in pitstop_df.columns:
        fastest_team = None
    else:
        fastest_pitstop = pitstop_df['fastest_pitstop_time'].min()
        fastest_team_rows = pitstop_df.loc[pitstop_df['fastest_pitstop_time'] == fastest_pitstop, 'constructor_name']
        fastest_team = fastest_team_rows.iloc[0] if not fastest_team_rows.empty else None


    # World record benchmark
    WORLD_RECORD_TIME = 1.80

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

        entry['fastest_pitstop_bonus'] = 0  # Always initialize first

        if constructor == fastest_team:
            entry['fastest_pitstop_bonus'] += 5
            if p_time is not None and p_time < WORLD_RECORD_TIME:
                entry['fastest_pitstop_bonus'] += 15



        # Constructor penalties (Sprint DSQ + Race DSQ)
        sprint_dsq_count = group['sprint_disqualified'].sum()
        race_dsq_count = group['status'].apply(lambda x: 'DSQ' in str(x).upper()).sum()

        total_dsq = sprint_dsq_count + race_dsq_count
        entry['disqualification_penalty'] = total_dsq * -10


        # Final Constructor Fantasy Points
        entry['constructor_fantasy_points'] = (
        entry['driver_points_total'] +
        entry['qualifying_bonus'] +
        entry['pitstop_points'] +
        entry['fastest_pitstop_bonus'] +
        entry['disqualification_penalty']
        )


        constructor_rows.append(entry)

    return pd.DataFrame(constructor_rows)



def process_race(season, round_no):
    try:
        event = fastf1.get_event(season, round_no)
        gp_name = event['EventName']
        logger.info(f"\n===== Starting Processing for Round {round_no} - {gp_name} =====")

        race_df = get_race_results_fastf1(season, round_no)
        if race_df.empty:
            logger.warning(f"Skipping Round {round_no} - {gp_name}: No race data found.")
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

        #Telemetry data
        session = get_session_with_retry(season, round_no, 'R')
        session = load_session_with_retry(session)
        telemetry_df = get_driver_telemetry(session)
        final_data = final_data.merge(telemetry_df, on='driver_id', how='left')

        # Weather data
        weather_df = session.weather_data
        weather_df['session_time'] = weather_df['Time'].dt.total_seconds()
        avg_weather = {
            'track_temp_avg': weather_df['TrackTemp'].mean(),
            'air_temp_avg': weather_df['AirTemp'].mean(),
            'humidity_avg': weather_df['Humidity'].mean(),
            'pressure_avg': weather_df['Pressure'].mean(),
            'rainfall_avg': weather_df['Rainfall'].mean(),
            'wind_speed_avg': weather_df['WindSpeed'].mean(),
            'wind_direction_avg': weather_df['WindDirection'].mean()
        }

        # Pitstop data
        pitstop_df = get_team_fastest_pitstops(season, round_no)
        constructor_data = calculate_constructor_points(final_data, pitstop_df)

        # Add metadata
        final_data['season'] = season
        final_data['round'] = round_no
        final_data['event_name'] = gp_name

        constructor_data['season'] = season
        constructor_data['round'] = round_no
        constructor_data['event_name'] = gp_name

        for k, v in avg_weather.items():
            final_data[k] = v
            constructor_data[k] = v

        # Saving data to cache
        os.makedirs("cache/pickles", exist_ok=True)
        final_data.to_pickle(f"cache/pickles/driver_{season}_r{round_no}.pkl")
        constructor_data.to_pickle(f"cache/pickles/constructor_{season}_r{round_no}.pkl")

        logger.info(f"Processed Round {round_no} - {gp_name}")
        return final_data, constructor_data

    except Exception as e:
        logger.exception(f"Exception in Round {round_no} - {gp_name if 'gp_name' in locals() else 'Unknown GP'}")
        return None, None



def run_full_season(season, races):
    all_driver_data = []
    all_constructor_data = []

    for round_no in tqdm(races, desc=f"Processing {season} season"):
        if is_race_cached(season, round_no):
            logger.info(f"Skipping round {round_no} - already cached.")
            continue
        driver_df, constructor_df = process_race(season, round_no)
        if driver_df is not None:
            all_driver_data.append(driver_df)
        if constructor_df is not None:
            all_constructor_data.append(constructor_df)
        time.sleep(60)  # delay of 60 seconds between race API calls

    if all_driver_data:
        os.makedirs("GeneratedSpreadsheets", exist_ok=True)
        full_driver_df = pd.concat(all_driver_data, ignore_index=True)

        expected_rounds = set(races)
        processed_rounds = set(full_driver_df['round'].unique())
        missing_rounds = expected_rounds - processed_rounds
        if missing_rounds:
            logger.warning(f"Missing rounds in season {season}: {sorted(missing_rounds)}")

        full_driver_df.to_excel(f"GeneratedSpreadsheets/{season}_fantasy_driver_data.xlsx", index=False)
        logger.info(f"Saved full season driver data: {season}_fantasy_driver_data.xlsx")

    if all_constructor_data:
        os.makedirs("GeneratedSpreadsheets", exist_ok=True)
        full_constructor_df = pd.concat(all_constructor_data, ignore_index=True)
        full_constructor_df.to_excel(f"GeneratedSpreadsheets/{season}_fantasy_constructor_data.xlsx", index=False)
        logger.info(f"Saved full season constructor data: {season}_fantasy_constructor_data.xlsx")



def combine_all_seasons(output_folder="./GeneratedSpreadsheets", file_prefix="fantasy_driver_data"):
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



def verify_session_data(session, expected_columns=None, context=""):
    if session.results is None or session.results.empty:
        logger.warning(f"[{context}] Empty or missing results.")
        return False

    num_drivers = len(session.results)
    if num_drivers < 18:
        logger.warning(f"[{context}] Unexpected number of drivers: {num_drivers}")

    if expected_columns:
        missing = [col for col in expected_columns if col not in session.results or session.results[col].isna().all()]
        if missing:
            logger.warning(f"[{context}] Missing or all-null columns: {missing}")

    if context == "Race":
        if 'PitInTime' not in session.laps.columns or session.laps['PitInTime'].isna().all():
            logger.warning(f"[{context}] Pit stop data is missing or empty.")
    
    logger.info(f"[{context}] Data verification complete.")
    return True    



def main():
    for season, races in SEASON_RACE_MAP.items():
        logger.info(f"Starting season {season}...")
        run_full_season(season, races)
    
    combine_all_seasons("GeneratedSpreadsheets",file_prefix="fantasy_driver_data")
    combine_all_seasons("GeneratedSpreadsheets",file_prefix="fantasy_constructor_data")



# Run the pipeline
main()