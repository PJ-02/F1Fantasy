import pandas as pd
import fastf1
import logging
import os
import time

from tqdm import tqdm

from raceCalendars import races_2019, races_2020, races_2021, races_2022, races_2023, races_2024, races_2025, races_2026
from DataRetrieval.race import get_race_results_fastf1
from DataRetrieval.qualifying import get_qualifying_results_fastf1
from DataRetrieval.sprint import get_sprint_results_fastf1
from DataRetrieval.pitstops import get_team_fastest_pitstops
from DataRetrieval.telemetry import get_driver_telemetry

from Scoring.preprocess import preprocess_driver_data
from Scoring.driverFantasyPoints import calculate_fantasy_points
from Scoring.constructorFantasyPoints import calculate_constructor_points

from Utils.common import get_session_with_retry, load_session_with_retry



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
    2025: races_2025,
    2026: races_2026
}



def is_race_cached(season, round_no):
    driver_path = f"cache/pickles/driver_{season}_r{round_no}.pkl"
    constructor_path = f"cache/pickles/constructor_{season}_r{round_no}.pkl"
    return os.path.exists(driver_path) and os.path.exists(constructor_path)



def process_race(season, round_no):
    try:
        event = fastf1.get_event(season, round_no)
        gp_name = event['EventName']
        logger.info(f"\n===== Starting Processing for Round {round_no} - {gp_name} =====")

        # Load the race session once — reused for results, telemetry, pitstops, and weather
        session = get_session_with_retry(season, round_no, 'R')
        session = load_session_with_retry(session)

        race_df = get_race_results_fastf1(season, round_no, session=session)
        if race_df.empty:
            logger.warning(f"Skipping Round {round_no} - {gp_name}: No race data found.")
            return None, None

        quali_df = get_qualifying_results_fastf1(season, round_no)

        sprint_df = get_sprint_results_fastf1(season, round_no)
        had_sprint = not sprint_df.empty and 'driver_id' in sprint_df.columns

        merged_df = race_df.merge(quali_df, on="driver_id", how="left")

        if not sprint_df.empty and 'driver_id' in sprint_df.columns:
            merged_df = merged_df.merge(sprint_df, on="driver_id", how="left")
        else:
            merged_df['sprint_pos'] = None
            merged_df['sprint_status'] = ""
            merged_df['sprint_grid'] = None

        # Merge telemetry BEFORE scoring so overtake counts feed into fantasy points
        telemetry_df = get_driver_telemetry(session)
        if not telemetry_df.empty and 'driver_id' in telemetry_df.columns:
            merged_df = merged_df.merge(telemetry_df, on='driver_id', how='left')
        else:
            logger.warning(f"No telemetry data for Round {round_no} — overtakes will be 0.")

        clean_data = preprocess_driver_data(merged_df)
        final_data = calculate_fantasy_points(clean_data)

        # Weather data
        try:
            weather_df = session.weather_data
            avg_weather = {
                'track_temp_avg': weather_df['TrackTemp'].mean(),
                'air_temp_avg': weather_df['AirTemp'].mean(),
                'humidity_avg': weather_df['Humidity'].mean(),
                'pressure_avg': weather_df['Pressure'].mean(),
                'rainfall_avg': weather_df['Rainfall'].mean(),
                'wind_speed_avg': weather_df['WindSpeed'].mean(),
                'wind_direction_avg': weather_df['WindDirection'].mean()
            }
        except Exception as e:
            logger.warning(f"Weather data unavailable for Round {round_no}: {e}")
            avg_weather = {k: None for k in ['track_temp_avg', 'air_temp_avg', 'humidity_avg', 'pressure_avg', 'rainfall_avg', 'wind_speed_avg', 'wind_direction_avg']}

        # Pitstop data (pass already-loaded session to avoid double load)
        pitstop_df = get_team_fastest_pitstops(session)
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

        # Add sprint flag
        final_data['had_sprint'] = had_sprint
        constructor_data['had_sprint'] = had_sprint

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
            try:
                df = pd.read_excel(file)
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not read {file}: {e}")


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



if __name__ == '__main__':
    main()