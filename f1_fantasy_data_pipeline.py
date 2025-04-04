import requests
import pandas as pd


def get_race_results(season, round_no):
    # API URL for a specific season and race
    url = f"https://ergast.com/api/f1/{season}/{round_no}/results.json"
    response = requests.get(url)
    data = response.json()

    # Check if race exists
    if not data['MRData']['RaceTable']['Races']:
        print("Race data not found.")
        return pd.DataFrame()

    results = data['MRData']['RaceTable']['Races'][0]['Results']
    race_data = []

    for r in results:
        entry = {
            'driver_id': r['Driver']['driverId'],
            'driver_name': f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
            'constructor_name': r['Constructor']['name'],
            'grid': int(r['grid']),
            'position': int(r['position']) if r['position'].isdigit() else None,
            'status': r['status'],
            'fastest_lap_rank': int(r['FastestLap']['rank']) if 'FastestLap' in r else None,
            'fastest_lap_time': r['FastestLap']['Time']['time'] if 'FastestLap' in r else None
        }
        race_data.append(entry)

    return pd.DataFrame(race_data)


def preprocess_driver_data(df):
    df['DNF'] = df['status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)
    df['fastest_lap'] = df['fastest_lap_rank'].apply(lambda x: True if x == 1 else False)
    df['positions_gained'] = df['grid'] - df['position']
    df['positions_gained'] = df['positions_gained'].fillna(0)

    # Sprint fantasy features
    df['sprint_positions_gained'] = df['sprint_grid'] - df['sprint_pos']
    df['sprint_positions_gained'] = df['sprint_positions_gained'].fillna(0)
    df['sprint_dnf'] = df['sprint_status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)

    return df


def get_qualifying_results(season, round_no):
    url = f"https://ergast.com/api/f1/{season}/{round_no}/qualifying.json"
    response = requests.get(url)
    data = response.json()

    if not data['MRData']['RaceTable']['Races']:
        print("No qualifying data found.")
        return pd.DataFrame()

    qualifying = data['MRData']['RaceTable']['Races'][0]['QualifyingResults']
    qual_data = []
    for q in qualifying:
        entry = {
            'driver_id': q['Driver']['driverId'],
            'qualifying_pos': int(q['position'])
        }
        qual_data.append(entry)

    return pd.DataFrame(qual_data)


def get_sprint_results(season, round_no):
    url = f"https://ergast.com/api/f1/{season}/{round_no}/sprint.json"
    response = requests.get(url)
    data = response.json()

    if not data['MRData']['RaceTable']['Races']:
        print("No sprint data found.")
        return pd.DataFrame()

    sprints = data['MRData']['RaceTable']['Races'][0]['SprintResults']
    sprint_data = []
    for s in sprints:
        entry = {
            'driver_id': s['Driver']['driverId'],
            'sprint_pos': int(s['position']),
            'sprint_status': s['status'],
            'sprint_grid': int(s['grid'])  # Sprint grid start
        }
        sprint_data.append(entry)

    return pd.DataFrame(sprint_data)


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


def main():
    season = 2023
    round_no = 4  # Example race

    race_df = get_race_results(season, round_no)
    if race_df.empty:
        print("No race data found.")
        return

    # Fetch qualifying and sprint
    qual_df = get_qualifying_results(season, round_no)
    sprint_df = get_sprint_results(season, round_no)

    # Merge qualifying data
    merged_df = race_df.merge(qual_df, on="driver_id", how="left")

    # ✅ Only merge sprint if it has data
    if not sprint_df.empty and 'driver_id' in sprint_df.columns:
        merged_df = merged_df.merge(sprint_df, on="driver_id", how="left")
    else:
        print("⚠️ No Sprint data found, skipping Sprint merge.")
        # Add empty columns so your preprocessing doesn't break
        merged_df['sprint_pos'] = None
        merged_df['sprint_status'] = ""
        merged_df['sprint_grid'] = None

    # Preprocess combined data
    clean_data = preprocess_driver_data(merged_df)

    # ✅ Add Fantasy Points Calculation
    final_data = calculate_fantasy_points(clean_data)

    # Save final data
    final_data.to_excel("f1_driver_data_with_fantasy_points.xlsx", index=False)
    print("✅ Data saved to f1_driver_data_with_fantasy_points.xlsx")


# Run the pipeline
main()
