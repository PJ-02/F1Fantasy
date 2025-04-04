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


def main():
    season = 2023
    round_no = 6  # Example race

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

    # Save final data
    clean_data.to_excel("f1_driver_data_with_sprint_qual.xlsx", index=False)
    print("✅ Data saved to f1_driver_data_with_sprint_qual.xlsx")



# Run the pipeline
main()
