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
    # Add if driver did not finish (DNF)
    df['DNF'] = df['status'].apply(lambda x: 1 if 'Accident' in x or 'Retired' in x else 0)

    # Mark if driver had the fastest lap
    df['fastest_lap'] = df['fastest_lap_rank'].apply(lambda x: True if x == 1 else False)

    # Calculate how many positions gained
    df['positions_gained'] = df['grid'] - df['position']
    df['positions_gained'] = df['positions_gained'].fillna(0)

    return df

def main():
    season = 2023
    round_no = 5  # Try 5 for Miami 2023

    raw_data = get_race_results(season, round_no)
    if raw_data.empty:
        print("No race data available.")
        return

    clean_data = preprocess_driver_data(raw_data)
    clean_data.to_excel("f1_driver_data.xlsx", index=False)
    print("Data saved to f1_driver_data.xlsx")

# Run the pipeline
main()
