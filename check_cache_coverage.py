import os
from race_calendars import races_2024, races_2023, races_2022, races_2021, races_2020, races_2019, races_2018  # import your other years similarly

# Add all seasons and race calendars you want to check
season_race_map = {
    2024: races_2024,
    2023: races_2023,
    2022: races_2022,
    2021: races_2021,
    2020: races_2020,
    2019: races_2019,
    2018: races_2018
}

CACHE_DIR = "cache/json"

REQUIRED_SUFFIXES = ["race", "qualifying", "sprint"]


def build_expected_filenames(season, round_no):
    return [f"{suffix}_{season}_round_{round_no}.json" for suffix in REQUIRED_SUFFIXES]


def check_cache_files(season, races):
    missing = []
    complete = []
    not_started = []

    all_files = set(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else set()

    for round_no, gp_name in races:
        expected_files = build_expected_filenames(season, round_no)
        found = [file in all_files for file in expected_files]

        if all(found):
            complete.append((round_no, gp_name))
        elif any(found):
            missing.append((round_no, gp_name, [f for i, f in enumerate(expected_files) if not found[i]]))
        else:
            not_started.append((round_no, gp_name))

    return complete, missing, not_started


def main():
    for season, races in season_race_map.items():
        print(f"\nSeason {season}")
        complete, missing, not_started = check_cache_files(season, races)

        print(f"Complete races: {len(complete)}")
        for r in complete:
            print(f"     Round {r[0]} - {r[1]}")

        print(f"\nRaces with missing files: {len(missing)}")
        for r in missing:
            print(f"     Round {r[0]} - {r[1]} | Missing: {', '.join(r[2])}")

        print(f"\n? Not started yet: {len(not_started)}")
        for r in not_started:
            print(f"    Round {r[0]} - {r[1]}")


if __name__ == "__main__":
    main()
