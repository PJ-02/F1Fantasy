import pandas as pd

from Utils.common import logger

def get_driver_telemetry(session):
    telemetry_data = []
    try:
        # Use pick_drivers for each driver so the ID format (number vs abbreviation) is handled by FastF1
        all_laps = session.laps
    except Exception as e:
        logger.warning(f"Laps data not available for telemetry: {e}")
        return pd.DataFrame()

    for drv in session.drivers:
        try:
            driver_laps = session.laps.pick_drivers(drv).sort_values('LapNumber')
            if driver_laps.empty:
                continue

            # Overtake counting: each lap where race position improved vs previous lap
            overtakes = 0
            prev_pos = None
            for _, row in driver_laps.iterrows():
                current_pos = row['Position']
                if pd.notna(current_pos) and prev_pos is not None and pd.notna(prev_pos):
                    if current_pos < prev_pos:
                        overtakes += 1
                if pd.notna(current_pos):
                    prev_pos = current_pos

            # Use the driver abbreviation from the lap data as driver_id
            drv_abbr = driver_laps['Driver'].iloc[0].lower()

            fastest_lap = driver_laps.pick_fastest()
            if fastest_lap is None or fastest_lap.empty:
                telemetry_data.append({
                    'driver_id': drv_abbr,
                    'top_speed': None,
                    'avg_throttle': None,
                    'brake_usage': None,
                    'overtakes': overtakes
                })
                continue
            car_data = fastest_lap.get_car_data().add_distance()
            telemetry_data.append({
                'driver_id': drv_abbr,
                'top_speed': car_data['Speed'].max(),
                'avg_throttle': car_data['Throttle'].mean(),
                'brake_usage': car_data['Brake'].mean(),
                'overtakes': overtakes
            })

        except Exception as e:
            logger.warning(f"Failed to load telemetry for driver {drv}: {e}")
            continue
    return pd.DataFrame(telemetry_data)