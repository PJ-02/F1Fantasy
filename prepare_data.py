import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full combined dataset
driver_df = pd.read_excel("GeneratedSpreadsheets/all_seasons_fantasy_driver_data.xlsx")

# Preview it
print(driver_df.shape)
print(driver_df.columns)
print(driver_df.head())

columns_to_drop = ['driver_name', 'status', 'event_name', 'fastest_lap_time']
df = driver_df.drop(columns=columns_to_drop)

df = df.dropna()  # or df.fillna(0), depending on your preference

X = df.drop(columns=['fantasy_points_total'])  # features
y = df['fantasy_points_total']                 # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)