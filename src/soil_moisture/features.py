import pandas as pd
import numpy as np


def create_features(df):
    """Create additional features for the model."""
    df = df.copy()

    # Time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Time of day categories
    df['time_of_day'] = pd.cut(df['hour'],
                               bins=[0, 6, 12, 18, 24],
                               labels=['night', 'morning', 'afternoon', 'evening'],
                               include_lowest=True)
    df['is_night'] = (df['time_of_day'] == 'night').astype(int)
    df['is_morning'] = (df['time_of_day'] == 'morning').astype(int)
    df['is_afternoon'] = (df['time_of_day'] == 'afternoon').astype(int)
    df['is_evening'] = (df['time_of_day'] == 'evening').astype(int)

    # Interaction features
    df['temp_humidity_interaction'] = df['air_temperature_c'] * df['air_humidity_pct']
    df['soil_temp_light_interaction'] = df['soil_temperature_c'] * df['light_level_lux']

    # Rolling averages
    df = df.sort_values('datetime').reset_index(drop=True)
    df['air_temp_rolling_mean_6h'] = df['air_temperature_c'].rolling(window=72, min_periods=1).mean()
    df['soil_temp_rolling_mean_6h'] = df['soil_temperature_c'].rolling(window=72, min_periods=1).mean()
    df['light_rolling_mean_6h'] = df['light_level_lux'].rolling(window=72, min_periods=1).mean()

    # Temperature difference
    df['temp_diff'] = df['air_temperature_c'] - df['soil_temperature_c']

    # Light presence
    df['has_light'] = (df['light_level_lux'] > 0).astype(int)

    feature_cols = [c for c in df.columns if
                    c not in ['DATE', 'TIME', 'datetime', 'soil_moisture_pct', 'date_only', 'day', 'time_of_day']]
    print(f"\nTotal features created: {len(feature_cols)}")

    return df
