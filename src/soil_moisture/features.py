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


def add_forecast_features(df, horizon=1, lags=(1, 6, 12, 72), roll_windows=(6, 24, 72)):
    """
    Convert to a supervised forecasting frame:
      - target_future = soil_moisture_pct shifted -horizon (label at t is y_{t+h})
      - use only *past* info via lags/rollings (no same-time leakage)
    Assumes df is already sorted by 'datetime' ascending.
    """
    df = df.copy()

    # future label
    df['target_future'] = df['soil_moisture_pct'].shift(-horizon)

    # lags (past only)
    for k in lags:
        df[f'soil_moisture_raw_lag_{k}'] = df['soil_moisture_raw'].shift(k)
        df[f'soil_moisture_pct_lag_{k}'] = df['soil_moisture_pct'].shift(k)

    # rolling means (past only)
    for w in roll_windows:
        df[f'soil_raw_rollmean_{w}'] = df['soil_moisture_raw'].rolling(w, min_periods=1).mean()
        df[f'soil_pct_rollmean_{w}']  = df['soil_moisture_pct'].rolling(w, min_periods=1).mean()

    # drop rows without full supervision window
    first_valid = max(lags) if lags else 0
    if horizon > 0:
        df = df.iloc[first_valid: -horizon].copy()
    else:
        df = df.iloc[first_valid:].copy()

    return df

