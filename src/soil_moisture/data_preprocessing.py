import glob
import pandas as pd
import numpy as np


def combine_multiple_files(path):
    """Load multiple CSV files and combine them."""
    dfs = []
    files = glob.glob(path)

    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Total records loaded: {len(combined_df)}")

    return combined_df


def clean_sensor_data(data_file):
    """Clean sensor data for model training."""
    df = data_file.copy()

    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # Rename columns
    column_mapping = {
        'AIR_TEMPERATURE[C]': 'air_temperature_c',
        'AIR_HUMIDITY[%]': 'air_humidity_pct',
        'SOIL_TEMPERATURE[C]': 'soil_temperature_c',
        'SOIL_MOISTURE[%]': 'soil_moisture_pct',
        'SOIL_MOISTURE[RAW]': 'soil_moisture_raw',
        'LIGHT_LEVEL[LUX]': 'light_level_lux'
    }

    df = df.rename(columns=column_mapping)
    print(f"\nRenamed columns: {df.columns.tolist()}")

    # Handle LIGHT_LEVEL
    df['light_level_lux'] = df['light_level_lux'].fillna(0)
    df['light_level_lux'] = df['light_level_lux'].replace([np.inf, -np.inf], 0)

    # Handle other missing values
    numeric_columns = ['air_temperature_c', 'air_humidity_pct',
                       'soil_temperature_c', 'soil_moisture_pct',
                       'soil_moisture_raw']

    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].ffill().bfill()

    # Remove remaining NaN rows
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)

    if rows_before > rows_after:
        print(f"\nRemoved {rows_before - rows_after} rows with remaining missing values")

    # Handle duplicates
    duplicates = df.duplicated(subset=['DATE', 'TIME']).sum()
    if duplicates > 0:
        print(f"\nFound {duplicates} duplicate timestamps. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['DATE', 'TIME'], keep='first')

    # Convert to datetime
    df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'],
                                    format='%d.%m.%Y %H:%M:%S')

    # Validate data ranges
    df = df[(df['air_humidity_pct'] >= 0) & (df['air_humidity_pct'] <= 100)]
    df = df[(df['soil_moisture_pct'] >= 0) & (df['soil_moisture_pct'] <= 100)]
    df = df[df['light_level_lux'] >= 0]

    print(f"\nFinal data shape: {df.shape}")

    return df
