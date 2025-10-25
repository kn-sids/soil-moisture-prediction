def prepare_data(df, target_col='soil_moisture_pct', test_size=0.2):
    if 'datetime' not in df.columns:
        raise ValueError("Column 'datetime' is required for time-aware split.")

    # 1) Sort chronologically
    df_sorted = df.sort_values('datetime').reset_index(drop=True)

    # 2) Build features/target FROM THE SORTED FRAME
    exclude_cols = ['DATE','TIME','datetime', target_col, 'date_only','day','time_of_day']
    feature_cols = [c for c in df_sorted.columns if c not in exclude_cols]

    X = df_sorted[feature_cols]
    y = df_sorted[target_col]

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # 3) Chronological split index (use test_size)
    n = len(df_sorted)
    split_idx = int((1 - test_size) * n)
    if not (0 < split_idx < n):
        raise ValueError("test_size leads to empty train or test set.")

    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test,  y_test  = X.iloc[split_idx:], y.iloc[split_idx:]

    print(f"\nTrain set size: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/n*100:.1f}%)")

    return X_train, X_test, y_train, y_test, feature_cols
