import optuna.visualization as vis
from pathlib import Path
from .data_preprocessing import combine_multiple_files, clean_sensor_data
from .features import create_features, add_forecast_features
from .split import prepare_data
from .models import train_xgboost_model, train_random_forest_model, train_catboost_model
from .evaluate import evaluate_model
from .tune import run_study


if __name__ == "__main__":
    # Get the absolute path to the repo root (two levels up from this file)
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    file_path = str(data_dir / "*.csv")

    # Load and prepare data
    combined_data = combine_multiple_files(file_path)
    cleaned_data = clean_sensor_data(combined_data)
    df_with_features = create_features(cleaned_data)
    df_with_features = df_with_features.sort_values('datetime').reset_index(drop=True)
    supervised = add_forecast_features(df_with_features, horizon=1, lags=(1, 6, 12, 72), roll_windows=(6, 24, 72))
    X_train, X_test, y_train, y_test, feature_names = prepare_data(supervised, test_size=0.2)

    # Retrieve the best parameter values
    study = run_study(X_train, y_train, n_trials=10)
    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")
    # vis.plot_param_importances(study).show()
    # vis.plot_optimization_history(study).show()

    # Train XGBoost
    xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test, params=best_params)
    xgb_metrics = evaluate_model(xgb_model, "XGBoost", X_train, y_train, X_test, y_test, feature_names)

    # Train Random Forest
    rf_model = train_random_forest_model(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, "Random Forest", X_train, y_train, X_test, y_test, feature_names)

    # Train CatBoost model
    split_val = int(len(X_train) * 0.9)
    cb_model = train_catboost_model(
        X_train.iloc[:split_val], y_train.iloc[:split_val],
        X_train.iloc[split_val:], y_train.iloc[split_val:]
    )
    cb_metrics = evaluate_model(cb_model, "CatBoost", X_train, y_train, X_test, y_test, feature_names)

    # Compare models
    # compare_models([xgb_metrics, rf_metrics])
