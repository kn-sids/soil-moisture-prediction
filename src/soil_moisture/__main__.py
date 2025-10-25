import optuna.visualization as vis
from data_preprocessing import combine_multiple_files, clean_sensor_data
from features import create_features
from split import prepare_data
from models import train_xgboost_model, train_random_forest_model
from evaluate import evaluate_model
from pathlib import Path
from tune import run_study


if __name__ == "__main__":
    # Get the absolute path to the repo root (two levels up from this file)
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    file_path = str(data_dir / "*.csv")

    # Load and prepare data
    combined_data = combine_multiple_files(file_path)
    cleaned_data = clean_sensor_data(combined_data)
    df_with_features = create_features(cleaned_data)
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df_with_features)

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

    # Compare models
    # compare_models([xgb_metrics, rf_metrics])
