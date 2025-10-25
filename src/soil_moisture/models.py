import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


def train_xgboost_model(X_train, y_train, X_test, y_test, params=None):
    """Train XGBoost model."""
    print("\n" + "=" * 70)
    print("XGBOOST MODEL TRAINING")
    print("=" * 70)

    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1
        }

    print("\nModel Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    split_val = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:split_val], X_train.iloc[split_val:]
    y_tr, y_val = y_train.iloc[:split_val], y_train.iloc[split_val:]

    model = xgb.XGBRegressor(**params, eval_metric="rmse", n_estimators=2000, early_stopping_rounds=100)

    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )

    return model


def train_random_forest_model(X_train, y_train, params=None):
    """Train Random Forest model."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST MODEL TRAINING")
    print("=" * 70)

    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }

    print("\nModel Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    model = RandomForestRegressor(**params)

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    return model


def train_catboost_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train CatBoost model.
    """
    if params is None:
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "depth": 8,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "random_strength": 1.0,
            "bagging_temperature": 0.5,
            "iterations": 2000,  # high cap; use early stopping
            "early_stopping_rounds": 100,
            "random_seed": 42,
            "allow_writing_files": False,
            "verbose": False,
        }

    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    return model
