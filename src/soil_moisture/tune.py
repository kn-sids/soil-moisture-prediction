import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def objective_factory(X_train, y_train):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 1,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "n_estimators": 2000,
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for tr, va in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
            model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = model.predict(X_va)
            rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        val = float(np.mean(rmses))
        trial.set_user_attr("mean_rmse", val)
        return val
    return objective


def run_study(X_train, y_train, n_trials=20):
    study = optuna.create_study(study_name="example_xgboost_study", direction='minimize')
    study.optimize(objective_factory(X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    print("Direction:", study.direction)
    print("Best value (objective):", study.best_value)
    print("Best params:", study.best_params)
    return study
