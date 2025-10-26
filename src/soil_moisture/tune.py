import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


def _xgb_builder(trial):
    """Return an XGBRegressor configured from trial params."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        n_estimators=2000,
        early_stopping_rounds=100,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 8.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
        # eval_metric="rmse"
    )


def _rf_builder(trial):
    """Return a RandomForestRegressor configured from trial params."""
    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 300, 1200, step=100),
        max_depth=trial.suggest_int("max_depth", 8, 40),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def _catboost_builder(trial):
    """Return a CatBoostRegressor configured from trial params."""
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        depth=trial.suggest_int("depth", 4, 9),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 5.0, 15.0),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        random_strength=trial.suggest_float("random_strength", 1.0, 2.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        rsm=trial.suggest_float("rsm", 0.6, 0.9),
        iterations=2000,
        early_stopping_rounds=100,
        allow_writing_files=False,
        random_seed=42,
        verbose=False,
        thread_count=1,  # 1 during tuning to avoid oversubscription
    )


_BUILDERS = {
    "xgb": _xgb_builder,
    "rf": _rf_builder,
    "catboost": _catboost_builder,
}


def objective_factory(model_key, X_train, y_train):
    """
    Returns an Optuna objective that builds `model_key` for each trial and
    minimizes mean RMSE over TimeSeriesSplit folds.
    """
    if model_key not in _BUILDERS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from: {list(_BUILDERS)}")

    def objective(trial):
        builder = _BUILDERS[model_key]
        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for tr, va in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
            model = builder(trial)

            if model_key == "xgb":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            elif model_key == "catboost":
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            else:  # "rf"
                model.fit(X_tr, y_tr)

            preds = model.predict(X_va)
            rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        trial.set_user_attr("mean_rmse", mean_rmse)
        return mean_rmse

    return objective


def run_study(model_key, X_train, y_train, n_trials=20, study_name=None, seed=42):
    """
    Tune a single model. model_key ∈ {'xgb','rf','catboost'}.
    """
    if study_name is None:
        study_name = f"tune_{model_key}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler)
    study.optimize(objective_factory(model_key, X_train, y_train), n_trials=n_trials, show_progress_bar=True)

    print(f"\n[{model_key}] Direction:", study.direction)
    print(f"[{model_key}] Best value (RMSE):", study.best_value)
    print(f"[{model_key}] Best params:", study.best_params)
    return study


def run_all_studies(X_train, y_train, models=("xgb", "rf", "catboost"), n_trials=20, seed=42):
    """
    Tune multiple models and return a dict {model_key: study}.
    """
    results = {}
    for m in models:
        results[m] = run_study(m, X_train, y_train, n_trials=n_trials, study_name=f"tune_{m}", seed=seed)
    return results
