import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, feature_names):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print(f"{model_name.upper()} MODEL EVALUATION")
    print("=" * 70)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTRAINING SET METRICS:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")

    print("\nTEST SET METRICS:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    # Overfitting check
    rmse_diff = train_rmse - test_rmse
    r2_diff = train_r2 - test_r2

    print("\nOVERFITTING CHECK:")
    print(f"  RMSE difference: {rmse_diff:.4f}")
    print(f"  R² difference:   {r2_diff:.4f}")

    if abs(r2_diff) < 0.05 and abs(rmse_diff) < 1.0:
        print("Model appears well-balanced")
    elif train_r2 > test_r2 + 0.1:
        print("Possible overfitting detected")
    else:
        print("Model generalization looks good")

    # Create visualizations
    fig = plt.figure(figsize=(18, 12))

    # Predictions vs Actual (Train)
    plt.subplot(2, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.3, s=1)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual Soil Moisture (%)')
    plt.ylabel('Predicted Soil Moisture (%)')
    plt.title(f'Train Set: Predictions vs Actual\nR² = {train_r2:.4f}')
    plt.grid(True, alpha=0.3)

    # Predictions vs Actual (Test)
    plt.subplot(2, 3, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.3, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Soil Moisture (%)')
    plt.ylabel('Predicted Soil Moisture (%)')
    plt.title(f'Test Set: Predictions vs Actual\nR² = {test_r2:.4f}')
    plt.grid(True, alpha=0.3)

    # Residuals (Train)
    plt.subplot(2, 3, 3)
    train_residuals = y_train - y_train_pred
    plt.scatter(y_train_pred, train_residuals, alpha=0.3, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Soil Moisture (%)')
    plt.ylabel('Residuals')
    plt.title('Train Set: Residual Plot')
    plt.grid(True, alpha=0.3)

    # Residuals (Test)
    plt.subplot(2, 3, 4)
    test_residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, test_residuals, alpha=0.3, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Soil Moisture (%)')
    plt.ylabel('Residuals')
    plt.title('Test Set: Residual Plot')
    plt.grid(True, alpha=0.3)

    # Error Distribution
    plt.subplot(2, 3, 5)
    plt.hist(test_residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (%)')
    plt.ylabel('Frequency')
    plt.title(f'Test Set: Error Distribution\nMean: {test_residuals.mean():.4f}')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)

    # Actual vs Predicted over time
    plt.subplot(2, 3, 6)
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, y_test_pred, label='Predicted')
    plt.title('Test: Soil Moisture (%) — Actual vs Predicted')
    plt.xlabel('Index (chronological test points)')
    plt.ylabel('Soil Moisture (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.show()

    try:
        shap_summary_plots(model, X_train, X_test, model_name)
    except Exception as e:
        print(f"[SHAP] Skipped SHAP plots due to: {e}")

    return {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }


def compare_models(metrics_list):
    """Compare performance of multiple models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame(metrics_list)
    print("\n", comparison_df.to_string(index=False))

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    models = comparison_df['model_name']

    # RMSE Comparison
    axes[0, 0].bar(models, comparison_df['test_rmse'], color=['skyblue', 'lightcoral'])
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Test Set RMSE Comparison')
    axes[0, 0].grid(True, alpha=0.3)

    # MAE Comparison
    axes[0, 1].bar(models, comparison_df['test_mae'], color=['skyblue', 'lightcoral'])
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Test Set MAE Comparison')
    axes[0, 1].grid(True, alpha=0.3)

    # R² Comparison
    axes[1, 0].bar(models, comparison_df['test_r2'], color=['skyblue', 'lightcoral'])
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('Test Set R² Comparison')
    axes[1, 0].grid(True, alpha=0.3)

    # Train vs Test R² Comparison
    x = np.arange(len(models))
    width = 0.35
    axes[1, 1].bar(x - width / 2, comparison_df['train_r2'], width, label='Train', color='skyblue')
    axes[1, 1].bar(x + width / 2, comparison_df['test_r2'], width, label='Test', color='lightcoral')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Train vs Test R² Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Determine best model
    best_model_idx = comparison_df['test_r2'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'model_name']
    print(f"\nBest performing model: {best_model}")
    print(f"   Test R²: {comparison_df.loc[best_model_idx, 'test_r2']:.4f}")
    print(f"   Test RMSE: {comparison_df.loc[best_model_idx, 'test_rmse']:.4f}")


def shap_summary_plots(model, X_train, X_test, model_name="Model", max_points=2000, save_prefix="shap"):
    # Subsample test for plotting (avoid leakage: background from TRAIN only)
    X_bg = X_train.sample(min(1000, len(X_train)), random_state=42)
    Xs   = X_test.iloc[:max_points].copy()

    try:
        # Prefer TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        # SHAP’s API differs by version; try new, then fallback to old
        try:
            sv = explainer(Xs, check_additivity=False)
            values = sv.values
            feature_names = sv.feature_names
        except TypeError:
            values = explainer.shap_values(Xs)
            feature_names = list(Xs.columns)

    except Exception:
        # Fallback: model-agnostic explainer with a callable
        explainer = shap.Explainer(model.predict, X_bg)
        sv = explainer(Xs)
        values = sv.values
        feature_names = sv.feature_names

    # 1) Beeswarm
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(sv, show=False, max_display=20)
    plt.title(f"{model_name} — SHAP summary (beeswarm)")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_{model_name}_beeswarm.png", dpi=200); plt.show()

    # 2) Bar (global mean |SHAP|)
    plt.figure(figsize=(8, 6))
    shap.plots.bar(sv, show=False, max_display=20)
    plt.title(f"{model_name} — SHAP global importance")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_{model_name}_bar.png", dpi=200); plt.show()

    # 3) Dependence for the top feature
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = int(np.argsort(mean_abs)[::-1][0])
    top_feature = feature_names[top_idx]

    plt.figure(figsize=(8, 6))
    shap.plots.scatter(sv[:, top_feature], show=False)
    plt.title(f"{model_name} — SHAP dependence: {top_feature}")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_{model_name}_dependence_{top_feature}.png", dpi=200); plt.show()
