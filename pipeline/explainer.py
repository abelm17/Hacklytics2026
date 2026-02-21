import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def generate_shap_plots(model, scaler, df: pd.DataFrame,
                        feature_cols: list[str], output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    X = scaler.transform(df[feature_cols])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary.png", dpi=120)
    plt.close()

    # Feature importance bar
    fig, ax = plt.subplots(figsize=(8, 5))
    importance = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(importance)
    ax.barh([feature_cols[i] for i in sorted_idx], importance[sorted_idx], color="#4F8EF7")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance.png", dpi=120)
    plt.close()

    return shap_values

def get_top_features_for_image(shap_values: np.ndarray, idx: int,
                                feature_cols: list[str], top_n: int = 3) -> list[dict]:
    sv = shap_values[idx]
    sorted_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    return [{"feature": feature_cols[i], "shap": float(sv[i])} for i in sorted_idx]
