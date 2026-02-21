import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from config import RANDOM_SEED, TOP_K_PER_CLUSTER, SCORE_THRESHOLD

FEATURE_COLS = [
    "sharpness", "brightness", "brightness_var", "contrast",
    "color_entropy", "face_count", "largest_face_ratio",
    "eye_openness", "smile_score",
]

def build_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    """Heuristic labels for bootstrapping (no user input needed)."""
    s = df.copy()
    for col in FEATURE_COLS:
        mn, mx = s[col].min(), s[col].max()
        s[col] = (s[col] - mn) / (mx - mn + 1e-8)

    heuristic = (
        0.30 * s["sharpness"] +
        0.15 * s["contrast"] +
        0.15 * s["color_entropy"] +
        0.20 * s["eye_openness"] +
        0.10 * s["largest_face_ratio"] +
        0.10 * (1 - (s["brightness"] - 0.5).abs() * 2)
    )
    threshold = heuristic.quantile(0.70)
    return (heuristic >= threshold).astype(int)

def train_ranker(df: pd.DataFrame, user_selected: list[str] = None):
    """Train XGBoost ranker. Returns (model, scaler, feature_cols)."""
    X = df[FEATURE_COLS].copy()

    if user_selected and len(user_selected) >= 5:
        y = df["filename"].isin(user_selected).astype(int)
        pos = df[y == 1]
        neg = df[y == 0].sample(min(len(pos) * 3, len(df[y == 0])), random_state=RANDOM_SEED)
        idx = pd.concat([pos, neg]).index
        X, y = X.loc[idx], y.loc[idx]
    else:
        y = build_pseudo_labels(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)
    return model, scaler, FEATURE_COLS

def predict_scores(model, scaler, df: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(df[FEATURE_COLS])
    return model.predict_proba(X)[:, 1]

def select_images(df: pd.DataFrame) -> pd.DataFrame:
    """Select top-K per cluster by predicted_score."""
    df = df.copy()
    df["selected"] = False

    for cluster_id, group in df.groupby("cluster"):
        top_idx = group["predicted_score"].nlargest(TOP_K_PER_CLUSTER).index
        df.loc[top_idx, "selected"] = True

    df.loc[df["predicted_score"] >= SCORE_THRESHOLD + 0.2, "selected"] = True
    return df

def precision_at_k(df: pd.DataFrame, k: int = 3) -> float:
    hits = 0
    total = 0
    for _, grp in df.groupby("cluster"):
        if len(grp) >= k:
            top_k = grp.nlargest(k, "predicted_score")["selected"].sum()
            hits += top_k
            total += k
    return hits / total if total > 0 else 0.0
