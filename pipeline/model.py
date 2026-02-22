"""
Ranking model — photo-type aware.

Three separate heuristic scorers (one per photo type) are used to generate
pseudo-labels for bootstrapping. The XGBoost model is then trained with ALL
features but with photo_type as a categorical split signal, so it can
naturally learn type-specific feature weights.

Photo types: "subject" | "group" | "scenery"
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from config import RANDOM_SEED, TOP_K_PER_CLUSTER, SCORE_THRESHOLD

# Feature column groups 

UNIVERSAL_COLS = [
    "sharpness", "brightness", "brightness_var", "contrast", "color_entropy",
]

SUBJECT_COLS = [
    "subj_eye_openness", "subj_smile_score", "subj_face_sharpness",
    "subj_exposure_score", "subj_composition", "subj_face_area_ratio",
]

GROUP_COLS = [
    "grp_eyes_open_pct", "grp_min_eye_openness", "grp_expression_consistency",
    "grp_min_face_sharpness", "grp_face_visibility",
    "grp_composition", "grp_brightness", "grp_bg_contrast",
]

SCENERY_COLS = [
    "scen_global_sharpness", "scen_exposure_quality", "scen_dynamic_range",
    "scen_horizon_level", "scen_composition", "scen_color_score",
]

# photo_type encoded as int for XGBoost
PHOTO_TYPE_MAP = {"scenery": 0, "subject": 1, "group": 2}

ALL_FEATURE_COLS = (
    UNIVERSAL_COLS + SUBJECT_COLS + GROUP_COLS + SCENERY_COLS + ["photo_type_enc"]
)


# Heuristic scorers per photo type 

def _score_subject(s: pd.DataFrame) -> pd.Series:
    """
    Score = 0.35 * EyeScore
          + 0.25 * ExpressionScore (smile)
          + 0.20 * FaceSharpness
          + 0.10 * ExposureScore
          + 0.10 * CompositionScore
    """
    return (
        0.35 * s["subj_eye_openness"] +
        0.25 * s["subj_smile_score"] +
        0.20 * s["subj_face_sharpness"] +
        0.10 * s["subj_exposure_score"] +
        0.10 * s["subj_composition"]
    )


def _score_group(s: pd.DataFrame) -> pd.Series:
    """
    Score = 0.40 * EyesOpenPercent (weighted, largest faces matter more)
          + 0.25 * MinFaceSharpness  (penalise if any face is blurry)
          + 0.15 * ExpressionConsistency
          + 0.10 * FaceVisibility
          + 0.10 * CompositionScore

    An extra blink penalty: if min_eye_openness < 0.3 (someone clearly
    blinking) the score is multiplied down by 0.6.
    """
    base = (
        0.40 * s["grp_eyes_open_pct"] +
        0.25 * s["grp_min_face_sharpness"] +
        0.15 * s["grp_expression_consistency"] +
        0.10 * s["grp_face_visibility"] +
        0.10 * s["grp_composition"]
    )
    blink_penalty = s["grp_min_eye_openness"].apply(lambda v: 0.6 if v < 0.30 else 1.0)
    return base * blink_penalty


def _score_scenery(s: pd.DataFrame) -> pd.Series:
    """
    Score = 0.35 * GlobalSharpness
          + 0.25 * ExposureQuality
          + 0.15 * HorizonLevel
          + 0.15 * CompositionScore
          + 0.10 * ColorScore
    """
    return (
        0.35 * s["scen_global_sharpness"] +
        0.25 * s["scen_exposure_quality"] +
        0.15 * s["scen_horizon_level"] +
        0.15 * s["scen_composition"] +
        0.10 * s["scen_color_score"]
    )


# Pseudo-label builder

def build_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    """
    Generate heuristic 0/1 labels per photo type.
    Within each type, the top 30% by heuristic score are labelled 1.
    Ensures XGBoost learns type-appropriate quality signals.
    """
    labels = pd.Series(0, index=df.index)

    for ptype, scorer in [
        ("subject", _score_subject),
        ("group",   _score_group),
        ("scenery", _score_scenery),
    ]:
        mask = df["photo_type"] == ptype
        if mask.sum() == 0:
            continue
        sub = df[mask].copy()
        scores = scorer(sub)
        threshold = scores.quantile(0.70)
        labels[mask] = (scores >= threshold).astype(int).values

    return labels


# Training
def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode photo_type and return feature matrix. Also stamps photo_type_enc onto df in-place."""
    df["photo_type_enc"] = df["photo_type"].map(PHOTO_TYPE_MAP).fillna(0).astype(int)
    return df[ALL_FEATURE_COLS].copy()


def train_ranker(df: pd.DataFrame, user_selected: list[str] = None):
    """
    Train XGBoost ranker.
    Returns (model, scaler, feature_cols).
    """
    X = _prepare_features(df)

    if user_selected and len(user_selected) >= 5:
        y = df["filename"].isin(user_selected).astype(int)
        pos = df[y == 1]
        neg = df[y == 0].sample(
            min(len(pos) * 3, (y == 0).sum()), random_state=RANDOM_SEED
        )
        idx = pd.concat([pos, neg]).index
        X, y = X.loc[idx], y.loc[idx]
    else:
        y = build_pseudo_labels(df)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)
    return model, scaler, ALL_FEATURE_COLS


# Scoring & selection 

def predict_scores(model, scaler, df: pd.DataFrame) -> np.ndarray:
    X = _prepare_features(df)
    return model.predict_proba(scaler.transform(X))[:, 1]


def select_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select images per cluster with a hard cap to prevent burst bloat:

    - Clusters with 1–2 images   → select the best 1
    - Clusters with 3–6 images   → select the best 2  (one burst, two keepers max)
    - Clusters with 7+ images    → select the best 3  (hard cap regardless of size)

    Additionally, any image scoring >= SCORE_THRESHOLD + 0.2 globally is
    selected, but still capped at 3 per cluster to avoid flooding from a
    single long burst.
    """
    df = df.copy()
    df["selected"] = False

    for _cluster_id, group in df.groupby("cluster"):
        n = len(group)
        if n <= 15:
            cap = 1
        elif n <= 30:
            cap = 2
        else:
            cap = 3   # hard ceiling even for very large burst clusters

        top_idx = group["predicted_score"].nlargest(cap).index
        df.loc[top_idx, "selected"] = True

    # High-confidence global boost — but still respect per-cluster cap of 3
    for _cluster_id, group in df.groupby("cluster"):
        already_selected = group["selected"].sum()
        if already_selected >= 3:
            continue
        remaining_cap = 3 - already_selected
        high_conf = group[
            (~group["selected"]) & (group["predicted_score"] >= SCORE_THRESHOLD + 0.2)
        ].nlargest(remaining_cap, "predicted_score").index
        df.loc[high_conf, "selected"] = True

    return df


def precision_at_k(df: pd.DataFrame, k: int = 3) -> float:
    hits, total = 0, 0
    for _, grp in df.groupby("cluster"):
        if len(grp) >= k:
            hits  += grp.nlargest(k, "predicted_score")["selected"].sum()
            total += k
    return hits / total if total > 0 else 0.0
