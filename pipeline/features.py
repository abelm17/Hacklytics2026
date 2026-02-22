"""
Feature extraction — photo-type aware.

Photo types are determined by face_count:
  - SUBJECT  : face_count == 1
  - GROUP    : face_count >= 2
  - SCENERY  : face_count == 0

Every image gets ALL base features extracted.
Type-specific features are computed on top and stored with a prefix so
the model can use them. Features that are irrelevant for a photo type
are stored as NaN-safe neutrals (0.0 or 0.5) so XGBoost can ignore them
via the type-gating in model.py.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.stats import entropy as scipy_entropy

mp_face    = mp.solutions.face_detection
mp_mesh    = mp.solutions.face_mesh

# ── MediaPipe eye landmark indices (EAR) ─────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Mouth landmark indices (simple smile proxy via mouth-corner height) ───────
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291
MOUTH_TOP   = 13
MOUTH_BOTTOM = 14


def extract_all_features(records: list[dict]) -> list[dict]:
    """
    Adds feature keys to each record dict in-place.
    Initialises MediaPipe once and reuses across all images.
    """
    face_det  = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    face_mesh = mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=20,           # allow large groups
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    for rec in records:
        arr  = rec["array"]                           # RGB uint8
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # ── Universal features (all photo types) ─────────────────────────
        rec["sharpness"]      = _sharpness(gray)
        rec["brightness"]     = float(np.mean(gray)) / 255.0   # normalise to 0-1
        rec["brightness_var"] = float(np.var(gray))  / (255.0 ** 2)
        rec["contrast"]       = float(gray.std())    / 255.0
        rec["color_entropy"]  = _color_entropy(arr)

        # ── Face detection ────────────────────────────────────────────────
        det_results = face_det.process(arr)
        detections  = det_results.detections if det_results.detections else []
        face_count  = len(detections)
        rec["face_count"] = face_count

        # Determine photo type
        if face_count == 0:
            photo_type = "scenery"
        elif face_count == 1:
            photo_type = "subject"
        else:
            photo_type = "group"
        rec["photo_type"] = photo_type

        # ── Face mesh (needed for eye/expression features) ────────────────
        mesh_results = face_mesh.process(arr) if face_count > 0 else None
        all_landmarks = (mesh_results.multi_face_landmarks
                         if mesh_results and mesh_results.multi_face_landmarks
                         else [])

        h, w = arr.shape[:2]

        # ── Subject features (face_count == 1) ───────────────────────────
        if photo_type == "subject":
            bb      = detections[0].location_data.relative_bounding_box
            face_area = bb.width * bb.height
            rec["subj_face_area_ratio"] = float(face_area)

            if all_landmarks:
                lm = all_landmarks[0]
                rec["subj_eye_openness"]  = _ear_score(lm, w, h)
                rec["subj_smile_score"]   = _smile_score(lm, w, h)
                rec["subj_face_sharpness"] = _roi_sharpness(
                    gray, bb, h, w
                )
            else:
                rec["subj_eye_openness"]   = 0.5
                rec["subj_smile_score"]    = 0.5
                rec["subj_face_sharpness"] = rec["sharpness"]

            rec["subj_exposure_score"] = _exposure_score(gray)
            rec["subj_composition"]    = _composition_score(arr)

            # Neutral group / scenery fields
            rec.update(_neutral_group())
            rec.update(_neutral_scenery())

        # ── Group features (face_count >= 2) ─────────────────────────────
        elif photo_type == "group":
            face_areas = []
            for det in detections:
                bb = det.location_data.relative_bounding_box
                face_areas.append(bb.width * bb.height)

            # Sort landmarks by face area descending (largest face first)
            sorted_pairs = sorted(
                zip(face_areas, range(len(detections))),
                reverse=True
            )
            sorted_areas = [p[0] for p in sorted_pairs]

            # Weighted eye-open score: central/largest faces count more
            weights = np.array([1.0 / (i + 1) for i in range(len(all_landmarks))])
            weights = weights / weights.sum() if weights.sum() > 0 else weights

            ear_scores = []
            smile_scores = []
            face_sharpnesses = []

            for idx, lm in enumerate(all_landmarks):
                ear_scores.append(_ear_score(lm, w, h))
                smile_scores.append(_smile_score(lm, w, h))
                bb_i = detections[min(idx, len(detections)-1)].location_data.relative_bounding_box
                face_sharpnesses.append(_roi_sharpness(gray, bb_i, h, w))

            if ear_scores:
                w_arr = weights[:len(ear_scores)]
                w_arr = w_arr / w_arr.sum()
                rec["grp_eyes_open_pct"]      = float(np.average(ear_scores, weights=w_arr))
                rec["grp_min_eye_openness"]   = float(min(ear_scores))   # penalise any blinker
                rec["grp_expression_consistency"] = float(1.0 - np.std(smile_scores))
                rec["grp_min_face_sharpness"] = float(min(face_sharpnesses))
            else:
                rec["grp_eyes_open_pct"]          = 0.5
                rec["grp_min_eye_openness"]        = 0.5
                rec["grp_expression_consistency"]  = 0.5
                rec["grp_min_face_sharpness"]      = rec["sharpness"]

            rec["grp_face_visibility"]  = float(min(1.0, len(all_landmarks) / max(face_count, 1)))
            rec["grp_composition"]      = _composition_score(arr)
            rec["grp_brightness"]       = rec["brightness"]
            rec["grp_bg_contrast"]      = _background_face_contrast(arr, gray, detections, h, w)

            rec.update(_neutral_subject())
            rec.update(_neutral_scenery())

        # ── Scenery features (face_count == 0) ───────────────────────────
        else:
            rec["scen_global_sharpness"]  = rec["sharpness"]
            rec["scen_exposure_quality"]  = _exposure_score(gray)
            rec["scen_dynamic_range"]     = _dynamic_range(gray)
            rec["scen_horizon_level"]     = _horizon_level(gray)
            rec["scen_composition"]       = _composition_score(arr)
            rec["scen_color_score"]       = _color_score(arr)

            rec.update(_neutral_subject())
            rec.update(_neutral_group())

    face_det.close()
    face_mesh.close()
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Universal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalise to 0-1 with soft cap at 3000 (covers typical camera output)
    return float(min(lap / 3000.0, 1.0))

def _color_entropy(arr: np.ndarray) -> float:
    entropies = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=64, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropies.append(scipy_entropy(hist))
    return float(np.mean(entropies) / np.log(64))   # normalise to 0-1

def _exposure_score(gray: np.ndarray) -> float:
    """
    Peaks when mean brightness ~= 118-138 (slightly above mid-grey).
    Falls off for over/under exposure.
    """
    mean = np.mean(gray)
    # Gaussian centred at 128, std=60
    score = np.exp(-0.5 * ((mean - 128) / 60) ** 2)
    return float(score)

def _composition_score(arr: np.ndarray) -> float:
    """
    Rule-of-thirds proxy: measure edge energy at the third lines.
    Normalised to 0-1.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    h, w  = edges.shape

    # Thirds lines
    rows = [h // 3, 2 * h // 3]
    cols = [w // 3, 2 * w // 3]
    margin = max(5, h // 30)

    thirds_energy = 0.0
    for r in rows:
        thirds_energy += edges[max(0, r-margin):r+margin, :].mean()
    for c in cols:
        thirds_energy += edges[:, max(0, c-margin):c+margin].mean()

    total_energy = edges.mean() + 1e-6
    score = min(thirds_energy / (4 * total_energy * 1.5), 1.0)
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Subject / portrait helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ear_score(landmarks, w: int, h: int) -> float:
    """Eye Aspect Ratio → normalised 0-1 (1 = fully open)."""
    def ear(indices):
        pts = np.array([[landmarks.landmark[i].x * w,
                         landmarks.landmark[i].y * h] for i in indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C + 1e-6)
    avg = (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2.0
    return float(np.clip(avg / 0.28, 0.0, 1.0))   # EAR ~0.28 = fully open

def _smile_score(landmarks, w: int, h: int) -> float:
    """
    Mouth width vs height ratio as smile proxy.
    Wider relative to height → more smile-like.
    """
    ml = landmarks.landmark[MOUTH_LEFT]
    mr = landmarks.landmark[MOUTH_RIGHT]
    mt = landmarks.landmark[MOUTH_TOP]
    mb = landmarks.landmark[MOUTH_BOTTOM]

    mouth_w = abs(mr.x - ml.x) * w
    mouth_h = abs(mb.y - mt.y) * h + 1e-6
    ratio   = mouth_w / mouth_h
    # Typical ratio range: 2 (neutral) → 5 (big smile)
    score   = np.clip((ratio - 2.0) / 3.0, 0.0, 1.0)
    return float(score)

def _roi_sharpness(gray: np.ndarray, bb, h: int, w: int) -> float:
    """Laplacian variance on the face bounding box ROI."""
    x1 = max(0, int(bb.xmin * w))
    y1 = max(0, int(bb.ymin * h))
    x2 = min(w, int((bb.xmin + bb.width)  * w))
    y2 = min(h, int((bb.ymin + bb.height) * h))
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(min(cv2.Laplacian(roi, cv2.CV_64F).var() / 3000.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Group helpers
# ─────────────────────────────────────────────────────────────────────────────

def _background_face_contrast(arr: np.ndarray, gray: np.ndarray,
                               detections, h: int, w: int) -> float:
    """
    Mean brightness of face ROIs vs background — higher = faces pop from bg.
    """
    face_mask = np.zeros((h, w), dtype=bool)
    for det in detections:
        bb = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w))
        y1 = max(0, int(bb.ymin * h))
        x2 = min(w, int((bb.xmin + bb.width)  * w))
        y2 = min(h, int((bb.ymin + bb.height) * h))
        face_mask[y1:y2, x1:x2] = True

    face_px = gray[face_mask]
    bg_px   = gray[~face_mask]
    if face_px.size == 0 or bg_px.size == 0:
        return 0.5

    contrast = abs(float(face_px.mean()) - float(bg_px.mean())) / 255.0
    return float(min(contrast * 2.5, 1.0))   # scale so 0.4 difference → 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Scenery helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dynamic_range(gray: np.ndarray) -> float:
    """
    Spread of histogram: p5 to p95 range normalised to 0-1.
    Wider = more dynamic range captured.
    """
    p5  = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    return min((p95 - p5) / 220.0, 1.0)

def _horizon_level(gray: np.ndarray) -> float:
    """
    Detects dominant horizontal lines via Hough and measures how level they are.
    Returns 1.0 = perfectly level, 0.0 = highly tilted.
    Score is only meaningful for photos that have a clear horizon.
    Falls back to 0.5 for images with no strong lines.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return 0.5

    angles = []
    for line in lines[:20]:   # top 20 strongest lines
        rho, theta = line[0]
        # theta=0 or pi = vertical; theta=pi/2 = horizontal
        angle_from_horiz = abs(theta - np.pi / 2)   # 0 = horizontal
        angles.append(angle_from_horiz)

    if not angles:
        return 0.5

    mean_tilt = float(np.mean(angles))   # radians
    # 0 rad = perfectly level → 1.0; pi/4 rad (~45°) → 0.0
    score = max(0.0, 1.0 - mean_tilt / (np.pi / 4))
    return float(score)

def _color_score(arr: np.ndarray) -> float:
    """
    Combines saturation and colour distribution breadth.
    High saturation + good spread = high score.
    """
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0   # 0-1
    mean_sat  = float(sat.mean())

    # Hue spread: std of hue (0-180 in OpenCV) normalised
    hue = hsv[:, :, 0].astype(np.float32)
    hue_std = float(hue.std()) / 90.0   # 90 = max expected std

    score = 0.6 * mean_sat + 0.4 * min(hue_std, 1.0)
    return float(min(score, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Neutral value fillers (used when photo type doesn't apply)
# ─────────────────────────────────────────────────────────────────────────────

def _neutral_subject() -> dict:
    return {
        "subj_eye_openness":   0.0,
        "subj_smile_score":    0.0,
        "subj_face_sharpness": 0.0,
        "subj_exposure_score": 0.0,
        "subj_composition":    0.0,
        "subj_face_area_ratio":0.0,
    }

def _neutral_group() -> dict:
    return {
        "grp_eyes_open_pct":         0.0,
        "grp_min_eye_openness":      0.0,
        "grp_expression_consistency":0.0,
        "grp_min_face_sharpness":    0.0,
        "grp_face_visibility":       0.0,
        "grp_composition":           0.0,
        "grp_brightness":            0.0,
        "grp_bg_contrast":           0.0,
    }

def _neutral_scenery() -> dict:
    return {
        "scen_global_sharpness": 0.0,
        "scen_exposure_quality": 0.0,
        "scen_dynamic_range":    0.0,
        "scen_horizon_level":    0.0,
        "scen_composition":      0.0,
        "scen_color_score":      0.0,
    }
