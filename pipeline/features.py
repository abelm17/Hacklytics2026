import cv2
import numpy as np
import mediapipe as mp
from scipy.stats import entropy as scipy_entropy

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

def extract_all_features(records: list[dict]) -> list[dict]:
    """Adds feature keys to each record dict in-place."""
    face_det = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    face_mesh_proc = mp_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=5,
        refine_landmarks=True, min_detection_confidence=0.5
    )
    for rec in records:
        arr = rec["array"]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        rec["sharpness"] = _sharpness(gray)
        rec["brightness"] = float(np.mean(gray))
        rec["brightness_var"] = float(np.var(gray))
        rec["contrast"] = _contrast(gray)
        rec["color_entropy"] = _color_entropy(arr)

        face_feats = _face_features(arr, face_det, face_mesh_proc)
        rec.update(face_feats)

    face_det.close()
    face_mesh_proc.close()
    return records

def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _contrast(gray: np.ndarray) -> float:
    return float(gray.std())

def _color_entropy(arr: np.ndarray) -> float:
    entropies = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=64, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropies.append(scipy_entropy(hist))
    return float(np.mean(entropies))

def _face_features(arr: np.ndarray, face_det, face_mesh) -> dict:
    h, w = arr.shape[:2]
    rgb = arr  # already RGB
    results = face_det.process(rgb)

    face_count = 0
    largest_face_ratio = 0.0
    eye_openness = 1.0  # default: assume open if no face
    smile_score = 0.0

    if results.detections:
        face_count = len(results.detections)
        areas = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            area = bb.width * bb.height
            areas.append(area)
        largest_face_ratio = float(max(areas))

        mesh_results = face_mesh.process(rgb)
        if mesh_results.multi_face_landmarks:
            eye_openness = _eye_openness_score(mesh_results.multi_face_landmarks[0], w, h)

    return {
        "face_count": face_count,
        "largest_face_ratio": largest_face_ratio,
        "eye_openness": eye_openness,
        "smile_score": smile_score,
    }

# MediaPipe landmark indices for eye aspect ratio
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def _eye_openness_score(landmarks, w: int, h: int) -> float:
    def ear(indices):
        pts = np.array([[landmarks.landmark[i].x * w,
                         landmarks.landmark[i].y * h] for i in indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    left = ear(LEFT_EYE)
    right = ear(RIGHT_EYE)
    avg = (left + right) / 2
    return float(np.clip(avg / 0.3, 0, 1))
