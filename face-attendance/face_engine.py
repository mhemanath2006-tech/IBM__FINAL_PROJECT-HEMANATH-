"""Face detection / embedding / liveness helpers.

Uses a pretrained model (InsightFace ArcFace). No training happens anywhere:
each student's face becomes a few 512-number embeddings, and a scan is compared
against the logged-in student's own embeddings (1-to-1 verification).
"""
import base64
import json
import os

import cv2
import numpy as np

MIN_FACE_SCORE = float(os.environ.get("MIN_FACE_SCORE", "0.6"))
MIN_FACE_PIXELS = int(os.environ.get("MIN_FACE_PIXELS", "80"))
MAX_SIDE = 800


class FaceError(Exception):
    """A user-facing problem with a frame (no face, several faces, too small...)."""


class InsightEngine:
    def __init__(self):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=os.environ.get("FACE_MODEL", "buffalo_l"),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=(480, 480))

    def analyze(self, img_bgr):
        out = []
        for f in self.app.get(img_bgr):
            out.append(
                {
                    "embedding": np.asarray(f.normed_embedding, dtype=np.float32),
                    "kps": np.asarray(f.kps, dtype=np.float32),
                    "score": float(f.det_score),
                    "bbox": np.asarray(f.bbox, dtype=np.float32),
                }
            )
        return out


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = InsightEngine()
    return _engine


def set_engine(engine):
    """Used by tests to plug in a fake engine."""
    global _engine
    _engine = engine


def decode_image(data, max_bytes=2_500_000):
    if not isinstance(data, str) or not data:
        raise FaceError("Missing image.")
    if "," in data[:100]:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        raise FaceError("Image could not be read.")
    if len(raw) > max_bytes:
        raise FaceError("Image is too large.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FaceError("Image could not be read.")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def single_face(engine, img):
    """Return the one usable face in the frame or raise FaceError."""
    faces = [f for f in engine.analyze(img) if f["score"] >= MIN_FACE_SCORE]
    if not faces:
        raise FaceError("No face found. Face the camera in good light.")
    if len(faces) > 1:
        raise FaceError("More than one face in the frame.")
    f = faces[0]
    x1, y1, x2, y2 = f["bbox"]
    if min(x2 - x1, y2 - y1) < MIN_FACE_PIXELS:
        raise FaceError("Face is too small. Move closer to the camera.")
    return f


def yaw_ratio(kps):
    """Horizontal nose offset from the eye midpoint, relative to eye distance.

    Changes as a real 3D head turns; a flat photo moved in front of the camera
    barely changes it. kps order: left eye, right eye, nose, mouth corners (image
    coordinates, unmirrored). Person turning to THEIR left moves the nose toward
    larger x, so the ratio increases.
    """
    le, re, nose = kps[0], kps[1], kps[2]
    eye_mid = (le + re) / 2.0
    dist = float(np.linalg.norm(re - le)) or 1.0
    return float((nose[0] - eye_mid[0]) / dist)


def encode_templates(embeddings):
    return json.dumps(
        [base64.b64encode(np.asarray(e, np.float32).tobytes()).decode() for e in embeddings]
    )


def decode_templates(text):
    return np.stack(
        [np.frombuffer(base64.b64decode(s), dtype=np.float32) for s in json.loads(text)]
    )


def best_similarity(embedding, templates):
    """Cosine similarity (embeddings are L2-normalised) against the best stored one."""
    return float(np.max(templates @ np.asarray(embedding, np.float32)))
