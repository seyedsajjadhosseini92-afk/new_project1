"""
Person body detection (OpenCV HOG) + Re-ID (HSV colour histogram).

No additional packages required — uses only opencv-python and numpy,
which are already installed for the face-recognition pipeline.

Detection
---------
OpenCV's built-in HOG + SVM person detector.  Works on any machine
without extra downloads or internet access.

Re-ID features
--------------
128-dim unit-normalised HSV histogram split into upper / lower body halves.
Upper half captures shirt/jacket colour; lower half captures trousers colour.
Cosine similarity is used for matching, same as face recognition.
"""

import logging
import os

import cv2
import numpy as np

from core.config import CONFIG

logger = logging.getLogger(__name__)


class BodyEngine:
    def __init__(self):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info('HOG person detector ready (no extra packages needed)')

        self._embeddings: list[np.ndarray] = []
        self._ids:        list[str]        = []
        self.reload()

    # ── Embedding helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _half_hist(img: np.ndarray) -> np.ndarray:
        """32-bin H + 32-bin S histogram for one body region."""
        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        return np.concatenate([h_hist, s_hist])

    def _embed(self, crop: np.ndarray) -> np.ndarray | None:
        """128-dim unit-normalised HSV histogram (upper + lower body halves)."""
        if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 16:
            return None
        crop = cv2.resize(crop, (64, 128))
        mid  = crop.shape[0] // 2
        feat = np.concatenate([
            self._half_hist(crop[:mid]),   # upper body — shirt / jacket colour
            self._half_hist(crop[mid:]),   # lower body — trousers colour
        ])
        norm = np.linalg.norm(feat) + 1e-9
        return feat / norm

    @staticmethod
    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # ── Persistence ───────────────────────────────────────────────────────────

    def reload(self):
        self._embeddings, self._ids = [], []
        if not os.path.exists(CONFIG.people_dir):
            return
        folders = sorted(
            [f for f in os.listdir(CONFIG.people_dir)
             if f.isdigit() and
             os.path.isdir(os.path.join(CONFIG.people_dir, f))],
            key=int,
        )
        for pid in folders:
            path = os.path.join(CONFIG.people_dir, pid, 'body.jpg')
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            emb = self._embed(img)
            if emb is not None:
                self._embeddings.append(emb)
                self._ids.append(pid)
        logger.info('Loaded %d body Re-ID embeddings', len(self._ids))

    def save_body(self, pid: str, body_crop: np.ndarray):
        path = os.path.join(CONFIG.people_dir, pid, 'body.jpg')
        cv2.imwrite(path, body_crop)
        emb = self._embed(body_crop)
        if emb is None:
            return
        if pid in self._ids:
            self._embeddings[self._ids.index(pid)] = emb
        else:
            self._embeddings.append(emb)
            self._ids.append(pid)

    def remove_person(self, pid: str):
        if pid in self._ids:
            idx = self._ids.index(pid)
            self._ids.pop(idx)
            self._embeddings.pop(idx)

    def has_body(self, pid: str) -> bool:
        return pid in self._ids

    # ── Detection & matching ──────────────────────────────────────────────────

    def detect_persons(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return list of [x1,y1,x2,y2] for every detected person."""
        detections, _ = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        if len(detections) == 0:
            return []
        return [np.array([x, y, x + w, y + h]) for x, y, w, h in detections]

    def find_person(self, body_crop: np.ndarray) -> tuple[str | None, float]:
        emb = self._embed(body_crop)
        if emb is None or not self._embeddings:
            return None, 0.0
        scores = [self._sim(emb, e) for e in self._embeddings]
        best   = int(np.argmax(scores))
        score  = scores[best]
        if score >= CONFIG.body_similarity_threshold:
            return self._ids[best], score
        return None, score


body_engine = BodyEngine()
