"""
Face detection and recognition engine built on InsightFace (ArcFace).

Model selection (set in config.py)
-----------------------------------
buffalo_sc  — small / fast model, ~4× faster than buffalo_l.
              Suitable for real-time detection at 640×480.
buffalo_l   — large / accurate model, better for low-quality cameras
              or side-profile faces. Use when FPS is not the priority.

Speed tips
----------
- Keep det_size at (320, 320) or smaller for faster detection.
- Call detect() only every N frames (controlled by CONFIG.detection_interval).
- Use OpenVINO provider if an Intel CPU/iGPU is available.
"""

import logging
import os
import threading

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from core.config import CONFIG

logger = logging.getLogger(__name__)


class FaceEngine:
    def __init__(self):
        self._app = FaceAnalysis(
            name=CONFIG.model_name,
            providers=CONFIG.providers,
        )
        self._app.prepare(ctx_id=0, det_size=CONFIG.det_size)

        self._lock = threading.Lock()
        self._embeddings: list[np.ndarray] = []
        self._ids: list[str] = []

        os.makedirs(CONFIG.people_dir, exist_ok=True)
        self.reload()

    # ── Face database ────────────────────────────────────────────────────────

    def reload(self):
        """Re-load all face embeddings from disk.

        Prefers embeddings.npy (multi-angle, accumulated at runtime) over
        re-computing from photo.jpg.  Falls back to photo.jpg when the .npy
        file is absent (first run or legacy data).
        """
        self._embeddings = []
        self._ids = []

        folders = sorted(
            [f for f in os.listdir(CONFIG.people_dir)
             if os.path.isdir(os.path.join(CONFIG.people_dir, f)) and f.isdigit()],
            key=int,
        )
        for pid in folders:
            emb_path   = os.path.join(CONFIG.people_dir, pid, 'embeddings.npy')
            photo_path = os.path.join(CONFIG.people_dir, pid, 'photo.jpg')

            if os.path.exists(emb_path):
                embs = np.load(emb_path)
                for emb in embs:
                    self._embeddings.append(emb)
                    self._ids.append(pid)
            elif os.path.exists(photo_path):
                img = cv2.imread(photo_path)
                if img is None:
                    continue
                faces = self._app.get(self._preprocess(img))
                if not faces:
                    continue
                emb = faces[0].embedding
                self._embeddings.append(emb)
                self._ids.append(pid)
                np.save(emb_path, np.array([emb]))  # persist for future startups

        logger.info(
            'Loaded %d face embeddings for %d people',
            len(self._embeddings), len(set(self._ids))
        )

    def save_person(self, embeddings: list, face_img: np.ndarray) -> str:
        """Save a new person with all accumulated angle embeddings. Returns new ID."""
        with self._lock:
            return self._save_person(embeddings, face_img)

    def _save_person(self, embeddings: list, face_img: np.ndarray) -> str:
        existing = [int(f) for f in os.listdir(CONFIG.people_dir) if f.isdigit()]
        pid = str(max(existing) + 1) if existing else '1'

        person_dir = os.path.join(CONFIG.people_dir, pid)
        os.makedirs(person_dir, exist_ok=True)
        cv2.imwrite(os.path.join(person_dir, 'photo.jpg'), face_img)
        np.save(os.path.join(person_dir, 'embeddings.npy'), np.array(embeddings))

        for emb in embeddings:
            self._embeddings.append(emb)
            self._ids.append(pid)
        logger.info('Registered new person: %s  (%d angle embeddings)',
                    pid, len(embeddings))
        return pid



    def remove_person(self, pid: str):
        """Remove ALL embeddings (all angles) for a person."""
        indices = [i for i, id_ in enumerate(self._ids) if id_ == pid]
        for idx in reversed(indices):
            self._ids.pop(idx)
            self._embeddings.pop(idx)

    def add_angle(self, pid: str, embedding: np.ndarray) -> bool:
        """Store a new viewing-angle embedding for a registered person.

        Skips if the person already has CONFIG.max_face_angles embeddings, or if
        the new embedding is too similar to an existing one (same angle).
        Returns True when a new angle is actually stored.
        """
        existing = [e for e, i in zip(self._embeddings, self._ids) if i == pid]
        if len(existing) >= CONFIG.max_face_angles:
            return False
        if existing and max(self.cosine_sim(embedding, e) for e in existing) > CONFIG.angle_diversity_threshold:
            return False
        self._embeddings.append(embedding)
        self._ids.append(pid)
        self._persist_angles(pid)
        return True

    def _persist_angles(self, pid: str):
        """Write all stored embeddings for a person to embeddings.npy."""
        embs = np.array([e for e, i in zip(self._embeddings, self._ids) if i == pid])
        np.save(os.path.join(CONFIG.people_dir, pid, 'embeddings.npy'), embs)

    # ── Detection & matching ─────────────────────────────────────────────────

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """Gamma correction for dark frames + CLAHE for local contrast."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(gray.mean())
        if mean_lum < 80:
            gamma = 1.0 + (80.0 - mean_lum) / 100.0
            lut = (np.linspace(0.0, 1.0, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
            frame = cv2.LUT(frame, lut)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def blur_score(crop: np.ndarray) -> float:
        """Laplacian variance — higher means sharper."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def frontal_score(face) -> float:
        """1.0 = full frontal, 0.0 = pure profile. Uses 5-pt landmarks."""
        kps = getattr(face, 'kps', None)
        if kps is None or len(kps) < 3:
            return 1.0
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_dist = abs(right_eye[0] - left_eye[0]) + 1e-6
        return float(max(0.0, 1.0 - abs(nose[0] - eye_center_x) / eye_dist))

    def detect(self, frame: np.ndarray) -> list:
        """Detect faces with lighting correction and quality filtering."""
        faces = self._app.get(self._preprocess(frame))
        return [
            f for f in faces
            if float(getattr(f, 'det_score', 1.0)) >= CONFIG.min_det_score
            and (f.bbox[2] - f.bbox[0]) >= CONFIG.min_face_size
        ]

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def find_person(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """
        Match per-person (max over each person's angles), then require a margin
        over the second-best candidate to reject ambiguous identifications.
        """
        if not self._embeddings:
            return None, 0.0

        pid_best: dict[str, float] = {}
        for emb, pid in zip(self._embeddings, self._ids):
            s = self.cosine_sim(embedding, emb)
            if s > pid_best.get(pid, -1.0):
                pid_best[pid] = s

        ranked = sorted(pid_best.items(), key=lambda x: x[1], reverse=True)
        best_pid, best_score = ranked[0]

        if best_score < CONFIG.similarity_threshold:
            return None, best_score

        # Reject if too close to second-best (ambiguous match)
        if len(ranked) >= 2:
            second_score = ranked[1][1]
            if second_score >= CONFIG.similarity_threshold * 0.85 \
                    and (best_score - second_score) < CONFIG.id_margin:
                return None, best_score

        return best_pid, best_score

    @property
    def known_count(self) -> int:
        return len(self._ids)


engine = FaceEngine()
