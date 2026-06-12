"""
Central configuration for the Security System.
Edit values here — no need to touch other files.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ── Camera ───────────────────────────────────────────────────────────────
    camera_index:      int = 0
    resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (320, 240), (480, 360), (640, 480), (960, 720), (1280, 720)
    ])
    default_resolution: int = 2   # index → 640×480

    # ── Face Recognition ─────────────────────────────────────────────────────
    # 'buffalo_sc' → fast, real-time (recommended)
    # 'buffalo_l'  → more accurate, better on profiles — slower
    model_name: str = 'buffalo_l'
    providers: List[str] = field(default_factory=lambda: [
        'OpenVINOExecutionProvider', 'CPUExecutionProvider'
    ])
    det_size: Tuple[int, int] = (640, 640)
    similarity_threshold: float = 0.52
    max_face_angles:      int   = 12    # max stored embeddings per person

    # Minimum detection confidence — faces below this score are ignored.
    min_det_score: float = 0.70
    # Minimum face width in pixels — ignore very small/distant faces.
    min_face_size: int = 50
    # Laplacian variance below this = blurry frame, skip for registration.
    blur_threshold: float = 60.0
    # Max cosine similarity between stored angles — lower = more diverse angles captured.
    angle_diversity_threshold: float = 0.80
    # Minimum margin between best and second-best match to avoid ambiguous IDs.
    id_margin: float = 0.04
    # Minimum frontal score (0=profile, 1=frontal) to store angle embeddings.
    min_frontal_score: float = 0.50

    # Frames a new face must stay visible before being registered.
    # At detection_interval=5 and 30fps → ~6 detections/sec → 30 frames ≈ 5 sec.
    confirm_frames: int = 30

    # ── Tracking ─────────────────────────────────────────────────────────────
    lost_timeout_sec:  int = 5    # seconds before a missing track is pruned
    seen_interval_sec: int = 30   # seconds between "seen" log entries

    # Door line position as a fraction of frame width (0.0 = left, 1.0 = right).
    # Change this value to match where your door actually is in the camera frame.
    door_position_pct: float = 0.5

    # Skip N-1 frames between face detections (higher = faster UI, less CPU load).
    detection_interval: int = 5

    # ── Paths ─────────────────────────────────────────────────────────────────
    people_dir:    str = os.path.join(BASE_DIR, 'people')
    db_path:       str = os.path.join(BASE_DIR, 'data.db')
    watchlist_dir: str = os.path.join(BASE_DIR, 'watchlist')

    # ── Deduplication ─────────────────────────────────────────────────────────
    # If two person records share any embedding with similarity >= this value,
    # they are treated as the same person and merged (one record deleted).
    dedup_threshold: float = 0.58

    # ── Watchlist alarm ───────────────────────────────────────────────────────
    # Cosine similarity required to trigger a watchlist alarm.
    watchlist_alarm_threshold: float = 0.52

    # ── UI ────────────────────────────────────────────────────────────────────
    camera_refresh_ms: int = 30     # display update interval (~33 fps)
    stats_refresh_ms:  int = 5000   # people list + stats refresh interval


CONFIG = Config()
