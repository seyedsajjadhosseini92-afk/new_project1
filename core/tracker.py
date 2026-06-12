"""
Multi-person tracker — single camera with door-line ENTRY / EXIT.

How it works
------------
A vertical door line (CONFIG.door_position_pct × frame width) divides
the frame into two zones.

1. Each detected face is matched against registered embeddings.
   The same person can have up to CONFIG.max_face_angles stored embeddings
   (accumulated automatically from different viewing angles for better accuracy).
2. Unmatched faces get a temporary 't_N' ID; after CONFIG.confirm_frames
   consecutive frames the person is registered.
3. When a registered person crosses the door line, the direction determines
   ENTRY (right→left) or EXIT (left→right) and the event is logged.
4. Tracks not seen for CONFIG.lost_timeout_sec are pruned.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import CONFIG
from core.database import db
from core.face_engine import engine


@dataclass
class _Track:
    pid:            str
    embedding:      np.ndarray
    x:              int
    side:           str           # 'L' or 'R' relative to door line
    first_seen:     datetime = field(default_factory=datetime.now)
    last_seen:      datetime = field(default_factory=datetime.now)
    crossed:        bool     = False
    confirm_frames: int      = 0
    # Multi-angle accumulation during confirmation phase
    angle_embs:     list     = field(default_factory=list)
    best_score:     float    = 0.0
    best_crop:      object   = None   # Optional[np.ndarray]


# (bbox, pid, is_confirmed, frames_remaining, match_score)
DrawData = Tuple[np.ndarray, str, bool, int, float]


class Tracker:
    def __init__(self):
        self._lock           = threading.Lock()
        self._tracks:        Dict[str, _Track]   = {}
        self._last_seen_log: Dict[str, datetime] = {}
        self._temp_counter:  int                 = 0
        self.active_pids:    List[str]           = []

    def update(
        self,
        faces:  list,
        frame:  np.ndarray,
        door_x: int,
    ) -> List[DrawData]:
        with self._lock:
            return self._update(faces, frame, door_x)

    def _update(self, faces, frame, door_x) -> List[DrawData]:
        now          = datetime.now()
        current_ids: List[str]     = []
        draw_data:   List[DrawData] = []
        h, w = frame.shape[:2]

        for face in faces:
            emb:  np.ndarray = face.embedding
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            cx   = (x1 + x2) // 2
            side = 'R' if cx > door_x else 'L'

            # ── Match against registered database ─────────────────────────────
            pid, match_score = engine.find_person(emb)

            # ── Match against unregistered temp tracks (best match + proximity) ──
            if pid is None:
                best_tid, best_sim = None, CONFIG.similarity_threshold * 0.85
                for tid, trk in self._tracks.items():
                    if not tid.startswith('t_'):
                        continue
                    sim = engine.cosine_sim(emb, trk.embedding)
                    if sim > best_sim and abs(cx - trk.x) < w * 0.25:
                        best_sim = sim
                        best_tid = tid
                if best_tid:
                    pid = best_tid

            # ── Brand-new face → create temp track ───────────────────────────
            if pid is None:
                pid = f't_{self._temp_counter}'
                self._temp_counter += 1
                self._tracks[pid] = _Track(pid=pid, embedding=emb, x=cx, side=side)

            if pid not in self._tracks:
                self._tracks[pid] = _Track(pid=pid, embedding=emb, x=cx, side=side)

            trk           = self._tracks[pid]
            trk.embedding = emb
            trk.x         = cx
            trk.last_seen = now

            # ── Unconfirmed temp person ───────────────────────────────────────
            if pid.startswith('t_'):
                trk.confirm_frames += 1
                rem = max(0, CONFIG.confirm_frames - trk.confirm_frames)
                current_ids.append(pid)

                # Accumulate diverse, frontal embeddings from different angles
                frontal = engine.frontal_score(face)
                det_score = float(getattr(face, 'det_score', 1.0))
                if not trk.angle_embs:
                    trk.angle_embs.append(emb.copy())
                elif (len(trk.angle_embs) < CONFIG.max_face_angles
                        and frontal >= CONFIG.min_frontal_score
                        and max(engine.cosine_sim(emb, e)
                                for e in trk.angle_embs) < 0.85):
                    trk.angle_embs.append(emb.copy())

                # Keep the highest combined quality (sharpness × confidence × frontal)
                candidate = frame[y1c:y2c, x1c:x2c]
                if candidate.size > 0:
                    sharpness = engine.blur_score(candidate)
                    quality = det_score * min(sharpness / 200.0, 1.0) * frontal
                    if quality > trk.best_score and sharpness >= CONFIG.blur_threshold:
                        trk.best_score = quality
                        trk.best_crop  = candidate.copy()

                if trk.confirm_frames >= CONFIG.confirm_frames:
                    embs  = trk.angle_embs if trk.angle_embs else [emb]
                    photo = trk.best_crop  if trk.best_crop is not None \
                            else frame[y1c:y2c, x1c:x2c]
                    if photo.size > 0:
                        new_pid = engine.save_person(embs, photo)
                        db.add_person(new_pid)
                        del self._tracks[pid]
                        current_ids.remove(pid)
                        self._tracks[new_pid] = _Track(
                            pid=new_pid, embedding=emb, x=cx, side=side)
                        current_ids.append(new_pid)
                        draw_data.append((bbox, new_pid, True, 0, 0.0))
                        continue

                draw_data.append((bbox, pid, False, rem, 0.0))
                continue

            # ── Confirmed person: accumulate angle embedding ──────────────────
            engine.add_angle(pid, emb)

            # ── Door-crossing detection ───────────────────────────────────────
            if trk.side != side and not trk.crossed:
                action = 'ENTRY' if trk.side == 'R' and side == 'L' else 'EXIT'
                db.log_visit(pid, action)
                trk.crossed = True
            elif trk.side == side:
                trk.crossed = False
            trk.side = side

            # ── Periodic seen log ─────────────────────────────────────────────
            last_log = self._last_seen_log.get(pid)
            if last_log is None or \
                    (now - last_log).total_seconds() >= CONFIG.seen_interval_sec:
                db.log_seen(pid)
                self._last_seen_log[pid] = now

            current_ids.append(pid)
            draw_data.append((bbox, pid, True, 0, match_score))

        # ── Prune stale tracks ────────────────────────────────────────────────
        stale = [
            pid for pid, trk in self._tracks.items()
            if pid not in current_ids
            and (now - trk.last_seen).total_seconds() > CONFIG.lost_timeout_sec
        ]
        for pid in stale:
            del self._tracks[pid]

        self.active_pids = [p for p in current_ids if not p.startswith('t_')]
        return draw_data
