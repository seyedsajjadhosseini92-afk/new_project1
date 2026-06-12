# Security System — Face Recognition Dashboard

Real-time face detection, person tracking, and entry/exit logging with a modern dark-themed dashboard.

---

## Features

- **Real-time face recognition** using InsightFace ArcFace (buffalo_sc for speed)
- **Entry / Exit detection** via a virtual door line that divides the camera frame
- **Auto-registration** of new people after a configurable number of confirmation frames
- **Modern dashboard** built with CustomTkinter — dark theme, live camera feed, person cards
- **Visit history & charts** — per-person visit log and a 14-day bar chart
- **Rename / Delete** people directly from the dashboard
- **Thread-safe SQLite** database with automatic migration from v1 schema

---

## Project Structure

```
new_project1/
├── core/
│   ├── config.py       — All tunable settings (model, threshold, paths …)
│   ├── database.py     — Thread-safe SQLite layer (people, visits, seen)
│   ├── face_engine.py  — InsightFace wrapper (detect, match, save, remove)
│   └── tracker.py      — Multi-person tracker with door-crossing logic
├── ui/
│   └── app.py          — CustomTkinter dashboard (3-panel layout)
├── people/             — Auto-created; one sub-folder per registered person
│   └── 1/
│       ├── photo.jpg
│       └── info.txt
├── data.db             — SQLite database (auto-created)
├── main.py             — Entry point: python main.py
├── requirements.txt
└── .vscode/
    ├── settings.json
    └── launch.json
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Intel CPU / iGPU?** Uncomment the `openvino` line in `requirements.txt` for faster inference via OpenVINO.

### 2. Run

```bash
python main.py
```

Or press **F5** in VS Code (select "Run Dashboard" configuration).

---

## Configuration

Edit `core/config.py` — no other files need to change.

| Setting | Default | Description |
|---------|---------|-------------|
| `model_name` | `buffalo_sc` | `buffalo_sc` = fast; `buffalo_l` = more accurate |
| `det_size` | `(320, 320)` | Smaller → faster detection |
| `similarity_threshold` | `0.45` | Cosine similarity cutoff for face matching |
| `confirm_frames` | `20` | Frames before a new face is registered |
| `detection_interval` | `3` | Run detection every N frames (higher → faster UI) |
| `camera_index` | `0` | Webcam index (0 = default camera) |

---

## How Door Detection Works

```
Camera frame:

LEFT side           DOOR LINE           RIGHT side
(ENTRY direction)   (centre)            (EXIT direction)

Person crosses:  RIGHT → LEFT  =  ENTRY logged
                 LEFT  → RIGHT =  EXIT  logged
```

A crossing is only logged once per direction change — the person must
return to their original side before the next event is recorded.

---

## Database Schema

```sql
people  (id TEXT PK, name TEXT, added TEXT)
visits  (id INT PK, person_id TEXT, action TEXT, time TEXT)
seen    (id INT PK, person_id TEXT, time TEXT)
```

- **visits** — records every ENTRY / EXIT event
- **seen** — records that a person was visible (logged every 30 s)

---

## Speed Tips

1. Use `buffalo_sc` (default) — 4× faster than `buffalo_l`
2. Increase `detection_interval` to 4 or 5 for lower-end CPUs
3. Reduce `det_size` to `(256, 256)` for even faster detection
4. Use a lower camera resolution (320×240) if FPS is still low
5. Install OpenVINO (`pip install openvino`) on Intel hardware
