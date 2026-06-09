"""
app.py
Streamlit web UI for the NumPlate Detection pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NumPlate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 18px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    [data-testid="metric-container"]:hover { transform: translateY(-2px); }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 14px;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(102, 126, 234, 0.9);
    }

    /* Data table */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 4px; }

    /* Headings */
    h1 { background: linear-gradient(135deg, #667eea, #f093fb);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { color: #cbd5e0; font-weight: 600; }

    /* Status badge */
    .status-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-running  { background: rgba(72,199,142,0.2); color: #48c78e; border: 1px solid #48c78e; }
    .badge-idle     { background: rgba(100,116,139,0.2); color: #94a3b8; border: 1px solid #475569; }
    .badge-done     { background: rgba(99,179,237,0.2);  color: #63b3ed; border: 1px solid #63b3ed; }
    .badge-error    { background: rgba(252,129,74,0.2);  color: #fc814a; border: 1px solid #fc814a; }

    /* Plate tag */
    .plate-tag {
        font-family: 'Courier New', monospace;
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 8px; padding: 3px 10px;
        font-weight: 700; font-size: 0.95rem; color: #a78bfa;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Session gate card */
    .session-gate {
        max-width: 520px;
        margin: 80px auto;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(102,126,234,0.35);
        border-radius: 20px;
        padding: 44px 48px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 40px rgba(102,126,234,0.18);
    }
    .session-gate h2 {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .session-gate p { color: #94a3b8; font-size: 0.9rem; margin-bottom: 24px; }
    .session-chip {
        display: inline-block;
        background: rgba(102,126,234,0.15);
        border: 1px solid rgba(102,126,234,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #a78bfa;
        font-weight: 600;
        cursor: pointer;
        margin: 3px;
        transition: background 0.15s;
    }
    .session-chip:hover { background: rgba(102,126,234,0.3); }
    .active-session-banner {
        background: linear-gradient(90deg, rgba(102,126,234,0.18), rgba(118,75,162,0.12));
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.85rem;
        color: #c4b5fd;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
SESSIONS_DIR = ROOT / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)


def _session_db_uri(session_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_name).strip("_")
    return f"sqlite:///{SESSIONS_DIR / safe}.db"


def _list_sessions() -> list[str]:
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.db"))


CONFIG_PATH = ROOT / "configs" / "config.yaml"


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_device(prefer_gpu: bool) -> str:
    if prefer_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
        except ImportError:
            pass
    return "cpu"


def _load_pipeline_components(cfg: dict, device: str):
    """Import and instantiate all pipeline components (cached per session)."""
    sys.path.insert(0, str(ROOT))

    from models.ocr_engine import build_ocr_engine
    from models.plate_detector import PlateDetector
    from models.vehicle_detector import VehicleDetector
    from processing.postprocessor import PlateValidator, TrackBuffer
    from processing.preprocessor import Preprocessor
    from storage.database import DatabaseManager
    from tracking.tracker import DeepSORTTracker

    vd_cfg = cfg.get("vehicle_detector", {})
    vehicle_detector = VehicleDetector(
        weights=vd_cfg.get("weights", "yolov8n.pt"),
        confidence=vd_cfg.get("confidence", 0.45),
        iou_threshold=vd_cfg.get("iou_threshold", 0.50),
        device=device,
        target_classes=vd_cfg.get("target_classes", [2, 3, 5, 7]),
        imgsz=vd_cfg.get("imgsz", 640),
    )

    pd_cfg = cfg.get("plate_detector", {})
    plate_detector = PlateDetector(
        weights=pd_cfg.get("weights", "models/weights/plate_detector-3/weights/plate_detector_best.pt"),
        fallback_weights=pd_cfg.get("fallback_weights", "yolov8n.pt"),
        confidence=pd_cfg.get("confidence", 0.30),
        iou_threshold=pd_cfg.get("iou_threshold", 0.45),
        device=device,
        imgsz=pd_cfg.get("imgsz", 320),
    )

    ocr_cfg = cfg.get("ocr", {"backend": "easyocr", "languages": ["en"], "gpu": False})
    ocr_engine = build_ocr_engine(ocr_cfg)

    trk_cfg = cfg.get("tracker", {})
    tracker = DeepSORTTracker(
        max_age=trk_cfg.get("max_age", 30),
        n_init=trk_cfg.get("n_init", 1),
        max_cosine_distance=trk_cfg.get("max_cosine_distance", 0.4),
        nn_budget=trk_cfg.get("nn_budget", 100),
    )

    pre_cfg = cfg.get("preprocessing", {})
    clahe_cfg = pre_cfg.get("clahe", {})
    preprocessor = Preprocessor(
        clahe_enabled=clahe_cfg.get("enabled", True),
        clip_limit=clahe_cfg.get("clip_limit", 3.0),
        tile_grid_size=tuple(clahe_cfg.get("tile_grid_size", [8, 8])),
        plate_crop_ratio=pre_cfg.get("plate_crop_ratio", 0.85),
        min_plate_height=pre_cfg.get("min_plate_height", 64),
        min_plate_width=pre_cfg.get("min_plate_width", 100),
    )

    post_cfg = cfg.get("postprocessing", {})
    validator = PlateValidator(
        pattern=post_cfg.get("regex_pattern", r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")
    )
    track_buffer = TrackBuffer(
        validator=validator,
        strategy=post_cfg.get("dedup_strategy", "majority_vote"),
        min_reads=post_cfg.get("min_reads_before_flush", 1),
    )

    store_cfg = cfg.get("storage", {})
    db = DatabaseManager(
        db_uri=store_cfg.get("db_uri", "sqlite:///detections.db"),
        auto_create_tables=store_cfg.get("auto_create_tables", True),
    )

    return vehicle_detector, plate_detector, ocr_engine, tracker, preprocessor, track_buffer, db


def _draw_overlay(
    frame: np.ndarray,
    tracks,
    plate_labels: dict,      # tid -> validated plate text
    plate_confs: dict,
    plate_boxes: dict,       # tid -> [(px1,py1,px2,py2)] in frame coords
    raw_ocr_labels: dict,    # tid -> raw OCR text (even if regex-rejected)
) -> np.ndarray:
    """Draw vehicle boxes + plate boxes + plate text on a frame."""
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        tid = track.track_id
        label = plate_labels.get(tid, "")
        conf  = plate_confs.get(tid, 0.0)
        raw   = raw_ocr_labels.get(tid, "")

        # Vehicle box — bright green, thick border for visibility
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 128), 4)
        # Inner highlight for extra contrast
        cv2.rectangle(annotated, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (0, 200, 80), 1)

        # Plate sub-boxes — orange (validated) or yellow (raw)
        box_color = (0, 165, 255) if label else (0, 220, 255)
        for (bx1, by1, bx2, by2) in plate_boxes.get(tid, []):
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2)

        # Label: show validated plate or raw OCR in different colours
        font_scale = 0.65
        thickness = 2
        if label:
            text = f"{label}  {conf:.0%}"
            bg_color = (0, 255, 128)   # green — validated
            fg_color = (0, 0, 0)
        elif raw:
            text = f"~{raw}"           # tilde prefix = raw/unvalidated
            bg_color = (0, 200, 255)   # yellow — raw OCR, failed regex
            fg_color = (0, 0, 0)
        else:
            text = f"#{tid}"
            bg_color = (0, 255, 128)
            fg_color = (0, 0, 0)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), bg_color, -1)
        cv2.putText(annotated, text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg_color, thickness)

    return annotated


def _fetch_db_df(db) -> pd.DataFrame:
    """Pull all detections into a formatted DataFrame."""
    detections = db.get_all_detections()  # now returns plain dicts
    if not detections:
        return pd.DataFrame(columns=["Plate", "Confidence", "Vehicle", "Frame", "Time", "Reads"])
    rows = []
    for d in detections:
        rows.append({
            "Plate": d["plate_number"],
            "Confidence": f"{d['confidence']:.1%}",
            "Vehicle": d["vehicle_class"] or "vehicle",
            "Frame": d["frame_no"],
            "Time": str(d["first_seen_time"])[:19] if d["first_seen_time"] else "—",
            "Reads": d["total_reads"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Session gate — must set a session name before anything else renders
# ---------------------------------------------------------------------------

if "session_name" not in st.session_state:
    st.session_state["session_name"] = ""

if not st.session_state["session_name"]:
    st.markdown(
        """
        <div class="session-gate">
            <h2>🚗 NumPlate Detection</h2>
            <p>Enter a session name to begin. Each session stores its detections separately,
            so you can switch between runs without data mixing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        gate_col = st.columns([1, 2, 1])[1]  # centre column
        with gate_col:
            new_name = st.text_input(
                "Session name",
                placeholder="e.g. highway_cam_01",
                label_visibility="collapsed",
                key="_gate_name_input",
            )
            existing = _list_sessions()
            if existing:
                st.markdown("**Resume an existing session:**")
                for sname in existing:
                    if st.button(f"📂 {sname}", key=f"_resume_{sname}", use_container_width=True):
                        st.session_state["session_name"] = sname
                        st.rerun()
            if st.button("▶️  Start session", use_container_width=True, type="primary",
                         disabled=not new_name.strip()):
                st.session_state["session_name"] = new_name.strip()
                st.rerun()
    st.stop()

# Active session
_SESSION_NAME = st.session_state["session_name"]
_SESSION_DB_URI = _session_db_uri(_SESSION_NAME)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Active session banner
    st.markdown(
        f"<div class='active-session-banner'>🗂️ Session: <b>{_SESSION_NAME}</b></div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Switch session", use_container_width=True):
        st.session_state["session_name"] = ""
        st.rerun()

    st.markdown("## ⚙️ Configuration")
    st.divider()

    cfg = _load_yaml(CONFIG_PATH)

    vehicle_conf = st.slider("Vehicle confidence", 0.1, 0.9,
                             float(cfg.get("vehicle_detector", {}).get("confidence", 0.45)), 0.05)
    plate_conf   = st.slider("Plate confidence",   0.1, 0.9,
                             float(cfg.get("plate_detector", {}).get("confidence", 0.30)), 0.05)
    skip_frames  = st.slider("Skip frames (process every Nth)", 1, 10,
                             int(cfg.get("source", {}).get("skip_frames", 2)))
    min_reads    = st.slider("Min OCR reads before commit", 1, 5,
                             int(cfg.get("postprocessing", {}).get("min_reads_before_flush", 1)))
    use_gpu      = st.toggle("Use GPU (if available)", value=False)
    disable_regex = st.toggle("⚠️ Disable plate regex filter", value=False,
                              help="Turn off to accept any OCR output as a plate number. "
                                   "Useful if your plates don't match the Indian format.")

    st.divider()
    st.markdown("### 📊 Session Database")
    _db_file = SESSIONS_DIR / f"{_SESSION_NAME}.db"
    st.caption(f"`sessions/{_SESSION_NAME}.db`")

    if st.button("🗑️ Clear this session's data", use_container_width=True):
        if _db_file.exists():
            import gc
            gc.collect()
            try:
                _db_file.unlink()
                st.success("Session data cleared.")
                st.rerun()
            except PermissionError as e:
                st.error(
                    f"Could not delete the database file because it is locked by another process/connection. "
                    f"Please try switching sessions or restarting the application.\n\nError: {e}"
                )

    st.divider()
    st.markdown(
        "<small style='color:#64748b'>NumPlate Detection Pipeline<br>"
        "YOLOv8 · DeepSORT · EasyOCR</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title(f"🚗 NumPlate Detection — {_SESSION_NAME}")
st.markdown(
    f"Upload a video to detect vehicles and read license plates in real-time. "
    f"Results are stored in <b>session '{_SESSION_NAME}'</b> only.",
    unsafe_allow_html=True,
)

# Upload area
uploaded = st.file_uploader(
    "📤 Upload a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Supported formats: MP4, AVI, MOV, MKV, WebM",
)

# Two-column layout: video | database
col_vid, col_db = st.columns([5, 2], gap="large")

with col_vid:
    st.markdown("### 🎬 Live Detection Feed")
    frame_placeholder = st.empty()
    progress_bar      = st.progress(0)
    status_placeholder = st.empty()

with col_db:
    st.markdown("### 🗄️ Detected Plates")
    stats_cols = st.columns(3)
    metric_vehicles  = stats_cols[0].empty()
    metric_plates    = stats_cols[1].empty()
    metric_fps       = stats_cols[2].empty()
    st.divider()
    table_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Initial DB display — scoped to current session
# ---------------------------------------------------------------------------

try:
    from storage.database import DatabaseManager as _DBM
    _init_db = _DBM(db_uri=_SESSION_DB_URI, auto_create_tables=True)
    try:
        init_df = _fetch_db_df(_init_db)
    finally:
        _init_db.close()
except Exception:
    init_df = pd.DataFrame()

if not init_df.empty:
    table_placeholder.dataframe(
        init_df, use_container_width=True, hide_index=True,
        column_config={"Confidence": st.column_config.TextColumn("Confidence 🎯")}
    )
    metric_plates.metric("📋 Total Plates", len(init_df))
else:
    table_placeholder.info("No detections yet. Upload and process a video.")

metric_vehicles.metric("🚗 Vehicles", 0)
metric_fps.metric("⚡ FPS", "—")

# ---------------------------------------------------------------------------
# Run button + processing loop
# ---------------------------------------------------------------------------

if uploaded is not None:
    st.divider()
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_clicked = st.button("▶️  Start Detection", use_container_width=True)

    if run_clicked:
        # Save upload to temp file
        suffix = Path(uploaded.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        device = _resolve_device(use_gpu)
        status_placeholder.markdown(
            '<span class="status-badge badge-running">● LOADING MODELS…</span>',
            unsafe_allow_html=True,
        )

        try:
            # Override config with sidebar values
            cfg_run = _load_yaml(CONFIG_PATH)
            cfg_run.setdefault("vehicle_detector", {})["confidence"] = vehicle_conf
            cfg_run.setdefault("plate_detector", {})["confidence"]   = plate_conf
            cfg_run.setdefault("postprocessing", {})["min_reads_before_flush"] = min_reads
            cfg_run.setdefault("tracker", {})["n_init"] = 1  # always confirm fast
            # Scope DB to this session
            cfg_run.setdefault("storage", {})["db_uri"] = _SESSION_DB_URI

            (vehicle_detector, plate_detector, ocr_engine,
             tracker, preprocessor, track_buffer, db) = _load_pipeline_components(cfg_run, device)

        except Exception as e:
            status_placeholder.markdown(
                f'<span class="status-badge badge-error">● ERROR: {e}</span>',
                unsafe_allow_html=True,
            )
            st.stop()

        # Open video
        cap = cv2.VideoCapture(tmp_path)
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_native    = cap.get(cv2.CAP_PROP_FPS) or 30.0

        plate_labels: dict = {}
        plate_confs:  dict = {}
        plate_boxes:  dict = {}  # tid -> [(px1,py1,px2,py2), ...] in frame coords
        raw_ocr_labels: dict = {}  # tid -> raw OCR text (before regex validation)
        ocr_conf_cache: dict = {}  # tid -> best conf seen so far (throttle OCR)
        vehicle_classes: dict = {}  # tid -> class name (car, bus, motorcycle, truck)
        frames_proc   = 0
        plates_committed = 0
        frame_idx     = 0
        t_start       = time.monotonic()
        all_seen_vehicle_ids: set = set()  # cumulative; never decreases

        # OCR confidence threshold: only skip re-reading after a VALIDATED read
        _OCR_SKIP_THRESH = 0.80

        status_placeholder.markdown(
            '<span class="status-badge badge-running">● PROCESSING…</span>',
            unsafe_allow_html=True,
        )

        try:  # try/finally ensures flush_all always runs
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Skip frames
                if frame_idx % skip_frames != 0:
                    continue

                frames_proc += 1

                # ── Vehicle detection ──────────────────────────────────────
                vehicle_dets = vehicle_detector.detect(frame)
                active_tracks, lost_ids = tracker.update(vehicle_dets, frame)

                # Flush lost tracks
                for tid in lost_ids:
                    committed = track_buffer.flush(tid)
                    if committed:
                        ts = datetime.utcnow()
                        vclass = vehicle_classes.get(tid, "vehicle")
                        inserted = db.upsert_detection(
                            plate_number=committed.plate_number,
                            track_id=committed.track_id,
                            confidence=committed.confidence,
                            frame_no=committed.first_frame,
                            first_seen_time=ts,
                            total_reads=committed.total_reads,
                            vehicle_class=vclass,
                        )
                        if inserted:
                            plates_committed += 1
                    plate_labels.pop(tid, None)
                    plate_confs.pop(tid, None)
                    plate_boxes.pop(tid, None)
                    ocr_conf_cache.pop(tid, None)
                    raw_ocr_labels.pop(tid, None)
                    vehicle_classes.pop(tid, None)

                # ── Per-track plate detection + OCR ─────────────────────
                h_f, w_f = frame.shape[:2]
                for track in active_tracks:
                    tid = track.track_id
                    vehicle_classes[tid] = track.class_name
                    x1, y1, x2, y2 = (int(v) for v in track.bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_f - 1, x2), min(h_f - 1, y2)
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue

                    # Detect ALL plates in this vehicle crop (not just best)
                    plate_dets = plate_detector.detect(vehicle_crop, vehicle_offset=(x1, y1))
                    if not plate_dets:
                        continue

                    # Store all plate boxes for drawing (in frame coords)
                    plate_boxes[tid] = [
                        (
                            max(0, int(det.bbox[0]) + x1),
                            max(0, int(det.bbox[1]) + y1),
                            min(w_f - 1, int(det.bbox[2]) + x1),
                            min(h_f - 1, int(det.bbox[3]) + y1),
                        )
                        for det in plate_dets
                    ]

                    # OCR throttle: skip ONLY if a validated read already hit threshold
                    if ocr_conf_cache.get(tid, 0.0) >= _OCR_SKIP_THRESH:
                        continue

                    # Run OCR on the best-confidence plate crop only
                    best_det = plate_dets[0]
                    bpx1, bpy1, bpx2, bpy2 = (int(v) for v in best_det.bbox)
                    bpx1, bpy1 = max(0, bpx1), max(0, bpy1)
                    bpx2, bpy2 = (
                        min(vehicle_crop.shape[1] - 1, bpx2),
                        min(vehicle_crop.shape[0] - 1, bpy2),
                    )
                    plate_crop = vehicle_crop[bpy1:bpy2, bpx1:bpx2]
                    if plate_crop.size > 0:
                        plate_crop = preprocessor.process(plate_crop)
                        text, conf = ocr_engine.read(plate_crop)
                        if text:
                            raw_ocr_labels[tid] = text  # always show raw read on frame
                            if disable_regex:
                                # Bypass regex: push directly into the track buffer
                                from processing.postprocessor import OCRRead as _OCRRead
                                clean_text = text.upper().strip().replace(" ", "").replace("-", "")
                                track_buffer._buffer[tid].append(
                                    _OCRRead(text=clean_text, confidence=conf, frame_no=frame_idx)
                                )
                                if tid not in track_buffer._first_frame:
                                    track_buffer._first_frame[tid] = frame_idx
                                plate_labels[tid] = clean_text
                                plate_confs[tid]  = conf
                                # Only count validated reads for throttle
                                if conf > ocr_conf_cache.get(tid, 0.0):
                                    ocr_conf_cache[tid] = conf
                            else:
                                accepted = track_buffer.add(tid, text, conf, frame_idx)
                                if accepted:
                                    # Get the corrected text from the buffer (last entry)
                                    buffered = track_buffer._buffer[tid][-1]
                                    plate_labels[tid] = buffered.text
                                    plate_confs[tid]  = buffered.confidence
                                    # Only count VALIDATED reads for throttle
                                    if conf > ocr_conf_cache.get(tid, 0.0):
                                        ocr_conf_cache[tid] = conf

                # ── Annotate + display ─────────────────────────────────
                annotated = _draw_overlay(frame, active_tracks, plate_labels, plate_confs, plate_boxes, raw_ocr_labels)
                # Convert BGR → RGB for Streamlit
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

                # ── Progress & metrics ─────────────────────────────────────
                pct = min(frame_idx / total_frames, 1.0)
                progress_bar.progress(pct)

                elapsed = time.monotonic() - t_start
                cur_fps = frames_proc / elapsed if elapsed > 0 else 0.0

                for track in active_tracks:
                    all_seen_vehicle_ids.add(track.track_id)
                metric_vehicles.metric("🚗 Vehicles (total)", len(all_seen_vehicle_ids))
                metric_plates.metric("📋 Plates Committed", plates_committed)
                metric_fps.metric("⚡ FPS", f"{cur_fps:.1f}")

                # ── Update DB table every 5 processed frames ────────────────
                if frames_proc % 5 == 0:
                    df = _fetch_db_df(db)
                    if not df.empty:
                        table_placeholder.dataframe(
                            df, use_container_width=True, hide_index=True,
                            column_config={"Confidence": st.column_config.TextColumn("Confidence 🎯")}
                        )
                    else:
                        table_placeholder.info("No plates committed yet…")

        finally:
            # ── End of video: flush ALL remaining tracks ─────────────────
            # This runs even if the processing loop errors out
            remaining = track_buffer.flush_all()
            for det in remaining:
                ts = datetime.utcnow()
                vclass = vehicle_classes.get(det.track_id, "vehicle")
                inserted = db.upsert_detection(
                    plate_number=det.plate_number,
                    track_id=det.track_id,
                    confidence=det.confidence,
                    frame_no=det.first_frame,
                    first_seen_time=ts,
                    total_reads=det.total_reads,
                    vehicle_class=vclass,
                )
                if inserted:
                    plates_committed += 1
            if 'db' in locals():
                db.close()

        cap.release()
        os.unlink(tmp_path)

        progress_bar.progress(1.0)
        status_placeholder.markdown(
            '<span class="status-badge badge-done">✓ COMPLETE</span>',
            unsafe_allow_html=True,
        )

        # Final DB table
        final_df = _fetch_db_df(db)
        table_placeholder.dataframe(
            final_df, use_container_width=True, hide_index=True,
            column_config={"Confidence": st.column_config.TextColumn("Confidence 🎯")}
        )
        metric_plates.metric("📋 Plates Committed", plates_committed)

        # CSV download
        if not final_df.empty:
            csv_bytes = final_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️  Download CSV",
                data=csv_bytes,
                file_name="detections.csv",
                mime="text/csv",
                use_container_width=False,
            )

        st.success(
            f"✅ Processing complete! "
            f"**{frames_proc}** frames processed · "
            f"**{plates_committed}** plates committed · "
            f"Session: **{_SESSION_NAME}**"
        )
