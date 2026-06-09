"""
run.py
CLI entry point for the NumPlate Detection pipeline.

Usage examples:
    python run.py --source data/raw/sample.mp4
    python run.py --source 0 --show
    python run.py --source rtsp://192.168.1.10/stream --no-gpu --export-csv
    python run.py --source data/raw/highway.mp4 --config configs/config.yaml --export-csv
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="numplate",
        description="Cascaded license plate detection pipeline — YOLOv8 + DeepSORT + EasyOCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video file path, RTSP URL, or '0' for webcam.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=None,
        help="Show real-time annotated window (overrides config).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the display window (overrides config).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU-only mode (overrides config).",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export detection CSV at the end of the run.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Override CSV export path from config.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Process every Nth frame (overrides config). 1 = every frame.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write annotated output video to disk.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path for the output video file (overrides config).",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] Config file not found at '{path}' — using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_pipeline(cfg: dict, args: argparse.Namespace):
    """Instantiate every component and wire them into a NumPlatePipeline."""
    from models.ocr_engine import build_ocr_engine
    from models.plate_detector import PlateDetector
    from models.vehicle_detector import VehicleDetector
    from pipeline import NumPlatePipeline
    from processing.postprocessor import PlateValidator, TrackBuffer
    from processing.preprocessor import Preprocessor
    from storage.database import DatabaseManager
    from tracking.tracker import DeepSORTTracker
    from utils.logger import setup_logger
    from utils.visualizer import Visualizer

    # --- Logging ---
    log_cfg = cfg.get("logging", {})
    setup_logger(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/pipeline.log"),
        max_bytes=log_cfg.get("max_bytes", 10_485_760),
        backup_count=log_cfg.get("backup_count", 3),
    )

    # --- GPU override ---
    device = "cpu" if args.no_gpu else "auto"

    # --- Vehicle detector ---
    vd_cfg = cfg.get("vehicle_detector", {})
    vehicle_detector = VehicleDetector(
        weights=vd_cfg.get("weights", "yolov8n.pt"),
        confidence=vd_cfg.get("confidence", 0.45),
        iou_threshold=vd_cfg.get("iou_threshold", 0.50),
        device=device,
        target_classes=vd_cfg.get("target_classes", [2, 3, 5, 7]),
        imgsz=vd_cfg.get("imgsz", 640),
    )

    # --- Plate detector ---
    pd_cfg = cfg.get("plate_detector", {})
    plate_detector = PlateDetector(
        weights=pd_cfg.get("weights", "models/weights/plate_detector_best.pt"),
        fallback_weights=pd_cfg.get("fallback_weights", "yolov8n.pt"),
        confidence=pd_cfg.get("confidence", 0.40),
        iou_threshold=pd_cfg.get("iou_threshold", 0.45),
        device=device,
        imgsz=pd_cfg.get("imgsz", 320),
    )

    # --- OCR ---
    ocr_cfg = cfg.get("ocr", {"backend": "easyocr", "languages": ["en"], "gpu": not args.no_gpu})
    if args.no_gpu:
        ocr_cfg["gpu"] = False
    ocr_engine = build_ocr_engine(ocr_cfg)

    # --- Tracker ---
    trk_cfg = cfg.get("tracker", {})
    tracker = DeepSORTTracker(
        max_age=trk_cfg.get("max_age", 30),
        n_init=trk_cfg.get("n_init", 3),
        max_cosine_distance=trk_cfg.get("max_cosine_distance", 0.4),
        nn_budget=trk_cfg.get("nn_budget", 100),
    )

    # --- Preprocessor ---
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

    # --- Post-processor / TrackBuffer ---
    post_cfg = cfg.get("postprocessing", {})
    validator = PlateValidator(
        pattern=post_cfg.get(
            "regex_pattern", r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
        )
    )
    track_buffer = TrackBuffer(
        validator=validator,
        strategy=post_cfg.get("dedup_strategy", "majority_vote"),
        min_reads=post_cfg.get("min_reads_before_flush", 3),
    )

    # --- Storage ---
    store_cfg = cfg.get("storage", {})
    db = DatabaseManager(
        db_uri=store_cfg.get("db_uri", "sqlite:///detections.db"),
        auto_create_tables=store_cfg.get("auto_create_tables", True),
    )

    # --- Visualizer ---
    vis_cfg = cfg.get("visualizer", {})
    show_window = True
    if args.no_show:
        show_window = False
    elif args.show:
        show_window = True
    else:
        show_window = vis_cfg.get("show_window", True)

    visualizer = Visualizer(
        vehicle_box_color=tuple(vis_cfg.get("vehicle_box_color", [0, 255, 128])),
        plate_box_color=tuple(vis_cfg.get("plate_box_color", [0, 165, 255])),
        text_color=tuple(vis_cfg.get("text_color", [255, 255, 255])),
        font_scale=vis_cfg.get("font_scale", 0.65),
        line_thickness=vis_cfg.get("line_thickness", 2),
    ) if vis_cfg.get("enabled", True) else None

    # --- Output video ---
    save_video = args.save_video or vis_cfg.get("save_output_video", False)
    output_video = (
        args.output_video
        or vis_cfg.get("output_video_path", "output/result.mp4")
    )

    # --- Skip frames ---
    src_cfg = cfg.get("source", {})
    skip_frames = args.skip_frames or src_cfg.get("skip_frames", 1)

    return NumPlatePipeline(
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        ocr_engine=ocr_engine,
        tracker=tracker,
        preprocessor=preprocessor,
        track_buffer=track_buffer,
        db=db,
        visualizer=visualizer,
        show_window=show_window,
        skip_frames=skip_frames,
        save_video=save_video,
        output_video_path=output_video,
    ), db, store_cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve video source
    source_cfg = cfg.get("source", {})
    source = args.source or source_cfg.get("path", "0")

    # Convert "0" string to int for webcam
    if source == "0":
        source = 0

    print(f"\n{'='*55}")
    print(f"  NumPlate Detection Pipeline")
    print(f"{'='*55}")
    print(f"  Source  : {source}")
    print(f"  Config  : {args.config}")
    print(f"  GPU     : {'no' if args.no_gpu else 'auto'}")
    print(f"{'='*55}\n")

    pipeline, db, store_cfg = build_pipeline(cfg, args)

    try:
        stats = pipeline.run(source)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # CSV export
        if args.export_csv:
            csv_path = args.csv_path or store_cfg.get(
                "csv_export_path", "output/detections.csv"
            )
            exported = db.export_to_csv(csv_path)
            print(f"\n[INFO] Detections exported → {exported}")

        print(f"\n  Total unique vehicles detected: {db.get_vehicle_count()}")


if __name__ == "__main__":
    main()
