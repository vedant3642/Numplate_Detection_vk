"""
scripts/train_plate_detector.py
Fine-tune YOLOv8 on a license plate dataset.

Usage:
    python scripts/train_plate_detector.py \
        --data data/plates_dataset/data.yaml \
        --epochs 100 \
        --imgsz 640 \
        --batch 16 \
        --device 0

After training, best weights will be saved to:
    models/weights/plate_detector_best.pt

Then update configs/config.yaml:
    plate_detector:
      weights: "models/weights/plate_detector_best.pt"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def _auto_device() -> str:
    """Return '0' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for license plate detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/plates_dataset/data.yaml",
        help="Path to the dataset data.yaml (YOLO format).",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="yolov8n.pt",
        help="Starting weights (yolov8n.pt, yolov8s.pt, etc.).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (-1 = auto-batch). Default 8 is safe for CPU; increase for GPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: '0' for GPU 0, 'cpu' for CPU. Auto-detected if not set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/weights",
        help="Directory to copy the best weights to.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="YOLOv8 training run output directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="plate_detector",
        help="Run name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve device: auto-detect if not specified
    device = args.device if args.device is not None else _auto_device()

    if not os.path.exists(args.data):
        print(
            f"[ERROR] Dataset not found at '{args.data}'.\n"
            f"Run scripts/download_dataset.py first."
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  YOLOv8 Plate Detector Fine-Tuning")
    print(f"{'='*55}")
    print(f"  Base weights : {args.base_weights}")
    print(f"  Dataset      : {args.data}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Device       : {device}")
    print(f"{'='*55}\n")

    model = YOLO(args.base_weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation
        degrees=15.0,       # rotation ±15°
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Early stopping
        patience=20,
        # Save best
        save=True,
        save_period=-1,     # only save best
    )

    # Copy best weights to models/weights/
    best_pt = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs(args.output_dir, exist_ok=True)
        dest = os.path.join(args.output_dir, "plate_detector_best.pt")
        shutil.copy2(best_pt, dest)
        print(f"\n[SUCCESS] Best weights saved to: {os.path.abspath(dest)}")
        print(
            "\n[NEXT STEP] Update configs/config.yaml:\n"
            "  plate_detector:\n"
            f"    weights: \"{dest}\""
        )
    else:
        print(
            f"\n[WARN] best.pt not found at expected path '{best_pt}'.\n"
            f"Check the training output in: {os.path.join(args.project, args.name)}"
        )

    # Quick validation on the val split
    print("\n[INFO] Running validation on val split...")
    val_results = model.val()
    print(f"  mAP50      : {val_results.box.map50:.3f}")
    print(f"  mAP50-95   : {val_results.box.map:.3f}")
    print(f"  Precision  : {val_results.box.p.mean():.3f}")
    print(f"  Recall     : {val_results.box.r.mean():.3f}")


if __name__ == "__main__":
    main()
