from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Roboflow plate dataset in YOLOv8 format."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Your Roboflow API key (get it at https://app.roboflow.com/settings/api).",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Roboflow workspace slug (shown in the project URL).",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Roboflow project slug.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Dataset version number.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/plates_dataset",
        help="Directory where the dataset will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        print(
            "[ERROR] roboflow is not installed.\n"
            "Install it with:  pip install roboflow"
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Connecting to Roboflow...")
    rf = Roboflow(api_key=args.api_key)

    print(f"[INFO] Downloading {args.workspace}/{args.project} v{args.version} ...")
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download("yolov8", location=args.output_dir, overwrite=True)

    # Verify files actually landed
    yaml_path = os.path.join(args.output_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(
            f"\n[ERROR] Download completed but data.yaml not found at: {yaml_path}\n"
            "  The dataset may have been extracted to a different subfolder.\n"
            f"  dataset.location = {getattr(dataset, 'location', 'unknown')}"
        )
        sys.exit(1)

    print(f"\n[SUCCESS] Dataset downloaded to: {os.path.abspath(args.output_dir)}")
    print(f"  data.yaml path: {yaml_path}")
    print(
        "\n[NEXT STEP] Run training:\n"
        f"  python scripts/train_plate_detector.py "
        f"--data {os.path.join(args.output_dir, 'data.yaml')}"
    )


if __name__ == "__main__":
    main()
