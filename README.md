# NumPlate Detection Pipeline

A production-grade, cascaded license plate detection and recognition system for Indian traffic surveillance. Detects vehicles, localizes license plates within vehicle crops, reads plate text via OCR, tracks vehicles across frames, deduplicates reads, and persists results to a database.

---

## Architecture

```
Video frames
    │
    ▼
Vehicle Detection (YOLOv8 — COCO)
    │  car, truck, bus, motorcycle
    ▼
DeepSORT Tracking ──────────────────┐
    │  track_id per vehicle          │ lost track → flush TrackBuffer
    ▼                                │
Vehicle crop ROI                     │
    │                                │
    ▼                                │
Plate Detection (YOLOv8 fine-tuned)  │
    │                                │
    ▼                                │
Preprocessing (CLAHE + trim + upscale)│
    │                                │
    ▼                                │
OCR (EasyOCR / PaddleOCR)           │
    │                                │
    ▼                                │
Regex Validation + TrackBuffer ──────┘
    │  majority vote / best confidence
    ▼
Upsert → SQLite DB / PostgreSQL
    │
    ▼
CSV Export + Real-time HUD display
```

---

## Project Structure

```
NumPlate_Detection/
├── configs/
│   └── config.yaml              ← All tunable parameters
├── data/
│   ├── raw/                     
│   └── plates_dataset/          ← Roboflow dataset (after download)
├── models/
│   ├── vehicle_detector.py
│   ├── plate_detector.py
│   ├── ocr_engine.py
│   └── weights/                 ← Fine-tuned .pt files go here
├── tracking/
│   └── tracker.py
├── processing/
│   ├── preprocessor.py
│   └── postprocessor.py
├── storage/
│   ├── models.py
│   └── database.py
├── utils/
│   ├── visualizer.py
│   └── logger.py
├── scripts/
│   ├── download_dataset.py
│   └── train_plate_detector.py
├── pipeline.py
├── run.py
├── requirements.txt
└── README.md
```
