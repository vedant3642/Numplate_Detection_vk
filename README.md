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
│   ├── raw/                     ← Input videos here
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

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: install the CUDA-enabled PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Usage

### Quick start (zero fine-tuning required)

Drop a video into `data/raw/` and run:

```bash
python run.py --source data/raw/your_video.mp4 --show
```

> The vehicle detector uses pre-trained COCO weights (`yolov8n.pt`), which detects cars/trucks out of the box. The plate detector falls back to the same model until you fine-tune it.

### Webcam

```bash
python run.py --source 0 --show
```

### RTSP stream

```bash
python run.py --source rtsp://192.168.1.10/stream --no-show --export-csv
```

### All options

```
--source        Video file, RTSP URL, or 0 for webcam
--config        Path to config.yaml (default: configs/config.yaml)
--show          Force show display window
--no-show       Disable display window
--no-gpu        Force CPU-only inference
--export-csv    Write CSV at end of run
--csv-path      Override CSV output path
--skip-frames   Process every Nth frame (e.g. 2 = skip every other)
--save-video    Save annotated output video
--output-video  Path for output video
```

### Export CSV

```bash
python run.py --source data/raw/video.mp4 --export-csv --csv-path output/results.csv
```

---

## Fine-Tuning the Plate Detector (Recommended)

The base COCO model detects vehicles well but is not specialized for plate localization. Fine-tuning on a plate dataset significantly improves accuracy.

### Step 1 — Get a Roboflow API key

Sign up at [roboflow.com](https://roboflow.com) → Settings → API Key.

### Step 2 — Download a dataset

```bash
python scripts/download_dataset.py \
    --api-key YOUR_API_KEY \
    --workspace your-workspace \
    --project license-plate-detection \
    --version 1
```

Recommended Roboflow projects:
- `license-plate-detection` (generic, multi-country)
- `indian-license-plate` (India-specific, best for this pipeline)
- `vehicle-registration-plates` (multi-angle, challenging)

### Step 3 — Train

```bash
python scripts/train_plate_detector.py \
    --data data/plates_dataset/data.yaml \
    --epochs 100 \
    --batch 16 \
    --device 0
```

Training takes ~1–2 hours on an NVIDIA T4/RTX 3060. Best weights are saved to `models/weights/plate_detector_best.pt` automatically.

### Step 4 — Update config

`configs/config.yaml` already points to `models/weights/plate_detector_best.pt`. No change needed after training.

---

## Output Schema

Each unique vehicle detection is stored as one row:

| Column | Type | Description |
|---|---|---|
| `plate_number` | TEXT (PK) | Normalised plate string (e.g. `MH12AB1234`) |
| `first_seen_time` | DATETIME | UTC timestamp of first frame |
| `track_id` | INT | DeepSORT track ID |
| `confidence` | FLOAT | Average OCR confidence (0–1) |
| `frame_no` | INT | Frame number of first appearance |
| `vehicle_class` | TEXT | COCO class (car, truck, bus, motorcycle) |
| `total_reads` | INT | Number of valid OCR reads accumulated |

The count of unique plates = **total vehicle count**.

---

## Configuration Reference

All parameters live in `configs/config.yaml`. Key settings:

| Section | Key | Description |
|---|---|---|
| `source` | `path` | Default video source |
| `source` | `skip_frames` | Process every Nth frame |
| `vehicle_detector` | `confidence` | YOLOv8 detection threshold |
| `plate_detector` | `weights` | Path to fine-tuned plate weights |
| `ocr` | `backend` | `"easyocr"` or `"paddleocr"` |
| `postprocessing` | `regex_pattern` | Plate format validation regex |
| `postprocessing` | `dedup_strategy` | `"majority_vote"` or `"best_confidence"` |
| `storage` | `db_uri` | SQLAlchemy database URL |
| `visualizer` | `show_window` | Show display window by default |

---

## Indian Plate Specifics

The pipeline includes two India-specific optimisations enabled by default:

1. **CLAHE preprocessing** — corrects yellow/white background glare on Indian plates before OCR
2. **IND emblem crop** — trims the leftmost 15% of each plate region to remove the blue IND symbol

The default regex validates Indian plates in both old and new BH series formats:
```
^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$
```

---

## Switching OCR Backend

Change one line in `configs/config.yaml`:

```yaml
ocr:
  backend: "paddleocr"   # was "easyocr"
```

Install PaddleOCR:
```bash
pip install paddleocr paddlepaddle
```

---

## Switching Database

For PostgreSQL:
```yaml
storage:
  db_uri: "postgresql://user:password@localhost:5432/numplate"
```

Install the driver:
```bash
pip install psycopg2-binary
```

---

## Performance Tips

| Tip | Impact |
|---|---|
| Use `--skip-frames 2` | ~2× FPS improvement with minimal accuracy loss |
| Use `yolov8n.pt` (nano) | Fastest; suitable for real-time on CPU |
| Use `yolov8s.pt` (small) | Better accuracy; needs GPU |
| Set `ocr.gpu: false` | Required if no CUDA GPU |
| Reduce `plate_detector.imgsz` to 256 | Faster plate detection |

---

## Logs

Rotating logs are written to `logs/pipeline.log`. Adjust level in `config.yaml`:
```yaml
logging:
  level: "INFO"   # DEBUG | INFO | WARNING | ERROR
```
