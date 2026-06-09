# video_annotator

Modular tool for downloading alert videos from S3, annotating them with DMS detection data from metadata.txt, and uploading results.

## Setup

```bash
# Install ffmpeg
brew install ffmpeg

# Install Python dependencies
pip install -r video_annotator/requirements.txt
```

## Usage

```bash
# From the repo root:
cd /Users/prithviram/work/git/work

# List available annotators
python -m video_annotator.main --list-annotators

# Process a single alert with specific annotations
python -m video_annotator.main --alert-id 5815863985 --annotate face_bbox nose shoulders

# Process alerts from CSV with all annotations
python -m video_annotator.main --csv video_annotator/sample_alerts.csv --annotate all

# Process and upload to S3
python -m video_annotator.main --alert-id 5815863985 --annotate face_bbox nose --s3-upload s3://netradyne-sharing/analytics/prithvi/annotated/

# Custom output directory and FPS
python -m video_annotator.main --csv alerts.csv --annotate frame_info event_window --output-dir ./my_output --fps 10
```

## Available Annotators

| Name | Description |
|------|-------------|
| `face_bbox` | Face bounding box |
| `person_bbox` | Person bounding box |
| `nose` | Nose keypoint with label |
| `shoulders` | Shoulder keypoints (lsh, rsh) with connecting lines |
| `ears` | Ear keypoints (lear, rear) |
| `eye_scores` | Eye state scores (CLOS/OPEN/SQNT/OCCL) and eye bboxes |
| `head_pose` | Head pitch/yaw/roll angles |
| `event_window` | Timeline showing event active window |
| `frame_info` | Frame number, event status, speed |
| `mouth_kps` | Mouth keypoints with V/H ratio |

## Adding New Annotators

1. Create a new file in `video_annotator/annotators/` (e.g. `my_annotator.py`)
2. Subclass `BaseAnnotator` and implement `annotate()`
3. Import and register in `video_annotator/annotators/registry.py`

Example:
```python
from .base import BaseAnnotator

class MyAnnotator(BaseAnnotator):
    name = "my_thing"
    description = "Draw my custom annotation"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        # Your drawing code here
        value = detection.get("my_field")
        if value is not None:
            draw.text((10, 100), f"my_field: {value}", fill="white")
```

## Project Structure

```
video_annotator/
    __init__.py
    __main__.py            # python -m video_annotator.main
    main.py                # CLI entry point and pipeline orchestration
    config.py              # Constants and defaults
    data_sources/
        base.py            # Abstract data source interface
        mongodb.py         # MongoDB fetcher (video_requests_v2 + alert)
        postgres.py        # PostgreSQL fetcher (stub for AVID support)
    downloader/
        s3.py              # S3 download/upload with credential fallback
    metadata/
        parser.py          # Parse metadata.txt for events + detections
    video/
        extractor.py       # ffmpeg frame extraction
        assembler.py       # Reassemble annotated frames to video
    annotators/
        base.py            # Abstract annotator base class
        registry.py        # Dynamic annotator registry
        bbox.py            # face_bbox, person_bbox
        keypoints.py       # nose, shoulders, ears
        eye_scores.py      # Eye detection scores
        head_pose.py       # Pitch/yaw/roll
        event_window.py    # Event timeline
        frame_info.py      # Frame number overlay
        mouth.py           # Mouth keypoints
```

## Data Flow

1. Alert IDs from CLI (`--alert-id`) or CSV (`--csv`)
2. MongoDB lookup: `video_requests_v2` (preferred) -> `alert` collection (fallback)
3. S3 download: inputVideo.mp4 + metadata.txt
4. Parse metadata for detections and event timing
5. Extract frames at configured FPS
6. Apply selected annotators to each frame
7. Reassemble frames into output video
8. Optional S3 upload
