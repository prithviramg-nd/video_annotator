"""
Central configuration for the video_annotator tool.
All constants and defaults are defined here for easy modification.
"""

import json
import os
import sys

# ── MongoDB (non-secret constants) ──────────────────────────────────────────
MONGO_DB = "analytics"
MONGO_COLLECTION_VIDEO_REQUESTS = "video_requests_v2"
MONGO_COLLECTION_ALERT = "alert"


# ── Credentials loader ─────────────────────────────────────────────────────
def load_credentials(filepath: str) -> dict:
    """Load database credentials from a JSON file.

    Expected format::

        {
            "mongo_uri": "mongodb://user:pass@host:port/?...",
            "postgres_uri": "postgresql://user:pass@host:port/db"  // optional, for future use
        }

    Returns the parsed dict.  Raises SystemExit on error so callers get a
    clear message instead of a traceback.
    """
    path = os.path.expanduser(filepath)
    if not os.path.isfile(path):
        print(f"Error: credentials file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path) as f:
            creds = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in credentials file: {exc}", file=sys.stderr)
        sys.exit(1)
    if "mongo_uri" not in creds:
        print(
            "Error: credentials file must contain a 'mongo_uri' key",
            file=sys.stderr,
        )
        sys.exit(1)
    return creds

# ── Video processing ────────────────────────────────────────────────────────
FPS = 10                       # output video / detection frame rate
VIDEO_OFFSET_MS = 5000         # padding around event window (ms) — 5s before + 5s after
TRIM_VIDEO_FPS = 30            # fps used to interpret trim folder frame numbers

# ── Temporary storage ───────────────────────────────────────────────────────
# Use a repo-local temp/ directory for all scratch files (frames, downloads,
# logs).  The directory is .gitignored and cleaned up after each alert.
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
LOG_FILE = os.path.join(TEMP_DIR, "video_annotator.log")

# ── Annotation defaults ─────────────────────────────────────────────────────
DEFAULT_FONT_SIZE = 24
DEFAULT_SMALL_FONT_SIZE = 16
BBOX_WIDTH = 2
KEYPOINT_RADIUS = 5

# ── Annotation colors ───────────────────────────────────────────────────────
COLORS = {
    "face_bbox": "blue",
    "person_bbox": "lime",
    "nose": "red",
    "lsh": "green",
    "rsh": "green",
    "lear": "cyan",
    "rear": "cyan",
    "left_eye_bbox": "yellow",
    "right_eye_bbox": "yellow",
    "event_active": (255, 0, 0),
    "event_inactive": (255, 255, 255),
    "frame_info": "white",
    "head_pose": "yellow",
    "eye_scores_left": "red",
    "eye_scores_right": "orange",
    "mouth_ydist_var": (255, 165, 0),   # orange - mouth y-dist variance graph
    "nose_x_var": (0, 255, 200),        # teal - nose x ratio variance graph
}

# ── Eye state names ─────────────────────────────────────────────────────────
RAW_EYE_STATE_NAMES = ["CLOS", "OPEN", "SQNT", "OCCL"]
