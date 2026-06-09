"""
Central configuration for the video_annotator tool.
All constants and defaults are defined here for easy modification.
"""

import os
import tempfile

# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = (
    "mongodb://mongo_dp_ro:vam1aBcp@analytics-dashboard-mongo-db.netradyne.info:27019/"
    "?readPreference=primary&ssl=false"
)
MONGO_DB = "analytics"
MONGO_COLLECTION_VIDEO_REQUESTS = "video_requests_v2"
MONGO_COLLECTION_ALERT = "alert"

# ── Video processing ────────────────────────────────────────────────────────
FPS = 10                       # output video / detection frame rate
VIDEO_OFFSET_MS = 5000         # padding around event window (ms)
TRIM_VIDEO_FPS = 30            # fps used to interpret trim folder frame numbers

# ── Temporary storage ───────────────────────────────────────────────────────
TEMP_DIR = os.path.join(tempfile.gettempdir(), "video_annotator_temp")

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
}

# ── Eye state names ─────────────────────────────────────────────────────────
RAW_EYE_STATE_NAMES = ["CLOS", "OPEN", "SQNT", "OCCL"]
