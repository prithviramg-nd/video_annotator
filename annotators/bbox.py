"""Bounding box annotators: face_bbox and person_bbox."""

from PIL import Image, ImageDraw

from ..config import BBOX_WIDTH, COLORS
from .base import BaseAnnotator


def _normalize_bbox(bbox):
    """
    Normalize bbox coordinates: ensure x_min <= x_max, y_min <= y_max.
    Returns (x_min, y_min, x_max, y_max) or None if degenerate (zero area).
    """
    x1, y1, x2, y2 = bbox
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    # Skip degenerate boxes (zero area or all zeros)
    if x_max <= x_min or y_max <= y_min:
        return None
    return (x_min, y_min, x_max, y_max)


class FaceBBoxAnnotator(BaseAnnotator):
    name = "face_bbox"
    description = "Draw face bounding box (face_bbox field)"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        bbox = detection.get("face_bbox")
        if bbox is None:
            return
        coords = _normalize_bbox(bbox)
        if coords is None:
            return
        draw.rectangle(coords, outline=COLORS["face_bbox"], width=BBOX_WIDTH)


class PersonBBoxAnnotator(BaseAnnotator):
    name = "person_bbox"
    description = "Draw person bounding box (person_bbox field)"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        bbox = detection.get("person_bbox")
        if bbox is None:
            return
        coords = _normalize_bbox(bbox)
        if coords is None:
            return
        draw.rectangle(coords, outline=COLORS["person_bbox"], width=BBOX_WIDTH)
