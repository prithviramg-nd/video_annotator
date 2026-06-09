"""Bounding box annotators: face_bbox and person_bbox."""

from PIL import Image, ImageDraw

from ..config import BBOX_WIDTH, COLORS
from .base import BaseAnnotator


class FaceBBoxAnnotator(BaseAnnotator):
    name = "face_bbox"
    description = "Draw face bounding box (face_bbox field)"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        bbox = detection.get("face_bbox")
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=COLORS["face_bbox"], width=BBOX_WIDTH)


class PersonBBoxAnnotator(BaseAnnotator):
    name = "person_bbox"
    description = "Draw person bounding box (person_bbox field)"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        bbox = detection.get("person_bbox")
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        draw.rectangle(
            [x1, y1, x2, y2], outline=COLORS["person_bbox"], width=BBOX_WIDTH
        )
