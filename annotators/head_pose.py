"""Head pose (pitch, yaw, roll) annotator."""

from PIL import Image, ImageDraw

from ..config import COLORS
from .base import BaseAnnotator


class HeadPoseAnnotator(BaseAnnotator):
    name = "head_pose"
    description = "Display head pose angles (pitch, yaw, roll) near face bbox"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        pyr = detection.get("head_pyr")
        if pyr is None:
            return

        p, y, r = pyr
        font = self.get_small_font()

        # Position near face bbox if available, else top-right
        face_bbox = detection.get("face_bbox")
        if face_bbox is not None:
            x1, y1, _, _ = face_bbox
            text_x, text_y = x1, max(0, y1 - 25)
        else:
            text_x, text_y = img.width - 300, 50

        draw.text(
            (text_x, text_y),
            f"P:{p:.1f} Y:{y:.1f} R:{r:.1f}",
            fill=COLORS["head_pose"],
            font=font,
        )
