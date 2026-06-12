"""Eye detection scores annotator."""

import numpy as np
from PIL import Image, ImageDraw

from ..config import COLORS, RAW_EYE_STATE_NAMES
from .base import BaseAnnotator


class EyeScoresAnnotator(BaseAnnotator):
    name = "eye_scores"
    description = "Display left/right eye state scores (CLOS/OPEN/SQNT/OCCL) and eye bboxes"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        font = self.get_small_font()

        # Left eye scores
        le_scores = detection.get("left_eye_det_scores")
        if le_scores is not None:
            sorted_pairs = sorted(
                enumerate(np.round(np.array(le_scores), 2)),
                key=lambda x: x[1],
                reverse=True,
            )
            text = "L: " + ", ".join(
                f"{RAW_EYE_STATE_NAMES[idx]}({score})"
                for idx, score in sorted_pairs
            )
            draw.text((10, 10), text, fill=COLORS["eye_scores_left"], font=font)

        # Right eye scores
        re_scores = detection.get("right_eye_det_scores")
        if re_scores is not None:
            sorted_pairs = sorted(
                enumerate(np.round(np.array(re_scores), 2)),
                key=lambda x: x[1],
                reverse=True,
            )
            text = "R: " + ", ".join(
                f"{RAW_EYE_STATE_NAMES[idx]}({score})"
                for idx, score in sorted_pairs
            )
            draw.text((10, 30), text, fill=COLORS["eye_scores_right"], font=font)

        # Left eye bbox
        le_bbox = detection.get("left_eye_bbox")
        if le_bbox is not None:
            x1, y1, x2, y2 = le_bbox
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if x_max > x_min and y_max > y_min:
                draw.rectangle(
                    [x_min, y_min, x_max, y_max],
                    outline=COLORS["left_eye_bbox"],
                    width=1,
                )

        # Right eye bbox
        re_bbox = detection.get("right_eye_bbox")
        if re_bbox is not None:
            x1, y1, x2, y2 = re_bbox
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if x_max > x_min and y_max > y_min:
                draw.rectangle(
                    [x_min, y_min, x_max, y_max],
                    outline=COLORS["right_eye_bbox"],
                    width=1,
                )
