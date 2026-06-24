"""
Variance graph annotators.

Two rolling-variance graph overlays:
1. MouthYDistVarianceAnnotator - rolling variance of mouth upper-lip to lower-lip
   y-distance over a 1-second window (10 frames at 10 fps).
2. NoseXRatioVarianceAnnotator - rolling variance of
   (nose_x - rear_x) / (lear_x - rear_x + eps) ratio over a 1-second window.
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ..config import COLORS, FPS
from .base import BaseAnnotator

# Rolling window = 1 second at configured FPS
VARIANCE_WINDOW = FPS  # 10 frames for 10 fps


class MouthYDistVarianceAnnotator(BaseAnnotator):
    """Plot rolling variance of mouth upper-lip to lower-lip y-distance."""

    name = "mouth_ydist_var"
    description = (
        "Graph: rolling 1-sec variance of mouth upper-lip / lower-lip y-distance"
    )

    def __init__(self):
        super().__init__()
        self._cached_variances = None
        self._cached_y_dists = None
        self._cached_id = None

    def _compute(self, detections):
        """Pre-compute y-dist and rolling variance for all frames."""
        y_dists = []
        for det in detections:
            mouth_kps = det.get("mouth_kps")
            if mouth_kps is not None and len(mouth_kps) >= 8:
                # Index 3 = upper lip y, index 7 = lower lip y
                upper_y = mouth_kps[3]
                lower_y = mouth_kps[7]
                y_dists.append(abs(lower_y - upper_y))
            else:
                # Use previous value if available, else 0
                y_dists.append(y_dists[-1] if y_dists else 0.0)

        series = pd.Series(y_dists)
        variances = series.rolling(
            window=VARIANCE_WINDOW, min_periods=VARIANCE_WINDOW
        ).var().tolist()

        return y_dists, variances

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        detections = kwargs.get("detections")
        if detections is None:
            return

        # Cache computation keyed on id of detections list
        det_id = id(detections)
        if self._cached_id != det_id:
            self._cached_y_dists, self._cached_variances = self._compute(detections)
            self._cached_id = det_id

        variances = self._cached_variances
        y_dists = self._cached_y_dists

        event_start = kwargs.get("event_start_frame", 0)
        event_end = kwargs.get("event_end_frame", total_frames - 1)

        # ── Graph region (bottom-left area by default) ───────────────────
        g_height = 120
        g_bottom = img.height - 10
        g_top = g_bottom - g_height
        g_left = 10
        g_right = img.width // 2 - 10

        # If mouth keypoints are obstructed by the graph, move graph to 20% from top
        if detection is not None and detection.get("mouth_kps") is not None:
            if detection["mouth_kps"][3] > img.height * 0.65: # then mouth is low in frame, move graph up
                g_top = int(img.height * 0.20)
                g_bottom = g_top + g_height

        font = self.get_small_font()
        color = COLORS.get("mouth_ydist_var", (255, 165, 0))  # orange

        # Background
        draw.rectangle(
            [g_left - 2, g_top - 22, g_right + 2, g_bottom + 2],
            fill=(0, 0, 0, 180) if img.mode == "RGBA" else (20, 20, 20),
            outline="gray",
        )

        # Compute scale from valid variances up to current frame
        valid_vars = [
            v for v in variances[: frame_idx + 1]
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        max_var = max(max(valid_vars) * 1.2, 0.1) if valid_vars else 1.0

        # Title
        draw.text(
            (g_left + 4, g_top - 20),
            f"Mouth Y-dist Var (win={VARIANCE_WINDOW})",
            fill=color,
            font=font,
        )

        # Scale labels
        draw.text((g_right - 60, g_top), f"{max_var:.2f}", fill="white", font=font)
        draw.text((g_right - 40, g_bottom - 14), "0", fill="white", font=font)

        # Grid lines
        for frac in [0.25, 0.5, 0.75]:
            gy = int(g_bottom - frac * g_height)
            draw.line((g_left, gy, g_right, gy), fill="gray", width=1)

        # Event boundaries
        for boundary in [event_start, event_end]:
            if 0 <= boundary < total_frames:
                bx = int(
                    g_left + (g_right - g_left) * boundary / max(total_frames - 1, 1)
                )
                draw.line((bx, g_top, bx, g_bottom), fill="red", width=1)

        # Plot variance up to current frame
        prev_x, prev_y = None, None
        for k in range(min(frame_idx + 1, total_frames)):
            v = variances[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                prev_x, prev_y = None, None
                continue

            x_c = int(g_left + (g_right - g_left) * k / max(total_frames - 1, 1))
            y_c = int(g_bottom - (min(v, max_var) / max_var) * g_height)
            y_c = max(g_top, min(g_bottom, y_c))

            draw.ellipse((x_c - 2, y_c - 2, x_c + 2, y_c + 2), fill=color)

            if prev_x is not None:
                draw.line((prev_x, prev_y, x_c, y_c), fill=color, width=2)

            prev_x, prev_y = x_c, y_c

        # Current frame marker (vertical line)
        cx = int(g_left + (g_right - g_left) * frame_idx / max(total_frames - 1, 1))
        draw.line((cx, g_top, cx, g_bottom), fill="white", width=1)

        # Current value text
        cur_var = variances[frame_idx]
        if cur_var is not None and not (isinstance(cur_var, float) and np.isnan(cur_var)):
            draw.text(
                (g_left + 4, g_top + 2),
                f"val: {cur_var:.4f}",
                fill="white",
                font=font,
            )


class NoseXRatioVarianceAnnotator(BaseAnnotator):
    """Plot rolling variance of nose-x position ratio between ears."""

    name = "nose_x_var"
    description = (
        "Graph: rolling 1-sec variance of nose_x ratio between left/right ears"
    )

    def __init__(self):
        super().__init__()
        self._cached_variances = None
        self._cached_ratios = None
        self._cached_id = None

    def _compute(self, detections):
        """
        Pre-compute nose x percentile between ears and rolling variance.

        ratio = (nose_x - rear_x) / (lear_x - rear_x + eps)
        This gives the nose's relative horizontal position between the ears.
        variance is computed over a rolling 1-second window.
        """
        ratios = []
        for det in detections:
            nose = det.get("nose")
            lear = det.get("lear")
            rear = det.get("rear")

            if nose is not None and lear is not None and rear is not None:
                nose_x = nose[0]
                lear_x = lear[0]
                rear_x = rear[0]
                denom = lear_x - rear_x + 1e-6
                ratio = (nose_x - rear_x) / denom
                ratios.append(ratio)
            else:
                # Use previous value if available, else 0.5 (centered)
                ratios.append(ratios[-1] if ratios else 0.5)

        series = pd.Series(ratios)
        # Multiply by 10000 as per the reference nose_movement_check logic
        variances = (
            series.rolling(window=VARIANCE_WINDOW, min_periods=VARIANCE_WINDOW)
            .var()
            .multiply(10000)
            .tolist()
        )

        return ratios, variances

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        detections = kwargs.get("detections")
        if detections is None:
            return

        # Cache computation keyed on id of detections list
        det_id = id(detections)
        if self._cached_id != det_id:
            self._cached_ratios, self._cached_variances = self._compute(detections)
            self._cached_id = det_id

        variances = self._cached_variances
        ratios = self._cached_ratios

        event_start = kwargs.get("event_start_frame", 0)
        event_end = kwargs.get("event_end_frame", total_frames - 1)

        # ── Graph region (bottom-right area by default) ──────────────────
        g_height = 120
        g_bottom = img.height - 10
        g_top = g_bottom - g_height
        g_left = img.width // 2 + 10
        g_right = img.width - 10

        # If nose keypoints are obstructed by the graph, move graph to 20% from top
        if detection is not None and detection.get("nose") is not None:
            if detection["nose"][1] > img.height * 0.65: # then face is low in frame, move graph up
                g_top = int(img.height * 0.20)
                g_bottom = g_top + g_height

        font = self.get_small_font()
        color = COLORS.get("nose_x_var", (0, 255, 200))  # teal/green

        # Background
        draw.rectangle(
            [g_left - 2, g_top - 22, g_right + 2, g_bottom + 2],
            fill=(0, 0, 0, 180) if img.mode == "RGBA" else (20, 20, 20),
            outline="gray",
        )

        # Compute scale from valid variances up to current frame
        valid_vars = [
            v for v in variances[: frame_idx + 1]
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        max_var = max(max(valid_vars) * 1.2, 1.0) if valid_vars else 50.0

        # Title
        draw.text(
            (g_left + 4, g_top - 20),
            f"Nose-X Ratio Var*1e4 (win={VARIANCE_WINDOW})",
            fill=color,
            font=font,
        )

        # Scale labels
        draw.text((g_right - 60, g_top), f"{max_var:.1f}", fill="white", font=font)
        draw.text((g_right - 40, g_bottom - 14), "0", fill="white", font=font)

        # Grid lines
        for frac in [0.25, 0.5, 0.75]:
            gy = int(g_bottom - frac * g_height)
            draw.line((g_left, gy, g_right, gy), fill="gray", width=1)

        # Threshold line at 50 (reference: thresh=50 in nose_movement_check)
        if max_var > 50:
            thresh_y = int(g_bottom - (50 / max_var) * g_height)
            draw.line((g_left, thresh_y, g_right, thresh_y), fill="red", width=1)
            draw.text(
                (g_left + 4, thresh_y - 14), "thresh=50", fill="red", font=font
            )

        # Event boundaries
        for boundary in [event_start, event_end]:
            if 0 <= boundary < total_frames:
                bx = int(
                    g_left + (g_right - g_left) * boundary / max(total_frames - 1, 1)
                )
                draw.line((bx, g_top, bx, g_bottom), fill="red", width=1)

        # Plot variance up to current frame
        prev_x, prev_y = None, None
        for k in range(min(frame_idx + 1, total_frames)):
            v = variances[k]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                prev_x, prev_y = None, None
                continue

            x_c = int(g_left + (g_right - g_left) * k / max(total_frames - 1, 1))
            y_c = int(g_bottom - (min(v, max_var) / max_var) * g_height)
            y_c = max(g_top, min(g_bottom, y_c))

            draw.ellipse((x_c - 2, y_c - 2, x_c + 2, y_c + 2), fill=color)

            if prev_x is not None:
                draw.line((prev_x, prev_y, x_c, y_c), fill=color, width=2)

            prev_x, prev_y = x_c, y_c

        # Current frame marker (vertical line)
        cx = int(g_left + (g_right - g_left) * frame_idx / max(total_frames - 1, 1))
        draw.line((cx, g_top, cx, g_bottom), fill="white", width=1)

        # Current value text
        cur_var = variances[frame_idx]
        if cur_var is not None and not (isinstance(cur_var, float) and np.isnan(cur_var)):
            draw.text(
                (g_left + 4, g_top + 2),
                f"val: {cur_var:.2f}",
                fill="white",
                font=font,
            )
