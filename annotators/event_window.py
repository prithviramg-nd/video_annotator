"""Event window timeline annotator - shows a progress bar for the event."""

from PIL import Image, ImageDraw

from ..config import COLORS
from .base import BaseAnnotator


class EventWindowAnnotator(BaseAnnotator):
    name = "event_window"
    description = "Draw event timeline showing active event window at bottom of frame"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        event_start_frame = kwargs.get("event_start_frame")
        event_end_frame = kwargs.get("event_end_frame")

        if event_start_frame is None or event_end_frame is None:
            return

        # Draw timeline dots
        countdown = frame_idx
        while countdown >= 0:
            in_event = event_start_frame <= countdown <= event_end_frame
            color = COLORS["event_active"] if in_event else COLORS["event_inactive"]
            y_center = 120 if in_event else 200
            x_center = (img.width // max(total_frames, 1)) * countdown + 10
            draw.ellipse(
                (x_center - 5, y_center - 5, x_center + 5, y_center + 5),
                fill=color,
            )
            if countdown != frame_idx:
                prev_x = (img.width // max(total_frames, 1)) * (countdown + 1) + 10
                prev_in = event_start_frame <= (countdown + 1) <= event_end_frame
                prev_y = 120 if prev_in else 200
                prev_col = COLORS["event_active"] if prev_in else COLORS["event_inactive"]
                draw.line((prev_x, prev_y, x_center, y_center), fill=prev_col, width=2)
            countdown -= 1
