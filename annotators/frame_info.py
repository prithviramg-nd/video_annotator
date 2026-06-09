"""Frame info annotator - displays frame number and metadata."""

from PIL import Image, ImageDraw

from ..config import COLORS
from .base import BaseAnnotator


class FrameInfoAnnotator(BaseAnnotator):
    name = "frame_info"
    description = "Display frame number, detection framenum, and event status"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        font = self.get_font()
        event_start_frame = kwargs.get("event_start_frame")
        event_end_frame = kwargs.get("event_end_frame")
        framenum = detection.get("framenum", frame_idx)

        # Frame number
        in_event = False
        if event_start_frame is not None and event_end_frame is not None:
            in_event = event_start_frame <= frame_idx <= event_end_frame

        status = " [EVENT]" if in_event else ""
        color = "lime" if in_event else COLORS["frame_info"]

        draw.text(
            (img.width - 280, 10),
            f"Frame: {framenum:03d}{status}",
            fill=color,
            font=font,
        )

        # Speed if available
        speed = detection.get("speed")
        if speed is not None:
            small_font = self.get_small_font()
            draw.text(
                (img.width - 280, 40),
                f"Speed: {speed:.1f}",
                fill=COLORS["frame_info"],
                font=small_font,
            )
