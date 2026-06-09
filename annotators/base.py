"""
Base annotator class.

All annotators inherit from this and implement the `annotate` method.
Each annotator is responsible for drawing one type of annotation on a frame.
"""

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


class BaseAnnotator(ABC):
    """
    Abstract base class for frame annotators.

    Each annotator receives a frame (PIL Image), the detection dict for that
    frame, and draws its specific annotations on it.

    To add a new annotation type:
      1. Create a new file in annotators/
      2. Subclass BaseAnnotator
      3. Implement annotate()
      4. Register it in registry.py
    """

    # Human-readable name for CLI help
    name: str = "base"
    description: str = "Base annotator"

    def __init__(self):
        self._font = None
        self._small_font = None

    def get_font(self, size: int = 24) -> ImageFont.FreeTypeFont:
        """Get a font, falling back to default if system fonts unavailable."""
        if self._font is None or self._font.size != size:
            try:
                # Try common macOS font paths
                for font_path in [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/System/Library/Fonts/SFNSMono.ttf",
                    "/Library/Fonts/Arial.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                ]:
                    try:
                        self._font = ImageFont.truetype(font_path, size)
                        return self._font
                    except (OSError, IOError):
                        continue
                self._font = ImageFont.load_default()
            except Exception:
                self._font = ImageFont.load_default()
        return self._font

    def get_small_font(self, size: int = 16) -> ImageFont.FreeTypeFont:
        """Get a smaller font."""
        if self._small_font is None or self._small_font.size != size:
            try:
                for font_path in [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/System/Library/Fonts/SFNSMono.ttf",
                    "/Library/Fonts/Arial.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                ]:
                    try:
                        self._small_font = ImageFont.truetype(font_path, size)
                        return self._small_font
                    except (OSError, IOError):
                        continue
                self._small_font = ImageFont.load_default()
            except Exception:
                self._small_font = ImageFont.load_default()
        return self._small_font

    @abstractmethod
    def annotate(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        detection: dict,
        frame_idx: int,
        total_frames: int,
        **kwargs,
    ) -> None:
        """
        Draw annotations on the frame.

        Args:
            img: PIL Image to annotate (in-place).
            draw: ImageDraw instance for the image.
            detection: detection dict for this frame from metadata.
            frame_idx: 0-based index of this frame.
            total_frames: total number of frames.
            **kwargs: additional context (event_start_frame, event_end_frame, etc.)
        """
        ...
