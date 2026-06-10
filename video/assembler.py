"""
Video assembly from annotated frames using ffmpeg.
"""

import os
import subprocess
from typing import Optional

from loguru import logger

from ..config import FPS


class VideoAssembler:
    """Assemble annotated frames back into a video."""

    def __init__(self, fps: int = FPS):
        self.fps = fps

    def assemble(
        self,
        frames_dir: str,
        output_path: str,
        pattern: str = "%06d.jpg",
        alert_id: int = None,
    ) -> str:
        """
        Create an MP4 video from sequentially numbered frame images.

        Args:
            frames_dir: directory containing the frame images
            output_path: where to write the output MP4
            pattern: ffmpeg pattern for frame filenames
            alert_id: optional alert ID for log context

        Returns:
            Path to the assembled video.
        """
        tag = f"[{alert_id}] " if alert_id else ""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", os.path.join(frames_dir, pattern),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"{tag}Video assembly failed: {result.stderr[-500:]}")
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

        logger.info(f"{tag}Assembled video: {output_path}")
        return output_path
