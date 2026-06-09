"""
Frame extraction from video using ffmpeg.
"""

import glob
import os
import subprocess
from typing import List, Optional

from loguru import logger

from ..config import FPS


class FrameExtractor:
    """Extract frames from a video file at a given FPS."""

    def __init__(self, fps: int = FPS):
        self.fps = fps
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install with: brew install ffmpeg"
            )

    def extract_full(self, video_path: str, output_dir: str) -> List[str]:
        """
        Extract all frames from the entire video at self.fps.
        Returns sorted list of frame file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={self.fps}",
            "-start_number", "0",
            os.path.join(output_dir, "%06d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg extraction failed: {result.stderr[-500:]}")
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

        frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def extract_segment(
        self,
        video_path: str,
        output_dir: str,
        start_ms: float,
        duration_ms: float,
    ) -> List[str]:
        """
        Extract frames from a specific time segment.
        Returns sorted list of frame file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_ms / 1000.0:.3f}",
            "-i", video_path,
            "-t", f"{duration_ms / 1000.0:.3f}",
            "-vf", f"fps={self.fps}",
            "-start_number", "0",
            os.path.join(output_dir, "%06d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg segment extraction failed: {result.stderr[-500:]}")
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}")

        frames = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))
        logger.info(
            f"Extracted {len(frames)} frames (segment "
            f"{start_ms:.0f}-{start_ms + duration_ms:.0f}ms) from {video_path}"
        )
        return frames

    @staticmethod
    def get_video_duration_ms(video_path: str) -> float:
        """Get video duration in milliseconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip()) * 1000
