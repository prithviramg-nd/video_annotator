"""
S3 download/upload handler using AWS CLI.

Uses `aws s3 cp` and `aws s3 ls` subprocess calls instead of boto3.
This avoids SSL/fork issues on macOS when used with multiprocessing (p_tqdm).

Handles:
  - Downloading video files (inputVideo.mp4) with multi-strategy resolution
  - Downloading metadata.txt
  - Uploading annotated videos to S3
"""

import os
import re
import subprocess
from typing import Optional, Tuple
from urllib.parse import urlparse

from loguru import logger

from ..config import TRIM_VIDEO_FPS


class S3Handler:
    """Download from and upload to S3 using AWS CLI."""

    def s3_object_exists(self, s3_path: str) -> bool:
        """Check if an S3 object exists using `aws s3 ls`."""
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", s3_path],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except Exception as e:
            logger.warning(f"S3 access error checking {s3_path}: {e}")
            return False

    def download_file(self, s3_path: str, local_path: str) -> bool:
        """Download a file from S3 using `aws s3 cp`."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, local_path],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Downloaded {s3_path} -> {local_path}")
                return True
            else:
                logger.error(f"Download failed for {s3_path}: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Download timed out for {s3_path}")
            return False
        except Exception as e:
            logger.error(f"Download failed for {s3_path}: {e}")
            return False

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        """Upload a local file to S3 using `aws s3 cp`."""
        try:
            result = subprocess.run(
                [
                    "aws", "s3", "cp", local_path, s3_path,
                    "--content-type", "video/mp4",
                    "--content-disposition", "inline",
                ],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"Uploaded {local_path} -> {s3_path}")
                return True
            else:
                logger.error(f"Upload failed for {s3_path}: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Upload timed out for {s3_path}")
            return False
        except Exception as e:
            logger.error(f"Upload failed for {s3_path}: {e}")
            return False

    def resolve_video_path(
        self, video_s3_path: str, video_http_path: str = None
    ) -> Tuple[Optional[str], float]:
        """
        Resolve the actual video location on S3 using multiple strategies.

        For video_requests_v2 source: video_s3_path already points to
        8/inputVideo.mp4 directly.

        For alert source: video_http_path may point to trimmedVideos or
        outputVideo.mp4. We try:
          1. {session_key}8/inputVideo.mp4
          2. {session_key}8/trimmedVideos/alert/{trim_folder}/inputVideo.mp4
          3. {session_key}inputVideo.mp4

        Returns (s3_path, trim_start_ms).
        trim_start_ms is non-zero only for trimmed videos.
        """
        # Strategy 0: Direct path (from video_requests_v2)
        if video_s3_path and self.s3_object_exists(video_s3_path):
            logger.info(f"Video found at direct path: {video_s3_path}")
            return video_s3_path, 0.0

        if not video_http_path:
            return None, 0.0

        # Parse the HTTP videoPath for session info
        parsed = urlparse(video_http_path)
        bucket = parsed.hostname.split(".")[0]
        key = parsed.path.lstrip("/")

        # Extract trim folder if present
        trim_match = re.search(r"/trimmedVideos/alert/([^/]+)/", key)
        trim_folder = trim_match.group(1) if trim_match else None

        # Extract session key
        cam_match = re.search(r"^(.*?)/\d+/", key)
        if cam_match:
            session_key = cam_match.group(1) + "/"
        else:
            # Try before trimmedVideos or outputVideo
            alt_match = re.search(r"^(.*?)/(?:trimmedVideos|outputVideo)", key)
            if alt_match:
                session_key = alt_match.group(1) + "/"
            else:
                session_key = key[: key.rfind("/") + 1]

        # Strategy 1: /8/inputVideo.mp4
        s3_try = f"s3://{bucket}/{session_key}8/inputVideo.mp4"
        if self.s3_object_exists(s3_try):
            logger.info(f"Strategy 1 (8/inputVideo.mp4): {s3_try}")
            return s3_try, 0.0

        # Strategy 2: trimmedVideos
        if trim_folder:
            s3_try = f"s3://{bucket}/{session_key}8/trimmedVideos/alert/{trim_folder}/inputVideo.mp4"
            if self.s3_object_exists(s3_try):
                parts = trim_folder.split("_")
                trim_start_frame = int(parts[1]) if len(parts) > 1 else 0
                trim_start_ms = trim_start_frame / TRIM_VIDEO_FPS * 1000
                logger.info(
                    f"Strategy 2 (trimmedVideos/{trim_folder}): {s3_try} "
                    f"trim_start_ms={trim_start_ms:.0f}"
                )
                return s3_try, trim_start_ms

        # Strategy 3: root-level inputVideo.mp4
        s3_try = f"s3://{bucket}/{session_key}inputVideo.mp4"
        if self.s3_object_exists(s3_try):
            logger.info(f"Strategy 3 (root inputVideo.mp4): {s3_try}")
            return s3_try, 0.0

        logger.error(f"Could not locate video for session {session_key}")
        return None, 0.0
