"""
S3 download/upload handler using AWS CLI.

Uses `aws s3 cp` and `aws s3 ls` subprocess calls instead of boto3.
This avoids SSL/fork issues on macOS when used with multiprocessing (p_tqdm).

Handles:
  - Downloading video files (inputVideo.mp4) with multi-strategy resolution
  - Downloading metadata.txt
  - Uploading annotated videos to S3
"""

import json
import os
import re
import subprocess
from typing import List, Optional, Tuple
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

    def _s3_list_keys(self, s3_prefix: str) -> List[str]:
        """
        List objects under an S3 prefix using `aws s3 ls --recursive`.
        Returns list of relative key suffixes (everything after the prefix).
        """
        # Ensure prefix ends with /
        if not s3_prefix.endswith("/"):
            s3_prefix += "/"
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", s3_prefix, "--recursive"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return []
            keys = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    keys.append(parts[-1])
            return keys
        except Exception as e:
            logger.warning(f"S3 list error for {s3_prefix}: {e}")
            return []

    def download_file(self, s3_path: str, local_path: str, alert_id: int = None) -> bool:
        """Download a file from S3 using `aws s3 cp`."""
        tag = f"[{alert_id}] " if alert_id else ""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, local_path],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"{tag}Downloaded {s3_path} -> {local_path}")
                return True
            else:
                logger.error(f"{tag}Download failed for {s3_path}: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"{tag}Download timed out for {s3_path}")
            return False
        except Exception as e:
            logger.error(f"{tag}Download failed for {s3_path}: {e}")
            return False

    def upload_file(self, local_path: str, s3_path: str, alert_id: int = None) -> bool:
        """Upload a local file to S3 using `aws s3 cp`."""
        tag = f"[{alert_id}] " if alert_id else ""
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
                logger.info(f"{tag}Uploaded {local_path} -> {s3_path}")
                return True
            else:
                logger.error(f"{tag}Upload failed for {s3_path}: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"{tag}Upload timed out for {s3_path}")
            return False
        except Exception as e:
            logger.error(f"{tag}Upload failed for {s3_path}: {e}")
            return False

    # Buckets to try: the original bucket from the HTTP URL, plus common
    # alternates (protected ↔ non-protected).
    _BUCKET_ALTERNATES = {
        "fleetdata-protected-production": "fleetdata-production",
        "fleetdata-production": "fleetdata-protected-production",
    }

    # Video filenames to look for when scanning subdirectories
    _DMS_VIDEO_NAMES = {"8.mp4", "8_transcoded.mp4", "dmsVideo.mp4"}

    def _resolve_subfolder_video(
        self, video_s3_path: str, alert_id: int = None,
    ) -> Optional[Tuple[str, float]]:
        """
        Handle SQP subfolder layout where 8.mp4 lives in a numbered
        subdirectory (e.g., s3://bucket/prefix/0/8.mp4 instead of
        s3://bucket/prefix/8.mp4).

        Lists files under the prefix and searches for DMS video files
        in any subdirectory.  Also checks for trimmed_video_params.json
        in the same subfolder to determine the trim start offset.

        Returns (resolved_s3_path, trim_start_ms) if found, else None.
        """
        tag = f"[{alert_id}] " if alert_id else ""

        # Parse video_s3_path: s3://bucket/prefix/8.mp4  ->  prefix = "bucket/prefix/"
        # Remove s3:// and split
        without_scheme = video_s3_path[len("s3://"):]
        bucket = without_scheme.split("/")[0]
        key = "/".join(without_scheme.split("/")[1:])
        # prefix_dir is everything up to the filename
        prefix_dir = key[: key.rfind("/") + 1]  # e.g. "N406.../uuid/"

        s3_prefix = f"s3://{bucket}/{prefix_dir}"
        keys = self._s3_list_keys(s3_prefix)
        if not keys:
            return None

        # Look for DMS video files in listed keys
        resolved = None
        subfolder = None
        for listed_key in keys:
            filename = listed_key.split("/")[-1]
            if filename in self._DMS_VIDEO_NAMES:
                resolved = f"s3://{bucket}/{listed_key}"
                # Extract subfolder (e.g., "0" from "prefix/0/8.mp4")
                rel_path = listed_key[len(prefix_dir):]  # "0/8.mp4"
                parts = rel_path.split("/")
                if len(parts) > 1:
                    subfolder = parts[0]
                break

        if resolved is None:
            return None

        # Check for trimmed_video_params.json in the subfolder
        trim_start_ms = 0.0
        if subfolder is not None:
            trim_params_key = f"{prefix_dir}{subfolder}/trimmed_video_params.json"
            if trim_params_key in keys:
                trim_start_ms = self._get_trim_start_from_params(
                    bucket, trim_params_key, tag,
                )

        logger.info(
            f"{tag}Strategy 0b (subfolder search): found {resolved}"
            + (f" (trim_start={trim_start_ms:.0f}ms)" if trim_start_ms > 0 else "")
        )
        return resolved, trim_start_ms

    def _get_trim_start_from_params(
        self, bucket: str, params_key: str, tag: str = "",
    ) -> float:
        """
        Download and parse trimmed_video_params.json to get the trim
        start offset for camera 8.

        Returns trim_start_ms (float), or 0.0 if not determinable.
        """
        s3_path = f"s3://{bucket}/{params_key}"
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, "-"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return 0.0
            data = json.loads(result.stdout)
            params = data.get("trimmed_video_params", {})
            # Look for camera 8 trim info
            cam8 = params.get("8", {})
            start_offset = cam8.get("start_offset")
            if start_offset is not None:
                logger.debug(
                    f"{tag}trimmed_video_params camera 8: "
                    f"start={start_offset}, end={cam8.get('end_offset')}"
                )
                return float(start_offset)
            return 0.0
        except Exception as e:
            logger.warning(f"{tag}Failed to read trimmed_video_params: {e}")
            return 0.0

    def resolve_video_path(
        self, video_s3_path: str, video_http_path: str = None,
        alert_id: int = None,
    ) -> Tuple[Optional[str], float]:
        """
        Resolve the actual video location on S3 using multiple strategies.

        For video_requests_v2 source: video_s3_path already points to
        8/inputVideo.mp4 directly.

        For alert source: video_http_path may point to trimmedVideos or
        outputVideo.mp4. We try (in both original and alternate buckets):
          1. {session_key}8/inputVideo.mp4
          2. {session_key}8/trimmedVideos/alert/{trim_folder}/inputVideo.mp4
          3. {session_key}inputVideo.mp4
          4. {session_key}8/previewVideos/alerts/{alert_id}/inputVideo.mp4

        Returns (s3_path, trim_start_ms).
        trim_start_ms is non-zero only for trimmed videos.
        """
        # Strategy 0: Direct path (from video_requests_v2)
        if video_s3_path and self.s3_object_exists(video_s3_path):
            logger.info(f"Video found at direct path: {video_s3_path}")
            return video_s3_path, 0.0

        # Strategy 0b: Subfolder search for nd-training-data-production
        # (SQP layout: 8.mp4 lives in a numbered subdirectory like 0/8.mp4)
        if video_s3_path:
            result = self._resolve_subfolder_video(video_s3_path, alert_id)
            if result:
                return result  # (s3_path, trim_start_ms)

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

        # Build list of buckets to try: original first, then alternate
        buckets = [bucket]
        alt_bucket = self._BUCKET_ALTERNATES.get(bucket)
        if alt_bucket:
            buckets.append(alt_bucket)

        # Try each strategy across ALL buckets before moving to the next
        # strategy.  This avoids picking a weaker match (e.g. root-level
        # inputVideo.mp4 in a protected bucket) over a stronger match
        # (8/inputVideo.mp4 in the alternate bucket).

        # Strategy 1: /8/inputVideo.mp4
        for try_bucket in buckets:
            s3_try = f"s3://{try_bucket}/{session_key}8/inputVideo.mp4"
            if self.s3_object_exists(s3_try):
                logger.info(f"Strategy 1 (8/inputVideo.mp4): {s3_try}")
                return s3_try, 0.0

        # Strategy 2: trimmedVideos
        if trim_folder:
            for try_bucket in buckets:
                s3_try = f"s3://{try_bucket}/{session_key}8/trimmedVideos/alert/{trim_folder}/inputVideo.mp4"
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
        for try_bucket in buckets:
            s3_try = f"s3://{try_bucket}/{session_key}inputVideo.mp4"
            if self.s3_object_exists(s3_try):
                logger.info(f"Strategy 3 (root inputVideo.mp4): {s3_try}")
                return s3_try, 0.0

        # Strategy 4: previewVideos (short clips stored per alert)
        if alert_id:
            for try_bucket in buckets:
                s3_try = f"s3://{try_bucket}/{session_key}8/previewVideos/alerts/{alert_id}/inputVideo.mp4"
                if self.s3_object_exists(s3_try):
                    logger.info(f"Strategy 4 (previewVideos): {s3_try}")
                    return s3_try, 0.0

        tag = f"[{alert_id}] " if alert_id else ""
        logger.error(f"{tag}Could not locate video for session {session_key}")
        return None, 0.0
