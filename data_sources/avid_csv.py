"""
AVID CSV data source.

Reads a CSV with columns: avid, [event_code], [start_offset], [end_offset], [json_path]
Uses the AVC API (same as syncalert.py) to resolve S3 paths for each AVID,
then downloads the DMS video (dmsVideo.mp4 / 8.mp4) using AWS CLI (aws s3 cp/ls).

Clipping and annotation are independent features toggled by CSV columns:
  - start_offset + end_offset present  ->  clip video to event window
  - json_path present                  ->  annotate frames with detections
  - Both, either, or neither can be provided independently.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ── AVC API config ──────────────────────────────────────────────────────────
AVC_API_ENDPOINT = "https://analytics-kpis.netradyne.info/avc_api"


@dataclass
class AvidData:
    """Data for one AVID entry from the CSV."""
    avid: str
    event_code: Optional[str] = None
    start_offset: Optional[int] = None        # ms
    end_offset: Optional[int] = None          # ms
    json_path: Optional[str] = None           # local path to summary.json
    video_local_path: Optional[str] = None    # set after download

    @property
    def has_event_code(self) -> bool:
        return self.event_code is not None and str(self.event_code).strip() != ""

    @property
    def should_clip(self) -> bool:
        """True if both start_offset and end_offset are present."""
        return self.start_offset is not None and self.end_offset is not None

    @property
    def should_annotate(self) -> bool:
        """True if json_path is present."""
        return self.json_path is not None and str(self.json_path).strip() != ""


def _requests_retry_session(
    retries=5,
    backoff_factor=1,
    status_forcelist=(400, 429, 500, 502, 503, 504),
    session=None,
):
    """Retry strategy for requests (from syncalert.py)."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def query_avc_api(avid: str, env: str = "production") -> Optional[dict]:
    """
    Call the AVC API with an AVID to get S3 paths.

    Returns the API response dict on success, None on failure.
    """
    api_env = "secondary" if env == "staging" else "primary"
    params = {
        "input_data": {"avid": avid},
        "anonymize_environment": api_env,
        "source": "debug",
        "api_version": "v2",
    }
    logger.info(f"[{avid}] Querying AVC API...")
    try:
        response = _requests_retry_session(retries=5).post(
            AVC_API_ENDPOINT, json=params, timeout=60,
        )
        result = response.json()
        if result.get("msg") == "success":
            logger.info(f"[{avid}] AVC API success")
            return result
        else:
            logger.error(f"[{avid}] AVC API returned: {result.get('msg', result)}")
            return None
    except Exception as e:
        logger.error(f"[{avid}] AVC API error: {e}")
        return None


def _s3_ls(s3_path: str) -> List[str]:
    """List objects at an S3 path using `aws s3 ls`. Returns list of keys."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path, "--recursive"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
        # Each line: "2024-01-01 00:00:00  12345 path/to/file"
        keys = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                keys.append(parts[-1])
        return keys
    except Exception as e:
        logger.warning(f"aws s3 ls failed for {s3_path}: {e}")
        return []


def _s3_cp(s3_path: str, local_path: str, avid: str = "") -> bool:
    """Download a file from S3 using `aws s3 cp`."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, local_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            logger.info(f"[{avid}] Downloaded {s3_path} -> {local_path}")
            return True
        else:
            logger.error(f"[{avid}] Download failed for {s3_path}: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"[{avid}] Download timed out for {s3_path}")
        return False
    except Exception as e:
        logger.error(f"[{avid}] Download failed for {s3_path}: {e}")
        return False


def _s3_object_exists(s3_path: str) -> bool:
    """Check if an S3 object exists using `aws s3 ls`."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


def download_dms_video(avid: str, api_result: dict, download_dir: str) -> Optional[str]:
    """
    Download the DMS video (dmsVideo.mp4 / 8.mp4) for an AVID using AWS CLI.
    Mimics syncalert.py's download_data() approach but uses subprocess aws s3 cp/ls.

    Strategy:
      1. Get s3_bucket from AVC API result
      2. List files in the S3 prefix
      3. Look for dmsVideo.mp4 or 8.mp4 (at root or in subdirectories)
      4. Download using aws s3 cp

    Returns the local path to the downloaded video, or None on failure.
    """
    s3_path_list = api_result.get("s3_bucket")
    if s3_path_list is None:
        logger.error(f"[{avid}] No s3_bucket in API response")
        return None

    if not isinstance(s3_path_list, list):
        s3_path_list = [s3_path_list]

    for s3_path in s3_path_list:
        s3_tokens = s3_path.split("/")
        bucket_name = s3_tokens[0]
        s3_prefix = "/".join(s3_tokens[1:]).rstrip("/")
        s3_base = f"s3://{bucket_name}/{s3_prefix}"

        logger.info(f"[{avid}] Looking for DMS video in {s3_base}")

        # Try direct paths first (faster than listing)
        # Priority: 8.mp4 at root, then dmsVideo.mp4 at root
        for filename in ("8.mp4", "dmsVideo.mp4"):
            s3_direct = f"{s3_base}/{filename}"
            if _s3_object_exists(s3_direct):
                local_path = os.path.join(download_dir, "dmsVideo.mp4")
                if _s3_cp(s3_direct, local_path, avid=avid):
                    return local_path

        # If direct paths fail, list all files and search
        keys = _s3_ls(s3_base + "/")
        if not keys:
            logger.warning(f"[{avid}] No files found in {s3_base}")
            continue

        # Search for DMS video in any subdirectory
        dms_key = None
        for key in keys:
            filename = key.split("/")[-1]
            if filename in ("dmsVideo.mp4", "8.mp4"):
                dms_key = key
                break

        if dms_key is None:
            available = [k.split("/")[-1] for k in keys]
            logger.warning(
                f"[{avid}] No DMS video found in {s3_base}. "
                f"Available files: {available}"
            )
            continue

        # Download
        s3_full = f"s3://{bucket_name}/{dms_key}"
        local_path = os.path.join(download_dir, "dmsVideo.mp4")
        if _s3_cp(s3_full, local_path, avid=avid):
            return local_path

    logger.error(f"[{avid}] Could not find/download DMS video from any S3 path")
    return None


def parse_avid_csv(csv_path: str) -> List[AvidData]:
    """
    Parse the AVID CSV file into a list of AvidData.

    Supports columns: event_code, start_offset, end_offset, avid, json_path
    All columns except 'avid' are optional.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Validate required column
    if "avid" not in df.columns:
        raise ValueError(f"CSV must contain an 'avid' column. Found: {list(df.columns)}")

    results = []
    for _, row in df.iterrows():
        avid = str(row["avid"]).strip()
        if not avid:
            continue

        # Optional columns
        event_code = None
        if "event_code" in df.columns:
            val = row.get("event_code")
            if pd.notna(val) and str(val).strip():
                event_code = str(val).strip()

        start_offset = None
        if "start_offset" in df.columns:
            val = row.get("start_offset")
            if pd.notna(val):
                start_offset = int(val)

        end_offset = None
        if "end_offset" in df.columns:
            val = row.get("end_offset")
            if pd.notna(val):
                end_offset = int(val)

        json_path = None
        if "json_path" in df.columns:
            val = row.get("json_path")
            if pd.notna(val) and str(val).strip():
                json_path = str(val).strip()

        results.append(AvidData(
            avid=avid,
            event_code=event_code,
            start_offset=start_offset,
            end_offset=end_offset,
            json_path=json_path,
        ))

    n_clip = sum(1 for r in results if r.should_clip)
    n_annot = sum(1 for r in results if r.should_annotate)
    logger.info(
        f"Parsed {len(results)} entries from {csv_path} "
        f"(clip: {n_clip}, annotate: {n_annot})"
    )
    return results
