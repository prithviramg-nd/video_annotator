#!/usr/bin/env python3
"""
video_annotator - CLI tool for downloading, annotating, and uploading alert videos.

Usage:
    # Process alert IDs from CSV with specific annotations
    python -m video_annotator.main --csv alerts.csv --annotate face_bbox nose shoulders

    # Process a single alert ID with all annotations
    python -m video_annotator.main --alert-id 5815863985 --annotate all

    # List available annotators
    python -m video_annotator.main --list-annotators

    # Process with only frame_info and event_window (no detection overlays)
    python -m video_annotator.main --csv alerts.csv --annotate frame_info event_window
"""

# Allow running directly: python main.py --help
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "video_annotator"

import argparse
import multiprocessing
import os
import re
import shutil
import sys
from typing import List, Optional

import p_tqdm
import pandas as pd
from loguru import logger
from PIL import Image, ImageDraw

from .annotators.registry import AnnotatorRegistry
from .config import FPS, LOG_FILE, TEMP_DIR, VIDEO_OFFSET_MS, load_credentials
from .data_sources.base import AlertData
from .data_sources.mongodb import MongoDBSource
from .downloader.s3 import S3Handler
from .metadata.parser import MetadataParser
from .video.assembler import VideoAssembler
from .video.extractor import FrameExtractor


# ── Process-local connections ────────────────────────────────────────────────
# Each worker process creates its own S3 client lazily.
# MongoDB is queried once in the main process before parallelization.

_worker_s3_handler: Optional[S3Handler] = None


def _get_s3_handler() -> S3Handler:
    """Get or create a process-local S3 client."""
    global _worker_s3_handler
    if _worker_s3_handler is None:
        _worker_s3_handler = S3Handler()
    return _worker_s3_handler


# ── Pipeline ─────────────────────────────────────────────────────────────────

def process_alert(
    alert_data: AlertData,
    annotator_names: List[str],
    output_dir: str,
    s3_upload_path: Optional[str] = None,
    fps: int = FPS,
    video_offset_ms: int = VIDEO_OFFSET_MS,
) -> bool:
    """
    Full pipeline for one alert (MongoDB data already fetched):
      1. Download video and metadata from S3
      2. Parse metadata for event timing and detections
      3. Extract frames
      4. Annotate each frame with selected annotators
      5. Reassemble into video
      6. Optionally upload to S3

    Returns True on success.
    """
    alert_id = alert_data.alert_id
    s3_handler = _get_s3_handler()

    logger.info(f"{'='*60}")
    logger.info(f"Processing alert_id={alert_id}")
    logger.info(f"{'='*60}")
    logger.info(
        f"[{alert_id}] Source: {alert_data.source}, "
        f"event_code: {alert_data.event_code}"
    )

    # ── 1. Download video and metadata ───────────────────────────────────────
    work_dir = os.path.join(TEMP_DIR, str(alert_id))
    os.makedirs(work_dir, exist_ok=True)

    # Resolve and download video
    video_s3, trim_start_ms = s3_handler.resolve_video_path(
        alert_data.video_s3_path,
        alert_data.video_http_path,
        alert_id=alert_id,
    )
    if video_s3 is None:
        logger.error(f"[{alert_id}] Could not locate video on S3")
        _cleanup(work_dir)
        return False

    local_video = os.path.join(work_dir, "inputVideo.mp4")
    if not s3_handler.download_file(video_s3, local_video, alert_id=alert_id):
        logger.error(f"[{alert_id}] Video download failed")
        _cleanup(work_dir)
        return False

    # Download metadata
    if alert_data.metadata_s3_path is None:
        logger.error(f"[{alert_id}] No metadata path available")
        _cleanup(work_dir)
        return False

    local_metadata = os.path.join(work_dir, "metadata.txt")
    if not s3_handler.download_file(alert_data.metadata_s3_path, local_metadata, alert_id=alert_id):
        logger.error(f"[{alert_id}] Metadata download failed")
        _cleanup(work_dir)
        return False

    # ── 3. Parse metadata ────────────────────────────────────────────────────
    try:
        parser = MetadataParser(local_metadata)
        detections = parser.get_detections(alert_id=alert_id)
        if not detections:
            logger.error(f"[{alert_id}] No detections found in metadata")
            _cleanup(work_dir)
            return False
    except Exception as e:
        logger.error(f"[{alert_id}] Metadata parse error: {e}")
        _cleanup(work_dir)
        return False

    # Find event timing
    event_code = alert_data.event_code
    event = parser.get_event_by_code(event_code, alert_id=alert_id)

    # Determine start/end offsets
    if event is not None:
        start_offset_ms = event.start_offset
        end_offset_ms = event.end_offset
        logger.info(
            f"[{alert_id}] Event {event_code}: "
            f"{start_offset_ms}ms - {end_offset_ms}ms"
        )
    elif alert_data.start_offset is not None and alert_data.end_offset is not None:
        start_offset_ms = alert_data.start_offset
        end_offset_ms = alert_data.end_offset
        logger.info(
            f"[{alert_id}] Using offsets from alert collection: "
            f"{start_offset_ms}ms - {end_offset_ms}ms"
        )
    else:
        # No event timing - process entire video
        start_offset_ms = 0
        end_offset_ms = len(detections) * (1000 // fps)
        logger.warning(
            f"[{alert_id}] No event timing found, using full video"
        )

    # ── 4. Extract frames (segment around event) ───────────────────────────
    # Only extract video_offset_ms before and after the event to reduce
    # output size (e.g. video_offset_ms(5s) + event_duration + video_offset_ms(5s) instead of full 1min).
    frames_dir = os.path.join(work_dir, "frames")
    try:
        extractor = FrameExtractor(fps=fps)

        # Calculate segment boundaries in the video's own timeline
        adj_event_start = max(0, start_offset_ms - trim_start_ms)
        adj_event_end = end_offset_ms - trim_start_ms

        video_duration_ms = FrameExtractor.get_video_duration_ms(local_video)

        segment_start_ms = max(0, adj_event_start - video_offset_ms)
        segment_end_ms = min(video_duration_ms, adj_event_end + video_offset_ms)
        segment_duration_ms = segment_end_ms - segment_start_ms

        logger.info(
            f"[{alert_id}] Extracting segment: "
            f"{segment_start_ms:.0f}ms - {segment_end_ms:.0f}ms "
            f"(duration: {segment_duration_ms:.0f}ms, "
            f"event: {adj_event_start}ms - {adj_event_end}ms, "
            f"padding: {video_offset_ms}ms)"
        )

        frame_files = extractor.extract_segment(
            local_video, frames_dir,
            start_ms=segment_start_ms,
            duration_ms=segment_duration_ms,
            alert_id=alert_id,
        )
        if not frame_files:
            logger.error(f"[{alert_id}] No frames extracted")
            _cleanup(work_dir)
            return False

        # Slice detections to match the extracted segment
        seg_start_frame = int(segment_start_ms * fps / 1000)
        seg_end_frame = int(segment_end_ms * fps / 1000)
        detections = detections[seg_start_frame:seg_end_frame]

    except Exception as e:
        logger.error(f"[{alert_id}] Frame extraction failed: {e}")
        _cleanup(work_dir)
        return False

    # ── 5. Annotate frames ───────────────────────────────────────────────────
    try:
        annotators = AnnotatorRegistry.get(annotator_names)
    except ValueError as e:
        logger.error(f"[{alert_id}] {e}")
        _cleanup(work_dir)
        return False

    # Calculate event frame indices relative to the extracted segment
    event_start_frame = int((adj_event_start - segment_start_ms) * fps / 1000)
    event_end_frame = int((adj_event_end - segment_start_ms) * fps / 1000)

    num_frames = min(len(frame_files), len(detections))
    logger.info(
        f"[{alert_id}] Annotating {num_frames} frames "
        f"(event frames: {event_start_frame}-{event_end_frame}), "
        f"annotators: {annotator_names}"
    )

    for idx in range(num_frames):
        detection = detections[idx]
        frame_path = frame_files[idx]

        try:
            with Image.open(frame_path) as img:
                draw = ImageDraw.Draw(img)
                for annotator in annotators:
                    annotator.annotate(
                        img=img,
                        draw=draw,
                        detection=detection,
                        frame_idx=idx,
                        total_frames=num_frames,
                        event_start_frame=event_start_frame,
                        event_end_frame=event_end_frame,
                    )
                img.save(frame_path)
        except Exception as e:
            logger.error(f"[{alert_id}] Error annotating frame {idx}: {e}")

    # ── 6. Assemble video ────────────────────────────────────────────────────
    video_name = f"{alert_id}_{event_code}_{event_start_frame}_{event_end_frame}.mp4"
    output_video = os.path.join(output_dir, video_name)
    try:
        assembler = VideoAssembler(fps=fps)
        assembler.assemble(frames_dir, output_video, alert_id=alert_id)
    except Exception as e:
        logger.error(f"[{alert_id}] Video assembly failed: {e}")
        _cleanup(work_dir)
        return False

    logger.info(f"[{alert_id}] Annotated video: {output_video}")

    # ── 7. Upload to S3 ─────────────────────────────────────────────────────
    if s3_upload_path:
        s3_dest = f"{s3_upload_path.rstrip('/')}/{video_name}"
        if s3_handler.upload_file(output_video, s3_dest, alert_id=alert_id):
            logger.info(f"[{alert_id}] Uploaded to {s3_dest}")
            # Remove local copy after successful upload
            try:
                os.remove(output_video)
                logger.debug(f"[{alert_id}] Removed local file: {output_video}")
            except OSError as e:
                logger.warning(f"[{alert_id}] Could not remove local file: {e}")
        else:
            logger.error(f"[{alert_id}] Upload failed, keeping local file: {output_video}")

    # ── 8. Cleanup temp files ────────────────────────────────────────────────
    _cleanup(work_dir)

    # ── 9. Done ───────────────────────────────────────────────────────────────
    logger.info(f"[{alert_id}] Done")

    return True


def _cleanup(path: str):
    """Remove temporary directory."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"Cleanup failed for {path}: {e}")


# ── S3 credential pre-flight ────────────────────────────────────────────────

def _preflight_s3_check() -> bool:
    """
    Verify S3 access works before spawning workers.
    If access fails, prompt for credentials in the main process so that
    child processes inherit the env vars.
    Returns True if S3 is accessible.
    """
    import subprocess

    def _s3_accessible() -> bool:
        try:
            result = subprocess.run(
                ["aws", "s3", "ls"],
                capture_output=True, text=True, timeout=15,
            )
            return result.returncode == 0
        except Exception:
            return False

    if _s3_accessible():
        logger.info("S3 access verified")
        return True

    logger.warning("S3 access failed")
    print("\n" + "=" * 60)
    print("AWS credentials required for S3 access.")
    print("Paste your credentials below (all three export lines).")
    print("Press Enter on an empty line when done.")
    print("=" * 60)

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line.strip())

    for line in lines:
        match = re.match(r'export\s+(\w+)=["\']?(.*?)["\']?\s*$', line)
        if match:
            key, value = match.group(1), match.group(2)
            os.environ[key] = value
            logger.info(f"Set {key} from user input")

    # Re-verify
    if _s3_accessible():
        logger.info("S3 access verified after credential update")
        return True

    logger.error("S3 access still failing")
    return False


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, annotate, and upload alert videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process one alert with face bbox and keypoints
  python -m video_annotator.main --alert-id 5815863985 --annotate face_bbox nose shoulders

  # Process alerts from CSV with all annotations
  python -m video_annotator.main --csv alerts.csv --annotate all

  # Process and upload to S3
  python -m video_annotator.main --csv alerts.csv --annotate face_bbox --s3-upload s3://bucket/prefix/

  # Control parallelism
  python -m video_annotator.main --csv alerts.csv --annotate all --workers 4

  # List available annotation types
  python -m video_annotator.main --list-annotators
""",
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--alert-id",
        type=int,
        help="Single alert ID to process",
    )
    input_group.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file containing alert IDs (column: alert_id)",
    )

    # Annotation configuration
    parser.add_argument(
        "--annotate",
        nargs="+",
        default=["frame_info", "event_window"],
        help=(
            "Annotation types to apply. Use 'all' for everything. "
            "Default: frame_info event_window. "
            "Available: " + ", ".join(AnnotatorRegistry.available())
        ),
    )
    parser.add_argument(
        "--list-annotators",
        action="store_true",
        help="List all available annotator types and exit",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(TEMP_DIR, "annotated_videos"),
        help="Local directory for output videos (default: temp/annotated_videos)",
    )
    parser.add_argument(
        "--s3-upload",
        type=str,
        default=None,
        help="S3 path to upload annotated videos (e.g. s3://bucket/prefix/)",
    )

    # Processing options
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help=f"Frames per second for extraction (default: {FPS})",
    )
    parser.add_argument(
        "--video-offset",
        type=int,
        default=VIDEO_OFFSET_MS,
        help=f"Padding around event window in ms (default: {VIDEO_OFFSET_MS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help=f"Number of parallel workers (default: {multiprocessing.cpu_count()} = all CPUs)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: stdout only)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--credentials-file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "my_creds.json"),
        help=(
            "Path to JSON file with database credentials "
            "(must contain 'mongo_uri'). If not provided, you will be "
            "prompted to enter the path interactively."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── List annotators ──────────────────────────────────────────────────────
    if args.list_annotators:
        print("\nAvailable annotators:")
        print("-" * 50)
        for name, desc in sorted(AnnotatorRegistry.describe().items()):
            print(f"  {name:20s} {desc}")
        print()
        sys.exit(0)

    # ── Validate input ───────────────────────────────────────────────────────
    if args.alert_id is None and args.csv is None:
        parser.error("Provide either --alert-id or --csv")

    # ── Configure logging ────────────────────────────────────────────────────
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)

    # Always log to temp/video_annotator.log
    os.makedirs(TEMP_DIR, exist_ok=True)
    log_dest = args.log_file or LOG_FILE
    logger.add(log_dest, level="DEBUG", mode="w")
    logger.info(f"Log file: {log_dest}")

    # ── Load database credentials ──────────────────────────────────────────
    creds_path = args.credentials_file
    if creds_path is None:
        print("\nEnter path to credentials JSON file")
        print("  (must contain 'mongo_uri', optionally 'postgres_uri'):")
        creds_path = input("> ").strip()
        if not creds_path:
            logger.error("No credentials file provided")
            sys.exit(1)
    credentials = load_credentials(creds_path)
    logger.info(f"Loaded credentials from: {creds_path}")

    # ── S3 upload path ───────────────────────────────────────────────────────
    s3_upload = args.s3_upload
    if s3_upload is None:
        print("\nEnter S3 path for uploading annotated videos")
        print("  e.g. s3://netradyne-sharing/analytics/prithvi/annotated/")
        print("  (press Enter to skip S3 upload):")
        s3_input = input("> ").strip().strip("\r\n\t ")
        if s3_input:
            s3_upload = s3_input
            logger.info(f"S3 upload path: {s3_upload}")
        else:
            logger.info("S3 upload skipped (videos saved locally only)")

    # ── Pre-flight S3 check ──────────────────────────────────────────────────
    # Validate credentials in main process BEFORE forking workers.
    # Env vars set here are inherited by all child processes.
    if not _preflight_s3_check():
        logger.error("Cannot proceed without S3 access (needed for downloading videos)")
        sys.exit(1)

    # ── Resolve annotators ───────────────────────────────────────────────────
    if "all" in args.annotate:
        annotator_names = AnnotatorRegistry.available()
    else:
        annotator_names = args.annotate

    # Validate annotator names
    try:
        AnnotatorRegistry.get(annotator_names)
    except ValueError as e:
        parser.error(str(e))

    # ── Collect alert IDs ────────────────────────────────────────────────────
    if args.alert_id:
        alert_ids = [args.alert_id]
    else:
        df = pd.read_csv(args.csv)
        # Support both 'alert_id' and 'alertId' column names
        col = None
        for candidate in ["alert_id", "alertId", "alert_ids", "ALERT_ID"]:
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            parser.error(
                f"CSV must contain an 'alert_id' column. "
                f"Found columns: {list(df.columns)}"
            )
        alert_ids = df[col].dropna().astype(int).tolist()

    # ── Fetch alert data from MongoDB (batch query) ────────────────────────
    logger.info(f"Fetching alert data from MongoDB for {len(alert_ids)} alert(s)...")
    data_source = MongoDBSource(uri=credentials["mongo_uri"])
    results_map = data_source.fetch_batch(alert_ids)
    data_source.close()

    alert_data_list = [results_map[aid] for aid in alert_ids if aid in results_map]
    not_found = [aid for aid in alert_ids if aid not in results_map]

    if not_found:
        logger.warning(
            f"{len(not_found)} alert(s) not found in MongoDB: {not_found}"
        )
    if not alert_data_list:
        logger.error("No alerts found in MongoDB. Nothing to process.")
        sys.exit(1)

    total = len(alert_data_list)
    num_workers = min(args.workers, total)

    logger.info(f"Processing {total} alert(s) with {num_workers} workers")
    logger.info(f"Annotators: {annotator_names}")
    logger.info(f"Output: {args.output_dir}")
    if s3_upload:
        logger.info(f"S3 upload: {s3_upload}")

    # ── Setup ────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # ── Process alerts in parallel ───────────────────────────────────────────
    def _worker(alert_data: AlertData) -> bool:
        """Worker function for p_tqdm. Only needs S3 (no MongoDB)."""
        try:
            return process_alert(
                alert_data=alert_data,
                annotator_names=annotator_names,
                output_dir=args.output_dir,
                s3_upload_path=s3_upload,
                fps=args.fps,
                video_offset_ms=args.video_offset,
            )
        except Exception as e:
            logger.error(f"Unhandled error for alert {alert_data.alert_id}: {e}")
            return False

    if total == 1:
        # Single alert: run directly (no multiprocessing overhead)
        results = [_worker(alert_data_list[0])]
    else:
        # Parallel processing with progress bar
        results = p_tqdm.p_map(
            _worker,
            alert_data_list,
            num_cpus=num_workers,
            desc="Annotating videos",
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    success_count = sum(1 for r in results if r)
    fail_count = total - success_count

    logger.info(f"{'='*60}")
    logger.info(f"Done. Success: {success_count}, Failed: {fail_count}, Total: {total}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
