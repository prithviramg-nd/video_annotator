"""Keypoint annotators: nose, shoulders (lsh/rsh), ears (lear/rear)."""

from PIL import Image, ImageDraw

from ..config import COLORS, KEYPOINT_RADIUS
from .base import BaseAnnotator


class NoseAnnotator(BaseAnnotator):
    name = "nose"
    description = "Draw nose keypoint with label"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        nose = detection.get("nose")
        if nose is None:
            return
        nx, ny = nose
        r = KEYPOINT_RADIUS
        draw.ellipse((nx - r, ny - r, nx + r, ny + r), fill=COLORS["nose"])
        font = self.get_small_font()
        draw.text((nx + 10, ny - 10), f"nose({nx},{ny})", fill=COLORS["nose"], font=font)


class ShoulderAnnotator(BaseAnnotator):
    name = "shoulders"
    description = "Draw shoulder keypoints (lsh, rsh) with labels and connecting lines to nose"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        r = KEYPOINT_RADIUS
        font = self.get_small_font()
        nose = detection.get("nose")

        lsh = detection.get("lsh")
        if lsh is not None:
            lx, ly = lsh
            draw.ellipse((lx - r, ly - r, lx + r, ly + r), fill=COLORS["lsh"])
            draw.text((lx + 10, ly - 20), f"lsh({lx},{ly})", fill=COLORS["lsh"], font=font)
            # Draw line from nose to lsh
            if nose is not None:
                nx, ny = nose
                draw.line((nx, ny, lx, ly), fill="orange", width=1)

        rsh = detection.get("rsh")
        if rsh is not None:
            rx, ry = rsh
            draw.ellipse((rx - r, ry - r, rx + r, ry + r), fill=COLORS["rsh"])
            draw.text((rx + 10, ry - 20), f"rsh({rx},{ry})", fill=COLORS["rsh"], font=font)
            # Draw line from nose to rsh
            if nose is not None:
                nx, ny = nose
                draw.line((nx, ny, rx, ry), fill="orange", width=1)


class EarAnnotator(BaseAnnotator):
    name = "ears"
    description = "Draw ear keypoints (lear, rear) with labels"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        r = KEYPOINT_RADIUS
        font = self.get_small_font()

        lear = detection.get("lear")
        if lear is not None:
            lx, ly = lear
            draw.ellipse((lx - r, ly - r, lx + r, ly + r), fill=COLORS["lear"])
            draw.text((lx + 10, ly - 10), f"lear", fill=COLORS["lear"], font=font)

        rear = detection.get("rear")
        if rear is not None:
            rx, ry = rear
            draw.ellipse((rx - r, ry - r, rx + r, ry + r), fill=COLORS["rear"])
            draw.text((rx + 10, ry - 10), f"rear", fill=COLORS["rear"], font=font)
