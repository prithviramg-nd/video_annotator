"""Mouth keypoints annotator."""

from PIL import Image, ImageDraw

from .base import BaseAnnotator


class MouthKeypointsAnnotator(BaseAnnotator):
    name = "mouth_kps"
    description = "Draw mouth keypoints (4 points: L, U, R, B) with V/H ratio"

    def annotate(self, img, draw, detection, frame_idx, total_frames, **kwargs):
        mouth_kps = detection.get("mouth_kps")
        if mouth_kps is None or len(mouth_kps) < 8:
            return

        font = self.get_small_font()
        pts = [(mouth_kps[j], mouth_kps[j + 1]) for j in range(0, 8, 2)]
        colors = ["cyan", "magenta", "cyan", "yellow"]
        labels = ["L", "U", "R", "B"]
        r = 5

        for pi, (px, py) in enumerate(pts):
            draw.ellipse((px - r, py - r, px + r, py + r), fill=colors[pi])
            draw.text((px + 8, py - 12), labels[pi], fill=colors[pi], font=font)

        # Horizontal and vertical lines
        draw.line((pts[0][0], pts[0][1], pts[2][0], pts[2][1]), fill="cyan", width=2)
        draw.line((pts[1][0], pts[1][1], pts[3][0], pts[3][1]), fill="yellow", width=2)

        # V/H ratio
        h_dist = abs(pts[2][0] - pts[0][0])
        v_dist = abs(pts[3][1] - pts[1][1])
        if h_dist > 1e-6:
            ratio = v_dist / h_dist
            draw.text(
                (pts[2][0] + 15, pts[2][1]),
                f"V/H: {ratio:.3f}",
                fill="white",
                font=font,
            )
