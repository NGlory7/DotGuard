"""
DotGuard - Threat Visualizer
Generates rich annotated side-by-side inspection images showing:
1) Raw Rasterized Dots
2) Morphologically Reconstituted Silhouettes
3) Detected Threat Bounding Boxes & Confidence Labels
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np

class Visualizer:
    SEVERITY_COLORS = {
        "CRITICAL": (0, 0, 255),    # Red
        "HIGH": (0, 140, 255),       # Orange
        "MEDIUM": (0, 220, 255),     # Yellow
        "LOW": (255, 200, 0),        # Cyan
        "BENIGN": (0, 255, 0)        # Green
    }

    def create_annotated_panel(self, raw_img: np.ndarray, binary_img: np.ndarray, threats: List[Dict[str, Any]]) -> np.ndarray:
        """
        Creates a side-by-side 3-panel comparison image:
        [Raw Dots] | [Morphed Silhouette] | [Annotated Detections]
        """
        h, w = raw_img.shape[:2]

        # Convert grayscale images to 3-channel BGR
        raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR) if len(raw_img.shape) == 2 else raw_img.copy()
        morph_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR) if len(binary_img.shape) == 2 else binary_img.copy()
        annotated_bgr = morph_bgr.copy()

        # Draw detections on annotated panel
        for threat in threats:
            bbox = threat.get("bbox")
            if not bbox:
                continue
            x, y, bw, bh = bbox
            sev = threat.get("severity", "MEDIUM")
            color = self.SEVERITY_COLORS.get(sev, (0, 0, 255))
            label = f"{threat.get('label', 'THREAT')} ({threat.get('confidence', 0.0):.2f})"

            # Draw rectangle
            cv2.rectangle(annotated_bgr, (x, y), (x + bw, y + bh), color, 2)

            # Draw background tag for text
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.4
            thick = 1
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            tag_y1 = max(0, y - th - 6)
            tag_y2 = y
            cv2.rectangle(annotated_bgr, (x, tag_y1), (x + tw + 6, tag_y2), color, -1)
            cv2.putText(annotated_bgr, label, (x + 3, tag_y2 - 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Add headers to each panel
        header_h = 35
        panels = [
            ("1. Raw Rasterized Dots", raw_bgr),
            ("2. Morphological Silhouette", morph_bgr),
            ("3. Threat Detections", annotated_bgr)
        ]

        canvas_w = w * 3 + 40
        canvas_h = h + header_h + 20
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 25) # Dark gray background

        for idx, (title, panel_img) in enumerate(panels):
            x_offset = 10 + idx * (w + 10)
            y_offset = header_h + 10

            # Draw header
            cv2.putText(canvas, title, (x_offset, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = panel_img

        return canvas

    def save_analysis_image(self, raw_img: np.ndarray, binary_img: np.ndarray, threats: List[Dict[str, Any]], output_path: str):
        canvas = self.create_annotated_panel(raw_img, binary_img, threats)
        cv2.imwrite(output_path, canvas)
