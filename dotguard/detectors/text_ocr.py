"""
DotGuard - Visual Word & Letter Detector
Detects if the dot/ASCII art is spelling out text, words, or bypass phrases.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

class TextOCRDetector:
    def __init__(self):
        self._tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        try:
            import pytesseract
            # Test if tesseract binary responds
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def detect_text_lines_heuristic(self, contours: List[np.ndarray], features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects letter-like connected components arranged horizontally in lines.
        Even without OCR, this flags when dot art is being used to write hidden words.
        """
        if len(features) < 3:
            return []

        # Filter components with letter-like dimensions (aspect ratio between 0.2 and 1.8)
        letter_candidates = []
        for feat in features:
            ar = feat["aspect_ratio"]
            w, h = feat["bbox"][2], feat["bbox"][3]
            if 0.15 <= ar <= 2.2 and h >= 12 and w >= 6:
                letter_candidates.append(feat)

        if len(letter_candidates) < 3:
            return []

        # Sort left to right
        letter_candidates.sort(key=lambda f: f["bbox"][0])

        # Cluster by baseline / Y-coordinate
        lines = []
        for cand in letter_candidates:
            cy = cand["bbox"][1] + cand["bbox"][3] / 2
            h = cand["bbox"][3]
            placed = False
            for line in lines:
                line_cy = line["avg_cy"]
                line_h = line["avg_h"]
                if abs(cy - line_cy) < (line_h * 0.4):
                    line["items"].append(cand)
                    # Update average
                    line["avg_cy"] = sum(c["bbox"][1] + c["bbox"][3] / 2 for c in line["items"]) / len(line["items"])
                    line["avg_h"] = sum(c["bbox"][3] for c in line["items"]) / len(line["items"])
                    placed = True
                    break
            if not placed:
                lines.append({"avg_cy": cy, "avg_h": h, "items": [cand]})

        findings = []
        for line in lines:
            items = line["items"]
            if len(items) >= 3:
                # Calculate bounding box of the entire text line
                min_x = min(c["bbox"][0] for c in items)
                min_y = min(c["bbox"][1] for c in items)
                max_x = max(c["bbox"][0] + c["bbox"][2] for c in items)
                max_y = max(c["bbox"][1] + c["bbox"][3] for c in items)

                findings.append({
                    "category": "TEXT_SPELLED_OUT",
                    "label": f"Hidden Word/Text Detected ({len(items)} characters visual line)",
                    "character_count": len(items),
                    "confidence": 0.85,
                    "bbox": (min_x, min_y, max_x - min_x, max_y - min_y),
                    "severity": "MEDIUM"
                })

        return findings

    def run_ocr_if_available(self, binary_img: np.ndarray) -> str:
        """Runs pytesseract if available on the system."""
        if not self._tesseract_available:
            return ""
        try:
            import pytesseract
            # White text on black bg or inverted for tesseract
            inv = cv2.bitwise_not(binary_img)
            text = pytesseract.image_to_string(inv, config="--psm 6")
            return text.strip()
        except Exception:
            return ""

    def analyze(self, binary_img: np.ndarray, contours: List[np.ndarray], features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_lines = self.detect_text_lines_heuristic(contours, features)
        ocr_text = self.run_ocr_if_available(binary_img)

        return {
            "text_detected": len(text_lines) > 0 or bool(ocr_text),
            "lines": text_lines,
            "ocr_extracted_text": ocr_text if ocr_text else None
        }
