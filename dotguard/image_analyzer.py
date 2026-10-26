"""
DotGuard - Direct Image Threat Analyzer
Unified pipeline for scanning real image files (.jpg, .png) for Weapons, NSFW, and Threats.
"""

from typing import Dict, Any, Optional, Union
import os
import cv2
import numpy as np

from .detectors.image_nsfw import ImageNSFWDetector
from .detectors.image_weapon import ImageWeaponDetector
from .detectors.image_text_threat import ImageTextThreatDetector

class ImageThreatAnalyzer:
    SEVERITY_COLORS = {
        "CRITICAL": (0, 0, 255),    # Red
        "HIGH": (0, 140, 255),       # Orange
        "MEDIUM": (0, 220, 255),     # Yellow
        "LOW": (255, 200, 0),        # Cyan
        "BENIGN": (0, 255, 0)        # Green
    }

    def __init__(self):
        self.nsfw_detector = ImageNSFWDetector()
        self.weapon_detector = ImageWeaponDetector()
        self.text_threat_detector = ImageTextThreatDetector()

    def analyze_image(self, image_input: Union[str, np.ndarray], save_visual_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Scans a direct image (file path or OpenCV BGR numpy array) for threats.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {
                    "status": "ERROR_FILE_NOT_FOUND",
                    "file_path": image_input,
                    "threat_score": 0.0,
                    "risk_level": "ERROR"
                }
            img = cv2.imread(image_input)
            if img is None:
                return {
                    "status": "ERROR_INVALID_IMAGE",
                    "file_path": image_input,
                    "threat_score": 0.0,
                    "risk_level": "ERROR"
                }
        else:
            img = image_input

        h, w = img.shape[:2]

        # 1. NSFW Analysis
        nsfw_res = self.nsfw_detector.analyze(img)

        # 2. Weapon Analysis
        weapon_res = self.weapon_detector.analyze(img)

        # 3. Text & Threat Keywords
        text_res = self.text_threat_detector.analyze(img)

        # Compile all threats
        threats = []
        threats.extend(nsfw_res.get("findings", []))
        threats.extend(weapon_res.get("findings", []))
        threats.extend(text_res.get("threat_keywords", []))

        # Calculate Threat Score (0 - 100)
        threat_score = 0.0
        for t in threats:
            sev = t.get("severity", "LOW")
            conf = t.get("confidence", 0.5)
            if sev == "CRITICAL":
                threat_score = max(threat_score, 90.0 + conf * 10.0)
            elif sev == "HIGH":
                threat_score = max(threat_score, 70.0 + conf * 15.0)
            elif sev == "MEDIUM":
                threat_score = max(threat_score, 45.0 + conf * 15.0)
            elif sev == "LOW":
                threat_score = max(threat_score, 20.0 + conf * 10.0)

        threat_score = min(100.0, round(threat_score, 1))

        if threat_score >= 85.0:
            risk_level = "CRITICAL"
        elif threat_score >= 65.0:
            risk_level = "HIGH"
        elif threat_score >= 40.0:
            risk_level = "MEDIUM"
        elif threat_score > 0.0:
            risk_level = "LOW"
        else:
            risk_level = "CLEAN"

        # Generate annotated visual report if requested
        if save_visual_path:
            self._save_annotated_image(img, threats, save_visual_path)

        return {
            "status": "ANALYZED",
            "media_type": "DIRECT_IMAGE",
            "resolution": {"width": w, "height": h},
            "risk_level": risk_level,
            "threat_score": threat_score,
            "nsfw": nsfw_res,
            "weapons": weapon_res,
            "text_threats": text_res,
            "threats_count": len(threats),
            "threats": threats,
            "visualization_saved": save_visual_path if save_visual_path else None
        }

    def _save_annotated_image(self, bgr_img: np.ndarray, threats: list, out_path: str):
        annotated = bgr_img.copy()
        for t in threats:
            bbox = t.get("bbox")
            if not bbox:
                continue
            x, y, bw, bh = bbox
            sev = t.get("severity", "MEDIUM")
            color = self.SEVERITY_COLORS.get(sev, (0, 0, 255))
            label = f"{t.get('label', 'THREAT')} ({t.get('confidence', 0.0):.2f})"

            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thick = 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            tag_y1 = max(0, y - th - 6)
            tag_y2 = y
            cv2.rectangle(annotated, (x, tag_y1), (x + tw + 6, tag_y2), color, -1)
            cv2.putText(annotated, label, (x + 3, tag_y2 - 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(out_path, annotated)
