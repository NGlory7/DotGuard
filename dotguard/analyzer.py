"""
DotGuard - Threat Analysis Orchestrator
Main engine combining rasterization, morphological vision, and threat detection.
"""

from typing import Dict, Any, Optional
import os
import cv2
import numpy as np

from .rasterizer import Rasterizer
from .preprocessor import Preprocessor
from .detectors.stego import StegoDetector
from .detectors.geometry import GeometryDetector
from .detectors.nsfw_rules import NSFWDetector
from .detectors.text_ocr import TextOCRDetector
from .visualizer import Visualizer

class DotGuardAnalyzer:
    def __init__(self, dot_size: int = 6, dot_spacing: int = 2):
        self.rasterizer = Rasterizer(dot_size=dot_size, dot_spacing=dot_spacing)
        self.preprocessor = Preprocessor(bridge_radius=dot_spacing + 2)
        self.stego_detector = StegoDetector()
        self.geometry_detector = GeometryDetector()
        self.nsfw_detector = NSFWDetector()
        self.text_detector = TextOCRDetector()
        self.visualizer = Visualizer()

    def analyze_text(self, text: str, save_visual_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes complete threat analysis pipeline on incoming text.
        """
        if not text or not text.strip():
            return {
                "status": "EMPTY_INPUT",
                "risk_level": "CLEAN",
                "threat_score": 0.0,
                "threats": []
            }

        # 1. Steganography check
        stego_result = self.stego_detector.analyze(text)

        # 2. Rasterization to image
        raw_img, art_type = self.rasterizer.rasterize(text)

        # 3. Morphological preprocessing
        binary_silhouette = self.preprocessor.connect_dots(raw_img)
        contours = self.preprocessor.extract_contours(binary_silhouette)
        features_list = [self.preprocessor.compute_shape_features(c) for c in contours]

        # 4. Geometry and hate symbol detection
        geom_threats = self.geometry_detector.analyze_contours(binary_silhouette, contours, features_list)

        # 5. NSFW / Anatomical silhouette detection
        nsfw_threats = self.nsfw_detector.analyze_contours(contours, features_list)

        # 6. Text / Letter / OCR detection
        text_result = self.text_detector.analyze(binary_silhouette, contours, features_list)

        # Combine all findings
        threats = []
        threats.extend(geom_threats)
        threats.extend(nsfw_threats)
        if text_result.get("text_detected"):
            threats.extend(text_result.get("lines", []))

        # Calculate overall threat score (0 to 100)
        threat_score = 0.0
        if stego_result.get("steganography_detected"):
            threat_score = max(threat_score, 75.0)

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

        # Assign risk level
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

        # 7. Optionally save visualization
        if save_visual_path:
            out_dir = os.path.dirname(save_visual_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            self.visualizer.save_analysis_image(raw_img, binary_silhouette, threats, save_visual_path)

        return {
            "status": "ANALYZED",
            "art_type": art_type,
            "image_dimensions": {"width": raw_img.shape[1], "height": raw_img.shape[0]},
            "risk_level": risk_level,
            "threat_score": threat_score,
            "contours_evaluated": len(contours),
            "threats_count": len(threats),
            "threats": threats,
            "steganography": stego_result,
            "visual_text": text_result,
            "visualization_saved": save_visual_path if save_visual_path else None
        }
