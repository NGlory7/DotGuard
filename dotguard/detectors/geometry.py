"""
DotGuard - Geometric & Hate Symbol Detector
Analyzes contour morphology, rotational symmetry, and geometric signatures
to identify hate symbols (e.g. Swastika), weapon silhouettes, and malicious shapes.
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

class GeometryDetector:
    def check_4fold_rotational_symmetry(self, binary_mask: np.ndarray, feat: Dict[str, Any]) -> float:
        """
        Calculates 4-fold (90-degree) rotational symmetry score around the shape's centroid.
        Hate symbols like the swastika or iron cross have high 90-deg rotational symmetry.
        Returns IoU (0.0 to 1.0).
        """
        cx, cy = feat.get("centroid", (0, 0))
        h, w = binary_mask.shape[:2]
        if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
            return 0.0

        # Rotate entire mask around contour centroid
        rot_mat = cv2.getRotationMatrix2D((float(cx), float(cy)), 90, 1.0)
        rot90 = cv2.warpAffine(binary_mask, rot_mat, (w, h))

        intersection = np.logical_and(binary_mask > 0, rot90 > 0).sum()
        union = np.logical_or(binary_mask > 0, rot90 > 0).sum()

        if union == 0:
            return 0.0
        return float(intersection) / float(union)

    def detect_swastika_signature(self, features: Dict[str, Any], sym_score: float) -> Tuple[bool, float]:
        """
        Heuristic detection for swastika shape:
        - Aspect ratio roughly 1.0 (0.70 - 1.40)
        - Moderate solidity (0.20 - 0.70)
        - High 90-degree rotational symmetry around centroid (>0.40)
        """
        ar = features["aspect_ratio"]
        solidity = features["solidity"]

        score = 0.0
        if 0.70 <= ar <= 1.40:
            score += 0.25
        if 0.20 <= solidity <= 0.72:
            score += 0.25
        if sym_score >= 0.38:
            score += 0.50 * min(1.0, (sym_score - 0.35) / 0.15)

        detected = (score >= 0.70 and sym_score >= 0.40)
        return detected, min(1.0, score)

    def detect_weapon_blade_signature(self, features: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Detects elongated sharp knife/blade silhouettes regardless of orientation/angle.
        Uses rotation-invariant minimum area rectangle elongation.
        """
        elongation = features.get("oriented_elongation", 0.0)
        solidity = features["solidity"]

        # High elongation regardless of rotation angle (e.g. diagonal blade)
        if elongation >= 3.8 and solidity >= 0.40:
            conf = min(1.0, 0.50 + (elongation - 3.8) * 0.12)
            return True, round(conf, 3)
        return False, 0.0

    def analyze_contours(self, binary_mask: np.ndarray, contours: List[np.ndarray], features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        findings = []

        for c, feat in zip(contours, features_list):
            bbox = feat["bbox"]
            sym_90 = self.check_4fold_rotational_symmetry(binary_mask, feat)
            feat["sym_90"] = sym_90

            # Test 1: Swastika / Cross hate symbol
            is_swastika, sw_conf = self.detect_swastika_signature(feat, sym_90)
            if is_swastika:
                findings.append({
                    "category": "HATE_SYMBOL",
                    "label": "Swastika / Rotational Cross Pattern",
                    "confidence": round(sw_conf, 3),
                    "bbox": bbox,
                    "severity": "CRITICAL"
                })

            # Test 2: Blade / Sharp Weapon
            is_blade, bl_conf = self.detect_weapon_blade_signature(feat)
            if is_blade and not is_swastika:
                findings.append({
                    "category": "WEAPON",
                    "label": "Elongated Blade / Knife Silhouette",
                    "confidence": round(bl_conf, 3),
                    "bbox": bbox,
                    "severity": "HIGH"
                })

        return findings
