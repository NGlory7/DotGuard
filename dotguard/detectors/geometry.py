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
        Supports both solid filled silhouettes and outline-drawn ASCII blades.
        Uses rotation-invariant minimum area rectangle elongation.
        """
        elongation = features.get("oriented_elongation", 0.0)
        solidity = features["solidity"]

        # High elongation regardless of rotation angle (supports hollow outline blades down to solidity 0.22)
        if elongation >= 3.6 and solidity >= 0.22:
            conf = min(1.0, 0.55 + (elongation - 3.6) * 0.10)
            return True, round(conf, 3)
        return False, 0.0

    def detect_firearm_handgun_signature(self, contour: np.ndarray, features: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Detects handgun / pistol / revolver silhouettes:
        - L-shaped geometric structure (barrel and perpendicular/angled grip)
        - Aspect ratio roughly 1.3 to 3.5
        - Moderate solidity (0.35 to 0.80)
        - Characteristic deep concavity defect (>40% of min dimension) between barrel and grip
        """
        x, y, w, h = features["bbox"]
        solidity = features["solidity"]
        ar = features["aspect_ratio"]

        major_ar = max(ar, 1.0 / ar) if ar > 0 else 0
        if not (1.3 <= major_ar <= 3.5):
            return False, 0.0
        if not (0.35 <= solidity <= 0.80):
            return False, 0.0

        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 10:
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    # In OpenCV defects shape is (N, 4) or (N, 1, 4)
                    max_d = 0.0
                    for i in range(defects.shape[0]):
                        depth_val = defects[i, 0, 3] if len(defects.shape) == 3 else defects[i, 3]
                        d_px = depth_val / 256.0
                        if d_px > max_d:
                            max_d = d_px

                    min_dim = min(w, h)
                    if min_dim > 0 and max_d >= (min_dim * 0.40):
                        conf = min(0.95, 0.60 + (max_d / min_dim) * 0.35)
                        return True, round(float(conf), 2)
            except Exception:
                pass

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

            # Test 2: Handgun / Revolver L-shape silhouette
            is_handgun, hg_conf = self.detect_firearm_handgun_signature(c, feat)
            if is_handgun and not is_swastika:
                findings.append({
                    "category": "WEAPON_FIREARM",
                    "label": "Handgun / Revolver Silhouette (Barrel & Grip)",
                    "confidence": round(hg_conf, 3),
                    "bbox": bbox,
                    "severity": "HIGH"
                })

            # Test 3: Blade / Sharp Weapon
            is_blade, bl_conf = self.detect_weapon_blade_signature(feat)
            if is_blade and not is_swastika and not is_handgun:
                findings.append({
                    "category": "WEAPON",
                    "label": "Elongated Blade / Knife Silhouette",
                    "confidence": round(bl_conf, 3),
                    "bbox": bbox,
                    "severity": "HIGH"
                })

        return findings
