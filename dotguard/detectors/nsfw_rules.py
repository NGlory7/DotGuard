"""
DotGuard - NSFW & Anatomical Silhouette Detector
Detects explicit anatomical shapes, copypasta vulgar silhouettes, and erotic contours.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

class NSFWDetector:
    def detect_phallic_structure(self, contour: np.ndarray, feat: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Detects phallic anatomical structures:
        - Elongated shape (aspect ratio 1.8 to 4.5)
        - Characteristic convexity defects (deep notch at base between lobes)
        - Mass distribution (asymmetric center of mass along major axis)
        """
        area = feat["area"]
        ar = feat["aspect_ratio"]
        solidity = feat["solidity"]

        # Typical range: elongated vertically or horizontally
        major_ar = max(ar, 1.0 / ar) if ar > 0 else 0
        if not (1.6 <= major_ar <= 5.0):
            return False, 0.0

        if not (0.35 <= solidity <= 0.85):
            return False, 0.0

        # Check convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 10:
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    # Look for significant deep defects (at least 2-3 deep indentations)
                    deep_defects = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0  # OpenCV fixed-point depth
                        if depth > (min(feat["bbox"][2], feat["bbox"][3]) * 0.25):
                            deep_defects += 1

                    if deep_defects >= 2:
                        conf = 0.55 + min(0.35, deep_defects * 0.1)
                        return True, round(conf, 3)
            except Exception:
                pass

        return False, 0.0

    def detect_bilobed_contours(self, contours: List[np.ndarray], features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects twin symmetric lobes (e.g. breast or buttock silhouettes)
        positioned side by side with similar areas and vertical alignment.
        """
        findings = []
        n = len(contours)
        if n < 2:
            return findings

        for i in range(n):
            for j in range(i + 1, n):
                f1, f2 = features[i], features[j]
                # Filter out tiny noise contours (e.g. serrated blade teeth or small symbols)
                if f1["area"] < 300 or f2["area"] < 300:
                    continue

                area_ratio = min(f1["area"], f2["area"]) / max(f1["area"], f2["area"]) if max(f1["area"], f2["area"]) > 0 else 0
                if area_ratio < 0.65:
                    continue

                x1, y1, w1, h1 = f1["bbox"]
                x2, y2, w2, h2 = f2["bbox"]

                # Check horizontal adjacency and vertical overlap
                y_diff = abs(y1 - y2)
                x_dist = abs((x1 + w1 / 2) - (x2 + w2 / 2))

                # Both should have rounded circularity
                if f1["circularity"] > 0.45 and f2["circularity"] > 0.45:
                    if y_diff < (h1 * 0.3) and x_dist < (w1 * 2.5):
                        findings.append({
                            "category": "NSFW_ANATOMY",
                            "label": "Dual Symmetric Lobe Contour (Explicit/Nude pattern)",
                            "confidence": 0.72,
                            "bbox": (min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(y1, y2)),
                            "severity": "HIGH"
                        })
        return findings

    def detect_single_bilobed_contour(self, contour: np.ndarray, feat: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Detects a single connected contour representing twin symmetric lobes
        (breasts / buttocks / testicles) joined by cleavage, hands, or central bridge.
        """
        x, y, w, h = feat["bbox"]
        bilat_sym = feat.get("bilat_sym", 0.0)
        ar = feat["aspect_ratio"]
        solidity = feat["solidity"]
        extent = feat["extent"]

        # Must be horizontally symmetric with twin solid rounded lobes (exclude hollow rectangular frames)
        if bilat_sym < 0.65 or not (0.75 <= ar <= 3.2) or solidity < 0.55 or extent < 0.45:
            return False, 0.0

        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 10:
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    max_d = 0.0
                    for i in range(defects.shape[0]):
                        depth_val = defects[i, 0, 3] if len(defects.shape) == 3 else defects[i, 3]
                        d_px = depth_val / 256.0
                        if d_px > max_d:
                            max_d = d_px

                    # Deep central notch/cleft dividing the two lobes
                    if max_d >= (h * 0.22):
                        conf = min(0.92, 0.68 + (bilat_sym - 0.65) * 0.50)
                        return True, round(float(conf), 2)
            except Exception:
                pass

        return False, 0.0

    def analyze_contours(self, contours: List[np.ndarray], features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        findings = []

        # Check individual phallic contours and connected twin lobes
        for c, feat in zip(contours, features_list):
            is_phallic, conf = self.detect_phallic_structure(c, feat)
            if is_phallic:
                findings.append({
                    "category": "NSFW_VULGAR",
                    "label": "Explicit Phallic / Obscene Contour",
                    "confidence": conf,
                    "bbox": feat["bbox"],
                    "severity": "CRITICAL"
                })

            is_bilobed, b_conf = self.detect_single_bilobed_contour(c, feat)
            if is_bilobed and not is_phallic:
                findings.append({
                    "category": "NSFW_ANATOMY",
                    "label": "Dual Symmetric Lobe Contour (Breast / Buttock silhouette)",
                    "confidence": b_conf,
                    "bbox": feat["bbox"],
                    "severity": "HIGH"
                })

        # Check separate twin lobe patterns
        bilobed = self.detect_bilobed_contours(contours, features_list)
        findings.extend(bilobed)

        return findings
