"""
DotGuard - Direct Image NSFW & Nudity Detector
Combines skin-tone segmentation (YCrCb + HSV) and optional ONNX deep learning classification.
"""

from typing import Dict, Any, List, Optional
import os
import cv2
import numpy as np

class ImageNSFWDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join("models", "nsfw.onnx")
        self.net = None
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except Exception:
                self.net = None

    def segment_skin_regions(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Dual-space skin color segmentation using YCrCb and HSV color models.
        Robust across diverse lighting and skin tones.
        """
        # 1. YCrCb Color Space
        ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
        # Cr: 133 to 173, Cb: 77 to 127
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # 2. HSV Color Space
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([25, 200, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Combine both spaces
        combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        dilated = cv2.dilate(cleaned, kernel, iterations=2)
        return dilated

    def analyze(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        h, w = bgr_img.shape[:2]
        total_pixels = h * w

        # Skin coverage analysis
        skin_mask = self.segment_skin_regions(bgr_img)
        skin_pixels = int(np.count_nonzero(skin_mask))
        skin_ratio = float(skin_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

        # Find major skin clusters/contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_clusters = []
        for c in contours:
            area = cv2.contourArea(c)
            # Clusters covering over 8% of the image
            if area > (total_pixels * 0.08):
                x, y, bw, bh = cv2.boundingRect(c)
                large_clusters.append({
                    "bbox": (x, y, bw, bh),
                    "area_ratio": round(area / total_pixels, 3)
                })

        findings = []
        is_nsfw = False
        confidence = 0.0

        # Heuristic Thresholds: High skin exposure ratio
        if skin_ratio >= 0.40 and len(large_clusters) >= 1:
            is_nsfw = True
            confidence = min(0.95, 0.50 + skin_ratio * 0.8)
            findings.append({
                "category": "NSFW_NUDITY",
                "label": f"High Nudity / Skin Exposure ({skin_ratio * 100:.1f}%)",
                "confidence": round(confidence, 3),
                "bbox": large_clusters[0]["bbox"] if large_clusters else (0, 0, w, h),
                "severity": "CRITICAL" if skin_ratio > 0.55 else "HIGH"
            })
        elif skin_ratio >= 0.25 and len(large_clusters) >= 1:
            findings.append({
                "category": "NSFW_SUSPICIOUS",
                "label": f"Moderate Skin Exposure ({skin_ratio * 100:.1f}%)",
                "confidence": 0.65,
                "bbox": large_clusters[0]["bbox"] if large_clusters else (0, 0, w, h),
                "severity": "MEDIUM"
            })

        # Deep learning pass if ONNX model loaded
        onnx_pred = None
        if self.net is not None:
            try:
                blob = cv2.dnn.blobFromImage(bgr_img, 1.0 / 255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)
                self.net.setInput(blob)
                out = self.net.forward()
                # Assuming [prob_safe, prob_nsfw]
                prob_nsfw = float(out[0][1])
                onnx_pred = round(prob_nsfw, 3)
                if prob_nsfw > 0.60:
                    is_nsfw = True
                    findings.append({
                        "category": "NSFW_MODEL",
                        "label": f"Neural NSFW Classification ({prob_nsfw * 100:.1f}%)",
                        "confidence": prob_nsfw,
                        "bbox": (0, 0, w, h),
                        "severity": "CRITICAL"
                    })
            except Exception:
                pass

        return {
            "is_nsfw": is_nsfw,
            "skin_ratio": round(skin_ratio, 3),
            "skin_clusters_count": len(large_clusters),
            "onnx_prediction": onnx_pred,
            "findings": findings
        }
