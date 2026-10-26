"""
DotGuard - Direct Image Weapon & Firearm Detector
Supports YOLO ONNX object detection via OpenCV DNN with geometric contour fallbacks.
"""

from typing import Dict, Any, List, Optional
import os
import cv2
import numpy as np

class ImageWeaponDetector:
    # Standard COCO classes for weapons/dangerous tools
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    WEAPON_CLASSES = {"knife", "scissors", "baseball bat", "gun", "pistol", "rifle", "sword", "weapon"}

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join("models", "weapons.onnx")
        self.net = None
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
            except Exception:
                self.net = None

    def detect_via_onnx(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLO / ONNX object detection model through OpenCV DNN."""
        if self.net is None:
            return []

        h, w = bgr_img.shape[:2]
        # Standard YOLO input 640x640
        blob = cv2.dnn.blobFromImage(bgr_img, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Parse YOLOv8 output shape: [1, 84, 8400] -> transpose to [8400, 84]
        findings = []
        try:
            if len(outputs.shape) == 3:
                preds = np.transpose(outputs[0], (1, 0)) # [8400, 84]
                boxes = []
                confidences = []
                class_ids = []

                for row in preds:
                    scores = row[4:]
                    class_id = int(np.argmax(scores))
                    conf = float(scores[class_id])
                    if conf > 0.40:
                        class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}"
                        if class_name.lower() in self.WEAPON_CLASSES:
                            cx, cy, bw, bh = row[0:4]
                            x = int((cx - bw / 2) * w / 640.0)
                            y = int((cy - bh / 2) * h / 640.0)
                            box_w = int(bw * w / 640.0)
                            box_h = int(bh * h / 640.0)

                            boxes.append([x, y, box_w, box_h])
                            confidences.append(conf)
                            class_ids.append(class_id)

                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.45)
                if len(indices) > 0:
                    for i in indices.flatten():
                        cname = self.COCO_CLASSES[class_ids[i]]
                        findings.append({
                            "category": "WEAPON_DETECTED",
                            "label": f"Weapon Identified: {cname.upper()}",
                            "confidence": round(float(confidences[i]), 3),
                            "bbox": tuple(boxes[i]),
                            "severity": "CRITICAL"
                        })
        except Exception:
            pass

        return findings

    def detect_heuristic_blades(self, bgr_img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Heuristic edge & high-gradient contour analysis for sharp elongated blades / metallic weapons.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        total_pixels = h * w

        # Canny edge detector for sharp metallic silhouettes
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 200)

        # Dilate slightly to connect broken edge lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        findings = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < (total_pixels * 0.015):
                continue

            rect = cv2.minAreaRect(c)
            (_, _), (dim1, dim2), _ = rect
            major = max(dim1, dim2)
            minor = min(dim1, dim2)
            if minor <= 0:
                continue

            elongation = major / minor
            # Extreme elongation indicative of blade / knife / barrel
            if elongation >= 4.5 and major > (min(h, w) * 0.25):
                x, y, bw, bh = cv2.boundingRect(c)
                findings.append({
                    "category": "SUSPICIOUS_SHAPE",
                    "label": "Elongated Sharp Contour (Possible Blade/Barrel)",
                    "confidence": round(min(0.85, 0.45 + (elongation - 4.5) * 0.08), 2),
                    "bbox": (x, y, bw, bh),
                    "severity": "HIGH"
                })

        return findings

    def analyze(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        onnx_findings = self.detect_via_onnx(bgr_img)
        heuristic_findings = []
        if not onnx_findings:
            heuristic_findings = self.detect_heuristic_blades(bgr_img)

        all_findings = onnx_findings + heuristic_findings
        has_weapon = len(all_findings) > 0

        return {
            "weapon_detected": has_weapon,
            "findings_count": len(all_findings),
            "findings": all_findings
        }
