"""
DotGuard - Morphological Preprocessor
Bridges gaps between isolated dots to reconstitute coherent object silhouettes and contours.
"""

from typing import Tuple, List, Dict, Any
import cv2
import numpy as np

class Preprocessor:
    def __init__(self, bridge_radius: int = 5):
        self.bridge_radius = bridge_radius

    def connect_dots(self, img: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Applies morphological dilation and closing to bridge spaces between dots
        and form continuous shapes.
        """
        # Elliptical kernel connects diagonal and orthogonal dots nicely
        ksize = self.bridge_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # Close small holes
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        # Dilate slightly to form cohesive boundary
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # Smooth edges
        blurred = cv2.GaussianBlur(dilated, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        return thresh

    def extract_contours(self, binary_img: np.ndarray, min_area: float = 100.0) -> List[np.ndarray]:
        """
        Finds external contours that represent meaningful shapes.
        Filters out tiny noise artifacts.
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        # Sort largest contour first
        valid.sort(key=cv2.contourArea, reverse=True)
        return valid

    def compute_shape_features(self, contour: np.ndarray) -> Dict[str, Any]:
        """
        Calculates mathematical descriptors for a contour:
        - Area
        - Aspect Ratio (w / h)
        - Solidity (Area / Convex Hull Area)
        - Extent (Area / Bounding Rect Area)
        - Circularity / Compactness (4 * pi * Area / Perimeter^2)
        - Hu Moments (scale/rotation invariant features)
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0.0
        aspect_ratio = float(w) / h if h > 0 else 0.0

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

        # Hu Moments
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        # Log scale hu moments
        hu_log = []
        for val in hu:
            if abs(val) > 1e-12:
                hu_log.append(-1.0 * np.copysign(1.0, val) * np.log10(abs(val)))
            else:
                hu_log.append(0.0)

        # Centroid (center of mass)
        cx = int(moments['m10'] / moments['m00']) if moments['m00'] > 0 else x + w // 2
        cy = int(moments['m01'] / moments['m00']) if moments['m00'] > 0 else y + h // 2

        # Rotation-invariant minimum area bounding box (oriented bbox)
        min_rect = cv2.minAreaRect(contour)
        (_, _), (dim1, dim2), angle = min_rect
        major_axis = max(dim1, dim2)
        minor_axis = min(dim1, dim2)
        oriented_elongation = float(major_axis) / minor_axis if minor_axis > 0 else 0.0

        return {
            "area": area,
            "perimeter": perimeter,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "oriented_rect": min_rect,
            "oriented_elongation": oriented_elongation,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "extent": extent,
            "circularity": circularity,
            "hu_moments": hu_log
        }
