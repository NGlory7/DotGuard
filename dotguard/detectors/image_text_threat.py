"""
DotGuard - Image Text & Threat Keyword Detector
Extracts text from images / memes and cross-references against Threat & Violence lexicons (TR / EN).
"""

from typing import Dict, Any, List, Tuple
import re
import cv2
import numpy as np

class ImageTextThreatDetector:
    # Threat & Violence Lexicon (Turkish & English)
    THREAT_KEYWORDS = {
        # Extreme Violence / Threats
        "kill": ("VIOLENCE_THREAT", "CRITICAL"),
        "murder": ("VIOLENCE_THREAT", "CRITICAL"),
        "massacre": ("VIOLENCE_THREAT", "CRITICAL"),
        "bomb": ("TERROR_WEAPON", "CRITICAL"),
        "shoot": ("VIOLENCE_THREAT", "HIGH"),
        "gun": ("WEAPON_KEYWORD", "HIGH"),
        "guns": ("WEAPON_KEYWORD", "HIGH"),
        "weapon": ("WEAPON_KEYWORD", "HIGH"),
        "weapons": ("WEAPON_KEYWORD", "HIGH"),
        "pistol": ("WEAPON_KEYWORD", "HIGH"),
        "revolver": ("WEAPON_KEYWORD", "HIGH"),
        "rifle": ("WEAPON_KEYWORD", "HIGH"),
        "suicide": ("SELF_HARM", "HIGH"),
        "execution": ("VIOLENCE_THREAT", "CRITICAL"),
        "terrorist": ("TERROR_THREAT", "HIGH"),
        "decapitate": ("VIOLENCE_THREAT", "CRITICAL"),
        
        # Turkish Threat Keywords
        "öldür": ("VIOLENCE_THREAT", "CRITICAL"),
        "öldüreceğim": ("VIOLENCE_THREAT", "CRITICAL"),
        "gebert": ("VIOLENCE_THREAT", "CRITICAL"),
        "katliam": ("VIOLENCE_THREAT", "CRITICAL"),
        "bomba": ("TERROR_WEAPON", "CRITICAL"),
        "infaz": ("VIOLENCE_THREAT", "CRITICAL"),
        "silah": ("WEAPON_KEYWORD", "HIGH"),
        "tabanca": ("WEAPON_KEYWORD", "HIGH"),
        "tüfek": ("WEAPON_KEYWORD", "HIGH"),
        "bıçak": ("WEAPON_KEYWORD", "HIGH"),
        "intihar": ("SELF_HARM", "HIGH"),
        "kurşun": ("VIOLENCE_THREAT", "HIGH"),
        "terörist": ("TERROR_THREAT", "HIGH"),
        
        # Hate Speech & Extremism
        "nazi": ("HATE_SPEECH", "CRITICAL"),
        "swastika": ("HATE_SPEECH", "CRITICAL"),
        "hitler": ("HATE_SPEECH", "HIGH"),
        
        # Cyber Threats / Doxxing
        "doxx": ("CYBER_THREAT", "HIGH"),
        "leak": ("DATA_LEAK", "MEDIUM"),
        "ddos": ("CYBER_ATTACK", "MEDIUM"),
        "sızıntı": ("DATA_LEAK", "MEDIUM"),
    }

    def __init__(self):
        self._tesseract = self._init_tesseract()

    def _init_tesseract(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return pytesseract
        except Exception:
            return None

    def detect_text_regions_mser(self, bgr_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Finds text-like bounding boxes using OpenCV MSER (Maximally Stable Extremal Regions)."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        h, w = gray.shape[:2]

        boxes = []
        for r in regions:
            x, y, bw, bh = cv2.boundingRect(r)
            ar = bw / float(bh) if bh > 0 else 0
            if 0.2 <= ar <= 6.0 and 8 <= bh <= (h * 0.4) and 8 <= bw <= (w * 0.8):
                boxes.append([x, y, bw, bh])

        if not boxes:
            return []

        # Merge overlapping text boxes
        indices = cv2.dnn.NMSBoxes(boxes, [1.0] * len(boxes), 0.5, 0.4)
        merged = []
        if len(indices) > 0:
            for idx in indices.flatten():
                merged.append(tuple(boxes[idx]))
        return merged

    def scan_text_content(self, text: str) -> List[Dict[str, Any]]:
        """Scans extracted string against threat keyword database."""
        findings = []
        clean_text = text.lower()
        words = re.findall(r'\b\w+\b', clean_text)

        for w in words:
            if w in self.THREAT_KEYWORDS:
                category, severity = self.THREAT_KEYWORDS[w]
                findings.append({
                    "category": category,
                    "keyword_found": w,
                    "label": f"Threat Word Detected: '{w.upper()}' ({category})",
                    "severity": severity,
                    "confidence": 0.90
                })

        return findings

    def analyze(self, bgr_img: np.ndarray) -> Dict[str, Any]:
        text_boxes = self.detect_text_regions_mser(bgr_img)
        extracted_text = ""

        if self._tesseract is not None:
            try:
                extracted_text = self._tesseract.image_to_string(bgr_img).strip()
            except Exception:
                pass

        keyword_threats = self.scan_text_content(extracted_text)

        return {
            "has_text_regions": len(text_boxes) > 0,
            "text_regions_count": len(text_boxes),
            "extracted_text": extracted_text if extracted_text else None,
            "threat_keywords": keyword_threats,
            "threats_count": len(keyword_threats)
        }
