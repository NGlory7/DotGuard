"""
Unit tests for DotGuard threat analysis engine
"""

import unittest
import numpy as np
from dotguard import DotGuardAnalyzer, ImageThreatAnalyzer
from tests.test_samples import SAMPLE_BENIGN_BOX, SAMPLE_SWASTIKA, SAMPLE_STEGO, SAMPLE_BLADE

class TestDotGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = DotGuardAnalyzer()
        cls.img_analyzer = ImageThreatAnalyzer()

    def test_benign_box_is_clean(self):
        rep = self.analyzer.analyze_text(SAMPLE_BENIGN_BOX)
        self.assertEqual(rep["risk_level"], "CLEAN")
        self.assertEqual(rep["threat_score"], 0.0)

    def test_swastika_detected_critical(self):
        rep = self.analyzer.analyze_text(SAMPLE_SWASTIKA)
        self.assertIn(rep["risk_level"], ["HIGH", "CRITICAL"])
        self.assertTrue(any(t["category"] == "HATE_SYMBOL" for t in rep["threats"]))

    def test_stego_detected_with_payload(self):
        rep = self.analyzer.analyze_text(SAMPLE_STEGO)
        self.assertTrue(rep["steganography"]["steganography_detected"])
        self.assertEqual(rep["steganography"]["zero_width"]["decoded_hidden_payload"], "SECRET_THREAT_URL")

    def test_blade_detected_weapon(self):
        rep = self.analyzer.analyze_text(SAMPLE_BLADE)
        self.assertTrue(any(t["category"] == "WEAPON" for t in rep["threats"]))

    def test_empty_string(self):
        rep = self.analyzer.analyze_text("")
        self.assertEqual(rep["risk_level"], "CLEAN")

    def test_direct_image_clean(self):
        clean_img = np.zeros((200, 200, 3), dtype=np.uint8)
        clean_img[:, :] = (100, 100, 100)
        rep = self.img_analyzer.analyze_image(clean_img)
        self.assertEqual(rep["risk_level"], "CLEAN")

    def test_direct_image_nsfw_skin(self):
        nsfw_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # BGR skin color fill
        nsfw_img[30:270, 30:270] = (120, 155, 215)
        rep = self.img_analyzer.analyze_image(nsfw_img)
        self.assertIn(rep["risk_level"], ["HIGH", "CRITICAL"])
        self.assertTrue(any("NSFW" in t["category"] for t in rep["threats"]))

if __name__ == "__main__":
    unittest.main()
