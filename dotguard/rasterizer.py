"""
DotGuard - Text to Image Rasterizer
Converts Unicode Braille and ASCII art text into clean OpenCV image matrices.
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import platform

class Rasterizer:
    BRAILLE_START = 0x2800
    BRAILLE_END = 0x28FF

    def __init__(self, dot_size: int = 6, dot_spacing: int = 2):
        """
        :param dot_size: Diameter in pixels for each braille dot when rendered.
        :param dot_spacing: Margin in pixels between adjacent dots.
        """
        self.dot_size = dot_size
        self.dot_spacing = dot_spacing
        self.step = dot_size + dot_spacing
        self._font = self._load_default_font()

    # Repoyla gelen yazı tipi: her işletim sisteminde aynı piksel çıktısı (tespit eşikleri buna göre ayarlı).
    BUNDLED_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "DejaVuSansMono.ttf")
    # Karakter hücresi genişliği (px). Başka bir yazı tipine düşülürse çıktı bu genişliğe ölçeklenir,
    # böylece şekil oranları (en/boy, girinti derinliği) yazı tipinden bağımsız kalır.
    CELL_WIDTH = 9.0

    def _load_default_font(self):
        """Prefers the bundled DejaVu Sans Mono; falls back to a system monospaced font."""
        font_candidates = [self.BUNDLED_FONT]
        if platform.system() == "Windows":
            windir = os.environ.get("WINDIR", r"C:\Windows")
            font_candidates += [
                os.path.join(windir, "Fonts", "consola.ttf"),
                os.path.join(windir, "Fonts", "cour.ttf"),
                os.path.join(windir, "Fonts", "lucon.ttf"),
            ]
        else:
            font_candidates += [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            ]

        for path in font_candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size=16)
                except Exception:
                    pass
        return ImageFont.load_default()

    def is_braille_char(self, ch: str) -> bool:
        code = ord(ch)
        return self.BRAILLE_START <= code <= self.BRAILLE_END

    def detect_art_type(self, text: str) -> str:
        """
        Returns 'braille', 'ascii', or 'empty'.
        """
        clean = [c for c in text if not c.isspace()]
        if not clean:
            return 'empty'
        braille_count = sum(1 for c in clean if self.is_braille_char(c))
        if braille_count / len(clean) >= 0.4:
            return 'braille'
        return 'ascii'

    def braille_to_bitmatrix(self, text: str) -> np.ndarray:
        """
        Direct bitwise mapping from Unicode Braille patterns (U+2800..U+28FF)
        to a deterministic binary matrix (0: background, 1: dot).
        Each character is 4 rows x 2 columns.
        """
        lines = text.splitlines()
        if not lines:
            return np.zeros((4, 2), dtype=np.uint8)

        max_cols = max(len(line) for line in lines)
        total_rows = len(lines) * 4
        total_cols = max_cols * 2

        matrix = np.zeros((total_rows, total_cols), dtype=np.uint8)

        for line_idx, line in enumerate(lines):
            r_base = line_idx * 4
            for char_idx, ch in enumerate(line):
                c_base = char_idx * 2
                code = ord(ch)
                if self.is_braille_char(ch):
                    mask = code - self.BRAILLE_START
                    # Braille standard 8-dot mapping:
                    # Dot 1: (0,0), Dot 4: (0,1)
                    # Dot 2: (1,0), Dot 5: (1,1)
                    # Dot 3: (2,0), Dot 6: (2,1)
                    # Dot 7: (3,0), Dot 8: (3,1)
                    if mask & 0x01: matrix[r_base + 0, c_base + 0] = 1
                    if mask & 0x02: matrix[r_base + 1, c_base + 0] = 1
                    if mask & 0x04: matrix[r_base + 2, c_base + 0] = 1
                    if mask & 0x08: matrix[r_base + 0, c_base + 1] = 1
                    if mask & 0x10: matrix[r_base + 1, c_base + 1] = 1
                    if mask & 0x20: matrix[r_base + 2, c_base + 1] = 1
                    if mask & 0x40: matrix[r_base + 3, c_base + 0] = 1
                    if mask & 0x80: matrix[r_base + 3, c_base + 1] = 1
                elif ch in ('█', '▓', '#', '@'):
                    # Full block representation for mixed art
                    matrix[r_base:r_base+4, c_base:c_base+2] = 1
                elif ch in ('░', '▒', '*', '+'):
                    matrix[r_base+1:r_base+3, c_base:c_base+2] = 1

        return matrix

    def rasterize_braille(self, text: str, inverted: bool = False) -> np.ndarray:
        """
        Renders braille text into an OpenCV grayscale image (0-255).
        Foreground dots are white (255) by default, background is black (0).
        """
        bit_mat = self.braille_to_bitmatrix(text)
        rows, cols = bit_mat.shape

        img_h = rows * self.step + self.dot_spacing
        img_w = cols * self.step + self.dot_spacing
        img = np.zeros((img_h, img_w), dtype=np.uint8)

        radius = max(1, self.dot_size // 2)

        # Draw dots as circles or solid boxes
        for r in range(rows):
            for c in range(cols):
                if bit_mat[r, c] == 1:
                    center_x = c * self.step + self.step // 2
                    center_y = r * self.step + self.step // 2
                    cv2.circle(img, (center_x, center_y), radius, 255, -1)

        if inverted:
            img = cv2.bitwise_not(img)
        return img

    def rasterize_ascii_font(self, text: str, inverted: bool = False) -> np.ndarray:
        """
        Renders generic ASCII art using Pillow font rasterizer to grayscale image.
        """
        lines = text.splitlines()
        if not lines:
            return np.zeros((100, 100), dtype=np.uint8)

        # Estimate image dimensions
        # Use dummy image to calculate text bounding box
        dummy_img = Image.new("L", (10, 10))
        draw = ImageDraw.Draw(dummy_img)

        # Determine line heights and max width
        line_height = 20
        max_w = 100
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self._font)
            w = bbox[2] - bbox[0]
            if w > max_w:
                max_w = w

        total_h = max(100, len(lines) * line_height + 20)
        total_w = max_w + 30

        pil_img = Image.new("L", (total_w, total_h), color=0)
        draw = ImageDraw.Draw(pil_img)

        y = 10
        for line in lines:
            draw.text((15, y), line, fill=255, font=self._font)
            y += line_height

        img = np.array(pil_img)

        # Yazı tipinin karakter genişliği CELL_WIDTH'ten farklıysa yatay olarak normalleştir
        try:
            advance = float(self._font.getlength("M"))
        except Exception:
            advance = self.CELL_WIDTH
        if advance > 0 and abs(advance - self.CELL_WIDTH) > 0.05:
            scale = self.CELL_WIDTH / advance
            img = cv2.resize(img, (max(1, int(round(img.shape[1] * scale))), img.shape[0]), interpolation=cv2.INTER_AREA)

        if inverted:
            img = cv2.bitwise_not(img)
        return img

    def rasterize(self, text: str) -> Tuple[np.ndarray, str]:
        """
        Automatically detects art type and produces a normalized OpenCV binary/grayscale image.
        Returns: (image_mat, detected_type)
        """
        art_type = self.detect_art_type(text)
        if art_type == 'braille':
            img = self.rasterize_braille(text)
        else:
            img = self.rasterize_ascii_font(text)
        return img, art_type
