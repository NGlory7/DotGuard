"""
DotGuard - Steganography & Invisible Payload Detector
Uncovers zero-width character payloads and Braille bitstream steganography.
"""

from typing import Dict, Any, List, Optional
import re

class StegoDetector:
    # Zero-width / invisible Unicode codepoints frequently used to bypass filters or smuggle payloads
    ZERO_WIDTH_CHARS = {
        '\u200B': 'ZERO_WIDTH_SPACE',
        '\u200C': 'ZERO_WIDTH_NON_JOINER',
        '\u200D': 'ZERO_WIDTH_JOINER',
        '\uFEFF': 'ZERO_WIDTH_NO_BREAK_SPACE',
        '\u2060': 'WORD_JOINER',
        '\u200E': 'LEFT_TO_RIGHT_MARK',
        '\u200F': 'RIGHT_TO_LEFT_MARK',
        '\u202A': 'LEFT_TO_RIGHT_EMBEDDING',
        '\u202B': 'RIGHT_TO_LEFT_EMBEDDING',
        '\u202C': 'POP_DIRECTIONAL_FORMATTING',
        '\u202D': 'LEFT_TO_RIGHT_OVERRIDE',
        '\u202E': 'RIGHT_TO_LEFT_OVERRIDE',
    }

    def detect_zero_width_payload(self, text: str) -> Dict[str, Any]:
        """
        Detects hidden invisible characters and attempts binary/base decoding.
        """
        found_chars = []
        binary_bits = []
        for ch in text:
            if ch in self.ZERO_WIDTH_CHARS:
                found_chars.append(ch)
                # Common encoding: \u200B as 0, \u200C or \u200D as 1
                if ch == '\u200B':
                    binary_bits.append('0')
                elif ch in ('\u200C', '\u200D', '\uFEFF'):
                    binary_bits.append('1')

        decoded_text = ""
        if len(binary_bits) >= 8 and len(binary_bits) % 8 == 0:
            try:
                chars = []
                for i in range(0, len(binary_bits), 8):
                    byte_str = "".join(binary_bits[i:i+8])
                    chars.append(chr(int(byte_str, 2)))
                decoded = "".join(chars)
                if all(32 <= ord(c) <= 126 or c in '\r\n\t' for c in decoded):
                    decoded_text = decoded
            except Exception:
                pass

        has_stego = len(found_chars) > 0
        threat_level = "HIGH" if (decoded_text or len(found_chars) > 5) else ("MEDIUM" if has_stego else "NONE")

        return {
            "has_hidden_chars": has_stego,
            "hidden_char_count": len(found_chars),
            "decoded_hidden_payload": decoded_text if decoded_text else None,
            "threat_level": threat_level
        }

    def decode_braille_byte_stream(self, text: str) -> Optional[str]:
        """
        Checks if braille character offsets (code - 0x2800) encode consecutive printable ASCII bytes.
        """
        braille_chars = [c for c in text if 0x2800 <= ord(c) <= 0x28FF]
        if len(braille_chars) < 4:
            return None

        extracted = []
        for c in braille_chars:
            byte_val = ord(c) - 0x2800
            # Printable ASCII range 32 - 126
            if 32 <= byte_val <= 126:
                extracted.append(chr(byte_val))
            else:
                extracted.append('?')

        ratio_printable = sum(1 for c in extracted if c != '?') / len(extracted)
        # If over 60% maps to printable characters, it's likely an encoded ASCII payload
        if ratio_printable >= 0.6:
            return "".join(extracted)
        return None

    def analyze(self, text: str) -> Dict[str, Any]:
        zw_result = self.detect_zero_width_payload(text)
        braille_payload = self.decode_braille_byte_stream(text)

        has_threat = zw_result["has_hidden_chars"] or (braille_payload is not None)
        return {
            "steganography_detected": has_threat,
            "zero_width": zw_result,
            "braille_byte_payload": braille_payload
        }
