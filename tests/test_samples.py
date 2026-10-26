"""
DotGuard - Sample Library for Testing and Verification
Contains both benign and threat patterns rendered in Unicode Braille and ASCII.
"""

# 1. Benign: Harmless rectangular frame / box
SAMPLE_BENIGN_BOX = """
⣏⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⣹
⣿⠀⠀⠀HELLO WORLD⠀⠀⠀⠀⠀⣿
⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿
⣇⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣸
"""

# 2. Hate Symbol: Braille Swastika pattern (commonly used in chat attacks)
SAMPLE_SWASTIKA = """
⣿⣿⣿⣿⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⠀⠀⠀⠀⠀⣿⣿⣿⣿⠀⠀⠀⠀
⣿⣿⣿⣿⠀⠀⠀⠀⠀⣿⣿⣿⣿⠀⠀⠀⠀
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⠀⠀⠀⠀⣿⣿⣿⣿⠀⣿⣿⣿⣿⠀⠀⠀⠀
⠀⠀⠀⠀⣿⣿⣿⣿⠀⣿⣿⣿⣿⠀⠀⠀⠀
⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⠀⠀⠀⠀
"""

# 3. Steganography: Hidden zero-width payload encoding "PWNED"
# Binary for 'P' (01010000), 'W' (01010111), 'N' (01011110)...
# Let's programmatically generate a clean braille art with zero-width characters inside
def _create_stego_sample(secret_msg: str = "FLAG{HACK}") -> str:
    bits = "".join(f"{ord(c):08b}" for c in secret_msg)
    # 0 -> \u200B (Zero-width space), 1 -> \u200C (Zero-width non-joiner)
    zw_payload = "".join('\u200C' if b == '1' else '\u200B' for b in bits)
    return f"⣷⣄{zw_payload}⡀⠀⣀⡀⠀⣀⣠⣾\n⣿⣿⣷⣾⣿⣿⣷⣾⣿⣿\n⠘⢿⣿⣿⣿⣿⣿⣿⣿⠃"

SAMPLE_STEGO = _create_stego_sample("SECRET_THREAT_URL")

# 4. Weapon / Blade: Long knife silhouette
SAMPLE_BLADE = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⠟⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⠟⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣴⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢀⣴⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

# 5. Word bypass: Text drawn with dots spelling "DIE"
SAMPLE_TEXT_WORD = """
⣿⣿⣿⣦⠀⠀⣿⣿⠀⠀⣿⣿⣿⣿⣿
⣿⠀⠀⣿⠀⠀⣿⣿⠀⠀⣿⠀⠀⠀⠀
⣿⠀⠀⣿⠀⠀⣿⣿⠀⠀⣿⣿⣿⣿⠀
⣿⠀⠀⣿⠀⠀⣿⣿⠀⠀⣿⠀⠀⠀⠀
⣿⣿⣿⠟⠀⠀⣿⣿⠀⠀⣿⣿⣿⣿⣿
"""
