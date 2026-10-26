#!/usr/bin/env python3
"""
DotGuard CLI - Standalone Threat Inspector for Unicode Braille, ASCII Art & Direct Images
Usage:
    python cli.py                          (Launches interactive prompt)
    python cli.py scan <path_or_file>      (Auto-detects .txt or image)
    python cli.py scan-text "<text>"       (Scans a string directly)
    python cli.py scan -                   (Reads input from piped stdin)
    python cli.py demo                     (Runs test demo on sample patterns)
"""

import sys
import os
import argparse
import json

# Ensure UTF-8 output on Windows consoles to prevent cp1254/cp1252 charmap encoding crashes
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotguard import DotGuardAnalyzer, ImageThreatAnalyzer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def print_banner():
    banner = r"""
  ____        _    ____                     _ 
 |  _ \  ___ | |_ / ___|_   _  __ _ _ __ __| |
 | | | |/ _ \| __| |  _| | | |/ _` | '__/ _` |
 | |_| | (_) | |_| |_| | |_| | (_| | | | (_| |
 |____/ \___/ \__|\____|\__,_|\__,_|_|  \__,_|
  Standalone Threat & NSFW Inspector for Chat Art & Images
    """
    print(banner)

def format_report_console(report: dict):
    risk = report.get("risk_level", "CLEAN")
    score = report.get("threat_score", 0.0)
    art_type = report.get("art_type") or report.get("media_type", "unknown")
    dims = report.get("image_dimensions") or report.get("resolution", {})
    threats = report.get("threats", [])
    stego = report.get("steganography", {})

    color_reset = "\033[0m"
    if risk == "CRITICAL":
        risk_color = "\033[91;1m"  # Bold Red
    elif risk == "HIGH":
        risk_color = "\033[91m"    # Red
    elif risk == "MEDIUM":
        risk_color = "\033[93m"    # Yellow
    else:
        risk_color = "\033[92m"    # Green

    print(f"\n{'='*64}")
    print(f" [*] THREAT ANALYSIS REPORT")
    print(f"{'='*64}")
    print(f" Input Type       : {art_type.upper()}")
    print(f" Resolution       : {dims.get('width', 0)} x {dims.get('height', 0)} px")
    print(f" Risk Assessment  : {risk_color}{risk}{color_reset}")
    print(f" Threat Score     : {risk_color}{score} / 100{color_reset}")
    if "contours_evaluated" in report:
        print(f" Contours Scanned : {report.get('contours_evaluated', 0)}")
    print(f"{'-'*64}")

    # Steganography details
    if stego and stego.get("steganography_detected"):
        print(f" [!] STEGANOGRAPHY / HIDDEN PAYLOAD DETECTED:")
        zw = stego.get("zero_width", {})
        if zw.get("has_hidden_chars"):
            print(f"     - Invisible Zero-Width Characters: {zw.get('hidden_char_count')} found")
            if zw.get("decoded_hidden_payload"):
                print(f"     - Decoded Payload: '{zw.get('decoded_hidden_payload')}'")
        if stego.get("braille_byte_payload"):
            print(f"     - Braille Decoded ASCII: '{stego.get('braille_byte_payload')}'")
        print(f"{'-'*64}")

    # Threat details
    if threats:
        print(f" [!] DETECTED VIOLATIONS & THREATS ({len(threats)}):")
        for idx, t in enumerate(threats, 1):
            sev = t.get("severity", "MEDIUM")
            label = t.get("label", "Threat")
            conf = t.get("confidence", 0.0)
            bbox = t.get("bbox", (0, 0, 0, 0))
            print(f"   {idx}. [{sev}] {label}")
            print(f"      Confidence: {conf * 100:.1f}% | Location: {bbox}")
    else:
        print(f" [OK] No hate symbols, weapons, NSFW anatomy, or hidden threats detected.")

    if report.get("visualization_saved"):
        print(f"\n [i] Annotated inspection image saved to:")
        print(f"     file:///{os.path.abspath(report['visualization_saved'])}")
    print(f"{'='*64}\n")

def clean_path_input(path_str: str) -> str:
    """Strips quotes and whitespace when user drags & drops a file into terminal."""
    return path_str.strip().strip("'\"")

def interactive_mode():
    print_banner()
    analyzer = DotGuardAnalyzer()
    img_analyzer = ImageThreatAnalyzer()

    while True:
        print("\n=== DotGuard Interactive Menu ===")
        print("  [1] Paste a message / chat art (multiline)")
        print("  [2] Scan a text file (.txt)")
        print("  [3] Scan an image file (.jpg, .png, etc.)")
        print("  [4] Run built-in demo samples")
        print("  [0] Exit")
        choice = input("\nSelect an option [0-4]: ").strip()

        if choice == "1":
            print("\nPaste your message below. Type 'DONE' on an empty line and press Enter to finish:")
            print("-" * 50)
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip() == "DONE":
                        break
                    lines.append(line)
                except EOFError:
                    break
            text_content = "\n".join(lines)
            if not text_content.strip():
                print("Empty input. Returning to menu.")
                continue

            save_vis = input("\nSave annotated inspection image? (Enter file name like report.png or leave empty): ").strip()
            save_vis = clean_path_input(save_vis) if save_vis else None
            rep = analyzer.analyze_text(text_content, save_visual_path=save_vis)
            format_report_console(rep)

        elif choice == "2":
            file_path = input("\nEnter or drag-and-drop .txt file path: ").strip()
            file_path = clean_path_input(file_path)
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                continue
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            save_vis = input("Save annotated inspection image? (Enter file name or leave empty): ").strip()
            save_vis = clean_path_input(save_vis) if save_vis else None
            rep = analyzer.analyze_text(content, save_visual_path=save_vis)
            format_report_console(rep)

        elif choice == "3":
            img_path = input("\nEnter or drag-and-drop image file path: ").strip()
            img_path = clean_path_input(img_path)
            if not os.path.exists(img_path):
                print(f"Error: File not found: {img_path}")
                continue
            save_vis = input("Save annotated inspection image? (Enter file name or leave empty): ").strip()
            save_vis = clean_path_input(save_vis) if save_vis else None
            rep = img_analyzer.analyze_image(img_path, save_visual_path=save_vis)
            format_report_console(rep)

        elif choice == "4":
            run_demo(analyzer)

        elif choice == "0":
            print("\nExiting DotGuard. Stay safe!\n")
            sys.exit(0)
        else:
            print("Invalid selection, please try again.")

def run_demo(analyzer: DotGuardAnalyzer):
    print("\nRunning DotGuard Demo on synthetic threat patterns...\n")
    from tests.test_samples import SAMPLE_BENIGN_BOX, SAMPLE_SWASTIKA, SAMPLE_STEGO, SAMPLE_BLADE

    samples = [
        ("Benign Geometric Box", SAMPLE_BENIGN_BOX, "output_benign.png"),
        ("Hate Symbol (Swastika Pattern)", SAMPLE_SWASTIKA, "output_swastika.png"),
        ("Hidden Zero-Width Steganography", SAMPLE_STEGO, "output_stego.png"),
        ("Blade / Weapon Silhouette", SAMPLE_BLADE, "output_blade.png"),
    ]

    for name, text_content, visual_file in samples:
        print(f">>> Scanning: {name}...")
        rep = analyzer.analyze_text(text_content, save_visual_path=visual_file)
        format_report_console(rep)

def main():
    parser = argparse.ArgumentParser(description="DotGuard - Standalone Chat Art & Image Threat Inspector")
    subparsers = parser.add_subparsers(dest="command")

    # scan (auto-detects text vs image vs stdin)
    p_scan = subparsers.add_parser("scan", help="Scan a file (.txt, .jpg, .png) or use '-' for stdin pipe")
    p_scan.add_argument("target_path", help="Path to file or '-' for stdin")
    p_scan.add_argument("--visual", help="Path to save annotated inspection PNG", default=None)
    p_scan.add_argument("--json", action="store_true", help="Output raw JSON")

    # scan-file
    p_file = subparsers.add_parser("scan-file", help="Scan a text file containing Braille or ASCII art")
    p_file.add_argument("file_path", help="Path to text file")
    p_file.add_argument("--visual", help="Path to save annotated inspection PNG", default=None)
    p_file.add_argument("--json", action="store_true", help="Output raw JSON")

    # scan-image
    p_img = subparsers.add_parser("scan-image", help="Scan an image file (.jpg, .png, etc.)")
    p_img.add_argument("image_path", help="Path to image file")
    p_img.add_argument("--visual", help="Path to save annotated inspection PNG", default=None)
    p_img.add_argument("--json", action="store_true", help="Output raw JSON")

    # scan-text
    p_text = subparsers.add_parser("scan-text", help="Scan a raw string directly from command line")
    p_text.add_argument("text", help="Text string to analyze")
    p_text.add_argument("--visual", help="Path to save annotated inspection PNG", default=None)
    p_text.add_argument("--json", action="store_true", help="Output raw JSON")

    # demo
    subparsers.add_parser("demo", help="Run quick demo on built-in synthetic test samples")

    # interactive
    subparsers.add_parser("interactive", help="Start interactive CLI menu")

    args = parser.parse_args()

    # If no argument provided, start interactive mode
    if not args.command:
        interactive_mode()
        return

    analyzer = DotGuardAnalyzer()
    img_analyzer = ImageThreatAnalyzer()

    if args.command == "interactive":
        interactive_mode()
        return

    if args.command == "demo":
        print_banner()
        run_demo(analyzer)
        sys.exit(0)

    if args.command == "scan-text":
        report = analyzer.analyze_text(args.text, save_visual_path=args.visual)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_banner()
            format_report_console(report)
        sys.exit(1 if report["threat_score"] >= 65.0 else 0)

    # File or Stdin scanning
    target = getattr(args, "target_path", None) or getattr(args, "file_path", None) or getattr(args, "image_path", None)
    target = clean_path_input(target) if target else None

    if target == "-":
        # Stdin pipe
        content = sys.stdin.read()
        report = analyzer.analyze_text(content, save_visual_path=args.visual)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_banner()
            format_report_console(report)
        sys.exit(1 if report["threat_score"] >= 65.0 else 0)

    if not target or not os.path.exists(target):
        print(f"Error: File not found: {target}", file=sys.stderr)
        sys.exit(2)

    ext = os.path.splitext(target)[1].lower()
    if args.command == "scan-image" or ext in IMAGE_EXTENSIONS:
        report = img_analyzer.analyze_image(target, save_visual_path=args.visual)
    else:
        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        report = analyzer.analyze_text(content, save_visual_path=args.visual)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_banner()
        format_report_console(report)
    sys.exit(1 if report["threat_score"] >= 65.0 else 0)

if __name__ == "__main__":
    main()
