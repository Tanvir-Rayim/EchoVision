import argparse
import json
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description="Run OCR on text-line crops (Bangla + English)")
    p.add_argument("--images", required=True, help="Folder containing text-line crop images")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--lang", default="en,bn", help="Comma-separated languages (default: en,bn)")
    args = p.parse_args()

    # PaddleOCR is a strong default for Bangla+English.
    from paddleocr import PaddleOCR

    langs = [x.strip() for x in args.lang.split(",") if x.strip()]
    # PaddleOCR lang codes: use 'en' for English, and for Bangla often 'bn' (depends on version).
    # We initialize with a primary lang; mixed-script still generally works.
    primary_lang = "en" if "en" in langs else (langs[0] if langs else "en")

    ocr = PaddleOCR(lang=primary_lang, use_angle_cls=True)

    images_dir = Path(args.images)
    img_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    results_out = []
    for img_path in sorted(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        ocr_res = ocr.ocr(img, cls=True)

        # Standardize output
        lines = []
        if ocr_res and len(ocr_res) > 0:
            for item in ocr_res[0]:
                bbox, (text, score) = item
                lines.append({"text": text, "score": float(score), "bbox": bbox})

        results_out.append({"image": img_path.name, "lines": lines})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
