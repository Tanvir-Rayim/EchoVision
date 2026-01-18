import argparse
import json
from pathlib import Path

import cv2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    p = argparse.ArgumentParser(description="Run YOLO carrier detection and crop signboards/product labels")
    p.add_argument("--images", required=True, help="Input images folder")
    p.add_argument("--model", required=True, help="Path to YOLO model (e.g., runs/detect/train/weights/best.pt)")
    p.add_argument("--out", required=True, help="Output folder for crops + metadata")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--pad", type=float, default=0.03, help="Padding ratio around bbox (relative to max(w,h))")
    args = p.parse_args()

    # Lazy import so users can run --help without ultralytics installed
    from ultralytics import YOLO

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    img_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    all_meta = []

    for img_path in sorted(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        r = results[0]

        # r.boxes.xyxy, r.boxes.cls, r.boxes.conf
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            pad_px = int(args.pad * max(w, h))
            x1i = clamp(int(x1) - pad_px, 0, w - 1)
            y1i = clamp(int(y1) - pad_px, 0, h - 1)
            x2i = clamp(int(x2) + pad_px, 0, w - 1)
            y2i = clamp(int(y2) + pad_px, 0, h - 1)
            if x2i <= x1i or y2i <= y1i:
                continue

            crop = img[y1i:y2i, x1i:x2i]
            crop_name = f"{img_path.stem}__{i:03d}__cls{cls_id}__{conf:.2f}.png"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            all_meta.append(
                {
                    "image": img_path.name,
                    "crop": str(Path("crops") / crop_name),
                    "bbox_xyxy": [x1i, y1i, x2i, y2i],
                    "class_id": cls_id,
                    "conf": conf,
                    "image_size": [w, h],
                }
            )

    (out_dir / "crops.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"Crops: {len(all_meta)}")
    print(f"Saved: {out_dir / 'crops.json'}")


if __name__ == "__main__":
    main()
