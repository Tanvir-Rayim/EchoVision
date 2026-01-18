import argparse
import json
from pathlib import Path

from PIL import Image


def _shape_to_bbox(shape):
    # Labelme rectangle: points = [[x1,y1],[x2,y2]] (can be any order)
    pts = shape.get("points") or []
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _yolo_line(class_id, x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main():
    p = argparse.ArgumentParser(description="Convert Labelme JSON rectangles to YOLO txt labels")
    p.add_argument("--images", required=True, help="Folder containing images + labelme .json")
    p.add_argument("--out", required=True, help="Output dataset folder (will create images/ and labels/)")
    p.add_argument("--classes", required=True, help="Path to labels.txt (one class per line)")
    p.add_argument("--ext", default="", help="Optional image extension filter (e.g. .jpg). Empty = all")
    args = p.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    labels_txt = Path(args.classes)

    classes = [line.strip() for line in labels_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_to_id = {name: i for i, name in enumerate(classes)}

    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_paths = []
    for pth in images_dir.iterdir():
        if not pth.is_file():
            continue
        if pth.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if args.ext and pth.suffix.lower() != args.ext.lower():
            continue
        img_paths.append(pth)

    converted = 0
    skipped = 0

    for img_path in sorted(img_paths):
        json_path = img_path.with_suffix(img_path.suffix + ".json")
        if not json_path.exists():
            json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            skipped += 1
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Use actual image size
        with Image.open(img_path) as im:
            w, h = im.size

        yolo_lines = []
        for shape in data.get("shapes", []):
            if (shape.get("shape_type") or "rectangle") != "rectangle":
                continue
            label = shape.get("label")
            if label not in class_to_id:
                continue
            bbox = _shape_to_bbox(shape)
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            yolo_lines.append(_yolo_line(class_to_id[label], x1, y1, x2, y2, w, h))

        # Write label file (even if empty, to match YOLO expectations)
        out_label_path = out_labels / (img_path.stem + ".txt")
        out_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        # Copy image
        (out_images / img_path.name).write_bytes(img_path.read_bytes())

        converted += 1

    print(f"Converted: {converted}")
    print(f"Skipped (no json): {skipped}")
    print(f"Classes: {len(classes)}")


if __name__ == "__main__":
    main()
