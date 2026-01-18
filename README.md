# Echo – Signboard + Product Text Extraction (Research)

For the full end-to-end process, see `docs/pipeline.md`.

## Recommended approach (2-stage)

1. **Detect text carriers** (full image)
   - Model: YOLO (e.g., `yolov11s`)
   - Classes (suggested):
     - `signboard` (road/store/instruction boards)
     - `product_label` (bottle/package label areas)

2. **Detect text zones** (on cropped carriers)
   - Classes (suggested):
     - `text_line` (line-level boxes; best for OCR)

3. **OCR (Bangla + English)**
   - Use a scene-text OCR engine on each `text_line` crop.
   - Practical starting point: PaddleOCR (Bangla + English), then evaluate and fine-tune if needed.

## Why minimal classes first

Avoid making dozens of classes like `expiry_date`, `ingredients`, `store_name` early.
Instead:
- Detect carriers + text lines
- OCR everything
- Then do **information extraction** on the OCR text (regex + rules + lightweight classifier) to find expiry dates, ingredients, etc.

## Labeling in this repo

This workspace includes `label_me.py` which launches Labelme with consistent label sets.

### Install labelme (once)

```powershell
python -m pip install -r .\requirements.txt
```

### Stage 1 labeling (carriers)

```powershell
python .\label_me.py carrier
```

### Stage 2 labeling (text lines)

```powershell
python .\label_me.py text
```

Labelme saves `.json` annotations next to your images.

## Helper scripts

### Convert Labelme → YOLO

Carrier dataset (uses `labels.carrier.txt` created by `label_me.py carrier`):

```powershell
python .\scripts\convert_labelme_to_yolo.py --images .\Photos --out .\datasets\carrier --classes .\labels.carrier.txt
```

Text-line dataset (run this on your *carrier-crops* folder that contains images + json annotations):

```powershell
python .\scripts\convert_labelme_to_yolo.py --images .\carrier_crops_annotated --out .\datasets\text --classes .\labels.text.txt
```

### Detect & crop carriers (after training Stage A)

```powershell
python .\scripts\detect_and_crop_carriers.py --images .\Photos --model .\runs\detect\train\weights\best.pt --out .\outputs\carrier_crops
```

### OCR on text-line crops

```powershell
python .\scripts\ocr_text_lines.py --images .\outputs\text_line_crops --out .\outputs\ocr_results.json --lang en,bn
```

## Dataset size guidance (rule-of-thumb)

These are practical ranges for research prototypes (varies by image quality and diversity):

### Stage 1: carrier detection
- **Prototype**: 200–500 images (diverse), ~1k–5k labeled instances
- **Solid**: 1,000–3,000 images, ~5k–30k instances
- **Multi-domain robust** (road + shop + products + lighting changes): 3,000–10,000+ images

### Stage 2: text-line detection (on crops)
- Count boxes, not images:
  - **Prototype**: 500–1,500 carrier crops, ~5k–30k `text_line` boxes
  - **Solid**: 2,000–10,000 crops, ~30k–200k boxes

### OCR recognition fine-tuning (optional)
- If using a pretrained OCR model first, you can often start without fine-tuning.
- If you fine-tune recognition: aim for **5k–20k** labeled line images with transcripts (Bangla/English mix).

## Splits

Split by **original photo** (not by crops) to avoid leakage:
- Train 70–80%
- Val 10–20%
- Test 10%
