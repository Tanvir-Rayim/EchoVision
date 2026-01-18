# Ideal Pipeline & Overall Process (Signboards + Product Labels → Bangla/English Text)

This document describes a practical, research-friendly end-to-end pipeline:
- detect **where** relevant text exists (signboards, product labels)
- detect **text zones** inside those regions
- run **OCR** (Bangla + English)
- extract **structured fields** (product name, ingredients, expiry date, place/store names)

The key idea is to **separate “where to look” from “what it says”**.

---

## 0) Problem framing (what YOLO can and cannot do)

- YOLO models do **object detection** (bounding boxes + class labels).
- YOLO does **not** directly output the **characters/words**.
- For readable text strings you need an OCR recognizer.

So the best-practice pipeline is:

1) **Carrier detection** (full image): signboard / product label
2) **Text-zone detection** (on crops): text lines
3) **OCR recognition** (on text-line crops): Bangla/English text
4) **Information extraction** (on OCR text): expiry dates, ingredients, names, etc.

---

## 1) Data collection (what to capture)

To generalize across road signs, store signs, instruction boards, bottles, and packaging:

- Lighting: indoor/outdoor, day/night, glare, shadows
- Distance: close-up labels and far signboards
- Motion blur: handheld phone blur
- Angles: oblique views, perspective distortion
- Fonts: printed, stylized, mixed Bangla/English
- Background clutter: streets, shelves, reflections

**Tip (research):** keep a spreadsheet of “domain tags” per image (road/store/product). Don’t make these detector classes initially—use them for analysis and balanced splits.

---

## 2) Labeling strategy (recommended)

### Stage A — Carrier detection labels (full images)
Goal: find regions that *contain* text you care about.

**Classes (minimal, scalable):**
- `signboard`
- `product_label`

Label a single tight rectangle around the whole signboard/label area.

**Why not many classes now?**
Because your downstream objective (expiry date, ingredients, store name) is better solved after OCR as **text understanding**, not as more detection classes.

### Stage B — Text-zone labels (inside carrier crops)
Goal: segment readable chunks for OCR.

**Classes:**
- `text_line`

Label at **line level** whenever possible.
- Line-level crops typically improve OCR accuracy and reduce “word soup”.

If you later need more detail, expand Stage B to:
- `text_line` (primary)
- `text_block` (optional, for dense paragraphs)

---

## 3) Dataset size (rule-of-thumb targets)

You can start small for a prototype, but robust multi-domain performance needs more.

### Carrier detector (Stage A)
- Prototype: **200–500 images** (diverse)
- Solid: **1,000–3,000 images**
- Multi-domain robust: **3,000–10,000+ images**

### Text-line detector (Stage B)
Count **boxes**, not images.
- Prototype: **5k–30k** `text_line` boxes
- Solid: **30k–200k** `text_line` boxes

### OCR (recognition)
- Start with pretrained Bangla+English OCR.
- Fine-tune only if needed; then you typically need **5k–20k** labeled line images with transcripts.

---

## 4) Train/val/test split (avoid leakage)

Split by **original photo**, not by crops.

Recommended:
- Train: 70–80%
- Val: 10–20%
- Test: 10%

If you have repeated scenes (same store, same shelf, same street), keep them in the same split to avoid inflated accuracy.

---

## 5) Training plan (iteration loop)

A practical research loop:

1) Label a **seed set** (e.g., 200–500 images)
2) Train Stage A detector
3) Run inference on unlabeled images
4) Add images where model fails (hard cases)
5) Retrain
6) Once Stage A is decent, generate crops and label Stage B text lines
7) Train Stage B detector
8) Evaluate end-to-end OCR and field extraction

This is active-learning in spirit (without requiring fancy tooling).

---

## 6) OCR choice (Bangla + English)

Recommended starting points:
- **PaddleOCR**: strong end-to-end scene text for many scripts; good practical baseline.
- **EasyOCR**: simple setup; quality varies by domain.

Notes:
- If you already do Stage B (text-line boxes), you can run OCR recognition directly on those crops.
- Handle rotation if your data has angled text (some OCR stacks can auto-detect orientation).

---

## 7) Information extraction (expiry date, ingredients, names)

After OCR, solve your “research questions” using text processing.

### Expiry date (example)
Use regex patterns + normalization:
- `EXP`, `MFG`, `BEST BEFORE`, `USE BY`
- date patterns: `DD/MM/YYYY`, `MM/YY`, etc.

### Ingredients
Often a long block; you may:
- find an “Ingredients” anchor word
- take the next N lines until a stop condition

### Names of places/stores
Often prominent lines on signboards; you may:
- prefer largest text-line boxes (heuristic)
- cluster OCR lines by vertical position

This step is where you can publish meaningful research: accuracy of extracted fields, robustness across domains, Bangla/English mixing, etc.

---

## 8) Evaluation (what to report)

Report both component metrics and end-to-end metrics:

### Detection
- mAP@0.5 and mAP@0.5:0.95 for Stage A and Stage B
- small-object performance for Stage B (text lines)

### OCR
- Character Error Rate (CER)
- Word Error Rate (WER) (optional)

### End-to-end field extraction
- Expiry-date extraction accuracy (exact match / partial match)
- Ingredient extraction F1 (token-level or line-level)

---

## 9) Implementation notes (this repo)

This repo includes `label_me.py` with two profiles:

- Carrier labeling:
  - `python .\label_me.py carrier`

- Text-line labeling:
  - `python .\label_me.py text`

Labelme saves `.json` next to images. You’ll convert these to YOLO-format datasets for training.
