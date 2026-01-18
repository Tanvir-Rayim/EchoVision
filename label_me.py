"""label_me.py

Labelme launcher for Echo workspace.

This script helps you annotate images using Labelme with a consistent label set.
It supports multiple "profiles" (e.g., signboard/product detection vs text-zone
annotation) so you can do the recommended 2-stage pipeline.
"""

import subprocess
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PHOTOS_DIR = PROJECT_ROOT / "Photos"

# Label sets for different annotation passes.
# Keep this intentionally small: you can always refine later.
LABEL_PROFILES = {
    # Stage 1: find "text carriers" in full images.
    # Use two classes if you regularly have both street/store signboards AND product labels.
    "carrier": [
        "signboard",
        "product_label",
    ],
    # Stage 2: inside each detected crop, label text zones.
    # Prefer line-level boxes for better OCR downstream.
    "text": [
        "text_line",
    ],
}

DEFAULT_PROFILE = "carrier"

def create_labels_file(classes, profile_name):
    """Create labels.txt for Labelme."""
    labels_file = PROJECT_ROOT / f"labels.{profile_name}.txt"
    with open(labels_file, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(cls + '\n')
    print(f"✓ Created {labels_file.name} with {len(classes)} classes")
    return labels_file

def create_config(classes, profile_name):
    """Create labelme config."""
    config = {
        "auto_save": True,
        "display_label_popup": True,
        "keep_prev": True,
        "keep_prev_mode": True,
        "keep_prev_scale": True,
        "keep_prev_brightness": True,
        "keep_prev_contrast": True,
        "logger_level": "info",
        "shape_color": "auto",
        "shift_auto_shape_color": 0,
        "sort_labels": True,
        "validate_label": "exact",
        "labels": classes
    }
    
    config_file = PROJECT_ROOT / f"labelme_config.{profile_name}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Created {config_file.name}")
    return config_file

def show_instructions(profile_name, classes):
    """Display usage instructions."""
    photos_hint = str(PHOTOS_DIR) if PHOTOS_DIR.exists() else "<your images folder>"
    class_list = "\n  - " + "\n  - ".join(classes)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 Labelme Quick Start Guide                    ║
╚══════════════════════════════════════════════════════════════╝

PROFILE: {profile_name}
CLASSES:{class_list}

🎯 KEYBOARD SHORTCUTS:
──────────────────────────────────────────────────────────────
  Ctrl+R        - Create rectangle (bounding box)
  D             - Next image
  A             - Previous image
  Del           - Delete selected box
  Ctrl+S        - Save
  Ctrl+D        - Duplicate current shape
  Ctrl+J        - Edit label
  Ctrl+Shift+A  - Save automatically and go to next

📋 WORKFLOW:
──────────────────────────────────────────────────────────────
1. When Labelme opens:
    • File → Open Dir → Select: {photos_hint}
   • Click on first image in file list
   
2. Start Annotating:
   • Press Ctrl+R to create rectangle
   • Click and drag around signboard
   • Select class from dropdown list
   • It auto-saves!
   
3. Complete batch:
   • JSON files saved alongside images
   • Convert to YOLO format after annotation
   • Move to next batch

💡 FEATURES:
──────────────────────────────────────────────────────────────
• Auto-save enabled (no manual save needed!)
• Zoom with mouse wheel
• Pan with middle mouse button
• Undo with Ctrl+Z
• All classes pre-loaded
• Brightness/Contrast adjustment available

🚀 AFTER ANNOTATION:
──────────────────────────────────────────────────────────────
Convert to YOLO format:
  python convert_labelme_to_yolo.py

Then organize:
  python organize_annotations.py

╔══════════════════════════════════════════════════════════════╗
║  Starting Labelme...                                         ║
╚══════════════════════════════════════════════════════════════╝
""")


def parse_profile(argv):
    """Parse profile name from CLI args."""
    if len(argv) < 2:
        return DEFAULT_PROFILE
    profile = argv[1].strip().lower()
    return profile

def main():
    """Launch Labelme"""
    profile_name = parse_profile(sys.argv)
    if profile_name not in LABEL_PROFILES:
        known = ", ".join(sorted(LABEL_PROFILES.keys()))
        print(f"❌ Unknown profile: {profile_name}")
        print(f"Available: {known}")
        print("Usage: python label_me.py [carrier|text]")
        sys.exit(2)

    classes = LABEL_PROFILES[profile_name]

    # Create config files
    labels_file = create_labels_file(classes, profile_name)
    _ = create_config(classes, profile_name)
    
    # Show instructions
    show_instructions(profile_name, classes)
    
    # Launch labelme with config
    try:
        print("\n🚀 Opening Labelme...\n")
        if PHOTOS_DIR.exists():
            print(f"📂 Open Dir → Select: {PHOTOS_DIR}\\n")
        else:
            print("📂 Open Dir → Select your images folder\n")
        
        # Launch with labels file
        subprocess.run([
            sys.executable, "-m", "labelme",
            "--labels", str(labels_file),
            "--nodata",  # Don't save image data in JSON
            "--autosave"  # Enable auto-save
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 Labelme closed. Progress saved!")
    except Exception as e:
        print(f"\n❌ Error launching labelme: {e}")
        print("\nTry running manually: labelme")

if __name__ == "__main__":
    main()
