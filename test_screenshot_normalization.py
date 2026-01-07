"""
Test screenshot normalization before PSP deaging.
This demonstrates the fix for screenshot alignment issues.
"""

import sys
import os

# Add paths
sys.path.insert(0, 'Phase2/Deaging/psp/encoder4editing')

from utils.alignment import is_likely_screenshot, normalize_screenshot, load_image_safely
from PIL import Image
import numpy as np


def test_screenshot_detection(image_path):
    """Test if an image is detected as a screenshot"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # Test detection
    is_screenshot = is_likely_screenshot(img_np)
    
    print(f"Screenshot detected: {is_screenshot}")
    
    if is_screenshot:
        print("\n📸 This image appears to be a screenshot or resampled image")
        print("   Normalization will be applied automatically in the pipeline")
        
        # Show normalization
        normalized = normalize_screenshot(img)
        output_path = "test_normalized_" + os.path.basename(image_path)
        normalized.save(output_path)
        print(f"   ✓ Normalized version saved: {output_path}")
    else:
        print("\n📷 This image appears to be an original photo")
        print("   No normalization needed")
    
    # Test the integrated loader
    print("\n🔧 Testing integrated load_image_safely():")
    loaded = load_image_safely(image_path)
    print(f"   ✓ Image loaded successfully")
    print(f"   Size: {loaded.size}")
    
    return is_screenshot


def main():
    print("\n" + "="*60)
    print("SCREENSHOT NORMALIZATION TEST")
    print("="*60)
    print("\nThis test demonstrates the screenshot normalization fix.")
    print("\nThe pipeline now:")
    print("  1. Detects screenshots using high-frequency analysis")
    print("  2. Normalizes screenshots BEFORE face detection")
    print("  3. Suppresses aliasing and sharpening artifacts")
    print("  4. Stabilizes landmarks for consistent alignment")
    print("\n" + "="*60)
    
    # Test images
    test_images = [
        "images/48000956.avif",
        # Add more test images
    ]
    
    results = []
    for img_path in test_images:
        if os.path.exists(img_path):
            is_screenshot = test_screenshot_detection(img_path)
            results.append((img_path, is_screenshot))
        else:
            print(f"\n⚠️  Skipping {img_path} - file not found")
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if results:
        for img_path, is_screenshot in results:
            status = "📸 Screenshot" if is_screenshot else "📷 Original"
            print(f"{status}: {img_path}")
    else:
        print("\n⚠️  No test images found.")
        print("Add image paths to test_images list to test.")
    
    print("\n" + "="*60)
    print("HOW IT WORKS")
    print("="*60)
    print("""
The normalization pipeline:

BEFORE (old behavior):
  Input → Face Detection → Alignment → PSP Model
          ❌ Screenshots fail here due to artifacts

AFTER (fixed):
  Input → Screenshot Detection → Normalization → Face Detection → Alignment → PSP Model
          ✓ Screenshots normalized to behave like originals

Screenshot normalization:
  1. Gaussian blur (sigma=0.6) - removes oversharpening/aliasing
  2. Downscale→upscale - restores continuous gradients
  3. Contrast normalization - standardizes dynamic range

This makes screenshots "behave like" originals for face processing,
without attempting to recreate lost information (which is impossible).

Detection criteria:
  - High-frequency energy > 12.0 (Laplacian variance)
  - Indicates artificial sharpening from screenshots

Integration:
  - Automatically applied in load_image_safely()
  - Used by get_landmark() BEFORE face detection
  - Transparent to existing deaging pipeline code
    """)


if __name__ == "__main__":
    main()
