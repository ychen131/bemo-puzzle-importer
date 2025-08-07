#!/usr/bin/env python3
"""
Test script for validating HSV color ranges and generating color visualization.

This script creates sample color patches and tests the HSV range detection
to ensure colors can be correctly identified and are sufficiently distinct.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))

from src.constants import HSV_COLOR_RANGES, HSV_COLOR_RANGES_ALTERNATIVE


def create_color_test_image():
    """
    Create a test image with color patches for each Tangram piece color.
    
    Returns:
        BGR image with color patches arranged in a grid
    """
    # Create a blank image
    image = np.zeros((400, 700, 3), dtype=np.uint8)
    
    # Define RGB values for each color (approximate)
    rgb_colors = {
        "red": (255, 0, 0),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203)
    }
    
    # Draw color patches
    patch_size = 80
    margin = 10
    x_start = margin
    y_start = margin
    
    for i, (color_name, rgb) in enumerate(rgb_colors.items()):
        # Calculate position (2 rows, 4 columns max)
        col = i % 4
        row = i // 4
        
        x = x_start + col * (patch_size + margin)
        y = y_start + row * (patch_size + margin)
        
        # Draw filled rectangle
        bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
        cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), bgr, -1)
        
        # Add label
        cv2.putText(image, color_name, (x, y + patch_size + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def test_color_detection(image, hsv_ranges, range_name="Primary"):
    """
    Test color detection using the provided HSV ranges.
    
    Args:
        image: BGR test image
        hsv_ranges: Dictionary of HSV color ranges
        range_name: Name of the range set for reporting
        
    Returns:
        Dictionary of detection results
    """
    print(f"\nTesting {range_name} HSV ranges:")
    print("-" * 40)
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detection_results = {}
    
    for color_name, ranges in hsv_ranges.items():
        print(f"\nTesting {color_name}:")
        
        # Create mask
        if "lower2" in ranges:
            # Handle red color with two ranges
            mask1 = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
            mask2 = cv2.inRange(hsv_image, ranges["lower2"], ranges["upper2"])
            mask = cv2.bitwise_or(mask1, mask2)
            print(f"  Range 1: H({ranges['lower'][0]}-{ranges['upper'][0]}) "
                  f"S({ranges['lower'][1]}-{ranges['upper'][1]}) "
                  f"V({ranges['lower'][2]}-{ranges['upper'][2]})")
            print(f"  Range 2: H({ranges['lower2'][0]}-{ranges['upper2'][0]}) "
                  f"S({ranges['lower2'][1]}-{ranges['upper2'][1]}) "
                  f"V({ranges['lower2'][2]}-{ranges['upper2'][2]})")
        else:
            # Single range for other colors
            mask = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
            print(f"  Range: H({ranges['lower'][0]}-{ranges['upper'][0]}) "
                  f"S({ranges['lower'][1]}-{ranges['upper'][1]}) "
                  f"V({ranges['lower'][2]}-{ranges['upper'][2]})")
        
        # Count detected pixels
        detected_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (detected_pixels / total_pixels) * 100
        
        print(f"  Detected: {detected_pixels}/{total_pixels} pixels ({percentage:.2f}%)")
        
        detection_results[color_name] = {
            "pixels": detected_pixels,
            "percentage": percentage,
            "mask": mask
        }
        
        # Evaluate detection quality
        if detected_pixels > 100:
            if percentage < 5:
                status = "✅ Good - Specific detection"
            elif percentage < 15:
                status = "⚠️  Moderate - May include noise"
            else:
                status = "❌ Poor - Too broad, likely detecting multiple colors"
        else:
            status = "❌ Poor - Too restrictive, not detecting target color"
        
        print(f"  Status: {status}")
    
    return detection_results


def analyze_color_separation(results):
    """
    Analyze how well colors are separated (no overlap in detection).
    
    Args:
        results: Detection results from test_color_detection
    """
    print(f"\nColor Separation Analysis:")
    print("-" * 40)
    
    # Check for reasonable detection levels
    well_detected = []
    poorly_detected = []
    
    for color, result in results.items():
        if 100 <= result["pixels"] <= 20000:  # Reasonable range
            well_detected.append(color)
        else:
            poorly_detected.append((color, result["pixels"]))
    
    print(f"Well-detected colors ({len(well_detected)}): {', '.join(well_detected)}")
    
    if poorly_detected:
        print("Problematic detections:")
        for color, pixels in poorly_detected:
            if pixels < 100:
                print(f"  {color}: Too restrictive ({pixels} pixels)")
            else:
                print(f"  {color}: Too broad ({pixels} pixels)")
    
    # Overall assessment
    if len(well_detected) >= 6:  # At least 6 out of 7 should work well
        print("✅ Overall color separation: GOOD")
        return True
    elif len(well_detected) >= 4:
        print("⚠️  Overall color separation: MODERATE - Needs tuning")
        return False
    else:
        print("❌ Overall color separation: POOR - Requires significant adjustment")
        return False


def create_detection_visualization(image, results, output_path="debug/hsv_detection.png"):
    """
    Create a visualization showing the original image and detection masks.
    
    Args:
        image: Original BGR test image
        results: Detection results with masks
        output_path: Where to save the visualization
    """
    # Ensure debug directory exists
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Create visualization grid
    colors = list(results.keys())
    rows = (len(colors) + 3) // 4  # 4 colors per row
    
    # Calculate grid dimensions
    img_height, img_width = image.shape[:2]
    grid_width = 4 * img_width
    grid_height = (rows + 1) * img_height  # +1 for original image
    
    # Create the grid image
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place original image at top
    grid[0:img_height, 0:img_width] = image
    cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Place detection masks
    for i, (color_name, result) in enumerate(results.items()):
        row = (i // 4) + 1  # +1 to skip original image row
        col = i % 4
        
        y_start = row * img_height
        x_start = col * img_width
        
        # Convert mask to 3-channel for visualization
        mask_color = cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)
        
        # Place in grid
        grid[y_start:y_start + img_height, x_start:x_start + img_width] = mask_color
        
        # Add label
        cv2.putText(grid, color_name, (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    cv2.imwrite(output_path, grid)
    print(f"Detection visualization saved to: {output_path}")


def test_range_overlap():
    """
    Test for potential overlap between color ranges.
    """
    print(f"\nTesting Range Overlap:")
    print("-" * 40)
    
    colors = list(HSV_COLOR_RANGES.keys())
    overlaps_found = False
    
    for i, color1 in enumerate(colors):
        for j, color2 in enumerate(colors[i+1:], i+1):
            ranges1 = HSV_COLOR_RANGES[color1]
            ranges2 = HSV_COLOR_RANGES[color2]
            
            # Check for hue overlap (simplified check)
            h1_min, h1_max = ranges1["lower"][0], ranges1["upper"][0]
            h2_min, h2_max = ranges2["lower"][0], ranges2["upper"][0]
            
            # Handle red wrap-around case
            if "lower2" in ranges1:
                h1_max = ranges1["upper2"][0]
            if "lower2" in ranges2:
                h2_max = ranges2["upper2"][0]
            
            # Check for overlap
            if not (h1_max < h2_min or h2_max < h1_min):
                print(f"⚠️  Potential overlap: {color1} (H:{h1_min}-{h1_max}) "
                      f"and {color2} (H:{h2_min}-{h2_max})")
                overlaps_found = True
    
    if not overlaps_found:
        print("✅ No significant hue overlaps detected")
    
    return not overlaps_found


def main():
    """Run HSV color range validation tests."""
    print("HSV Color Range Validation")
    print("=" * 50)
    
    # Create test image
    print("Creating color test image...")
    test_image = create_color_test_image()
    
    # Save test image for reference
    cv2.imwrite("debug/color_test_patches.png", test_image)
    print("Test image saved to: debug/color_test_patches.png")
    
    # Test primary ranges
    primary_results = test_color_detection(test_image, HSV_COLOR_RANGES, "Primary")
    primary_separation = analyze_color_separation(primary_results)
    
    # Test alternative ranges
    alt_results = test_color_detection(test_image, HSV_COLOR_RANGES_ALTERNATIVE, "Alternative")
    alt_separation = analyze_color_separation(alt_results)
    
    # Test for range overlaps
    no_overlap = test_range_overlap()
    
    # Create visualizations
    create_detection_visualization(test_image, primary_results, "debug/primary_detection.png")
    create_detection_visualization(test_image, alt_results, "debug/alternative_detection.png")
    
    # Overall assessment
    print(f"\n" + "=" * 50)
    print("HSV Color Range Assessment:")
    
    tests_passed = 0
    total_tests = 3
    
    if primary_separation:
        print("✅ Primary ranges: PASSED")
        tests_passed += 1
    else:
        print("❌ Primary ranges: FAILED")
    
    if alt_separation:
        print("✅ Alternative ranges: PASSED")
        tests_passed += 1
    else:
        print("❌ Alternative ranges: FAILED")
    
    if no_overlap:
        print("✅ Range separation: PASSED")
        tests_passed += 1
    else:
        print("❌ Range separation: FAILED")
    
    print(f"\nResults: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 2:
        print("🎉 HSV color ranges are suitable for Tangram detection!")
        return 0
    else:
        print("⚠️  HSV color ranges need adjustment before use.")
        return 1


if __name__ == "__main__":
    sys.exit(main())