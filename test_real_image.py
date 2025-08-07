#!/usr/bin/env python3
"""
Quick test script to analyze a real Tangram puzzle image using our constants.

This script provides early validation of our HSV color ranges and specifications
before implementing the full processing pipeline.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))

from src.constants import (
    HSV_COLOR_RANGES, HSV_COLOR_RANGES_ALTERNATIVE, COLOR_TO_PIECE_TYPE,
    WORLD_SPACE_WIDTH, WORLD_SPACE_HEIGHT, MIN_CONTOUR_AREA
)


def load_and_scale_image(image_path: str):
    """
    Load and scale image to world space while maintaining aspect ratio.
    
    Args:
        image_path: Path to input image
        
    Returns:
        Scaled image in BGR format
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    original_height, original_width = image.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")
    
    # Calculate scaling to fit world space while maintaining aspect ratio
    scale_x = WORLD_SPACE_WIDTH / original_width
    scale_y = WORLD_SPACE_HEIGHT / original_height
    scale = min(scale_x, scale_y)  # Use smaller scale to maintain aspect ratio
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    print(f"Scaled image size: {new_width}x{new_height} (scale: {scale:.3f})")
    
    return scaled_image


def detect_colors_in_image(image, hsv_ranges, range_name="Primary"):
    """
    Detect each color in the image using HSV ranges.
    
    Args:
        image: BGR input image
        hsv_ranges: Dictionary of HSV color ranges
        range_name: Name for reporting
        
    Returns:
        Dictionary of detection results
    """
    print(f"\nTesting {range_name} HSV ranges on real image:")
    print("-" * 50)
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detection_results = {}
    total_pixels = image.shape[0] * image.shape[1]
    
    for color_name, ranges in hsv_ranges.items():
        print(f"\nAnalyzing {color_name}:")
        
        # Create mask
        if "lower2" in ranges:
            # Handle red color with two ranges
            mask1 = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
            mask2 = cv2.inRange(hsv_image, ranges["lower2"], ranges["upper2"])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Single range for other colors
            mask = cv2.inRange(hsv_image, ranges["lower"], ranges["upper"])
        
        # Count detected pixels
        detected_pixels = cv2.countNonZero(mask)
        percentage = (detected_pixels / total_pixels) * 100
        
        print(f"  Detected: {detected_pixels:,} pixels ({percentage:.2f}% of image)")
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        
        print(f"  Found {len(contours)} total contours, {len(valid_contours)} above minimum area")
        
        if valid_contours:
            # Calculate areas of valid contours
            areas = [cv2.contourArea(c) for c in valid_contours]
            areas.sort(reverse=True)  # Largest first
            
            print(f"  Largest contour areas: {areas[:3]}")  # Show top 3
            
            # Expected piece type for this color
            piece_type = COLOR_TO_PIECE_TYPE.get(color_name, "unknown")
            print(f"  Expected piece type: {piece_type}")
        
        detection_results[color_name] = {
            "pixels": detected_pixels,
            "percentage": percentage,
            "contours": len(contours),
            "valid_contours": len(valid_contours),
            "areas": [cv2.contourArea(c) for c in valid_contours] if valid_contours else [],
            "mask": mask
        }
        
        # Quick assessment
        if len(valid_contours) == 1:
            status = "✅ EXCELLENT - Found exactly 1 piece"
        elif len(valid_contours) == 0:
            status = "❌ NOT DETECTED - No valid contours found"
        elif len(valid_contours) <= 3:
            status = "⚠️  MODERATE - Multiple contours (may need tuning)"
        else:
            status = "❌ POOR - Too many contours (likely noise)"
        
        print(f"  Assessment: {status}")
    
    return detection_results


def create_detection_overview(image, results, output_path="debug/real_image_analysis.png"):
    """
    Create a visual overview of color detection on the real image.
    
    Args:
        image: Original BGR image
        results: Detection results
        output_path: Where to save the visualization
    """
    # Ensure debug directory exists
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Create a grid layout: original + 7 color masks
    # 2 rows, 4 columns
    img_height, img_width = image.shape[:2]
    grid_width = 4 * img_width
    grid_height = 2 * img_height
    
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place original image
    grid[0:img_height, 0:img_width] = image
    cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Place detection masks
    colors = list(results.keys())
    for i, (color_name, result) in enumerate(results.items()):
        # Calculate position (skip first slot which has original)
        pos = i + 1
        row = pos // 4
        col = pos % 4
        
        y_start = row * img_height
        x_start = col * img_width
        
        # Convert mask to color for better visualization
        mask = result["mask"]
        
        # Create colored overlay
        colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        colored_mask[mask > 0] = [255, 255, 255]  # White for detected areas
        
        # Place in grid
        grid[y_start:y_start + img_height, x_start:x_start + img_width] = colored_mask
        
        # Add label with detection info
        label = f"{color_name} ({result['valid_contours']} pieces)"
        cv2.putText(grid, label, (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Save visualization
    cv2.imwrite(output_path, grid)
    print(f"\nDetection overview saved to: {output_path}")


def analyze_piece_distribution(results):
    """
    Analyze the distribution of detected pieces and assess completeness.
    
    Args:
        results: Detection results from color analysis
    """
    print(f"\n" + "=" * 50)
    print("PIECE DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    total_pieces_found = 0
    expected_pieces = 7
    colors_with_pieces = []
    colors_without_pieces = []
    
    for color_name, result in results.items():
        valid_contours = result["valid_contours"]
        piece_type = COLOR_TO_PIECE_TYPE.get(color_name, "unknown")
        
        if valid_contours > 0:
            colors_with_pieces.append((color_name, piece_type, valid_contours))
            total_pieces_found += valid_contours
        else:
            colors_without_pieces.append((color_name, piece_type))
    
    print(f"Total pieces detected: {total_pieces_found}/{expected_pieces}")
    print(f"Colors with pieces ({len(colors_with_pieces)}):")
    for color, piece_type, count in colors_with_pieces:
        print(f"  ✅ {color} → {piece_type} ({count} contour{'s' if count > 1 else ''})")
    
    if colors_without_pieces:
        print(f"\nColors without pieces ({len(colors_without_pieces)}):")
        for color, piece_type in colors_without_pieces:
            print(f"  ❌ {color} → {piece_type} (not detected)")
    
    # Overall assessment
    print(f"\nOVERALL ASSESSMENT:")
    if total_pieces_found == expected_pieces and len(colors_with_pieces) == expected_pieces:
        print("🎉 EXCELLENT: All 7 pieces detected with distinct colors!")
    elif total_pieces_found >= 5:
        print("✅ GOOD: Most pieces detected, minor tuning needed")
    elif total_pieces_found >= 3:
        print("⚠️  MODERATE: Some pieces detected, HSV ranges need adjustment")
    else:
        print("❌ POOR: Few pieces detected, major HSV tuning required")
    
    return total_pieces_found, len(colors_with_pieces)


def main():
    """
    Run real image analysis using our current constants.
    """
    print("Real Tangram Image Analysis")
    print("=" * 50)
    
    # Check if input image exists
    image_path = "input/tree.JPG"
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("Please place a Tangram puzzle image in the input/ directory")
        return 1
    
    try:
        # Load and scale image
        print(f"Loading image: {image_path}")
        image = load_and_scale_image(image_path)
        
        # Save scaled image for reference
        cv2.imwrite("debug/scaled_input.png", image)
        print(f"Scaled image saved to: debug/scaled_input.png")
        
        # Test primary HSV ranges
        primary_results = detect_colors_in_image(image, HSV_COLOR_RANGES, "Primary")
        
        # Test alternative HSV ranges
        alt_results = detect_colors_in_image(image, HSV_COLOR_RANGES_ALTERNATIVE, "Alternative")
        
        # Create visualizations
        create_detection_overview(image, primary_results, "debug/real_primary_detection.png")
        create_detection_overview(image, alt_results, "debug/real_alternative_detection.png")
        
        # Analyze results
        print(f"\n" + "=" * 50)
        print("PRIMARY RANGES ANALYSIS:")
        primary_pieces, primary_colors = analyze_piece_distribution(primary_results)
        
        print(f"\n" + "=" * 50)
        print("ALTERNATIVE RANGES ANALYSIS:")
        alt_pieces, alt_colors = analyze_piece_distribution(alt_results)
        
        # Final recommendation
        print(f"\n" + "=" * 50)
        print("RECOMMENDATIONS:")
        
        if primary_pieces >= alt_pieces:
            print("✅ Use PRIMARY HSV ranges for this image")
            better_results = primary_results
        else:
            print("✅ Use ALTERNATIVE HSV ranges for this image")
            better_results = alt_results
        
        if max(primary_pieces, alt_pieces) >= 5:
            print("🎯 HSV ranges are working well! Ready to proceed with Task 3")
        else:
            print("⚠️  HSV ranges need calibration for this image before Task 3")
            print("   Consider adjusting ranges based on this image's lighting/colors")
        
        print(f"\n📊 Debug outputs saved to debug/ directory")
        print(f"   • Scaled input: debug/scaled_input.png")
        print(f"   • Primary detection: debug/real_primary_detection.png")
        print(f"   • Alternative detection: debug/real_alternative_detection.png")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())