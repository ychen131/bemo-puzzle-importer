"""
Test script for GrabCut core functionality.

This script tests the basic functionality of the GrabCut enhancement module
using a real image from the input directory.
"""

import sys
import os
import cv2
import numpy as np
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.grabcut_enhancer import (
    apply_grabcut_refinement,
    hybrid_color_segmentation,
    validate_grabcut_quality,
    grabcut_performance_monitor,
    create_grabcut_debug_visualization
)
from src.processor import load_and_scale_image, convert_to_hsv, create_color_mask


def test_grabcut_basic_functionality():
    """Test basic GrabCut functionality with a real image."""
    print("=== Testing GrabCut Core Functionality ===\n")
    
    # Use the cat image for testing
    image_path = "input/cat.JPG"
    
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return False
    
    try:
        # Load and prepare image
        print("1. Loading and preparing image...")
        bgr_image = load_and_scale_image(image_path)
        hsv_image = convert_to_hsv(bgr_image)
        print(f"   Image size: {bgr_image.shape[:2]}")
        
        # Test with a few colors
        test_colors = ["red", "green", "blue"]
        
        for color_name in test_colors:
            print(f"\n2. Testing {color_name} detection...")
            
            # Create HSV mask
            hsv_mask = create_color_mask(hsv_image, color_name)
            hsv_area = cv2.countNonZero(hsv_mask)
            print(f"   HSV detected area: {hsv_area} pixels")
            
            if hsv_area == 0:
                print(f"   No {color_name} detected, skipping GrabCut test")
                continue
            
            # Test GrabCut refinement
            start_time = time.time()
            refined_mask = apply_grabcut_refinement(bgr_image, hsv_mask, iterations=3)
            end_time = time.time()
            
            refined_area = cv2.countNonZero(refined_mask)
            processing_time = end_time - start_time
            
            print(f"   GrabCut refined area: {refined_area} pixels")
            print(f"   Processing time: {processing_time:.3f} seconds")
            
            # Test quality validation
            quality_score = validate_grabcut_quality(hsv_mask, refined_mask)
            print(f"   Quality score: {quality_score:.3f}")
            
            # Test performance monitoring
            performance = grabcut_performance_monitor(
                start_time, 
                1.0,  # Assume 100% success for this test
                bgr_image.shape[:2],
                3
            )
            print(f"   Performance category: {performance['performance_category']}")
            
            # Test hybrid segmentation
            hybrid_mask = hybrid_color_segmentation(bgr_image, hsv_image, color_name)
            hybrid_area = cv2.countNonZero(hybrid_mask)
            print(f"   Hybrid segmentation area: {hybrid_area} pixels")
            
            # Save debug visualization if area is significant
            if hsv_area > 1000:  # Only for colors with substantial area
                debug_viz = create_grabcut_debug_visualization(
                    bgr_image, hsv_mask, refined_mask, quality_score
                )
                
                debug_path = f"debug/grabcut_test_{color_name}.png"
                os.makedirs("debug", exist_ok=True)
                cv2.imwrite(debug_path, debug_viz)
                print(f"   Debug visualization saved: {debug_path}")
        
        print("\n=== GrabCut Core Tests Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n=== Testing Error Handling ===\n")
    
    try:
        # Test with None inputs
        result = apply_grabcut_refinement(None, None)
        print("ERROR: Should have raised ValueError for None inputs")
        return False
    except ValueError as e:
        print(f"✓ Correctly handled None inputs: {e}")
    
    try:
        # Test with mismatched dimensions
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = apply_grabcut_refinement(image, mask)
        print("ERROR: Should have raised ValueError for mismatched dimensions")
        return False
    except ValueError as e:
        print(f"✓ Correctly handled dimension mismatch: {e}")
    
    try:
        # Test with invalid color name
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = hybrid_color_segmentation(image, hsv_image, "invalid_color")
        print("ERROR: Should have raised ValueError for invalid color")
        return False
    except ValueError as e:
        print(f"✓ Correctly handled invalid color: {e}")
    
    print("✓ All error handling tests passed")
    return True


if __name__ == "__main__":
    print("Starting GrabCut Core Module Tests...\n")
    
    # Run basic functionality tests
    basic_success = test_grabcut_basic_functionality()
    
    # Run error handling tests
    error_success = test_error_handling()
    
    if basic_success and error_success:
        print("\n🎉 All tests passed! GrabCut core module is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)