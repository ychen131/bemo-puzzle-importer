#!/usr/bin/env python3
"""
Environment validation script for Bemo Puzzle Creator.

This script tests the complete environment setup including:
- Python module imports
- OpenCV functionality
- NumPy operations
- Basic computer vision operations
"""

import sys
import traceback
from pathlib import Path


def test_basic_imports():
    """Test that all required packages can be imported."""
    print("Testing basic imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV imported successfully (version: {cv2.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    try:
        import pytest
        print(f"✅ Pytest imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Pytest: {e}")
        return False
    
    return True


def test_opencv_functionality():
    """Test basic OpenCV operations."""
    print("\nTesting OpenCV functionality...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [0, 255, 0]  # Green square
        
        # Test color conversion
        hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        print("✅ Color conversion (BGR to HSV) works")
        
        # Test contour detection
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✅ Contour detection works (found {len(contours)} contours)")
        
        # Test area calculation
        if contours:
            area = cv2.contourArea(contours[0])
            print(f"✅ Area calculation works (area: {area} pixels)")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenCV functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_numpy_operations():
    """Test NumPy mathematical operations."""
    print("\nTesting NumPy operations...")
    
    try:
        import numpy as np
        
        # Test array creation and operations
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = np.mean(arr)
        print(f"✅ NumPy array operations work (mean: {result})")
        
        # Test mathematical functions
        angles = np.array([0, np.pi/2, np.pi])
        cos_values = np.cos(angles)
        print(f"✅ NumPy mathematical functions work")
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy operations test failed: {e}")
        return False


def test_module_structure():
    """Test that our project modules can be imported."""
    print("\nTesting project module structure...")
    
    try:
        # Add src to path for testing
        src_path = Path(__file__).parent / "src"
        if src_path not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Test module imports
        import src
        print("✅ Main src package imports")
        
        import src.constants
        print("✅ Constants module imports")
        
        import src.processor
        print("✅ Processor module imports")
        
        import src.shape_validator
        print("✅ Shape validator module imports")
        
        import src.main
        print("✅ Main module imports")
        
        # Test constants access
        from src.constants import WORLD_SPACE_WIDTH, HSV_COLOR_RANGES
        print(f"✅ Constants accessible (world width: {WORLD_SPACE_WIDTH})")
        
        return True
        
    except Exception as e:
        print(f"❌ Module structure test failed: {e}")
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = ["src", "input", "output", "debug"]
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            all_exist = False
    
    return all_exist


def main():
    """Run all environment tests."""
    print("Bemo Puzzle Creator - Environment Validation")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("OpenCV Functionality", test_opencv_functionality),
        ("NumPy Operations", test_numpy_operations),
        ("Directory Structure", test_directory_structure),
        ("Module Structure", test_module_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Environment Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Environment setup is complete and ready for development!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())