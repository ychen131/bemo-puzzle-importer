#!/usr/bin/env python3
"""
Test script for validating Tangram mathematical constants and specifications.

This script verifies that all mathematical relationships are correct and
that the constants match the Tangram mathematical foundation.
"""

import sys
import math
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent / "src"
if src_path not in sys.path:
    sys.path.insert(0, str(src_path))

from src.constants import (
    TOTAL_TANGRAM_AREA, PIECE_AREAS, PIECE_AREA_RATIOS, EXPECTED_PIECE_COUNTS,
    EDGE_LENGTHS, ANGLES, PIECE_GEOMETRY, COLOR_TO_PIECE_TYPE, PIECE_Z_INDICES,
    get_expected_area_for_piece_type, validate_area_ratio, get_piece_category_from_area_ratio,
    WORLD_SPACE_WIDTH, WORLD_SPACE_HEIGHT, PIXELS_PER_WORLD_UNIT
)


def test_area_conservation():
    """Test that all piece areas sum to the expected total."""
    print("Testing area conservation...")
    
    # Calculate total area from individual pieces
    calculated_total = 0
    for piece_type, count in EXPECTED_PIECE_COUNTS.items():
        piece_area = PIECE_AREAS[piece_type]
        calculated_total += piece_area * count
    
    print(f"Expected total area: {TOTAL_TANGRAM_AREA}")
    print(f"Calculated total area: {calculated_total}")
    
    if abs(calculated_total - TOTAL_TANGRAM_AREA) < 0.001:
        print("✅ Area conservation test PASSED")
        return True
    else:
        print("❌ Area conservation test FAILED")
        return False


def test_area_ratios():
    """Test that area ratios are mathematically correct."""
    print("\nTesting area ratios...")
    
    # Get small triangle area as base unit
    small_area = PIECE_AREAS["small_triangle"]
    print(f"Small triangle area (base): {small_area}")
    
    all_correct = True
    for piece_type, expected_ratio in PIECE_AREA_RATIOS.items():
        actual_area = PIECE_AREAS[piece_type]
        calculated_ratio = actual_area / small_area
        
        print(f"{piece_type}: area={actual_area}, ratio={calculated_ratio:.1f}, expected={expected_ratio}")
        
        if abs(calculated_ratio - expected_ratio) > 0.001:
            print(f"❌ {piece_type} ratio incorrect")
            all_correct = False
        else:
            print(f"✅ {piece_type} ratio correct")
    
    if all_correct:
        print("✅ Area ratios test PASSED")
    else:
        print("❌ Area ratios test FAILED")
    
    return all_correct


def test_edge_length_relationships():
    """Test that edge length relationships are mathematically correct."""
    print("\nTesting edge length relationships...")
    
    unit = EDGE_LENGTHS["unit"]
    sqrt2 = EDGE_LENGTHS["sqrt2"]
    double = EDGE_LENGTHS["double"]
    double_sqrt2 = EDGE_LENGTHS["double_sqrt2"]
    
    # Verify mathematical relationships
    tests = [
        (sqrt2, math.sqrt(2), "√2 calculation"),
        (double, 2 * unit, "2 × unit calculation"),
        (double_sqrt2, 2 * math.sqrt(2), "2√2 calculation"),
        (sqrt2 / unit, math.sqrt(2), "√2/unit ratio")
    ]
    
    all_correct = True
    for actual, expected, description in tests:
        print(f"{description}: {actual:.6f} vs {expected:.6f}")
        if abs(actual - expected) > 0.000001:
            print(f"❌ {description} FAILED")
            all_correct = False
        else:
            print(f"✅ {description} PASSED")
    
    return all_correct


def test_angle_specifications():
    """Test that angle specifications are correct."""
    print("\nTesting angle specifications...")
    
    # Verify degree to radian conversions
    tests = [
        (ANGLES["right"]["radians"], math.pi / 2, "90° to radians"),
        (ANGLES["acute"]["radians"], math.pi / 4, "45° to radians"),
        (ANGLES["obtuse"]["radians"], 3 * math.pi / 4, "135° to radians")
    ]
    
    all_correct = True
    for actual, expected, description in tests:
        print(f"{description}: {actual:.6f} vs {expected:.6f}")
        if abs(actual - expected) > 0.000001:
            print(f"❌ {description} FAILED")
            all_correct = False
        else:
            print(f"✅ {description} PASSED")
    
    return all_correct


def test_piece_geometry_consistency():
    """Test that piece geometry specifications are internally consistent."""
    print("\nTesting piece geometry consistency...")
    
    all_correct = True
    
    for piece_type, geometry in PIECE_GEOMETRY.items():
        print(f"\nTesting {piece_type}:")
        
        # Check vertex count consistency
        vertices = geometry["vertices"]
        angles = geometry["angles"]
        edges = geometry["edges"][0]  # First (and only) edge set
        
        if len(angles) != vertices:
            print(f"❌ {piece_type}: angle count ({len(angles)}) != vertex count ({vertices})")
            all_correct = False
        else:
            print(f"✅ {piece_type}: vertex/angle count consistent ({vertices})")
        
        if len(edges) != vertices:
            print(f"❌ {piece_type}: edge count ({len(edges)}) != vertex count ({vertices})")
            all_correct = False
        else:
            print(f"✅ {piece_type}: vertex/edge count consistent ({vertices})")
        
        # Check angle sum for polygons
        expected_angle_sum = (vertices - 2) * 180
        actual_angle_sum = sum(angles)
        
        if abs(actual_angle_sum - expected_angle_sum) > 0.1:
            print(f"❌ {piece_type}: angle sum {actual_angle_sum}° != expected {expected_angle_sum}°")
            all_correct = False
        else:
            print(f"✅ {piece_type}: angle sum correct ({actual_angle_sum}°)")
        
        # Check area consistency with PIECE_AREAS
        expected_area = PIECE_AREAS[piece_type]
        geometry_area = geometry["area"]
        
        if abs(geometry_area - expected_area) > 0.001:
            print(f"❌ {piece_type}: geometry area {geometry_area} != expected {expected_area}")
            all_correct = False
        else:
            print(f"✅ {piece_type}: area consistent ({geometry_area})")
    
    return all_correct


def test_color_mapping_completeness():
    """Test that color to piece type mapping is complete."""
    print("\nTesting color mapping completeness...")
    
    # Check that we have exactly 7 colors for 7 pieces
    colors = set(COLOR_TO_PIECE_TYPE.keys())
    piece_types = set(COLOR_TO_PIECE_TYPE.values())
    
    if len(colors) != 7:
        print(f"❌ Expected 7 colors, got {len(colors)}: {colors}")
        return False
    else:
        print(f"✅ Correct number of colors (7): {colors}")
    
    if len(piece_types) != 7:
        print(f"❌ Expected 7 piece types, got {len(piece_types)}: {piece_types}")
        return False
    else:
        print(f"✅ Correct number of piece types (7): {piece_types}")
    
    # Check that z-indices are defined for all piece types
    missing_z = []
    for piece_type in piece_types:
        if piece_type not in PIECE_Z_INDICES:
            missing_z.append(piece_type)
    
    if missing_z:
        print(f"❌ Missing z-indices for: {missing_z}")
        return False
    else:
        print("✅ All piece types have z-indices defined")
    
    return True


def test_utility_functions():
    """Test the utility functions for area validation."""
    print("\nTesting utility functions...")
    
    all_correct = True
    
    # Test get_expected_area_for_piece_type
    test_cases = [
        ("largeTriangle1", 2.0),
        ("largeTriangle2", 2.0),
        ("mediumTriangle", 1.0),
        ("smallTriangle1", 0.5),
        ("smallTriangle2", 0.5),
        ("square", 1.0),
        ("parallelogram", 1.0)
    ]
    
    for piece_type, expected_area in test_cases:
        try:
            actual_area = get_expected_area_for_piece_type(piece_type)
            if abs(actual_area - expected_area) > 0.001:
                print(f"❌ {piece_type}: expected {expected_area}, got {actual_area}")
                all_correct = False
            else:
                print(f"✅ {piece_type}: area lookup correct ({actual_area})")
        except Exception as e:
            print(f"❌ {piece_type}: error in area lookup: {e}")
            all_correct = False
    
    # Test validate_area_ratio
    if validate_area_ratio(1.0, 1.0):
        print("✅ Area validation: exact match works")
    else:
        print("❌ Area validation: exact match failed")
        all_correct = False
    
    if validate_area_ratio(1.01, 1.0):
        print("✅ Area validation: within tolerance works")
    else:
        print("❌ Area validation: within tolerance failed")
        all_correct = False
    
    if not validate_area_ratio(1.1, 1.0):
        print("✅ Area validation: outside tolerance rejected")
    else:
        print("❌ Area validation: outside tolerance accepted")
        all_correct = False
    
    # Test get_piece_category_from_area_ratio
    ratio_tests = [
        (1, "small_triangle"),
        (2, "medium_triangle"),
        (4, "large_triangle")
    ]
    
    for ratio, expected_category in ratio_tests:
        actual_category = get_piece_category_from_area_ratio(ratio)
        if actual_category == expected_category:
            print(f"✅ Area ratio {ratio} → {actual_category}")
        else:
            print(f"❌ Area ratio {ratio} → {actual_category}, expected {expected_category}")
            all_correct = False
    
    return all_correct


def test_world_space_specifications():
    """Test world space coordinate system specifications."""
    print("\nTesting world space specifications...")
    
    # Check that dimensions are reasonable
    if WORLD_SPACE_WIDTH > 0 and WORLD_SPACE_HEIGHT > 0:
        print(f"✅ World space dimensions valid: {WORLD_SPACE_WIDTH}×{WORLD_SPACE_HEIGHT}")
    else:
        print(f"❌ Invalid world space dimensions: {WORLD_SPACE_WIDTH}×{WORLD_SPACE_HEIGHT}")
        return False
    
    # Check pixels per world unit
    if PIXELS_PER_WORLD_UNIT > 0:
        print(f"✅ Pixels per world unit valid: {PIXELS_PER_WORLD_UNIT}")
    else:
        print(f"❌ Invalid pixels per world unit: {PIXELS_PER_WORLD_UNIT}")
        return False
    
    return True


def main():
    """Run all validation tests for constants."""
    print("Tangram Constants Validation")
    print("=" * 50)
    
    tests = [
        ("Area Conservation", test_area_conservation),
        ("Area Ratios", test_area_ratios),
        ("Edge Length Relationships", test_edge_length_relationships),
        ("Angle Specifications", test_angle_specifications),
        ("Piece Geometry Consistency", test_piece_geometry_consistency),
        ("Color Mapping Completeness", test_color_mapping_completeness),
        ("Utility Functions", test_utility_functions),
        ("World Space Specifications", test_world_space_specifications)
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
    print(f"Constants Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mathematical specifications are correct!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())