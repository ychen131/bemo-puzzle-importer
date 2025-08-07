"""
Geometric validation system for Tangram piece identification.

This module provides functions to identify and validate Tangram pieces
using mathematical properties such as area ratios, vertex counts, and angles.
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
try:
    from .constants import PIECE_AREA_RATIOS, TOLERANCE, MIN_CONTOUR_AREA
except ImportError:
    # Fallback for direct execution
    from constants import PIECE_AREA_RATIOS, TOLERANCE, MIN_CONTOUR_AREA


def calculate_contour_area(contour: np.ndarray) -> float:
    """
    Calculate the area of a contour in pixels.
    
    Args:
        contour: OpenCV contour array
        
    Returns:
        Area in pixels as a float
    """
    if contour is None or len(contour) < 3:
        return 0.0
    
    # Use OpenCV's contourArea function for accurate calculation
    area = cv2.contourArea(contour)
    return float(area)


def calculate_base_unit_area(contours: list) -> float:
    """
    Calculate the base unit area from the smallest triangles in the set.
    
    Args:
        contours: List of all detected contours
        
    Returns:
        Base unit area (area of smallest triangle)
        
    Raises:
        ValueError: If unable to determine base unit area
    """
    if not contours:
        raise ValueError("No contours provided")
    
    # Calculate areas for all contours
    areas = []
    for contour in contours:
        area = calculate_contour_area(contour)
        if area > 0:  # Only include valid areas
            areas.append(area)
    
    if not areas:
        raise ValueError("No valid contours found")
    
    # Sort areas to find the smallest ones
    areas.sort()
    
    # In a Tangram set, we expect 2 small triangles with the same area
    # Find the smallest area that appears at least once
    min_area = areas[0]
    
    # Validate that we have at least one small triangle
    if min_area < MIN_CONTOUR_AREA:
        raise ValueError(f"Smallest contour area ({min_area}) is below minimum threshold ({MIN_CONTOUR_AREA})")
    
    return min_area


def classify_piece_by_area(contour: np.ndarray, base_unit_area: float) -> Optional[str]:
    """
    Classify a piece type based on its area ratio to the base unit.
    
    Args:
        contour: OpenCV contour to classify
        base_unit_area: Reference area of the smallest triangle
        
    Returns:
        Piece type name or None if no match found
    """
    if contour is None or base_unit_area <= 0:
        return None
    
    # Calculate the area of this contour
    contour_area = calculate_contour_area(contour)
    
    if contour_area <= 0:
        return None
    
    # Calculate the ratio to the base unit
    area_ratio = contour_area / base_unit_area
    
    # Find the best matching piece type using the predefined ratios
    best_match = None
    min_difference = float('inf')
    
    for piece_type, expected_ratio in PIECE_AREA_RATIOS.items():
        difference = abs(area_ratio - expected_ratio)
        
        # Use a more generous tolerance for area matching
        # Calculate relative tolerance: 5% of the expected ratio or minimum 0.1
        tolerance_threshold = max(0.1, 0.05 * expected_ratio)
        
        if difference <= tolerance_threshold and difference < min_difference:
            min_difference = difference
            best_match = piece_type
    
    return best_match


def validate_piece_geometry(contour: np.ndarray, piece_type: str) -> bool:
    """
    Validate that a contour matches the expected geometry for a piece type.
    
    Args:
        contour: OpenCV contour to validate
        piece_type: Expected piece type name
        
    Returns:
        True if geometry is valid, False otherwise
    """
    # TODO: Implement geometric validation (vertex count, angles, etc.)
    pass


def calculate_piece_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid (center of mass) of a piece contour.
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Tuple of (x, y) coordinates of the centroid
    """
    # TODO: Implement centroid calculation using cv2.moments()
    pass


def calculate_piece_rotation(contour: np.ndarray) -> float:
    """
    Calculate the rotation angle of a piece relative to its standard orientation.
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Rotation angle in radians
    """
    # TODO: Implement rotation calculation using cv2.minAreaRect()
    pass


def validate_complete_tangram_set(classified_pieces: Dict[str, int]) -> bool:
    """
    Validate that the detected pieces form a complete Tangram set.
    
    Args:
        classified_pieces: Dictionary of piece_type -> count
        
    Returns:
        True if set is complete and valid, False otherwise
    """
    # TODO: Implement complete set validation
    pass