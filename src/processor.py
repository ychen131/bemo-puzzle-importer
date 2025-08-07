"""
Core image processing pipeline for Tangram piece isolation and analysis.

This module handles image loading, scaling, color-based piece isolation,
and contour extraction for Tangram puzzle processing.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from .constants import WORLD_SPACE_WIDTH, WORLD_SPACE_HEIGHT, HSV_COLOR_RANGES, MIN_CONTOUR_AREA, CONTOUR_EPSILON_FACTOR


def load_and_scale_image(image_path: str) -> np.ndarray:
    """
    Load an image and scale it to world space dimensions while maintaining aspect ratio.
    
    Args:
        image_path: Path to the input image file
        
    Returns:
        Scaled image as numpy array in BGR format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    # Check if file exists
    import os
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate scaling factor while maintaining aspect ratio
    width_scale = WORLD_SPACE_WIDTH / original_width
    height_scale = WORLD_SPACE_HEIGHT / original_height
    scale_factor = min(width_scale, height_scale)  # Use smaller scale to fit within bounds
    
    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # Resize image
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create world space canvas and center the scaled image
    world_image = np.zeros((WORLD_SPACE_HEIGHT, WORLD_SPACE_WIDTH, 3), dtype=np.uint8)
    
    # Calculate centering offsets
    x_offset = (WORLD_SPACE_WIDTH - new_width) // 2
    y_offset = (WORLD_SPACE_HEIGHT - new_height) // 2
    
    # Place scaled image in center of world space canvas
    world_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = scaled_image
    
    return world_image


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSV color space for color-based filtering.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Image converted to HSV format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def create_color_mask(hsv_image: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask for a specific color range with morphological cleaning.
    
    Args:
        hsv_image: Input image in HSV format
        color_name: Name of the color to isolate (must be in HSV_COLOR_RANGES)
        
    Returns:
        Clean binary mask where white pixels represent the target color
        
    Raises:
        KeyError: If color_name is not in HSV_COLOR_RANGES
    """
    if color_name not in HSV_COLOR_RANGES:
        raise KeyError(f"Color '{color_name}' not found in HSV_COLOR_RANGES")
    
    color_range = HSV_COLOR_RANGES[color_name]
    
    # Handle red color which has two ranges due to HSV hue wraparound
    if color_name == "red" and "lower2" in color_range:
        # Create mask for first range
        mask1 = cv2.inRange(hsv_image, 
                           np.array(color_range["lower"]), 
                           np.array(color_range["upper"]))
        
        # Create mask for second range
        mask2 = cv2.inRange(hsv_image, 
                           np.array(color_range["lower2"]), 
                           np.array(color_range["upper2"]))
        
        # Combine both masks using bitwise OR
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Standard single range mask creation
        mask = cv2.inRange(hsv_image, 
                          np.array(color_range["lower"]), 
                          np.array(color_range["upper"]))
    
    # Apply morphological operations to clean up the mask
    mask = clean_mask(mask)
    
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up a binary mask using morphological operations to remove noise and fill holes.
    
    Args:
        mask: Binary mask to clean
        
    Returns:
        Cleaned binary mask
    """
    # Create morphological kernels
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove noise with opening (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    
    # Fill holes with closing (dilation followed by erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, medium_kernel)
    
    # Additional noise removal
    mask = cv2.medianBlur(mask, 3)
    
    return mask


def filter_contours_by_quality(contours: List[np.ndarray]) -> List[np.ndarray]:
    """
    Filter contours based on quality metrics to remove invalid shapes.
    
    Args:
        contours: List of contours to filter
        
    Returns:
        List of high-quality contours
    """
    quality_contours = []
    
    for contour in contours:
        # Basic area filter
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        
        # Maximum area filter - reject extremely large contours (likely background)
        if area > 50000:  # Adjust based on typical piece size
            continue
        
        # Convexity filter - Tangram pieces should be reasonably convex
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            convexity = area / hull_area
            if convexity < 0.6:  # Stricter convexity requirement
                continue
        
        # Aspect ratio filter - reject extremely elongated shapes
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 4.0:  # Stricter aspect ratio
                continue
        
        # Perimeter to area ratio filter - reject very complex or noisy contours
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            complexity = (perimeter * perimeter) / area
            if complexity > 40:  # Stricter complexity filter
                continue
        
        # Vertex count filter - Tangram pieces should have 3-4 vertices when simplified
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertex_count = len(approx)
        if vertex_count < 3 or vertex_count > 6:  # Allow some flexibility
            continue
        
        quality_contours.append(contour)
    
    return quality_contours


def get_largest_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Get the largest contour from a list, representing the main piece.
    
    Args:
        contours: List of contours
        
    Returns:
        Largest contour or None if list is empty
    """
    if not contours:
        return None
    
    return max(contours, key=cv2.contourArea)


def simplify_contour(contour: np.ndarray) -> np.ndarray:
    """
    Simplify a contour to reduce noise while preserving shape.
    
    Args:
        contour: Input contour to simplify
        
    Returns:
        Simplified contour
    """
    # Calculate epsilon for contour approximation (percentage of perimeter)
    epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
    
    # Approximate contour to reduce noise
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    
    return simplified


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Find and filter high-quality contours in a binary mask.
    
    Args:
        mask: Binary mask image
        
    Returns:
        List of high-quality contours
    """
    # Find contours using external retrieval mode and simple approximation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by quality metrics
    quality_contours = filter_contours_by_quality(contours)
    
    # Simplify contours to reduce noise
    simplified_contours = []
    for contour in quality_contours:
        simplified = simplify_contour(contour)
        simplified_contours.append(simplified)
    
    return simplified_contours


def extract_pieces_from_image(image_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Complete pipeline to extract all Tangram pieces from a colored image.
    
    Args:
        image_path: Path to the input Tangram image
        
    Returns:
        List of tuples containing (color_name, contour) for each detected piece
        
    Raises:
        ValueError: If unable to detect pieces for each color
    """
    # Step 1: Load and scale image to world space
    image = load_and_scale_image(image_path)
    
    # Step 2: Convert to HSV color space
    hsv_image = convert_to_hsv(image)
    
    # Step 3: Extract the largest piece for each color
    detected_pieces = []
    
    for color_name in HSV_COLOR_RANGES.keys():
        # Create cleaned color mask
        mask = create_color_mask(hsv_image, color_name)
        
        # Find high-quality contours in the mask
        contours = find_contours(mask)
        
        # Get the largest contour for this color (representing the main piece)
        largest_contour = get_largest_contour(contours)
        
        if largest_contour is not None:
            detected_pieces.append((color_name, largest_contour))
    
    # Step 4: Validate detection results
    detected_colors = [color for color, _ in detected_pieces]
    expected_colors = list(HSV_COLOR_RANGES.keys())
    
    if len(detected_pieces) == 0:
        raise ValueError("No pieces detected. Check HSV color ranges and image quality.")
    
    missing_colors = set(expected_colors) - set(detected_colors)
    if missing_colors:
        print(f"Warning: Missing colors: {missing_colors}")
    
    print(f"Successfully detected {len(detected_pieces)} pieces: {detected_colors}")
    
    return detected_pieces