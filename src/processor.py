"""
Core image processing pipeline for Tangram piece isolation and analysis.

This module handles image loading, scaling, color-based piece isolation,
and contour extraction for Tangram puzzle processing.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from .constants import WORLD_SPACE_WIDTH, WORLD_SPACE_HEIGHT, HSV_COLOR_RANGES, MIN_CONTOUR_AREA


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
    Create a binary mask for a specific color range.
    
    Args:
        hsv_image: Input image in HSV format
        color_name: Name of the color to isolate (must be in HSV_COLOR_RANGES)
        
    Returns:
        Binary mask where white pixels represent the target color
        
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
    
    return mask


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Find contours in a binary mask and filter by minimum area.
    
    Args:
        mask: Binary mask image
        
    Returns:
        List of contours that meet minimum area requirements
    """
    # Find contours using external retrieval mode and simple approximation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by minimum area to remove noise
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_CONTOUR_AREA:
            filtered_contours.append(contour)
    
    return filtered_contours


def extract_pieces_from_image(image_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Complete pipeline to extract all Tangram pieces from a colored image.
    
    Args:
        image_path: Path to the input Tangram image
        
    Returns:
        List of tuples containing (color_name, contour) for each detected piece
        
    Raises:
        ValueError: If unable to detect exactly 7 pieces
    """
    # Step 1: Load and scale image to world space
    image = load_and_scale_image(image_path)
    
    # Step 2: Convert to HSV color space
    hsv_image = convert_to_hsv(image)
    
    # Step 3: Extract pieces for each color
    detected_pieces = []
    
    for color_name in HSV_COLOR_RANGES.keys():
        # Create color mask
        mask = create_color_mask(hsv_image, color_name)
        
        # Find contours in the mask
        contours = find_contours(mask)
        
        # Add all valid contours for this color
        for contour in contours:
            detected_pieces.append((color_name, contour))
    
    # Step 4: Validate that we have exactly 7 pieces
    if len(detected_pieces) != 7:
        raise ValueError(f"Expected exactly 7 Tangram pieces, but detected {len(detected_pieces)} pieces. "
                        f"Found pieces: {[color for color, _ in detected_pieces]}")
    
    return detected_pieces