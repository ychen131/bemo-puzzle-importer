"""
GrabCut Enhancement Module for Bemo Puzzle Importer.

This module provides GrabCut-based enhancement for HSV color segmentation,
improving contour precision and reducing background noise through automatic
refinement of color masks.

GrabCut Integration Strategy:
- Uses existing HSV masks as initial foreground/background estimates
- Applies GrabCut algorithm for automatic refinement
- Provides fallback to original HSV detection when GrabCut fails
- Includes quality validation and performance monitoring
"""

from typing import Dict, Tuple, Optional, Any
import time
import cv2
import numpy as np
from .constants import HSV_COLOR_RANGES


def apply_grabcut_refinement(
    bgr_image: np.ndarray, 
    hsv_mask: np.ndarray, 
    iterations: int = 5
) -> np.ndarray:
    """
    Apply GrabCut refinement using HSV mask as initial guess.
    
    This function takes an existing HSV-based color mask and uses it as the initial
    estimate for OpenCV's GrabCut algorithm. The GrabCut algorithm then refines
    the segmentation by analyzing color and texture patterns in the original image.
    
    Args:
        bgr_image: Original BGR image (3-channel uint8)
        hsv_mask: Binary mask from HSV color detection (single channel uint8)
        iterations: Number of GrabCut iterations to perform (default: 5)
        
    Returns:
        Refined binary mask (single channel uint8) with values 0 or 255
        
    Raises:
        ValueError: If input dimensions don't match or invalid input types
    """
    # Input validation
    if bgr_image is None or hsv_mask is None:
        raise ValueError("Input image and mask cannot be None")
    
    if bgr_image.shape[:2] != hsv_mask.shape[:2]:
        raise ValueError(f"Image shape {bgr_image.shape[:2]} doesn't match mask shape {hsv_mask.shape[:2]}")
    
    if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
        raise ValueError("BGR image must be 3-channel")
    
    if len(hsv_mask.shape) != 2:
        raise ValueError("HSV mask must be single channel")
    
    try:
        # Convert HSV mask to GrabCut mask format
        # GrabCut uses 4 values: GC_BGD(0), GC_FGD(1), GC_PR_BGD(2), GC_PR_FGD(3)
        grabcut_mask = np.zeros(hsv_mask.shape, dtype=np.uint8)
        
        # Areas with high confidence (HSV detected as foreground)
        grabcut_mask[hsv_mask == 255] = cv2.GC_FGD  # Sure foreground
        
        # Areas with high confidence (HSV detected as background)
        grabcut_mask[hsv_mask == 0] = cv2.GC_BGD    # Sure background
        
        # Optional: Add probable regions around edges for better refinement
        # Create a border region around foreground for probable classification
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(hsv_mask, kernel, iterations=2)
        eroded = cv2.erode(hsv_mask, kernel, iterations=2)
        
        # Probable foreground: dilated area minus sure foreground
        prob_fg = dilated - hsv_mask
        grabcut_mask[prob_fg == 255] = cv2.GC_PR_FGD
        
        # Probable background: area between eroded and original
        prob_bg = hsv_mask - eroded
        grabcut_mask[prob_bg == 255] = cv2.GC_PR_BGD
        
        # Initialize GrabCut models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut algorithm
        cv2.grabCut(
            bgr_image, 
            grabcut_mask, 
            None,  # No rectangle initialization (using mask instead)
            bgd_model, 
            fgd_model, 
            iterations, 
            cv2.GC_INIT_WITH_MASK
        )
        
        # Convert GrabCut result to binary mask
        # Combine sure foreground and probable foreground
        refined_mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 
            255, 
            0
        ).astype('uint8')
        
        return refined_mask
        
    except Exception as e:
        # Fallback to original mask on any error
        print(f"GrabCut refinement failed: {e}. Using original HSV mask.")
        return hsv_mask.copy()


def hybrid_color_segmentation(
    bgr_image: np.ndarray, 
    hsv_image: np.ndarray, 
    color_name: str
) -> np.ndarray:
    """
    Perform hybrid color segmentation combining HSV detection with GrabCut refinement.
    
    This function integrates HSV color detection with GrabCut refinement in a single
    step, providing an enhanced segmentation that maintains the reliability of HSV
    detection with the precision of GrabCut refinement.
    
    Args:
        bgr_image: Original BGR image
        hsv_image: HSV-converted image
        color_name: Name of color to segment (must exist in HSV_COLOR_RANGES)
        
    Returns:
        Enhanced binary mask combining HSV and GrabCut techniques
        
    Raises:
        ValueError: If color_name is not recognized
    """
    from .processor import create_color_mask  # Import here to avoid circular imports
    
    if color_name not in HSV_COLOR_RANGES:
        raise ValueError(f"Unknown color: {color_name}. Available colors: {list(HSV_COLOR_RANGES.keys())}")
    
    # Step 1: Create initial HSV-based mask
    hsv_mask = create_color_mask(hsv_image, color_name)
    
    # Step 2: Apply GrabCut refinement
    refined_mask = apply_grabcut_refinement(bgr_image, hsv_mask)
    
    return refined_mask


def validate_grabcut_quality(
    original_mask: np.ndarray, 
    refined_mask: np.ndarray,
    min_improvement_ratio: float = 0.05
) -> float:
    """
    Compare GrabCut result quality against original HSV mask.
    
    This function analyzes multiple quality metrics to determine if the GrabCut
    refinement improved the segmentation quality. The quality score ranges from
    0.0 (poor) to 1.0 (excellent).
    
    Quality Metrics:
    - Contour count (fewer contours usually indicate cleaner segmentation)
    - Area preservation (refined area should be similar to original)
    - Edge smoothness (measured by perimeter-to-area ratio)
    - Connectivity (fewer disconnected components is better)
    
    Args:
        original_mask: Original HSV-based binary mask
        refined_mask: GrabCut-refined binary mask
        min_improvement_ratio: Minimum improvement needed to consider GrabCut successful
        
    Returns:
        Quality score between 0.0 and 1.0 (higher is better)
    """
    if original_mask is None or refined_mask is None:
        return 0.0
    
    if original_mask.shape != refined_mask.shape:
        return 0.0
    
    try:
        # Metric 1: Contour count (fewer is usually better for clean shapes)
        orig_contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refined_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        orig_contour_count = len(orig_contours)
        refined_contour_count = len(refined_contours)
        
        # Score: 1.0 if refined has fewer/equal contours, scaled down if more
        if refined_contour_count <= orig_contour_count:
            contour_score = 1.0
        else:
            contour_score = max(0.0, 1.0 - (refined_contour_count - orig_contour_count) * 0.2)
        
        # Metric 2: Area preservation (should be similar)
        orig_area = cv2.countNonZero(original_mask)
        refined_area = cv2.countNonZero(refined_mask)
        
        if orig_area == 0:
            area_score = 0.0
        else:
            area_ratio = refined_area / orig_area
            # Good if area is between 80% and 120% of original
            if 0.8 <= area_ratio <= 1.2:
                area_score = 1.0
            elif 0.6 <= area_ratio <= 1.4:
                area_score = 0.7
            else:
                area_score = 0.3
        
        # Metric 3: Edge smoothness (perimeter to area ratio)
        def calculate_smoothness(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if area <= 0:
                return 0.0
            
            # Lower ratio indicates smoother edges
            return (perimeter * perimeter) / (4 * np.pi * area)
        
        orig_smoothness = calculate_smoothness(original_mask)
        refined_smoothness = calculate_smoothness(refined_mask)
        
        # Score: 1.0 if refined is smoother, scaled based on improvement
        if orig_smoothness == 0:
            smoothness_score = 0.5
        else:
            improvement = (orig_smoothness - refined_smoothness) / orig_smoothness
            if improvement >= min_improvement_ratio:
                smoothness_score = min(1.0, 0.5 + improvement * 2.0)
            else:
                smoothness_score = 0.5
        
        # Metric 4: Connectivity (number of connected components)
        orig_components = cv2.connectedComponents(original_mask)[0]
        refined_components = cv2.connectedComponents(refined_mask)[0]
        
        # Score: 1.0 if refined has fewer/equal components
        if refined_components <= orig_components:
            connectivity_score = 1.0
        else:
            connectivity_score = max(0.0, 1.0 - (refined_components - orig_components) * 0.3)
        
        # Combine metrics with weights
        weights = {
            'contour': 0.3,
            'area': 0.3,
            'smoothness': 0.25,
            'connectivity': 0.15
        }
        
        final_score = (
            weights['contour'] * contour_score +
            weights['area'] * area_score +
            weights['smoothness'] * smoothness_score +
            weights['connectivity'] * connectivity_score
        )
        
        return min(1.0, max(0.0, final_score))
        
    except Exception as e:
        print(f"Quality validation failed: {e}")
        return 0.0


def grabcut_performance_monitor(
    start_time: float, 
    success_rate: float,
    image_size: Tuple[int, int],
    iterations: int
) -> Dict[str, Any]:
    """
    Monitor and report GrabCut performance metrics.
    
    This function tracks the performance characteristics of GrabCut operations
    to help optimize parameters and identify potential issues.
    
    Args:
        start_time: Timestamp when GrabCut operation started
        success_rate: Current success rate (0.0 to 1.0)
        image_size: (width, height) of processed image
        iterations: Number of GrabCut iterations used
        
    Returns:
        Dictionary containing performance metrics and recommendations
    """
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate processing rate (pixels per second)
    total_pixels = image_size[0] * image_size[1]
    pixels_per_second = total_pixels / processing_time if processing_time > 0 else 0
    
    # Performance categories
    performance_category = "Unknown"
    if processing_time < 0.1:
        performance_category = "Excellent"
    elif processing_time < 0.5:
        performance_category = "Good"
    elif processing_time < 1.0:
        performance_category = "Acceptable"
    else:
        performance_category = "Slow"
    
    # Generate recommendations
    recommendations = []
    
    if processing_time > 1.0:
        recommendations.append("Consider reducing iterations or image size")
    
    if success_rate < 0.7:
        recommendations.append("Consider adjusting HSV ranges or GrabCut parameters")
    
    if pixels_per_second < 100000:  # Less than 100K pixels/second
        recommendations.append("Performance may be limited by system resources")
    
    return {
        "processing_time_seconds": round(processing_time, 3),
        "pixels_per_second": int(pixels_per_second),
        "performance_category": performance_category,
        "image_size": image_size,
        "iterations": iterations,
        "success_rate": round(success_rate, 3),
        "recommendations": recommendations,
        "timestamp": end_time
    }


def create_grabcut_debug_visualization(
    original_image: np.ndarray,
    hsv_mask: np.ndarray,
    grabcut_mask: np.ndarray,
    quality_score: float
) -> np.ndarray:
    """
    Create a debug visualization showing HSV vs GrabCut comparison.
    
    Args:
        original_image: Original BGR image
        hsv_mask: Original HSV mask
        grabcut_mask: GrabCut refined mask
        quality_score: Quality score from validation
        
    Returns:
        Combined visualization image for debugging
    """
    try:
        height, width = original_image.shape[:2]
        
        # Create side-by-side comparison
        comparison = np.zeros((height, width * 3, 3), dtype=np.uint8)
        
        # Left: Original image
        comparison[:, :width] = original_image
        
        # Middle: HSV mask overlay
        hsv_overlay = original_image.copy()
        hsv_colored = cv2.applyColorMap(hsv_mask, cv2.COLORMAP_JET)
        hsv_overlay = cv2.addWeighted(hsv_overlay, 0.7, hsv_colored, 0.3, 0)
        comparison[:, width:width*2] = hsv_overlay
        
        # Right: GrabCut mask overlay
        grabcut_overlay = original_image.copy()
        grabcut_colored = cv2.applyColorMap(grabcut_mask, cv2.COLORMAP_VIRIDIS)
        grabcut_overlay = cv2.addWeighted(grabcut_overlay, 0.7, grabcut_colored, 0.3, 0)
        comparison[:, width*2:width*3] = grabcut_overlay
        
        # Add quality score text
        text = f"Quality Score: {quality_score:.2f}"
        cv2.putText(comparison, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(comparison, "Original", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "HSV", (width+10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "GrabCut", (width*2+10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return comparison
        
    except Exception as e:
        print(f"Debug visualization creation failed: {e}")
        return original_image.copy()