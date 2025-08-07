"""
Geometric validation system for Tangram piece identification.

This module provides functions to identify and validate Tangram pieces
using mathematical properties such as area ratios, vertex counts, and angles.
"""

from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np
try:
    from .constants import PIECE_AREA_RATIOS, TOLERANCE, MIN_CONTOUR_AREA, PIECE_GEOMETRY, ANGLE_TOLERANCE, CONTOUR_EPSILON_FACTOR, EDGE_TOLERANCE, EDGE_LENGTHS, EXPECTED_PIECE_COUNTS
except ImportError:
    # Fallback for direct execution
    from constants import PIECE_AREA_RATIOS, TOLERANCE, MIN_CONTOUR_AREA, PIECE_GEOMETRY, ANGLE_TOLERANCE, CONTOUR_EPSILON_FACTOR, EDGE_TOLERANCE, EDGE_LENGTHS, EXPECTED_PIECE_COUNTS


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


def count_vertices(contour: np.ndarray) -> int:
    """
    Count the number of vertices in a contour using polygon approximation.
    
    Args:
        contour: OpenCV contour to analyze
        
    Returns:
        Number of vertices in the approximated polygon
    """
    if contour is None or len(contour) < 3:
        return 0
    
    # Calculate epsilon for contour approximation
    epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
    
    # Approximate contour to polygon
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    
    return len(approx_polygon)


def calculate_contour_angles(contour: np.ndarray) -> list:
    """
    Calculate the interior angles of a contour polygon.
    
    Args:
        contour: OpenCV contour to analyze
        
    Returns:
        List of interior angles in degrees
    """
    if contour is None or len(contour) < 3:
        return []
    
    # Approximate contour to polygon
    epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx_polygon) < 3:
        return []
    
    # Calculate angles between consecutive edges
    angles = []
    num_vertices = len(approx_polygon)
    
    for i in range(num_vertices):
        # Get three consecutive points
        p1 = approx_polygon[i - 1][0]  # Previous point
        p2 = approx_polygon[i][0]      # Current point
        p3 = approx_polygon[(i + 1) % num_vertices][0]  # Next point
        
        # Calculate vectors
        v1 = p1 - p2  # Vector from current to previous
        v2 = p3 - p2  # Vector from current to next
        
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms > 0:
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            angles.append(angle_degrees)
    
    return angles


def validate_piece_geometry(contour: np.ndarray, piece_type: str) -> bool:
    """
    Validate that a contour matches the expected geometry for a piece type.
    
    Args:
        contour: OpenCV contour to validate
        piece_type: Expected piece type name
        
    Returns:
        True if geometry is valid, False otherwise
    """
    if contour is None or piece_type not in PIECE_GEOMETRY:
        return False
    
    expected_geometry = PIECE_GEOMETRY[piece_type]
    
    # 1. Validate vertex count
    vertex_count = count_vertices(contour)
    expected_vertices = expected_geometry["vertices"]
    
    if vertex_count != expected_vertices:
        return False
    
    # 2. Validate angles
    calculated_angles = calculate_contour_angles(contour)
    expected_angles = expected_geometry["angles"]
    
    if len(calculated_angles) != len(expected_angles):
        return False
    
    # Sort both angle lists for comparison
    calc_sorted = sorted(calculated_angles)
    exp_sorted = sorted(expected_angles)
    
    # Check if angles match within tolerance
    for calc_angle, exp_angle in zip(calc_sorted, exp_sorted):
        angle_diff = abs(calc_angle - exp_angle)
        if angle_diff > ANGLE_TOLERANCE:
            return False
    
    # 3. Additional validation for right triangles
    if expected_geometry.get("is_right_triangle", False):
        # Check that we have exactly one 90-degree angle
        right_angles = [a for a in calculated_angles if abs(a - 90) <= ANGLE_TOLERANCE]
        if len(right_angles) != 1:
            return False
    
    return True


def classify_piece_by_geometry(contour: np.ndarray) -> Optional[str]:
    """
    Classify a piece type based on geometric properties (vertices and angles).
    
    Args:
        contour: OpenCV contour to classify
        
    Returns:
        Piece type name or None if no match found
    """
    if contour is None:
        return None
    
    vertex_count = count_vertices(contour)
    calculated_angles = calculate_contour_angles(contour)
    
    # Try to match against known piece types
    for piece_type, geometry in PIECE_GEOMETRY.items():
        if geometry["vertices"] == vertex_count:
            if validate_piece_geometry(contour, piece_type):
                return piece_type
    
    return None


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


def calculate_edge_lengths(contour: np.ndarray) -> list:
    """
    Calculate the edge lengths of a contour polygon.
    
    Args:
        contour: OpenCV contour to analyze
        
    Returns:
        List of edge lengths in pixels
    """
    if contour is None or len(contour) < 3:
        return []
    
    # Approximate contour to polygon
    epsilon = CONTOUR_EPSILON_FACTOR * cv2.arcLength(contour, True)
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx_polygon) < 3:
        return []
    
    # Calculate distances between consecutive vertices
    edge_lengths = []
    num_vertices = len(approx_polygon)
    
    for i in range(num_vertices):
        p1 = approx_polygon[i][0]
        p2 = approx_polygon[(i + 1) % num_vertices][0]
        
        # Calculate Euclidean distance
        edge_length = np.linalg.norm(p2 - p1)
        edge_lengths.append(edge_length)
    
    return edge_lengths


def normalize_edge_lengths(edge_lengths: list) -> list:
    """
    Normalize edge lengths relative to the shortest edge.
    
    Args:
        edge_lengths: List of edge lengths in pixels
        
    Returns:
        List of normalized edge length ratios
    """
    if not edge_lengths:
        return []
    
    min_length = min(edge_lengths)
    if min_length == 0:
        return []
    
    return [length / min_length for length in edge_lengths]


def validate_edge_ratios(contour: np.ndarray, piece_type: str) -> bool:
    """
    Validate that edge length ratios match expected values for a piece type.
    
    Args:
        contour: OpenCV contour to validate
        piece_type: Expected piece type name
        
    Returns:
        True if edge ratios are valid, False otherwise
    """
    if contour is None or piece_type not in PIECE_GEOMETRY:
        return False
    
    # Calculate actual edge lengths
    edge_lengths = calculate_edge_lengths(contour)
    if not edge_lengths:
        return False
    
    # Get expected edge pattern for this piece type
    expected_geometry = PIECE_GEOMETRY[piece_type]
    expected_edges = expected_geometry.get("edges", [])
    
    if not expected_edges:
        return True  # No edge constraints defined
    
    expected_edge_list = expected_edges[0]  # Take first edge pattern
    
    # Check if vertex count matches
    if len(edge_lengths) != len(expected_edge_list):
        return False
    
    # Normalize both actual and expected edge lengths
    normalized_actual = normalize_edge_lengths(edge_lengths)
    normalized_expected = normalize_edge_lengths(expected_edge_list)
    
    # Sort both for comparison (since orientation may vary)
    actual_sorted = sorted(normalized_actual)
    expected_sorted = sorted(normalized_expected)
    
    # Check if ratios match within tolerance
    for actual_ratio, expected_ratio in zip(actual_sorted, expected_sorted):
        relative_error = abs(actual_ratio - expected_ratio) / expected_ratio
        if relative_error > EDGE_TOLERANCE:
            return False
    
    return True


def classify_piece_by_edges(contour: np.ndarray) -> Optional[str]:
    """
    Classify a piece type based on edge length ratios.
    
    Args:
        contour: OpenCV contour to classify
        
    Returns:
        Piece type name or None if no match found
    """
    if contour is None:
        return None
    
    edge_lengths = calculate_edge_lengths(contour)
    if not edge_lengths:
        return None
    
    # Try to match against known piece types
    for piece_type in PIECE_GEOMETRY.keys():
        if validate_edge_ratios(contour, piece_type):
            return piece_type
    
    return None


def calculate_piece_rotation(contour: np.ndarray) -> float:
    """
    Calculate the rotation angle of a piece relative to its standard orientation.
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Rotation angle in radians
    """
    if contour is None or len(contour) < 3:
        return 0.0
    
    # Use minimum area rectangle to find orientation
    rect = cv2.minAreaRect(contour)
    angle_degrees = rect[2]
    
    # Convert to radians and normalize
    angle_radians = np.radians(angle_degrees)
    
    return angle_radians


def complete_geometric_validation(contour: np.ndarray, piece_type: str) -> bool:
    """
    Complete geometric validation using area, vertices, angles, and edge lengths.
    
    Args:
        contour: OpenCV contour to validate
        piece_type: Expected piece type name
        
    Returns:
        True if all geometric properties are valid, False otherwise
    """
    if contour is None or piece_type not in PIECE_GEOMETRY:
        return False
    
    # 1. Validate vertex count and angles
    if not validate_piece_geometry(contour, piece_type):
        return False
    
    # 2. Validate edge length ratios
    if not validate_edge_ratios(contour, piece_type):
        return False
    
    return True


def enhanced_piece_classification(contour: np.ndarray, base_unit_area: float) -> Optional[str]:
    """
    Enhanced piece classification using area ratios, geometry, and edge validation.
    
    Args:
        contour: OpenCV contour to classify
        base_unit_area: Reference area of the smallest triangle
        
    Returns:
        Piece type name or None if no match found
    """
    if contour is None or base_unit_area <= 0:
        return None
    
    # Try different classification methods
    area_classification = classify_piece_by_area(contour, base_unit_area)
    geometry_classification = classify_piece_by_geometry(contour)
    edge_classification = classify_piece_by_edges(contour)
    
    # Count how many methods agree on each classification
    classifications = [area_classification, geometry_classification, edge_classification]
    valid_classifications = [c for c in classifications if c is not None]
    
    if not valid_classifications:
        return None
    
    # Find the most common classification
    from collections import Counter
    classification_counts = Counter(valid_classifications)
    most_common = classification_counts.most_common(1)[0]
    candidate_type, vote_count = most_common
    
    # Require at least 2 out of 3 methods to agree, or complete validation
    if vote_count >= 2:
        # Additional validation: check complete geometric properties
        if complete_geometric_validation(contour, candidate_type):
            return candidate_type
    
    # Fallback: if only one method succeeds but passes complete validation
    if len(valid_classifications) == 1:
        candidate_type = valid_classifications[0]
        if complete_geometric_validation(contour, candidate_type):
            return candidate_type
    
    return None


def comprehensive_piece_classification(contour: np.ndarray, base_unit_area: float) -> dict:
    """
    Comprehensive analysis of a piece with all classification methods.
    
    Args:
        contour: OpenCV contour to analyze
        base_unit_area: Reference area of the smallest triangle
        
    Returns:
        Dictionary with detailed classification results
    """
    if contour is None:
        return {}
    
    result = {
        'area': calculate_contour_area(contour),
        'vertices': count_vertices(contour),
        'angles': calculate_contour_angles(contour),
        'edge_lengths': calculate_edge_lengths(contour),
        'area_classification': None,
        'geometry_classification': None,
        'edge_classification': None,
        'enhanced_classification': None,
        'validation_results': {}
    }
    
    if base_unit_area > 0:
        result['area_ratio'] = result['area'] / base_unit_area
        result['area_classification'] = classify_piece_by_area(contour, base_unit_area)
    
    result['geometry_classification'] = classify_piece_by_geometry(contour)
    result['edge_classification'] = classify_piece_by_edges(contour)
    
    if base_unit_area > 0:
        result['enhanced_classification'] = enhanced_piece_classification(contour, base_unit_area)
    
    # Test validation against all piece types
    for piece_type in PIECE_GEOMETRY.keys():
        result['validation_results'][piece_type] = {
            'geometry_valid': validate_piece_geometry(contour, piece_type),
            'edges_valid': validate_edge_ratios(contour, piece_type),
            'complete_valid': complete_geometric_validation(contour, piece_type)
        }
    
    return result


def validate_complete_tangram_set(classified_pieces: Dict[str, int]) -> bool:
    """
    Validate that the detected pieces form a complete Tangram set.
    
    Args:
        classified_pieces: Dictionary of piece_type -> count
        
    Returns:
        True if set is complete and valid, False otherwise
    """
    
    # Check if we have the exact expected count for each piece type
    for piece_type, expected_count in EXPECTED_PIECE_COUNTS.items():
        actual_count = classified_pieces.get(piece_type, 0)
        if actual_count != expected_count:
            return False
    
    # Check that we don't have any extra piece types
    for piece_type in classified_pieces:
        if piece_type not in EXPECTED_PIECE_COUNTS:
            return False
    
    return True


# =====================================
# OpenCV Integration Functions (Task 4.4)
# =====================================

def cv2_geometric_analyzer(contour: np.ndarray, debug: bool = False) -> Dict:
    """
    Complete OpenCV-integrated geometric analysis of a contour.
    
    Args:
        contour: OpenCV contour to analyze
        debug: Whether to include debug information
        
    Returns:
        Dictionary with comprehensive geometric analysis using OpenCV functions
    """
    if contour is None or len(contour) < 3:
        return {}
    
    analysis = {
        'contour_properties': {},
        'opencv_features': {},
        'geometric_validation': {},
        'classification_results': {}
    }
    
    # Basic contour properties using OpenCV
    analysis['contour_properties'] = {
        'area': cv2.contourArea(contour),
        'perimeter': cv2.arcLength(contour, True),
        'vertices': len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)),
        'is_convex': cv2.isContourConvex(contour),
        'centroid': None,
        'orientation': 0.0
    }
    
    # Calculate centroid using OpenCV moments
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        analysis['contour_properties']['centroid'] = (cx, cy)
    
    # Calculate orientation using minimum area rectangle
    if len(contour) >= 5:
        rect = cv2.minAreaRect(contour)
        analysis['contour_properties']['orientation'] = rect[2]
        analysis['opencv_features']['min_area_rect'] = {
            'center': rect[0],
            'size': rect[1],
            'angle': rect[2]
        }
    
    # Convex hull analysis
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    analysis['opencv_features']['convex_hull'] = {
        'area': hull_area,
        'convexity': analysis['contour_properties']['area'] / hull_area if hull_area > 0 else 0
    }
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    analysis['opencv_features']['bounding_rect'] = {
        'x': x, 'y': y, 'width': w, 'height': h,
        'aspect_ratio': w / h if h > 0 else 0
    }
    
    # Contour approximation with different epsilons
    analysis['opencv_features']['approximations'] = {}
    for epsilon_factor in [0.01, 0.02, 0.03]:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        analysis['opencv_features']['approximations'][f'epsilon_{epsilon_factor}'] = {
            'vertices': len(approx),
            'points': approx.tolist() if debug else len(approx)
        }
    
    return analysis


def cv2_piece_classifier(contour: np.ndarray, base_unit_area: float = None) -> Dict:
    """
    OpenCV-optimized piece classification using integrated cv2 functions.
    
    Args:
        contour: OpenCV contour to classify
        base_unit_area: Reference area for ratio calculations
        
    Returns:
        Classification results with OpenCV optimization
    """
    if contour is None:
        return {}
    
    # Use OpenCV functions for efficient geometric analysis
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Efficient polygon approximation
    epsilon = 0.02 * perimeter
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    vertex_count = len(approx_polygon)
    
    # Fast convexity check
    is_convex = cv2.isContourConvex(contour)
    
    # Aspect ratio from bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Circularity metric
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Classification based on OpenCV-derived features
    classification = {
        'opencv_metrics': {
            'area': area,
            'perimeter': perimeter,
            'vertices': vertex_count,
            'is_convex': is_convex,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity
        },
        'piece_classification': None,
        'confidence': 0.0
    }
    
    # Rule-based classification using OpenCV metrics
    if vertex_count == 3 and is_convex:
        # Triangle classification
        if base_unit_area and area > 0:
            area_ratio = area / base_unit_area
            if 0.8 <= area_ratio <= 1.2:
                classification['piece_classification'] = 'small_triangle'
                classification['confidence'] = 0.9
            elif 1.8 <= area_ratio <= 2.2:
                classification['piece_classification'] = 'medium_triangle'
                classification['confidence'] = 0.85
            elif 3.8 <= area_ratio <= 4.2:
                classification['piece_classification'] = 'large_triangle'
                classification['confidence'] = 0.85
        else:
            classification['piece_classification'] = 'small_triangle'  # Default
            classification['confidence'] = 0.6
    
    elif vertex_count == 4 and is_convex:
        # Quadrilateral classification
        if 0.9 <= aspect_ratio <= 1.1:
            classification['piece_classification'] = 'square'
            classification['confidence'] = 0.9
        else:
            classification['piece_classification'] = 'parallelogram'
            classification['confidence'] = 0.8
    
    # Enhance with our existing comprehensive analysis
    if base_unit_area:
        enhanced_result = enhanced_piece_classification(contour, base_unit_area)
        if enhanced_result:
            classification['piece_classification'] = enhanced_result
            classification['confidence'] = min(classification['confidence'] + 0.1, 1.0)
    
    return classification


def cv2_validation_pipeline(contours: list, debug: bool = False) -> Dict:
    """
    Complete validation pipeline using OpenCV functions throughout.
    
    Args:
        contours: List of OpenCV contours to process
        debug: Whether to include debug information
        
    Returns:
        Complete validation results using OpenCV integration
    """
    if not contours:
        return {'error': 'No contours provided'}
    
    pipeline_results = {
        'input_summary': {
            'total_contours': len(contours),
            'valid_contours': 0,
            'base_unit_area': 0
        },
        'contour_analysis': [],
        'piece_classifications': {},
        'validation_summary': {
            'total_classified': 0,
            'classification_confidence': {},
            'tangram_set_valid': False
        }
    }
    
    # Filter contours using OpenCV functions
    valid_contours = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= MIN_CONTOUR_AREA:
            # Additional OpenCV-based quality checks
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            if convexity >= 0.6:  # Reasonable convexity
                valid_contours.append(contour)
                
                if debug:
                    pipeline_results['contour_analysis'].append({
                        'index': i,
                        'area': area,
                        'convexity': convexity,
                        'status': 'valid'
                    })
    
    pipeline_results['input_summary']['valid_contours'] = len(valid_contours)
    
    if not valid_contours:
        pipeline_results['error'] = 'No valid contours after OpenCV filtering'
        return pipeline_results
    
    # Calculate base unit area using OpenCV
    try:
        base_unit_area = calculate_base_unit_area(valid_contours)
        pipeline_results['input_summary']['base_unit_area'] = base_unit_area
    except ValueError as e:
        pipeline_results['error'] = f'Base unit calculation failed: {e}'
        return pipeline_results
    
    # Classify each contour using OpenCV-optimized methods
    piece_counts = {}
    total_confidence = 0
    classified_count = 0
    
    for i, contour in enumerate(valid_contours):
        # OpenCV-optimized classification
        classification = cv2_piece_classifier(contour, base_unit_area)
        
        piece_type = classification.get('piece_classification')
        confidence = classification.get('confidence', 0)
        
        if piece_type:
            piece_counts[piece_type] = piece_counts.get(piece_type, 0) + 1
            total_confidence += confidence
            classified_count += 1
            
            if debug:
                # Add comprehensive analysis for debug
                comprehensive = comprehensive_piece_classification(contour, base_unit_area)
                classification['comprehensive_analysis'] = comprehensive
        
        pipeline_results['contour_analysis'].append({
            'index': i,
            'classification': classification,
            'opencv_analysis': cv2_geometric_analyzer(contour, debug=debug)
        })
    
    # Summarize results
    pipeline_results['piece_classifications'] = piece_counts
    pipeline_results['validation_summary']['total_classified'] = classified_count
    
    if classified_count > 0:
        avg_confidence = total_confidence / classified_count
        pipeline_results['validation_summary']['classification_confidence'] = {
            'average': avg_confidence,
            'total_pieces': classified_count
        }
    
    # Validate complete Tangram set
    pipeline_results['validation_summary']['tangram_set_valid'] = validate_complete_tangram_set(piece_counts)
    
    return pipeline_results


def cv2_process_image_with_validation(image_path: str, debug: bool = False) -> Dict:
    """
    Complete image processing and validation pipeline using OpenCV integration.
    
    Args:
        image_path: Path to the image file
        debug: Whether to include debug information and save debug images
        
    Returns:
        Complete processing results with OpenCV integration
    """
    try:
        from .processor import extract_pieces_from_image
    except ImportError:
        from processor import extract_pieces_from_image
    
    results = {
        'image_path': image_path,
        'processing_status': 'started',
        'piece_extraction': {},
        'validation_results': {},
        'final_summary': {}
    }
    
    try:
        # Extract pieces using our enhanced processor
        detected_pieces = extract_pieces_from_image(image_path)
        
        results['piece_extraction'] = {
            'detected_count': len(detected_pieces),
            'detected_colors': [color for color, _ in detected_pieces],
            'status': 'success'
        }
        
        # Extract contours for validation
        contours = [contour for _, contour in detected_pieces]
        
        # Run OpenCV validation pipeline
        validation_results = cv2_validation_pipeline(contours, debug=debug)
        results['validation_results'] = validation_results
        
        # Create final summary
        classified_pieces = validation_results.get('piece_classifications', {})
        total_classified = validation_results.get('validation_summary', {}).get('total_classified', 0)
        is_complete_set = validation_results.get('validation_summary', {}).get('tangram_set_valid', False)
        
        results['final_summary'] = {
            'total_pieces_detected': len(detected_pieces),
            'total_pieces_classified': total_classified,
            'classification_rate': total_classified / len(detected_pieces) if detected_pieces else 0,
            'piece_distribution': classified_pieces,
            'is_complete_tangram_set': is_complete_set,
            'processing_status': 'completed'
        }
        
        results['processing_status'] = 'completed'
        
    except Exception as e:
        results['processing_status'] = 'failed'
        results['error'] = str(e)
    
    return results