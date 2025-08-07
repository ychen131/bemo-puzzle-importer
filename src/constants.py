"""
Constants and specifications for Tangram piece processing.

This module contains HSV color ranges for piece identification, mathematical
specifications for geometric validation, and world space dimensions for
coordinate transformation.

Mathematical Foundation:
- All pieces derive from a square with total area = 8 square units
- Area ratios: Small(0.5), Medium(1), Large(2), Square(1), Parallelogram(1)
- Edge lengths: 1, √2, 2, 2√2 units
- Standard angles: 45°, 90°, 135°
"""

from typing import Dict, Tuple, List
import numpy as np
import math

# =============================================================================
# WORLD SPACE COORDINATE SYSTEM
# =============================================================================

# World space dimensions (target resolution for processing)
WORLD_SPACE_WIDTH = 800
WORLD_SPACE_HEIGHT = 600

# Coordinate system origin (top-left corner in image coordinates)
WORLD_ORIGIN_X = 0
WORLD_ORIGIN_Y = 0

# Pixels per world unit for scaling calculations
PIXELS_PER_WORLD_UNIT = 100  # 1 world unit = 100 pixels

# =============================================================================
# TANGRAM MATHEMATICAL SPECIFICATIONS
# =============================================================================

# Total area of complete Tangram set (in square units)
TOTAL_TANGRAM_AREA = 8.0

# Piece area specifications (in square units, exact mathematical values)
PIECE_AREAS = {
    "small_triangle": 0.5,      # Two pieces: 2 × 0.5 = 1.0
    "medium_triangle": 1.0,     # One piece:  1 × 1.0 = 1.0  
    "large_triangle": 2.0,      # Two pieces: 2 × 2.0 = 4.0
    "square": 1.0,              # One piece:  1 × 1.0 = 1.0
    "parallelogram": 1.0        # One piece:  1 × 1.0 = 1.0
}                               # Total: 8.0 square units

# Area ratios relative to small triangle (small = 1 unit for easier calculations)
PIECE_AREA_RATIOS = {
    "small_triangle": 1,        # Base unit: 0.5 → 1
    "medium_triangle": 2,       # 1.0 → 2  
    "large_triangle": 4,        # 2.0 → 4
    "square": 2,                # 1.0 → 2
    "parallelogram": 2          # 1.0 → 2
}

# Expected piece counts in a complete Tangram set
EXPECTED_PIECE_COUNTS = {
    "small_triangle": 2,
    "medium_triangle": 1,
    "large_triangle": 2,
    "square": 1,
    "parallelogram": 1
}

# Edge length specifications (in world units)
EDGE_LENGTHS = {
    "unit": 1.0,                    # Base unit length
    "sqrt2": math.sqrt(2),          # √2 ≈ 1.414
    "double": 2.0,                  # 2 × unit
    "double_sqrt2": 2 * math.sqrt(2) # 2√2 ≈ 2.828
}

# Standard angles in degrees and radians
ANGLES = {
    "right": {
        "degrees": 90,
        "radians": math.pi / 2
    },
    "acute": {
        "degrees": 45,
        "radians": math.pi / 4
    },
    "obtuse": {
        "degrees": 135,
        "radians": 3 * math.pi / 4
    }
}

# Geometric properties for each piece type
PIECE_GEOMETRY = {
    "small_triangle": {
        "vertices": 3,
        "edges": [(1.0, 1.0, math.sqrt(2))],  # Two legs of length 1, hypotenuse √2
        "angles": [90, 45, 45],  # degrees
        "area": 0.5,
        "is_right_triangle": True
    },
    "medium_triangle": {
        "vertices": 3,
        "edges": [(math.sqrt(2), math.sqrt(2), 2.0)],  # Two legs √2, hypotenuse 2
        "angles": [90, 45, 45],  # degrees
        "area": 1.0,
        "is_right_triangle": True
    },
    "large_triangle": {
        "vertices": 3,
        "edges": [(2.0, 2.0, 2 * math.sqrt(2))],  # Two legs 2, hypotenuse 2√2
        "angles": [90, 45, 45],  # degrees
        "area": 2.0,
        "is_right_triangle": True
    },
    "square": {
        "vertices": 4,
        "edges": [(1.0, 1.0, 1.0, 1.0)],  # All sides length 1
        "angles": [90, 90, 90, 90],  # degrees
        "area": 1.0,
        "is_right_triangle": False
    },
    "parallelogram": {
        "vertices": 4,
        "edges": [(math.sqrt(2), 1.0, math.sqrt(2), 1.0)],  # Alternating √2, 1
        "angles": [45, 135, 45, 135],  # degrees
        "area": 1.0,
        "is_right_triangle": False
    }
}

# =============================================================================
# HSV COLOR RANGES
# =============================================================================

# HSV color ranges for Tangram pieces (in OpenCV HSV format: H=0-179, S=0-255, V=0-255)
# These ranges are optimized for robust detection across various lighting conditions
# and account for camera variations while maintaining distinct color separation.
HSV_COLOR_RANGES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "red": {
        # Red wraps around HSV hue spectrum, so we need two ranges
        "lower": (0, 100, 100),      # Primary red range
        "upper": (10, 255, 255),
        "lower2": (170, 100, 100),   # Secondary red range (high hue values)
        "upper2": (179, 255, 255)
    },
    "orange": {
        "lower": (11, 100, 100),     # Distinct from red and yellow
        "upper": (25, 255, 255)
    },
    "yellow": {
        "lower": (26, 80, 120),      # Lower saturation threshold for yellow
        "upper": (35, 255, 255)
    },
    "green": {
        "lower": (36, 100, 80),      # Wide range for various green shades
        "upper": (85, 255, 255)
    },
    "blue": {
        "lower": (86, 120, 80),      # Distinct from green and purple
        "upper": (125, 255, 255)
    },
    "purple": {
        "lower": (126, 100, 80),     # Violet/purple range
        "upper": (155, 255, 255)
    },
    "pink": {
        "lower": (150, 50, 100),     # Pink/magenta range (using alternative range)
        "upper": (175, 255, 255)     # Better detection for pink
    }
}

# Alternative HSV ranges for challenging lighting conditions
# These can be used as fallback ranges if primary detection fails
HSV_COLOR_RANGES_ALTERNATIVE: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "red": {
        "lower": (0, 70, 70),
        "upper": (15, 255, 255),
        "lower2": (165, 70, 70),
        "upper2": (179, 255, 255)
    },
    "orange": {
        "lower": (16, 70, 100),
        "upper": (30, 255, 255)
    },
    "yellow": {
        "lower": (20, 50, 100),      # More permissive for yellow
        "upper": (40, 255, 255)
    },
    "green": {
        "lower": (35, 70, 60),
        "upper": (90, 255, 255)
    },
    "blue": {
        "lower": (80, 80, 60),
        "upper": (130, 255, 255)
    },
    "purple": {
        "lower": (120, 70, 60),
        "upper": (160, 255, 255)
    },
    "pink": {
        "lower": (150, 50, 100),
        "upper": (175, 255, 255)
    }
}

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

# Tolerance for mathematical calculations
TOLERANCE = 0.01  # 1% tolerance for area and geometric calculations
ANGLE_TOLERANCE = 2.0  # degrees tolerance for angle measurements
EDGE_TOLERANCE = 0.05  # 5% tolerance for edge length comparisons

# Minimum contour area (in pixels) to be considered a valid piece
MIN_CONTOUR_AREA = 100

# Maximum allowed deviation for area validation
MAX_AREA_DEVIATION = 0.02  # 2% maximum deviation from expected area

# Contour approximation parameters
CONTOUR_EPSILON_FACTOR = 0.02  # Percentage of perimeter for contour approximation

# =============================================================================
# PIECE TYPE MAPPINGS FOR BEMO APP
# =============================================================================

# Color to piece type mapping (for JSON export)
COLOR_TO_PIECE_TYPE = {
    "red": "largeTriangle1",
    "orange": "largeTriangle2", 
    "yellow": "mediumTriangle",
    "green": "smallTriangle1",
    "blue": "smallTriangle2",
    "purple": "square",
    "pink": "parallelogram"
}

# Reverse mapping for validation
PIECE_TYPE_TO_COLOR = {v: k for k, v in COLOR_TO_PIECE_TYPE.items()}

# Default z-index values for layering
PIECE_Z_INDICES = {
    "largeTriangle1": 1,
    "largeTriangle2": 2,
    "mediumTriangle": 3,
    "smallTriangle1": 4,
    "smallTriangle2": 5,
    "square": 6,
    "parallelogram": 7
}

# =============================================================================
# UTILITY FUNCTIONS FOR MATHEMATICAL VALIDATION
# =============================================================================

def get_expected_area_for_piece_type(piece_type: str) -> float:
    """
    Get the expected area for a specific piece type.
    
    Args:
        piece_type: Type of Tangram piece
        
    Returns:
        Expected area in square units
        
    Raises:
        KeyError: If piece_type is not recognized
    """
    piece_category = piece_type.replace("1", "").replace("2", "")
    if piece_category.endswith("Triangle"):
        piece_category = piece_category.replace("Triangle", "_triangle").lower()
    
    return PIECE_AREAS[piece_category]


def validate_area_ratio(calculated_area: float, expected_area: float) -> bool:
    """
    Validate that calculated area is within tolerance of expected area.
    
    Args:
        calculated_area: Measured area from contour
        expected_area: Expected area from specifications
        
    Returns:
        True if within tolerance, False otherwise
    """
    if expected_area == 0:
        return False
    
    ratio = abs(calculated_area - expected_area) / expected_area
    return ratio <= MAX_AREA_DEVIATION


def get_piece_category_from_area_ratio(area_ratio: float) -> str:
    """
    Determine piece category based on area ratio to smallest piece.
    
    Args:
        area_ratio: Ratio of piece area to smallest triangle area
        
    Returns:
        Most likely piece category name
    """
    best_match = None
    min_difference = float('inf')
    
    for piece_type, expected_ratio in PIECE_AREA_RATIOS.items():
        difference = abs(area_ratio - expected_ratio)
        if difference < min_difference:
            min_difference = difference
            best_match = piece_type
    
    return best_match if min_difference <= TOLERANCE * max(PIECE_AREA_RATIOS.values()) else None