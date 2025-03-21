# /learn-ai-ml-dl/phase0/linear_algebra/src/vector_ops.py

"""
Vector operations implemented from scratch.
This module provides fundamental vector operations without using NumPy.
"""

def create_vector(components):
    """Create a vector from a list of components."""
    return components.copy()

def vector_add(v1, v2):
    """Add two vectors element-wise."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    return [v1[i] + v2[i] for i in range(len(v1))]

def vector_subtract(v1, v2):
    """Subtract second vector from first vector element-wise."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    return [v1[i] - v2[i] for i in range(len(v1))]

def scalar_multiply(scalar, vector):
    """Multiply a vector by a scalar."""
    return [scalar * component for component in vector]

def dot_product(v1, v2):
    """Calculate the dot product of two vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    return sum(v1[i] * v2[i] for i in range(len(v1)))

def magnitude(vector):
    """Calculate the magnitude (length) of a vector."""
    return (sum(component ** 2 for component in vector)) ** 0.5

def normalize(vector):
    """Return a unit vector in the same direction."""
    mag = magnitude(vector)
    if mag == 0:
        raise ValueError("Cannot normalize a zero vector")
    return [component / mag for component in vector]

def vector_to_string(vector):
    """Return a string representation of a vector."""
    return "(" + ", ".join(str(component) for component in vector) + ")"