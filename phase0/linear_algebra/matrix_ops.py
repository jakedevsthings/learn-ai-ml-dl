# /learn-ai-ml-dl/phase0/linear_algebra/matrix_ops.py

"""
Matrix operations implemented from scratch.
This module provides fundamental matrix operations without using NumPy.
"""

def create_matrix(rows, cols, init_val=0):
    """Create a matrix with the specified dimensions."""
    return [[init_val for _ in range(cols)] for _ in range(rows)]

def matrix_from_rows(rows):
    """Create a matrix from a list of row vectors."""
    return [row.copy() for row in rows]

def matrix_dimensions(matrix):
    """Return the dimensions of a matrix as (rows, columns)."""
    return len(matrix), len(matrix[0]) if matrix else 0

def matrix_add(m1, m2):
    """Add two matrices element-wise."""
    rows1, cols1 = matrix_dimensions(m1)
    rows2, cols2 = matrix_dimensions(m2)
    
    if (rows1, cols1) != (rows2, cols2):
        raise ValueError("Matrices must have the same dimensions")
    
    result = create_matrix(rows1, cols1)
    for i in range(rows1):
        for j in range(cols1):
            result[i][j] = m1[i][j] + m2[i][j]
    
    return result

def matrix_subtract(m1, m2):
    """Subtract second matrix from first matrix element-wise."""
    rows1, cols1 = matrix_dimensions(m1)
    rows2, cols2 = matrix_dimensions(m2)
    
    if (rows1, cols1) != (rows2, cols2):
        raise ValueError("Matrices must have the same dimensions")
    
    result = create_matrix(rows1, cols1)
    for i in range(rows1):
        for j in range(cols1):
            result[i][j] = m1[i][j] - m2[i][j]
    
    return result

def matrix_multiply(m1, m2):
    """Multiply two matrices."""
    rows1, cols1 = matrix_dimensions(m1)
    rows2, cols2 = matrix_dimensions(m2)
    
    if cols1 != rows2:
        raise ValueError(f"Matrix dimensions incompatible for multiplication: {(rows1, cols1)} and {(rows2, cols2)}")
    
    result = create_matrix(rows1, cols2)
    for i in range(rows1):
        for j in range(cols2):
            result[i][j] = sum(m1[i][k] * m2[k][j] for k in range(cols1))
    
    return result

def matrix_scalar_multiply(scalar, matrix):
    """Multiply a matrix by a scalar."""
    rows, cols = matrix_dimensions(matrix)
    result = create_matrix(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            result[i][j] = scalar * matrix[i][j]
    
    return result

def matrix_transpose(matrix):
    """Transpose a matrix."""
    rows, cols = matrix_dimensions(matrix)
    result = create_matrix(cols, rows)
    
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    
    return result

def identity_matrix(n):
    """Create an n×n identity matrix."""
    result = create_matrix(n, n)
    for i in range(n):
        result[i][i] = 1
    return result


def is_linearly_independent(vectors):
    """
    Determines if a set of vectors is linearly independent.
    Args:
        vectors: list of vectors (each a list of floats or ints)
    Returns:
        True if linearly independent, False otherwise
    """
    import copy

    if not vectors:
        return True  # empty set is trivially independent

    # Transpose: convert list of column vectors to row-wise matrix
    matrix = [list(row) for row in zip(*vectors)]
    rows = len(matrix)
    cols = len(matrix[0])
    mat = copy.deepcopy(matrix)

    pivot_count = 0
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while mat[i][lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead == cols:
                    break
        if lead == cols:
            break
        # Swap to bring pivot row up
        mat[i], mat[r] = mat[r], mat[i]

        # Normalize pivot row
        lv = mat[r][lead]
        mat[r] = [mrx / lv for mrx in mat[r]]

        # Eliminate below and above
        for i in range(rows):
            if i != r:
                lv = mat[i][lead]
                mat[i] = [iv - lv * rv for rv, iv in zip(mat[r], mat[i])]

        pivot_count += 1
        lead += 1

    # If number of pivot columns = number of input vectors → independent
    return pivot_count == len(vectors)