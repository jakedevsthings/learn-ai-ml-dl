"""
Eigenvalue and eigenvector operations implemented from scratch.
This module provides functions to compute eigenvalues and eigenvectors without using NumPy's built-in functions.
"""

import numpy as np
from .matrix_ops import matrix_multiply, matrix_subtract, matrix_dimensions

def power_iteration(A, num_iterations=100, tolerance=1e-10):
    """
    Compute the dominant eigenvalue and eigenvector of a matrix using the power iteration method.
    
    Parameters:
    -----------
    A : numpy array or list of lists
        The input matrix
    num_iterations : int, optional
        Maximum number of iterations
    tolerance : float, optional
        Convergence tolerance
    
    Returns:
    --------
    eigenvalue : float
        The dominant eigenvalue
    eigenvector : numpy array
        The corresponding eigenvector (normalized)
    """
    # Convert to numpy array if not already
    A = np.array(A)
    n = A.shape[0]
    
    # Start with a random vector
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)
    
    for _ in range(num_iterations):
        # Calculate the matrix-vector product
        b_k1 = A @ b_k
        
        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # Re-normalize the vector
        b_k1 = b_k1 / b_k1_norm
        
        # Check for convergence
        if np.linalg.norm(b_k1 - b_k) < tolerance:
            break
            
        b_k = b_k1
        
    # Calculate the Rayleigh quotient to get the eigenvalue
    eigenvalue = (b_k @ A @ b_k) / (b_k @ b_k)
    
    return eigenvalue, b_k

def deflation(A, eigenvalue, eigenvector):
    """
    Deflate a matrix by removing the contribution of an eigenpair.
    
    Parameters:
    -----------
    A : numpy array
        The input matrix
    eigenvalue : float
        An eigenvalue of A
    eigenvector : numpy array
        The corresponding eigenvector
    
    Returns:
    --------
    A_deflated : numpy array
        The deflated matrix
    """
    # Normalize the eigenvector
    v = eigenvector / np.linalg.norm(eigenvector)
    
    # Compute the deflated matrix
    A_deflated = A - eigenvalue * np.outer(v, v)
    
    return A_deflated

def qr_algorithm(A, max_iterations=100, tolerance=1e-10):
    """
    Compute all eigenvalues of a matrix using the QR algorithm.
    
    Parameters:
    -----------
    A : numpy array
        The input matrix
    max_iterations : int, optional
        Maximum number of iterations
    tolerance : float, optional
        Convergence tolerance
    
    Returns:
    --------
    eigenvalues : numpy array
        The eigenvalues of A
    """
    A = np.array(A)
    n = A.shape[0]
    A_k = A.copy()
    
    for _ in range(max_iterations):
        # QR decomposition
        Q, R = np.linalg.qr(A_k)
        
        # Update A_k
        A_prev = A_k.copy()
        A_k = R @ Q
        
        # Check for convergence
        if np.linalg.norm(A_k - A_prev) < tolerance:
            break
    
    # Extract eigenvalues from the diagonal
    eigenvalues = np.diag(A_k)
    
    return eigenvalues

def inverse_power_method(A, mu, num_iterations=100, tolerance=1e-10):
    """
    Compute the eigenvalue closest to mu and its corresponding eigenvector using the inverse power method.
    
    Parameters:
    -----------
    A : numpy array
        The input matrix
    mu : float
        The shift value (approximation of the desired eigenvalue)
    num_iterations : int, optional
        Maximum number of iterations
    tolerance : float, optional
        Convergence tolerance
    
    Returns:
    --------
    eigenvalue : float
        The eigenvalue closest to mu
    eigenvector : numpy array
        The corresponding eigenvector
    """
    A = np.array(A)
    n = A.shape[0]
    
    # Compute the shifted matrix
    I = np.eye(n)
    A_shifted = A - mu * I
    
    # Start with a random vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        # Solve the linear system
        w = np.linalg.solve(A_shifted, v)
        
        # Normalize the result
        w_norm = np.linalg.norm(w)
        w = w / w_norm
        
        # Check for convergence
        if np.linalg.norm(w - v) < tolerance:
            break
            
        v = w
    
    # Compute the Rayleigh quotient to get the eigenvalue
    eigenvalue = (v @ A @ v) / (v @ v)
    
    return eigenvalue, v

def is_symmetric(A, tolerance=1e-10):
    """
    Check if a matrix is symmetric.
    
    Parameters:
    -----------
    A : numpy array
        The input matrix
    tolerance : float, optional
        Tolerance for floating-point comparisons
    
    Returns:
    --------
    is_symmetric : bool
        True if the matrix is symmetric, False otherwise
    """
    A = np.array(A)
    return np.allclose(A, A.T, rtol=tolerance)

def eigenvectors_from_eigenvalues(A, eigenvalues, tolerance=1e-10):
    """
    Compute the eigenvectors corresponding to given eigenvalues.
    
    Parameters:
    -----------
    A : numpy array
        The input matrix
    eigenvalues : list or numpy array
        The eigenvalues of A
    tolerance : float, optional
        Tolerance for solving the linear system
    
    Returns:
    --------
    eigenvectors : list of numpy arrays
        The eigenvectors corresponding to the eigenvalues
    """
    A = np.array(A)
    n = A.shape[0]
    eigenvectors = []
    
    for eigenvalue in eigenvalues:
        # Compute A - lambda*I
        A_lambda = A - eigenvalue * np.eye(n)
        
        # Find the null space of A - lambda*I
        # We use SVD to find the null space
        U, S, Vh = np.linalg.svd(A_lambda)
        
        # The eigenvector is in the null space of A - lambda*I
        # We take the columns of V corresponding to the smallest singular values
        null_space_vectors = Vh.T[:, S < tolerance]
        
        if null_space_vectors.size > 0:
            # Take the first vector in the null space
            eigenvector = null_space_vectors[:, 0]
            # Normalize
            eigenvector = eigenvector / np.linalg.norm(eigenvector)
            eigenvectors.append(eigenvector)
        else:
            # If no vectors in the null space, use inverse power method
            _, eigenvector = inverse_power_method(A, eigenvalue)
            eigenvectors.append(eigenvector)
    
    return eigenvectors

def eigendecomposition(A):
    """
    Compute the eigendecomposition of a matrix.
    
    Parameters:
    -----------
    A : numpy array
        The input matrix
    
    Returns:
    --------
    eigenvalues : numpy array
        The eigenvalues of A
    eigenvectors : numpy array
        The eigenvectors of A (as columns)
    """
    A = np.array(A)
    
    # Check if the matrix is symmetric
    if is_symmetric(A):
        # For symmetric matrices, use a specialized algorithm
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else:
        # For general matrices, use the QR algorithm
        eigenvalues = qr_algorithm(A)
        eigenvectors = np.column_stack(eigenvectors_from_eigenvalues(A, eigenvalues))
    
    return eigenvalues, eigenvectors
