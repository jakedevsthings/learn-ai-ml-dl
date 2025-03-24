"""
Principal Component Analysis (PCA) implemented from scratch.
This module provides functions to perform PCA without using scikit-learn's built-in functions.
"""

import numpy as np
from .eigen import eigendecomposition

def compute_covariance_matrix(X):
    """
    Compute the covariance matrix of a data matrix.
    
    Parameters:
    -----------
    X : numpy array
        Data matrix with shape (n_samples, n_features)
        Each row is a sample, each column is a feature
    
    Returns:
    --------
    cov_matrix : numpy array
        Covariance matrix with shape (n_features, n_features)
    """
    # Center the data (subtract mean)
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    return cov_matrix

def pca_from_scratch(X, n_components=None):
    """
    Perform Principal Component Analysis from scratch.
    
    Parameters:
    -----------
    X : numpy array
        Data matrix with shape (n_samples, n_features)
        Each row is a sample, each column is a feature
    n_components : int, optional
        Number of principal components to keep
        If None, keep all components
    
    Returns:
    --------
    components : numpy array
        Principal components (eigenvectors) with shape (n_features, n_components)
    explained_variance : numpy array
        Variance explained by each component
    transformed_data : numpy array
        Data projected onto the principal components with shape (n_samples, n_components)
    """
    # Convert to numpy array if not already
    X = np.array(X)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = compute_covariance_matrix(X)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigendecomposition(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Determine number of components to keep
    if n_components is None:
        n_components = X.shape[1]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    # Project data onto principal components
    transformed_data = X_centered @ components
    
    return components, explained_variance, transformed_data

def explained_variance_ratio(explained_variance):
    """
    Compute the explained variance ratio for each component.
    
    Parameters:
    -----------
    explained_variance : numpy array
        Variance explained by each component
    
    Returns:
    --------
    explained_variance_ratio : numpy array
        Ratio of variance explained by each component
    """
    total_variance = np.sum(explained_variance)
    return explained_variance / total_variance

def cumulative_explained_variance_ratio(explained_variance):
    """
    Compute the cumulative explained variance ratio.
    
    Parameters:
    -----------
    explained_variance : numpy array
        Variance explained by each component
    
    Returns:
    --------
    cumulative_explained_variance_ratio : numpy array
        Cumulative ratio of variance explained
    """
    ratios = explained_variance_ratio(explained_variance)
    return np.cumsum(ratios)

def reconstruct_from_pca(transformed_data, components, original_mean):
    """
    Reconstruct the original data from its PCA representation.
    
    Parameters:
    -----------
    transformed_data : numpy array
        Data projected onto the principal components
    components : numpy array
        Principal components (eigenvectors)
    original_mean : numpy array
        Mean of the original data
    
    Returns:
    --------
    reconstructed_data : numpy array
        Reconstructed data in the original space
    """
    reconstructed_data = transformed_data @ components.T
    reconstructed_data = reconstructed_data + original_mean
    
    return reconstructed_data

def choose_n_components(explained_variance, variance_threshold=0.95):
    """
    Choose the number of components needed to explain a certain amount of variance.
    
    Parameters:
    -----------
    explained_variance : numpy array
        Variance explained by each component
    variance_threshold : float, optional
        Threshold for cumulative explained variance ratio
    
    Returns:
    --------
    n_components : int
        Number of components needed to explain the specified variance
    """
    cumulative_ratios = cumulative_explained_variance_ratio(explained_variance)
    n_components = np.argmax(cumulative_ratios >= variance_threshold) + 1
    
    return n_components
