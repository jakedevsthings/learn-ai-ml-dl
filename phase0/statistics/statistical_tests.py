# /learn-ai-ml-dl/phase0/statistics/statistical_tests.py

"""
Implementation of statistical tests and confidence intervals.
"""

import math
import numpy as np
from scipy import stats
from typing import Tuple, Optional, List, Dict, Any, Union


def mean_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate the mean and confidence interval for a data sample.
    
    Parameters:
    -----------
    data : List[float]
        The data sample
    confidence : float
        Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
    --------
    Tuple[float, float, float]
        Mean, lower bound, and upper bound of the confidence interval
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_error = stats.sem(data)
    
    # Get the critical value from t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    # Calculate the margin of error
    margin_error = t_critical * std_error
    
    # Calculate confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    
    return mean, lower_bound, upper_bound


def t_test(sample1: List[float], sample2: List[float], equal_var: bool = True) -> Dict[str, Any]:
    """
    Perform a two-sample t-test.
    
    Parameters:
    -----------
    sample1, sample2 : List[float]
        The two samples to compare
    equal_var : bool
        Whether to assume equal variances (default: True)
        
    Returns:
    --------
    Dict
        Dictionary containing t-statistic, p-value, and interpretation
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    
    # Interpret the result
    alpha = 0.05  # Standard significance level
    interpretation = "Reject null hypothesis (means are different)" if p_value < alpha else "Fail to reject null hypothesis"
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "interpretation": interpretation,
        "significant": p_value < alpha
    }


def chi_square_test(observed: List[int], expected: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Perform a chi-square goodness-of-fit test.
    
    Parameters:
    -----------
    observed : List[int]
        Observed frequencies
    expected : List[float], optional
        Expected frequencies (if None, assumes uniform distribution)
        
    Returns:
    --------
    Dict
        Dictionary containing chi-square statistic, p-value, and interpretation
    """
    observed = np.array(observed)
    
    if expected is None:
        # Assume uniform distribution
        expected = np.ones_like(observed) * observed.sum() / len(observed)
    else:
        expected = np.array(expected)
    
    # Calculate chi-square statistic and p-value
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    # Interpret the result
    alpha = 0.05  # Standard significance level
    interpretation = "Reject null hypothesis (distributions differ)" if p_value < alpha else "Fail to reject null hypothesis"
    
    return {
        "chi2_statistic": chi2_stat,
        "p_value": p_value,
        "interpretation": interpretation,
        "significant": p_value < alpha
    }


def f_test(sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
    """
    Perform an F-test for equality of variances.
    
    Parameters:
    -----------
    sample1, sample2 : List[float]
        The two samples to compare
        
    Returns:
    --------
    Dict
        Dictionary containing F-statistic, p-value, and interpretation
    """
    # Convert to numpy arrays
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    # Calculate variances
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    # Ensure the larger variance is in the numerator
    if var1 < var2:
        var1, var2 = var2, var1
        sample1, sample2 = sample2, sample1
    
    # Calculate F-statistic
    f_stat = var1 / var2
    
    # Degrees of freedom
    dfn = len(sample1) - 1  # Numerator df
    dfd = len(sample2) - 1  # Denominator df
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * min(stats.f.cdf(f_stat, dfn, dfd), 1 - stats.f.cdf(f_stat, dfn, dfd))
    
    # Interpret the result
    alpha = 0.05  # Standard significance level
    interpretation = "Reject null hypothesis (variances are different)" if p_value < alpha else "Fail to reject null hypothesis"
    
    return {
        "f_statistic": f_stat,
        "numerator_df": dfn,
        "denominator_df": dfd,
        "p_value": p_value,
        "interpretation": interpretation,
        "significant": p_value < alpha
    }


def bootstrap_confidence_interval(
    data: List[float], 
    statistic_func: callable,
    confidence: float = 0.95,
    n_resamples: int = 10000
) -> Tuple[float, float, float]:
    """
    Calculate a bootstrap confidence interval for a statistic.
    
    Parameters:
    -----------
    data : List[float]
        The data sample
    statistic_func : callable
        Function to compute the statistic (e.g., np.mean, np.median)
    confidence : float
        Confidence level (default: 0.95 for 95% confidence)
    n_resamples : int
        Number of bootstrap resamples (default: 10000)
        
    Returns:
    --------
    Tuple[float, float, float]
        Statistic, lower bound, and upper bound of the bootstrap confidence interval
    """
    data = np.array(data)
    n = len(data)
    
    # Calculate the statistic on the original data
    original_stat = statistic_func(data)
    
    # Generate bootstrap resamples and calculate statistics
    bootstrap_stats = []
    for _ in range(n_resamples):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(resample))
    
    # Calculate percentile-based confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrap_stats, 100 * alpha)
    upper_bound = np.percentile(bootstrap_stats, 100 * (1 - alpha))
    
    return original_stat, lower_bound, upper_bound


def permutation_test(
    sample1: List[float], 
    sample2: List[float],
    statistic_func: callable = lambda x, y: np.mean(x) - np.mean(y),
    n_permutations: int = 10000
) -> Dict[str, Any]:
    """
    Perform a permutation test to compare two samples.
    
    Parameters:
    -----------
    sample1, sample2 : List[float]
        The two samples to compare
    statistic_func : callable
        Function to compute the test statistic (default: difference in means)
    n_permutations : int
        Number of permutations (default: 10000)
        
    Returns:
    --------
    Dict
        Dictionary containing observed statistic, p-value, and interpretation
    """
    # Convert to numpy arrays
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    # Combine samples
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    n2 = len(sample2)
    n = n1 + n2
    
    # Calculate observed statistic
    observed_stat = statistic_func(sample1, sample2)
    
    # Generate permutation distribution
    perm_stats = []
    for _ in range(n_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined)
        # Split into two groups
        perm_sample1 = combined[:n1]
        perm_sample2 = combined[n1:]
        # Calculate statistic
        perm_stat = statistic_func(perm_sample1, perm_sample2)
        perm_stats.append(perm_stat)
    
    # Calculate two-sided p-value
    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    
    # Interpret the result
    alpha = 0.05  # Standard significance level
    interpretation = "Reject null hypothesis (samples differ)" if p_value < alpha else "Fail to reject null hypothesis"
    
    return {
        "observed_statistic": observed_stat,
        "p_value": p_value,
        "interpretation": interpretation,
        "significant": p_value < alpha
    }