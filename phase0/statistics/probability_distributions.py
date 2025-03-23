# /learn-ai-ml-dl/phase0/statistics/probability_distributions.py

"""
Implementation of probability distributions from scratch.
"""

import math
import random

class Distribution:
    """Base class for probability distributions."""
    
    def sample(self):
        """Generate a random sample from the distribution."""
        raise NotImplementedError("Subclasses must implement sample method.")
    
    def pdf(self, x):
        """Probability density function or probability mass function at x."""
        raise NotImplementedError("Subclasses must implement pdf method.")
    
    def cdf(self, x):
        """Cumulative distribution function at x."""
        raise NotImplementedError("Subclasses must implement cdf method.")
    
    def mean(self):
        """Expected value of the distribution."""
        raise NotImplementedError("Subclasses must implement mean method.")
    
    def variance(self):
        """Variance of the distribution."""
        raise NotImplementedError("Subclasses must implement variance method.")
    
    def std_dev(self):
        """Standard deviation of the distribution."""
        return math.sqrt(self.variance())


class UniformDistribution(Distribution):
    """Uniform distribution on the interval [a, b]."""
    
    def __init__(self, a=0, b=1):
        """
        Create a uniform distribution on [a, b].
        
        Parameters:
        -----------
        a : float
            Lower bound of the interval
        b : float
            Upper bound of the interval
        """
        if a >= b:
            raise ValueError("Lower bound must be less than upper bound")
        
        self.a = a
        self.b = b
    
    def sample(self):
        """Generate a random sample from the uniform distribution."""
        return random.uniform(self.a, self.b)
    
    def pdf(self, x):
        """Uniform probability density function at x."""
        if self.a <= x <= self.b:
            return 1 / (self.b - self.a)
        else:
            return 0
    
    def cdf(self, x):
        """Uniform cumulative distribution function at x."""
        if x < self.a:
            return 0
        elif x > self.b:
            return 1
        else:
            return (x - self.a) / (self.b - self.a)
    
    def mean(self):
        """Expected value of the uniform distribution."""
        return (self.a + self.b) / 2
    
    def variance(self):
        """Variance of the uniform distribution."""
        return (self.b - self.a)**2 / 12


# /learn-ai-ml-dl/phase0/statistics/src/probability_distributions.py
# (Add this to the existing file)

class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution."""
    
    def __init__(self, mu=0, sigma=1):
        """
        Create a normal distribution with mean mu and standard deviation sigma.
        
        Parameters:
        -----------
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution
        """
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")
        
        self.mu = mu
        self.sigma = sigma
    
    def sample(self):
        """
        Generate a random sample from the normal distribution.
        Uses the Box-Muller transform.
        """
        # Box-Muller transform
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        
        # Transform to desired mean and standard deviation
        return self.mu + self.sigma * z
    
    def pdf(self, x):
        """Normal probability density function at x."""
        return (1 / (self.sigma * math.sqrt(2 * math.pi))) * \
               math.exp(-(x - self.mu)**2 / (2 * self.sigma**2))
    
    def cdf(self, x):
        """
        Normal cumulative distribution function at x.
        Uses an approximation of the error function.
        """
        # Standardize x
        z = (x - self.mu) / (self.sigma * math.sqrt(2))
        
        # Approximation of the error function
        t = 1.0 / (1.0 + 0.5 * abs(z))
        erf = 1 - t * math.exp(-z*z - 1.26551223 + t * (1.00002368 + t * \
              (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * \
              (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * \
              (-0.82215223 + t * 0.17087277)))))))))
        
        if z >= 0:
            return 0.5 + 0.5 * erf
        else:
            return 0.5 - 0.5 * erf
    
    def mean(self):
        """Expected value of the normal distribution."""
        return self.mu
    
    def variance(self):
        """Variance of the normal distribution."""
        return self.sigma**2