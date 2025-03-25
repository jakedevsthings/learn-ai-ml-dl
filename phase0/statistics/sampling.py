# /learn-ai-ml-dl/phase0/statistics/sampling.py

"""
Implementation of statistical sampling methods.
"""

import math
import random
import numpy as np
from typing import List, Callable, Tuple, Optional

class SamplingMethod:
    """Base class for sampling methods."""
    
    def sample(self, size: int = 1) -> List[float]:
        """Generate samples using the specified method."""
        raise NotImplementedError("Subclasses must implement sample method.")
    

class RejectionSampling(SamplingMethod):
    """
    Rejection sampling implementation.
    
    This method generates samples from a target distribution by using a proposal 
    distribution and accepting/rejecting samples based on their likelihood ratio.
    """
    
    def __init__(self, 
                 target_pdf: Callable[[float], float], 
                 proposal_pdf: Callable[[float], float],
                 proposal_sampler: Callable[[], float],
                 M: float):
        """
        Initialize rejection sampling.
        
        Parameters:
        -----------
        target_pdf : callable
            The target probability density function f(x)
        proposal_pdf : callable
            The proposal probability density function g(x)
        proposal_sampler : callable
            Function that generates samples from the proposal distribution
        M : float
            A constant such that f(x) ≤ M * g(x) for all x
        """
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        self.M = M
        
    def sample(self, size: int = 1) -> List[float]:
        """
        Generate samples from the target distribution.
        
        Parameters:
        -----------
        size : int
            Number of samples to generate
            
        Returns:
        --------
        List[float]
            Samples from the target distribution
        """
        samples = []
        attempts = 0
        max_attempts = size * 1000  # Prevent infinite loops
        
        while len(samples) < size and attempts < max_attempts:
            attempts += 1
            
            # Sample from proposal distribution
            x = self.proposal_sampler()
            
            # Calculate acceptance probability
            acceptance_ratio = self.target_pdf(x) / (self.M * self.proposal_pdf(x))
            
            # Accept or reject
            if random.random() < acceptance_ratio:
                samples.append(x)
                
        if len(samples) < size:
            print(f"Warning: Could only generate {len(samples)} samples after {attempts} attempts.")
            
        return samples


class ImportanceSampling:
    """
    Importance sampling implementation.
    
    This method is used to estimate properties of a target distribution by sampling
    from a different proposal distribution.
    """
    
    def __init__(self, 
                 target_pdf: Callable[[float], float], 
                 proposal_pdf: Callable[[float], float],
                 proposal_sampler: Callable[[], float]):
        """
        Initialize importance sampling.
        
        Parameters:
        -----------
        target_pdf : callable
            The target probability density function f(x)
        proposal_pdf : callable
            The proposal probability density function g(x)
        proposal_sampler : callable
            Function that generates samples from the proposal distribution
        """
        self.target_pdf = target_pdf
        self.proposal_pdf = proposal_pdf
        self.proposal_sampler = proposal_sampler
        
    def estimate_expectation(self, 
                             function: Callable[[float], float], 
                             n_samples: int = 1000) -> Tuple[float, float]:
        """
        Estimate the expectation of a function under the target distribution.
        
        Parameters:
        -----------
        function : callable
            The function h(x) whose expectation E[h(X)] we want to estimate
        n_samples : int
            Number of samples to use for the estimate
            
        Returns:
        --------
        Tuple[float, float]
            Estimated expectation and standard error
        """
        # Generate samples from proposal distribution
        samples = [self.proposal_sampler() for _ in range(n_samples)]
        
        # Calculate importance weights
        weights = [self.target_pdf(x) / self.proposal_pdf(x) for x in samples]
        
        # Calculate weighted function values
        weighted_values = [function(x) * w for x, w in zip(samples, weights)]
        
        # Estimate expectation
        expectation = sum(weighted_values) / sum(weights)
        
        # Estimate standard error
        if n_samples > 1:
            variance = sum([(w * function(x) - expectation * w)**2 
                           for x, w in zip(samples, weights)]) / sum(weights)**2
            std_error = math.sqrt(variance / (n_samples - 1))
        else:
            std_error = float('nan')
            
        return expectation, std_error


class MCMC(SamplingMethod):
    """
    Base class for Markov Chain Monte Carlo sampling methods.
    """
    
    def __init__(self, 
                 target_pdf: Callable[[float], float], 
                 initial_state: float = 0.0):
        """
        Initialize MCMC sampler.
        
        Parameters:
        -----------
        target_pdf : callable
            The target probability density function (up to a normalizing constant)
        initial_state : float
            Starting state for the Markov chain
        """
        self.target_pdf = target_pdf
        self.current_state = initial_state
        
    def propose(self) -> float:
        """Generate a proposed next state."""
        raise NotImplementedError("Subclasses must implement propose method.")
        
    def acceptance_probability(self, proposed_state: float) -> float:
        """Calculate probability of accepting the proposed state."""
        raise NotImplementedError("Subclasses must implement acceptance_probability method.")
    
    def step(self) -> float:
        """
        Perform one step of the MCMC algorithm.
        
        Returns:
        --------
        float
            The new state after this step
        """
        # Propose new state
        proposed_state = self.propose()
        
        # Calculate acceptance probability
        acceptance_prob = self.acceptance_probability(proposed_state)
        
        # Accept or reject the proposed state
        if random.random() < acceptance_prob:
            self.current_state = proposed_state
            
        return self.current_state
    
    def sample(self, size: int = 1, burn_in: int = 100, thin: int = 1) -> List[float]:
        """
        Generate samples using MCMC.
        
        Parameters:
        -----------
        size : int
            Number of samples to generate
        burn_in : int
            Number of initial samples to discard
        thin : int
            Keep every 'thin' samples after burn-in
            
        Returns:
        --------
        List[float]
            Samples from the target distribution
        """
        # Burn-in period
        for _ in range(burn_in):
            self.step()
            
        # Generate samples with thinning
        samples = []
        for i in range(size * thin):
            self.step()
            if i % thin == 0:
                samples.append(self.current_state)
                
        return samples


class MetropolisHastings(MCMC):
    """
    Metropolis-Hastings algorithm implementation.
    
    A general MCMC method for sampling from a probability distribution.
    """
    
    def __init__(self, 
                 target_pdf: Callable[[float], float], 
                 proposal_sampler: Callable[[float], float],
                 proposal_pdf: Optional[Callable[[float, float], float]] = None,
                 initial_state: float = 0.0):
        """
        Initialize Metropolis-Hastings sampler.
        
        Parameters:
        -----------
        target_pdf : callable
            The target probability density function (up to a normalizing constant)
        proposal_sampler : callable
            Function that generates proposed states given the current state
        proposal_pdf : callable, optional
            Probability density of proposal distribution q(x'|x)
            If None, assumes symmetric proposal distribution
        initial_state : float
            Starting state for the Markov chain
        """
        super().__init__(target_pdf, initial_state)
        self.proposal_sampler = proposal_sampler
        self.proposal_pdf = proposal_pdf
        
    def propose(self) -> float:
        """Generate a proposed next state."""
        return self.proposal_sampler(self.current_state)
        
    def acceptance_probability(self, proposed_state: float) -> float:
        """Calculate probability of accepting the proposed state."""
        # Compute ratio of target densities
        target_ratio = self.target_pdf(proposed_state) / self.target_pdf(self.current_state)
        
        # For symmetric proposal, proposal ratio is 1
        if self.proposal_pdf is None:
            proposal_ratio = 1.0
        else:
            # Compute ratio of proposal densities
            proposal_ratio = (self.proposal_pdf(self.current_state, proposed_state) / 
                             self.proposal_pdf(proposed_state, self.current_state))
            
        return min(1.0, target_ratio * proposal_ratio)


class GibbsSampling:
    """
    Gibbs sampling implementation for multivariate distributions.
    
    This method samples from a multivariate distribution by iteratively sampling from
    conditional distributions of each variable given the current values of the others.
    """
    
    def __init__(self, 
                 conditionals: List[Callable[[List[float]], float]], 
                 initial_state: List[float]):
        """
        Initialize Gibbs sampler.
        
        Parameters:
        -----------
        conditionals : List[callable]
            List of functions to sample from conditional distributions
            conditionals[i] generates a sample for the i-th dimension
        initial_state : List[float]
            Starting state for the Markov chain
        """
        self.conditionals = conditionals
        self.current_state = initial_state.copy()
        self.n_dims = len(initial_state)
        
    def step(self) -> List[float]:
        """
        Perform one step of the Gibbs sampler.
        
        Returns:
        --------
        List[float]
            The new state after this step
        """
        for i in range(self.n_dims):
            # Sample from conditional distribution for dimension i
            self.current_state[i] = self.conditionals[i](self.current_state)
            
        return self.current_state.copy()
    
    def sample(self, size: int = 1, burn_in: int = 100, thin: int = 1) -> List[List[float]]:
        """
        Generate samples using Gibbs sampling.
        
        Parameters:
        -----------
        size : int
            Number of samples to generate
        burn_in : int
            Number of initial samples to discard
        thin : int
            Keep every 'thin' samples after burn-in
            
        Returns:
        --------
        List[List[float]]
            Samples from the target distribution
        """
        # Burn-in period
        for _ in range(burn_in):
            self.step()
            
        # Generate samples with thinning
        samples = []
        for i in range(size * thin):
            self.step()
            if i % thin == 0:
                samples.append(self.current_state.copy())
                
        return samples