# /learn-ai-ml-dl/phase0/statistics/bayesian_inference.py

"""
Bayesian inference implementation for machine learning applications.
"""

import math
from collections import defaultdict

class BayesianModel:
    """Implementation of a basic Bayesian model."""
    
    def __init__(self):
        """Initialize the Bayesian model with empty priors."""
        self.prior = {}
        self.likelihood = {}
        self.posterior = {}
    
    def set_prior(self, prior):
        """
        Set the prior probabilities.
        
        Parameters:
        -----------
        prior : dict
            Dictionary mapping each hypothesis to its prior probability
        """
        # Normalize prior probabilities to ensure they sum to 1
        total = sum(prior.values())
        self.prior = {h: p/total for h, p in prior.items()}
    
    def set_likelihood(self, likelihood):
        """
        Set the likelihood function.
        
        Parameters:
        -----------
        likelihood : dict
            Nested dictionary mapping hypotheses to data values to probabilities
            likelihood[hypothesis][data] = P(data|hypothesis)
        """
        self.likelihood = likelihood
    
    def update(self, data):
        """
        Update the model with new data using Bayes' rule.
        
        Parameters:
        -----------
        data : any
            The observed data
        
        Returns:
        --------
        posterior : dict
            The updated posterior probabilities
        """
        # Calculate the posterior for each hypothesis
        unnormalized_posterior = {}
        
        for hypothesis in self.prior:
            if hypothesis in self.likelihood and data in self.likelihood[hypothesis]:
                # P(H|D) ∝ P(D|H) * P(H)
                unnormalized_posterior[hypothesis] = (
                    self.likelihood[hypothesis][data] * self.prior[hypothesis]
                )
            else:
                unnormalized_posterior[hypothesis] = 0
        
        # Normalize the posterior
        total = sum(unnormalized_posterior.values())
        if total > 0:
            self.posterior = {h: p/total for h, p in unnormalized_posterior.items()}
        else:
            # If all probabilities are zero, set equal probabilities
            self.posterior = {h: 1/len(self.prior) for h in self.prior}
        
        # Update prior for next iteration
        self.prior = self.posterior.copy()
        
        return self.posterior
    
    def predict(self, possible_data):
        """
        Make predictions about future data.
        
        Parameters:
        -----------
        possible_data : list
            List of possible data values
            
        Returns:
        --------
        predictions : dict
            Dictionary mapping possible data values to their probabilities
        """
        predictions = {}
        
        for data in possible_data:
            # Sum over all hypotheses: P(D) = ∑ P(D|H) * P(H)
            prob = 0
            for hypothesis in self.prior:
                if hypothesis in self.likelihood and data in self.likelihood[hypothesis]:
                    prob += self.likelihood[hypothesis][data] * self.prior[hypothesis]
            
            predictions[data] = prob
        
        return predictions