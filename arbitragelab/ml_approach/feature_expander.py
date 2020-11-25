import numpy as np


class FeatureExpander:
    
    def __init__(self, methods=[]):
        self.methods = methods
        
    def _chebyshev(self, x, degree):
        return np.polynomial.chebyshev.chebvander(x, degree)
    
    def _legendre(self, x, degree):
        return np.polynomial.legendre.legvander(x, degree)
    
    def _laguerre(self, x, degree):
        return np.polynomial.laguerre.lagvander(x, degree)
    
    def _power(self, x, degree):
        return np.polynomial.polynomial.polyvander(x, degree)
    
    def fit(self, X, y=None):
        """
        Compute number of output features.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        Returns
        -------
        self : instance
        """
        n_samples, n_features = self._validate_data(
            X, accept_sparse=True).shape
        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        self.n_input_features_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self

    def transform(self, X):
        """
        Transform data to polynomial features
        """