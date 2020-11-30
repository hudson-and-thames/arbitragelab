import numpy as np

class FeatureExpander:
    
    def __init__(self, methods=[], n_orders=1):
        """
        
        :param methods: (list) Possible expansion methods [chebyshev, legendre, laguerre, power].
        :param n_orders: (int) Number of orders.
        """
        self.methods = methods
        self.n_orders = n_orders
        
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
        :param X: (np.array) dataset
        """
        self.dataset = X
        return self

    def transform(self) -> list:
        """
        Transform data to polynomial features
        
        :return: List of lists of the expanded values.
        """
        new_dataset = []
        
        for x in self.dataset:
            expanded_x = list(x)
            for deg in range(1, self.n_orders):
                for meth in self.methods:
                    expanded_x.extend( np.ravel( getattr(self, '_' + meth)(x, deg) ) )
                
            new_dataset.append( np.ravel(expanded_x).tolist() )
            
        return new_dataset