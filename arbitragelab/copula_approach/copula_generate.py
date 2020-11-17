# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Module that houses all copula classes and the parent copula class. Also
include a Switcher class to create copula by its name and parameters, to
emulate a switch functionality.
"""
import numpy as np
from scipy.optimize import brentq as brentq
from scipy.special import gamma as gm
from scipy.stats import t as student_t
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.integrate import quad


class Copula:
    """Copula class houses common functions for each coplula subtype."""
    def __init__(self):
        pass
        
    def describe(self):
        """
        Describe the copula's name and parameter.
        """
        pass
    
    def sample_plot(self):
        """
        Quick plotting from sampling points based on copula's P.D.F.
        """
        pass

class Gumbel(Copula):
    """Gumbel Copula"""
    def __init__(self, threshold=1e-10, theta=None):
        super().__init__()
        # Lower than this amount will be considered 0.
        self.threshold = threshold
        self.theta = theta  # Gumbel copula parameter.

    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        :param num: (int) Number of points to generate.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data. 
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        # Distribution of C(U1, U2). To be used for numerically solving the inverse.
        def _Kc(w):
            return w * (1 - np.log(w) / theta)

        # Generate pairs of indep uniform dist vectors.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Gumbel copulas from the independent uniform pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from Gumbel copula.

        :param v1: (float) i.i.d. uniform random variable in [0,1].
        :param v2: (float) i.i.d. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) conditional probability function, for numerical inverse.
        """
        # Numerically root finding for w1, where Kc(w1) = v2. 
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2, self.threshold, 1)
        else:
            w = 0  # Below the threshold, gives 0 as the root
        u1 = np.exp(v1 ** (1 / theta) * np.log(w))
        u2 = np.exp((1 - v1) ** (1 / theta) * np.log(w))

        return u1, u2

    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        theta = self.theta
        # Preparameters parameters.
        u_part = (-np.log(u)) ** theta
        v_part = (-np.log(v)) ** theta
        expo = (u_part + v_part) ** (1 / theta)
        
        # Assembling for P.D.F.
        pdf = 1 / (u * v) \
            * ( np.exp(-expo)
               * u_part / (-np.log(u)) * v_part / (-np.log(v))
               * (theta + expo - 1)
               * (u_part + v_part) ** (1 / theta - 2))
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        theta = self.theta
        # Preparameters parameters.
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta)
        
        # Assembling for P.D.F.
        cdf = np.exp(-expo)
        return cdf

    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        theta = self.theta
        expo = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** ((1 - theta) / theta)
        result = self._C(u, v) * expo * (-np.log(v)) ** (theta - 1) / v
        return result

    def _theta_hat(self, tau):
        """Calculate theta hat from Kendall's tau from sample data"""
        return 1 / (1 - tau)


class Frank(Copula):
    """Frank Copula"""
    def __init__(self, threshold=1e-10, theta=None):
        super().__init__()
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input

    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1]
        
        :param num: (int) Number of points to generate.
        :param theta: (float) All reals except for 0, measurement of correlation.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data. 
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Frank copulas from the unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta)

        return sample_pairs

    def _generate_one_pair(self, u1, v2, theta):
        """
        Helper func to generate one pair of vectors from Frank copula.

        :param v1: (float) i.i.d. uniform random variable in [0,1].
        :param v2: (float) i.i.d. uniform random variable in [0,1].
        """
        u2 = -1 / theta * np.log(1 + (v2 * (1 - np.exp(-theta))) /
                                 (v2 * (np.exp(-theta * u1) - 1)
                                  - np.exp(-theta * u1)))

        return u1, u2

    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        theta = self.theta
        et = np.exp(theta)
        eut = np.exp(u * theta)
        evt = np.exp(v * theta)
        pdf = (et * eut * evt * (et - 1) * theta /
                (et + eut * evt - et * eut - et * evt)**2)
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        theta = self.theta
        cdf = -1 / theta * np.log(
            1 + (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
            / (np.exp(-theta) - 1))
        return cdf

    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v). 
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        theta = self.theta
        enut = np.exp(-u * theta)
        envt = np.exp(-v * theta)
        ent = np.exp(-theta)
        result = (envt * (enut - 1)
                  / ((ent - 1) + (enut - 1) * (envt - 1)))
        return result

    def _theta_hat(self, tau):
        """Calculate theta hat from Kendall's tau from sample data"""
        def debye1(theta):
            """Debye function D_1(theta)"""
            result = quad(lambda x: x / theta / (np.exp(x) - 1), 0, theta)
            return result[0]

        def kendall_tau(theta):
            return 1 - 4 / theta + 4 * debye1(theta) / theta
        
        # Numerically find the root.
        result = brentq(lambda theta: kendall_tau(theta) - tau, -100, 100)
        return result

class Clayton(Copula):
    """Clayton copula"""
    def __init__(self, threshold=1e-10, theta=None):
        super().__init__()
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input

    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        Note: Large theta might suffer from accuracy issues.

        :param num: (int) Number of points to generate.
        :param theta: (float) Range in [-1, +inf) \ {0}., measurement of correlation.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data. 
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Frank copulas from the unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta)

        return sample_pairs

    def _generate_one_pair(self, u1, v2, theta):
        """
        Helper func to generate one pair of vectors from Clayton copula.

        :param v1: (float) i.i.d. uniform random variable in [0,1].
        :param v2: (float) i.i.d. uniform random variable in [0,1].
        """
        u2 = np.power(u1 ** (-theta) * (v2 ** (-theta / (1 + theta)) - 1) + 1,
                      -1 / theta)

        return u1, u2

    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_part = u ** (-1 - theta)
        v_part = v ** (-1 - theta)
        pdf = ((1 + theta) * u_part * v_part
               * (-1 + u_part * u + v_part * v) ** (-2 - 1 / theta))
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        theta = self.theta
        cdf = np.max(u ** (-theta) + v ** (-theta) - 1,
                     0) ** (-1 / theta)
        return cdf

    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        theta = self.theta
        unt = u ** (-theta)
        vnt = v ** (-theta)
        t_power = 1 / theta + 1
        result = vnt / v / np.power(unt + vnt - 1, t_power)

        return result

    def _theta_hat(self, tau):
        """Calculate theta hat from Kendall's tau from sample data"""
        return 2 * tau / (1 - tau)


class Joe(Copula):
    """Joe Copula"""
    def __init__(self, theta=None, threshold=1e-10, ):
        if theta < 1:
            raise ValueError("theta should be in [1, +inf).")
        self.theta = theta  # Default input
        # Lower than this amount will be considered 0
        self.threshold = threshold
        super().__init__()
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        :param num: (int) Number of points to generate.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data. 
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        def _Kc(w):
            return w - 1 / theta * (
                    (np.log(1 - (1 - w)**theta)) * (1 - (1 - w)**theta)
                    / ((1 - w)**(theta - 1)))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Joe copulas from the unif i.i.d. pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from Joe copula.

        :param v1: (float) i.i.d. uniform random variable in [0,1].
        :param v2: (float) i.i.d. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) conditional probability function, for numerical inverse.
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = 0  # Below the threshold, gives 0 as the root
        u1 = 1 - (1 - (1 - (1 - w)**theta)**v1)**(1 / theta)

        u2 = 1 - (1 - (1 - (1 - w)**theta)**(1 - v1))**(1 / theta)

        return u1, u2

    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_part = (1 - u)**theta
        v_part = (1 - v)**theta
        pdf = (u_part / (1 - u) * v_part / (1 - v)
                * (u_part + v_part - u_part * v_part)**(1 / theta - 2)
                * (theta - (u_part - 1) * (v_part - 1)))
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_part = (1 - u) ** theta
        v_part = (1 - v) ** theta
        cdf = 1 - ((u_part + v_part - u_part * v_part)
                   ** (1 / theta))
        return cdf
    
    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        theta = self.theta
        u_part = (1 - u) ** theta
        v_part = (1 - v) ** theta
        result = -(-1 + u_part) * (u_part + v_part - u_part * v_part)**(-1 + 1/theta) \
            * v_part / (1 - v)
        
        return result


class N13(Copula):
    """N13 Copula (Nelsen 13)"""
    def __init__(self, threshold=1e-10, theta=None):
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input
        super().__init__()

    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        :param num: (int) Number of points to generate.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data. 
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        def _Kc(w):
            return w + 1 / theta * (
                    w - w * np.power((1 - np.log(w)), 1 - theta) - w * np.log(w))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute N13 copulas from the i.i.d. unif pairs
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from N13 copula.

        :param v1: (float) i.i.d. uniform random variable in [0,1].
        :param v2: (float) i.i.d. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) conditional probability function, for numerical inverse.
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = 0  # Below the threshold, gives 0 as the root.
        u1 = np.exp(
            1 - (v1 * ((1 - np.log(w))**theta - 1) + 1)**(1 / theta))

        u2 = np.exp(
            1 - ((1 - v1) * ((1 - np.log(w))**theta - 1) + 1)**(1 / theta))

        return u1, u2
    
    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_part = (1 - np.log(u))**theta
        v_part = (1 - np.log(v))**theta
        Cuv = self._C(u, v)
        
        numerator = (Cuv * u_part * v_part
                     * (-1 + theta + (-1 + u_part + v_part)**(1/theta))
                     * (-1 + u_part + v_part)**(1/theta) )
        
        denominator = u * v * (1 - np.log(u)) * (1 - np.log(v)) * (-1 + u_part + v_part)**2
        
        pdf = numerator / denominator
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_part = (1 - np.log(u))**theta
        v_part = (1 - np.log(v))**theta
        cdf = np.exp(
            1 - (-1 + u_part + v_part)**(1/theta))

        return cdf
    
    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        theta = self.theta
        u_part = (1 - np.log(u))**theta
        v_part = (1 - np.log(v))**theta
        Cuv = self._C(u, v)

        numerator = Cuv * (-1 + u_part + v_part)**(1/theta) * v_part
        denominator = v * (-1 + u_part + v_part) * (1 - np.log(v))

        result = numerator / denominator
        
        return result


class N14:
    """N14 Copula (Nelsen 14)."""
    
    def __init__(self, threshold=1e-10, theta=None):
        # Lower than this amount will be considered 0.
        self.threshold = threshold
        self.theta = theta  # Default input.
        super().__init__()

    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        :param num: (int) Number of points to generate.
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param unif_vec: (np.array) Shape=(num, 2) array, two independent uniformly distributed sets of data. 
            Default uses numpy pseudo-random generators.
        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")

        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input.

        def _Kc(w):
            return -w * (-2 + w**(1/theta))

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))

        # Compute Gumbel copulas from the unif pairs.
        sample_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            sample_pairs[row] = self._generate_one_pair(pair[0],
                                                        pair[1],
                                                        theta=theta,
                                                        Kc=_Kc)

        return sample_pairs

    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from N14 copula.

        :param v1: (float) i.i.d. uniform random variable in [0,1].
        :param v2: (float) i.i.d. uniform random variable in [0,1].
        :param theta: (float) Range in [1, +inf), measurement of correlation.
        :param Kc: (func) conditional probability function, for numerical inverse.
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2,
                       self.threshold, 1 - self.threshold)
        else:
            w = 0  # Below the threshold, gives 0 as the root.
        u1 = (1 + (v1 * (w**(-1 / theta) - 1)**theta) ** (1 / theta))**(-theta)
        u2 = (1 + ((1 - v1) * (w**(-1 / theta) - 1)**theta)**(1 / theta))**(-theta)

        return u1, u2
    
    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_ker = -1 + np.power(u, 1/theta)
        v_ker = -1 + np.power(v, 1/theta)
        u_part = (-1 + np.power(u, -1/theta))**theta
        v_part = (-1 + np.power(v, -1/theta))**theta
        cdf_ker =  1 + (u_part + v_part)**(1/theta)
        
        numerator = (u_part * v_part * (cdf_ker -1)
                     * (-1 + theta + 2 * theta * (cdf_ker -1)))
        
        denominator = ((u_part + v_part)**2 * cdf_ker**(2 + theta)
                       * u * v * u_ker * v_ker * theta)
        
        pdf = numerator / denominator
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        theta = self.theta
        u_part = (-1 + np.power(u, -1/theta))**theta
        v_part = (-1 + np.power(v, -1/theta))**theta
        cdf = (1 + (u_part + v_part)**(1/theta))**(-theta)

        return cdf
    
    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        theta = self.theta
        u_ker = -1 + np.power(u, -1/theta)
        v_ker = -1 + np.power(v, -1/theta)
        u_part = (-1 + np.power(u, -1/theta))**theta
        v_part = (-1 + np.power(v, -1/theta))**theta
        cdf_ker =  1 + (u_part + v_part)**(1/theta)

        numerator = v_part * (cdf_ker - 1)
        denominator = v**(1 + 1/theta) * v_ker * (u_part + v_part) * cdf_ker**(1 + theta)

        result = numerator / denominator
        
        return result


class Gaussian(Copula):
    """Bivariate Gaussian Copula"""
    def __init__(self, cov=None):
        self.cov = cov  # Covariance matrix
        # Correlation
        self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))
        super().__init__()

    def generate_pairs(self, num=None, cov=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        :param num: (int) Number of points to generate.
        :param cov: (np.array) 2 by 2 covariance matrix.

        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if cov is None:
            cov = self.cov

        gaussian_pairs = self._generate_corr_gaussian(num, cov)
        sample_pairs = norm.cdf(gaussian_pairs)

        return sample_pairs

    def _generate_corr_gaussian(self, num, cov):
        """
        Helper func to sample from a bivariate Gaussian dist.

        :param num: (int) Number of samples.
        :param cov: (np.array) Covariance matrix.
        """
        # Generate bivariate normal with mean 0 and intended covariance.
        rand_generator = np.random.default_rng()
        result = rand_generator.multivariate_normal(mean=[0, 0],
                                                    cov=cov,
                                                    size=num)

        return result

    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        rho = self.rho
        inv_cdf_u = norm.ppf(u)
        inv_cdf_v = norm.ppf(v)
        
        pdf = np.exp(- inv_cdf_u * inv_cdf_v * rho) / np.sqrt(1 - rho**2)
        return pdf

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        corr = [[1, self.rho], [self.rho, 1]]  # Correlation matrix.
        inv_cdf_u = norm.ppf(u)  # Inverse cdf of standard normal.
        inv_cdf_v = norm.ppf(v)
        mvn_dist = mvn(mean=[0, 0], cov=corr)  # Joint cdf of multivar normal.
        cdf = mvn_dist.cdf((inv_cdf_u, inv_cdf_v))
        return cdf

    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        rho = self.rho
        inv_cdf_u = norm.ppf(u)
        inv_cdf_v = norm.ppf(v)
        sqrt_det_corr = np.sqrt(1 - rho * rho)
        result = norm.cdf((inv_cdf_u - rho * inv_cdf_v)
                          / sqrt_det_corr)

        return result

    def _theta_hat(self, tau):
        """Calculate theta hat from Kendall's tau from sample data"""
        return np.sin(tau * np.pi / 2)


class Student:
    """Bivariate Student-t Copula, need degree of freedom nu."""
    def __init__(self, nu=None, cov=None):
        self.cov = cov  # Covariance matrix.
        self.nu = nu  # Degree of freedom.
        # Correlation from covariance matrix.
        self.rho = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

    def generate_pairs(self, num=None, nu=None, cov=None):
        """
        Generate pairs according to P.D.F., stored in a 2D np.array.
        
        User may choose to side load independent uniformly distributed data in [0, 1].
        
        :param num: (int) Number of points to generate.
        :param nu: (float) Degree of freedom.
        :param cov: (np.array) 2 by 2 covariance matrix.

        :return sample_pairs: (np.array) Shape=(num, 2) array, sampled data for this copula.
        """
        if cov is None:
            cov = self.cov
        if nu is None:
            nu = self.nu

        student_pairs = self._generate_corr_student(num, cov, nu)
        t_dist = student_t(df=nu)
        copula_pairs = t_dist.cdf(student_pairs)

        return copula_pairs

    def _generate_corr_student(self, num, cov, nu):
        """
        Helper func to sample from a bivariate Student-t dist.

        :param num: (int) Number of samples.
        :param cov: (np.array) Covariance matrix.
        :param nu: (float) Degree of freedom.
        """
        # Sample from bivariate Normal with cov=cov.
        rand_generator = np.random.default_rng()
        normal = rand_generator.multivariate_normal(mean=[0, 0], cov=cov, size=num)
        # Sample from Chi-square with df=nu.
        chisq = rand_generator.chisquare(df=nu, size=num)
        result = np.zeros((num, 2))
        for row_idx, row in enumerate(result):
            row[0] = normal[row_idx][0] / np.sqrt(chisq[row_idx] / nu)
            row[1] = normal[row_idx][1] / np.sqrt(chisq[row_idx] / nu)

        return result

    def _c(self, u, v):
        """
        Calculate probability density of the biavriate copula: P(U=u, V=v).
        
        Result is analytical.
        """
        # rho = self.rho
        nu = self.nu
        rho = self.rho
        corr = [[1, rho],
                [rho, 1]]
        t_dist = student_t(df=nu)
        y1 = t_dist.ppf(u)
        y2 = t_dist.ppf(v)

        numerator = self._bv_t_dist(x=(y1, y2),
                               mu=(0, 0),
                               cov=corr,
                               df=nu)
        denominator = t_dist.pdf(y1) * t_dist.pdf(y2)

        pdf = numerator / denominator

        return pdf

    def condi_cdf(self, u, v):
        """
        Calculate conditional cumulative density function: P(U<=u | V=v).
        
        Result is analytical.
        
        Note: This probability is symmmetric about (u, v).
        """
        rho = self.rho
        nu = self.nu
        t_dist = student_t(nu)
        t_dist_nup1 = student_t(nu + 1)
        inv_cdf_u = t_dist.ppf(u)
        inv_cdf_v = t_dist.ppf(v)
        numerator = (inv_cdf_u - rho * inv_cdf_v) * np.sqrt(nu + 1)
        denominator = np.sqrt((1 - rho**2) * (inv_cdf_v**2 + nu))

        result = t_dist_nup1.cdf(numerator / denominator)

        return result

    def _C(self, u, v):
        """
        Calculate cumulative density of the bivariate copula: P(U<=u, V<=v).
        
        Result is analytical.
        """
        corr = [[1, self.rho], [self.rho, 1]]  # correlation mtx
        t_dist = student_t(df=self.nu)
        var1 = t_dist.ppf(u)  # Inverse cdf of standard normal
        var2 = t_dist.ppf(v)
        mvt_dist = mvn(mean=[0, 0], cov=corr)  # Joint cdf of multivar normal
        cdf = mvt_dist.cdf((var1, var2))
        return cdf

    def _theta_hat(self, tau):
        """Calculate theta hat from Kendall's tau from sample data"""
        return np.sin(tau * np.pi / 2)
    
    @staticmethod
    def _bv_t_dist(x, mu, cov, df):
        """
        Bivariate Student-t probability density.
    
        :param x: (list_like) a pair of values, shape=(2, ).
        :param mu: (list_like) mean for the distribution, shape=(2, ).
        :param cov: (list_like) covariance matrix, shape=(2, 2).
        :param df: (float) degree of freedom.
        :return: (float) the probability density.
        """
        x1 = x[0] - mu[0]
        x2 = x[1] - mu[1]
        c11 = cov[0][0]
        c12 = cov[0][1]
        c21 = cov[1][0]
        c22 = cov[1][1]
        det_cov = c11 * c22 - c12 * c21
        # Pseudo code: (x.transpose)(cov.inverse)(x)/ Det(cov)
        xT_covinv_x = (-2 * c12 * x1 * x2 + c11 * (x1 ** 2 + x2 ** 2)) / det_cov
    
        numerator = gm((2 + df) / 2)
        denominator = (gm(df / 2) * df * np.pi * np.sqrt(det_cov)
                       * np.power(1 + xT_covinv_x / df, (2 + df) / 2))
    
        result = numerator / denominator
        return result


class Switcher:
    """
    Switch class to emulate switch functionality.
    
    Create copula by its string name.
    """
    def __init__(self):
        self.theta = None
        self.cov = None
        self.nu = None

    def choose_copula(self, **kwargs):
        """
        Choose a method to instanticate a copula.
        
        User need to input copula's name and necessary parameters as kwargs.

        :param kwargs: (dict) Key word arguments to generate a copula by its name.
            copula_name: (str) Name of the copula.
            theta: (float) A measurement of correlation.
            cov: (np.array) Covariance matrix, only useful for Gaussian and Student-t.
            nu: (float) Degree of freedom, only useful for Student-t.
        :return method: (func) The method that creates the wanted copula. Eventually the returned object
            is a copula.
        """
        copula_name = kwargs.get('copula_name')
        self.theta = kwargs.get('theta', None)
        self.cov = kwargs.get('cov', None)
        self.nu = kwargs.get('nu', None)
        method_name = '_create_' + str(copula_name).lower()
        method = getattr(self, method_name,
                         lambda: print("Invalid copula name"))
        return method()

    def _create_gumbel(self):
        my_copula = Gumbel(theta=self.theta)
        return my_copula

    def _create_frank(self):
        my_copula = Frank(theta=self.theta)
        return my_copula

    def _create_clayton(self):
        my_copula = Clayton(theta=self.theta)
        return my_copula

    def _create_joe(self):
        my_copula = Joe(theta=self.theta)
        return my_copula

    def _create_n13(self):
        my_copula = N13(theta=self.theta)
        return my_copula

    def _create_n14(self):
        my_copula = N14(theta=self.theta)
        return my_copula

    def _create_gaussian(self):
        my_copula = Gaussian(cov=self.cov)
        return my_copula

    def _create_student(self):
        my_copula = Student(nu=self.nu, cov=self.cov)
        return my_copula
