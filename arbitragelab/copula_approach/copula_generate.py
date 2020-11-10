# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:59:47 2020

@author: Hansen
"""
import numpy as np
from scipy.optimize import brentq as brentq
from scipy.special import erfinv as erfinv
from scipy.special import gamma as gm
from scipy.stats import t as student_t
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.integrate import quad
import matplotlib.pyplot as plt

def mv_t_dist(x, mu, Sigma, df):
        """
        Multivariate t-student density. Returns the density
        of the function at points specified by x.
    
        input:
            x = parameter (n-d numpy array; will be forced to 2d)
            mu = mean (d dimensional numpy array)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
    
        Edited from: http://stackoverflow.com/a/29804411/3521179
        """

        x = np.atleast_2d(x) # requires x as 2d
        nD = Sigma.shape[0] # dimensionality
    
        numerator = gm(1.0 * (nD + df) / 2.0)
    
        denominator = (
                gm(1.0 * df / 2.0) * 
                np.power(df * np.pi, 1.0 * nD / 2.0) *  
                np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
                np.power(
                    1.0 + (1.0 / df) *
                    np.diagonal(
                        np.dot( np.dot(x - mu, np.linalg.inv(Sigma)),
                               (x - mu).T)
                    ), 
                    1.0 * (nD + df) / 2.0
                    )
                )
        
        result = 1.0 * numerator / denominator
        return result[0]

class Gumbel:
    def __init__(self, threshold=1e-10, theta=None):
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        
        theta in [1, +inf)
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")
        
        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        def _Kc(w):
            return w*(1-np.log(w) / theta)
        # Generate pairs of indep uniform dist vectors.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        
        # Compute Gumbel copulas from the independent uniform pairs
        copula_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            copula_pairs[row] = self._generate_one_pair(pair[0],
                                                      pair[1],
                                                      theta=theta,
                                                      Kc=_Kc)

        return copula_pairs
    
    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from Gumbel copula
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2, self.threshold, 1)
        else:
            w = 0  # Below the threshold, gives 0 as the root
        u1 = np.exp(v1**(1/theta) * np.log(w))
        u2 = np.exp((1-v1)**(1/theta) * np.log(w))
        
        return u1, u2
    
    def _c(self, u, v):
        """
        P.D.F. of the biavriate copula
        """
        theta = self.theta
        u_part = (-np.log(u))**theta
        v_part = (-np.log(v))**theta
        expo = (u_part + v_part)**(1/theta)
        pdf = 1/(u*v)*(
            np.exp(-expo)
            * u_part/(-np.log(u)) * v_part/(-np.log(v))
            * (theta + expo - 1)
            * (u_part + v_part)**(1/theta-2)
            )
        return pdf
    
    def _C(self, u, v):
        """
        C.D.F. of the copula
        """
        theta = self.theta
        expo = ((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)
        cdf = np.exp(-expo)
        return cdf
    
    def _theta_hat(self, tau):
        return 1/(1-tau)
    
class Frank:
    def __init__(self, threshold=1e-10, theta=None):
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        
        theta in Reals\{0}.
        
        Large theta might have accuracy issues
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")
        
        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        
        # Compute Frank copulas from the unif pairs
        copula_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            copula_pairs[row] = self._generate_one_pair(pair[0],
                                                      pair[1],
                                                      theta=theta)

        return copula_pairs
    
    def _generate_one_pair(self, u1, v2, theta):
        """
        Helper func to generate one pair of vectors from Frank copula
        """
        u2 = -1/theta * np.log(1 + (v2*(1-np.exp(-theta)))/
                               (v2*(np.exp(-theta*u1)-1)
                                - np.exp(-theta*u1))
                               )
        
        return u1, u2
    
    def _c(self, u, v):
        """
        P.D.F. of the biavriate copula
        """
        theta = self.theta
        et = np.exp(theta)
        eut = np.exp(u*theta)
        evt = np.exp(v*theta)
        pdf = (
            et * eut * evt * (et - 1) * theta /
            (et + eut*evt - et*eut - et*evt)**2
            )
        return pdf
        
    def _C(self, u, v):
        """
        C.D.F. of the copula
        """
        theta = self.theta
        cdf = -1/theta*np.log(
            1 + (np.exp(-theta*u) - 1)*(np.exp(-theta*v) - 1)
            /(np.exp(-theta) - 1)
            )
        return cdf
    
    def _theta_hat(self, tau):
        """
        Estimate theta hat from Kendall's tau from sample data
        """
        def debye1(theta):
            """Debye function D_1(theta)"""
            result = quad(lambda x: x / theta / (np.exp(x)-1), 0, theta)
            return result[0]
        
        def kendall_tau(theta):
            return 1 - 4/theta + 4*debye1(theta)/theta
        
        result = brentq(lambda theta: kendall_tau(theta) - tau, -100, 100)
        return result
    
class Clayton:
    def __init__(self, threshold=1e-10, theta=None):

        if theta == 0 or theta < -1:
            raise ValueError('theta should be in [-1, +inf) \ {0}.')
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        
        theta in [-1, +inf) \ {0}.
        
        Large theta might have accuracy issues
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")
        
        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        
        # Compute Frank copulas from the unif pairs
        copula_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            copula_pairs[row] = self._generate_one_pair(pair[0],
                                                      pair[1],
                                                      theta=theta)

        return copula_pairs
    
    def _generate_one_pair(self, u1, v2, theta):
        """
        Helper func to generate one pair of vectors from Clayton copula
        """
        u2 = np.power(u1**(-theta) * (v2**(-theta/(1+theta)) - 1) + 1,
                      -1/theta)
        
        return u1, u2
    
    def _c(self, u, v):
        """
        P.D.F. of the biavriate copula
        """
        theta = self.theta
        u_part = u**(-1 - theta)
        v_part = v**(-1 - theta)
        pdf = (
            (1 + theta) * u_part * v_part
            * (-1 + u_part*u + v_part*v)**(-2 - 1/theta)
            )
        return pdf
        
    def _C(self, u, v):
        """
        C.D.F. of the copula
        """
        theta = self.theta
        cdf = np.max(u**(-theta) + v**(-theta) - 1,
                     0)**(-1/theta)
        return cdf
    
    def _theta_hat(self, tau):
        return 2*tau/(1-tau)
    
class Joe:
    def __init__(self, theta=None, threshold=1e-10,):
        if theta < 1:
            raise ValueError("theta should be in [1, +inf).")
        self.theta = theta  # Default input
        # Lower than this amount will be considered 0
        self.threshold = threshold
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        
        theta in [1, +inf)
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")
        
        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        def _Kc(w):
            return w - 1/theta *(
                (np.log(1 - (1-w)**theta)) * (1 - (1-w)**theta)
                / ((1-w)**(theta-1))
                )
        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        
        # Compute Gumbel copulas from the unif pairs
        copula_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            copula_pairs[row] = self._generate_one_pair(pair[0],
                                                      pair[1],
                                                      theta=theta,
                                                      Kc=_Kc)

        return copula_pairs
    
    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from Joe copula
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2,
                       self.threshold, 1-self.threshold)
        else:
            w = 0  # Below the threshold, gives 0 as the root
        u1 = 1 - (1 - (1 - (1-w)**theta)**v1)**(1/theta)
        
        u2 = 1 - (1 - (1 - (1-w)**theta)**(1-v1))**(1/theta)
        
        return u1, u2
    
    def _c(self, u, v):
        """
        P.D.F. of the biavriate copula
        """
        theta = self.theta
        u_part = (1 - u)**theta
        v_part = (1 - v)**theta
        pdf = (
        u_part/(1-u) * v_part/(1-v)
        * (u_part + v_part - u_part*v_part)**(1/theta-2)
        * (theta - (u_part - 1)*(v_part - 1))
        )
        return pdf
        
    def _C(self, u, v):
        """
        C.D.F. of the copula
        """
        theta = self.theta
        u_part = (1 - u)**theta
        v_part = (1 - v)**theta
        cdf = 1 - (
            u_part + v_part - u_part*v_part
            )**(1/theta)
        return cdf
    
class N13:
    def __init__(self, threshold=1e-10, theta=None):
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        
        theta in [1, +inf)
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")
        
        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        def _Kc(w):
            return w + 1/theta *(
                w - w*np.power((1 - np.log(w)), 1-theta) - w*np.log(w)
                )
        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        
        # Compute Gumbel copulas from the unif pairs
        copula_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            copula_pairs[row] = self._generate_one_pair(pair[0],
                                                      pair[1],
                                                      theta=theta,
                                                      Kc=_Kc)

        return copula_pairs
    
    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from Joe copula
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2,
                       self.threshold, 1-self.threshold)
        else:
            w = 0  # Below the threshold, gives 0 as the root
        u1 = np.exp(
            1 - (v1 * ((1-np.log(w))**theta - 1) + 1)**(1/theta)
            )

        u2 = np.exp(
            1 - ((1-v1) * ((1-np.log(w))**theta - 1) + 1)**(1/theta)
            )
        
        return u1, u2
    
class N14:
    def __init__(self, threshold=1e-10, theta=None):
        # Lower than this amount will be considered 0
        self.threshold = threshold
        self.theta = theta  # Default input
        
    def generate_pairs(self, num=None, theta=None, unif_vec=None):
        """
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        
        theta in [1, +inf)
        """
        if num is None and unif_vec is None:
            raise ValueError("Please either input num or unif_vec")
        
        if theta is None and self.theta is not None:
            theta = self.theta  # Use the default input

        def _Kc(w):
            return w**((theta-1)/theta)
        # Generate pairs of indep uniform dist vectors. Use numpy to generate.
        if unif_vec is None:
            unif_vec = np.random.uniform(low=0, high=1, size=(num, 2))
        
        # Compute Gumbel copulas from the unif pairs
        copula_pairs = np.zeros_like(unif_vec)
        for row, pair in enumerate(unif_vec):
            copula_pairs[row] = self._generate_one_pair(pair[0],
                                                      pair[1],
                                                      theta=theta,
                                                      Kc=_Kc)

        return copula_pairs
    
    def _generate_one_pair(self, v1, v2, theta, Kc):
        """
        Helper func to generate one pair of vectors from copula
        """
        if v2 > self.threshold:
            w = brentq(lambda w1: Kc(w1) - v2,
                       self.threshold, 1-self.threshold)
        else:
            w = 0  # Below the threshold, gives 0 as the root
        u1 = (
            1 + (v1 * (w**(1/theta) - 1)**theta)**(1/theta)
            )**theta

        u2 = (
            1 + ((1-v1) * (w**(1/theta) - 1)**theta)**(1/theta)
            )**theta
        
        return u1, u2
    
class Gaussian:
    def __init__(self, cov=None):
        self.cov = cov  # Covariance matrix
        self.rho = cov[0][1] / (np.sqrt(cov[0][0])*np.sqrt(cov[1][1]))

    def generate_pairs(self, num=None, cov=None):
        """        
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
        """
        if cov is None:
            cov = self.cov

        gaussian_pairs = self._generate_corr_gaussian(num, cov)
        copula_pairs = norm.cdf(gaussian_pairs)
        
        return copula_pairs

    def _generate_corr_gaussian(self, num, cov):
        """
        Helper func to generate correlated Gaussian pairs
        """
        rand_generator = np.random.default_rng()
        result = rand_generator.multivariate_normal(mean=[0,0],
                                                    cov=cov,
                                                    size=num)
        
        return result
    
    def _c(self, u, v):
        """
        P.D.F. of the biavriate copula
        """
        rho = self.rho
        a = np.sqrt(2) * erfinv(2*u - 1)
        b = np.sqrt(2) * erfinv(2*v - 1)
        pdf = np.exp(
            - ((a**2 + b**2) * rho**2 - 2*a*b*rho)
            / (2 * (1 - rho**2))
            ) / np.sqrt(1 - rho**2)
        return pdf
        
    def _C(self, u, v):
        """
        C.D.F. of the copula
        """
        corr = [[1, self.rho],[self.rho, 1]]  # correlation mtx
        var1 = norm.ppf(u)  # Inverse cdf of standard normal
        var2 = norm.ppf(v)
        mvn_dist = mvn(mean=[0,0], cov=corr)  # Joint cdf of multivar normal
        cdf = mvn_dist.cdf((var1, var2))
        return cdf
    
    def _theta_hat(self, tau):
        return np.sin(tau * np.pi / 2)

class Student:
    def __init__(self, nu=None, cov=None):
        self.cov = cov
        self.nu = nu
        self.rho = cov[0][1] / (np.sqrt(cov[0][0])*np.sqrt(cov[1][1]))
    
    def generate_pairs(self, num=None, nu=None, cov=None):
        """        
        Generate pairs stored in an 2D np array.
        
        Array dimension  = (num, 2)
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
        Generate correlated t pairs from multi-var Normal and Chi-squared
        """
        rand_generator = np.random.default_rng()
        normal = rand_generator.multivariate_normal(mean=[0,0], cov=cov, size=num)
        chisq = rand_generator.chisquare(df=nu)
        result = normal/np.sqrt(chisq/nu)
        
        return result
    
    def _c(self, u, v):
        """
        P.D.F. of the biavriate copula
        """
        # rho = self.rho
        nu = self.nu
        Sigma = np.array(self.cov)
        t_dist = student_t(df=nu)
        y1 = t_dist.ppf(u)
        y2 = t_dist.ppf(v)
        
        numerator = mv_t_dist(x=(y1, y2),
                              mu=(0,0),
                              Sigma=Sigma,
                              df=nu)
        denominator = t_dist.pdf(y1) * t_dist.pdf(y2)
        
        pdf = numerator / denominator
        
        # pdf = (
        #     gm((nu+2)/2)*gm(nu/2) / (gm((nu+1)/2)**2)
        #     * (1 + (y1**2 + y2**2 -2*y1*y2*rho)/(rho**2 - 1)/nu)**(-(nu+2)/2)
        #     / (((1 + y1**2/nu) * (1 + y2**2/nu))**(-(nu+1)/2))
        #     ) / np.sqrt(1 - rho**2)
        return pdf
        
    def _C(self, u, v):
        """
        C.D.F. of the copula
        """
        corr = [[1, self.rho],[self.rho, 1]]  # correlation mtx
        t_dist = student_t(df=self.nu)
        var1 = t_dist.ppf(u)  # Inverse cdf of standard normal
        var2 = t_dist.ppf(v)
        mvt_dist = mvn(mean=[0,0], cov=corr)  # Joint cdf of multivar normal
        cdf = mvt_dist.cdf((var1, var2))
        return cdf
    
    def _theta_hat(self, tau):
        return np.sin(tau * np.pi / 2)

class Switcher:
    def __init__(self):
        pass
    
    def choose_copula(self, **kwargs):
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
    
#%% test class for Gumbel
# theta = 2
# GC = Gumbel(theta=theta)
# result = GC.generate_pairs(num=1000)
# print(GC._c(0.5, 0.7))

# #%% plotting for Gumbel
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'Gumbel Copula, $\theta$ = {}'.format(theta))
# plt.show()

#%% test class for Frank
# FC = Frank(theta = 2)
# result = FC.generate_pairs(num=1000)
# theta_hat = FC._theta_hat(0.2)
# print(FC._c(0.5, 0.7))
# print(FC._C(0.5, 0.7))
# #%% plotting for Frank
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'Frank Copula, $\theta$ = {}'.format(FC.theta))
# plt.show()

# #%% test class for Clayton
# theta = 2
# CC = Clayton(theta = theta)
# result = CC.generate_pairs(num=1000)
# print(CC._c(0.5, 0.7))
# print(CC._C(0.5, 0.7))
# #%% plotting for Clayton
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'Clayton Copula, $\theta$ = {}'.format(theta))
# plt.show()

# #%% test class for Joe
# theta = 6
# JC = Joe(theta = theta)
# result = JC.generate_pairs(num=1000)
# print(JC._c(0.5, 0.7))
# print(JC._C(0.5, 0.7))

# #%% plotting for Joe
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'Joe Copula, $\theta$ = {}'.format(theta))
# plt.show()

# #%% test class for Gaussian
# cov = [[1, 0.5],
#        [0.5, 1]]
# GC = Gaussian(cov=cov)
# #result = GC.generate_pairs(num=3000, cov=cov)
# print(GC._C(0.5, 0.3))
# print(GC._c(0.5, 0.3))
# #%% plotting for Gaussian
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'Gaussian Copula')
# plt.show()

# #%% test class for Student
# cov = [[1, 0.5],
#        [0.5, 1]]
# nu = 3
# SC = Student(nu = nu, cov=cov)
# #print(SC._C(0.5, 0.3))
# print(GC._c(0.5, 0.3))
# result = SC.generate_pairs(num=3000, cov=cov, nu=3)

# #%% plotting for Student
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'Simulated Samples: Student-t Copula, $\nu = {}$'.format(df))
# plt.show()

# #%% test class for N13
# N13C = N13()
# theta = 4
# result = N13C.generate_pairs(num=1000, theta=theta)

# #%% plotting for N13
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'N13 Copula, $\theta = {}$'.format(theta))
# plt.show()

# #%% test class for N14
# N14C = N14()
# theta = 4
# result = N14C.generate_pairs(num=1000, theta=theta)

# #%% plotting for N14
# plt.figure(dpi=300)
# plt.scatter(result[:,0], result[:,1], s = 1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(r'N13 Copula, $\theta = {}$'.format(theta))
# plt.show()

#%% Test for multi_v_dist
# print(mv_t_dist(x=(2,3),
#                 mu=(0,0),
#                 Sigma=np.array([[2, 1],
#                                 [1, 2]]),
#                 df=5))