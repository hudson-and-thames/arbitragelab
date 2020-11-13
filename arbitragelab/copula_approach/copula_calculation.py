# Copyright 2020, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
"""
Back end module that handles maximum likelihood related copula calculations.

Functions include:
    Finding (marginal) cumulative distribution function from data.
    Maximum likelihood estimation of theta_hat (empirical theta) from data.
    Calculate the sum log likelihood given a copula and data.
    Calculate SIC (Schwarz information criterion).
    Calculate AIC (Akaike information criterion).
    Calculate HQIC (Hannan-Quinn information criterion).
"""
import copula_generate as cg
import numpy as np
from scipy.stats import kendalltau
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import norm
from scipy.stats import t as student_t
from statsmodels.distributions.empirical_distribution import ECDF

theta_copula_names = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14']
cov_copula_names = ['Gaussian', 'Student']

def find_marginal_cdf(x, empirical=True):
    """
    Find the cumulative density function (CDF) from scaled data. i.e., P(X<=x)

    User can choose between empirical CDF or a CDF selected by maximum likelihood. 
    :param x: (np.array) Data, need to be scaled to interval [0, 1].
    :param empirical: (bool) Whether to use empirical estimation for CDF.
    :return fitted_cdf: (func) The cumulative density function from data.
    """
    if empirical:
        # To prevent probability 0 and 1 from occuring, we augment the data set a bit.
        augment = [-0.00001, 1.00001]
        new_x = np.insert(x, 0, augment)
        # Use empirical cumulative density function on data.
        fitted_cdf = ECDF(new_x)
        return fitted_cdf
    else: # Choose a distribution by maximum likelihood estimation.
        return fitted_cdf

def ml_theta_hat(x, y, copula_name):
    """
    Calculate empirical theta (theta_hat) for a type of copula by maximum likelihood.
    
    x, y need to be uniformly distributed respectively. Use Kendall's tau value to
    calculate theta hat.

    Note: Gaussian and Student-t copula do not use this function.
    :param x: (np.array) 1D vector data.
    :param y: (np.array) 1D vector data.
    :param copula_name: (str) Name of the copula.
    :return theta_hat: (float) Empirical theta for the copula.
    """
    # Calculate Kendall's tau from data.
    tau = kendalltau(x, y)[0]
    # Calculate theta from the desired copula.
    dud_cov = [[1,0],[0,1]]  # To create copula by name. Not involved in calculations.
    # Create copula by its name. Fulfil switch functionality.
    Switch = cg.Switcher()
    my_copula = Switch.choose_copula(copula_name=copula_name,
                                     cov=dud_cov)
    # Translate Kendall's tau into theta.
    theta_hat = my_copula._theta_hat(tau)
    
    return theta_hat

def log_ml(x, y, copula_name, nu=None):
    """
    Fit a type of copula using maximum likelihood.
    
    User provide the name of the copula (and degree of freedom nu, if it is 'Student-t'), then this method
    fits the copyla type by maximum likelihood. Moreover, it calculates log maximum likelihood.
    
    :param x: (np.array) 1D vector data. Need to be uniformly distributed.
    :param y: (np.array) 1D vector data. Need to be uniformly distributed.
    :param copula_name: (str) Name of the copula.
    :param nu: (float) Degree of freedom for Student-t copula.
    :return (tuple):
        log_likelihood_sum (float): Logarithm of max likelihood value from data.
        my_copula (Copula): Copula with its parameter fitted to data.
    """
    # Find log max likelihood given all the data.
    Switch = cg.Switcher()
    if copula_name in theta_copula_names:
        # Get the max likelihood theta_hat for theta from data.
        theta = ml_theta_hat(x, y, copula_name)
        my_copula = Switch.choose_copula(copula_name=copula_name,
                                         theta=theta)
    elif copula_name == 'Gaussian':
        # 1. Calculate covariance matrix using sklearn.
        # Correct matrix dimension for fitting in sklearn.
        unif_data = np.array([x,y]).reshape(2,-1).T 
        value_data = norm.ppf(unif_data)  # Change from quantile to value.
        # Getting empirical covariance matrix.
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        
        # 2. Construct copula with fitted parameter.
        my_copula = Switch.choose_copula(copula_name=copula_name,
                                         cov=cov_hat)
    elif copula_name == 'Student':
        # 1. Calculate covariance matrix using sklearn.
        # Correct matrix dimension for fitting in sklearn.
        unif_data = np.array([x,y]).reshape(2,-1).T
        t_dist = student_t(df=nu)
        value_data = t_dist.ppf(unif_data)  # Change from quantile to value.
        # Getting empirical covariance matrix.
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        
        # 2. Construct copula with fitted parameter.
        my_copula = Switch.choose_copula(copula_name=copula_name,
                                         cov=cov_hat,
                                         nu=nu)
    
    # Likelihood quantity for each pair of data, stored in a list.
    likelihood_list = [my_copula._c(xi, yi) for (xi, yi) in zip(x,y)]
    # Sum of logarithm of likelihood data.
    log_likelihood_sum = np.sum(np.log(likelihood_list))
    
    return log_likelihood_sum, my_copula

def sic(log_likelihood: float, n: int, k=1):
    """
    Schwarz information criterion (SIC), aka Bayesian information criterion (BIC).
    
    :param log_likelihood (float): Sum of log likelihood of some data.
    :param n (int): Number of instances.
    :param k (int): Number of parametrs estimated by max likelihood.
    :return sic_value (float): Value of SIC.
    """
    sic_value = np.log(n)*k - 2*log_likelihood
    return sic_value

def aic(log_likelihood: float, n: int, k=1):
    """
    Akaike information criterion.
    
    :param log_likelihood (float): Sum of log likelihood of some data.
    :param n (int): Number of instances.
    :param k (int): Number of parametrs estimated by max likelihood.
    :return sic_value (float): Value of AIC.
    """
    aic_value = (2*n/(n-k-1))*k - 2*log_likelihood
    return aic_value

def hqic(log_likelihood: float, n: int, k=1):
    """
    Hannan-Quinn information criterion.
    
    :param log_likelihood (float): Sum of log likelihood of some data.
    :param n (int): Number of instances.
    :param k (int): Number of parametrs estimated by max likelihood.
    :return sic_value (float): Value of HQIC.
    """
    hqic_value = 2*np.log(np.log(n))*k - 2*log_likelihood
    return hqic_value

# #%% Test
# # 1. generate x, y from Gumbel copula
# GC = cg.Gumbel(theta=3)
# data = GC.generate_pairs(num=1000)
# # 2. max likelihood estimate of theta hat
# theta_hat = ml_theta_hat(data[:,0], data[:,1], copula_name='Gumbel')
# # 3. sic estimation
# ll = log_ml(data[:,0], data[:,1], copula_name='Gumbel')
# sic_value = sic(log_likelihood=ll, n=len(data[:,0]))
# aic_value = aic(log_likelihood=ll, n=len(data[:,0]))
# hqic_value = hqic(log_likelihood=ll, n=len(data[:,0]))

# print('SIC = {}'.format(sic_value))
# print('AIC = {}'.format(aic_value))
# print('HQIC = {}'.format(hqic_value))

# #%% Test for Gaussian
# # 1. generate x, y from Gaussian copula
# cov = [[2, 0.5],
#        [0.5, 2]]
# GaussianC = cg.Gaussian(cov=cov)
# data = GaussianC.generate_pairs(num=3000)
# x = norm.ppf(data[:, 0])
# y = norm.ppf(data[:, 1])
# # 2. max likelihood estimate of theta hat
# theta_hat = ml_theta_hat(x, y, copula_name='Gaussian')
# # 3. max likelihood estimate of covariance matrix
# cov_hat = EmpiricalCovariance().fit(norm.ppf(data)).covariance_

# #%% Test for Student t
# # 1. generate x, y from Student copula
# cov = [[2, 1],
#        [1, 2]]
# nu = 5
# StudentC = cg.Student(cov=cov, nu=nu)
# data = StudentC.generate_pairs(num=3000)
# t_dist = student_t(df=nu)
# x = t_dist.ppf(data[:, 0])
# y = t_dist.ppf(data[:, 1])
# # 2. max likelihood estimate of theta hat
# theta_hat = ml_theta_hat(x, y, copula_name='Student')
# # 3. max likelihood estimate of covariance matrix
# cov_hat = EmpiricalCovariance().fit(t_dist.ppf(data)).covariance_

# #%% ML Test Student T
# # 1. generate x, y from Student copula
# cov = [[2, 1],
#        [1, 2]]
# nu = 5
# StudentC = cg.Student(cov=cov, nu=nu)
# data = StudentC.generate_pairs(num=3000)
# # 2. IC estimation
# ll, like_list = log_ml(data[:,0], data[:,1], copula_name='Student', nu=nu)
# sic_value = sic(log_likelihood=ll, n=len(data[:,0]))
# aic_value = aic(log_likelihood=ll, n=len(data[:,0]))
# hqic_value = hqic(log_likelihood=ll, n=len(data[:,0]))

# print('SIC = {}'.format(sic_value))
# print('AIC = {}'.format(aic_value))
# print('HQIC = {}'.format(hqic_value))

# #%% plot student_t density
# cov = [[2, 1],
#        [1, 2]]
# nu = 5
# StudentC = cg.Student(cov=cov, nu=nu)
# data = StudentC.generate_pairs(num=3000)

# from mpl_toolkits import mplot3d
# x_coord = np.linspace(0.001,1-0.001,100)
# y_coord = np.linspace(0.001,1-0.001,100)

# X, Y = np.meshgrid(x_coord, y_coord)
# Z_coord = [[0]*100 for i in range(100)]
# for i in range(100):
#     for j in range(100):
#         Z_coord[i][j] = [x_coord[i], y_coord[j]]
        
# Z = [[None]*100 for i in range(100)]
# for i in range(100):
#     for j in range(100):
#         Z[i][j] = StudentC._c(Z_coord[i][j][0], Z_coord[i][j][1])
# #%%
# Z =np.array(Z)
# plt.figure(dpi=300)
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_zlim(0, 5);
# ax.set_title('surface');

