# -*- coding: utf-8 -*-
"""
Module that handles copula calculations
Created on Sun Nov  8 22:06:40 2020

@author: Hansen
"""
import copula_generate as cg
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import norm
from scipy.stats import t as student_t
import matplotlib.pyplot as plt

# if __name__ != '__main__':
theta_copula_names = ['Gumbel', 'Clayton', 'Frank', 'Joe', 'N13', 'N14']
cov_copula_names = ['Gaussian', 'Student']

def find_marginal_dist(x):
    pass

def ml_theta_hat(x, y, copula_name: str):
    # 1. Calculate Kendall's tau from data
    tau = kendalltau(x, y)[0]
    
    # 2. Calculate theta from the desired copula
    dud_cov = [[1,0],[0,1]]
    Switch = cg.Switcher()
    my_copula = Switch.choose_copula(copula_name=copula_name,
                                     cov=dud_cov)
    theta_hat = my_copula._theta_hat(tau)
    
    return theta_hat

def log_ml(x, y, copula_name: str, nu: int=None):
    """
    Log of max likelihood of a given copula type.
    
    x, y need to be uniformly distributed.
    """
    # Find log max likelihood given all the data
    Switch = cg.Switcher()
    if copula_name in theta_copula_names:
        theta = ml_theta_hat(x, y, copula_name)
        my_copula = Switch.choose_copula(copula_name=copula_name,
                                         theta=theta)
    elif copula_name == 'Gaussian':
        unif_data = np.array([x,y]).reshape(2,-1).T  # correct dim for fitting
        value_data = norm.ppf(unif_data)
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        my_copula = Switch.choose_copula(copula_name=copula_name,
                                         cov=cov_hat)
    elif copula_name == 'Student':
        unif_data = np.array([x,y]).reshape(2,-1).T  # correct dim for fitting
        t_dist = student_t(df=nu)
        value_data = t_dist.ppf(unif_data)
        cov_hat = EmpiricalCovariance().fit(value_data).covariance_
        print('cov_hat', cov_hat)
        my_copula = Switch.choose_copula(copula_name=copula_name,
                                         cov=cov_hat,
                                         nu=nu)
    
    likelihood_list = [my_copula._c(xi, yi) for (xi, yi) in zip(x,y)]
    log_likelihood_sum = np.sum(np.log(likelihood_list))
    
    return log_likelihood_sum, likelihood_list

def sic(log_likelihood: float, n: int, k=1):
    sic_value = np.log(n)*k - 2*log_likelihood
    return sic_value

def aic(log_likelihood: float, n: int, k=1):
    aic_value = (2*n/(n-k-1))*k - 2*log_likelihood
    return aic_value

def hqic(log_likelihood: float, n: int, k=1):
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
    
