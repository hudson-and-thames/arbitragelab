"""
Notebook for implementing copula strategies.

@author: Hansen
"""
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
# from sklearn import preprocessing
from scipy.stats import norm
from scipy.stats import t as student_t

import arbitragelab.copula_approach.copula_strategy as copula_strategy
import arbitragelab.copula_approach.copula_generate as cg
import arbitragelab.copula_approach.copula_calculation as ccalc

# pylint: disable=invalid-name, protected-access

pair_prices = pd.read_csv(r'arbitragelab\BKD_ESC_2009_2011.csv')
CS = copula_strategy.CopulaStrategy()
BKD_series = pair_prices['BKD'].to_numpy()
ESC_series = pair_prices['ESC'].to_numpy()
dates = pair_prices['Date'].to_numpy()

#%% Training and testing split
training_length = 670

BKD_clr = CS.cum_log_return(BKD_series)
ESC_clr = CS.cum_log_return(ESC_series)

BKD_train = BKD_clr[ : training_length]
ESC_train = ESC_clr[ : training_length]

BKD_test = BKD_clr[training_length : ]
ESC_test = ESC_clr[training_length : ]

# Empirical CDF for the training set
cdf1 = ccalc.find_marginal_cdf(BKD_clr)
cdf2 = ccalc.find_marginal_cdf(ESC_clr)

#%%
u1 = cdf1(BKD_train)
u2 = cdf2(ESC_train)

unif_data = np.array([u1, u2]).reshape(2, -1).T
value_data = norm.ppf(unif_data)  # Change from quantile to value.
# Getting empirical covariance matrix.
cov_hat = EmpiricalCovariance().fit(value_data).covariance_

# 2. Construct copula with fitted parameter.
Switch = cg.Switcher()
my_copula = Switch.choose_copula(copula_name='Student',
                                 cov=cov_hat, nu=5)

likelihood_list = [my_copula._c(xi, yi) for (xi, yi) in zip(u1, u2)]
#logml, copula = ccalc.log_ml(x=u1, y=u2, copula_name='Student', nu=669)
#%%
nu = 669
rho = my_copula.rho
corr = [[1, rho],
        [rho, 1]]
t_dist = student_t(df=nu)
y1 = t_dist.ppf(0.5)
y2 = t_dist.ppf(0.7)
