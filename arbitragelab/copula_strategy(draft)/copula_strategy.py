# -*- coding: utf-8 -*-
"""
Master module that graphs and fits copula
Created on Sun Nov  8 19:12:38 2020

@author: Hansen
"""
import copula_generate
import matplotlib.pyplot as plt
import pandas as pd

class Copula_Strategy:
    def __init__(self, copula_names: list):
        self.copula_names = copula_names
        self.theta_copula_names = ['Gumbel', 'Clayton', 'Frank',
                                   'Joe', 'N13', 'N14']
        self.cov_copula_names = ['Gaussian', 'Student']
        self.all_copula_names = self.theta_copula_names \
            + self.cov_copula_names
            
    def max_likelihood_theta(self, data: pd.DataFrame, copula_name: str):
        """
        Generate the max likelihood estimation of theta
        """
        pass

    def graph_copula(self, copula_name, **kwargs):
        """
        Graph the sample from a given copula
        """
        # print(kwargs)
        num = kwargs.get('num', 2000)
        dpi = kwargs.get('dpi', 300)
        s = kwargs.get('s', 1)
        plot_style = kwargs.get('plot_style', 'default')
        theta = kwargs.get('theta', None)
        cov = kwargs.get('cov', None)
        nu = kwargs.get('nu', None)

        plt.style.use(plot_style)
        if copula_name in self.theta_copula_names:
            # Generate data for plotting
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    theta=theta)
            result = my_copula.generate_pairs(num=num)

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.scatter(result[:,0], result[:,1], s = s)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(r'{} Copula, $\theta={}$'
                         .format(copula_name,
                                 theta))
            fig.show()

        elif copula_name == 'Gaussian':
            # Generate data for plotting
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    cov=cov)
            result = my_copula.generate_pairs(num=num)

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.scatter(result[:,0], result[:,1], s = s)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(r'{} Copula, $\rho={}$'
                         .format(copula_name,
                                 my_copula.rho))
            fig.show()
            
        elif copula_name == 'Student':
            # Generate data for plotting
            my_copula = self._create_copula_by_name(copula_name=copula_name,
                                                    cov=cov,
                                                    nu=nu)
            result = my_copula.generate_pairs(num=num)

            fig = plt.figure(dpi=dpi)
            ax = fig.add_subplot(111)
            ax.scatter(result[:,0], result[:,1], s = s)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(r'{} Copula, $\rho={}$, $\nu={}$'
                         .format(copula_name,
                                 my_copula.rho,
                                 my_copula.nu))
            fig.show()

    def _create_copula_by_name(self, **kwargs):
        Switch = copula_generate.Switcher()
        result = Switch.choose_copula(**kwargs)
        return result
        
    
# #%% Test Copula Strategy
# CS = Copula_Strategy(['Student', 'Gumbel'])
# cov = [[1, 0.8],
#         [0.8, 1]]
# nu = 3
# CS.graph_copula('Student', dpi=300, num=2500, plot_style='default',
#                 cov=cov, nu=nu)


# #%%
# SC = copula_generate.Student(cov=cov, df=nu)
# SC.generate_pairs(1000)