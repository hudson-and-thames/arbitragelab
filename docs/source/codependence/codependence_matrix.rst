.. _codependence-codependence_matrix:

===================
Codependence Matrix
===================

.. raw:: html

    <div style="position: relative;
                padding-bottom: 56.25%;
                margin-bottom: 5%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                height: auto;">

        <iframe src="https://www.youtube.com/embed/YyMouLPj2QA?start=668"
                frameborder="0"
                allowfullscreen
                style="position: absolute;
                       top: 0;
                       left: 0;
                       width: 100%;
                       height: 100%;">
        </iframe>
        <br/>
    </div>

|

The functions in this part of the module are used to generate dependence and distance matrices using the codependency and
distance metrics described previously.

1. **Dependence Matrix** function is used to compute codependences between elements in a given dataframe of elements
   using various codependence metrics like Mutual Information, Variation of Information, Distance Correlation,
   Spearman's Rho, GPR distance, and GNPR distance.

2. **Distance Matrix** function can be used to compute a distance matrix from a given codependency matrix using
   distance metrics like angular, squared angular and absolute angular.

.. Note::
    **Underlying Literature**

    The following sources elaborate extensively on the topic:

    - `Codependence (Presentation Slides) <https://ssrn.com/abstract=3512994>`__ *by* Marcos Lopez de Prado.

Implementation
##############

.. py:currentmodule:: arbitragelab.codependence.codependence_matrix
.. autofunction:: get_dependence_matrix
.. autofunction:: get_distance_matrix


Example
#######

.. code-block::

   import pandas as pd
   from arbitragelab.codependence import (get_dependence_matrix, get_distance_matrix)

    # Import dataframe of returns for assets in a portfolio
    asset_returns = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

    # Calculate distance correlation matrix
    distance_corr = get_dependence_matrix(asset_returns, dependence_method='distance_correlation')

    # Calculate Pearson correlation matrix
    pearson_corr = asset_returns.corr()

    # Calculate absolute angular distance from a Pearson correlation matrix
    abs_angular_dist = absolute_angular_distance(pearson_corr)

Presentation Slides
###################

.. image:: images/codependence_slides.png
   :scale: 40 %
   :align: center
   :target: https://drive.google.com/file/d/1pamteuYyc06r1q-BR3VFsxwa3c7-7oeK/view

References
##########

* `de Prado, M.L., 2020. Codependence (Presentation Slides). Available at SSRN 3512994. <https://ssrn.com/abstract=3512994>`_
