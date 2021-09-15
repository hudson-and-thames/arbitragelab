# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module validates your API key. Please note that making changes to this file or any of the files in this library is
a violation of the license agreement. If you have any problems, please contact us: research@hudsonthames.org.
"""
from arbitragelab.util import segment


class Golem:
    """
    Class to verify the API Key of the end user. Make sure that you have declared your environment variables. See the
    installation documentation for more details.
    """

    # pylint: disable=missing-function-docstring
    def __init__(self):
        self.__validate()

    # pylint: disable=missing-function-docstring
    @staticmethod
    def __import_libraries():
        # pylint: disable=import-outside-toplevel, unused-import
        import arbitragelab.codependence as codependence
        import arbitragelab.cointegration_approach as cointegration_approach
        import arbitragelab.copula_approach as copula_approach
        import arbitragelab.distance_approach as distance_approach
        import arbitragelab.ml_approach as ml_approach
        import arbitragelab.optimal_mean_reversion as optimal_mean_reversion
        import arbitragelab.other_approaches as other_approaches
        import arbitragelab.stochastic_control_approach as stochastic_control_approach
        import arbitragelab.tearsheet as tearsheet
        import arbitragelab.time_series_approach as time_series_approach
        import arbitragelab.hedge_ratios as hedge_ratios
        import arbitragelab.pairs_selection as pairs_selection
        import arbitragelab.util as util
        segment.track('Import')

    # pylint: disable=missing-function-docstring
    def __validate(self):
        if segment.VALIDATOR.isvalid:
            self.__import_libraries()
        else:
            print('Invalid API Key, please check your account or ENV Variables.')
