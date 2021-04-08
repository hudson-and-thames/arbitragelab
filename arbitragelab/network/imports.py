# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module validates your API key. Please note that making changes to this file or any of the files in this library is
a violation of the license agreement. If you have any problems, please contact us: research@hudsonthames.org.
"""

import os
import json
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

from arbitragelab.util import devadarsh

API_KEY_ENV_VAR = "ARBLAB_API_KEY"


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
    def __check_api_key():
        # Check environment variables is present
        if API_KEY_ENV_VAR in os.environ:
            # Validate key
            old_key, new_key = '', ''

            # Check old server
            try:
                with urlopen("https://us-central1-hudson-and-thames.cloudfunctions.net/checkKey/" + os.environ[API_KEY_ENV_VAR]) as response:
                    response_content = response.read()
                    json_response = json.loads(response_content)
                    old_key = json_response['status'] == 'active'
            except HTTPError:
                pass
            except URLError as err:
                raise ConnectionError('Can not reach the server. Please check your connection or firewall.') from err

            # Check new server
            try:
                if not old_key:
                    with urlopen("https://hudson-thames.ew.r.appspot.com/api/access/" + os.environ[API_KEY_ENV_VAR]) as response:
                        response_content = response.read().decode()
                        new_key = response_content == 'OK'
            except HTTPError as err:
                raise Exception(" ARBLAB_API_KEY is not valid.") from err

            # Return the results
            return old_key or new_key
        # Else the API KEy has not been registered.
        raise Exception(" ARBLAB_API_KEY not found in your environment variables. Please check the install instructions.")

    # pylint: disable=missing-function-docstring
    @staticmethod
    def __import_libraries():
        # pylint: disable=import-outside-toplevel, unused-import
        import arbitragelab.ml_approach as ml_approach
        import arbitragelab.codependence as codependence
        import arbitragelab.cointegration_approach as cointegration_approach
        import arbitragelab.copula_approach as copula_approach
        import arbitragelab.distance_approach as distance_approach
        import arbitragelab.pca_approach as pca_approach
        import arbitragelab.other_approaches as other_approaches
        import arbitragelab.util as util
        import arbitragelab.optimal_mean_reversion as optimal_mean_reversion
        import arbitragelab.time_series_approach as time_series_approach
        devadarsh.track('Import')

    # pylint: disable=missing-function-docstring
    def __validate(self):
        if self.__check_api_key():
            self.__import_libraries()
        else:
            raise Exception("Invalid API Key: Please check the install instructions as well as your account.")
