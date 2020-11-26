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

API_KEY_ENV_VAR = "ARBLAB_API_KEY"


class Security:
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
            with urlopen("https://us-central1-hudson-and-thames.cloudfunctions.net/checkKey/" + os.environ[API_KEY_ENV_VAR]) as response:
                response_content = response.read()
                json_response = json.loads(response_content)
                return json_response['status'] == 'active'
        else:
            raise Exception(" ARBLAB_API_KEY not found in your environment variables.")

    # pylint: disable=missing-function-docstring
    @staticmethod
    def __import_libraries():
        # pylint: disable=import-outside-toplevel, unused-import
        import arbitragelab.codependence as codependence
        import arbitragelab.cointegration_approach as cointegration_approach
        import arbitragelab.distance_approach as distance_approach
        import arbitragelab.other_approaches as other_approaches
        import arbitragelab.util as util
        import arbitragelab.optimal_mean_reversion as optimal_mean_reversion

    # pylint: disable=missing-function-docstring
    def __validate(self):
        if self.__check_api_key():
            self.__import_libraries()
        else:
            raise Exception("Invalid API Key: Please check the install instructions as well as your account.")
