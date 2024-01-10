# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module allows us to track how the library is used and measure statistics such as growth and retention.
"""

import os

# from datetime import datetime as dt
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from requests import get


class Analytics:
    """
    Validates the API Key and tracks imports on load.
    """

    def __init__(self):
        # Check valid API key
        self.__isvalid = self.__check_api_key()

    def is_valid(self):
        """
        Returns the result of the APIKey Validation.
        :return: (Bool) APIKEY Valid
        """
        return self.__isvalid

    @staticmethod
    def __check_api_key():
        """
        Validate the API Kay.

        :return: (Bool) Valid API Key or not
        """
        # Check environment variables is present
        if API_KEY_ENV_VAR in os.environ:
            # site for portal.hudsonthames.org -- new system
            site1 = (
                "https://portal.hudsonthames.org/api/verify/"
                + os.environ[API_KEY_ENV_VAR]
                + "/arbitrage"
            )
            # site for hudson-thames.ew.r.appspot.com -- old system
            site2 = (
                "https://hudson-thames.ew.r.appspot.com/api/access/"
                + os.environ[API_KEY_ENV_VAR]
            )

            # Validate API key with portal.hudsonthames.org
            try:
                with urlopen(site1) as response:
                    response_content = response.read().decode()
                    if response_content == "true":
                        return True
            except (HTTPError, URLError):
                pass

            # Validate API key with hudson-thames.ew.r.appspot.com
            try:
                with urlopen(site2) as response:
                    response_content = response.read().decode()
                    if response_content == "OK":
                        return True
            except (HTTPError, URLError):
                pass

            return False

        # Else the API KEy has not been registered.
        raise Exception(
            " ARBLAB_API_KEY not found in your environment variables. Please check the install instructions."
        )


# Get user data functions
def get_apikey():
    """
    Identify the user by API key
    """
    try:
        apikey = os.environ[API_KEY_ENV_VAR]
    except KeyError:
        apikey = "Bandit"

    return apikey


# Validate functions
def validate_env_variable(env_variable_name):
    """
    Check that a environment variables are setup correctly and that they are valid.
    """
    try:
        is_valid = bool(os.environ[env_variable_name])
    except KeyError:
        is_valid = False

    return is_valid


def is_build_server():
    """
    Check if device is build server.
    """
    try:
        is_dev = bool(validate_env_variable("IS_CIRCLECI"))
    except KeyError:
        is_dev = False

    return is_dev


def track(func):
    """
    Tracks a function call.

    :param func: String - name of function.
    """
    pass


# ----------------------------------------------------------------------------------
# Body
# Env Var
API_KEY_ENV_VAR = "ARBLAB_API_KEY"
IP = None
API_KEY = get_apikey()
IS_DEV = is_build_server()
TRACK_CALLS = {}

try:
    IP = get("http://checkip.amazonaws.com/").text.strip()
except ConnectionError as err:
    raise ConnectionError(
        "Can not reach the Amazon CheckIP server. Please check your connection or firewall."
    ) from err

# Connect with DB
VALIDATOR = Analytics()
