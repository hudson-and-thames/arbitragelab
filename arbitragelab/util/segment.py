# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module allows us to track how the library is used and measure statistics such as growth and retention.
"""

import os
from datetime import datetime as dt
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from requests import get
import analytics as segment
import getmac


class Analytics:
    """
    Validates the API Key and tracks imports on load.
    """
    def __init__(self):
        # Check valid API key
        self.isvalid = self.__check_api_key()

        # Identify new session
        identify()

    @staticmethod
    def __check_api_key():
        """
        Validate the API Kay.

        :return: (Bool) Valid API Key or not
        """
        # Check environment variables is present
        if API_KEY_ENV_VAR in os.environ:
            new_key = ''
            site = "https://hudson-thames.ew.r.appspot.com/api/access/" + os.environ[API_KEY_ENV_VAR]

            # Validate api key
            try:
                with urlopen(site) as response:
                    # Get validation message.
                    response_content = response.read().decode()
                    # Confirm that it is valid.
                    new_key = response_content == 'OK'
            except HTTPError as err:
                raise Exception(" ARBLAB_API_KEY is not valid.") from err
            except URLError as err:
                raise ConnectionError('Can not reach the server. Please check your connection or firewall.') from err
            # Return the results
            return new_key

        # Else the API KEy has not been registered.
        raise Exception(
            " ARBLAB_API_KEY not found in your environment variables. Please check the install instructions.")


# Get user data functions
def get_apikey():
    """
    Identify the user by API key
    """
    try:
        apikey = os.environ[API_KEY_ENV_VAR]
    except KeyError:
        apikey = 'Bandit'

    return apikey


def get_mac():
    """
    Identify the device by MAC Address
    """
    mac = getmac.get_mac_address()

    if mac is None:
        raise Exception(" No MAC Address is found on this device, must have a MAC Address.")

    return mac


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
        is_dev = bool(validate_env_variable('IS_CIRCLECI'))
    except KeyError:
        is_dev = False

    return is_dev


# Segment functions
def identify():
    """
    Identify the user and device.
    """
    # Validate not build server
    if not IS_DEV:
        segment.identify(MAC, {'mac_address': MAC,
                               'api_key': API_KEY,
                               'IP': IP,
                               'created_at': dt.now()})


def track(func):
    """
    Tracks a function call.

    :param func: String - name of function.
    """
    # Validate key
    if VALIDATOR.isvalid:
        # Validate not build server
        if not IS_DEV:
            # If 1st time func called
            if func not in TRACK_CALLS:
                TRACK_CALLS[func] = True
                segment.track(MAC, func, {'time': dt.now()})
    else:
        raise Exception(" ARBLAB_API_KEY is not valid.")


# ----------------------------------------------------------------------------------
# Body
# Env Var
API_KEY_ENV_VAR = "ARBLAB_API_KEY"
SEGMENT = 'r7uCHEvWWUshccLG6CYTOaZ3j3gA9Wpf'
IP = None
LOCATION = None
API_KEY = get_apikey()
MAC = get_mac()

# Todo: change ISDEV
# IS_DEV = is_build_server()
IS_DEV = False
TRACK_CALLS = {}


try:
    IP = get('http://checkip.amazonaws.com/').text.strip()
except ConnectionError as err:
    raise ConnectionError('Can not reach the server. Please check your connection or firewall.') from err

# Connect with DB
segment.write_key = SEGMENT
VALIDATOR = Analytics()
