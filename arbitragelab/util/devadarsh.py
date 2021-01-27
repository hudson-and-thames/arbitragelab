# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://hudson-and-thames-arbitragelab.readthedocs-hosted.com/en/latest/additional_information/license.html

"""
This module allows us to track how the library is used and measure statistics such as growth and lifetime.
"""

import os
from datetime import datetime as dt
from requests import get
import analytics


# pylint: disable=missing-function-docstring
def validate_env_variable(env_variable_name):
    try:
        is_valid = bool(os.environ[env_variable_name])
    except KeyError:
        is_valid = False

    return is_valid


# pylint: disable=missing-function-docstring
def get_user():
    try:
        user = os.environ[API_KEY_ENV_VAR]
    except KeyError:
        user = 'Robert Angier'

    return user


# pylint: disable=missing-function-docstring
def validate_alum():
    try:
        is_circle = bool(validate_env_variable('IS_CIRCLECI'))
        is_rtd = bool(validate_env_variable('IS_RTD'))
        is_alum = get_user()[-4:] == '62a0'
        is_dev = is_circle or is_rtd or is_alum
    except KeyError:
        is_dev = False

    return is_dev


# Identify
# pylint: disable=missing-function-docstring
def identify():
    if not IS_DEV:
        analytics.identify(USER, {"name": USER,
                                  'created_at': dt.now()})


# Generic function for pinging the server
# pylint: disable=missing-function-docstring
def page(url):
    if not IS_DEV:
        analytics.page(USER, 'ArbitrageLab', 'Import',
                       {"url": url,
                        'time': dt.now()},
                       context=LOCATION)


# pylint: disable=missing-function-docstring
def track(func):
    if not IS_DEV:
        analytics.track(USER, func, {'time': dt.now()})


# Env Var
API_KEY_ENV_VAR = "ARBLAB_API_KEY"  # User ID
SEGMENT = 'r7uCHEvWWUshccLG6CYTOaZ3j3gA9Wpf'

IP = None
LOCATION = None
USER = get_user()
IS_DEV = validate_alum()

# pylint: disable=bare-except
try:
    IP = get('https://api.ipify.org').text
    LOCATION = {"ip": IP}
except:
    LOCATION = None

# Connect with DB
analytics.write_key = SEGMENT
