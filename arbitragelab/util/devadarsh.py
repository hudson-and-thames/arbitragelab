# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt

"""
This module allows us to track how the library is used and measure statistics such as growth and lifetime.
"""

import os
from datetime import datetime as dt
from requests import get
import analytics

# Env Var
IS_CIRCLECI = False  # Don't run inside CircleCI
API_KEY_ENV_VAR = "ARBLAB_API_KEY"  # User ID

# pylint: disable=bare-except
try:
    IS_CIRCLECI = bool(os.environ['IS_CIRCLECI'])
except:
    pass

# Set Location and User
IP = None
USER = "unknown"

# pylint: disable=bare-except
try:
    IP = get('https://api.ipify.org').text
    LOCATION = {"ip": IP}
except:
    LOCATION = None

try:
    USER = os.environ[API_KEY_ENV_VAR]
except KeyError:
    USER = 'Angier'

# Connect with DB
analytics.write_key = 'r7uCHEvWWUshccLG6CYTOaZ3j3gA9Wpf'


# Identify
def identify():
    """
    Identify user.
    """
    if not IS_CIRCLECI:
        analytics.identify(USER, {"name": USER,
                                  'created_at': dt.now()})


# Generic function for pinging the server
def page(url):
    """
    Page opened.
    :param url: the page url
    """
    if not IS_CIRCLECI:
        analytics.page(USER, 'ArbitrageLab', 'Import',
                       {"url": url,
                        'time': dt.now()},
                       context=LOCATION)


def track(func):
    """
    User action.
    :param func: action name
    """
    if not IS_CIRCLECI:
        analytics.track(USER, func, {'time': dt.now()})
