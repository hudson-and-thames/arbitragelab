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
API_KEY_ENV_VAR = "ARBLAB_API_KEY"  # User ID
SEGMENT = 'r7uCHEvWWUshccLG6CYTOaZ3j3gA9Wpf'


try:
    IS_CIRCLECI = bool(os.environ['IS_CIRCLECI'])
except KeyError:
    IS_CIRCLECI = False

# Set Location and User
IP = None
USER = "unknown"
IS_DEV = IS_CIRCLECI or (os.environ["ARBLAB_API_KEY"][-4:] == '62a0')

# pylint: disable=bare-except
try:
    IP = get('https://api.ipify.org').text
    LOCATION = {"ip": IP}
except:
    LOCATION = None

try:
    USER = os.environ[API_KEY_ENV_VAR]
except KeyError:
    USER = 'Robert Angier'

# Connect with DB
analytics.write_key = SEGMENT


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
