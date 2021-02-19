# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/arbitragtelab/blob/master/LICENSE.txt


# Always prefer setuptools over distutils
from setuptools import setup

setup()

# Bump version
# Make sure install codes are updated in the docs (Installation guide).
# Update Changelog release
# bumpversion major/minor/patch --allow-dirty
# Double-check if you did a Git Push
# git push origin [0.1.0]
# On Github, go to tags and use the GUI to push a Release.

# Create package
# python setup.py bdist_wheel
# twine upload dist/*  (This is official repo)
