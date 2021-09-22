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

# Obfuscation steps
# 1. Create package: python setup.py bdist_wheel
# 2. Unzip
# 3. Obfuscate: pyarmor obfuscate --platform windows.x86_64 --platform linux.x86_64 --platform darwin.x86_64 --obf-code=0 --recursive --output dist/arbitragelab arbitragelab/__init__.py
# 4. Repackage
# 5. install
# 6. test
# 7. deploy on client repo