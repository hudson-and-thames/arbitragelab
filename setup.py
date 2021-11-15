# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/arbitragtelab/blob/master/LICENSE.txt


# Always prefer setuptools over distutils
from setuptools import setup

setup()

# Old instructions
# Create package
# python setup.py bdist_wheel
# twine upload dist/*  (This is official repo)
# ----------------------------------------------------------------------------------

# Pull new commits
# Bump version
# Update Changelog release
# Update version in docs cfg and library setup.cfg
# Update install location on install.rst
# Make sure you double check pushing all changes to git: git push

# Obfuscation steps
# 1. Create package: python setup.py bdist_wheel
# 2. Unzip the dist.whl file
# 2.2 cd into the unzipped dir
# 3. Obfuscate: pyarmor obfuscate --platform windows.x86_64 --platform linux.x86_64 --platform darwin.x86_64 --obf-code=0 --recursive --output dist/mlfinlab mlfinlab/__init__.py
# 4. Add back datasets in mlfinlab
# 4. Repackage
# 5. install
# 6. test
# 7. deploy on client repo

# Tagging
# git tag [1.4.0]
# git push origin [1.4.0]
# On Github, go to tags and use the GUI to push a Release.
