# Always prefer setuptools over distutils
from setuptools import setup

setup()

# ----------------------------------------------------------------------------------

# Pull new commits
# Bump version
# Update Changelog release
# Update version in docs cfg and library setup.cfg
# Make sure you double check pushing all changes to git: git push

# Tagging
# git tag [1.4.0]
# git push origin [1.4.0]
# On Github, go to tags and use the GUI to push a Release.

# Create package
# python setup.py bdist_wheel
# twine upload dist/*  (This is official repo)
