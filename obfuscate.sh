#!
# This script builds, unpacks, obfuscates, and then rebuilds the python wheel
# package

# Set the version
# TODO: Do this automatically in the future by reading from setup.cfg (or
# similar)
VERSION=0.9.0

# Start clean
rm -rf dist/

# Build the wheel
python setup.py bdist_wheel

# Unpack the wheel
python -m wheel unpack dist/arbitragelab-$VERSION-py3-none-any.whl --dest dist/

# Generate a PyArmor license
pyarmor-7 licenses c1_version # --expired 2024-02-01 c1_version <-- use this to make license expire

# Obfuscate using PyArmor
pyarmor-7 obfuscate \
    --with-license licenses/c1_version/license.lic \
    --platform windows.x86_64 \
    --platform linux.x86_64 \
    --platform darwin.x86_64 \
    --platform darwin.aarch64 \
    --platform linux.x86 \
    --obf-code=0 \
    --recursive \
    --output dist/arbitragelab-$VERSION/arbitragelab arbitragelab/__init__.py

# Repack wheel
python -m wheel pack dist/arbitragelab-$VERSION --dest dist

# Clean-up
rm -rf dist/arbitragelab-$VERSION
rm -rf build
rm -rf licenses
