# adapted from setup.py from https://github.com/facebookresearch/llama.git

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="llm",
    version="0.0.1",
    packages=find_packages(),
    # install_requires=get_requirements("requirements.txt"),
)
