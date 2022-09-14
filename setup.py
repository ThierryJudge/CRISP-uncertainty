#!/usr/bin/env python
import pathlib

from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
PATH_ROOT = pathlib.Path(__file__).parent


def load_long_description():  # noqa: D103
    text = open(PATH_ROOT / "README.md", encoding="utf-8").read()
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace(".svg", ".png")
    return text


setup(
    name="crisp-uncertainty",
    version="0.0.1",
    description="CRISP Uncertainty.",
    author="Thierry Judge",
    author_email="thierry.judge@usherbrooke.ca",
    url="https://github.com/ThierryJudge/CRISP-uncertainty",
    license="Apache-2.0",
    packages=find_packages(exclude=["vital", "vital/*"]),
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    setup_requires=[],
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
