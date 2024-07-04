#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  Copyright (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
import os

from setuptools import find_packages
from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


PACKAGENAME: str = "fastwonn"

setup(
    name=PACKAGENAME,
    version="0.0.3",
    author="Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/fastwonn",
    description="Fast, GPU-friendly, differentiable computation of Intrinsic Dimension via the TwoNN algorithm (Facco et al., 2017)",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Differentiable Programming",
        "Intrinsic Dimension",
        "Machine Learning",
        "Manifold Learning",
        "Maximum Likelihood Estimation",
        "PyTorch",
        "TwoNN",
    ],
    license="MIT",
    packages=[
        package for package in find_packages() if package.startswith(PACKAGENAME)
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=["torch>=2", "torchsort>=0.1.9"],
    extras_require={
        "KeOps": ["keopscore>=2.2.3", "pykeops>=2.2.3"],
        "FAISS CPU": ["faiss-cpu>=1.8.0"],
        "FAISS GPU (all CUDAs, older)": ["faiss-gpu>=1.7.2"],
        "FAISS GPU (CUDA 11 only, newer)": ["faiss-gpu-cu11>=1.8.0.2"],
        "FAISS GPU (CUDA 12 only, newer)": ["faiss-gpu-cu12>=1.8.0.2"],
    },
    include_package_data=True,
    zip_safe=False,
)
