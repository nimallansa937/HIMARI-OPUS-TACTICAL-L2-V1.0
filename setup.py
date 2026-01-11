"""
HIMARI OPUS 2 Layer 2 - Setup
"""

from setuptools import setup, find_packages

setup(
    name="himari_layer2",
    version="2.1.1",
    description="HIMARI OPUS 2 Layer 2 Tactical Decision Layer",
    author="HIMARI OPUS Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
