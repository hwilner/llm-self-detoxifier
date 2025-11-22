"""
Setup script for the SASA package.
"""

import os
from setuptools import setup, find_packages

setup(
    name="sasa",
    version="0.1.0",
    author="hwilner",
    description="An implementation of Self-disciplined Autoregressive Sampling (SASA) for LLM detoxification.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/hwilner/llm-self-detoxifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
    ],
)
