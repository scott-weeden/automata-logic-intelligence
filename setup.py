#!/usr/bin/env python3
"""
Setup configuration for automata-theory library
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="automata-theory",
    version="1.0.0",
    author="Scott Weeden",
    author_email="scott.weeden@example.edu",
    description="A comprehensive library for learning automata theory with visceral undecidability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scott-weeden/automata-theory",
    project_urls={
        "Bug Tracker": "https://github.com/scott-weeden/automata-theory/issues",
        "Documentation": "https://automata-theory.readthedocs.io",
        "Source Code": "https://github.com/scott-weeden/automata-theory",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "graphviz>=0.20",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "pytest-timeout>=2.0.0",
        "pytest-mock>=3.0.0",
        "colorama>=0.4.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "isort>=5.0.0",
            "pre-commit>=2.0.0",
            "tox>=4.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
        "visualization": [
            "matplotlib>=3.0.0",
            "networkx>=2.0.0",
            "pygraphviz>=1.0.0",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "gunicorn>=20.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automata=automata_cli:cli",
            "automata-test=automata_test:main",
            "automata-viz=automata_viz:main",
        ],
    },
    include_package_data=True,
    package_data={
        "automata_theory": [
            "examples/*.json",
            "templates/*.html",
            "static/*.css",
            "data/*.txt",
        ],
    },
    zip_safe=False,
    test_suite="tests",
    tests_require=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],

)