"""
Setup script for Intelligent Systems project.
Implements core AI algorithms: search, games, MDPs, reinforcement learning.
"""

from setuptools import setup, find_packages

setup(
    name="intelligent-systems",
    version="1.0.0",
    description="AI algorithms for search, games, and decision making",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0"
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "jupyter>=1.0.0"],
        "viz": ["seaborn>=0.11.0", "networkx>=2.6.0"]
    }
)
