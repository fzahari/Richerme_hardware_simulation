"""
Setup configuration for Richerme Quantum Hardware library.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="richerme-quantum-hardware",
    version="1.0.0",
    author="Federico Zahariev",
    author_email="fzahari@iastate.edu",
    description="Python library for synthesizing quantum gates on trapped-ion analog hardware",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/federicozahariev/richerme-quantum-hardware",
    packages=find_packages(),
    py_modules=["richerme_ion_analog", "rich_sim", "rich_sim_h2", "rich_sim_h2o"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
        ],
        "qiskit": [
            "qiskit>=0.34.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "richerme-demo=examples.demo_zxx:main",
        ],
    },
    keywords="quantum computing, trapped ions, gate synthesis, quantum simulation",
    project_urls={
        "Bug Reports": "https://github.com/federicozahariev/richerme-quantum-hardware/issues",
        "Source": "https://github.com/federicozahariev/richerme-quantum-hardware",
        "Documentation": "https://github.com/federicozahariev/richerme-quantum-hardware/blob/main/README.md",
    },
)