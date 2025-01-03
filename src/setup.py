# src/setup.py

import setuptools
from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lstm-model-trainer",  # Replace with your desired package name
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A PyQt5 application for training and evaluating machine learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dadudekc/lstm-model-trainer",  # Replace with your repository URL
    packages=find_packages(),
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    install_requires=[
        "PyQt5>=5.15.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
        "matplotlib>=3.0.0",
        "numpy>=1.18.0",
        "tensorflow>=2.0.0",
        "keras-tuner>=1.0.0"
    ],
    entry_points={
        'console_scripts': [
            'pyqt-model-trainer=training_app.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
