


### `setup.py`

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eticas",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for calculating fairness metrics in ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/eticas",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)

