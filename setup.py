# `setup.py`
from setuptools import setup, find_packages
import os
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="eticas",
    version="0.1.0",
    author="Eticas AI",
    author_email="it@eticas.ai",
    description="A Python library for calculating fairness metrics in ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eticasai/eticas-audit",
    install_requires=[
        "numpy==2.1.2",
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "pyarrow==19.0.0",
        "scipy==1.15.1",
        "coverage==7.6.10",
        "requests==2.32.3"
    ],
    packages=find_packages(include=['eticas', 'eticas.*'])
)

