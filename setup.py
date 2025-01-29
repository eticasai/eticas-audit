# `setup.py`
from setuptools import setup
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
INSTALL_REQUIRES = [
      'numpy==2.1.2', 'pandas==2.2.3', 'scikit-learn==1.5.2', 'pyarrow==19.0.0'
      ]

setup(
    name="eticas",
    version="0.1.0",
    author="Eticas AI",
    author_email="it@eticas.ai",
    description="A Python library for calculating fairness metrics in ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eticasai/eticas-audit",
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.11.9',
)

