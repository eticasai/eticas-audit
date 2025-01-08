


### `setup.py`

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
INSTALL_REQUIRES = [
      'scikit-learn==1.5.2','bnlearn==0.10.2','networkx==3.4.2','matplotlib==3.9.2','pgmpy==0.1.26','numpy==1.26.4', 'pandas==2.2.3','scipy==1.11.4','statsmodels==0.14.4'
      ]


setup(
    name="eticas",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for calculating fairness metrics in ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/eticas",
    
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.11.9',
)

