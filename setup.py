import os
from setuptools import setup, find_packages


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="noisy-ml",
    version="0.1dev",
    license="Apache License 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="Learning from Noisy Labels",
    url="https://github.com/eaplatanios/noisy-labels",
    install_requires=[
        "bert-serving-client",
        "click",
        "nltk",
        "numpy>=1.5",
        "pandas",
        "pyyaml",
        "scikit-learn>=0.20.1",
        "six",
        "tensorflow>=1.11",
        "tqdm",
        "xlrd",
    ],
)
