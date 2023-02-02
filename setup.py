from distutils.core import setup

setup(
    name="jhn_ai",
    version="2.0",
    description="extended validation of sklearn models",
    author="jhn-nt",
    packages=["jhn_ai"],
    install_requires=[
        "scikit-learn>=1.2.1",
        "imbalanced-learn>=0.10.1",
        "numpy>=1.23.3",
        "pandas",
        "tqdm",
        "scipy>=1.10.0",
    ],
)
