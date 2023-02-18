from distutils.core import setup

setup(
    name="jhn_ai",
    version="2.0",
    description="extended validation of sklearn models",
    author="jhn-nt",
    packages=["jhn_ai"],
    package_data={"jhn_ai": ["*.json"]},
    install_requires=[
        "scikit-learn>=1.2.1",
        "numpy>=1.23.3",
        "pandas>=1.4.3",
        "tqdm>=4.64.1",
    ],
)
