from distutils.core import setup

setup(
    name="cross-learn",
    version="1.0",
    description="extensive scoring of crossvalidation loops.",
    author="jhn-nt",
    packages=["crlearn"],
    package_data={"crlearn": ["config.json"]},
    install_requires=[
        "scikit-learn>=1.2.1",
        "numpy<=1.24,>=1.18",
        "pandas>=1.4.3",
        "tqdm>=4.64.1",
    ],
)
