from distutils.core import setup

setup(
    name="cross-learn",
    version="1.01",
    description="extensive scoring of crossvalidation loops.",
    author="jhn-nt",
    packages=["crlearn"],
    package_data={"crlearn": ["config.json"]},
    install_requires=[
        "scikit-learn>=1.2.1",
        "tqdm>=4.64.1",
        "scipy>=1.10.1",
    ],
)
