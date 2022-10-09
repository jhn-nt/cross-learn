from distutils.core import setup

setup(
    name="jhn_ai",
    version="1.1",
    description="extended validation of sklearn models",
    author="Giovanni Angelotti, MSc",
    packages=["jhn_ai"],
    install_requires=["sklearn", "imblearn", "numpy>=1.23.3", "pandas", "tqdm"],
)
