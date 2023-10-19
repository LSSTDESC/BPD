from setuptools import find_packages, setup

setup(
    name="bpd",
    version="0.0.1",
    url="https://github.com/LSSTDESC/BPD.git",
    author="Ismael Mendoza",
    author_email="imendoza@umich.edu",
    description="Bayesian Pixel Domain inference of galaxy properties",
    packages=find_packages(),
    install_requires=["numpy", "jax", "galsim"],
)
