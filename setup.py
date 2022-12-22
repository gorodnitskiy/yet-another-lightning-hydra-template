from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    description=(
        "Deep Learning training pipeline template"
        "based on pytorch_lightning and hydra"
    ),
    author="Your Name",
    author_email="Your Email",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(exclude=["tests"]),
)
