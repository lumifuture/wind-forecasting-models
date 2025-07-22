from setuptools import setup, find_packages

setup(
    name="wind-models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0"
    ],
    description="Shared PyTorch models for wind power forecasting",
    author="lumifuture",
)
