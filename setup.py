from setuptools import setup, find_packages

setup(
    name="seqrecord",
    version="0.1.8",
    description="various bugs fix",
    author_email="shuhang0chen@gmail.com",
    maintainer_email="shuhang0chen@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "torch", "torchdata", "aiofiles"],
)
