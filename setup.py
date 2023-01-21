from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ml_tools",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    description='Common Machine Learning Tools',
    url='https://github.com/Clayrisee/ml-tools',
    author=['Haikal Ardikatama'])
