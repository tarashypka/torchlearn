from setuptools import setup, find_packages


setup(
    name='torchlearn',
    version='0.1',
    description='Pointless sketches with pytorch',
    author='Taras Shypka',
    author_email='tarashypka@gmail.com',
    packages=find_packages(exclude=['tests', 'tests.*']),
    python_requires='>=3.6',
    test_suite='tests')
