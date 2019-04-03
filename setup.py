from setuptools import setup


setup(
    name='torchlearn',
    version='0.1',
    description='Pointless sketches with pytorch',
    author='Taras Shypka',
    author_email='tarashypka@gmail.com',
    packages=['torchlearn', 'torchlearn.utils', 'torchlearn.vectorizer'],
    python_requires='>=3.6',
    test_suite='tests'
)