from setuptools import setup


setup(
    name='torchlearn',
    version='0.1',
    description='Pointless sketches with pytorch',
    author='Taras Shypka',
    author_email='tarashypka@gmail.com',
    packages=['torchlearn', 'torchlearn.utils', 'torchlearn.vectorizer'],
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.14.3',
        'torch==0.4.1',
        'torchtext==0.3.0',
        'dill==0.2.8.2',
        'scikit-learn==0.19.1'
    ],
    test_suite='tests'
)