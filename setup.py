from setuptools import setup, find_packages

setup(
    name='sefef',
    version='0.0.1',
    license="MIT license",
    description='SeFEF: A Seizure Forecast Evaluation Framework',
    readme="README.md",
    author="Ana Sofia Carmo",
    author_email="anascacais@gmail.com",
    packages=find_packages(include=['sefef', 'sefef.*']),
    setup_requires=['pytest-runner', 'flake8'],
    test_suite="tests",
    tests_require=['pytest'],
)
