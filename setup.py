from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()


def read_readme():
    with open("README.rst", encoding="utf-8") as f:
        return f.read()


setup(
    name='sefef',
    version='2.3.1',
    license="BSD 3-clause",
    description='SeFEF: Seizure Forecast Evaluation Framework',
    long_description=read_readme(),
    long_description_content_type="text/x-rst",
    readme="README.rst",
    author="Ana Sofia Carmo",
    author_email="anascacais@gmail.com",
    packages=find_packages(include=['sefef', 'sefef.*']),
    install_requires=required,
    setup_requires=['pytest-runner', 'flake8'],
    test_suite="tests",
    tests_require=['pytest'],
)
