#!/usr/bin/env python

__version__ = '0.9.8'

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['lognflow']

test_requirements = ['pytest>=3.7', ]

setup(
    author="Alireza Sadri",
    author_email='Alireza.Sadri@monash.edu',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    description="State of the art analysis tools for 4D STEM",
    entry_points={
        'console_scripts': [
            'mcemtools=mcemtools.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mcemtools',
    name='mcemtools',
    packages=find_packages(include=['mcemtools', 'mcemtools.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/arsadri/mcemtools',
    version = __version__,
    zip_safe=False,
)