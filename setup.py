#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as req_file:
    requirements = req_file.read()


setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Zachary R Uhlmann",
    author_email='zach.uhlmann@usda.gov',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="visual and graphical diagnostic of repeat 3D array collections - i.e. satellite imagery - for quick mapping of potentially erroneous data",
    entry_points={
        'console_scripts': [
            'raqc=raqc.cli:main'
        ]
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    package_data={'raqc':['CoreConfig.ini', 'recipes.ini']},
    keywords='raqc',
    name='raqc',
#    packages=find_packages(include=['raqc']),
    packages=['raqc'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zuhlmann/raqc',
    version='0.1.1',
    zip_safe=False,
)
