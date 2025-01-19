#formatting for setup inspired by Objax repository: https://github.com/google/objax/blob/master/setup.py

import re
from pkg_resources import parse_requirements
from setuptools import find_packages, setup

README_FILE = 'README.md'
REQUIREMENTS_FILE = 'requirements.txt'
VERSION_FILE = '_version.py'

VERSION_REGEXP = r'^__version__ = \'(\d+\.\d+\.\d+)\''
r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
   raise RuntimeError(f'Unable to find version string in {VERSION_FILE}.')
version = r.group(1)

long_description = open(README_FILE, encoding='utf-8').read()

install_requires = [str(r) for r in parse_requirements(open(REQUIREMENTS_FILE, 'rt'))]

name = 'Emails_Business_Personal'

setup(
    name=name,
    version=version,
    description='Emails classification on business and personal categories',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author='Milena Sosic',
    author_email='milena.sosic@gmail.com',
    packages=find_packages(),
    install_requires=install_requires
)

#create the data directories
import os
modules = [
   f for f in os.listdir(name)
   if os.path.isdir(os.path.join(name,f))
]
for fname in modules:
   fpath = os.path.join(name,fname)
   module_flag = os.path.isdir(fpath)
   if module_flag:
      subfiles = os.listdir(fpath)

