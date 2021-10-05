"""
To upload the package to PyPi, run

> python setup.py sdist bdist_wheel

This will generate 2 files:
- dist
    - ?.whl
    - ?.tar.gaz

To test the distribution, execute

>python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

which can be installed locally via

>python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps
         <module_name>

To complete a local install from setup.py, execute
>python setup.py install        - to get .egg file
>ptrgon setup.py install_lib    - to get copies of all .py files and the
                                - module directories.
"""
from setuptools import setup, find_packages


def readme():
    # for long description
    with open('README.md') as f:
        return f.read()


module_name = 'pymm'

setup(name=module_name,
      version='0.0',
      description='Markov models for categorical data',
      long_description=readme(),
      url='https://github.com/andrew31416/py-mm',
      license='MIT',
      install_requires=['numpy',
                        'scipy',
                        'typing'
                        ],
      packages=find_packages(),
      classifiers=['Development Status :: 1 - Planning']
      )
