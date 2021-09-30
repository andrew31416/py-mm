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
      packages=find_packages(include=['src','src.*']),
      package_dir={'pymm': 'src.pymm'},
      classifiers=['Development Status :: 1 - Planning']
      )