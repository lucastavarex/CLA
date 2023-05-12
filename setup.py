from setuptools import setup, find_packages
from setuptools.extension import Extension
import subprocess
import sys

def install(package):
  subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
  from Cython.Build import cythonize
except:
  install('cython')
  from Cython.Build import cythonize

extensions = [
  Extension(
    "cla.*",
    ["cla/*.py"],
  ),
]

setup(
  name = 'CLA',
  version = '0.1.0',
  license='GPL-3.0',
  professor='Luis Sagrilo',
  description = 'Código desenvolvido para o curso de Álgebra Linear Computacional @ UFRJ',
  packages=find_packages(),
  author = 'Lucas Tavares Da Silva Ferreira',
  author_email = 'lucas.tavares@poli.ufrj.br',
  url = 'https://github.com/lucastavarex/CLA/',
  install_requires=[
    'Cython>=0.29.17',
    'sympy==1.6.2',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
