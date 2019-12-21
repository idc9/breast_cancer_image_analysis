from setuptools import setup, find_packages

# the analysis was done with Python version 3.7.2.
# glob, os, shutil, datetime, codecs, copy, numbers
# TODO: explore

install_requires = ['numpy==1.15.4',
                    'matplotlib==3.0.2',
                    'pandas==0.24.0',
                    'scipy==1.2.0',
                    'sklearn==0.20.2',
                    'torch==1.0.0',
                    'torchvision==0.2.1',
                    'statsmodels==0.9.0',
                    'json==2.0.9',
                    'imageio==2.4.1',
                    'PIL==5.4.1',
                    'scikit-image==0.14.1'
                    'joblib==0.13.1',
                    'tqdm==4.41.0',
                    'csv==1.0',
                    'jive==0.2.0']

setup(name='cbcs_joint',
      version='0.0.1',
      description='Code to reproduce Joint and individual analysis of breast cancer histologic images and genomic covariates',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
