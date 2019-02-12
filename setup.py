from setuptools import setup

setup(name='metricscreen',
      version='0.1',
      description='A package for nonlinear feature selection.',
      url='http://github.com/keliliu/metricscreen',
      author='Keli Liu',
      author_email='keliliu@stanford.edu',
      packages=['metricscreen'],
      install_requires=[
          'numpy>=1.14',
          'scipy>=1.1'
      ],
      zip_safe=False)