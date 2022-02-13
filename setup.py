from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='ds2l_som',
   version='1.0.0',
   description='DS2L-SOM is a Local Density-based Simultaneous Two-level Algorithm for Topographic Clustering',
   long_description=long_description,
   author='Sandro Martens',
   packages=['ds2l_som'],
   install_requires=['numpy', 'networkx', "Pandas", "MiniSom", "scikit-learn"],
)
