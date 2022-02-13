from setuptools import setup


with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='DBGSOM',
   version='1.0.0',
   description='A directed batch growing approach to enhance the topology preservation of self-organizing maps ',
   long_description=long_description,
   author='Sandro Martens',
   packages=['DBGSOM'],
   install_requires=['numpy', 'networkx', 'scipy'],
)
