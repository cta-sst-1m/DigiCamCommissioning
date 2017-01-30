from distutils.core import setup

setup(
    name='DigiCamCommissioning',
    version='v0.0.1',
    packages=['utils', 'spectra_fit','data_treatement'],
    url='https://github.com/cocov/DigiCamCommissioning',
    license='GNU GPL 3.0',
    author='cocov',
    author_email='victor.coco@cern.ch',
    description='', requires=['numpy', 'ctapipe', 'yaml']
)
