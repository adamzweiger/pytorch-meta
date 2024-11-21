from setuptools import setup, find_packages
from os import path
import sys

from io import open

extras = {
    'tcga': ['pandas>=1.0.0', 'academictorrents>=2.1.0', 'six>=1.11.0'],
    'test': ['flaky', 'pytest']
}

here = path.abspath(path.dirname(__file__))

sys.path.insert(0, path.join(here, 'torchmeta'))
from version import VERSION

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torchmeta',
    version=VERSION,
    description='Dataloaders for meta-learning in Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author='Tristan Deleu',
    author_email='tristan.deleu@gmail.com',
    url='https://github.com/tristandeleu/pytorch-meta',
    keywords=['meta-learning', 'pytorch', 'few-shot', 'few-shot learning'],
    packages=find_packages(exclude=['data', 'contrib', 'docs', 'tests', 'examples']),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.20.0',
        'Pillow>=9.0.0',
        'h5py>=3.0.0',
        'tqdm>=4.50.0',
        'requests>=2.25.0',
        'ordered-set>=4.0.0'
    ],
    extras_require=extras,
    package_data={'torchmeta': ['torchmeta/datasets/assets/*']},
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
    ],
)
