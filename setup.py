from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='pyt_splade',
    version='0.0.2',
    description='PyT wrapper for SPLADE',
    url='https://github.com/cmacdonald/pyt_splade',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=['pyt_splade'] + ['pyt_splade.' + i for i in find_packages('pyt_splade')],
    # as per splade
    include_package_data=True,
    license="Creative Commons Attribution-NonCommercial-ShareAlike",
    long_description=readme,
    install_requires=[
        'splade', 'python-terrier>=0.11.0', 'pyterrier_alpha',
    ],
)