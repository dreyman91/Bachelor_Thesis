# setup.py
from setuptools import setup, find_packages

setup(
    name='failure-api',
    version='0.1.0',
    description='A modular communication failure API for Multi-Agent Reinforcement Learning environments',
    author='Oluwadamilare Adegun',
    author_email='adegun_dare@icloud.com',
    url='https://github.com/dreyman91/Bachelor_Thesis/tree/main/Failure_API/src/failure_api',  # Optional
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'pettingzoo',
        'gymnasium',
        'scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
