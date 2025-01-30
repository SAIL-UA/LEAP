from setuptools import setup, find_packages

setup(
    name='LEAP',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib'],  # Specify dependencies
    url='https://github.com/SAIL-UA/LEAP',
    author='Xishi Zhu',
    author_email='xzhu39@crimson.ua.edu',
    description='A package for loading sensor data',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
