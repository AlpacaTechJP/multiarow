from setuptools import setup, find_packages

setup(
    name="multiarow",
    version="0.1",
    url="https://github.com/AlpacaDB/multiarow",
    description="Multiclass AROW implementation",
    author="AlpacaDB, Inc.",
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='online learning multiclass',
    packages=find_packages(),
    install_requires=['numpy'],
)
