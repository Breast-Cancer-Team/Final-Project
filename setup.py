# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='Final-Project', 
    version='1.0.0',
    description='Diagnoses breast cancer samples as benign or malignant',  
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/Breast-Cancer-Team/Final-Project',  
    author='Wensy Chan, Kevin Cao, Keiton Guan, Sarah Wait', 
    author_email='wensy22@uw.edu, kevcao22@uw.edu, keitong@uw.edu, swait@uw.edu',  

    classifiers=[  
        'Intended Audience :: Pathologists',
        'Topic :: Scientific/Engineering :: Medical Science Apps',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.8.5',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.8.5',
    install_requires=[
        "ipython>=7.21.0"
        "matplotlib>=3.3.4",
        "numpy>=1.18.1",
        "pandas>=0.25.3",
        "pytest>=5.3.4",
        "pytest-cov>=2.8.1",
        "seaborn>=0.11.1"
        "scikit-learn>=0.24.1"
    tests_require={
        'pytest',
        'pytest-cov'
    },
    project_urls={ 
        'Kaggle Dataset': 'https://www.kaggle.com/uciml/breast-cancer-wisconsin-data',
    },
)