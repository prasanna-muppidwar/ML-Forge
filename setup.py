from setuptools import setup, find_packages

setup(
    name='AlgoKit',
    version='0.1.0',
    author='Prasanna Muppidwar',
    author_email='prasannamuppidwar16@gmail.com',
    description='An Opensource easy to use Machine Learning Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_library',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
