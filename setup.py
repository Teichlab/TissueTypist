from setuptools import setup, find_packages

setup(
    name='TissueTypist',
    version='0.0.1',
    description='Tissue label transfer functions',
    url='https://github.com/kazukane/TissueTypist',
    packages=find_packages(exclude=['docs', 'notebooks']),
    install_requires=[
        'anndata',
        'pandas',
        'numpy',
        'statsmodels',
        'scipy',
        'scikit-learn',
        'scanpy',
        'squidpy'
    ],
    # package_data={
    #     "drug2cell": ["*.pkl"]
    # },
    author='Krzysztof Polanski, Kazumasa Kanemaru',
    # author_email='kp9@sanger.ac.uk',
    # license='non-commercial license'
)
