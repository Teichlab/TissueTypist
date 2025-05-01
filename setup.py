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
        'matplotlib',
        'seaborn',
        'statsmodels',
        'scipy',
        'scikit-learn',
        'scanpy',
        'squidpy',
        'spatialdata',
        'joblib'
    ],
    include_package_data=True, # picks up MANIFEST.in
    author='Krzysztof Polanski, Kazumasa Kanemaru',
    # author_email='kp9@sanger.ac.uk',
    # license='non-commercial license'
)
