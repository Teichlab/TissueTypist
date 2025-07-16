from setuptools import setup, find_packages

setup(
    name='TissueTypist',
    version='0.0.1',
    description='Tissue label transfer functions',
    url='https://github.com/Teichlab/TissueTypist',
    packages=find_packages(exclude=['docs', 'notebooks']),
    install_requires=[
        'anndata',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'statsmodels',
        'scipy',
        'scikit-learn==1.5.2',
        'scanpy==1.11.2',
        'squidpy==1.6.5',
        'spatialdata',
        'joblib'
    ],
    include_package_data=True, # picks up MANIFEST.in
    author='Kazumasa Kanemaru, Krzysztof Polanski',
    # author_email='kp9@sanger.ac.uk',
    # license='non-commercial license'
)
