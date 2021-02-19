from setuptools import setup

setup(
    name='galpro',
    version='0.1.1',
    packages=['galpro'],
    install_requires=['numpy', 'scikit-learn>=0.22.1', 'joblib', 'h5py',
                      'matplotlib', 'statsmodels==0.11.0', 'seaborn==0.10.1', 'scipy>=1.6.0', 'pandas'],
    url='https://galpro.readthedocs.io/',
    license='MIT License',
    author='Sunil Mucesh',
    author_email='smucesh@hotmail.co.uk',
    description='Machine learning galaxy properties',
    project_urls={
        "readthedocs": "https://galpro.readthedocs.io",
        "GitHub": "https://github.com/smucesh/galpro",
        "arXiv": "https://arxiv.org/abs/2012.05928"
    }
)
