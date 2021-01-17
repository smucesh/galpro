from setuptools import setup

setup(
    name='galpro',
    version='0.2.7',
    packages=['galpro'],
    install_requires=['numpy', 'scikit-learn>=0.22.1', 'joblib', 'h5py',
                      'matplotlib', 'statsmodels', 'seaborn', 'scipy>=1.5.2', 'pandas'],
    url='https://galpro.readthedocs.io/',
    license='MIT License',
    author='Sunil Mucesh',
    author_email='smucesh@hotmail.co.uk',
    description='Machine learning galaxy properties',
    project_urls={
        "readthedocs": "https://galpro.readthedocs.io",
        "GitHub": "https://github.com/smucesh/galpro",
    }
)
