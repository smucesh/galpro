from setuptools import setup

setup(
    name='galpro',
    version='0.1.9',
    packages=['galpro'],
    install_requires=['numpy', 'scikit-learn', 'joblib', 'h5py',
                      'matplotlib', 'statsmodels', 'seaborn', 'scipy', 'pandas'],
    url='https://galpro.readthedocs.io/',
    license='MIT License',
    author='Sunil Mucesh',
    author_email='smucesh@hotmail.co.uk',
    description='Machine learning galaxy properties and redshift',
    project_urls={
        "readthedocs": "https://galpro.readthedocs.io",
        "GitHub": "https://github.com/smucesh/galpro",
    }
)
