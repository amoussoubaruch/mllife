from setuptools import setup, find_packages

setup(
    name='mllife',
    version='1.0',
    packages=find_packages(),
    install_requires=
        [
            "scikit-learn==0.24.2",
            "plotly==4.7.1",
            "mlflow==1.8.0",
            "google.cloud.storage==1.32.0",
            "pandas==1.3.5",

        ],
    include_package_data=True,    
    description='MLFlow Usage package',
    author='Baruch Amoussou-djangban',
    license='MIT'
)
