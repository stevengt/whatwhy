from setuptools import setup, find_packages

setup(
    name="whatwhy",
    version="0.0.1",
    description="A collection of scripts for predicting the 'why' of a sentence.",
    url="https://github.com/stevengt/whatwhy",
    author="Steven Thomas",
    author_email="stevent3115@gmail.com",
    python_requires='>=3.6',
    install_requires=[
        "boto3 >= 1.10.2",
        "dask >= 2.6.0",
        "giveme5w1h >= 1.0.17",
        "jamspell >= 0.0.11",
        "numpy >= 1.17.2",
        "pandas >= 0.25.1",
        "twython >= 3.7.0",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "whatwhy = whatwhy.text_processing.main:main"
        ]
    }
)