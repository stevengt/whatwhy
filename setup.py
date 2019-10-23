from setuptools import setup, find_packages

setup(
    name="whatwhy",
    version="0.0.1",
    description="A collection of scripts for predicting the 'why' of a sentence.",
    url="https://github.com/stevengt/whatwhy",
    author="Steven Thomas",
    author_email="stevent3115@gmail.com",
    python_requires='>=3.7',
    install_requires=[
        "dask >= 2.6.0",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz",
        "Flask >= 1.1.1",
        "giveme5w1h >= 1.0.17",
        "numpy >= 1.17.2",
        "pandas >= 0.25.1",
        "spacy >= 2.2.1",
        "twython >= 3.7.0",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "wh-phrase-extractor-server = whatwhy.webservices.wh_phrase_extractor.server:main"
        ]
    }
)