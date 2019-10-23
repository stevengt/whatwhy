import setuptools

setuptools.setup(
    name="whatwhy",
    version="0.0.1",
    author="Steven Thomas",
    author_email="stevent3115@gmail.com",
    description="A collection of scripts for predicting the 'why' of a sentence.",
    url="https://github.com/stevengt/whatwhy",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    entry_points={
        "console_scripts": [
            "wh-phrase-extractor-server = whatwhy.webservices.wh_phrase_extractor.server:main"
        ]
    }
)