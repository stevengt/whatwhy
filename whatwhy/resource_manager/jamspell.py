import os
import shutil
import logging
import tarfile
import requests
from whatwhy import get_resources_folder
from .nltk import configure_nltk

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def get_jamspell_model_file_name():
    configure_jamspell()
    jamspell_model_file_name = os.path.join(get_jamspell_resources_folder(), "en.bin")
    if not os.path.exists(jamspell_model_file_name):
        logger.warning("jamspell language model file was not found.")
        download_jamspell_language_model()
    return jamspell_model_file_name

def configure_jamspell():
    assert_swig3_is_installed()
    configure_nltk()

def get_jamspell_resources_folder():
    jamspell_resources_folder = os.path.join(get_resources_folder(), "jamspell")
    if not os.path.isdir(jamspell_resources_folder):
        os.mkdir(jamspell_resources_folder)
    return jamspell_resources_folder

def assert_swig3_is_installed():
    is_swig3_installed = False
    swig3_executable_names = ["swig", "swig3.0", "swig.exe", "swig3.0.exe"]
    for swig3_executable_name in swig3_executable_names:
        if shutil.which(swig3_executable_name) is not None:
            is_swig3_installed = True
    error_message = "swig3.0 library was not found. Please install using the installation instructions at https://github.com/swig/swig/wiki/Getting-Started"
    assert is_swig3_installed, error_message

def download_jamspell_language_model():
    logger.info("Downloading jamspell language model")
    model_url = "https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz"

    tar_file_name = os.path.join(get_jamspell_resources_folder(), "en.tar.gz")
    with requests.get(model_url, stream=True) as compressed_model:
        with open(tar_file_name, "wb") as tar_file:
            tar_file.write(compressed_model.content)

    with tarfile.open(tar_file_name) as tar_file:
        tar_file.extractall(path=get_jamspell_resources_folder())
    os.remove(tar_file_name)
