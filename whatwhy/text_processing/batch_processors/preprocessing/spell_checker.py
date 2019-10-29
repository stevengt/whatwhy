import os
import shutil
import subprocess
import logging
import tarfile
import requests
import jamspell
from whatwhy import RESOURCES_FOLDER

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

JAMSPELL_RESOURCES_FOLDER = os.path.join(RESOURCES_FOLDER, "jamspell")
JAMSPELL_MODEL_FILE_NAME = os.path.join(JAMSPELL_RESOURCES_FOLDER, "en.bin")

def get_spell_checker():    
    assert_swig3_is_installed()
    if not is_jamspell_language_model_downloaded():
        download_jamspell_language_model()
    spell_checker = jamspell.TSpellCorrector()
    spell_checker.LoadLangModel(JAMSPELL_MODEL_FILE_NAME)
    return spell_checker

def create_resources_folder_if_does_not_exist():
    if not os.path.isdir(RESOURCES_FOLDER):
        os.mkdir(RESOURCES_FOLDER)
    if not os.path.isdir(JAMSPELL_RESOURCES_FOLDER):
        os.mkdir(JAMSPELL_RESOURCES_FOLDER)

def assert_swig3_is_installed():
    is_swig3_installed = False
    swig3_executable_names = ["swig", "swig3.0", "swig.exe", "swig3.0.exe"]
    for swig3_executable_name in swig3_executable_names:
        if shutil.which(swig3_executable_name) is not None:
            is_swig3_installed = True
    error_message = "swig3.0 library was not found. Please install using the installation instructions at https://github.com/swig/swig/wiki/Getting-Started"
    assert is_swig3_installed, error_message

def is_jamspell_language_model_downloaded():
    if os.path.exists(JAMSPELL_MODEL_FILE_NAME):
        return True
    logger.warning("jamspell language model file was not found.")
    return False

def download_jamspell_language_model():
    logger.info("Downloading jamspell language model")
    create_resources_folder_if_does_not_exist()

    model_url = "https://github.com/bakwc/JamSpell-models/raw/master/en.tar.gz"
    tar_file_name = os.path.join(JAMSPELL_RESOURCES_FOLDER, "en.tar.gz")
    with requests.get(model_url, stream=True) as compressed_model:
        with open(tar_file_name, "wb") as tar_file:
            tar_file.write(compressed_model.content)

    with tarfile.open(tar_file_name) as tar_file:
        tar_file.extractall(path=JAMSPELL_RESOURCES_FOLDER)
    os.remove(tar_file_name)
