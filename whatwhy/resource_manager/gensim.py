import os
import logging
import zipfile
import requests
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from whatwhy import get_resources_folder

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def get_glove_wiki_gigaword_50_model():
    gensim_resources_folder = get_gensim_resources_folder()
    model_file_name = os.path.join(gensim_resources_folder, "glove.6B.50d.word2vec.txt")
    if not os.path.exists(model_file_name):
        logger.warning("glove-wiki-gigaword-50 model file was not found.")
        download_glove_wiki_gigaword_50_model()
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file_name)
    return model

def get_gensim_resources_folder():
    gensim_resources_folder = os.path.join(get_resources_folder(), "gensim")
    if not os.path.isdir(gensim_resources_folder):
        os.mkdir(gensim_resources_folder)
    return gensim_resources_folder

def download_glove_wiki_gigaword_50_model():
    logger.info("Downloading glove-wiki-gigaword-50 model")
    model_url = "http://nlp.stanford.edu/data/glove.6B.zip"

    zip_file_name = os.path.join(get_gensim_resources_folder(), "glove.6B.zip")
    with requests.get(model_url, stream=True) as compressed_model:
        with open(zip_file_name, "wb") as zip_file:
            zip_file.write(compressed_model.content)

    with zipfile.ZipFile(zip_file_name, "r") as zip_file:
        zip_file.extractall(get_gensim_resources_folder())
    os.remove(zip_file_name)

    glove_file_name = os.path.join(get_gensim_resources_folder(), "glove.6B.50d.txt")
    word2vec_file_name =  os.path.join(get_gensim_resources_folder(), "glove.6B.50d.word2vec.txt")
    glove2word2vec(glove_file_name, word2vec_file_name)
