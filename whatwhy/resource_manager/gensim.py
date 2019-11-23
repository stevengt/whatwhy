import os
import shutil
import logging
import zipfile
import gzip

import requests
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from whatwhy import get_resources_folder

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def get_custom_word2vec_model():
    gensim_resources_folder = get_gensim_resources_folder()
    model_file_name = os.path.join(gensim_resources_folder, "whatwhy-custom.kv")
    if not os.path.exists(model_file_name):
        raise FileNotFoundError("Custom word2vec model was not found.")
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file_name)
    return model

def create_and_save_word2vec_model( tokens_lists,
                                    embedded_vector_size=100,
                                    min_token_count=10,
                                    window=10,
                                    workers=10,
                                    iter=10 ):

    gensim_resources_folder = get_gensim_resources_folder()
    model_file_name = os.path.join(gensim_resources_folder, "whatwhy-custom.kv")

    model = gensim.models.Word2Vec( tokens_lists,
                                    size=embedded_vector_size,
                                    window=window,
                                    min_count=min_token_count,
                                    workers=workers,
                                    iter=iter)
    if os.path.exists(model_file_name):
        os.remove(model_file_name)
    model.wv.save_word2vec_format(model_file_name)

def get_google_news_model():
    gensim_resources_folder = get_gensim_resources_folder()
    model_file_name = os.path.join(gensim_resources_folder, "GoogleNews-vectors-negative300.bin")
    if not os.path.exists(model_file_name):
        logger.warning(f"{model_file_name} model file was not found.")
        download_google_news_model()
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file_name, binary=True)
    return model

def download_google_news_model():
    logger.info("Downloading Google-News model.")
    model_url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

    file_name = os.path.join(get_gensim_resources_folder(), "GoogleNews-vectors-negative300.bin")
    zip_file_name = os.path.join(get_gensim_resources_folder(), "GoogleNews-vectors-negative300.bin.gz")

    with requests.get(model_url, stream=True) as compressed_model:
        with open(zip_file_name, "wb") as zip_file:
            zip_file.write(compressed_model.content)

    with gzip.open(zip_file_name, "rb") as zip_file:
        with open(file_name, "wb") as model:
            shutil.copyfileobj(zip_file, model)
    
    os.remove(zip_file_name)

def get_glove_wiki_gigaword_model(num_dimensions):
    valid_dimensions = (50, 100, 200, 300)
    if num_dimensions not in valid_dimensions:
        raise ValueError(f"num_dimensions must be in {valid_dimensions}.")
    
    file_name = f"word2vec.glove.6B.{num_dimensions}d.txt"
    gensim_resources_folder = get_gensim_resources_folder()
    model_file_name = os.path.join(gensim_resources_folder, file_name)
    if not os.path.exists(model_file_name):
        logger.warning(f"{model_file_name} model file was not found.")
        download_glove_wiki_gigaword_models()
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file_name)
    return model

def get_gensim_resources_folder():
    gensim_resources_folder = os.path.join(get_resources_folder(), "gensim")
    if not os.path.isdir(gensim_resources_folder):
        os.mkdir(gensim_resources_folder)
    return gensim_resources_folder

def download_glove_wiki_gigaword_models():
    logger.info("Downloading glove-wiki-gigaword models.")
    model_url = "http://nlp.stanford.edu/data/glove.6B.zip"

    zip_file_name = os.path.join(get_gensim_resources_folder(), "glove.6B.zip")
    with requests.get(model_url, stream=True) as compressed_model:
        with open(zip_file_name, "wb") as zip_file:
            zip_file.write(compressed_model.content)

    with zipfile.ZipFile(zip_file_name, "r") as zip_file:
        zip_file.extractall(get_gensim_resources_folder())
    os.remove(zip_file_name)

    glove_file_names = ["glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt", "glove.6B.300d.txt"]
    for glove_file_name in glove_file_names:
        full_glove_file_path = os.path.join(get_gensim_resources_folder(), glove_file_name)
        word2vec_file_name = os.path.join(get_gensim_resources_folder(), "word2vec." + glove_file_name)
        glove2word2vec(full_glove_file_path, word2vec_file_name)
