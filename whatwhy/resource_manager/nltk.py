import os
import logging
import nltk
from whatwhy import get_resources_folder

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def configure_nltk():
    """Downloads any required NLTK data if not already downloaded."""
    nltk_resources_folder = get_nltk_resources_folder()
    nltk.data.path.append(nltk_resources_folder)

    try:
        nltk.data.find("tokenizers/punkt")
    except:
        logger.warning("NLTK punkt tokenizer was not found.")
        logger.info("Downloading NLTK punkt tokenizer.")
        nltk.download("punkt", download_dir=nltk_resources_folder)

    try:
        nltk.data.find("corpora/wordnet")
    except:
        logger.warning("NLTK wordnet corpus was not found.")
        logger.info("Downloading NLTK wordnet corpus.")
        nltk.download("wordnet", download_dir=nltk_resources_folder)

    try:
        nltk.data.find("corpora/stopwords")
    except:
        logger.warning("NLTK stopwords corpus was not found.")
        logger.info("Downloading NLTK stopwords corpus.")
        nltk.download("stopwords", download_dir=nltk_resources_folder)

def get_nltk_resources_folder():
    nltk_resources_folder = os.path.join(get_resources_folder(), "nltk")
    if not os.path.isdir(nltk_resources_folder):
        os.mkdir(nltk_resources_folder)
    return nltk_resources_folder
