import os

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESOURCES_FOLDER = os.path.join(PACKAGE_DIRECTORY, "resources")
QUESTION_WORDS = ["who", "what", "when", "where", "why", "how"]

def get_resources_folder():
    if not os.path.isdir(RESOURCES_FOLDER):
        os.mkdir(RESOURCES_FOLDER)
    return RESOURCES_FOLDER

def get_nltk_resources_folder():
    nltk_resources_folder = os.path.join(get_resources_folder(), "nltk")
    if not os.path.isdir(nltk_resources_folder):
        os.mkdir(nltk_resources_folder)
    return nltk_resources_folder

def configure_nltk():
    import nltk
    nltk_resources_folder = get_nltk_resources_folder()
    nltk.data.path.append(nltk_resources_folder)

    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt", download_dir=nltk_resources_folder)

    try:
        nltk.data.find("corpora/wordnet")
    except:
        nltk.download("wordnet", download_dir=nltk_resources_folder)

    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords", download_dir=nltk_resources_folder)
