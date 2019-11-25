import os
from whatwhy import get_resources_folder

def get_whatwhy_predictor_vectorizers_folder():
    whatwhy_predictor_vectorizers_folder = os.path.join(get_whatwhy_predictor_resources_folder(), "vectorizers")
    if not os.path.isdir(whatwhy_predictor_vectorizers_folder):
        os.mkdir(whatwhy_predictor_vectorizers_folder)
    return whatwhy_predictor_vectorizers_folder

def get_whatwhy_predictor_model_folder():
    whatwhy_predictor_model_folder = os.path.join(get_whatwhy_predictor_resources_folder(), "model")
    if not os.path.isdir(whatwhy_predictor_model_folder):
        os.mkdir(whatwhy_predictor_model_folder)
    return whatwhy_predictor_model_folder

def get_whatwhy_predictor_resources_folder():
    whatwhy_predictor_resources_folder = os.path.join(get_resources_folder(), "whatwhy_predictor")
    if not os.path.isdir(whatwhy_predictor_resources_folder):
        os.mkdir(whatwhy_predictor_resources_folder)
    return whatwhy_predictor_resources_folder
