import ast
import numpy as np

def get_text_as_list(text):
    if text is None or text is np.nan:
        return []
    else:
        return ast.literal_eval(text)

def get_default_token(word2vec_model):
    return "." if "." in word2vec_model.vocab.keys() else word2vec_model.index2word[0]
