import numpy as np
from textblob import TextBlob

def get_list_of_lemmatized_words_from_text(text):
    if text is None or text is np.nan:
        return []
    try:
        text_blob = TextBlob(text)
        return list( text_blob.words.singularize().lemmatize() )
    except:
        return []
