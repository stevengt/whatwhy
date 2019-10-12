import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def remove_reply_tag_from_tweet_text(text):
    if text is None or text is np.nan:
        return text
    reply_tag = re.match("@[^\s]+[\s]+", text)
    if reply_tag is not None:
        reply_tag = reply_tag.group(0)
        text = text.replace(reply_tag, "")
    return text

def remove_punctuation_from_text(text):
    if text is None or text is np.nan:
        return ""
    try:
        return text.translate( str.maketrans("", "", string.punctuation) )
    except:
        return text

def get_list_of_words_from_text(text):
    if text is None or text is np.nan:
        return []
    try:
        return nltk.word_tokenize(text)
    except:
        return []

def lemmatize_list_of_words(words):
    try:
        return [ lemmatizer.lemmatize(word) for word in words ]
    except:
        return words
