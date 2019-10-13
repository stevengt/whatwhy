import re
import numpy as np
from gingerit.gingerit import GingerIt
from textblob import TextBlob

spelling_and_grammar_parser = GingerIt()

def remove_reply_tag_from_tweet_text(text):
    if text is None or text is np.nan:
        return text
    reply_tag = re.match("@[^\s]+[\s]+", text)
    if reply_tag is not None:
        reply_tag = reply_tag.group(0)
        text = text.replace(reply_tag, "")
    return text

def autocorrect_spelling_and_grammar(text):
    return spelling_and_grammar_parser.parse(text)["result"]

def get_list_of_lemmatized_words_from_text(text):
    if text is None or text is np.nan:
        return []
    try:
        text_blob = TextBlob(text)
        return list( text_blob.words.singularize().lemmatize() )
    except:
        return []
