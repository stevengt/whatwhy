import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_glove_wiki_gigaword_50_model
from .helper_methods import get_text_as_list
from .vectorizer import TokenVectorizer

# word2vec_model = get_glove_wiki_gigaword_50_model()

# csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/wh_phrases.csv"
# df = get_df_from_file(csv_file_name)

# for question_type in QUESTION_WORDS:
#     token_col = question_type + " tokens"
#     df[token_col] = df[token_col].apply(get_text_as_list)

# what_tokens = df["what tokens"].tolist()[:5]
# why_tokens = df["why tokens"].tolist()[:5]

# what_token_vectorizer = TokenVectorizer(what_tokens, word2vec_model)
# why_token_vectorizer = TokenVectorizer(why_tokens, word2vec_model)

# embedded_what_tokens = what_token_vectorizer.get_embeddings()
# one_hot_why_tokens = why_token_vectorizer.get_one_hot_encodings()

from .mock_model import *
from .what_to_why_model import WhatToWhyModel

model = MockWord2VecModel()

what_tokens = MOCK_TOKENS_LISTS
why_tokens = MOCK_TOKENS_LISTS[::-1]

w2w_model = WhatToWhyModel(what_tokens, why_tokens, model)
w2w_model.compile()
w2w_model.fit()
w2w_model.compare_train_set_to_predictions()
