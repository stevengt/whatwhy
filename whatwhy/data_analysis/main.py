import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_glove_wiki_gigaword_50_model
from .helper_methods import get_text_as_list
from .whatwhy_predictor import WhatWhyPredictor

word2vec_model = get_glove_wiki_gigaword_50_model()

csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/wh_phrases.csv"
df = get_df_from_file(csv_file_name)

for question_type in QUESTION_WORDS:
    token_col = question_type + " tokens"
    df[token_col] = df[token_col].apply(get_text_as_list)

what_tokens = df["what tokens"].tolist()[:10]
why_tokens = df["why tokens"].tolist()[:10]


# from .mock_model import *
# word2vec_model = MockWord2VecModel()
# what_tokens = MOCK_TOKENS_LISTS
# why_tokens = MOCK_TOKENS_LISTS[::-1]

max_num_tokens_per_sample = 10
epochs = 10

w2w_model = WhatWhyPredictor(word2vec_model, max_num_tokens_per_sample=max_num_tokens_per_sample)
# w2w_model.fit_tokens(what_tokens, why_tokens, epochs=epochs)
w2w_model.fit_tokens(what_tokens, what_tokens, epochs=epochs)
w2w_model.compare_train_set_to_predictions()
