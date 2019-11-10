import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_glove_wiki_gigaword_50_model
from .helper_methods import get_text_as_list
from .whatwhy_predictor import WhatWhyPredictor
from .vocab_index import VocabularyIndex

word2vec_model = get_glove_wiki_gigaword_50_model()
vectorizers_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/vectorizers"

csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/wh_phrases.csv"
df = get_df_from_file(csv_file_name)

for question_type in QUESTION_WORDS:
    token_col = question_type + " tokens"
    df[token_col] = df[token_col].apply(get_text_as_list)

num_samples = 10000
what_tokens = df["what tokens"].tolist()[:num_samples]
why_tokens = df["why tokens"].tolist()[:num_samples]


# from .mock_model import *
# word2vec_model = MockWord2VecModel()
# what_tokens = MOCK_TOKENS_LISTS
# why_tokens = MOCK_TOKENS_LISTS[::-1]

# why_tokens = what_tokens

max_num_tokens_per_sample = 10
epochs = 1000
vocab_index = VocabularyIndex.from_lists(why_tokens)

w2w_model = WhatWhyPredictor(word2vec_model, max_num_tokens_per_sample=max_num_tokens_per_sample, vocab_index=vocab_index)

w2w_model.save_token_vectorizers_to_pickle_files(vectorizers_dir, what_tokens, why_tokens)

# w2w_model.load_token_vectorizers_from_pickle_files(vectorizers_dir)
# print(w2w_model.what_token_vectorizer.get_embeddings().shape)
# print(w2w_model.why_token_vectorizer.get_one_hot_encodings().shape)
# print(w2w_model.decoder.num_words_in_vocab)
# print(w2w_model.word2vec_model.vector_size)

# w2w_model.fit_tokens(what_tokens, why_tokens, epochs=epochs)
# w2w_model.compare_train_set_to_predictions()
