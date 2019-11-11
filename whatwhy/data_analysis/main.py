import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_glove_wiki_gigaword_model
from .helper_methods import get_text_as_list
from .whatwhy_predictor import WhatWhyPredictor
from .vocab_index import VocabularyIndex

# word2vec_model = None
word2vec_model = get_glove_wiki_gigaword_model(num_dimensions=300)

vectorizers_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/vectorizers"
model_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/tf-model"

csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/wh_phrases.csv"
df = get_df_from_file(csv_file_name)

for question_type in QUESTION_WORDS:
    token_col = question_type + " tokens"
    df[token_col] = df[token_col].apply(get_text_as_list)

num_samples = 20000
what_tokens = df["what tokens"].tolist()[:num_samples]
why_tokens = df["why tokens"].tolist()[:num_samples]

# what_tokens = None
# why_tokens = None

max_num_tokens_per_sample = 15
epochs = 100
vocab_index = None
vocab_index = VocabularyIndex.from_lists(why_tokens)

w2w_model = WhatWhyPredictor(word2vec_model, max_num_tokens_per_sample=max_num_tokens_per_sample, vocab_index=vocab_index)

w2w_model.save_token_vectorizers_to_pickle_files(vectorizers_dir, what_tokens, why_tokens)
# w2w_model.load_token_vectorizers_from_pickle_files(vectorizers_dir)
# w2w_model.load_seq2seq_model_from_saved_tf_model(model_dir)

# w2w_model.fit_tokens(what_tokens, why_tokens, epochs=epochs)
# w2w_model.compare_train_set_to_predictions()
# w2w_model.save_model(model_dir)

# predictions = w2w_model.predict_all(what_tokens)
# w2w_model.compare_predictions_to_actual(predictions, [ " ".join(tokens) for tokens in why_tokens ])
