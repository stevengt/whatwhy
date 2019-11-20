import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_google_news_model
from .helper_methods import get_text_as_list
from .whatwhy_predictor import WhatWhyPredictor
from .vocab_index import VocabularyIndex

# --------------------------------------

csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/wh_phrases.csv"

vectorizers_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/vectorizers"
# vectorizers_dir = "/home/ubuntu/whatwhy-data/vectorizers"

model_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/tf-model"
# model_dir = "/home/ubuntu/whatwhy-data/tf-model"

num_samples = 30000
max_num_tokens_per_sample = 15
epochs = 75
batch_size = 128

# --------------------------------------

def get_raw_what_and_why_tokens_from_csv(csv_file_name, num_samples):
    df = get_df_from_file(csv_file_name)
    
    for question_type in QUESTION_WORDS:
        token_col = question_type + " tokens"
        df[token_col] = df[token_col].apply(get_text_as_list)
    
    what_tokens = df["what tokens"].tolist()[:num_samples]
    why_tokens = df["why tokens"].tolist()[:num_samples]

    return what_tokens, why_tokens

def create_and_save_token_vectorizers_and_train_and_test_data(what_tokens, why_tokens, max_num_tokens_per_sample, vectorizers_dir):
    word2vec_model = get_google_news_model()
    vocab_index = VocabularyIndex.from_lists(why_tokens)
    w2w_model = WhatWhyPredictor(word2vec_model, max_num_tokens_per_sample=max_num_tokens_per_sample, vocab_index=vocab_index)
    w2w_model.save_token_vectorizers_to_pickle_files(vectorizers_dir, what_tokens, why_tokens)
    w2w_model.save_train_and_test_data_to_pickle_files(vectorizers_dir)

def load_what_why_predictor(vectorizers_dir, model_dir=None):
    w2w_model = WhatWhyPredictor()
    w2w_model.load_token_vectorizers_from_pickle_files(vectorizers_dir)
    w2w_model.load_train_and_test_data_from_pickle_files(vectorizers_dir)
    if model_dir is not None:
        w2w_model.load_seq2seq_model_from_saved_tf_model(model_dir)
    return w2w_model



# what_tokens, why_tokens = get_raw_what_and_why_tokens_from_csv(csv_file_name, num_samples)
# create_and_save_token_vectorizers_and_train_and_test_data(what_tokens, why_tokens, max_num_tokens_per_sample, vectorizers_dir)

# w2w_model = load_what_why_predictor(vectorizers_dir)
w2w_model = load_what_why_predictor(vectorizers_dir, model_dir)

# w2w_model.fit_tokens(epochs=epochs, batch_size=batch_size)
# w2w_model.save_model(model_dir)

w2w_model.compare_train_set_to_predictions()
# predictions = w2w_model.predict_all(what_tokens)
# w2w_model.compare_predictions_to_actual(predictions, [ " ".join(tokens) for tokens in why_tokens ])
