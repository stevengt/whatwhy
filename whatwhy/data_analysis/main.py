import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_google_news_model, get_glove_wiki_gigaword_model
from .helper_methods import get_text_as_list
from .whatwhy_predictor import WhatWhyPredictor
from .vocab_index import VocabularyIndex

# --------------------------------------

csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/wh_phrases2.csv"

vectorizers_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/vectorizers"
# vectorizers_dir = "/home/ubuntu/whatwhy-data/vectorizers"

model_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/all-the-news/tf-model"
# model_dir = "/home/ubuntu/whatwhy-data/tf-model"

num_samples = 5000
min_num_tokens_per_sample = 3
max_num_tokens_per_sample = 10
epochs = 500
batch_size = 16

# --------------------------------------

def get_raw_what_and_why_tokens_from_csv(csv_file_name, num_samples):
    df = get_df_from_file(csv_file_name)
    
    for question_type in QUESTION_WORDS:
        token_col = question_type + " tokens"
        df[token_col] = df[token_col].apply(get_text_as_list)
    
    tmp_what_tokens = df["what tokens"].tolist()
    tmp_why_tokens = df["why tokens"].tolist()

    what_tokens = []
    why_tokens = []

    count = 0
    for i in range(len(tmp_what_tokens)):
        if min_num_tokens_per_sample <= len(tmp_what_tokens[i]) <= max_num_tokens_per_sample:
            if min_num_tokens_per_sample <= len(tmp_why_tokens[i]) <= max_num_tokens_per_sample:
                what_tokens.append(tmp_what_tokens[i])
                why_tokens.append(tmp_why_tokens[i])
                count += 1
                if count >= num_samples:
                    break

    return what_tokens, why_tokens

def create_and_save_token_vectorizers_and_train_and_test_data(what_tokens, why_tokens, max_num_tokens_per_sample, vectorizers_dir):
    word2vec_model = get_google_news_model() #get_glove_wiki_gigaword_model(100)
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

# w2w_model.compare_train_set_to_predictions()
w2w_model.compare_test_set_to_predictions()
# predictions = w2w_model.predict_all(what_tokens)
# w2w_model.compare_predictions_to_actual(predictions, [ " ".join(tokens) for tokens in why_tokens ])
