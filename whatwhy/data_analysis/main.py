import numpy as np
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_google_news_model, get_glove_wiki_gigaword_model, get_custom_word2vec_model, create_and_save_word2vec_model
from .helper_methods import get_text_as_list, remove_uncommon_whatwhy_tokens, get_token_counts
from .whatwhy_predictor import WhatWhyPredictor
from .vocab_index import VocabularyIndex

# --------------------------------------

csv_file_name = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/financial-news-dataset/wh_phrases.csv"

vectorizers_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/financial-news-dataset/vectorizers"
# vectorizers_dir = "/home/ubuntu/whatwhy-data/vectorizers"

model_dir = "/home/stevengt/Documents/code/whatwhy-data/News-Articles/financial-news-dataset/tf-model"
# model_dir = "/home/ubuntu/whatwhy-data/tf-model"

num_samples = 10000
min_num_tokens_per_sample = 4
max_num_tokens_per_sample = 10
epochs = 100
batch_size = 16

# --------------------------------------

def get_raw_what_and_why_tokens_from_csv(csv_file_name, num_samples, max_num_tokens_per_sample, min_token_frequency):
    df = get_df_from_file(csv_file_name)
    df = df.drop_duplicates(subset=QUESTION_WORDS)

    for question_type in QUESTION_WORDS:
        token_col = question_type + " tokens"
        df[token_col] = df[token_col].apply(get_text_as_list)
    
    df = remove_uncommon_whatwhy_tokens(df, min_token_frequency)

    tmp_what_tokens = df["what tokens"].tolist()
    tmp_why_tokens = df["why tokens"].tolist()

    what_tokens = []
    why_tokens = []

    count = 0
    for i in range(len(tmp_what_tokens)):
        if min_num_tokens_per_sample <= len(tmp_what_tokens[i]):
            if min_num_tokens_per_sample <= len(tmp_why_tokens[i]):
                what_tokens.append(tmp_what_tokens[i][:max_num_tokens_per_sample])
                why_tokens.append(tmp_why_tokens[i][:max_num_tokens_per_sample])
                count += 1
                if count >= num_samples:
                    break

    return what_tokens, why_tokens

def create_and_save_token_vectorizers_and_train_and_test_data(what_tokens, why_tokens, max_num_tokens_per_sample, vectorizers_dir):
    word2vec_model = get_google_news_model()  # get_custom_word2vec_model()get_glove_wiki_gigaword_model(100) 
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

def create_and_save_whatwhy_word2vec_model():
    min_token_frequency = 1
    what_tokens, why_tokens = get_raw_what_and_why_tokens_from_csv(csv_file_name, num_samples=1e99, max_num_tokens_per_sample=100, min_token_frequency=min_token_frequency)
    all_tokens = what_tokens
    all_tokens.extend(why_tokens)
    create_and_save_word2vec_model( all_tokens,
                                    embedded_vector_size=200,
                                    min_token_frequency=min_token_frequency,
                                    window=5,
                                    workers=20,
                                    iter=10000 )

# df = get_df_from_file(csv_file_name)
# df = df.drop_duplicates(subset=["Processed Text"])
# df["Processed Text"] = df["Processed Text"].apply(get_text_as_list)
# token_counts = get_token_counts(df, "Processed Text")
# df["Processed Text"] = df["Processed Text"].apply(lambda x: [token for token in x if token_counts[token] >= 100 ])
# all_tokens = df["Processed Text"].tolist()
# create_and_save_word2vec_model( all_tokens,
#                                 embedded_vector_size=100,
#                                 min_token_frequency=100,
#                                 window=10,
#                                 workers=10,
#                                 iter=1000 )


# create_and_save_whatwhy_word2vec_model()
# model = get_custom_word2vec_model()
# print(model.vector_size)
# print(len(model.vocab.keys()))

# what_tokens, why_tokens = get_raw_what_and_why_tokens_from_csv(csv_file_name, num_samples, max_num_tokens_per_sample, min_token_frequency=30)
# print(len(what_tokens))
# print(len(why_tokens))
# create_and_save_token_vectorizers_and_train_and_test_data(what_tokens, why_tokens, max_num_tokens_per_sample, vectorizers_dir)

# w2w_model = load_what_why_predictor(vectorizers_dir)
# w2w_model.fit_tokens(epochs=epochs, batch_size=batch_size)
# w2w_model.save_model(model_dir)

w2w_model = load_what_why_predictor(vectorizers_dir, model_dir)
w2w_model.compare_test_set_to_predictions(max_num_examples=100)
# w2w_model.compare_train_set_to_predictions(max_num_examples=None)
# predictions = w2w_model.predict_all(what_tokens)
# w2w_model.compare_predictions_to_actual(predictions, [ " ".join(tokens) for tokens in why_tokens ])
