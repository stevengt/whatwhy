import argparse
from argparse import RawTextHelpFormatter
from whatwhy import QUESTION_WORDS
from whatwhy.text_processing.helper_methods import get_df_from_file
from whatwhy.resource_manager import get_google_news_model, get_whatwhy_predictor_vectorizers_folder, get_whatwhy_predictor_model_folder
from .helper_methods import get_text_as_list, remove_uncommon_whatwhy_tokens
from .whatwhy_predictor import WhatWhyPredictor
from .vocab_index import VocabularyIndex

description = """
This is a CLI for training and using a model to predict sequences of 'why' text from input 'what' text.

This script uses the GoogleNews gensim Word2Vec model to embed
text tokens into 300-dimensional vectors. The first time the
script is run it may take some time to download this model.

Also note that this model is quite large (about 3.6 GB), and any
models created by this script can also potentially be large
depending on the size of the provided data set. If you are using
a Linux system with a small root partition, then you should install
this script in a Python distribution (e.g., Anaconda) that stores
files on a large enough disk.
"""

def get_raw_what_and_why_tokens_from_csv(csv_file_name, min_num_tokens_per_sample, max_num_tokens_per_sample, min_token_frequency):
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

    for i in range(len(tmp_what_tokens)):
        if min_num_tokens_per_sample <= len(tmp_what_tokens[i]):
            if min_num_tokens_per_sample <= len(tmp_why_tokens[i]):
                what_tokens.append(tmp_what_tokens[i][:max_num_tokens_per_sample])
                why_tokens.append(tmp_why_tokens[i][:max_num_tokens_per_sample])

    return what_tokens, why_tokens

def create_and_save_whatwhy_predictor(what_tokens, why_tokens, max_num_tokens_per_sample):
    model_dir = get_whatwhy_predictor_model_folder()
    word2vec_model = get_google_news_model()
    vocab_index = VocabularyIndex.from_lists(why_tokens)
    predictor = WhatWhyPredictor(word2vec_model, max_num_tokens_per_sample=max_num_tokens_per_sample, vocab_index=vocab_index)
    predictor.set_what_and_why_token_vectorizers_from_lists(what_tokens, why_tokens)
    predictor.save_to_pickle_file(model_dir)
    return predictor

def train_and_save_whatwhy_predictor(predictor, epochs, batch_size):
    model_dir = get_whatwhy_predictor_model_folder()
    predictor.fit_tokens(epochs=epochs, batch_size=batch_size)
    predictor.save_seq2seq_model(model_dir)
    predictor.seq2seq_model = None # Serializing tensorflow objects is difficult, so save its model weights separately.
    predictor.save_to_pickle_file(model_dir)

def load_whatwhy_predictor():
    model_dir = get_whatwhy_predictor_model_folder()
    predictor = WhatWhyPredictor.load_from_pickle_file(model_dir)
    predictor.load_seq2seq_model_from_saved_tf_model(model_dir)
    return predictor

def main():
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)

    arggroup = parser.add_mutually_exclusive_group(required=True)
    arggroup.add_argument("--train", action="store_true", help="Train a prediction model using a supplied CSV file or previously loaded data set.\n" \
                                                              + "This will overwrite any previously trained models.")
    arggroup.add_argument("--predict", nargs="+", default=None, help="Uses a previously trained model to predict a sequence of 'why' text from the input 'what' text.")
    arggroup.add_argument("--compare-test", action="store_true", help="Uses a previously trained model to compare its predictions against its testing data set.")
    arggroup.add_argument("--compare-train", action="store_true", help="Uses a previously trained model to compare its predictions against its training data set.")

    parser.add_argument("-csv", "--csv-file-name", default=None, help="Name of a tab delimited local CSV file containing a data set for model training.\n" \
                                                                     + "If left blank, the most recently loaded data set will be used.\n" \
                                                                     + "CSV files must include columns labeled 'what tokens' and 'why tokens',\n" \
                                                                     + "each containing plain-text representations of a Python list of strings.")
    parser.add_argument("--min-token-frequency", type=int, default=30, help="The minimum number of times a token should occur in the dataset to be used for training a WhatWhyPredictor model.") 
    parser.add_argument("-min-tokens", "--min-tokens-per-sample", type=int, default=4, help="The minimum number of tokens a sample should contain to be used for training a WhatWhyPredictor model.")
    parser.add_argument("-max-tokens", "--max-tokens-per-sample", type=int, default=10, help="The maximum number of tokens a sample should contain for training a WhatWhyPredictor model. Any extra tokens will be truncated.")
    parser.add_argument("-bs", "--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    if args.train:
        predictor = None
        if args.csv_file_name is not None:
            what_tokens, why_tokens = get_raw_what_and_why_tokens_from_csv( args.csv_file_name,
                                                                            args.min_tokens_per_sample,
                                                                            args.max_tokens_per_sample,
                                                                            args.min_token_frequency )
            predictor = create_and_save_whatwhy_predictor( what_tokens,
                                                           why_tokens,
                                                           args.max_tokens_per_sample )
        else:
            predictor = load_whatwhy_predictor()
        train_and_save_whatwhy_predictor(predictor, args.epochs, args.batch_size)
    else:
        predictor = load_whatwhy_predictor()
        if args.predict is not None:
            predictor.predict(args.predict)
        elif args.compare_test:
            predictor.compare_test_set_to_predictions()
        elif args.compare_train:
            predictor.compare_train_set_to_predictions()

if __name__ == "__main__":
    main()
