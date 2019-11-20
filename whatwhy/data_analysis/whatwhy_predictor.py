import os
import pickle
from sklearn.model_selection import train_test_split
from .seq2seq_model import Seq2SeqModel
from .vectorizer import TokenVectorizer

class WhatWhyPredictor():
    """Predicts a sequence of text which answers the question 'why?' given some input 'what'."""

    def __init__(self, word2vec_model=None, seq2seq_model=None, max_num_tokens_per_sample=20, vocab_index=None):
        self.word2vec_model = word2vec_model
        self.seq2seq_model = seq2seq_model
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.vocab_index = vocab_index

        if seq2seq_model is not None:
            self.max_num_tokens_per_sample = seq2seq_model.num_tokens_per_sample

        self.what_token_vectorizer = None
        self.why_token_vectorizer = None
        
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # If word2vec_model is None, then the decoder should be loader from a pickle file instead.
        if word2vec_model is not None:
            self.decoder = TokenVectorizer( word2vec_model=word2vec_model,
                                            num_tokens_per_sample=self.max_num_tokens_per_sample,
                                            vocab_index=self.vocab_index )

    def load_seq2seq_model_from_saved_tf_model(self, model_dir):
        X_train, X_test, Y_train, Y_test = self.get_train_and_test_data()
        self.seq2seq_model = Seq2SeqModel(X_train, X_test, Y_train, Y_test).load_from_saved_tf_model(model_dir)

    def fit_tokens( self, lists_of_what_tokens=None,
                          lists_of_why_tokens=None,
                          epochs=1,
                          batch_size=None ):
        """Trains a Seq2SeqModel on lists that contain sequences (lists) of 'what' and 'why' tokens."""

        X_train, X_test, Y_train, Y_test = self.get_train_and_test_data(lists_of_what_tokens=lists_of_what_tokens, lists_of_why_tokens=lists_of_why_tokens)
        self.seq2seq_model = Seq2SeqModel(X_train, X_test, Y_train, Y_test)
        self.seq2seq_model.fit(epochs=epochs, batch_size=batch_size)

    def get_train_and_test_data( self, lists_of_what_tokens=None,
                                       lists_of_why_tokens=None,
                                       test_size=0.20,
                                       random_state=42 ):
    
        if self.X_train is None or self.X_test is None or self.Y_train is None or self.Y_test is None:
            what_token_vectorizer, why_token_vectorizer = self.get_what_and_why_token_vectorizers(lists_of_what_tokens, lists_of_why_tokens)
            embedded_what_tokens = what_token_vectorizer.get_embeddings()
            one_hot_why_tokens = why_token_vectorizer.get_one_hot_encodings()
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split( embedded_what_tokens,
                                                                                     one_hot_why_tokens,
                                                                                     test_size=test_size,
                                                                                     random_state=random_state )
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def save_train_and_test_data_to_pickle_files(self, dir_name, lists_of_what_tokens=None, lists_of_why_tokens=None):
        X_train, X_test, Y_train, Y_test = self.get_train_and_test_data(lists_of_what_tokens=lists_of_what_tokens, lists_of_why_tokens=lists_of_why_tokens)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        with open( os.path.join(dir_name, "X_train.p") , "wb" ) as out_file:
            pickle.dump(X_train, out_file, protocol=4)
        with open( os.path.join(dir_name, "X_test.p") , "wb" ) as out_file:
            pickle.dump(X_test, out_file, protocol=4)
        with open( os.path.join(dir_name, "Y_train.p") , "wb" ) as out_file:
            pickle.dump(Y_train, out_file, protocol=4)
        with open( os.path.join(dir_name, "Y_test.p") , "wb" ) as out_file:
            pickle.dump(Y_test, out_file, protocol=4)

    def load_train_and_test_data_from_pickle_files(self, dir_name):
        with open( os.path.join(dir_name, "X_train.p") , "rb" ) as in_file:
            self.X_train = pickle.load(in_file)
        with open( os.path.join(dir_name, "X_test.p") , "rb" ) as in_file:
            self.X_test = pickle.load(in_file)
        with open( os.path.join(dir_name, "Y_train.p") , "rb" ) as in_file:
            self.Y_train = pickle.load(in_file)
        with open( os.path.join(dir_name, "Y_test.p") , "rb" ) as in_file:
            self.Y_test = pickle.load(in_file)

    def get_what_and_why_token_vectorizers(self, lists_of_what_tokens=None, lists_of_why_tokens=None):
        if self.what_token_vectorizer is None:
            self.what_token_vectorizer = TokenVectorizer( word2vec_model=self.word2vec_model,
                                                          tokens_lists=lists_of_what_tokens,
                                                          num_tokens_per_sample=self.max_num_tokens_per_sample,
                                                          vocab_index=self.vocab_index )

        if self.why_token_vectorizer is None:
            self.why_token_vectorizer = TokenVectorizer( word2vec_model=self.word2vec_model,
                                                         tokens_lists=lists_of_why_tokens,
                                                         num_tokens_per_sample=self.max_num_tokens_per_sample,
                                                         vocab_index=self.vocab_index )

        return self.what_token_vectorizer, self.why_token_vectorizer

    def save_token_vectorizers_to_pickle_files(self, target_dir, lists_of_what_tokens=None, lists_of_why_tokens=None):
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        what_token_vectorizer, why_token_vectorizer = self.get_what_and_why_token_vectorizers(lists_of_what_tokens, lists_of_why_tokens)

        what_token_vectorizer.get_embeddings()
        why_token_vectorizer.get_one_hot_encodings()

        what_token_vectorizer.save_to_pickle_file( os.path.join(target_dir, "what_tokenizer.p") )
        why_token_vectorizer.save_to_pickle_file( os.path.join(target_dir, "why_tokenizer.p") )
        self.decoder.save_to_pickle_file( os.path.join(target_dir, "decoder.p") )

    def load_token_vectorizers_from_pickle_files(self, dir_name):
        # self.what_token_vectorizer = TokenVectorizer.load_from_pickle_file( os.path.join(dir_name, "what_tokenizer.p") )
        # self.why_token_vectorizer = TokenVectorizer.load_from_pickle_file( os.path.join(dir_name, "why_tokenizer.p") )
        self.decoder = TokenVectorizer.load_from_pickle_file( os.path.join(dir_name, "decoder.p") )
        self.word2vec_model = self.decoder.word2vec_model
        self.vocab_index = self.decoder.vocab_index
        self.max_num_tokens_per_sample = self.decoder.num_tokens_per_sample

    def predict(self, list_of_what_tokens):
        lists_of_what_tokens = [list_of_what_tokens]
        return self.predict_all(lists_of_what_tokens)[0]

    def predict_all(self, lists_of_what_tokens):
        embedded_what_tokens = TokenVectorizer( word2vec_model=self.word2vec_model,
                                                tokens_lists=lists_of_what_tokens,
                                                num_tokens_per_sample=self.max_num_tokens_per_sample,
                                                vocab_index=self.vocab_index ).get_embeddings()
        one_hot_predictions = self.seq2seq_model.predict_all(embedded_what_tokens)
        predictions = self.decoder.decode_multiple_one_hot_samples(one_hot_predictions)
        return predictions

    def compare_predictions_to_actual(self, predictions, actual_vals):
        for i, prediction in enumerate(predictions):
            print(f"Actual    : { actual_vals[i] }")
            print(f"Predicted : { prediction }")
            print("---------------------------------------------")

    def compare_test_set_to_predictions(self):
        X_test = self.X_test[:100,:,:]
        Y_test = self.Y_test[:100,:,:]
        actual_vals = self.decoder.decode_multiple_one_hot_samples(Y_test)
        one_hot_predictions = self.seq2seq_model.predict_all(X_test)
        predictions = self.decoder.decode_multiple_one_hot_samples(one_hot_predictions)
        self.compare_predictions_to_actual(predictions, actual_vals)

    def compare_train_set_to_predictions(self):
        X_train = self.X_train
        Y_train = self.Y_train
        actual_vals = self.decoder.decode_multiple_one_hot_samples(Y_train)
        one_hot_predictions = self.seq2seq_model.predict_all(X_train)
        predictions = self.decoder.decode_multiple_one_hot_samples(one_hot_predictions)
        self.compare_predictions_to_actual(predictions, actual_vals)

    def save_model(self, model_dir):
        self.seq2seq_model.save_model(model_dir)
