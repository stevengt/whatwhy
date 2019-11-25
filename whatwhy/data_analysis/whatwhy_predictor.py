import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from .seq2seq_model import Seq2SeqModel
from .vectorizer import TokenVectorizer

class WhatWhyPredictor():
    """
    Predicts a sequence of text which answers the question 'why?' given some input 'what'.
    
    The prediction model is trained by vectorizing lists of token sequences and passing
    the results to a Seq2SeqModel and calling its fit() method. After training, the
    predict() methods can be used to predict 'why' text from 'what' text.

    The Seq2SeqModel, vectorizers, and vectorized data sets can be specified manually
    or saved and loaded from files using the save/load methods.
    """

    def __init__(self, word2vec_model=None, max_num_tokens_per_sample=20, vocab_index=None):
        """
        Creates a WhatWhyPredictor instance using the specified parameters.
        If no parameters are specified, then they should be loaded from
        a file using the load() methods.

        Params:
            word2vec_model            : [Optional] A pre-trained gensim Word2Vec model.
            max_num_tokens_per_sample : [Optional] Maximum number of tokens to include in a sample sequence.
                                                   Any extra tokens will be truncated.
            vocab_index               : [Optional] A pre-built VocabularyIndex of the data set. This can
                                                   help reduce the size of one-hot encoded words in the
                                                   vocabulary, compared to that of pre-trained word2vec models.
        """

        self.word2vec_model = word2vec_model
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.vocab_index = vocab_index

        self.what_token_vectorizer = None
        self.why_token_vectorizer = None
        
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.indeces_train = None
        self.indeces_test = None

        # If word2vec_model is None, then the decoder should be loaded from a pickle file instead.
        if word2vec_model is not None:
            self.decoder = TokenVectorizer( word2vec_model=word2vec_model,
                                            num_tokens_per_sample=self.max_num_tokens_per_sample,
                                            vocab_index=self.vocab_index )

    def fit_tokens( self, lists_of_what_tokens=None,
                          lists_of_why_tokens=None,
                          epochs=1,
                          batch_size=None ):
        """Trains a Seq2SeqModel on lists that contain sequences (lists) of 'what' and 'why' tokens."""

        X_train, X_test, Y_train, Y_test, indeces_train, indeces_test = self.get_train_and_test_data( lists_of_what_tokens=lists_of_what_tokens,
                                                                                                      lists_of_why_tokens=lists_of_why_tokens )
        self.seq2seq_model = Seq2SeqModel(X_train, X_test, Y_train, Y_test)
        self.seq2seq_model.fit(epochs=epochs, batch_size=batch_size)

    def predict(self, list_of_what_tokens):
        """
        Predicts a string of 'why' text from an input sequence of 'what' tokens.
        
        The following instance fields should be initialized or loaded before calling this method.
            word2vec_model
            num_tokens_per_sample 
            seq2seq_model
            decoder
        """
        lists_of_what_tokens = [list_of_what_tokens]
        return self.predict_all(lists_of_what_tokens)[0]

    def predict_all(self, lists_of_what_tokens):
        """
        Predicts strings of 'why' text from input sequences of 'what' tokens.
        
        The following instance fields should be initialized or loaded before calling this method.
            word2vec_model
            num_tokens_per_sample 
            seq2seq_model
            decoder
        """
        embedded_what_tokens = TokenVectorizer( word2vec_model=self.word2vec_model,
                                                tokens_lists=lists_of_what_tokens,
                                                num_tokens_per_sample=self.max_num_tokens_per_sample,
                                                vocab_index=self.vocab_index ).get_embeddings()
        one_hot_predictions = self.seq2seq_model.predict_all(embedded_what_tokens)
        predictions = self.decoder.decode_multiple_one_hot_samples(one_hot_predictions)
        return predictions

    def compare_predictions_to_actual(self, input_tokens, predictions, actual_vals):
        for i, prediction in enumerate(predictions):
            print(f"'What' Input    : { ' '.join(input_tokens[i]) }")
            print(f"'Why' Actual    : { actual_vals[i] }")
            print(f"'Why' Predicted : { prediction }")
            print("---------------------------------------------")

    def compare_test_set_to_predictions(self, max_num_examples=None):
        if max_num_examples is None:
            max_num_examples = self.X_test.shape[0]
        X_test = self.X_test[:max_num_examples,:,:]
        Y_test = self.Y_test[:max_num_examples,:,:]
        indeces_test = self.indeces_test[:max_num_examples]
        input_tokens_test = [ self.what_token_vectorizer.tokens_lists[index] for index in indeces_test ]
        actual_vals = self.decoder.decode_multiple_one_hot_samples(Y_test)
        one_hot_predictions = self.seq2seq_model.predict_all(X_test)
        predictions = self.decoder.decode_multiple_one_hot_samples(one_hot_predictions)
        self.compare_predictions_to_actual(input_tokens_test, predictions, actual_vals)

    def compare_train_set_to_predictions(self, max_num_examples=None):
        if max_num_examples is None:
            max_num_examples = self.X_train.shape[0]
        X_train = self.X_train[:max_num_examples,:,:]
        Y_train = self.Y_train[:max_num_examples,:,:]
        indeces_train = self.indeces_train[:max_num_examples]
        input_tokens_train = [ self.what_token_vectorizer.tokens_lists[index] for index in indeces_train ]
        actual_vals = self.decoder.decode_multiple_one_hot_samples(Y_train)
        one_hot_predictions = self.seq2seq_model.predict_all(X_train)
        predictions = self.decoder.decode_multiple_one_hot_samples(one_hot_predictions)
        self.compare_predictions_to_actual(input_tokens_train, predictions, actual_vals)

    def get_what_and_why_token_vectorizers(self, lists_of_what_tokens=None, lists_of_why_tokens=None):
        """
        Returns TokenVectorizers for the lists of what/why token sequences.
        
        The instance fields 'word2vec_model', 'num_tokens_per_sample', and
        optionally 'vocab_index' should be initialized before calling this method.
        """
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

    def get_train_and_test_data( self, lists_of_what_tokens=None,
                                       lists_of_why_tokens=None,
                                       test_size=0.20,
                                       random_state=42 ):
        """
        Splits a data set of what/why tokens into test and train sets
        if they have not already been separated.
        """

        if self.X_train is None or self.X_test is None or self.Y_train is None or self.Y_test is None:
            what_token_vectorizer, why_token_vectorizer = self.get_what_and_why_token_vectorizers(lists_of_what_tokens, lists_of_why_tokens)
            embedded_what_tokens = what_token_vectorizer.get_embeddings()
            one_hot_why_tokens = why_token_vectorizer.get_one_hot_encodings()
            indeces = np.arange( len(what_token_vectorizer.tokens_lists) )
            self.X_train, self.X_test, self.Y_train, self.Y_test, self.indeces_train, self.indeces_test = train_test_split( embedded_what_tokens,
                                                                                                                            one_hot_why_tokens,
                                                                                                                            indeces,
                                                                                                                            test_size=test_size,
                                                                                                                            random_state=random_state )
        return self.X_train, self.X_test, self.Y_train, self.Y_test, self.indeces_train, self.indeces_test

    def save_model(self, model_dir):
        """
        Saves the underlying tensorflow.keras model's weights to
        a file 'model.h5' in the specified directory.
        """
        self.seq2seq_model.save_model(model_dir)

    def load_seq2seq_model_from_saved_tf_model(self, model_dir):
        """
        Intializes the Seq2SeqModel by loading weights from
        a file 'model.h5' in the specified directory.
        """
        X_train, X_test, Y_train, Y_test, indeces_train, indeces_test = self.get_train_and_test_data()
        self.seq2seq_model = Seq2SeqModel(X_train, X_test, Y_train, Y_test).load_from_saved_tf_model(model_dir)

    def save_train_and_test_data_to_pickle_files(self, dir_name, lists_of_what_tokens=None, lists_of_why_tokens=None):
        """
        Splits a data set of what/why tokens into test and train sets
        if they have not already been separated and saves them in pickle files.
        """
        X_train, X_test, Y_train, Y_test, indeces_train, indeces_test = self.get_train_and_test_data( lists_of_what_tokens=lists_of_what_tokens,
                                                                                                      lists_of_why_tokens=lists_of_why_tokens )
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
        with open( os.path.join(dir_name, "indeces_train.p") , "wb" ) as out_file:
            pickle.dump(indeces_train, out_file, protocol=4)
        with open( os.path.join(dir_name, "indeces_test.p") , "wb" ) as out_file:
            pickle.dump(indeces_test, out_file, protocol=4)

    def load_train_and_test_data_from_pickle_files(self, dir_name):
        with open( os.path.join(dir_name, "X_train.p") , "rb" ) as in_file:
            self.X_train = pickle.load(in_file)
        with open( os.path.join(dir_name, "X_test.p") , "rb" ) as in_file:
            self.X_test = pickle.load(in_file)
        with open( os.path.join(dir_name, "Y_train.p") , "rb" ) as in_file:
            self.Y_train = pickle.load(in_file)
        with open( os.path.join(dir_name, "Y_test.p") , "rb" ) as in_file:
            self.Y_test = pickle.load(in_file)
        with open( os.path.join(dir_name, "indeces_train.p") , "rb" ) as in_file:
            self.indeces_train = pickle.load(in_file)
        with open( os.path.join(dir_name, "indeces_test.p") , "rb" ) as in_file:
            self.indeces_test = pickle.load(in_file)

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
        self.what_token_vectorizer = TokenVectorizer.load_from_pickle_file( os.path.join(dir_name, "what_tokenizer.p") )
        self.why_token_vectorizer = TokenVectorizer.load_from_pickle_file( os.path.join(dir_name, "why_tokenizer.p") )
        self.decoder = TokenVectorizer.load_from_pickle_file( os.path.join(dir_name, "decoder.p") )
        self.word2vec_model = self.decoder.word2vec_model
        self.vocab_index = self.decoder.vocab_index
        self.max_num_tokens_per_sample = self.decoder.num_tokens_per_sample
