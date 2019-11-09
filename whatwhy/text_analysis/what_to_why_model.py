from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Activation, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from .vectorizer import TokenVectorizer

class WhatToWhyModel():

    def __init__(self, lists_of_what_tokens, lists_of_why_tokens, word2vec_model, max_num_tokens_per_sample=10):
        
        self.num_tokens_per_sample = max_num_tokens_per_sample
        self.word2vec_model = word2vec_model
        self.decoder = TokenVectorizer([], self.word2vec_model, self.num_tokens_per_sample)

        embedded_what_tokens = TokenVectorizer(lists_of_what_tokens, word2vec_model, max_num_tokens_per_sample).get_embeddings()
        one_hot_why_tokens = TokenVectorizer(lists_of_why_tokens, word2vec_model, max_num_tokens_per_sample).get_one_hot_encodings()

        self.embedded_vector_length = embedded_what_tokens.shape[-1]
        self.num_words_in_vocab = one_hot_why_tokens.shape[-1]

        X_train, X_test, y_train, y_test = train_test_split(embedded_what_tokens, one_hot_why_tokens, test_size=0.33, random_state = 42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = None

    def compile(self):
        input_shape = (self.num_tokens_per_sample, self.embedded_vector_length)
        num_units_in_hidden_layer = self.embedded_vector_length
        use_dropout = False
        
        model = Sequential()
        model.add( Input(shape=input_shape) )
        model.add( Masking(mask_value=0.0 ) )
        model.add( LSTM( num_units_in_hidden_layer, return_sequences=True ) )
        model.add( LSTM( num_units_in_hidden_layer, return_sequences=True ) )
        if use_dropout:
            model.add( Dropout(0.5) )
        model.add( TimeDistributed( Dense(self.num_words_in_vocab) ) )

        model.add( Masking(mask_value=0.0 ) )
        model.add( Activation('softmax') )

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy'])
        print(model.summary())

        self.model = model

    def fit(self):
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=16)

    def predict(self, list_of_what_tokens):
        lists_of_what_tokens = [list_of_what_tokens]
        return self.predict_all(lists_of_what_tokens)[0]

    def predict_all(self, lists_of_what_tokens):
        vectorizer = TokenVectorizer(lists_of_what_tokens, self.word2vec_model, self.num_tokens_per_sample)
        embedded_what_tokens = vectorizer.get_embeddings()
        self.predict_all_from_embeddings(embedded_what_tokens)

    def predict_all_from_embeddings(self, embedded_what_tokens):
        predictions_one_hot = self.model.predict(embedded_what_tokens)
        return self.decoder.decode_multiple_one_hot_samples(predictions_one_hot)

    def compare_test_set_to_predictions(self):
        actual_vals = self.decoder.decode_multiple_one_hot_samples(self.y_test)
        predictions = self.predict_all_from_embeddings(self.X_test)
        for i, prediction in enumerate(predictions):
            print(f"Actual    : { actual_vals[i] }")
            print(f"Predicted : { prediction }")
            print("---------------------------------------------")

    def compare_train_set_to_predictions(self):
        actual_vals = self.decoder.decode_multiple_one_hot_samples(self.y_train)
        predictions = self.predict_all_from_embeddings(self.X_train)
        for i, prediction in enumerate(predictions):
            print(f"Actual    : { actual_vals[i] }")
            print(f"Predicted : { prediction }")
            print("---------------------------------------------")
