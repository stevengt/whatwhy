import os
import numpy as np
import tensorflow
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Activation, Masking, Dropout
from tensorflow.keras.optimizers import Adam

class Seq2SeqModel():
    """
    A model which predicts one-hot encoded sequences of token data
    based on sequences of 'embedded' token data (i.e., each token is a vector).

    This class wraps a tensorflow.keras model that can be trained, saved, and loaded
    using a specified data set.
    """

    def __init__(self, X_train=None, X_test=None, Y_train=None, Y_test=None, pretrained_model=None):
        """
        Params:
            X_train, X_test  : An array of embedded input data with dimensions [num_samples, num_tokens_per_sample, embedded_vector_length].
            Y_train, Y_test  : An array of one-hot encoded output data with dimensions [num_samples, num_tokens_per_sample, num_token_categories].
            pretrained_model : [Optional] A pretrained tensorflow.keras model. If specified, all other arguments are ignored.
        """
        if pretrained_model is None:
            self.num_tokens_per_sample = X_train.shape[1]
            self.embedded_vector_length = X_train.shape[2]
            self.num_token_categories = Y_train.shape[2]

            self.X_train = X_train
            self.X_test = X_test
            self.Y_train = Y_train
            self.Y_test = Y_test

            self.model = None
        else:
            self.model = pretrained_model

    def load_from_saved_tf_model(self, model_dir):
        """
        Intializes a Seq2SeqModel by loading weights from
        a file 'model.h5' in the specified directory.
        """
        file_name = os.path.join(model_dir, "model.h5")
        self.model = self.get_new_model()
        self.model.load_weights(file_name)
        return self

    def get_new_model(self):
        """
        Returns a new instance of a tensorflow.keras model with LSTM layers
        and masking layers to ignore padded sections of token sequences.
        """
        input_shape = (self.num_tokens_per_sample, self.embedded_vector_length)
        num_units_in_hidden_layer = self.embedded_vector_length
        use_dropout = True
        
        model = Sequential()
        model.add( Input( shape=input_shape ) )
        model.add( Masking( mask_value=0.0 ) )

        model.add( Bidirectional( LSTM( num_units_in_hidden_layer, return_sequences=True ) ) )
        model.add( Bidirectional( LSTM( num_units_in_hidden_layer, return_sequences=True ) ) )

        if use_dropout:
            model.add( Dropout( 0.4 ) )
        model.add( TimeDistributed( Dense( self.num_token_categories ) ) )

        model.add( Masking( mask_value=0.0 ) )
        model.add( Activation( 'softmax' ) )

        model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'] )
        print( model.summary() )

        return model

    def fit(self, epochs=1, batch_size=None):
        """Trains the underlying tensorflow.keras model."""
        X_train = self.X_train
        Y_train = self.Y_train
        self.model = self.get_new_model()
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        """Returns a one-hot encoded prediction for a single embedded input vector."""
        X = [x]
        return self.predict_all(X)[0]

    def predict_all(self, X):
        """Returns one-hot encoded predictions for multiple embedded input vectors."""
        return self.model.predict(X)

    def save_model(self, model_dir):
        """
        Saves the underlying tensorflow.keras model's weights to
        a file 'model.h5' in the specified directory.
        """
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        file_name = os.path.join(model_dir, "model.h5")
        self.model.save(file_name)
