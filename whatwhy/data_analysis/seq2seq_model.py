from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Activation, Masking, Dropout
from tensorflow.keras.optimizers import Adam

class Seq2SeqModel():
    """
    A model which predicts one-hot encoded sequences of token data
    based on sequences of 'embedded' token data (i.e., each token is a vector).
    """

    def __init__(self, X, Y):
        """
        Params:
            X : An array of embedded input data with dimensions [num_samples, num_tokens_per_sample, embedded_vector_length].
            Y : An array of one-hot encoded output data with dimensions [num_samples, num_tokens_per_sample, num_token_categories]
        """

        self.num_tokens_per_sample = X.shape[1]
        self.embedded_vector_length = X.shape[2]
        self.num_token_categories = Y.shape[2]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 42)
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.model = None
        self.compile()

    def compile(self):
        input_shape = (self.num_tokens_per_sample, self.embedded_vector_length)
        num_units_in_hidden_layer = self.embedded_vector_length
        use_dropout = False
        
        model = Sequential()
        model.add( Input(shape=input_shape) )
        model.add( Masking(mask_value=0.0 ) )
        model.add( Bidirectional( LSTM( num_units_in_hidden_layer, return_sequences=True ) ) )
        model.add( Bidirectional( LSTM( num_units_in_hidden_layer, return_sequences=True ) ) )
        if use_dropout:
            model.add( Dropout(0.5) )
        model.add( TimeDistributed( Dense(self.num_token_categories) ) )

        model.add( Masking(mask_value=0.0 ) )
        model.add( Activation('softmax') )

        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['categorical_accuracy'])
        print(model.summary())

        self.model = model

    def fit(self, epochs=1, batch_size=None):
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        X = [x]
        return self.predict_all(X)[0]

    def predict_all(self, X):
        return self.model.predict(X)
