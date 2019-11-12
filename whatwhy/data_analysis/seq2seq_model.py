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
    """

    def __init__(self, X_train=None, X_test=None, Y_train=None, Y_test=None, pretrained_model=None):
        """
        Params:
            X_train, X_test : An array of embedded input data with dimensions [num_samples, num_tokens_per_sample, embedded_vector_length].
            Y_train, Y_test : An array of one-hot encoded output data with dimensions [num_samples, num_tokens_per_sample, num_token_categories]
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

    @classmethod
    def load_from_saved_tf_model(cls, model_dir):
        file_name = os.path.join(model_dir, "model.h5")
        model = tensorflow.keras.models.load_model(file_name)
        return cls(pretrained_model=model)

    def get_new_model(self):
        input_shape = (self.num_tokens_per_sample, self.embedded_vector_length)
        num_units_in_hidden_layer = self.embedded_vector_length
        use_dropout = True
        
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

        return model

    def fit(self, epochs=1, batch_size=None, num_cv_folds=3):
        """
        Trains the model using kfold cross-validation.
        The data should already be shuffled before running this method.
        """
        X_train = self.X_train
        Y_train = self.Y_train

        num_samples = X_train.shape[0]
        num_samples_per_fold = int(num_samples / num_cv_folds)
        cv_split_indeces = list( np.arange(num_samples)[::num_samples_per_fold] )
        if cv_split_indeces[-1] < num_samples - 1:
            if len(cv_split_indeces) < num_cv_folds + 1:
                cv_split_indeces.append(num_samples - 1)
            else:
                cv_split_indeces[-1] = num_samples - 1
        cv_split_indeces = np.asarray(cv_split_indeces)
        assert len(cv_split_indeces) == num_cv_folds + 1, f"Unable to split training data into {num_cv_folds} folds."

        best_model = None
        best_score = 0
        for cv_num in range(num_cv_folds):
            cv_indeces = np.arange(cv_split_indeces[cv_num], cv_split_indeces[cv_num + 1] + 1, dtype=int)
            train_indeces = [ n for n in range(num_samples) if n not in cv_indeces ]
            X = X_train[train_indeces, :, :]
            Y = Y_train[train_indeces, :, :]
            X_cv = X_train[cv_indeces, :, :]
            Y_cv = Y_train[cv_indeces, :, :]

            model = self.get_new_model()
            model.fit(X, Y, epochs=epochs, batch_size=batch_size)
            scores = model.evaluate(X_cv, Y_cv)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            if scores[1] > best_score:
                best_score = scores[1]
                best_model = model
        self.model = best_model

    def predict(self, x):
        X = [x]
        return self.predict_all(X)[0]

    def predict_all(self, X):
        return self.model.predict(X)

    def save_model(self, model_dir):
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        file_name = os.path.join(model_dir, "model.h5")
        self.model.save(file_name)
