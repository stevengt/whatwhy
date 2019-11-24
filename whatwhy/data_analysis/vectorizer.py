import pickle
import numpy as np
from .helper_methods import get_default_token

class TokenVectorizer():
    """
    Converts sequences of tokens to/from vector representations.
    
    Vector embeddings are retrieved from pre-trained Word2Vec models.
    
    One-hot encodings are created based on the size of the underlying
    Word2Vec model or, if specified, using a custom VocabularyIndex.
    """

    def __init__(self, word2vec_model, tokens_lists=None, num_tokens_per_sample=30, vocab_index=None):

        self.word2vec_model = word2vec_model
        self.num_tokens_per_sample = num_tokens_per_sample
        self.vocab_index = vocab_index

        self.embedded_vector_length = self.get_embedded_vector_length()
        self.num_words_in_vocab = self.get_num_words_in_vocab()
        self.embedded_tokens = None
        self.one_hot_encodings = None
        self.mask = np.zeros(self.num_words_in_vocab)
        
        self.end_of_sequence_token = get_default_token(word2vec_model)
        if self.vocab_index is not None:
            self.vocab_index.add_token(self.end_of_sequence_token)

        if tokens_lists is not None:
            self.tokens_lists = tokens_lists
            self.num_samples = len(tokens_lists)
            self.truncate_tokens_lists()
            self.add_end_of_sequence_tokens()

    @staticmethod
    def load_from_pickle_file(file_name):
        """Loads a TokenVectorizer instance from a pickle file."""
        with open(file_name, "rb") as in_file:
            return pickle.load(in_file)

    def save_to_pickle_file(self, file_name):
        """Saves a TokenVectorizer instance to a pickle file."""
        with open(file_name, "wb") as out_file:
            pickle.dump(self, out_file, protocol=4)

    def get_embedded_vector_length(self):
        return self.word2vec_model.vector_size

    def get_num_words_in_vocab(self):
        if self.vocab_index is not None:
            return self.vocab_index.vocab_size
        else:
            return len(self.word2vec_model.vocab.keys())

    def get_label_from_token(self, token):
        """Returns an integer label for the specified token."""
        if self.vocab_index is not None:
            return self.vocab_index.word2index[token]
        else:
            return self.word2vec_model.vocab[token].index

    def get_token_from_label(self, label):
        """Returns a token from the specified integer label."""
        if self.vocab_index is not None:
            return self.vocab_index.index2word[label]
        else:
            return self.word2vec_model.index2word[label]

    def truncate_tokens_lists(self):
        """
        Truncates any sequences with more than num_tokens_per_sample - 1 tokens.
        The last token in all sequences is reserved for the end-of-sequence token.
        """
        for i, tokens_list in enumerate(self.tokens_lists):
            self.tokens_lists[i] = tokens_list[:self.num_tokens_per_sample - 1]

    def add_end_of_sequence_tokens(self):
        """
        Adds an end-of-sequence token to all token sequences so that Seq2SeqModels
        can be trained to predict a sequence's length. By default, the EOS token is '.'
        """
        for tokens_list in self.tokens_lists:
            tokens_list.append(self.end_of_sequence_token)

    def get_embeddings(self):
        """
        Returns a 3D numpy array containing vector embeddings for all tokens that
        are supported by the underlying Word2Vec model.
        
        The return value is padded with zeros and has dimensions
        [num_samples, num_tokens_per_sample, embedded_vector_length].
        """
        if self.embedded_tokens is None:
            self.embedded_tokens = np.zeros([self.num_samples, self.num_tokens_per_sample, self.embedded_vector_length])
            for i, tokens_list in enumerate(self.tokens_lists):
                j = 0
                for token in tokens_list:
                    try:
                        self.embedded_tokens[i, j, :] = self.word2vec_model.get_vector(token)
                        j += 1
                    except:
                        pass
        return self.embedded_tokens

    def get_word2vec_labels(self):
        """
        Returns a 2D numpy array containing integer labels for all tokens that
        are supported by the underlying Word2Vec model or VocabularyIndex.

        The return value is padded with -1 and has dimensions
        [num_samples, num_tokens_per_sample].
        """
        # Use a default value of -1 for tokens which should not be one-hot encoded.
        labels = -1 * np.ones([self.num_samples, self.num_tokens_per_sample], dtype=int)
        for i, tokens_list in enumerate(self.tokens_lists):
            j = 0
            for token in tokens_list:
                try:
                    labels[i,j] = self.get_label_from_token(token)
                    j += 1
                except:
                    continue
        return labels

    def get_one_hot_encodings(self):
        """
        Returns a 3D numpy array containing one-hot encodings for all tokens that
        are supported by the underlying Word2Vec model or VocabularyIndex.

        One-hot encodings are created based on the size of the underlying
        Word2Vec model or, if specified, using a custom VocabularyIndex.

        The return value is padded with zeros and has dimensions
        [num_samples, num_tokens_per_sample, num_words_in_vocab].
        """
        if self.one_hot_encodings is None:
            labels = self.get_word2vec_labels()
            encodings = np.zeros([self.num_samples, self.num_tokens_per_sample, self.num_words_in_vocab], dtype=bool)
            for i, tokens_list in enumerate(self.tokens_lists):
                for j, label in enumerate(labels[i]):
                    if label != -1:
                        encodings[i, j, label] = True
            self.one_hot_encodings = encodings
        return self.one_hot_encodings

    def decode_single_one_hot_sample(self, encodings):
        """
        Converts a sequence of one-hot encoded tokens to a string using the
        underlying Word2Vec model or VocabularyIndex.
        """
        words = []
        for i in range(self.num_tokens_per_sample):
            word_label_one_hot = encodings[i,:]
            if not np.array_equal(word_label_one_hot, self.mask):
                word_label = np.argmax(word_label_one_hot)
                word = self.get_token_from_label(word_label)
                if word == self.end_of_sequence_token:
                    break
                words.append(word)
        return " ".join(words)

    def decode_multiple_one_hot_samples(self, encodings):
        """
        Converts multiple sequences of one-hot encoded tokens to a list of strings
        using the underlying Word2Vec model or VocabularyIndex.
        """
        decoded_samples = []
        num_samples = encodings.shape[0]
        for i in range(num_samples):
            sample = encodings[i,:,:]
            decoded_samples.append( self.decode_single_one_hot_sample(sample) )
        return decoded_samples

    def save_embeddings_to_pickle_file(self, file_name):
        """Saves all embedded token vectors to a pickle file."""
        embeddings = self.get_embeddings()
        with open(file_name, "wb") as out_file:
            pickle.dump(embeddings, out_file, protocol=4)
    
    def save_one_hot_encodings_to_pickle_file(self, file_name):
        """Saves all one-hot encoded token vectors to a pickle file."""
        encodings = self.get_one_hot_encodings()
        with open(file_name, "wb") as out_file:
            pickle.dump(encodings, out_file, protocol=4)
