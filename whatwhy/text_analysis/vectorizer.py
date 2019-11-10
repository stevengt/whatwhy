import numpy as np
from .helper_methods import get_default_token

class TokenVectorizer():

    def __init__(self, word2vec_model, tokens_lists=None, num_tokens_per_sample=30):

        self.word2vec_model = word2vec_model
        self.embedded_vector_length = self.get_embedded_vector_length()
        self.num_words_in_vocab = self.get_num_words_in_vocab()
        self.num_tokens_per_sample = num_tokens_per_sample
        self.embedded_tokens = None
        self.mask = np.zeros(self.num_words_in_vocab)
        self.end_of_sequence_token = "."

        if tokens_lists is not None:
            self.tokens_lists = tokens_lists
            self.num_samples = len(tokens_lists)
            self.truncate_tokens_lists()
            self.add_end_of_sequence_tokens()

    def get_embedded_vector_length(self):
        return self.word2vec_model.vector_size

    def get_num_words_in_vocab(self):
        return len(self.word2vec_model.vocab.keys())

    def truncate_tokens_lists(self):
        for i, tokens_list in enumerate(self.tokens_lists):
            self.tokens_lists[i] = tokens_list[:self.num_tokens_per_sample - 1]

    def add_end_of_sequence_tokens(self):
        for tokens_list in self.tokens_lists:
            tokens_list.append(self.end_of_sequence_token)

    def get_embeddings(self):
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

    def get_word2vec_indeces(self):
        # Use a default value of -1 for tokens which should not be one-hot encoded.
        indeces = -1 * np.ones([self.num_samples, self.num_tokens_per_sample], dtype=int)
        for i, tokens_list in enumerate(self.tokens_lists):
            j = 0
            for token in tokens_list:
                try:
                    indeces[i,j] = self.word2vec_model.vocab[token].index
                    j += 1
                except:
                    continue
        return indeces

    def get_one_hot_encodings(self):
        indeces = self.get_word2vec_indeces()
        encodings = np.zeros([self.num_samples, self.num_tokens_per_sample, self.num_words_in_vocab])
        for i, tokens_list in enumerate(self.tokens_lists):
            for j, index in enumerate(indeces[i]):
                if index != -1:
                    encodings[i, j, index] = 1
        return encodings

    def decode_single_one_hot_sample(self, encodings):
        words = []
        for i in range(self.num_tokens_per_sample):
            word_index_one_hot = encodings[i,:]
            if not np.array_equal(word_index_one_hot, self.mask):
                word_index = np.argmax(word_index_one_hot)
                word = self.word2vec_model.index2word[word_index]
                if word == self.end_of_sequence_token:
                    break
                words.append(word)
        return " ".join(words)

    def decode_multiple_one_hot_samples(self, encodings):
        decoded_samples = []
        num_samples = encodings.shape[0]
        for i in range(num_samples):
            sample = encodings[i,:,:]
            decoded_samples.append( self.decode_single_one_hot_sample(sample) )
        return decoded_samples
