import numpy as np
from .helper_methods import get_default_token

class TokenVectorizer():

    def __init__(self, tokens_lists, word2vec_model, num_tokens_per_sample=30):
        # All what/why sequences to be truncated or padded to num_tokens_per_sample
        self.tokens_lists = tokens_lists
        self.word2vec_model = word2vec_model
        self.embedded_vector_length = word2vec_model.vector_size
        self.num_words_in_vocab = len(word2vec_model.vocab.keys())
        self.num_samples = len(tokens_lists)
        self.num_tokens_per_sample = num_tokens_per_sample
        self.embedded_tokens = None
        self.masks = [ np.zeros(self.num_words_in_vocab), -1 * np.ones(self.num_words_in_vocab) ]

    def get_embeddings(self):
        if self.embedded_tokens is None:
            self.embedded_tokens = -1 * np.ones([self.num_samples, self.num_tokens_per_sample, self.embedded_vector_length])
            for i, tokens_list in enumerate(self.tokens_lists):
                j = 0
                for token in tokens_list:
                    if j >= self.num_tokens_per_sample:
                        break
                    try:
                        self.embedded_tokens[i, j, :] = self.word2vec_model.get_vector(token)
                        j += 1
                    except:
                        pass
        return self.embedded_tokens

    def get_word2vec_indeces(self):
        # Use a default value of -1 because 0 will be one-hot encoded, but not -1.
        indeces = -1 * np.ones([self.num_samples, self.num_tokens_per_sample], dtype=int)
        for i, tokens_list in enumerate(self.tokens_lists):
            j = 0
            for token in tokens_list:
                if j >= self.num_tokens_per_sample:
                    break
                try:
                    indeces[i,j] = self.word2vec_model.vocab[token].index
                    j += 1
                except:
                    continue
        return indeces

    def get_one_hot_encodings(self):
        indeces = self.get_word2vec_indeces()
        encodings = -1 * np.ones([self.num_samples, self.num_tokens_per_sample, self.num_words_in_vocab], dtype=int)
        for i, tokens_list in enumerate(self.tokens_lists):
            for j, index in enumerate(indeces[i]):
                if j >= self.num_tokens_per_sample:
                    break
                if index != -1:
                    encodings[i, j, :] = np.zeros(self.num_words_in_vocab)
                    encodings[i, j, index] = 1
        return encodings

    def decode_single_one_hot_sample(self, encodings):
        words = []
        for i in range(self.num_tokens_per_sample):
            word_index_one_hot = encodings[i,:]
            if not np.array_equal(word_index_one_hot, self.masks[0]) and not np.array_equal(word_index_one_hot, self.masks[1]):
                word_index = np.argmax(word_index_one_hot)
                word = self.word2vec_model.index2word[word_index]
                words.append(word)
        return " ".join(words)

    def decode_multiple_one_hot_samples(self, encodings):
        decoded_samples = []
        num_samples = encodings.shape[0]
        for i in range(num_samples):
            sample = encodings[i,:,:]
            decoded_samples.append( self.decode_single_one_hot_sample(sample) )
        return decoded_samples
