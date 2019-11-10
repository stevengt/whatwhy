
class VocabularyIndex():
    """
    Creates an index that maps integer labels to words.
    This can help reduce the size of one-hot encoded words
    in the vocabulary, compared to that of pre-trained word2vec models.
    """

    def __init__(self):
        self.vocab_size = 0
        self.word2index = {}
        self.index2word = {}
    
    @classmethod
    def from_list(cls, tokens_list):
        instance = cls()
        instance.add_tokens_from_list(tokens_list)
        return instance

    @classmethod
    def from_lists(cls, tokens_lists):
        instance = cls()
        instance.add_tokens_from_lists(tokens_lists)
        return instance

    def add_tokens_from_list(self, tokens_list):
        for token in tokens_list:
            self.add_token(token)
    
    def add_tokens_from_lists(self, tokens_lists):
        for tokens_list in tokens_lists:
            self.add_tokens_from_list(tokens_list)

    def add_token(self, token):
        if self.word2index.get(token) is None:
            self.word2index[token] = self.vocab_size
            self.index2word[self.vocab_size] = token
            self.vocab_size += 1
