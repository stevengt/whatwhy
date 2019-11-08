import numpy as np

class MockWordInfo():
    def __init__(self, index, vector):
        self.index = index
        self.vector = vector

class MockWord2VecModel():

    def __init__(self):
        self.vector_size = 6
        self.vocab = {
            " "     : MockWordInfo( 0, np.asarray([0,0,0,0,0,0]) ),
            "hello" : MockWordInfo( 1, np.asarray([0.1,0,0.2,0,3,0]) ),
            "world" : MockWordInfo( 2, np.asarray([0,0,1,2,1,0]) ),
            "foo"   : MockWordInfo( 3, np.asarray([5,6,4,1,2,3]) ),
            "bar"   : MockWordInfo( 4, np.asarray([0,0,0.5,0,3,5]) )
        }

        self.index2word = {
            0 : " ",
            1 : "hello",
            2 : "world",
            3 : "foo",
            4 : "bar"
        }
    
    def get_vector(self, word):
        return self.vocab[word].vector
    
MOCK_TOKENS_LISTS = [
    ["hello", "world", "foo", "bar"],
    ["hello", "bar", "world", "hello", "foo", "bar"],
    ["foo", "world", "hello", "bar"],
    ["world", "foo", "bar", "hello", "hello", "hello", "world"],
    ["bar", "foo"],
    ["world"],
    ["bar", "bar", "bar"]
]