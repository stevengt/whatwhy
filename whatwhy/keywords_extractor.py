import logging
import pandas as pd
import numpy as np
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
import data_cleaner

_5W1H_WORDS = ["Who", "What", "When", "Where", "Why", "How"]

class KeywordsExtractor():
    """
    Class for extracting the 5w1h keywords ('who', 'what', 'when', 'where', 'why', 'how')
    from a text column in a pandas DataFrame.
    """

    def __init__(self, df):
        self._df = df
        self._5w1h_extractor = MasterExtractor()

    def add_5w1h_keywords_to_df(self):
        self.preprocess_text()
        self.initialize_5w1h_columns()
        self.add_full_5w1h_texts_to_df()
        self.normalize_5w1h_columns()
        self.filter_words_from_5w1h_lists()
        logging.info("Done.")
        return self._df

    def preprocess_text(self):
        """Removes Twitter @reply tags and autocorrects spelling and grammar."""
        logging.info("Preprocessing text...")
        self._df["Text"] = self._df["Text"].map( data_cleaner.remove_reply_tag_from_tweet_text ) \
                                         .map( data_cleaner.autocorrect_spelling_and_grammar )
        
    def add_full_5w1h_texts_to_df(self, column="Text"):
        """Extracts the 5w1h phrases from all text."""
        
        def get_5w1h_phrase_or_empty_string(doc, question_type):
            try:
                phrase = doc.get_top_answer( question_type.lower() ).get_parts_as_text()
                if phrase is None:
                    phrase = ""
                return phrase
            except:
                return ""

        logging.info("Extracting 5w1h phrases from text...")
        for id, row in self._df.iterrows():
            try:
                text = row[column]
                if text is None or text is np.nan:
                    continue
                doc = Document.from_text(text)
                doc = self._5w1h_extractor.parse(doc)
                for question_type in _5W1H_WORDS:
                    self._df.at[id, question_type] = get_5w1h_phrase_or_empty_string(doc, question_type)
            except Exception as e:
                logging.debug(e)
                continue

    def initialize_5w1h_columns(self):
        for column_name in _5W1H_WORDS:
            self._df[column_name] = None

    def normalize_5w1h_columns(self):
        """Converts the raw 5w1h texts to lists of lowercase and lemmatized words."""
        logging.info("Normalizing and tokenizing 5w1h text...")
        self.convert_5w1h_columns_to_lowercase()
        self.tokenize_and_lemmatize_5w1h_columns()

    def convert_5w1h_columns_to_lowercase(self):
        for column_name in _5W1H_WORDS:
            self._df[column_name] = self._df[column_name].str.lower()

    def tokenize_and_lemmatize_5w1h_columns(self):
        for column_name in _5W1H_WORDS:
            self._df[column_name] = self._df[column_name].map( data_cleaner.get_list_of_lemmatized_words_from_text )

    def filter_words_from_5w1h_lists(self):
        """Extracts the most important keywords from lists of words in the 5w1h columns."""
        logging.info("Extracting keywords from 5w1h text...")
        for column_name in _5W1H_WORDS:
            self._df[column_name] = self._df[column_name].map( self.filter_words_from_list )

    def filter_words_from_list(self, words):
        """Extracts the most important keywords from a list of words."""
        words = self.remove_filler_words_from_list(words)
        most_common_english_words = set( self.get_most_common_english_words_from_list(words) )
        named_entities = set( self.get_named_entities_from_list(words) )
        words = most_common_english_words.union(named_entities)
        return list(words)

    # TODO
    def remove_filler_words_from_list(self, words):
        """Removes common 'filler' words from a list of words, such as 'a', 'the', or 'like'."""
        return words

    # TODO
    def get_most_common_english_words_from_list(self, words):
        return words
    
    # TODO
    def get_named_entities_from_list(self, words):
        return words
