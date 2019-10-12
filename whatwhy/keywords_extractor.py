import pandas as pd
import numpy as np
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
import data_cleaner

# TODO
# Consider aggregating "what", "who", "when", "where" words
# from Giveme5W1H into one "what" category, and aggregating
# "why" and "how" words into one "why" category.
class KeywordsExtractor():
    """Class for extracting the 'what' and 'why' keywords from a text column in a pandas DataFrame."""

    def __init__(self, df):
        self.df = df
        self.extractor = MasterExtractor()

    # TODO
    def preprocess_text(self):
        """Removes Twitter @reply tags and converts text to lowercase."""
        self.df["Text"] = self.df["Text"].map( data_cleaner.remove_reply_tag_from_tweet_text ).str.lower()

        # TODO
        # Autocorrect word spelling

    def add_full_what_and_why_texts_to_df(self, column="Text"):
        """Extracts the 'what' and 'why' phrases from all text using Giveme5W1H."""
        self.df["What"] = None
        self.df["Why"] = None  
        for id, row in self.df.iterrows():
            try:
                text = row[column]
                if text is None or text is np.nan:
                    continue
                doc = Document.from_text(text)
                doc = self.extractor.parse(doc)            
                what = doc.get_top_answer('what').get_parts_as_text()   
                why = doc.get_top_answer('why').get_parts_as_text()
                self.df.at[id, "What"] = what        
                self.df.at[id, "Why"] = why
            except:
                continue

    def get_lemmatized_words_from_text(self, text):
        text = data_cleaner.remove_punctuation_from_text(text)
        words = data_cleaner.get_list_of_words_from_text(text)
        words = data_cleaner.lemmatize_list_of_words(words)
        return words

    def tokenize_and_lemmatize_what_and_why_columns(self):
        self.df["What"] = self.df["What"].map( self.get_lemmatized_words_from_text )
        self.df["Why"] = self.df["Why"].map( self.get_lemmatized_words_from_text )

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

    def filter_words_from_list(self, words):
        """Extracts the most important keywords from a list of words."""
        words = self.remove_filler_words_from_list(words)
        most_common_english_words = set( self.get_most_common_english_words_from_list(words) )
        named_entities = set( self.get_named_entities_from_list(words) )
        words = most_common_english_words.union(named_entities)
        return list(words)

    def filter_words_from_what_and_why_lists(self):
        self.df["What"] = self.df["What"].map( self.filter_words_from_list )
        self.df["Why"] = self.df["Why"].map( self.filter_words_from_list )

    def add_what_and_why_keywords_to_df(self):
        self.preprocess_text()
        self.add_full_what_and_why_texts_to_df()
        # self.tokenize_and_lemmatize_what_and_why_columns()
        # self.filter_words_from_what_and_why_lists()
        return self.df

