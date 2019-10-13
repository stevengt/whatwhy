import logging
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
        self.what_and_why_extractor = MasterExtractor()

    def add_what_and_why_keywords_to_df(self):
        self.preprocess_text()
        self.add_full_what_and_why_texts_to_df()
        self.normalize_what_and_why_columns()
        self.filter_words_from_what_and_why_lists()
        logging.info("Done.")
        return self.df

    def preprocess_text(self):
        """Removes Twitter @reply tags and autocorrects spelling and grammar."""
        logging.info("Preprocessing text...")
        self.df["Text"] = self.df["Text"].map( data_cleaner.remove_reply_tag_from_tweet_text ) \
                                         .map( data_cleaner.autocorrect_spelling_and_grammar )
        
    def add_full_what_and_why_texts_to_df(self, column="Text"):
        """Extracts the 'what' and 'why' phrases from all text using Giveme5W1H."""
        logging.info("Extracting what/why phrases from text...")
        self.df["What"] = None
        self.df["Why"] = None  
        for id, row in self.df.iterrows():
            try:
                text = row[column]
                if text is None or text is np.nan:
                    continue
                doc = Document.from_text(text)
                doc = self.what_and_why_extractor.parse(doc)            
                what = doc.get_top_answer('what').get_parts_as_text()   
                why = doc.get_top_answer('why').get_parts_as_text()
                self.df.at[id, "What"] = what        
                self.df.at[id, "Why"] = why
            except:
                continue

    def normalize_what_and_why_columns(self):
        """Converts the raw what/why texts to lists of lowercase and lemmatized words."""
        logging.info("Normalizing and tokenizing what/why text...")
        self.convert_what_and_why_columns_to_lowercase()
        self.tokenize_and_lemmatize_what_and_why_columns()

    def convert_what_and_why_columns_to_lowercase(self):
        self.df["What"] = self.df["What"].str.lower()
        self.df["Why"] = self.df["Why"].str.lower()

    def tokenize_and_lemmatize_what_and_why_columns(self):
        self.df["What"] = self.df["What"].map( data_cleaner.get_list_of_lemmatized_words_from_text )
        self.df["Why"] = self.df["Why"].map( data_cleaner.get_list_of_lemmatized_words_from_text )

    def filter_words_from_what_and_why_lists(self):
        """Extracts the most important keywords from lists of words in the what/why columns."""
        logging.info("Extracting keywords from what/why text...")
        self.df["What"] = self.df["What"].map( self.filter_words_from_list )
        self.df["Why"] = self.df["Why"].map( self.filter_words_from_list )

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
