import logging
import pandas as pd
import numpy as np
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor
import spacy
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
        self._spacy_nlp = spacy.load("en_core_web_sm")
        self.column_names = {
            "raw text" : [ word + " Raw Text" for word in _5W1H_WORDS ],
            "keywords" : [ word + " Keywords" for word in _5W1H_WORDS] 
        }
        self.preprocess_text()

    def add_5w1h_keywords_to_df(self):
        self.initialize_5w1h_columns()
        self.add_raw_5w1h_texts_to_df()
        self.normalize_5w1h_keyword_columns()
        self.filter_words_from_5w1h_keyword_columns()
        logging.info("Done.")
        return self._df

    def preprocess_text(self):
        """Removes Twitter @reply tags and autocorrects spelling and grammar."""
        logging.info("Preprocessing text...")
        self._df["Preprocessed Text"] = self._df["Text"].map( data_cleaner.remove_reply_tag_from_tweet_text ) \
                                                        .map( data_cleaner.autocorrect_spelling_and_grammar )
        
    def add_raw_5w1h_texts_to_df(self, column="Preprocessed Text"):
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
                    raw_text = get_5w1h_phrase_or_empty_string(doc, question_type)
                    self._df.at[id, question_type + " Raw Text"] = raw_text
                    self._df.at[id, question_type + " Keywords"] = raw_text # This column will later be converted from raw text to keywords.
            except Exception as e:
                logging.debug(e)
                continue

    def initialize_5w1h_columns(self):
        for column_name in self.column_names["raw text"]:
            self._df[column_name] = None
        for column_name in self.column_names["keywords"]:
            self._df[column_name] = None

    def normalize_5w1h_keyword_columns(self):
        """Converts the raw 5w1h texts to lists of lowercase and lemmatized words."""
        logging.info("Normalizing and tokenizing 5w1h text...")
        self.convert_5w1h_keyword_columns_to_lowercase()
        self.tokenize_and_lemmatize_5w1h_keyword_columns()

    def convert_5w1h_keyword_columns_to_lowercase(self):
        for column_name in self.column_names["keywords"]:
            self._df[column_name] = self._df[column_name].str.lower()

    def tokenize_and_lemmatize_5w1h_keyword_columns(self):
        for column_name in self.column_names["keywords"]:
            self._df[column_name] = self._df[column_name].map( data_cleaner.get_list_of_lemmatized_words_from_text )

    def filter_words_from_5w1h_keyword_columns(self):
        """Extracts the most important keywords from the 5w1h columns."""
        logging.info("Extracting keywords from 5w1h text...")
        for question_type in _5W1H_WORDS:
            raw_text_column_name = question_type + " Raw Text"
            keywords_column_name = question_type + " Keywords"
            self._df[keywords_column_name] = self._df.apply( lambda row: \
                                                                    self.get_keywords_from_raw_and_tokenized_text( row[raw_text_column_name], \
                                                                                                                   row[keywords_column_name] ), \
                                                             axis=1 )

    def get_keywords_from_raw_and_tokenized_text(self, raw_text, words):
        """Extracts the most important keywords from a pair of raw/tokenized text."""
        keywords = self.remove_filler_words_from_list(words)
        keywords = set(keywords)

        # Get named entities and remove individual keywords which are part of a multi-word named entity.
        named_entities = set( self.get_named_entities_from_text(raw_text) )
        for named_entity in named_entities:
            named_entity_words = set( data_cleaner.get_list_of_lemmatized_words_from_text(named_entity) )
            keywords = keywords.difference(named_entity_words)
        keywords = list(keywords)

        most_common_english_words = set( self.get_most_common_english_words_from_list(keywords) )
        keywords = most_common_english_words.union(named_entities)
        return list(keywords)

    # TODO
    def remove_filler_words_from_list(self, words):
        """Removes common 'filler' words from a list of words, such as 'a', 'the', or 'like'."""
        return words

    # TODO
    def get_most_common_english_words_from_list(self, words):
        return words
    
    def get_named_entities_from_text(self, text):
        doc = self._spacy_nlp(text)
        return [ named_entity.text.lower() for named_entity in doc.ents ]
