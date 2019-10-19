import logging
import pandas as pd
import dask.dataframe as ddf
import numpy as np
import spacy
import data_cleaner
from five_w_one_h_extractor import _5W1H_WORDS, FiveWOneHExtractor


class KeywordsExtractor():
    """
    Class for extracting the 5w1h keywords ('who', 'what', 'when', 'where', 'why', 'how')
    from a text column in a pandas DataFrame.
    """

    def __init__(self, df):
        self._df = df
        self._spacy_nlp = spacy.load("en_core_web_sm")
        self.column_names = {
            "5w1h raw text" : [ word + " Raw Text" for word in _5W1H_WORDS ],
            "5w1h keywords" : [ word + " Keywords" for word in _5W1H_WORDS] 
        }
        self.preprocess_text()

    def add_5w1h_keywords_to_df(self):
        try:
            self.initialize_5w1h_columns()
            self.add_raw_5w1h_texts_to_df()
            self.normalize_5w1h_keyword_columns()
            self.filter_words_from_5w1h_keyword_columns()
            logging.info("Done.")
            return self._df
        except Exception as e:
            logging.error(e)

    def get_df_as_dask_df(self):
        return ddf.from_pandas(self._df, npartitions=4)

    def preprocess_text(self):
        """Removes Twitter @reply tags and autocorrects spelling and grammar."""
        
        def get_preprocessed_text_from_dask_df_partition(df_partition):
            return df_partition["Text"].map( data_cleaner.remove_reply_tag_from_tweet_text ) #\
                                                        # .map( data_cleaner.autocorrect_spelling_and_grammar )

        logging.info("Preprocessing text...")
        self._df["Preprocessed Text"] = self.get_df_as_dask_df().map_partitions( get_preprocessed_text_from_dask_df_partition ).compute(scheduler="processes")

    def add_raw_5w1h_texts_to_df(self):
        """Extracts the 5w1h phrases from all text."""

        def add_raw_5w1h_texts_to_dask_df_partition(df_partition):            
            try:
                _5w1h_extractor = FiveWOneHExtractor()
                df_partition[ self.column_names["5w1h raw text"] ] = df_partition.apply( lambda row: \
                                                                                                get_raw_5w1h_texts_from_text( row["Preprocessed Text"], \
                                                                                                                              _5w1h_extractor ), \
                                                                                         axis=1 )
                # The keyword columns initially contain raw text, and will be filtered later.
                df_partition[ self.column_names["5w1h keywords"] ] = df_partition[ self.column_names["5w1h raw text"] ]
            except Exception as e:
                logging.error(e)
            return df_partition

        def get_raw_5w1h_texts_from_text(text, _5w1h_extractor):
            empty_column_vals = pd.Series( [ None for question_type in _5W1H_WORDS ] )
            if text is None or text is np.nan:
                return empty_column_vals
            try:
                phrases_dict = _5w1h_extractor.get_5w1h_dict_from_text(text)
                return pd.Series( [ phrases_dict[question_type] for question_type in _5W1H_WORDS ] )
            except Exception as e:
                logging.warning(e)
                return empty_column_vals

        logging.info("Extracting 5w1h phrases from text...")
        self._df = self.get_df_as_dask_df().map_partitions( add_raw_5w1h_texts_to_dask_df_partition ).compute(scheduler="processes") 

    def initialize_5w1h_columns(self):
        for column_name in self.column_names["5w1h raw text"]:
            self._df[column_name] = None
        for column_name in self.column_names["5w1h keywords"]:
            self._df[column_name] = None

    def normalize_5w1h_keyword_columns(self):
        """Converts the raw 5w1h texts to lists of lowercase and lemmatized words."""

        def convert_5w1h_keyword_columns_to_lowercase():
            for column_name in self.column_names["5w1h keywords"]:
                self._df[column_name] = self._df[column_name].str.lower()
        
        def tokenize_and_lemmatize_5w1h_keyword_columns():
            for column_name in self.column_names["5w1h keywords"]:
                self._df[column_name] = self._df[column_name].map( data_cleaner.get_list_of_lemmatized_words_from_text )

        logging.info("Normalizing and tokenizing 5w1h text...")
        convert_5w1h_keyword_columns_to_lowercase()
        tokenize_and_lemmatize_5w1h_keyword_columns()

    def filter_words_from_5w1h_keyword_columns(self):
        """Extracts the most important keywords from the 5w1h columns."""
        
        def filter_words_from_5w1h_keyword_columns_in_dask_df_partition(df_partition):
            for question_type in _5W1H_WORDS:
                raw_text_column_name = question_type + " Raw Text"
                keywords_column_name = question_type + " Keywords"
                df_partition[keywords_column_name] = df_partition.apply( lambda row: \
                                                                                self.get_keywords_from_raw_and_tokenized_text( row[raw_text_column_name], \
                                                                                                                               row[keywords_column_name] ), \
                                                                         axis=1 )
            return df_partition

        logging.info("Extracting keywords from 5w1h text...")
        self._df = self.get_df_as_dask_df().map_partitions( filter_words_from_5w1h_keyword_columns_in_dask_df_partition ).compute(scheduler="processes") 

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
        if text is None or text is np.nan:
            return []
        try:
            doc = self._spacy_nlp(text)
            return [ named_entity.text.lower() for named_entity in doc.ents ]
        except:
            return []
