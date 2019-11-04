import string
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords 
from whatwhy import QUESTION_WORDS, configure_nltk
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_df_from_csv_string, get_csv_string_from_df

configure_nltk()

STOP_WORDS = set(stopwords.words('english'))

class BatchTokenizer(BatchProcessorBase):
    """Tokenizes and standardizes text-segments."""

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name="Preprocessed Text",
                       dest_col_name="Tokens",
                       include_cols=None):

        super().__init__(source=source,
                            dest=dest,
                            id_col_name=id_col_name,
                            source_col_name=source_col_name,
                            dest_col_name=dest_col_name,
                            include_cols=include_cols)
        
    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        batch_as_df[self.dest_col_name] = self.get_tokenized_column(batch_as_df, self.source_col_name)

        results_df_cols = [self.id_col_name, self.dest_col_name]
        results_df_cols.extend(self.include_cols)
        results_df = batch_as_df[results_df_cols]
        results_csv_string = get_csv_string_from_df(results_df)

        results = {
            "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return results

    def get_tokenized_column(self, df, col_name):
        return df[col_name].apply( self.get_list_of_lemmatized_words_from_text ) \
                           .apply( self.convert_to_lowercase ) \
                           .apply( self.remove_punctuation ) \
                           .apply( self.remove_non_alphabetic_tokens ) \
                           .apply( self.remove_stop_words ) \
                           .apply( self.remove_short_tokens )

    def get_list_of_lemmatized_words_from_text(self, text):
        if text is None or text is np.nan or text == "NOT PROCESSED":
            return []
        try:
            text_blob = TextBlob(text)
            return list( text_blob.words.singularize().lemmatize() )
        except:
            return []

    def convert_to_lowercase(self, tokens):
        return [ token.lower() for token in tokens ]

    def remove_punctuation(self, tokens):
        table = str.maketrans('', '', string.punctuation)
        return [ token.translate(table) for token in tokens ]

    def remove_non_alphabetic_tokens(self, tokens):
        return [ token for token in tokens if token.isalpha() ]

    def remove_stop_words(self, tokens):
        return [ token for token in tokens if not token in STOP_WORDS ]

    def remove_short_tokens(self, tokens):
        return [ token for token in tokens if len(token) > 2 ]

class BatchWHPhrasesTokenizer(BatchTokenizer):
    
    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        results_df_cols = [self.id_col_name]
        results_df_cols.extend(self.include_cols)

        for question_type in QUESTION_WORDS:
            source_col_name = question_type
            dest_col_name = question_type + " tokens"
            results_df_cols.extend([source_col_name, dest_col_name])
            batch_as_df[dest_col_name] = self.get_tokenized_column(batch_as_df, source_col_name)

        results_df = batch_as_df[results_df_cols]
        results_csv_string = get_csv_string_from_df(results_df)

        results = {
            "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return results