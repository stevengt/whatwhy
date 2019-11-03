import ast
import numpy as np
from nltk.corpus import stopwords
from whatwhy.text_processing.batch_processors import BatchProcessorBase
from whatwhy.text_processing.helper_methods import get_csv_string_from_df, get_df_from_csv_string

STOP_WORDS = set(stopwords.words('english'))

class BatchFilterer(BatchProcessorBase):
    """Filters out words from a list of word tokens"""

    def __init__(self, source,
                       dest,
                       id_col_name="ID",
                       source_col_name="Preprocessed Text",
                       dest_col_name="Processed Text",
                       include_cols=None,
                       should_remove_stop_words=True):

        super().__init__(source=source,
                            dest=dest,
                            id_col_name=id_col_name,
                            source_col_name=source_col_name,
                            dest_col_name=dest_col_name,
                            include_cols=include_cols)
        self.should_remove_stop_words = should_remove_stop_words

    def get_batch_results(self, batch):
        batch_as_df = get_df_from_csv_string(batch)
        batch_as_df[self.dest_col_name] = batch_as_df[self.source_col_name].apply(self.get_text_as_list)
        
        if self.should_remove_stop_words:
            batch_as_df[self.dest_col_name] = batch_as_df[self.dest_col_name].apply(self.remove_punctuation) \
                                                                             .apply(self.remove_non_alphabetic_characters) \
                                                                             .apply(self.remove_stop_words) \
                                                                             .apply(self.remove_short_tokens)
        results_df_cols = [self.id_col_name, self.dest_col_name]
        results_df_cols.extend(self.include_cols)
        results_df = batch_as_df[results_df_cols]
        results_csv_string = get_csv_string_from_df(results_df)

        results = {
            "target_results_file_name" : f"batch{batch_as_df[self.id_col_name].iloc[0]}.csv",
            "file_content" : results_csv_string
        }
        return results

    def get_text_as_list(self, text):
        if text is None or text is np.nan:
            return []
        else:
            return ast.literal_eval(text)

    def remove_punctuation(self, tokens):
        table = str.maketrans('', '', string.punctuation)
        return [token.translate(table) for token in tokens]

    def remove_non_alphabetic_characters(self, tokens):
        return [token for token in tokens if token.isalpha()]

    def remove_stop_words(self, tokens):
        return [ token for token in tokens if not token in STOP_WORDS ]

    def remove_short_tokens(self, tokens):
        return [token for token in tokens if len(tokens) > 1]
